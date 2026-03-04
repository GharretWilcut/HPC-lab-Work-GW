// task.c
//
// BFS relaxation kernel — UPMEM DPU, 16 tasklets.
//
// The host uploads fresh levels[] and parents[] before every kernel
// launch, so we just need one correct relaxation pass per invocation.
//
// Design — all-MRAM, two-phase per launch:
//
//   Phase 1  PROPOSE  (edge-striped, reads only)
//     Tasklet T processes edges [T*BATCH, T*BATCH+stride, ...].
//     For each edge (L,R) it reads lvl_L and lvl_R and, if either
//     direction can improve, records a candidate in a compact WRAM
//     proposal table indexed by local node id:
//       wram_prop[node] = {best_candidate_level, local_parent_index}
//     The proposal table fits in WRAM: 2 × uint32_t × MAX_PROP_NODES
//     per tasklet.  MAX_PROP_NODES is the WRAM-safe per-tasklet limit.
//     Because each tasklet writes only its OWN proposal table (indexed
//     by [tasklet_id][node]), there are zero write conflicts in Phase 1.
//
//   Phase 2  COMMIT  (node-striped, writes only)
//     Tasklet T owns nodes n where (n % NR_TASKLETS == T).
//     It scans ALL tasklets' proposal tables for node n, picks the best
//     (lowest level), compares against the current MRAM level, and if
//     better writes levels[n] and parents[n] to MRAM.
//     No locking needed: exclusive node ownership = no write conflicts.
//
// WRAM budget:
//   Per-tasklet proposal table: 2 fields × 4 bytes × MAX_PROP_NODES
//   With NR_TASKLETS=16 and MAX_PROP_NODES=512:
//     16 × 512 × 8 = 65536 bytes — exactly 64 KB.
//   We need a little headroom for stacks and globals, so use 480.
//     16 × 480 × 8 = 61440 bytes  (~60 KB) — leaves ~4 KB for stacks.
//
// Node ids can exceed MAX_PROP_NODES (DPUs have up to 63K nodes).
// Proposal tables are indexed by (node % MAX_PROP_NODES) with a tag
// to detect collisions — if a collision occurs we fall back to a
// direct mutex-protected MRAM write (rare, only for hash collisions).

#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <mram.h>
#include <defs.h>
#include <barrier.h>
#include <mutex.h>

#define INF            0xFFFFFFFF
#define EDGE_BATCH     8
#define MAX_PROP_NODES 480   // per-tasklet WRAM proposal slots

BARRIER_INIT(barrier, NR_TASKLETS);
MUTEX_INIT(fallback_mutex);  // only used on hash collision (rare)

typedef struct {
    uint32_t numNodes;
    uint32_t numEdges;
    uint32_t localToGlobal_m;
    uint32_t edges_m;
    uint32_t levels_m;
    uint32_t parents_m;
    uint32_t changed_m;
    uint32_t padding;
} DPUParams;

typedef struct { uint32_t left,  right;  } Edge;
typedef struct { uint32_t value, padding; } AlignedU32;

// Per-tasklet proposal slot
typedef struct {
    uint32_t node;          // which node this slot is for (INF = empty)
    uint32_t level;         // proposed level
    uint32_t parent_local;  // local index of parent node
    uint32_t _pad;
} PropSlot;

// 16 tasklets × 480 slots × 16 bytes = 122880 bytes — too large.
// Use 8 bytes per slot (drop _pad, pack node+level into one struct,
// parent separately) — still large.
//
// Simpler and correct: use 2 arrays of uint32_t per tasklet.
// prop_level[t][slot]  and  prop_parent[t][slot], where slot = node % MAX_PROP_NODES.
// We also need prop_node[t][slot] to detect collisions.
// 3 × 16 × 480 × 4 = 92160 bytes — still too large for 64 KB WRAM.
//
// Conclusion: any per-tasklet table large enough to avoid frequent
// collisions won't fit.  The correct UPMEM pattern for this problem
// is: keep ONE shared proposal array in WRAM sized to actual numNodes,
// but since numNodes can be 63K that's also too large.
//
// The only approach that fits AND is correct:
//   - Levels/parents stay in MRAM.
//   - Use a single mutex for writes.
//   - Accept the mutex contention — UPMEM tasklets are barrel-processor
//     style and hide latency well; the mutex is held only during the
//     short window of one mram_read + comparison + mram_write.
//   - Use double-checked locking to minimise lock acquisition.
//
// The previous "node ownership" version failed because it conflated
// edge ownership with node ownership.  The fix is simple:
//   ALL tasklets process ALL relaxation directions for their edge stripe.
//   Writes are serialised with a mutex.  No ownership filtering.

static bool shared_change;

int main()
{
    uint32_t id = me();

    if (id == 0) shared_change = false;

    DPUParams p;
    mram_read((__mram_ptr DPUParams *)DPU_MRAM_HEAP_POINTER,
              &p, sizeof(DPUParams));

    __mram_ptr Edge       *mram_edges   =
        (__mram_ptr Edge *)      (DPU_MRAM_HEAP_POINTER + p.edges_m);
    __mram_ptr AlignedU32 *mram_levels  =
        (__mram_ptr AlignedU32 *)(DPU_MRAM_HEAP_POINTER + p.levels_m);
    __mram_ptr AlignedU32 *mram_parents =
        (__mram_ptr AlignedU32 *)(DPU_MRAM_HEAP_POINTER + p.parents_m);
    __mram_ptr AlignedU32 *mram_l2g     =
        (__mram_ptr AlignedU32 *)(DPU_MRAM_HEAP_POINTER + p.localToGlobal_m);

    barrier_wait(&barrier);

    // ── Relaxation pass (edge-striped) ─────────────────────────────────
    // Each tasklet owns edges: id*EDGE_BATCH, id*EDGE_BATCH+stride, ...
    // For each edge both directions are attempted.
    // A write only happens when we find a strict improvement; the mutex
    // is acquired only in that case (double-checked pattern).

    Edge       ebuf[EDGE_BATCH] __attribute__((aligned(8)));
    AlignedU32 lbuf[2]          __attribute__((aligned(8)));  // [L], [R] levels
    AlignedU32 wbuf[1]          __attribute__((aligned(8)));  // write buffer
    AlignedU32 gbuf[1]          __attribute__((aligned(8)));  // l2g lookup

    bool local_change = false;

    uint32_t i      = id * EDGE_BATCH;
    uint32_t stride = NR_TASKLETS * EDGE_BATCH;

    while (i < p.numEdges) {
        uint32_t fetch = EDGE_BATCH;
        if (i + fetch > p.numEdges) fetch = p.numEdges - i;

        uint32_t fetch_bytes = fetch * sizeof(Edge);  // Edge=8 bytes, always aligned
        mram_read(&mram_edges[i], ebuf, fetch_bytes);

        for (uint32_t k = 0; k < fetch; k++) {
            uint32_t L = ebuf[k].left;
            uint32_t R = ebuf[k].right;

            // Read both levels — two 8-byte MRAM reads
            mram_read(&mram_levels[L], &lbuf[0], sizeof(AlignedU32));
            mram_read(&mram_levels[R], &lbuf[1], sizeof(AlignedU32));
            uint32_t lvl_L = lbuf[0].value;
            uint32_t lvl_R = lbuf[1].value;

            // ── Relax L → R ──────────────────────────────────────────
            if (lvl_L != INF && lvl_L + 1 < lvl_R) {
                mutex_lock(fallback_mutex);
                // Re-read inside lock to get the latest value
                mram_read(&mram_levels[R], wbuf, sizeof(AlignedU32));
                mram_read(&mram_levels[L], &lbuf[0], sizeof(AlignedU32));
                lvl_L = lbuf[0].value;
                if (lvl_L != INF && lvl_L + 1 < wbuf[0].value) {
                    wbuf[0].value   = lvl_L + 1;
                    wbuf[0].padding = 0;
                    mram_write(wbuf, &mram_levels[R], sizeof(AlignedU32));

                    mram_read(&mram_l2g[L], gbuf, sizeof(AlignedU32));
                    gbuf[0].padding = 0;
                    mram_write(gbuf, &mram_parents[R], sizeof(AlignedU32));
                    local_change = true;
                }
                mutex_unlock(fallback_mutex);
            }

            // Refresh levels after possible update
            mram_read(&mram_levels[L], &lbuf[0], sizeof(AlignedU32));
            mram_read(&mram_levels[R], &lbuf[1], sizeof(AlignedU32));
            lvl_L = lbuf[0].value;
            lvl_R = lbuf[1].value;

            // ── Relax R → L ──────────────────────────────────────────
            if (lvl_R != INF && lvl_R + 1 < lvl_L) {
                mutex_lock(fallback_mutex);
                mram_read(&mram_levels[L], wbuf, sizeof(AlignedU32));
                mram_read(&mram_levels[R], &lbuf[1], sizeof(AlignedU32));
                lvl_R = lbuf[1].value;
                if (lvl_R != INF && lvl_R + 1 < wbuf[0].value) {
                    wbuf[0].value   = lvl_R + 1;
                    wbuf[0].padding = 0;
                    mram_write(wbuf, &mram_levels[L], sizeof(AlignedU32));

                    mram_read(&mram_l2g[R], gbuf, sizeof(AlignedU32));
                    gbuf[0].padding = 0;
                    mram_write(gbuf, &mram_parents[L], sizeof(AlignedU32));
                    local_change = true;
                }
                mutex_unlock(fallback_mutex);
            }
        }

        i += stride;
    }

    // ── Merge change flags ─────────────────────────────────────────────
    barrier_wait(&barrier);

    if (local_change) {
        mutex_lock(fallback_mutex);
        shared_change = true;
        mutex_unlock(fallback_mutex);
    }

    barrier_wait(&barrier);

    // ── Write convergence flag ─────────────────────────────────────────
    if (id == 0) {
        AlignedU32 out __attribute__((aligned(8)));
        out.value   = shared_change ? 1 : 0;
        out.padding = 0;
        mram_write(&out,
                   (__mram_ptr AlignedU32 *)(DPU_MRAM_HEAP_POINTER + p.changed_m),
                   sizeof(AlignedU32));
    }

    return 0;
}