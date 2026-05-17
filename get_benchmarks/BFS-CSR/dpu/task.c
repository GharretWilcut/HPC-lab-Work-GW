// task.c — CSR-based BFS kernel for UPMEM DPU

#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <mram.h>
#include <defs.h>
#include <barrier.h>
#include <mutex.h>

#define INF            0xFFFFFFFF
#define NEIGHBOR_BATCH 32

BARRIER_INIT(barrier, NR_TASKLETS);
MUTEX_INIT(fallback_mutex);

// ── Shared structs (must match app.cpp) ─────────────────────────────────────

typedef struct {
    uint32_t numNodes;
    uint32_t numEdges;
    uint32_t localToGlobal_m;
    uint32_t nodePtrs_m;
    uint32_t neighborIdxs_m;
    uint32_t levels_m;
    uint32_t changed_m;
    uint32_t _pad;
} DPUParams;

typedef struct { uint32_t value; uint32_t parent; } AlignedU32;

static bool shared_change;

// ── Helpers ──────────────────────────────────────────────────────────────────

// Read nodePtrs[idx] safely with 8-byte alignment.
// Always reads a pair; returns the correct element.
static inline uint32_t read_nodeptr(__mram_ptr uint32_t *base, uint32_t idx)
{
    uint32_t aligned = idx & ~1u;
    uint32_t buf[2] __attribute__((aligned(8)));
    mram_read(&base[aligned], buf, 2 * sizeof(uint32_t));
    return buf[idx - aligned];   // buf[0] if idx was even, buf[1] if odd
}

// ── Main ─────────────────────────────────────────────────────────────────────
int main()
{
    uint32_t id = me();

    if (id == 0) shared_change = false;

    // ── Read parameters from MRAM ─────────────────────────────────────────
    DPUParams p;
    mram_read((__mram_ptr DPUParams *)DPU_MRAM_HEAP_POINTER,
              &p, sizeof(DPUParams));

    __mram_ptr uint32_t   *mram_nodePtrs     =
        (__mram_ptr uint32_t *)   (DPU_MRAM_HEAP_POINTER + p.nodePtrs_m);
    __mram_ptr uint32_t   *mram_neighborIdxs =
        (__mram_ptr uint32_t *)   (DPU_MRAM_HEAP_POINTER + p.neighborIdxs_m);
    __mram_ptr AlignedU32 *mram_levels       =
        (__mram_ptr AlignedU32 *) (DPU_MRAM_HEAP_POINTER + p.levels_m);
    __mram_ptr uint32_t   *mram_l2g          =
        (__mram_ptr uint32_t *)   (DPU_MRAM_HEAP_POINTER + p.localToGlobal_m);

    barrier_wait(&barrier);

    // ── Tasklet-local buffers ─────────────────────────────────────────────
    // niBuf must hold NEIGHBOR_BATCH values plus up to 1 alignment-padding
    // slot on each side, so allocate NEIGHBOR_BATCH + 2 and keep it even.
    uint32_t   niBuf[NEIGHBOR_BATCH + 2] __attribute__((aligned(8)));
    AlignedU32 srcLvl[1]                 __attribute__((aligned(8)));
    AlignedU32 dstLvl[1]                 __attribute__((aligned(8)));
    uint32_t   gBuf[2]                   __attribute__((aligned(8)));

    bool local_change = false;

    // ── Partition nodes across tasklets ───────────────────────────────────
    for (uint32_t u = id; u < p.numNodes; u += NR_TASKLETS) {

        // Read nodePtrs[u] and nodePtrs[u+1] using the aligned helper.
        uint32_t start = read_nodeptr(mram_nodePtrs, u);
        uint32_t end   = read_nodeptr(mram_nodePtrs, u + 1);

        if (start == end) continue;

        // Read u's current level.
        mram_read(&mram_levels[u], srcLvl, sizeof(AlignedU32));
        uint32_t lvl_u = srcLvl[0].value;
        if (lvl_u == INF) continue;

        // Read u's global id (l2g entries are uint32_t; read as aligned pair).
        uint32_t aligned_u = u & ~1u;
        mram_read(&mram_l2g[aligned_u], gBuf, 2 * sizeof(uint32_t));
        uint32_t global_u = gBuf[u - aligned_u];

        // Walk u's neighbor list in batches.
        uint32_t ni = start;
        while (ni < end) {

            uint32_t batch = end - ni;
            if (batch > NEIGHBOR_BATCH) batch = NEIGHBOR_BATCH;

            // ── Aligned MRAM read for neighbor indices ────────────────────
            // mram_read requires an 8-byte aligned source address and an
            // 8-byte-multiple transfer size.  ni may be odd, so we align
            // the read start down to the nearest even index and read extra
            // slots to compensate, discarding the leading padding slot when
            // ni is odd.
            uint32_t aligned_ni = ni & ~1u;
            uint32_t skip       = ni - aligned_ni;          // 0 or 1
            // Total elements to fetch = skip + batch, rounded up to even.
            uint32_t fetch      = (skip + batch + 1u) & ~1u;
            // fetch never exceeds NEIGHBOR_BATCH + 2 (buf size).

            mram_read(&mram_neighborIdxs[aligned_ni],
                      niBuf,
                      fetch * sizeof(uint32_t));

            for (uint32_t k = 0; k < batch; k++) {
                uint32_t v = niBuf[skip + k];   // skip the alignment pad

                // Optimistic check — avoid most lock acquisitions.
                mram_read(&mram_levels[v], dstLvl, sizeof(AlignedU32));
                if (lvl_u + 1 >= dstLvl[0].value) continue;

                // Potential improvement — lock and re-check.
                mutex_lock(fallback_mutex);

                mram_read(&mram_levels[u], srcLvl, sizeof(AlignedU32));
                mram_read(&mram_levels[v], dstLvl, sizeof(AlignedU32));
                lvl_u = srcLvl[0].value;

                if (lvl_u != INF && lvl_u + 1 < dstLvl[0].value) {
                    dstLvl[0].value  = lvl_u + 1;
                    dstLvl[0].parent = global_u;
                    mram_write(dstLvl, &mram_levels[v], sizeof(AlignedU32));
                    local_change = true;
                }

                mutex_unlock(fallback_mutex);
            }

            ni += batch;
        }
    }

    // ── Merge per-tasklet change flag ─────────────────────────────────────
    barrier_wait(&barrier);

    if (local_change) {
        mutex_lock(fallback_mutex);
        shared_change = true;
        mutex_unlock(fallback_mutex);
    }

    barrier_wait(&barrier);

    // ── Tasklet 0 writes result flag to MRAM ─────────────────────────────
    if (id == 0) {
        AlignedU32 out __attribute__((aligned(8)));
        out.value  = shared_change ? 1u : 0u;
        out.parent = 0;
        mram_write(&out,
                   (__mram_ptr AlignedU32 *)(DPU_MRAM_HEAP_POINTER + p.changed_m),
                   sizeof(AlignedU32));
    }

    return 0;
}