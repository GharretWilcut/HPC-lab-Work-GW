// task.c — Two-stage sparse output BFS kernel (single launch per iteration)
//
// Protocol per BFS iteration:
//
//  DPU side (single launch):
//   - Each tasklet traverses its edge stripe, relaxes edges, and accumulates
//     ChangedNode records in a private WRAM buffer (changedBuf[id][]).
//   - After the barrier, tasklet 0:
//       a) Computes prefix-sum write bases so each tasklet has a disjoint
//          slice of the output array.
//       b) Writes totalChanged (uint64_t) to countOnly_m.
//   - Each tasklet then flushes its WRAM buffer to its slice in mram_changed[].
//   - Tasklet 0 writes the final compact array size to flushSize_m.
//
//  CPU side:
//   Step 1 — pull countOnly_m from all DPUs (8 bytes each, one push_xfer).
//   Step 2 — compute maxChanged = max(totalChanged) across all DPUs.
//   Step 3 — pull mram_changed[0..maxChanged-1] from all DPUs that reported
//             count > 0, using maxChanged as the uniform xfer size.
//             (Each DPU wrote exactly its own count of valid records,
//             tightly packed from index 0; the rest of the buffer is stale
//             but never read since the CPU uses each DPU's own count.)

#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <mram.h>
#include <defs.h>
#include <barrier.h>
#include <mutex.h>

#define INF                      0xFFFFFFFF
#define EDGE_BATCH               96
#define MAX_CHANGED_PER_TASKLET  64   // 64 × 16 bytes × 16 tasklets = 16 KB WRAM

BARRIER_INIT(barrier, NR_TASKLETS);
MUTEX_INIT(fallback_mutex);

// -----------------------------------------------------------------------
// Structs — must match gharret_utils.h exactly
// -----------------------------------------------------------------------
typedef struct {
    uint32_t numNodes;
    uint32_t numEdges;
    uint32_t localToGlobal_m;
    uint32_t edges_m;
    uint32_t levels_m;
    uint32_t changedNodes_m;    // tightly-packed output: ChangedNode[totalChanged]
    uint32_t countOnly_m;       // uint64_t output: total changed node count
    uint32_t _pad;
} DPUParams;  // 32 bytes

typedef struct { uint32_t left,  right;  } Edge;
typedef struct { uint32_t value; uint32_t parent; } AlignedU32;

typedef struct {
    uint32_t localID;
    uint32_t value;
    uint32_t parent;
    uint32_t _pad;
} ChangedNode;  // 16 bytes

typedef struct { uint32_t value; uint32_t _pad; } L2GEntry;

// -----------------------------------------------------------------------
// WRAM — per-tasklet buffers, no cross-tasklet locking needed
// -----------------------------------------------------------------------
static ChangedNode changedBuf[NR_TASKLETS][MAX_CHANGED_PER_TASKLET]
    __attribute__((aligned(8)));
static uint32_t changedCnt[NR_TASKLETS];

// Prefix-sum write bases: tasklet t writes to mram_changed[writeBase[t]].
// Computed by tasklet 0 after the traversal barrier.
static uint32_t writeBase[NR_TASKLETS];

// -----------------------------------------------------------------------
int main()
// -----------------------------------------------------------------------
{
    uint32_t id = me();
    changedCnt[id] = 0;

    DPUParams p;
    mram_read((__mram_ptr DPUParams *)DPU_MRAM_HEAP_POINTER,
              &p, sizeof(DPUParams));

    __mram_ptr Edge        *mram_edges   =
        (__mram_ptr Edge *)       (DPU_MRAM_HEAP_POINTER + p.edges_m);
    __mram_ptr AlignedU32  *mram_levels  =
        (__mram_ptr AlignedU32 *) (DPU_MRAM_HEAP_POINTER + p.levels_m);
    __mram_ptr L2GEntry    *mram_l2g     =
        (__mram_ptr L2GEntry *)   (DPU_MRAM_HEAP_POINTER + p.localToGlobal_m);
    __mram_ptr ChangedNode *mram_changed =
        (__mram_ptr ChangedNode *)(DPU_MRAM_HEAP_POINTER + p.changedNodes_m);

    barrier_wait(&barrier);

    Edge       ebuf[EDGE_BATCH] __attribute__((aligned(8)));
    AlignedU32 lbuf[2]          __attribute__((aligned(8)));
    AlignedU32 wbuf[1]          __attribute__((aligned(8)));
    L2GEntry   gbuf[1]          __attribute__((aligned(8)));

    // -----------------------------------------------------------------------
    // Full edge traversal — accumulate changes in per-tasklet WRAM buffer
    // -----------------------------------------------------------------------
    uint32_t i      = id * EDGE_BATCH;
    uint32_t stride = NR_TASKLETS * EDGE_BATCH;

    while (i < p.numEdges) {
        uint32_t fetch = EDGE_BATCH;
        if (i + fetch > p.numEdges) fetch = p.numEdges - i;

        mram_read(&mram_edges[i], ebuf, fetch * sizeof(Edge));

        for (uint32_t k = 0; k < fetch; k++) {
            uint32_t L = ebuf[k].left;
            uint32_t R = ebuf[k].right;

            mram_read(&mram_levels[L], &lbuf[0], sizeof(AlignedU32));
            mram_read(&mram_levels[R], &lbuf[1], sizeof(AlignedU32));
            uint32_t lvl_L = lbuf[0].value;
            uint32_t lvl_R = lbuf[1].value;

            // Relax L -> R
            if (lvl_L != INF && lvl_L + 1 < lvl_R) {
                mutex_lock(fallback_mutex);
                mram_read(&mram_levels[R], wbuf,     sizeof(AlignedU32));
                mram_read(&mram_levels[L], &lbuf[0], sizeof(AlignedU32));
                lvl_L = lbuf[0].value;
                if (lvl_L != INF && lvl_L + 1 < wbuf[0].value) {
                    mram_read(&mram_l2g[L], gbuf, sizeof(L2GEntry));
                    wbuf[0].value  = lvl_L + 1;
                    wbuf[0].parent = gbuf[0].value;
                    mram_write(wbuf, &mram_levels[R], sizeof(AlignedU32));
                    if (changedCnt[id] < MAX_CHANGED_PER_TASKLET) {
                        changedBuf[id][changedCnt[id]].localID = R;
                        changedBuf[id][changedCnt[id]].value   = wbuf[0].value;
                        changedBuf[id][changedCnt[id]].parent  = wbuf[0].parent;
                        changedCnt[id]++;
                    }
                }
                mutex_unlock(fallback_mutex);
            }

            mram_read(&mram_levels[L], &lbuf[0], sizeof(AlignedU32));
            mram_read(&mram_levels[R], &lbuf[1], sizeof(AlignedU32));
            lvl_L = lbuf[0].value;
            lvl_R = lbuf[1].value;

            // Relax R -> L
            if (lvl_R != INF && lvl_R + 1 < lvl_L) {
                mutex_lock(fallback_mutex);
                mram_read(&mram_levels[L], wbuf,     sizeof(AlignedU32));
                mram_read(&mram_levels[R], &lbuf[1], sizeof(AlignedU32));
                lvl_R = lbuf[1].value;
                if (lvl_R != INF && lvl_R + 1 < wbuf[0].value) {
                    mram_read(&mram_l2g[R], gbuf, sizeof(L2GEntry));
                    wbuf[0].value  = lvl_R + 1;
                    wbuf[0].parent = gbuf[0].value;
                    mram_write(wbuf, &mram_levels[L], sizeof(AlignedU32));
                    if (changedCnt[id] < MAX_CHANGED_PER_TASKLET) {
                        changedBuf[id][changedCnt[id]].localID = L;
                        changedBuf[id][changedCnt[id]].value   = wbuf[0].value;
                        changedBuf[id][changedCnt[id]].parent  = wbuf[0].parent;
                        changedCnt[id]++;
                    }
                }
                mutex_unlock(fallback_mutex);
            }
        }
        i += stride;
    }

    // -----------------------------------------------------------------------
    // Barrier 1: all tasklets done traversing.
    // Tasklet 0 computes prefix-sum write bases and writes totalChanged.
    // -----------------------------------------------------------------------
    barrier_wait(&barrier);

    if (id == 0) {
        uint32_t total = 0;
        for (uint32_t t = 0; t < NR_TASKLETS; t++) {
            writeBase[t] = total;
            total += changedCnt[t];
        }
        // Write totalChanged to MRAM as a uint64_t (8-byte aligned)
        uint64_t count_out __attribute__((aligned(8)));
        count_out = (uint64_t)total;
        mram_write(&count_out,
                   (__mram_ptr void *)(DPU_MRAM_HEAP_POINTER + p.countOnly_m),
                   sizeof(uint64_t));
    }

    // -----------------------------------------------------------------------
    // Barrier 2: writeBase[] is now valid for all tasklets.
    // Each tasklet flushes its WRAM slice to its assigned MRAM region.
    // Records are tightly packed from index 0 (no per-tasklet gaps).
    // -----------------------------------------------------------------------
    barrier_wait(&barrier);

    if (changedCnt[id] > 0) {
        mram_write(changedBuf[id],
                   &mram_changed[writeBase[id]],
                   changedCnt[id] * sizeof(ChangedNode));
    }

    return 0;
}