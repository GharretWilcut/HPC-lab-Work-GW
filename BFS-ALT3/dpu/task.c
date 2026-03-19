// task.c

#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <mram.h>
#include <defs.h>
#include <barrier.h>
#include <mutex.h>
 
#define INF            0xFFFFFFFF
#define EDGE_BATCH     96      // edges fetched per tasklet per mram_read (
 
BARRIER_INIT(barrier, NR_TASKLETS);
MUTEX_INIT(fallback_mutex);
 
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
 
    Edge       ebuf[EDGE_BATCH] __attribute__((aligned(8)));
    AlignedU32 lbuf[2]          __attribute__((aligned(8)));
    AlignedU32 wbuf[1]          __attribute__((aligned(8)));
    AlignedU32 gbuf[1]          __attribute__((aligned(8)));
 
    bool local_change = false;
 
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
 
            // Relax L → R
            if (lvl_L != INF && lvl_L + 1 < lvl_R) {
                mutex_lock(fallback_mutex);
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
 
            // Relax R → L
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
 
    // Merge change flags
    barrier_wait(&barrier);
 
    if (local_change) {
        mutex_lock(fallback_mutex);
        shared_change = true;
        mutex_unlock(fallback_mutex);
    }
 
    barrier_wait(&barrier);
 
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