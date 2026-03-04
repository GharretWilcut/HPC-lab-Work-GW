// task.c 

#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <mram.h>
#include <defs.h>
#include <barrier.h>
#include <mutex.h>
#include <alloc.h>

#define INF          0xFFFFFFFF
#define EDGE_BATCH   8          // edges fetched per mram_read
#define MAX_NODES    4096       // adjust to your graph's maximum local nodes

BARRIER_INIT(barrier, NR_TASKLETS);
MUTEX_INIT(merge_mutex);

//structs in MRAM
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

typedef struct {
    uint32_t left;
    uint32_t right;
} Edge;

typedef struct {
    uint32_t value;
    uint32_t padding;
} AlignedU32;


static AlignedU32 wram_levels [MAX_NODES];
static AlignedU32 wram_parents[MAX_NODES];

static uint8_t dirty[NR_TASKLETS][MAX_NODES];

static bool shared_change;

//helper functions 

static void load_aligned_array(__mram_ptr AlignedU32 *src,
                               AlignedU32 *dst,
                               uint32_t count)
{
    uint32_t bytes = count * sizeof(AlignedU32); // each element = 8 bytes
    mram_read(src, dst, bytes);
}

static void store_aligned_array(AlignedU32 *src,
                                __mram_ptr AlignedU32 *dst,
                                uint32_t count)
{
    uint32_t bytes = count * sizeof(AlignedU32);
    mram_write(src, dst, bytes);
}


int main()
{
    uint32_t id = me();

    //load parameters from MRAM
    DPUParams p;
    mram_read((__mram_ptr DPUParams *)DPU_MRAM_HEAP_POINTER,
              &p, sizeof(DPUParams));

    __mram_ptr Edge      *mram_edges  =
        (__mram_ptr Edge *)     (DPU_MRAM_HEAP_POINTER + p.edges_m);
    __mram_ptr AlignedU32 *mram_levels =
        (__mram_ptr AlignedU32 *)(DPU_MRAM_HEAP_POINTER + p.levels_m);
    __mram_ptr AlignedU32 *mram_parents=
        (__mram_ptr AlignedU32 *)(DPU_MRAM_HEAP_POINTER + p.parents_m);
    __mram_ptr AlignedU32 *mram_l2g   =
        (__mram_ptr AlignedU32 *)(DPU_MRAM_HEAP_POINTER + p.localToGlobal_m);


    
    if (id == 0) {
        shared_change = false;
        load_aligned_array(mram_levels,  wram_levels,  p.numNodes);
        load_aligned_array(mram_parents, wram_parents, p.numNodes);
    }

    memset(dirty[id], 0, p.numNodes);

    barrier_wait(&barrier); 


    Edge batch[EDGE_BATCH];

    bool local_change = false;

    uint32_t i = id * EDGE_BATCH;         
    uint32_t stride = NR_TASKLETS * EDGE_BATCH;

    while (i < p.numEdges) {
        uint32_t fetch = EDGE_BATCH;
        if (i + fetch > p.numEdges)
            fetch = p.numEdges - i;

        uint32_t fetch_bytes = fetch * sizeof(Edge);
        if (fetch_bytes & 7) fetch_bytes = (fetch_bytes + 7) & ~7;

        mram_read(&mram_edges[i], batch, fetch_bytes);

        for (uint32_t k = 0; k < fetch; k++) {
            uint32_t L = batch[k].left;
            uint32_t R = batch[k].right;

            uint32_t lvl_L = wram_levels[L].value;
            uint32_t lvl_R = wram_levels[R].value;

            if (lvl_L != INF && lvl_L + 1 < lvl_R) {
                wram_levels [R].value   = lvl_L + 1;
                wram_parents[R].value   = wram_levels[L].value; 
                wram_parents[R].padding = L; 
                dirty[id][R] = 1;
                local_change = true;
                lvl_R = lvl_L + 1; 
            }

            // Relax R → L
            if (lvl_R != INF && lvl_R + 1 < lvl_L) {
                wram_levels [L].value   = lvl_R + 1;
                wram_parents[L].padding = R; 
                dirty[id][L] = 1;
                local_change = true;
            }
        }

        i += stride;
    }

    /* ── 4. Merge: resolve conflicts between tasklets ───────────────── */
    barrier_wait(&barrier);
    for (uint32_t n = 0; n < p.numNodes; n++) {
        if (!dirty[id][n]) continue;

        mutex_lock(merge_mutex);
        if (wram_levels[n].value < wram_levels[n].value) {
            //intentionally blank
        }
        {
            uint32_t parent_local = wram_parents[n].padding;
            AlignedU32 g;
            mram_read(&mram_l2g[parent_local], &g, sizeof(AlignedU32));
            wram_parents[n].value   = g.value;
            wram_parents[n].padding = 0;
        }
        mutex_unlock(merge_mutex);
    }

    barrier_wait(&barrier);


    if (id == 0) {
        bool any_dirty = false;
        for (uint32_t t = 0; t < NR_TASKLETS; t++) {
            for (uint32_t n = 0; n < p.numNodes; n++) {
                if (dirty[t][n]) { any_dirty = true; goto done_scan; }
            }
        }
        done_scan:;

        if (any_dirty) {
            store_aligned_array(wram_levels,  mram_levels,  p.numNodes);
            store_aligned_array(wram_parents, mram_parents, p.numNodes);
        }

        AlignedU32 out;
        out.value   = any_dirty ? 1 : 0;
        out.padding = 0;
        mram_write(&out,
                   (__mram_ptr AlignedU32 *)(DPU_MRAM_HEAP_POINTER + p.changed_m),
                   sizeof(AlignedU32));
    }

    return 0;
}