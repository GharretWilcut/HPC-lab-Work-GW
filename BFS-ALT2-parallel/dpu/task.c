#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>
#include <mram.h>
#include <perfcounter.h>
#include <defs.h>
#include <barrier.h>
#include <mutex.h>

#define INF 0xFFFFFFFF

MUTEX_INIT(level_mutex);
BARRIER_INIT(barrier, NR_TASKLETS);

/* MRAM structs */
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

bool shared_change;

int main() {

    if (me() == 0)
        shared_change = false;

    barrier_wait(&barrier);

    DPUParams p;
    __mram_ptr DPUParams *params_mram =
        (__mram_ptr DPUParams *)DPU_MRAM_HEAP_POINTER;

    mram_read(params_mram, &p, sizeof(DPUParams));

    __mram_ptr Edge *edges =
        (__mram_ptr Edge *)(DPU_MRAM_HEAP_POINTER + p.edges_m);

    __mram_ptr AlignedU32 *levels =
        (__mram_ptr AlignedU32 *)(DPU_MRAM_HEAP_POINTER + p.levels_m);

    __mram_ptr AlignedU32 *parents =
        (__mram_ptr AlignedU32 *)(DPU_MRAM_HEAP_POINTER + p.parents_m);

    __mram_ptr AlignedU32 *local_to_global =
        (__mram_ptr AlignedU32 *)(DPU_MRAM_HEAP_POINTER + p.localToGlobal_m);

    uint32_t id = me();

    bool local_change;

        if (id == 0)
            shared_change = false;

        barrier_wait(&barrier);

        local_change = false;

        for (uint32_t i = id; i < p.numEdges; i += NR_TASKLETS) {

            Edge e;
            mram_read(&edges[i], &e, sizeof(Edge));

            uint32_t left = e.left;
            uint32_t right = e.right;

            mutex_lock(level_mutex);

            AlignedU32 cur_left, cur_right;
            mram_read(&levels[left], &cur_left, sizeof(AlignedU32));
            mram_read(&levels[right], &cur_right, sizeof(AlignedU32));

            if (cur_left.value != INF &&
                cur_left.value + 1 < cur_right.value) {

                AlignedU32 new_val;
                new_val.value = cur_left.value + 1;
                new_val.padding = 0;

                mram_write(&new_val, &levels[right], sizeof(AlignedU32));

                AlignedU32 parent_val;
                mram_read(&local_to_global[left], &parent_val, sizeof(AlignedU32));
                mram_write(&parent_val, &parents[right], sizeof(AlignedU32));

                local_change = true;
            }

            if (cur_right.value != INF &&
                cur_right.value + 1 < cur_left.value) {

                AlignedU32 new_val;
                new_val.value = cur_right.value + 1;
                new_val.padding = 0;

                mram_write(&new_val, &levels[left], sizeof(AlignedU32));

                AlignedU32 parent_val;
                mram_read(&local_to_global[right], &parent_val, sizeof(AlignedU32));
                mram_write(&parent_val, &parents[left], sizeof(AlignedU32));

                local_change = true;
            }

            mutex_unlock(level_mutex);
        }
        barrier_wait(&barrier);

        if (local_change) {
            mutex_lock(level_mutex);
            shared_change = true;
            mutex_unlock(level_mutex);
        }

        barrier_wait(&barrier);


    if (id == 0) {
        AlignedU32 out;
        out.value = shared_change ? 1 : 0;
        out.padding = 0;

        __mram_ptr AlignedU32 *changed_ptr =
            (__mram_ptr AlignedU32 *)(DPU_MRAM_HEAP_POINTER + p.changed_m);

        mram_write(&out, changed_ptr, sizeof(AlignedU32));
    }
}

