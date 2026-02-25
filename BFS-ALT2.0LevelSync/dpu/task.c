#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <mram.h>
#include <defs.h>
#include <alloc.h>
#include <barrier.h>
#include <mutex.h>

#define NR_TASKLETS 16
#define INF 0xFFFFFFFF

// This is basically a hybrid CPU-DPU BFS implementation where the host CPU manages the global BFS state and each DPU is responsible 
// for processing its partition of the graph and updating local levels/parents. 
// The DPUs communicate back to the host using a "changed_flag" in DPU WRAM to indicate if they found any new nodes at the current level,
//  which tells the host whether another iteration is needed.

// This is a cross between the frontier approach that the example has and the complete partition approach that was in the original BFS-ALT2.0.
// Each DPU processes all its edges every iteration, 
// but only updates levels/parents for nodes that are at the current level.

//proposed improvements :
// --removing safe mutexes might fix this and make it faster, but we need to be careful
// --using broadcast to set the changed flag instead of each tasklet writing to main memory might also be faster, but 
// again we need to be careful about synchronization and ensuring all tasklets see the change in the same iteration


// Maximum nodes/edges per DPU partition (adjust as needed, but must fit in DPU MRAM)
#define MAX_NODES_PER_DPU 1000000
#define MAX_EDGES_PER_DPU 500000

typedef struct {
    uint32_t value;
    uint32_t padding;
} __attribute__((aligned(8))) AlignedU32;

typedef struct {
    uint32_t left;
    uint32_t right;
} __attribute__((aligned(8))) Edge;

typedef struct {
    uint32_t numNodes;
    uint32_t numEdges;
    uint32_t currentLevel;
    uint32_t _pad;           // keep struct size a multiple of 8
} DPUParams;


__mram_noinit DPUParams      param;
__mram_noinit AlignedU32     localToGlobal[MAX_NODES_PER_DPU];
__mram_noinit Edge           edges_arr[MAX_EDGES_PER_DPU];
__mram_noinit AlignedU32     levels[MAX_NODES_PER_DPU];
__mram_noinit AlignedU32     parents[MAX_NODES_PER_DPU];
__mram_noinit AlignedU32     changed_flag;

BARRIER_INIT(my_barrier, NR_TASKLETS);
MUTEX_INIT(my_mutex);

static volatile bool global_wram_change;

int main() {

    uint32_t id = me();
    // resets the global chaanged flag in DPU WRAM before starting the iteration
    if (id == 0) {
        mem_reset();
        global_wram_change = false;
    }

    barrier_wait(&my_barrier);

    DPUParams p;
    mram_read(&param, &p, sizeof(DPUParams));

    bool local_change = false;
    // Each tasklet processes a portion of the edges assigned to this DPU and updates levels/parents accordingly
    for (uint32_t i = id; i < p.numEdges; i += NR_TASKLETS) {

        Edge e;
        mram_read(&edges_arr[i], &e, sizeof(Edge));

        uint32_t left  = e.left;
        uint32_t right = e.right;

        AlignedU32 lvl_left, lvl_right;
        mram_read(&levels[left],  &lvl_left,  sizeof(AlignedU32));
        mram_read(&levels[right], &lvl_right, sizeof(AlignedU32));

        // Expand left → right
        if (lvl_left.value == p.currentLevel && lvl_right.value == INF) {
            mutex_lock(my_mutex);
            AlignedU32 check;
            mram_read(&levels[right], &check, sizeof(AlignedU32));
            if (check.value == INF) {
                AlignedU32 new_level = { p.currentLevel + 1, 0 };
                mram_write(&new_level, &levels[right], sizeof(AlignedU32));
                AlignedU32 parent_val;
                mram_read(&localToGlobal[left], &parent_val, sizeof(AlignedU32));
                mram_write(&parent_val, &parents[right], sizeof(AlignedU32));
                local_change = true;
            }
            mutex_unlock(my_mutex);
        }

        // Expand right → left
        if (lvl_right.value == p.currentLevel && lvl_left.value == INF) {
            mutex_lock(my_mutex);
            AlignedU32 check;
            mram_read(&levels[left], &check, sizeof(AlignedU32));
            if (check.value == INF) {
                AlignedU32 new_level = { p.currentLevel + 1, 0 };
                mram_write(&new_level, &levels[left], sizeof(AlignedU32));
                AlignedU32 parent_val;
                mram_read(&localToGlobal[right], &parent_val, sizeof(AlignedU32));
                mram_write(&parent_val, &parents[left], sizeof(AlignedU32));
                local_change = true;
            }
            mutex_unlock(my_mutex);
        }
    }

    barrier_wait(&my_barrier);
    // After processing all edges, if any tasklet found a change, set the global changed flag in WRAM so the host knows to continue iterating
    if (local_change) {
        mutex_lock(my_mutex);
        global_wram_change = true;
        mutex_unlock(my_mutex);
    }

    barrier_wait(&my_barrier);
    // Tasklet 0 writes the global changed flag back to main memory so the host can check if another iteration is needed
    if (id == 0 && global_wram_change) {
        AlignedU32 ch = { 1, 0 };
        mram_write(&ch, &changed_flag, sizeof(AlignedU32));
    }

    return 0;
}