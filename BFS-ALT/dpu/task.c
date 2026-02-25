/*
 * BFS with Local Traversal List (SLICED CSR VERSION)
 * Each DPU owns a local CSR slice and global neighbor IDs
 */
#include <stdio.h>
#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <mutex.h>
#include <perfcounter.h>

#include "../support/common.h"

struct NodeObject {
    uint32_t nodeId;     // Global node ID
    uint32_t parentId;
    uint32_t level;
};

struct DPUParamsLocal {
    uint32_t numGlobalNodes;
    uint32_t dpuStartNodeIdx;
    uint32_t dpuNumNodes;

    uint32_t dpuNodePtrs_m;       // LOCAL CSR (numNodes + 1)
    uint32_t dpuNeighborIdxs_m;   // LOCAL edge list (GLOBAL neighbor IDs)

    uint32_t dpuNodeObjects_m;
    uint32_t dpuTraversalList_m;
    uint32_t dpuTraversalCount_m;
    uint32_t maxTraversalCapacity;

    uint32_t sourceNodeId;
    uint32_t currentLevel;
};

static inline uint32_t load4B(uint32_t base_m, uint32_t idx, uint64_t *cache_w) {
    mram_read((__mram_ptr void const *)(base_m + idx * sizeof(uint32_t)),
              cache_w, sizeof(uint64_t));
    return ((uint32_t *)cache_w)[0];
}

BARRIER_INIT(initBarrier, NR_TASKLETS);
BARRIER_INIT(bfsBarrier, NR_TASKLETS);
MUTEX_INIT(traversalMutex);

__host uint32_t shared_traversal_count;

int main() {

    /* ---------------- Load parameters ---------------- */
    uint32_t params_m = (uint32_t)DPU_MRAM_HEAP_POINTER;
    struct DPUParamsLocal *p =
        (struct DPUParamsLocal *)mem_alloc(
            ROUND_UP_TO_MULTIPLE_OF_8(sizeof(struct DPUParamsLocal)));

    mram_read((__mram_ptr void const *)params_m, p,
              ROUND_UP_TO_MULTIPLE_OF_8(sizeof(struct DPUParamsLocal)));

    uint32_t startNodeIdx = p->dpuStartNodeIdx;
    uint32_t numNodes     = p->dpuNumNodes;

    uint32_t nodePtrs_m       = p->dpuNodePtrs_m;
    uint32_t neighborIdxs_m   = p->dpuNeighborIdxs_m;
    uint32_t nodeObjects_m    = p->dpuNodeObjects_m;
    uint32_t traversalList_m  = p->dpuTraversalList_m;

    uint32_t maxTraversalCapacity = p->maxTraversalCapacity;
    uint32_t sourceNodeId = p->sourceNodeId;
    uint32_t currentLevel = p->currentLevel;

    if (me() == 0) {
        uint64_t tmp;
        mram_read((__mram_ptr void *)p->dpuTraversalCount_m,
                  &tmp, sizeof(uint64_t));
        shared_traversal_count = (uint32_t)tmp;
    }
    barrier_wait(&initBarrier);

    if (numNodes == 0)
        return 0;

    uint64_t *cache_w = (uint64_t *)mem_alloc(sizeof(uint64_t));
    uint64_t *nodeObjCache_w = (uint64_t *)mem_alloc(16);

    /* ---------------- Work partition ---------------- */
    uint32_t nodesPerTasklet =
        (numNodes + NR_TASKLETS - 1) / NR_TASKLETS;

    uint32_t localStart = me() * nodesPerTasklet;
    uint32_t localEnd =
        (localStart + nodesPerTasklet > numNodes)
            ? numNodes
            : localStart + nodesPerTasklet;

    

    

    barrier_wait(&bfsBarrier);

    /* ---------------- BFS expansion ---------------- */
    for (uint32_t localIdx = localStart; localIdx < localEnd; ++localIdx) {

        mram_read((__mram_ptr void const *)(nodeObjects_m + localIdx * 16),
                  nodeObjCache_w, 16);
        struct NodeObject cur =
            ((struct NodeObject *)nodeObjCache_w)[0];

        if (cur.level != currentLevel)
            continue;

        uint32_t edgeStart =
            load4B(nodePtrs_m, localIdx, cache_w);
        uint32_t edgeEnd =
            load4B(nodePtrs_m, localIdx + 1, cache_w);

        for (uint32_t e = edgeStart; e < edgeEnd; ++e) {

            uint32_t neighborId =
                load4B(neighborIdxs_m, e, cache_w);

            struct NodeObject next = {
                .nodeId   = neighborId,
                .parentId = cur.nodeId,
                .level    = currentLevel + 1
            };

            if (neighborId >= startNodeIdx &&
                neighborId < startNodeIdx + numNodes) {

                uint32_t nLocal = neighborId - startNodeIdx;
                mram_read((__mram_ptr void const *)
                              (nodeObjects_m + nLocal * 16),
                          nodeObjCache_w, 16);

                struct NodeObject old =
                    ((struct NodeObject *)nodeObjCache_w)[0];

                if (old.level == UINT32_MAX) {
                    ((struct NodeObject *)nodeObjCache_w)[0] = next;
                    mram_write(nodeObjCache_w,
                               (__mram_ptr void *)
                                   (nodeObjects_m + nLocal * 16),
                               16);
                }

            } else {
                mutex_id_t m = MUTEX_GET(traversalMutex);
                mutex_lock(m);

                if (shared_traversal_count < maxTraversalCapacity) {
                    ((struct NodeObject *)nodeObjCache_w)[0] = next;
                    mram_write(nodeObjCache_w,
                               (__mram_ptr void *)
                                   (traversalList_m +
                                    shared_traversal_count * 16),
                               16);
                    shared_traversal_count++;
                }

                mutex_unlock(m);
            }
        }
    }

    barrier_wait(&bfsBarrier);

    /* ---------------- Write back traversal count ---------------- */
    if (me() == 0) {
        *cache_w = (uint64_t)shared_traversal_count;
        mram_write(cache_w,
                   (__mram_ptr void *)p->dpuTraversalCount_m,
                   sizeof(uint64_t));
        printf("DPU [%u..%u) traversal_count=%u\n",
               startNodeIdx,
               startNodeIdx + numNodes,
               shared_traversal_count);
    }

    return 0;
}
