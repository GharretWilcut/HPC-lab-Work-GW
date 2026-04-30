#ifndef DPU_PARAMS_LOCAL_H
#define DPU_PARAMS_LOCAL_H

#include <stdint.h>

// DPU Parameters for local traversal BFS
struct DPUParamsLocal {
    // Graph partitioning info
    uint32_t numGlobalNodes;        // Total nodes in entire graph
    uint32_t dpuStartNodeIdx;       // First node assigned to this DPU
    uint32_t dpuNumNodes;           // Number of nodes assigned to this DPU
    
    // Graph structure pointers (MRAM addresses)
    uint32_t dpuNodePtrs_m;         // Node adjacency list pointers
    uint32_t dpuNeighborIdxs_m;     // Neighbor indices array
    uint32_t dpuNodePtrsOffset;     // Offset to adjust node pointers
    
    // Local node state (MRAM addresses)
    uint32_t dpuNodeObjects_m;      // Array of NodeObjects for local nodes
    
    // Traversal list (MRAM addresses)
    uint32_t dpuTraversalList_m;    // Array of discovered NodeObjects
    uint32_t dpuTraversalCount_m;   // Pointer to count of traversed nodes
    uint32_t maxTraversalCapacity;  // Maximum capacity for traversal list
    
    // BFS source configuration
    uint32_t sourceNodeId;          // Global ID of BFS source node
    
    // Current iteration
    uint32_t currentLevel;          // Current BFS level being processed
};

#endif // DPU_PARAMS_LOCAL_H