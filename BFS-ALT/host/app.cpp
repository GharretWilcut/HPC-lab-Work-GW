// app_local.cpp - Host application for Local Traversal BFS on DPU
extern "C" {
#include <dpu.h>
#include <dpu_log.h>
}

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <getopt.h>
#include <algorithm>
#include <vector>
#include <unordered_map>

#include "mram-management.h"
#include "../support/common.h"
#include "../support/graph.h"
#include "../support/params.h"
#include "../support/timer.h"
#include "../support/utils.h"

#ifndef NR_DPUS
#define NR_DPUS 8  // Or whatever number you want
#endif
// NodeObject structure (must match DPU side)
struct NodeObject {
    uint32_t nodeId;
    uint32_t parentId;
    uint32_t level;
};

// Initialize a NodeObject
void initializeNodeObject(struct NodeObject* nodeObj, uint32_t id) {
    nodeObj->nodeId = id;
    nodeObj->parentId = UINT32_MAX;
    nodeObj->level = UINT32_MAX;
}

// DPU Parameters structure (must match DPU side)
struct DPUParamsLocal {
    uint32_t numGlobalNodes;
    uint32_t dpuStartNodeIdx;
    uint32_t dpuNumNodes;
    uint32_t dpuNodePtrs_m;
    uint32_t dpuNeighborIdxs_m;
    uint32_t dpuNodePtrsOffset;
    uint32_t dpuNodeObjects_m;
    uint32_t dpuTraversalList_m;
    uint32_t dpuTraversalCount_m;
    uint32_t maxTraversalCapacity;
    uint32_t sourceNodeId;
    uint32_t currentLevel;
};

#ifndef ENERGY
#define ENERGY 0
#endif

#if ENERGY
#include <dpu_probe.h>
#endif

#define DPU_BINARY "./bin/dpu_code_local"

// Change from dpu_t* to dpu_set_t
void copyNodeObjectsToDPU(dpu_set_t dpu, const NodeObject* src, uint32_t dst_m, uint32_t count) {
    size_t bufferSize = count * 16;
    uint8_t* buffer = (uint8_t*)calloc(bufferSize, 1);
    
    for (uint32_t i = 0; i < count; i++) {
        memcpy(buffer + i * 16, &src[i], sizeof(NodeObject));
    }
    
    copyToDPU(dpu, buffer, dst_m, bufferSize);
    free(buffer);
}

void copyNodeObjectsFromDPU(dpu_set_t dpu, uint32_t src_m, NodeObject* dst, uint32_t count) {
    size_t bufferSize = count * 16;
    uint8_t* buffer = (uint8_t*)calloc(bufferSize, 1);
    
    copyFromDPU(dpu, src_m, buffer, bufferSize);
    
    for (uint32_t i = 0; i < count; i++) {
        memcpy(&dst[i], buffer + i * 16, sizeof(NodeObject));
    }
    free(buffer);
}

// Helper function to merge traversal lists from all DPUs
void mergeTraversalLists(
    std::vector<NodeObject>& globalTraversalList,
    std::unordered_map<uint32_t, NodeObject>& bestNodes) {
    
    for (const auto& node : globalTraversalList) {
        auto it = bestNodes.find(node.nodeId);
        if (it == bestNodes.end() || node.level < it->second.level) {
            bestNodes[node.nodeId] = node;
        }
    }
}

int main(int argc, char** argv) {
    Params params = input_params(argc, argv);
    
    Timer timer;
    float loadTime = 0.0f, dpuTime = 0.0f, hostTime = 0.0f, retrieveTime = 0.0f;
    
    #if ENERGY
    dpu_probe_t probe;
    DPU_ASSERT(dpu_probe_init("energy_probe", &probe));
    double tenergy = 0.0;
    #endif
    
    // Allocate DPUs
    dpu_set_t dpuSet, dpu;
    uint32_t numDPUs;
    dpu_program_t* program = nullptr;
    
    DPU_ASSERT(dpu_alloc(NR_DPUS, nullptr, &dpuSet));
    DPU_ASSERT(dpu_load(dpuSet, DPU_BINARY, &program));
    DPU_ASSERT(dpu_get_nr_dpus(dpuSet, &numDPUs));
    
    PRINT_INFO(params.verbosity >= 1, "Allocated %u DPUs", numDPUs);
    PRINT_INFO(params.verbosity >= 1, "Reading graph %s", params.fileName);
    
    // Load graph
    COOGraph cooGraph = readCOOGraph(params.fileName);
    PRINT_INFO(params.verbosity >= 1, "    Graph has %d nodes and %d edges",
               cooGraph.numNodes, cooGraph.numEdges);
    
    CSRGraph csrGraph = coo2csr(cooGraph);
    freeCOOGraph(cooGraph);
    
    uint32_t numNodes = csrGraph.numNodes;
    uint32_t* nodePtrs = csrGraph.nodePtrs;
    uint32_t* neighborIdxs = csrGraph.neighborIdxs;
    
    // Initialize node objects (host-side tracking)
    NodeObject* nodeObjects = static_cast<NodeObject*>(
        calloc(numNodes, sizeof(NodeObject)));
    for (uint32_t i = 0; i < numNodes; ++i) {
        initializeNodeObject(&nodeObjects[i], i);
    }
    
    // Set source node
    uint32_t sourceNodeId = 0; // Start BFS from node 0
    nodeObjects[sourceNodeId].parentId = sourceNodeId;
    nodeObjects[sourceNodeId].level = 0;
    
    // Calculate nodes per DPU
    uint32_t numNodesPerDPU = ROUND_UP_TO_MULTIPLE_OF_64((numNodes - 1) / numDPUs + 1);
    PRINT_INFO(params.verbosity >= 1, "Assigning %u nodes per DPU", numNodesPerDPU);
    
    // Prepare DPU parameters
    DPUParamsLocal dpuParams[numDPUs];
    uint32_t dpuParams_m[numDPUs];
    
    // Maximum traversal capacity per DPU (conservative estimate)
    uint32_t maxTraversalCapacity = numNodes; // Can be tuned based on MRAM size
    
    // Initialize DPUs
    unsigned int dpuIdx = 0;
    DPU_FOREACH(dpuSet, dpu) {
        mram_heap_allocator_t allocator;
        init_allocator(&allocator);
        
        dpuParams_m[dpuIdx] = mram_heap_alloc(&allocator, 
            ROUND_UP_TO_MULTIPLE_OF_8(sizeof(DPUParamsLocal)));
        
        uint32_t dpuStartNodeIdx = dpuIdx * numNodesPerDPU;
        uint32_t dpuNumNodes = 0;
        if (dpuStartNodeIdx < numNodes) {
            dpuNumNodes = std::min(numNodes - dpuStartNodeIdx, numNodesPerDPU);
        }
        
        PRINT_INFO(params.verbosity >= 2, "    DPU %u:", dpuIdx);
        PRINT_INFO(params.verbosity >= 2, "        Receives %u nodes", dpuNumNodes);
        
        if (dpuNumNodes > 0) {
            uint32_t* dpuNodePtrs_h = &nodePtrs[dpuStartNodeIdx];
            uint32_t dpuNodePtrsOffset = dpuNodePtrs_h[0];
            uint32_t* dpuNeighborIdxs_h = neighborIdxs + dpuNodePtrsOffset;
            uint32_t dpuNumNeighbors = dpuNodePtrs_h[dpuNumNodes] - dpuNodePtrsOffset;
            
            // Allocate MRAM for graph structure
            uint32_t dpuNodePtrs_m = mram_heap_alloc(&allocator, 
                (dpuNumNodes + 1) * sizeof(uint32_t));
            uint32_t dpuNeighborIdxs_m = mram_heap_alloc(&allocator, 
                dpuNumNeighbors * sizeof(uint32_t));
            
            // Allocate MRAM for node objects (16 bytes each for alignment)
            uint32_t dpuNodeObjects_m = mram_heap_alloc(&allocator, 
                dpuNumNodes * 16);
            
            // Allocate MRAM for traversal list (16 bytes per entry)
            uint32_t dpuTraversalList_m = mram_heap_alloc(&allocator, 
                maxTraversalCapacity * 16);
            uint32_t dpuTraversalCount_m = mram_heap_alloc(&allocator, 
                sizeof(uint64_t));  // 8-byte aligned
            
            PRINT_INFO(params.verbosity >= 2, 
                      "        Total memory allocated: %d bytes", 
                      allocator.totalAllocated);
            
            // Set parameters
            dpuParams[dpuIdx].numGlobalNodes = numNodes;
            dpuParams[dpuIdx].dpuStartNodeIdx = dpuStartNodeIdx;
            dpuParams[dpuIdx].dpuNumNodes = dpuNumNodes;
            dpuParams[dpuIdx].dpuNodePtrs_m = dpuNodePtrs_m;
            dpuParams[dpuIdx].dpuNeighborIdxs_m = dpuNeighborIdxs_m;
            dpuParams[dpuIdx].dpuNodePtrsOffset = dpuNodePtrsOffset;
            dpuParams[dpuIdx].dpuNodeObjects_m = dpuNodeObjects_m;
            dpuParams[dpuIdx].dpuTraversalList_m = dpuTraversalList_m;
            dpuParams[dpuIdx].dpuTraversalCount_m = dpuTraversalCount_m;
            dpuParams[dpuIdx].maxTraversalCapacity = maxTraversalCapacity;
            dpuParams[dpuIdx].sourceNodeId = sourceNodeId;
            dpuParams[dpuIdx].currentLevel = 0;
            
            // Copy graph structure to DPU
            PRINT_INFO(params.verbosity >= 2, "        Copying data to DPU");
            startTimer(&timer);
            
            copyToDPU(dpu, reinterpret_cast<const uint8_t*>(dpuNodePtrs_h),
                     dpuNodePtrs_m, (dpuNumNodes + 1) * sizeof(uint32_t));
            
            copyToDPU(dpu, reinterpret_cast<const uint8_t*>(dpuNeighborIdxs_h),
                     dpuNeighborIdxs_m, dpuNumNeighbors * sizeof(uint32_t));
            
            // Initialize node objects on DPU using aligned copy
            NodeObject* dpuNodeObjects_h = &nodeObjects[dpuStartNodeIdx];
            copyNodeObjectsToDPU(dpu, dpuNodeObjects_h, dpuNodeObjects_m, dpuNumNodes);
            
            // Initialize traversal count to 0 (8-byte aligned)
            uint64_t initialCount = 0;
            copyToDPU(dpu, reinterpret_cast<const uint8_t*>(&initialCount),
                     dpuTraversalCount_m, sizeof(uint64_t));
            
            stopTimer(&timer);
            loadTime += getElapsedTime(timer);
        }
        
        // Copy parameters to DPU
        PRINT_INFO(params.verbosity >= 2, "        Copying parameters to DPU");
        startTimer(&timer);
        copyToDPU(dpu, reinterpret_cast<const uint8_t*>(&dpuParams[dpuIdx]),
                 dpuParams_m[dpuIdx], 
                 ROUND_UP_TO_MULTIPLE_OF_8(sizeof(DPUParamsLocal)));
        stopTimer(&timer);
        loadTime += getElapsedTime(timer);
        
        ++dpuIdx;
    }
    
    PRINT_INFO(params.verbosity >= 1, "    CPU-DPU Time: %f ms", loadTime * 1e3);
    
    // BFS iterations
    uint32_t currentLevel = 0;
    bool hasActiveNodes = true;
    std::unordered_map<uint32_t, NodeObject> pendingUpdates;
    
    while (hasActiveNodes) {
        PRINT_INFO(params.verbosity >= 1, 
                "Processing BFS level %u", currentLevel);
        
        // Launch DPUs
        startTimer(&timer);
        DPU_ASSERT(dpu_launch(dpuSet, DPU_SYNCHRONOUS));
        stopTimer(&timer);
        dpuTime += getElapsedTime(timer);
        
        // Collect results
        startTimer(&timer);
        std::vector<NodeObject> globalTraversalList;
        pendingUpdates.clear();
        
        dpuIdx = 0;
        DPU_FOREACH(dpuSet, dpu) {
            uint32_t dpuNumNodes = dpuParams[dpuIdx].dpuNumNodes;
            if (dpuNumNodes > 0) {
                // Get traversal count
                uint64_t dpuTraversalCount64;
                copyFromDPU(dpu, dpuParams[dpuIdx].dpuTraversalCount_m,
                        reinterpret_cast<uint8_t*>(&dpuTraversalCount64),
                        sizeof(uint64_t));
                uint32_t dpuTraversalCount = (uint32_t)dpuTraversalCount64;
                
                if (dpuTraversalCount > 0) {
                    // Retrieve traversal list
                    NodeObject* traversalBuffer = static_cast<NodeObject*>(
                        calloc(dpuTraversalCount, sizeof(NodeObject)));
                    copyNodeObjectsFromDPU(dpu, dpuParams[dpuIdx].dpuTraversalList_m,
                                        traversalBuffer, dpuTraversalCount);
                    for (uint32_t i = 0; i < dpuTraversalCount; ++i) {
                        globalTraversalList.push_back(traversalBuffer[i]);
                    }
                    free(traversalBuffer);
                }
                
                // Retrieve updated node objects
                uint32_t dpuStartNodeIdx = dpuIdx * numNodesPerDPU;
                copyNodeObjectsFromDPU(dpu, dpuParams[dpuIdx].dpuNodeObjects_m,
                                    &nodeObjects[dpuStartNodeIdx], dpuNumNodes);
                    
                if (params.verbosity >= 2) {
                    uint32_t nodesAtLevel1 = 0;
                    for (uint32_t i = 0; i < dpuNumNodes; i++) {
                        if (nodeObjects[dpuStartNodeIdx + i].level == 1) {
                            nodesAtLevel1++;
                        }
                    }
                    PRINT_INFO(params.verbosity >= 2, "    DPU %u has %u nodes at level 1", dpuIdx, nodesAtLevel1);
                    }
                }
                ++dpuIdx;
            }
        
        // Merge cross-partition discoveries
        mergeTraversalLists(globalTraversalList, pendingUpdates);
        
        // Apply pending updates
        for (const auto& pair : pendingUpdates) {
            uint32_t nodeId = pair.first;
            const NodeObject& newNode = pair.second;
            if (nodeObjects[nodeId].level == UINT32_MAX || 
                newNode.level < nodeObjects[nodeId].level) {
                nodeObjects[nodeId] = newNode;
            }
        }
        
        stopTimer(&timer);
        hostTime += getElapsedTime(timer);
        
        // Check if there is work for NEXT level
        hasActiveNodes = false;
        for (uint32_t i = 0; i < numNodes; ++i) {
            if (nodeObjects[i].level == currentLevel + 1) {
                hasActiveNodes = true;
                break;
            }
        }

        // Advance level ONLY if work exists
        if (hasActiveNodes) {
            ++currentLevel;
        }

        
        // If there's more work, update DPU parameters
        if (hasActiveNodes) {
            startTimer(&timer);
            dpuIdx = 0;
            DPU_FOREACH(dpuSet, dpu) {
                uint32_t dpuNumNodes = dpuParams[dpuIdx].dpuNumNodes;

                if (dpuNumNodes > 0) {
                    uint32_t dpuStartNodeIdx = dpuIdx * numNodesPerDPU;

                    // 1) Copy updated node objects
                    copyNodeObjectsToDPU(
                        dpu,
                        &nodeObjects[dpuStartNodeIdx],
                        dpuParams[dpuIdx].dpuNodeObjects_m,
                        dpuNumNodes
                    );

                    // 2) Reset traversal count
                    uint64_t resetCount = 0;
                    copyToDPU(
                        dpu,
                        reinterpret_cast<const uint8_t*>(&resetCount),
                        dpuParams[dpuIdx].dpuTraversalCount_m,
                        sizeof(uint64_t)
                    );

                    // 3) Update BFS level
                    dpuParams[dpuIdx].currentLevel = currentLevel;

                    // 4) Copy params back to DPU
                    copyToDPU(
                        dpu,
                        reinterpret_cast<const uint8_t*>(&dpuParams[dpuIdx]),
                        dpuParams_m[dpuIdx],
                        ROUND_UP_TO_MULTIPLE_OF_8(sizeof(DPUParamsLocal))
                    );
                }

                ++dpuIdx;
            }

            stopTimer(&timer);
            hostTime += getElapsedTime(timer);
        }
    }
    
    PRINT_INFO(params.verbosity >= 1, "DPU Kernel Time: %f ms", dpuTime * 1e3);
    PRINT_INFO(params.verbosity >= 1, "Inter-DPU Time: %f ms", hostTime * 1e3);
    #if ENERGY
    PRINT_INFO(params.verbosity >= 1, "    DPU Energy: %f J", tenergy);
    #endif
    
    // Verification: CPU BFS
    PRINT_INFO(params.verbosity >= 1, "Verifying with CPU BFS");
    uint32_t* nodeLevelReference = static_cast<uint32_t*>(
        calloc(numNodes, sizeof(uint32_t)));
    for (uint32_t i = 0; i < numNodes; ++i) {
        nodeLevelReference[i] = UINT32_MAX;
    }
    nodeLevelReference[sourceNodeId] = 0;
    
    std::vector<uint32_t> queue;
    queue.push_back(sourceNodeId);
    size_t queuePos = 0;
    
    while (queuePos < queue.size()) {
        uint32_t node = queue[queuePos++];
        uint32_t nodeLevel = nodeLevelReference[node];
        
        for (uint32_t i = nodePtrs[node]; i < nodePtrs[node + 1]; ++i) {
            uint32_t neighbor = neighborIdxs[i];
            if (nodeLevelReference[neighbor] == UINT32_MAX) {
                nodeLevelReference[neighbor] = nodeLevel + 1;
                queue.push_back(neighbor);
            }
        }
    }
    
    // Compare results
    uint32_t mismatches = 0;
    for (uint32_t i = 0; i < numNodes; ++i) {
        if (nodeObjects[i].level != nodeLevelReference[i]) {
            if (mismatches < 10) { // Print first 10 mismatches
                PRINT_ERROR("Mismatch at node %u: DPU=%u, CPU=%u",
                           i, nodeObjects[i].level, nodeLevelReference[i]);
            }
            ++mismatches;
        }
    }
    
    if (mismatches == 0) {
        PRINT_INFO(params.verbosity >= 1, "✓ Verification PASSED");
    } else {
        PRINT_ERROR("✗ Verification FAILED: %u mismatches", mismatches);
    }
    
    // Display DPU logs
    if (params.verbosity >= 2) {
        PRINT_INFO(params.verbosity >= 2, "Displaying DPU Logs:");
        dpuIdx = 0;
        DPU_FOREACH(dpuSet, dpu) {
            PRINT("DPU %u:", dpuIdx);
            dpu_error_t log_result = dpu_log_read(dpu, stdout);
            if (log_result != DPU_OK && log_result != DPU_ERR_LOG_CONTEXT_MISSING) {
                DPU_ASSERT(log_result);
            }
            ++dpuIdx;
        }
    }
    
    if (params.verbosity == 0) {
        PRINT("CPU-DPU: %f ms | DPU Kernel: %f ms | Inter-DPU: %f ms",
              loadTime * 1e3, dpuTime * 1e3, hostTime * 1e3);
    }
    
    // Cleanup
    freeCSRGraph(csrGraph);
    free(nodeObjects);
    free(nodeLevelReference);
    DPU_ASSERT(dpu_free(dpuSet));
    
    return 0;
}