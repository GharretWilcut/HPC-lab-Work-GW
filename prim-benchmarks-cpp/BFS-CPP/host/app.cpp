// app.cpp - Main application file for BFS-CPP DPU Host
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


#include "mram-management.h"
#include "../support/common.h"
#include "../support/graph.h"
#include "../support/params.h"
#include "../support/timer.h"
#include "../support/utils.h"

#ifndef ENERGY
#define ENERGY 0
#endif

#if ENERGY
#include <dpu_probe.h>
#endif

#define DPU_BINARY "./bin/dpu_code"

int main(int argc,char**argv){
    Params params = input_params(argc, argv);

    Timer timer;
    float loadTime = 0.0f, dpuTime = 0.0f, hostTime = 0.0f, retrieveTime = 0.0f;

    #if ENERGY
        dpu_probe_t probe;
        DPU_ASSERT(dpu_probe_init("energy_probe",&probe));
        double tenergy = 0.0;
    #endif

    //allocate dpus

    dpu_set_t dpuSet,dpu;
    uint32_t numDPUs;

    dpu_program_t *program = nullptr;

    DPU_ASSERT(dpu_alloc(NR_DPUS,nullptr,&dpuSet));
    DPU_ASSERT(dpu_load(dpuSet,DPU_BINARY,&program));
    DPU_ASSERT(dpu_get_nr_dpus(dpuSet,&numDPUs));

    PRINT_INFO(params.verbosity >=1, "Allocated %u DPUs", numDPUs);
    PRINT_INFO(params.verbosity >= 1, "Reading graph %s", params.fileName);
    
    COOGraph cooGraph = readCOOGraph(params.fileName);

    PRINT_INFO(params.verbosity >=1, "    Graph has %d nodes and %d edges",cooGraph.numNodes,cooGraph.numEdges);

    CSRGraph csrGraph = coo2csr(cooGraph);
    freeCOOGraph(cooGraph);

    uint32_t numNodes = csrGraph.numNodes;
    uint32_t* nodePtrs = csrGraph.nodePtrs;
    uint32_t* neighborIdxs = csrGraph.neighborIdxs;
    
    uint32_t* nodeLevel =
        static_cast<uint32_t*>(calloc(numNodes, sizeof(uint32_t)));
    uint64_t* visited =
        static_cast<uint64_t*>(calloc(numNodes / 64, sizeof(uint64_t)));

    uint64_t* currentFrontier =
        static_cast<uint64_t*>(calloc(numNodes / 64, sizeof(uint64_t)));

    uint64_t* nextFrontier =
        static_cast<uint64_t*>(calloc(numNodes / 64, sizeof(uint64_t)));

    setBit(nextFrontier[0],0);

    uint32_t level = 1;

    uint32_t numNodesPerDPU = 
        ROUND_UP_TO_MULTIPLE_OF_64((numNodes -1)/ numDPUs + 1);

    PRINT_INFO(params.verbosity >= 1, 
        "Assigning %u nodes per DPU",
        numNodesPerDPU);

    DPUParams dpuParams[numDPUs];
    uint32_t dpuParams_m[numDPUs];

    uint32_t globalRoot = 0;
    
    unsigned int dpuIdx = 0;
    DPU_FOREACH(dpuSet,dpu){
        mram_heap_allocator_t allocator;
        init_allocator(&allocator);

        dpuParams_m[dpuIdx] = 
            mram_heap_alloc(&allocator, sizeof(DPUParams));

        uint32_t dpuStartNodeIdx = dpuIdx * numNodesPerDPU;
        uint32_t dpuNumNodes = 0;
        if(dpuStartNodeIdx < numNodes){
            dpuNumNodes = std::min(numNodes - dpuStartNodeIdx, numNodesPerDPU);
        }

        dpuParams[dpuIdx].dpuNumNodes = dpuNumNodes;
        PRINT_INFO(params.verbosity >= 2, "    DPU %u:", dpuIdx);
        PRINT_INFO(params.verbosity >= 2, "        Receives %u nodes", dpuNumNodes);

        if (dpuNumNodes > 0){
            uint32_t* dpuNodePtrs_h = &nodePtrs[dpuStartNodeIdx];
            uint32_t dpuNodePtrsOffset = dpuNodePtrs_h[0];

            uint32_t* dpuNeighborIdxs_h = neighborIdxs + dpuNodePtrsOffset;
            uint32_t dpuNumNeighbors = dpuNodePtrs_h[dpuNumNodes] - dpuNodePtrsOffset;
            uint32_t* dpuNodeLevel_h = &nodeLevel[dpuStartNodeIdx];

            uint32_t dpuNodePtrs_m = mram_heap_alloc(&allocator,(dpuNumNodes + 1) * sizeof(uint32_t));
            uint32_t dpuNeighborIdxs_m = mram_heap_alloc(&allocator,dpuNumNeighbors * sizeof(uint32_t));
            uint32_t dpuNodeLevel_m = mram_heap_alloc(&allocator,dpuNumNodes * sizeof(uint32_t));
            uint32_t dpuVisited_m = mram_heap_alloc(&allocator, numNodes / 64 * sizeof(uint64_t));
            uint32_t dpuCurrentFrontier_m = mram_heap_alloc(&allocator,dpuNumNodes / 64 * sizeof(uint64_t));
            uint32_t dpuNextFrontier_m = mram_heap_alloc(&allocator, numNodes / 64 * sizeof(uint64_t));

            PRINT_INFO(params.verbosity >= 2, "        Total memory allocated is %d bytes", allocator.totalAllocated);

            dpuParams[dpuIdx].numNodes = numNodes;
            dpuParams[dpuIdx].dpuStartNodeIdx = dpuStartNodeIdx;
            dpuParams[dpuIdx].dpuNodePtrsOffset = dpuNodePtrsOffset;
            dpuParams[dpuIdx].level = level;
            dpuParams[dpuIdx].dpuNodePtrs_m = dpuNodePtrs_m;
            dpuParams[dpuIdx].dpuNeighborIdxs_m = dpuNeighborIdxs_m;
            dpuParams[dpuIdx].dpuNodeLevel_m = dpuNodeLevel_m;
            dpuParams[dpuIdx].dpuVisited_m = dpuVisited_m;
            dpuParams[dpuIdx].dpuCurrentFrontier_m = dpuCurrentFrontier_m;
            dpuParams[dpuIdx].dpuNextFrontier_m = dpuNextFrontier_m;
            dpuParams[dpuIdx].dpuNumNodes = dpuNumNodes;

            PRINT_INFO(params.verbosity >= 2, "        Copying data to DPU");
            startTimer(&timer);

            copyToDPU(
                dpu,
                reinterpret_cast<const uint8_t*>(dpuNodePtrs_h),
                dpuNodePtrs_m,
                (dpuNumNodes + 1) * sizeof(uint32_t)
            );
            
            copyToDPU(
                dpu,
                reinterpret_cast<const uint8_t*>(dpuNeighborIdxs_h),
                dpuNeighborIdxs_m,
                dpuNumNeighbors * sizeof(uint32_t)
            );
            
            copyToDPU(
                dpu,
                reinterpret_cast<const uint8_t*>(dpuNodeLevel_h),
                dpuNodeLevel_m,
                dpuNumNodes * sizeof(uint32_t)
            );
            
            copyToDPU(
                dpu,
                reinterpret_cast<const uint8_t*>(visited),
                dpuVisited_m,
                numNodes / 64 * sizeof(uint64_t)
            );
            
            copyToDPU(
                dpu,
                reinterpret_cast<const uint8_t*>(nextFrontier),
                dpuNextFrontier_m,
                numNodes / 64 * sizeof(uint64_t)
            );
            
            stopTimer(&timer);
            loadTime += getElapsedTime(timer);
        }

        PRINT_INFO(params.verbosity >= 2, "        Copying parameters to DPU");
        startTimer(&timer);
        copyToDPU(
            dpu,
            reinterpret_cast<const uint8_t*>(&dpuParams[dpuIdx]),
            dpuParams_m[dpuIdx],
            sizeof(DPUParams)
        );
        stopTimer(&timer);
        loadTime += getElapsedTime(timer);

        ++dpuIdx;

    }
    
    PRINT_INFO(params.verbosity >=1, "    CPU-DPU Time: %f ms", loadTime * 1e3);

    // Iterate until next frontier is empty
    uint32_t nextFrontierEmpty = 0;
    while(!nextFrontierEmpty) {

        PRINT_INFO(params.verbosity >= 1, "Processing current frontier for level %u", level);

        #if ENERGY
        DPU_ASSERT(dpu_probe_start(&probe));
        #endif

        // Run all DPUs
        PRINT_INFO(params.verbosity >= 1, "    Booting DPUs");
        startTimer(&timer);
        DPU_ASSERT(dpu_launch(dpuSet, DPU_SYNCHRONOUS));
        stopTimer(&timer);
        dpuTime += getElapsedTime(timer);
        PRINT_INFO(params.verbosity >= 2, "    Level DPU Time: %f ms", getElapsedTime(timer)*1e3);

        #if ENERGY
        DPU_ASSERT(dpu_probe_stop(&probe));
        double energy;
        DPU_ASSERT(dpu_probe_get(&probe, DPU_ENERGY, DPU_AVERAGE, &energy));
        tenergy += energy;
        #endif

        // Copy back next frontier from all DPUs and compute their union as the current frontier
        startTimer(&timer);
        dpuIdx = 0;
        DPU_FOREACH (dpuSet, dpu) {
            uint32_t dpuNumNodes = dpuParams[dpuIdx].dpuNumNodes;
            if(dpuNumNodes > 0) {
                if(dpuIdx == 0) {
                    copyFromDPU(dpu, dpuParams[dpuIdx].dpuNextFrontier_m, 
                               reinterpret_cast<uint8_t*>(currentFrontier), 
                               numNodes/64*sizeof(uint64_t));
                } else {
                    copyFromDPU(dpu, dpuParams[dpuIdx].dpuNextFrontier_m, 
                               reinterpret_cast<uint8_t*>(nextFrontier), 
                               numNodes/64*sizeof(uint64_t));
                    for(uint32_t i = 0; i < numNodes/64; ++i) {
                        currentFrontier[i] |= nextFrontier[i];
                    }
                }
            }
            ++dpuIdx;
        }

        // Check if the next frontier is empty, and copy data to DPU if not empty
        nextFrontierEmpty = 1;
        for(uint32_t i = 0; i < numNodes/64; ++i) {
            if(currentFrontier[i]) {
                nextFrontierEmpty = 0;
                break;
            }
        }
        
        if(!nextFrontierEmpty) {
            ++level;
            
            // First, collect and merge visited arrays from all DPUs
            dpuIdx = 0;
            DPU_FOREACH (dpuSet, dpu) {
                uint32_t dpuNumNodes = dpuParams[dpuIdx].dpuNumNodes;
                if(dpuNumNodes > 0) {
                    if(dpuIdx == 0) {
                        copyFromDPU(dpu, dpuParams[dpuIdx].dpuVisited_m, 
                                   reinterpret_cast<uint8_t*>(visited), 
                                   numNodes/64*sizeof(uint64_t));
                    } else {
                        uint64_t* tempVisited = static_cast<uint64_t*>(calloc(numNodes/64, sizeof(uint64_t)));
                        copyFromDPU(dpu, dpuParams[dpuIdx].dpuVisited_m, 
                                   reinterpret_cast<uint8_t*>(tempVisited), 
                                   numNodes/64*sizeof(uint64_t));
                        for(uint32_t i = 0; i < numNodes/64; ++i) {
                            visited[i] |= tempVisited[i];
                        }
                        free(tempVisited);
                    }
                }
                ++dpuIdx;
            }
            
            // Now send merged visited and current frontier to all DPUs
            dpuIdx = 0;
            DPU_FOREACH (dpuSet, dpu) {
                uint32_t dpuNumNodes = dpuParams[dpuIdx].dpuNumNodes;
                if(dpuNumNodes > 0) {
                    // Copy merged visited array
                    copyToDPU(dpu, reinterpret_cast<const uint8_t*>(visited), 
                             dpuParams[dpuIdx].dpuVisited_m, 
                             numNodes/64*sizeof(uint64_t));
                    
                    // Copy current frontier to all DPUs
                    copyToDPU(dpu, reinterpret_cast<const uint8_t*>(currentFrontier), 
                             dpuParams[dpuIdx].dpuNextFrontier_m, 
                             numNodes/64*sizeof(uint64_t));
                    
                    // Copy new level to DPU
                    dpuParams[dpuIdx].level = level;
                    copyToDPU(dpu, reinterpret_cast<const uint8_t*>(&dpuParams[dpuIdx]), 
                             dpuParams_m[dpuIdx], 
                             sizeof(DPUParams));
                }
                ++dpuIdx;
            }
        }

        stopTimer(&timer);
        hostTime += getElapsedTime(timer);
        PRINT_INFO(params.verbosity >= 2, "    Level Inter-DPU Time: %f ms", getElapsedTime(timer)*1e3);

    }
    
    PRINT_INFO(params.verbosity >= 1, "DPU Kernel Time: %f ms", dpuTime*1e3);
    PRINT_INFO(params.verbosity >= 1, "Inter-DPU Time: %f ms", hostTime*1e3);
    #if ENERGY
    PRINT_INFO(params.verbosity >= 1, "    DPU Energy: %f J", tenergy);
    #endif

    // Copy back node levels
    PRINT_INFO(params.verbosity >= 1, "Copying back the result");
    startTimer(&timer);
    dpuIdx = 0;
    DPU_FOREACH (dpuSet, dpu) {
        uint32_t dpuNumNodes = dpuParams[dpuIdx].dpuNumNodes;
        if(dpuNumNodes > 0) {
            uint32_t dpuStartNodeIdx = dpuIdx*numNodesPerDPU;
            copyFromDPU(dpu, dpuParams[dpuIdx].dpuNodeLevel_m, 
                       reinterpret_cast<uint8_t*>(nodeLevel + dpuStartNodeIdx), 
                       dpuNumNodes*sizeof(uint32_t));
        }
        ++dpuIdx;
    }
    stopTimer(&timer);
    retrieveTime += getElapsedTime(timer);
    PRINT_INFO(params.verbosity >= 1, "    DPU-CPU Time: %f ms", retrieveTime*1e3);
    
    if(params.verbosity == 0) {
        PRINT("CPU-DPU Time(ms): %f    DPU Kernel Time (ms): %f    Inter-DPU Time (ms): %f    DPU-CPU Time (ms): %f", 
              loadTime*1e3, dpuTime*1e3, hostTime*1e3, retrieveTime*1e3);
    }

    // Calculating result on CPU
    PRINT_INFO(params.verbosity >= 1, "Calculating result on CPU");
    uint32_t* nodeLevelReference = static_cast<uint32_t*>(calloc(numNodes, sizeof(uint32_t)));
    memset(nextFrontier, 0, numNodes/64*sizeof(uint64_t));
    memset(visited, 0, numNodes/64*sizeof(uint64_t));
    setBit(nextFrontier[0], 0);
    nextFrontierEmpty = 0;
    level = 1;
    
    while(!nextFrontierEmpty) {
        // Update current frontier and visited list based on the next frontier from the previous iteration
        for(uint32_t nodeTileIdx = 0; nodeTileIdx < numNodes/64; ++nodeTileIdx) {
            uint64_t nextFrontierTile = nextFrontier[nodeTileIdx];
            currentFrontier[nodeTileIdx] = nextFrontierTile;
            if(nextFrontierTile) {
                visited[nodeTileIdx] |= nextFrontierTile;
                nextFrontier[nodeTileIdx] = 0;
                for(uint32_t node = nodeTileIdx*64; node < (nodeTileIdx + 1)*64; ++node) {
                    if(isSet(nextFrontierTile, node%64)) {
                        nodeLevelReference[node] = level;
                    }
                }
            }
        }
        // Visit neighbors of the current frontier
        nextFrontierEmpty = 1;
        for(uint32_t nodeTileIdx = 0; nodeTileIdx < numNodes/64; ++nodeTileIdx) {
            uint64_t currentFrontierTile = currentFrontier[nodeTileIdx];
            if(currentFrontierTile) {
                for(uint32_t node = nodeTileIdx*64; node < (nodeTileIdx + 1)*64; ++node) {
                    if(isSet(currentFrontierTile, node%64)) {
                        uint32_t nodePtr = nodePtrs[node];
                        uint32_t nextNodePtr = nodePtrs[node + 1];
                        for(uint32_t i = nodePtr; i < nextNodePtr; ++i) {
                            uint32_t neighbor = neighborIdxs[i];
                            if(!isSet(visited[neighbor/64], neighbor%64)) {
                                setBit(nextFrontier[neighbor/64], neighbor%64);
                                nextFrontierEmpty = 0;
                            }
                        }
                    }
                }
            }
        }
        ++level;
    }

    // Verify the result
    PRINT_INFO(params.verbosity >= 1, "Verifying the result");
    for(uint32_t nodeIdx = 0; nodeIdx < numNodes; ++nodeIdx) {
        if(nodeLevel[nodeIdx] != nodeLevelReference[nodeIdx]) {
            PRINT_ERROR("Mismatch at node %u (CPU result = level %u, DPU result = level %u)", 
                       nodeIdx, nodeLevelReference[nodeIdx], nodeLevel[nodeIdx]);
        }
    }

    // Display DPU Logs (if any exist)
    if(params.verbosity >= 2) {
        PRINT_INFO(params.verbosity >= 2, "Displaying DPU Logs:");
        dpuIdx = 0;
        DPU_FOREACH (dpuSet, dpu) {
            PRINT("DPU %u:", dpuIdx);
            dpu_error_t log_result = dpu_log_read(dpu, stdout);
            if(log_result != DPU_OK && log_result != DPU_ERR_LOG_CONTEXT_MISSING) {
                DPU_ASSERT(log_result);
            }
            ++dpuIdx;
        }
    }

    // Deallocate data structures
    freeCSRGraph(csrGraph);
    free(nodeLevel);
    free(visited);
    free(currentFrontier);
    free(nextFrontier);
    free(nodeLevelReference);

    DPU_ASSERT(dpu_free(dpuSet));

    return 0;

}