// app.cpp — Parallel Relaxed Distributed BFS (UPMEM)


extern "C" {
#include <dpu.h>
#include <dpu_log.h>
}

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

#include "../support/gharret_utils.h"
#include "mram-management.h"

#ifndef ENERGY
#define ENERGY 0
#endif
#if ENERGY
extern "C" { #include <dpu_probe.h> }
#endif

#define DPU_BINARY   "./bin/dpu_code"
#define NR_DPUS      512
#define NR_TASKLETS  16

struct ChangedNode {
    uint32_t globalID;
    uint32_t value;
    uint32_t parent;
    uint32_t _pad;
};

struct NodeOwner {
    uint16_t dpuIdx;
    uint32_t localID;
};

struct DPUMeta {
    uint32_t numNodes;
    uint32_t numEdges;
};

int main(int argc, char **argv)
{
    int verbosity = 0;
    const char *filename = nullptr;
    for (int i = 1; i < argc; i++) {
        if (argv[i][0]=='-' && argv[i][1]=='v') {
            verbosity = (argv[i][2]!='\0') ? atoi(&argv[i][2])
                      : (i+1<argc)         ? atoi(argv[++i]) : 1;
        } else filename = argv[i];
    }
    if (!filename) {
        printf("usage: %s [-v 0|1|2|3] edge_list.txt\n", argv[0]);
        return 1;
    }

   
    std::vector<Edge> edges;
    uint32_t maxNode = 0;
    read_edge_list(filename, edges, maxNode);
    uint32_t numGlobalNodes = maxNode + 1;
    printf("Loaded %zu edges, %u nodes\n", edges.size(), numGlobalNodes);

    std::vector<std::vector<Edge>> dpuEdges(NR_DPUS);
    for (auto &e : edges)
        dpuEdges[std::min(e.u, e.v) % NR_DPUS].push_back(e);

    std::vector<AlignedU32> globalLevel(numGlobalNodes, {INF, INF});
    uint32_t root = 0;
    globalLevel[root] = {0, root};


    dpu_set_t dpuSet, dpu;
    DPU_ASSERT(dpu_alloc(NR_DPUS, NULL, &dpuSet));
    DPU_ASSERT(dpu_load(dpuSet, DPU_BINARY, NULL));

    dpu_set_t dpuHandles[NR_DPUS];
    { int idx=0; DPU_FOREACH(dpuSet,dpu) dpuHandles[idx++]=dpu; }

    uint32_t maxEdges = 0;
    for (int i = 0; i < NR_DPUS; i++)
        maxEdges = std::max(maxEdges, (uint32_t)dpuEdges[i].size());

    uint32_t maxRawIDs   = 2 * maxEdges;

    uint32_t params_row  = sizeof(DPUParams);
    uint32_t edges_row   = align_to_8(maxEdges  * sizeof(Edge));
    uint32_t l2g_row     = align_to_8(maxRawIDs * sizeof(uint32_t));
    
    uint32_t levels_row  = align_to_8(maxRawIDs * sizeof(AlignedU32));
    
    uint32_t changed_row = align_to_8(maxRawIDs * sizeof(ChangedNode));
    uint32_t count_row   = sizeof(uint64_t);

    mram_heap_allocator_t alloc;
    init_allocator(&alloc);

    uint32_t off_params  = mram_heap_alloc(&alloc, params_row);
    uint32_t off_edges   = mram_heap_alloc(&alloc, edges_row);
    uint32_t off_l2g     = mram_heap_alloc(&alloc, l2g_row);
    uint32_t off_levels  = mram_heap_alloc(&alloc, levels_row);
    uint32_t off_changed = mram_heap_alloc(&alloc, changed_row);
    uint32_t off_count   = mram_heap_alloc(&alloc, count_row);

    uint32_t total_mram = off_count + count_row;
    printf("MRAM layout: %.2f MB per DPU\n", total_mram / 1024.0 / 1024.0);
    if (total_mram > 64u * 1024 * 1024) {
        printf("ERROR: MRAM layout exceeds 64 MB!\n");
        return 1;
    }

   
    std::vector<uint8_t>  stageParams (NR_DPUS * params_row,  0);
    std::vector<uint8_t>  stageEdges  (NR_DPUS * edges_row,   0);
    std::vector<uint8_t>  stageLevels (NR_DPUS * levels_row,  0);
    std::vector<uint8_t>  stageChanged(NR_DPUS * changed_row, 0);
    std::vector<uint64_t> stageCount  (NR_DPUS, 0);
    std::vector<uint32_t> stageL2g   (NR_DPUS * maxRawIDs,    0);

    printf("Sending raw edges to DPUs...\n");
    double t_load = now_sec();

    for (int idx = 0; idx < NR_DPUS; idx++) {
        uint32_t numEdges = (uint32_t)dpuEdges[idx].size();

        DPUParams p = {};
        p.numNodes        = 0;
        p.numEdges        = numEdges;
        p.localToGlobal_m = off_l2g;
        p.edges_m         = off_edges;
        p.levels_m        = off_levels;
        p.changedNodes_m  = off_changed;
        p.countOnly_m     = off_count;
        p.isIndexPass     = 1;
        std::memcpy(stageParams.data() + idx * params_row, &p, sizeof(DPUParams));

        Edge *edst = reinterpret_cast<Edge *>(stageEdges.data() + idx * edges_row);
        std::copy(dpuEdges[idx].begin(), dpuEdges[idx].end(), edst);
    }

    { int idx=0; DPU_FOREACH(dpuSet,dpu) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, stageParams.data()+idx*params_row)); idx++;
    }}
    DPU_ASSERT(dpu_push_xfer(dpuSet,DPU_XFER_TO_DPU,DPU_MRAM_HEAP_POINTER_NAME,
                              off_params,params_row,DPU_XFER_DEFAULT));

    { int idx=0; DPU_FOREACH(dpuSet,dpu) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, stageEdges.data()+idx*edges_row)); idx++;
    }}
    DPU_ASSERT(dpu_push_xfer(dpuSet,DPU_XFER_TO_DPU,DPU_MRAM_HEAP_POINTER_NAME,
                              off_edges,edges_row,DPU_XFER_DEFAULT));

    double t_build_launch = now_sec();
    printf("Edge transfer: %.3f ms — launching index build...\n",
           (t_build_launch - t_load) * 1e3);

    DPU_ASSERT(dpu_launch(dpuSet, DPU_SYNCHRONOUS));

    double t_build_done = now_sec();
    printf("DPU index build: %.3f ms\n", (t_build_done - t_build_launch) * 1e3);

    { int idx=0; DPU_FOREACH(dpuSet,dpu) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, stageParams.data()+idx*params_row)); idx++;
    }}
    DPU_ASSERT(dpu_push_xfer(dpuSet,DPU_XFER_FROM_DPU,DPU_MRAM_HEAP_POINTER_NAME,
                              off_params,params_row,DPU_XFER_DEFAULT));

    std::vector<DPUMeta> meta(NR_DPUS);
    uint32_t maxNodes = 0;
    for (int idx = 0; idx < NR_DPUS; idx++) {
        const DPUParams *p = reinterpret_cast<const DPUParams *>(
            stageParams.data() + idx * params_row);
        meta[idx].numNodes = p->numNodes;
        meta[idx].numEdges = (uint32_t)dpuEdges[idx].size();
        maxNodes = std::max(maxNodes, p->numNodes);
    }

    printf("Pulling l2g tables for reverse index...\n");
    uint32_t l2g_pull_row = align_to_8(maxNodes * sizeof(uint32_t));
    { int idx=0; DPU_FOREACH(dpuSet,dpu) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, stageL2g.data() + (size_t)idx * maxRawIDs));
        idx++;
    }}
    DPU_ASSERT(dpu_push_xfer(dpuSet,DPU_XFER_FROM_DPU,DPU_MRAM_HEAP_POINTER_NAME,
                              off_l2g, l2g_pull_row, DPU_XFER_DEFAULT));

    printf("Building reverse index...\n");
    std::vector<std::vector<NodeOwner>> nodeOwners(numGlobalNodes);
    for (int idx = 0; idx < NR_DPUS; idx++) {
        const uint32_t *l2g = stageL2g.data() + (size_t)idx * maxRawIDs;
        for (uint32_t lid = 0; lid < meta[idx].numNodes; lid++)
            nodeOwners[l2g[lid]].push_back({(uint16_t)idx, lid});
    }

    double t_ready = now_sec();
    printf("Total setup (load+build+reverse): %.3f ms\n\n",
           (t_ready - t_load) * 1e3);

    for (int idx = 0; idx < NR_DPUS; idx++) {
        DPUParams *p = reinterpret_cast<DPUParams *>(
            stageParams.data() + idx * params_row);
        p->numNodes    = meta[idx].numNodes;
        p->isIndexPass = 0;
    }
    { int idx=0; DPU_FOREACH(dpuSet,dpu) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, stageParams.data()+idx*params_row)); idx++;
    }}
    DPU_ASSERT(dpu_push_xfer(dpuSet,DPU_XFER_TO_DPU,DPU_MRAM_HEAP_POINTER_NAME,
                              off_params,params_row,DPU_XFER_DEFAULT));

    bool     changed;
    uint32_t iteration = 0;
    double   totalUpload=0, totalDPU=0, totalDownload=0, totalMerge=0;

    std::vector<std::vector<uint32_t>> dirtySlots(NR_DPUS);
    std::vector<int>                   dirtyDPUs;

    if (verbosity >= 1) {
        printf("%-10s  %-10s  %-10s  %-12s  %-12s  %-14s  %-12s\n",
               "Iteration","Changed","ActiveDPUs","Upload(ms)","DPU(ms)",
               "Download(ms)","Merge(ms)");
        printf("----------  ----------  ----------  ------------  ------------"
               "  --------------  ------------\n");
    }

    do {
        changed = false;
        iteration++;


        double t_up = now_sec();

        if (iteration == 1) {
            for (int idx = 0; idx < NR_DPUS; idx++) {
                const uint32_t *l2g = stageL2g.data() + (size_t)idx * maxRawIDs;
                AlignedU32 *lrow = reinterpret_cast<AlignedU32 *>(
                    stageLevels.data() + (size_t)idx * levels_row);
                std::fill(lrow, lrow + meta[idx].numNodes, AlignedU32{INF, INF});
                for (uint32_t lid = 0; lid < meta[idx].numNodes; lid++)
                    lrow[lid] = globalLevel[l2g[lid]];
            }
            { int idx=0; DPU_FOREACH(dpuSet,dpu) {
                DPU_ASSERT(dpu_prepare_xfer(dpu,
                    stageLevels.data()+(size_t)idx*levels_row)); idx++;
            }}
            DPU_ASSERT(dpu_push_xfer(dpuSet,DPU_XFER_TO_DPU,
                                     DPU_MRAM_HEAP_POINTER_NAME,
                                     off_levels,levels_row,DPU_XFER_DEFAULT));
        } else if (!dirtyDPUs.empty()) {
            for (int idx : dirtyDPUs) {
                const uint32_t *l2g = stageL2g.data() + (size_t)idx * maxRawIDs;
                AlignedU32 *lrow = reinterpret_cast<AlignedU32 *>(
                    stageLevels.data() + (size_t)idx * levels_row);
                for (uint32_t lid : dirtySlots[idx])
                    lrow[lid] = globalLevel[l2g[lid]];
                DPU_ASSERT(dpu_prepare_xfer(dpuHandles[idx],
                    stageLevels.data() + (size_t)idx * levels_row));
                DPU_ASSERT(dpu_push_xfer(dpuHandles[idx], DPU_XFER_TO_DPU,
                                         DPU_MRAM_HEAP_POINTER_NAME,
                                         off_levels, levels_row,
                                         DPU_XFER_ASYNC));
            }
            DPU_ASSERT(dpu_sync(dpuSet));
        }

        totalUpload += now_sec() - t_up;

        double t_dpu = now_sec();
        DPU_ASSERT(dpu_launch(dpuSet, DPU_SYNCHRONOUS));
        double iterDPU = now_sec() - t_dpu;
        totalDPU += iterDPU;


        double t_dl = now_sec();

        { int idx=0; DPU_FOREACH(dpuSet,dpu) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, &stageCount[idx])); idx++;
        }}
        DPU_ASSERT(dpu_push_xfer(dpuSet,DPU_XFER_FROM_DPU,
                                 DPU_MRAM_HEAP_POINTER_NAME,
                                 off_count,sizeof(uint64_t),DPU_XFER_DEFAULT));

        std::vector<int> activeDPUs;
        activeDPUs.reserve(NR_DPUS);
        uint32_t maxChanged = 0;
        for (int idx = 0; idx < NR_DPUS; idx++) {
            uint32_t cnt = (uint32_t)stageCount[idx];
            if (cnt > 0) {
                activeDPUs.push_back(idx);
                if (cnt > maxChanged) maxChanged = cnt;
            }
        }

        if (maxChanged > 0) {
            uint32_t xfer_bytes = align_to_8(maxChanged * sizeof(ChangedNode));
            // Clamp to allocated buffer size
            if (xfer_bytes > changed_row) xfer_bytes = changed_row;
            for (int idx : activeDPUs) {
                DPU_ASSERT(dpu_prepare_xfer(dpuHandles[idx],
                    stageChanged.data() + (size_t)idx * changed_row));
                DPU_ASSERT(dpu_push_xfer(dpuHandles[idx],DPU_XFER_FROM_DPU,
                                         DPU_MRAM_HEAP_POINTER_NAME,
                                         off_changed, xfer_bytes,
                                         DPU_XFER_ASYNC));
            }
            DPU_ASSERT(dpu_sync(dpuSet));
        }

        totalDownload += now_sec() - t_dl;

     
        double t_merge = now_sec();
        int change_this_iter = 0;

        dirtyDPUs.clear();
        for (auto &s : dirtySlots) s.clear();

        for (int idx : activeDPUs) {
            uint32_t cnt = (uint32_t)stageCount[idx];
            const ChangedNode *cnodes = reinterpret_cast<const ChangedNode *>(
                stageChanged.data() + (size_t)idx * changed_row);

            for (uint32_t j = 0; j < cnt; j++) {
                const ChangedNode &cn = cnodes[j];
                uint32_t g = cn.globalID;

                if (cn.value < globalLevel[g].value) {
                    globalLevel[g] = {cn.value, cn.parent};
                    changed = true;
                    change_this_iter++;

                    for (const NodeOwner &o : nodeOwners[g])
                        dirtySlots[o.dpuIdx].push_back(o.localID);
                }
            }
        }

        for (int didx = 0; didx < NR_DPUS; didx++) {
            auto &s = dirtySlots[didx];
            if (s.empty()) continue;
            std::sort(s.begin(), s.end());
            s.erase(std::unique(s.begin(), s.end()), s.end());
            dirtyDPUs.push_back(didx);
        }

        totalMerge += now_sec() - t_merge;

        if (verbosity >= 1) {
            printf("%-10u  %-10d  %-10zu  %-12.3f  %-12.3f  %-14.3f  %-12.3f\n",
                   iteration, change_this_iter, activeDPUs.size(),
                   (now_sec()-t_up)    * 1e3,
                   iterDPU             * 1e3,
                   (now_sec()-t_dl)    * 1e3,
                   (now_sec()-t_merge) * 1e3);
        }

    } while (changed);

    double totalBFS = totalUpload + totalDPU + totalDownload + totalMerge;
    printf("\n===== TIMING SUMMARY =====\n");
    printf("Setup (load+build+reverse):  %8.3f ms\n", (t_ready - t_load) * 1e3);
    printf("BFS upload   (CPU->DPU):     %8.3f ms\n", totalUpload   * 1e3);
    printf("DPU kernel total:            %8.3f ms\n", totalDPU      * 1e3);
    printf("BFS download (DPU->CPU):     %8.3f ms\n", totalDownload * 1e3);
    printf("Host merge total:            %8.3f ms\n", totalMerge    * 1e3);
    printf("--------------------------\n");
    printf("Total BFS wall time:         %8.3f ms\n", totalBFS      * 1e3);
    printf("==========================\n");
    printf("\nConverged in %u iterations\n", iteration);

    if (verbosity >= 1) verify_levels(edges, globalLevel, root);

    if (verbosity >= 3) {
        printf("cpu reference bfs...\n");
        double t_cpu = now_sec();
        std::vector<uint32_t> cpuLevel, cpuParent;
        cpu_bfs(edges, numGlobalNodes, root, cpuLevel, cpuParent);
        t_cpu = now_sec() - t_cpu;
        printf("CPU BFS time: %.3f ms\n", t_cpu * 1e3);
        compare_bfs(globalLevel, cpuLevel, numGlobalNodes);
    }

    DPU_ASSERT(dpu_free(dpuSet));
    return 0;
}