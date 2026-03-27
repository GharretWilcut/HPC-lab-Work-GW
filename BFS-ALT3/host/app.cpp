// app.cpp — Parallel Relaxed Distributed BFS (UPMEM) — Optimised

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
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../support/gharret_utils.h"
#include "mram-management.h"

#ifndef ENERGY
#define ENERGY 0
#endif

#if ENERGY
extern "C" { #include <dpu_probe.h> }
#endif

#define DPU_BINARY  "./bin/dpu_code_local"
#define NR_DPUS     16


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
    if (!filename) { printf("usage: %s [-v 0|1|2] edge_list.txt\n",argv[0]); return 1; }

    std::vector<Edge> edges;
    uint32_t maxNode = 0;
    read_edge_list(filename, edges, maxNode);
    uint32_t numGlobalNodes = maxNode + 1;
    printf("Loaded %zu edges, %u nodes\n", edges.size(), numGlobalNodes);

    std::vector<uint8_t> nodeDPU(numGlobalNodes);
    for (uint32_t n = 0; n < numGlobalNodes; n++) nodeDPU[n] = n % NR_DPUS;

    std::vector<std::vector<Edge>> dpuEdges(NR_DPUS);
    for (auto &e : edges)
        dpuEdges[nodeDPU[std::min(e.u,e.v)]].push_back(e);

    std::vector<AlignedU32> globalLevel(numGlobalNodes, {INF, INF});
    uint32_t root = 0;
    globalLevel[root] = {0, root};   // root's parent is itself

    dpu_set_t dpuSet, dpu;
    DPU_ASSERT(dpu_alloc(NR_DPUS, NULL, &dpuSet));
    DPU_ASSERT(dpu_load(dpuSet, DPU_BINARY, NULL));

    std::vector<LocalMeta> meta(NR_DPUS);
    uint32_t maxNodes = 0;

    uint32_t dpuIdx = 0;
    DPU_FOREACH(dpuSet, dpu) {
        auto &m = meta[dpuIdx];
        for (auto &e : dpuEdges[dpuIdx]) {
            if (!m.g2l.count(e.u)) { m.g2l[e.u]=(uint32_t)m.l2g.size(); m.l2g.push_back(e.u); }
            if (!m.g2l.count(e.v)) { m.g2l[e.v]=(uint32_t)m.l2g.size(); m.l2g.push_back(e.v); }
        }
        m.numNodes = (uint32_t)m.l2g.size();
        maxNodes   = std::max(maxNodes, m.numNodes);
        dpuIdx++;
    }

    uint32_t xfer_levels_bytes  = align_to_8(maxNodes * sizeof(AlignedU32));
    uint32_t xfer_changed_bytes = sizeof(uint64_t);    // always 8 bytes

    uint32_t maxEdges = 0;
    for (int i = 0; i < NR_DPUS; i++)
        maxEdges = std::max(maxEdges, (uint32_t)dpuEdges[i].size());

    mram_heap_allocator_t fixed_alloc;
    init_allocator(&fixed_alloc);

    uint32_t fixed_params_off  = mram_heap_alloc(&fixed_alloc, sizeof(DPUParams));
    uint32_t fixed_l2g_off     = mram_heap_alloc(&fixed_alloc, align_to_8(maxNodes * sizeof(uint32_t)));
    uint32_t fixed_edges_off   = mram_heap_alloc(&fixed_alloc, align_to_8(maxEdges * sizeof(Edge)));
    uint32_t fixed_levels_off  = mram_heap_alloc(&fixed_alloc, xfer_levels_bytes);
    uint32_t fixed_changed_off = mram_heap_alloc(&fixed_alloc, xfer_changed_bytes);

    uint32_t params_row = sizeof(DPUParams);
    uint32_t l2g_row    = align_to_8(maxNodes * sizeof(uint32_t));
    uint32_t edges_row  = align_to_8(maxEdges * sizeof(Edge));
 
    std::vector<uint8_t> stageParams(NR_DPUS * params_row,  0);
    std::vector<uint8_t> stageL2g   (NR_DPUS * l2g_row,     0);
    std::vector<uint8_t> stageEdges (NR_DPUS * edges_row,   0);

    static_assert(sizeof(DPUParams) % 8 == 0, "DPUParams must be 8-byte aligned");

    if (verbosity >= 2) {
        printf("\n===== DPU PARTITION SUMMARY =====\n");
        printf("%-6s  %-10s  %-10s\n","DPU","Nodes","Edges");
        printf("------  ----------  ----------\n");
    }
 
    std::vector<std::vector<AlignedU32>> stageLevels(NR_DPUS, std::vector<AlignedU32>(maxNodes, {INF, INF}));
    std::vector<uint64_t>                stageChanged(NR_DPUS, 0);
 
    double t_fill = now_sec();
 
    for (int idx = 0; idx < NR_DPUS; idx++) {
        auto &m = meta[idx];
        uint32_t numEdges = (uint32_t)dpuEdges[idx].size();
 
        std::vector<Edge> localEdges(numEdges);
        for (uint32_t i = 0; i < numEdges; i++)
            localEdges[i] = { m.g2l[dpuEdges[idx][i].u],
                              m.g2l[dpuEdges[idx][i].v] };
 
        m.p.numNodes        = m.numNodes;
        m.p.numEdges        = numEdges;
        m.p.localToGlobal_m = fixed_l2g_off;
        m.p.edges_m         = fixed_edges_off;
        m.p.levels_m        = fixed_levels_off;
        m.p.changed_m       = fixed_changed_off;
        m.xfer_nodes        = maxNodes;
 
        if (verbosity >= 2)
            printf("%-6u  %-10u  %-10u\n", idx, m.numNodes, numEdges);
 
        std::memcpy(stageParams.data() + idx * params_row,
                    &m.p, sizeof(DPUParams));
 
        uint32_t *l2g_dst = reinterpret_cast<uint32_t *>(
            stageL2g.data() + idx * l2g_row);
        std::copy(m.l2g.begin(), m.l2g.end(), l2g_dst);
 
        Edge *edge_dst = reinterpret_cast<Edge *>(
            stageEdges.data() + idx * edges_row);
        std::copy(localEdges.begin(), localEdges.end(), edge_dst);
    }
 
    double t_push = now_sec();
 
    { int idx = 0; DPU_FOREACH(dpuSet, dpu) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, stageParams.data() + idx * params_row));
        idx++;
    }}
    DPU_ASSERT(dpu_push_xfer(dpuSet, DPU_XFER_TO_DPU,
                              DPU_MRAM_HEAP_POINTER_NAME,
                              fixed_params_off, params_row,
                              DPU_XFER_DEFAULT));
 
    { int idx = 0; DPU_FOREACH(dpuSet, dpu) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, stageL2g.data() + idx * l2g_row));
        idx++;
    }}
    DPU_ASSERT(dpu_push_xfer(dpuSet, DPU_XFER_TO_DPU,
                              DPU_MRAM_HEAP_POINTER_NAME,
                              fixed_l2g_off, l2g_row,
                              DPU_XFER_DEFAULT));
 
    { int idx = 0; DPU_FOREACH(dpuSet, dpu) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, stageEdges.data() + idx * edges_row));
        idx++;
    }}
    DPU_ASSERT(dpu_push_xfer(dpuSet, DPU_XFER_TO_DPU,
                              DPU_MRAM_HEAP_POINTER_NAME,
                              fixed_edges_off, edges_row,
                              DPU_XFER_DEFAULT));
 
    double loadTime = now_sec() - t_fill;   
 
    if (verbosity >= 2) {
        printf("=================================\n");
        printf("CPU fill time:              %8.3f ms\n", (t_push   - t_fill)  * 1e3);
        printf("push_xfer load (CPU\u2192DPU):   %8.3f ms\n", (now_sec() - t_push) * 1e3);
        printf("Total initial load:         %8.3f ms\n", loadTime * 1e3);
    }

    dpu_set_t dpuHandles[NR_DPUS];
    { int idx=0; DPU_FOREACH(dpuSet,dpu) dpuHandles[idx++]=dpu; }

    //  BFS loop 
    bool changed; uint32_t iteration = 0;
    double totalUpload=0, totalDPU=0, totalDownload=0, totalMerge=0;

    if (verbosity >= 1) {
        printf("\n%-10s  %-8s  %-12s  %-12s  %-12s  %-12s\n",
               "Iteration","Changed","Upload(ms)","DPU(ms)","Download(ms)","Merge(ms)");
        printf("----------  --------  ------------  ------------  ------------  ------------\n");
    }

    do {
        changed = false;
        iteration++;

        //  Upload levels (now carries parent too) via push_xfer
        double t_up = now_sec();

        for (int idx = 0; idx < NR_DPUS; idx++) {
            auto &m = meta[idx];
            std::fill(stageLevels[idx].begin(), stageLevels[idx].end(), AlignedU32{INF, INF});
            for (uint32_t i = 0; i < m.numNodes; i++) {
                uint32_t g = m.l2g[i];
                stageLevels[idx][i] = globalLevel[g];  // value+parent in one copy
            }
        }

        // Push levels (which now includes parent data — one xfer instead of two)
        { int idx=0; DPU_FOREACH(dpuSet,dpu) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, stageLevels[idx].data()));
            idx++;
        }}
        DPU_ASSERT(dpu_push_xfer(dpuSet, DPU_XFER_TO_DPU,
                                 DPU_MRAM_HEAP_POINTER_NAME,
                                 fixed_levels_off, xfer_levels_bytes,
                                 DPU_XFER_DEFAULT));

        std::fill(stageChanged.begin(), stageChanged.end(), 0ULL);
        { int idx=0; DPU_FOREACH(dpuSet,dpu) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, &stageChanged[idx]));
            idx++;
        }}
        DPU_ASSERT(dpu_push_xfer(dpuSet, DPU_XFER_TO_DPU,
                                 DPU_MRAM_HEAP_POINTER_NAME,
                                 fixed_changed_off, xfer_changed_bytes,
                                 DPU_XFER_DEFAULT));

        totalUpload += now_sec() - t_up;

        // Launch 
        double t_dpu = now_sec();
        DPU_ASSERT(dpu_launch(dpuSet, DPU_SYNCHRONOUS));
        double iterDPU = now_sec() - t_dpu;
        totalDPU += iterDPU;

        // Download 
        double t_dl = now_sec();

        // Stage 1: pull changed flags from all DPUs in parallel
        { int idx=0; DPU_FOREACH(dpuSet,dpu) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, &stageChanged[idx]));
            idx++;
        }}
        DPU_ASSERT(dpu_push_xfer(dpuSet, DPU_XFER_FROM_DPU,
                                 DPU_MRAM_HEAP_POINTER_NAME,
                                 fixed_changed_off, xfer_changed_bytes,
                                 DPU_XFER_DEFAULT));

        // Stage 2: only pull levels from DPUs that reported a change
        {
            for (int idx = 0; idx < NR_DPUS; idx++) {
                if (!stageChanged[idx]) continue;
                DPU_ASSERT(dpu_prepare_xfer(dpuHandles[idx], stageLevels[idx].data()));
            }
            for (int idx = 0; idx < NR_DPUS; idx++) {
                if (!stageChanged[idx]) continue;
                DPU_ASSERT(dpu_push_xfer(dpuHandles[idx], DPU_XFER_FROM_DPU,
                                        DPU_MRAM_HEAP_POINTER_NAME,
                                        fixed_levels_off, xfer_levels_bytes,
                                        DPU_XFER_DEFAULT));
            }
        }
        totalDownload += now_sec() - t_dl;

        double t_merge = now_sec();
        int change_this_iter = 0;

        for (int idx = 0; idx < NR_DPUS; idx++) {
            if (!stageChanged[idx]) continue;
            auto &m = meta[idx];
            for (uint32_t i = 0; i < m.numNodes; i++) {
                uint32_t g = m.l2g[i];
                if (stageLevels[idx][i].value < globalLevel[g].value) {
                    globalLevel[g] = stageLevels[idx][i];  // copies value+parent atomically
                    changed = true;
                    change_this_iter++;
                }
            }
        }

        totalMerge += now_sec() - t_merge;

        if (verbosity >= 1) {
            printf("%-10u  %-8d  %-12.3f  %-12.3f  %-12.3f  %-12.3f\n",
                   iteration, change_this_iter,
                   (now_sec()-t_up)    * 1e3,
                   iterDPU             * 1e3,
                   (now_sec()-t_dl)    * 1e3,
                   (now_sec()-t_merge) * 1e3);
        }

    } while (changed);

    double totalBFS = totalUpload + totalDPU + totalDownload + totalMerge;
    printf("\n===== TIMING SUMMARY =====\n");
    printf("Initial load (CPU\u2192DPU):    %8.3f ms\n", loadTime       * 1e3);
    printf("BFS upload   (CPU\u2192DPU):    %8.3f ms\n", totalUpload    * 1e3);
    printf("DPU kernel total:        %8.3f ms\n",        totalDPU       * 1e3);
    printf("BFS download (DPU\u2192CPU):    %8.3f ms\n", totalDownload  * 1e3);
    printf("Host merge total:        %8.3f ms\n",        totalMerge     * 1e3);
    printf("--------------------------\n");
    printf("Total BFS wall time:     %8.3f ms\n",        totalBFS       * 1e3);
    printf("==========================\n");

    printf("\nConverged in %u iterations\n", iteration);
    verify_levels(edges, globalLevel, root);
    if (verbosity >= 3){
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