// app.cpp — Parallel Relaxed Distributed BFS (UPMEM)
//
// Two-stage sparse download protocol:
//
//   Stage 1 — count pull (push_xfer, uniform 8 bytes, all DPUs):
//     Each DPU writes its totalChanged uint64_t to countOnly_m.
//     CPU reads all 512 in one broadcast pull.
//
//   Stage 2 — data pull (async push_xfer per active DPU + one dpu_sync):
//     CPU finds maxChanged = max(totalChanged across active DPUs).
//     Issues async push_xfer for each active DPU (fires in parallel),
//     then calls dpu_sync(dpuSet) once to wait for all completions.
//     Each DPU's valid records are tightly packed at the front of its buffer;
//     the CPU uses each DPU's own totalChanged to know how many are valid.
//
//   Sparse upload (iteration > 1):
//     Only dirty DPUs (those that need updated level data) participate.
//     Issues async push_xfer per dirty DPU + one dpu_sync barrier.
//     Clean DPUs are skipped entirely — no wasted bandwidth.

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
#include <unordered_map>
#include <vector>

#include "../support/gharret_utils.h"
#include "mram-management.h"

#ifndef ENERGY
#define ENERGY 0
#endif
#if ENERGY
extern "C" { #include <dpu_probe.h> }
#endif

#define DPU_BINARY               "./bin/dpu_code"
#define NR_DPUS                  512
#define NR_TASKLETS              16
#define MAX_CHANGED_PER_TASKLET  64
#define MAX_CHANGED_PER_DPU      ((uint32_t)NR_TASKLETS * MAX_CHANGED_PER_TASKLET)  // 1024

// ChangedNode mirrors task.c (16 bytes)
struct ChangedNode {
    uint32_t localID;
    uint32_t value;
    uint32_t parent;
    uint32_t _pad;
};

// Reverse-index entry: one DPU that holds a given global node
struct NodeOwner {
    uint16_t dpuIdx;
    uint32_t localID;
};

// -----------------------------------------------------------------------
int main(int argc, char **argv)
// -----------------------------------------------------------------------
{
    int verbosity = 0;
    const char *filename = nullptr;
    for (int i = 1; i < argc; i++) {
        if (argv[i][0]=='-' && argv[i][1]=='v') {
            verbosity = (argv[i][2]!='\0') ? atoi(&argv[i][2])
                      : (i+1<argc)         ? atoi(argv[++i]) : 1;
        } else filename = argv[i];
    }
    if (!filename) { printf("usage: %s [-v 0|1|2|3] edge_list.txt\n",argv[0]); return 1; }

    // -----------------------------------------------------------------------
    // Load graph
    // -----------------------------------------------------------------------
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

    // -----------------------------------------------------------------------
    // Allocate DPUs
    // -----------------------------------------------------------------------
    dpu_set_t dpuSet, dpu;
    DPU_ASSERT(dpu_alloc(NR_DPUS, NULL, &dpuSet));
    DPU_ASSERT(dpu_load(dpuSet, DPU_BINARY, NULL));

    // -----------------------------------------------------------------------
    // Build per-DPU metadata
    // -----------------------------------------------------------------------
    std::vector<LocalMeta> meta(NR_DPUS);
    uint32_t maxNodes = 0;
    {
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
    }

    // Build reverse index: nodeOwners[g] = list of (dpuIdx, localID)
    printf("Building reverse index...\n");
    std::vector<std::vector<NodeOwner>> nodeOwners(numGlobalNodes);
    for (int idx = 0; idx < NR_DPUS; idx++) {
        auto &m = meta[idx];
        for (uint32_t lid = 0; lid < m.numNodes; lid++)
            nodeOwners[m.l2g[lid]].push_back({(uint16_t)idx, lid});
    }

    // -----------------------------------------------------------------------
    // MRAM layout
    // -----------------------------------------------------------------------
    uint32_t maxEdges = 0;
    for (int i = 0; i < NR_DPUS; i++)
        maxEdges = std::max(maxEdges, (uint32_t)dpuEdges[i].size());

    uint32_t params_row        = sizeof(DPUParams);
    uint32_t l2g_row           = align_to_8(maxNodes * sizeof(uint32_t));
    uint32_t edges_row         = align_to_8(maxEdges * sizeof(Edge));
    uint32_t levels_row        = align_to_8(maxNodes * sizeof(AlignedU32));
    uint32_t changed_data_row  = align_to_8(MAX_CHANGED_PER_DPU * sizeof(ChangedNode));
    uint32_t count_row         = sizeof(uint64_t);

    mram_heap_allocator_t alloc;
    init_allocator(&alloc);

    uint32_t off_params        = mram_heap_alloc(&alloc, params_row);
    uint32_t off_l2g           = mram_heap_alloc(&alloc, l2g_row);
    uint32_t off_edges         = mram_heap_alloc(&alloc, edges_row);
    uint32_t off_levels        = mram_heap_alloc(&alloc, levels_row);
    uint32_t off_changed_data  = mram_heap_alloc(&alloc, changed_data_row);
    uint32_t off_count         = mram_heap_alloc(&alloc, count_row);

    // -----------------------------------------------------------------------
    // Stage buffers
    // -----------------------------------------------------------------------
    std::vector<uint8_t> stageParams      (NR_DPUS * params_row,       0);
    std::vector<uint8_t> stageL2g        (NR_DPUS * l2g_row,          0);
    std::vector<uint8_t> stageEdges      (NR_DPUS * edges_row,        0);
    std::vector<uint8_t> stageLevels     (NR_DPUS * levels_row,       0);
    std::vector<uint8_t> stageChangedData(NR_DPUS * changed_data_row, 0);
    std::vector<uint64_t> stageCount(NR_DPUS, 0);

    // -----------------------------------------------------------------------
    // One-time setup
    // -----------------------------------------------------------------------
    if (verbosity >= 2) {
        printf("\n===== DPU PARTITION SUMMARY =====\n");
        printf("%-6s  %-10s  %-10s\n","DPU","Nodes","Edges");
        printf("------  ----------  ----------\n");
    }

    double t_fill = now_sec();

    for (int idx = 0; idx < NR_DPUS; idx++) {
        auto &m = meta[idx];
        uint32_t numEdges = (uint32_t)dpuEdges[idx].size();

        std::vector<Edge> localEdges(numEdges);
        for (uint32_t i = 0; i < numEdges; i++)
            localEdges[i] = { m.g2l[dpuEdges[idx][i].u], m.g2l[dpuEdges[idx][i].v] };

        m.p.numNodes        = m.numNodes;
        m.p.numEdges        = numEdges;
        m.p.localToGlobal_m = off_l2g;
        m.p.edges_m         = off_edges;
        m.p.levels_m        = off_levels;
        m.p.changedNodes_m  = off_changed_data;
        m.p.countOnly_m     = off_count;
        m.p._pad            = 0;
        m.xfer_nodes        = maxNodes;

        if (verbosity >= 2) printf("%-6u  %-10u  %-10u\n", idx, m.numNodes, numEdges);

        std::memcpy(stageParams.data() + idx * params_row, &m.p, sizeof(DPUParams));

        uint32_t *l2g_dst = reinterpret_cast<uint32_t *>(stageL2g.data() + idx * l2g_row);
        std::copy(m.l2g.begin(), m.l2g.end(), l2g_dst);

        Edge *edge_dst = reinterpret_cast<Edge *>(stageEdges.data() + idx * edges_row);
        std::copy(localEdges.begin(), localEdges.end(), edge_dst);
    }

    double t_push = now_sec();

    { int idx=0; DPU_FOREACH(dpuSet,dpu) { DPU_ASSERT(dpu_prepare_xfer(dpu, stageParams.data()+idx*params_row)); idx++; }}
    DPU_ASSERT(dpu_push_xfer(dpuSet,DPU_XFER_TO_DPU,DPU_MRAM_HEAP_POINTER_NAME,off_params,params_row,DPU_XFER_DEFAULT));

    { int idx=0; DPU_FOREACH(dpuSet,dpu) { DPU_ASSERT(dpu_prepare_xfer(dpu, stageL2g.data()+idx*l2g_row)); idx++; }}
    DPU_ASSERT(dpu_push_xfer(dpuSet,DPU_XFER_TO_DPU,DPU_MRAM_HEAP_POINTER_NAME,off_l2g,l2g_row,DPU_XFER_DEFAULT));

    { int idx=0; DPU_FOREACH(dpuSet,dpu) { DPU_ASSERT(dpu_prepare_xfer(dpu, stageEdges.data()+idx*edges_row)); idx++; }}
    DPU_ASSERT(dpu_push_xfer(dpuSet,DPU_XFER_TO_DPU,DPU_MRAM_HEAP_POINTER_NAME,off_edges,edges_row,DPU_XFER_DEFAULT));

    double loadTime = now_sec() - t_fill;

    if (verbosity >= 2) {
        printf("=================================\n");
        printf("CPU fill time:              %8.3f ms\n", (t_push   - t_fill) * 1e3);
        printf("push_xfer load (CPU->DPU):  %8.3f ms\n", (now_sec()-t_push)  * 1e3);
        printf("Total initial load:         %8.3f ms\n", loadTime             * 1e3);
    }

    // Stash individual DPU handles for targeted async xfers
    dpu_set_t dpuHandles[NR_DPUS];
    { int idx=0; DPU_FOREACH(dpuSet,dpu) dpuHandles[idx++]=dpu; }

    // -----------------------------------------------------------------------
    // BFS loop
    // -----------------------------------------------------------------------
    bool     changed;
    uint32_t iteration = 0;
    double   totalUpload=0, totalDPU=0, totalDownload=0, totalMerge=0;

    std::vector<std::vector<uint32_t>> dirtySlots(NR_DPUS);
    std::vector<int>                   dirtyDPUs;

    if (verbosity >= 1) {
        printf("\n%-10s  %-10s  %-10s  %-12s  %-12s  %-14s  %-12s\n",
               "Iteration","Changed","ActiveDPUs","Upload(ms)","DPU(ms)","Download(ms)","Merge(ms)");
        printf("----------  ----------  ----------  ------------  ------------  --------------  ------------\n");
    }

    do {
        changed = false;
        iteration++;

        // ----------------------------------------------------------------
        // UPLOAD levels
        //   Iteration 1: full broadcast to all DPUs (single push_xfer).
        //   Iteration N: async push_xfer per dirty DPU + one dpu_sync.
        //                Clean DPUs are skipped entirely.
        // ----------------------------------------------------------------
        double t_up = now_sec();

        if (iteration == 1) {
            for (int idx = 0; idx < NR_DPUS; idx++) {
                auto &m = meta[idx];
                AlignedU32 *lrow = reinterpret_cast<AlignedU32 *>(
                    stageLevels.data() + idx * levels_row);
                std::fill(lrow, lrow + maxNodes, AlignedU32{INF, INF});
                for (uint32_t i = 0; i < m.numNodes; i++)
                    lrow[i] = globalLevel[m.l2g[i]];
            }
            { int idx=0; DPU_FOREACH(dpuSet,dpu) {
                DPU_ASSERT(dpu_prepare_xfer(dpu, stageLevels.data()+idx*levels_row)); idx++;
            }}
            DPU_ASSERT(dpu_push_xfer(dpuSet,DPU_XFER_TO_DPU,DPU_MRAM_HEAP_POINTER_NAME,
                                     off_levels,levels_row,DPU_XFER_DEFAULT));
        } else if (!dirtyDPUs.empty()) {
            // Update dirty slots in the stage buffer
            for (int idx : dirtyDPUs) {
                AlignedU32 *lrow = reinterpret_cast<AlignedU32 *>(
                    stageLevels.data() + idx * levels_row);
                for (uint32_t lid : dirtySlots[idx])
                    lrow[lid] = globalLevel[meta[idx].l2g[lid]];
            }

            // Fire async xfer for each dirty DPU — all run in parallel,
            // skipping clean DPUs entirely (zero wasted bandwidth).
            for (int idx : dirtyDPUs) {
                DPU_ASSERT(dpu_prepare_xfer(dpuHandles[idx],
                    stageLevels.data() + idx * levels_row));
                DPU_ASSERT(dpu_push_xfer(dpuHandles[idx], DPU_XFER_TO_DPU,
                                         DPU_MRAM_HEAP_POINTER_NAME,
                                         off_levels, levels_row,
                                         DPU_XFER_ASYNC));
            }
            // Single barrier waits for all async uploads to complete
            DPU_ASSERT(dpu_sync(dpuSet));
        }

        totalUpload += now_sec() - t_up;

        // ----------------------------------------------------------------
        // LAUNCH
        // ----------------------------------------------------------------
        double t_dpu = now_sec();
        DPU_ASSERT(dpu_launch(dpuSet, DPU_SYNCHRONOUS));
        double iterDPU = now_sec() - t_dpu;
        totalDPU += iterDPU;

        // ----------------------------------------------------------------
        // DOWNLOAD — two-stage
        //
        // Stage 1: pull countOnly (uint64_t = 8 bytes) from ALL DPUs.
        //   One push_xfer, uniform 8 bytes, all 512 DPUs in parallel.
        // ----------------------------------------------------------------
        double t_dl = now_sec();

        { int idx=0; DPU_FOREACH(dpuSet,dpu) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, &stageCount[idx]));
            idx++;
        }}
        DPU_ASSERT(dpu_push_xfer(dpuSet, DPU_XFER_FROM_DPU,
                                 DPU_MRAM_HEAP_POINTER_NAME,
                                 off_count, sizeof(uint64_t),
                                 DPU_XFER_DEFAULT));

        // Find active DPUs and the uniform xfer size for stage 2.
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

        // Stage 2: async push_xfer per active DPU + one dpu_sync.
        //   All active transfers fire in parallel; inactive DPUs are
        //   skipped entirely (no wasted bandwidth, no stale writes).
        //   Each DPU's valid records are at [0..stageCount[idx]-1].
        if (maxChanged > 0) {
            uint32_t xfer_bytes = align_to_8(maxChanged * sizeof(ChangedNode));
            for (int idx : activeDPUs) {
                DPU_ASSERT(dpu_prepare_xfer(dpuHandles[idx],
                    stageChangedData.data() + idx * changed_data_row));
                DPU_ASSERT(dpu_push_xfer(dpuHandles[idx], DPU_XFER_FROM_DPU,
                                         DPU_MRAM_HEAP_POINTER_NAME,
                                         off_changed_data, xfer_bytes,
                                         DPU_XFER_ASYNC));
            }
            // Single barrier waits for all async downloads to complete
            DPU_ASSERT(dpu_sync(dpuSet));
        }

        totalDownload += now_sec() - t_dl;

        // ----------------------------------------------------------------
        // MERGE — O(totalChanged) with O(1) reverse-index lookup per node
        // ----------------------------------------------------------------
        double t_merge = now_sec();
        int change_this_iter = 0;

        dirtyDPUs.clear();
        for (auto &s : dirtySlots) s.clear();

        for (int idx : activeDPUs) {
            auto &m = meta[idx];
            uint32_t cnt = (uint32_t)stageCount[idx];
            const ChangedNode *cnodes = reinterpret_cast<const ChangedNode *>(
                stageChangedData.data() + idx * changed_data_row);

            for (uint32_t j = 0; j < cnt; j++) {
                const ChangedNode &cn = cnodes[j];
                uint32_t g = m.l2g[cn.localID];

                if (cn.value < globalLevel[g].value) {
                    globalLevel[g] = {cn.value, cn.parent};
                    changed = true;
                    change_this_iter++;

                    for (const NodeOwner &o : nodeOwners[g])
                        dirtySlots[o.dpuIdx].push_back(o.localID);
                }
            }
        }

        // Deduplicate dirty slots and build dirtyDPUs
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

    // -----------------------------------------------------------------------
    // Results
    // -----------------------------------------------------------------------
    double totalBFS = totalUpload + totalDPU + totalDownload + totalMerge;
    printf("\n===== TIMING SUMMARY =====\n");
    printf("Initial load (CPU->DPU):   %8.3f ms\n", loadTime      * 1e3);
    printf("BFS upload   (CPU->DPU):   %8.3f ms\n", totalUpload   * 1e3);
    printf("DPU kernel total:          %8.3f ms\n", totalDPU      * 1e3);
    printf("BFS download (DPU->CPU):   %8.3f ms\n", totalDownload * 1e3);
    printf("Host merge total:          %8.3f ms\n", totalMerge    * 1e3);
    printf("--------------------------\n");
    printf("Total BFS wall time:       %8.3f ms\n", totalBFS      * 1e3);
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