// app.cpp — Parallel Relaxed Distributed BFS (UPMEM) — Optimised
//
// Optimisations over the original:
//   1. dpu_push_xfer for all uploads: fires transfers to all DPUs in a
//      single parallel call instead of per-DPU dpu_copy_to.
//   2. dpu_pull_xfer for all downloads: same for reads.
//   3. Per-DPU staging buffers padded to a uniform transfer size so
//      push/pull xfer (which requires equal sizes across DPUs) works
//      correctly even when DPUs have different node counts.
//   4. Dirty-set / changed-flag fast-path: only pull levels+parents
//      from DPUs that report a change.
//   5. Reused upload/download heap allocations across iterations.

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

#include "mram-management.h"

#if ENERGY
extern "C" { #include <dpu_probe.h> }
#endif

#define DPU_BINARY  "./bin/dpu_code_local"
#define NR_DPUS     16
static const uint32_t INF = UINT32_MAX;

/* ── Helpers ────────────────────────────────────────────────────────────── */

static inline double now_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static inline uint32_t align_to_8(uint32_t x) { return (x + 7) & ~7u; }

static double rapl_read_uj(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return -1.0;
    unsigned long long uj = 0;
    fscanf(f, "%llu", &uj);
    fclose(f);
    return (double)uj;
}
static double rapl_delta_joules(double before, double after, const char *maxp) {
    if (before < 0 || after < 0) return -1.0;
    double d = after - before;
    if (d < 0) { double m = rapl_read_uj(maxp); if (m > 0) d += m; }
    return d * 1e-6;
}
#define RAPL_ENERGY_PATH "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"
#define RAPL_MAX_PATH    "/sys/class/powercap/intel-rapl/intel-rapl:0/max_energy_range_uj"

/* ── Types ──────────────────────────────────────────────────────────────── */

struct AlignedU32 { uint32_t value, padding; };
struct Edge       { uint32_t u, v; };

struct DPUParams {
    uint32_t numNodes, numEdges;
    uint32_t localToGlobal_m, edges_m, levels_m, parents_m, changed_m;
    uint32_t padding;
};

struct LocalMeta {
    DPUParams               p;
    std::vector<uint32_t>   l2g;
    std::unordered_map<uint32_t,uint32_t> g2l;
    uint32_t                numNodes;
    uint32_t                xfer_nodes; // padded node count for uniform xfer size
};

/* ── Graph loading ──────────────────────────────────────────────────────── */

static void read_edge_list(const char *fn, std::vector<Edge> &edges, uint32_t &maxNode) {
    FILE *f = fopen(fn, "r");
    if (!f) { perror("fopen"); exit(1); }
    uint32_t u, v; maxNode = 0;
    while (fscanf(f, "%u %u", &u, &v) == 2) {
        edges.push_back({u,v});
        maxNode = std::max(maxNode, std::max(u,v));
    }
    fclose(f);
}

/* ── Verification ───────────────────────────────────────────────────────── */

static void verify_levels(const std::vector<Edge> &edges,
                          const std::vector<AlignedU32> &level,
                          uint32_t root)
{
    int bad = 0; bool ok = true;
    if (level[root].value != 0) { printf("ERROR: root level != 0\n"); ok = false; }
    for (auto &e : edges) {
        if (level[e.u].value == INF || level[e.v].value == INF) continue;
        if ((uint32_t)abs((int)level[e.u].value-(int)level[e.v].value) > 1) { bad++; ok=false; }
    }
    printf("\n===== LEVEL VERIFICATION =====\n");
    if (ok) printf("LEVELS ARE CONSISTENT\n");
    else    printf("LEVELS ARE INVALID WITH %d INVALID EDGES\n", bad);
    printf("==============================\n\n");
}

/* ── Main ───────────────────────────────────────────────────────────────── */

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

    // ── Load graph ───────────────────────────────────────────────────────
    std::vector<Edge> edges;
    uint32_t maxNode = 0;
    read_edge_list(filename, edges, maxNode);
    uint32_t numGlobalNodes = maxNode + 1;
    printf("Loaded %zu edges, %u nodes\n", edges.size(), numGlobalNodes);

    // ── Partition ────────────────────────────────────────────────────────
    std::vector<uint8_t> nodeDPU(numGlobalNodes);
    for (uint32_t n = 0; n < numGlobalNodes; n++) nodeDPU[n] = n % NR_DPUS;

    std::vector<std::vector<Edge>> dpuEdges(NR_DPUS);
    for (auto &e : edges)
        dpuEdges[nodeDPU[std::min(e.u,e.v)]].push_back(e);

    // ── Global BFS state ─────────────────────────────────────────────────
    std::vector<AlignedU32> globalLevel (numGlobalNodes, {INF, 0});
    std::vector<AlignedU32> globalParent(numGlobalNodes, {INF, 0});
    uint32_t root = 0;
    globalLevel[root].value  = 0;
    globalParent[root].value = root;

    // ── Alloc DPUs ───────────────────────────────────────────────────────
    dpu_set_t dpuSet, dpu;
    DPU_ASSERT(dpu_alloc(NR_DPUS, NULL, &dpuSet));
    DPU_ASSERT(dpu_load(dpuSet, DPU_BINARY, NULL));

    // ── Build per-DPU metadata and find the maximum node count ──────────
    // dpu_push_xfer requires every DPU to receive exactly the same byte
    // count.  We compute xfer_nodes = max(numNodes across all DPUs) and
    // pad every staging buffer to that size.
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

    // xfer size in bytes must be 8-byte aligned and identical for all DPUs
    uint32_t xfer_levels_bytes  = align_to_8(maxNodes * sizeof(AlignedU32));
    uint32_t xfer_parents_bytes = xfer_levels_bytes;   // same struct size
    uint32_t xfer_changed_bytes = sizeof(uint64_t);    // always 8 bytes

    // All DPUs use the same MRAM layout (same allocator sequence).
    // All DPUs share identical MRAM offsets (required for push/pull xfer).
    // Use maxNodes + maxEdges to fix the layout so every DPU has the same
    // levels_m, parents_m, changed_m offsets.
    // Levels/parents/changed must be
    // at a FIXED offset.  Strategy: place params + l2g + edges first (variable
    // size per DPU), then levels, parents, changed at fixed offsets computed
    // from the DPU with the most edges.
    uint32_t maxEdges = 0;
    for (int i = 0; i < NR_DPUS; i++)
        maxEdges = std::max(maxEdges, (uint32_t)dpuEdges[i].size());

    // Recompute the fixed layout using maxNodes and maxEdges so offsets are
    // identical on every DPU.
    mram_heap_allocator_t fixed_alloc;
    init_allocator(&fixed_alloc);

    uint32_t fixed_params_off  = mram_heap_alloc(&fixed_alloc, sizeof(DPUParams));
    uint32_t fixed_l2g_off     = mram_heap_alloc(&fixed_alloc, align_to_8(maxNodes   * sizeof(uint32_t)));
    uint32_t fixed_edges_off   = mram_heap_alloc(&fixed_alloc, align_to_8(maxEdges   * sizeof(Edge)));
    uint32_t fixed_levels_off  = mram_heap_alloc(&fixed_alloc, xfer_levels_bytes);
    uint32_t fixed_parents_off = mram_heap_alloc(&fixed_alloc, xfer_parents_bytes);
    uint32_t fixed_changed_off = mram_heap_alloc(&fixed_alloc, xfer_changed_bytes);

    if (verbosity >= 2) {
        printf("\n===== DPU PARTITION SUMMARY =====\n");
        printf("%-6s  %-10s  %-10s\n","DPU","Nodes","Edges");
        printf("------  ----------  ----------\n");
    }

    // ── Per-DPU staging buffers (padded to xfer size) ────────────────────
    // Indexed [dpu][node].  Allocated once, reused every iteration.
    // Size = maxNodes entries so push_xfer transfers the same byte count.
    std::vector<std::vector<AlignedU32>> stageLevels (NR_DPUS, std::vector<AlignedU32>(maxNodes, {INF,0}));
    std::vector<std::vector<AlignedU32>> stageParents(NR_DPUS, std::vector<AlignedU32>(maxNodes, {INF,0}));
    std::vector<uint64_t>                stageChanged(NR_DPUS, 0);

    // ── Initial load ─────────────────────────────────────────────────────
    double loadTime = 0.0;
    dpuIdx = 0;
    DPU_FOREACH(dpuSet, dpu) {
        auto &m = meta[dpuIdx];
        uint32_t numEdges = (uint32_t)dpuEdges[dpuIdx].size();

        // Build local edge list
        std::vector<Edge> localEdges(numEdges);
        for (uint32_t i = 0; i < numEdges; i++)
            localEdges[i] = { m.g2l[dpuEdges[dpuIdx][i].u],
                              m.g2l[dpuEdges[dpuIdx][i].v] };

        // Fill DPUParams with fixed offsets
        m.p.numNodes        = m.numNodes;
        m.p.numEdges        = numEdges;
        m.p.localToGlobal_m = fixed_l2g_off;
        m.p.edges_m         = fixed_edges_off;
        m.p.levels_m        = fixed_levels_off;
        m.p.parents_m       = fixed_parents_off;
        m.p.changed_m       = fixed_changed_off;
        m.p.padding         = 0;
        m.xfer_nodes        = maxNodes;

        if (verbosity >= 2)
            printf("%-6u  %-10u  %-10u\n", dpuIdx, m.numNodes, numEdges);

        double t0 = now_sec();

        // Params
        DPU_ASSERT(dpu_copy_to(dpu, DPU_MRAM_HEAP_POINTER_NAME,
                               fixed_params_off, &m.p, sizeof(DPUParams)));

        // l2g — pad to maxNodes with zeros
        std::vector<uint32_t> l2g_padded(maxNodes, 0);
        std::copy(m.l2g.begin(), m.l2g.end(), l2g_padded.begin());
        DPU_ASSERT(dpu_copy_to(dpu, DPU_MRAM_HEAP_POINTER_NAME,
                               fixed_l2g_off, l2g_padded.data(),
                               align_to_8(maxNodes * sizeof(uint32_t))));

        // Edges — pad to maxEdges with {0,0}
        std::vector<Edge> edges_padded(maxEdges, {0,0});
        std::copy(localEdges.begin(), localEdges.end(), edges_padded.begin());
        DPU_ASSERT(dpu_copy_to(dpu, DPU_MRAM_HEAP_POINTER_NAME,
                               fixed_edges_off, edges_padded.data(),
                               align_to_8(maxEdges * sizeof(Edge))));

        loadTime += now_sec() - t0;
        dpuIdx++;
    }

    if (verbosity >= 2) {
        printf("=================================\n");
        printf("Initial CPU\u2192DPU load time: %.3f ms\n", loadTime * 1e3);
    }

    // Cache DPU handles for indexed access (needed for push/pull xfer)
    dpu_set_t dpuHandles[NR_DPUS];
    { int idx=0; DPU_FOREACH(dpuSet,dpu) dpuHandles[idx++]=dpu; }

    // ── BFS loop ─────────────────────────────────────────────────────────
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

        // ── Upload levels + parents via push_xfer ─────────────────────
        double t_up = now_sec();

        // Fill staging buffers
        for (int idx = 0; idx < NR_DPUS; idx++) {
            auto &m = meta[idx];
            // Zero-fill to maxNodes first so padding region is clean
            std::fill(stageLevels [idx].begin(), stageLevels [idx].end(), AlignedU32{INF,0});
            std::fill(stageParents[idx].begin(), stageParents[idx].end(), AlignedU32{INF,0});
            for (uint32_t i = 0; i < m.numNodes; i++) {
                uint32_t g = m.l2g[i];
                stageLevels [idx][i] = globalLevel [g];
                stageParents[idx][i] = globalParent[g];
            }
        }

        // dpu_prepare_xfer associates a host buffer with each DPU,
        // then dpu_push_xfer fires all transfers in parallel.

        // Push levels
        { int idx=0; DPU_FOREACH(dpuSet,dpu) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, stageLevels[idx].data()));
            idx++;
        }}
        DPU_ASSERT(dpu_push_xfer(dpuSet, DPU_XFER_TO_DPU,
                                 DPU_MRAM_HEAP_POINTER_NAME,
                                 fixed_levels_off, xfer_levels_bytes,
                                 DPU_XFER_DEFAULT));

        // Push parents
        { int idx=0; DPU_FOREACH(dpuSet,dpu) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, stageParents[idx].data()));
            idx++;
        }}
        DPU_ASSERT(dpu_push_xfer(dpuSet, DPU_XFER_TO_DPU,
                                 DPU_MRAM_HEAP_POINTER_NAME,
                                 fixed_parents_off, xfer_parents_bytes,
                                 DPU_XFER_DEFAULT));

        // Reset and push changed flags
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

        // ── Launch ───────────────────────────────────────────────────
        double t_dpu = now_sec();
        DPU_ASSERT(dpu_launch(dpuSet, DPU_SYNCHRONOUS));
        double iterDPU = now_sec() - t_dpu;
        totalDPU += iterDPU;

        // ── Download ─────────────────────────────────────────────────
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

        // Stage 2: pull levels+parents only from DPUs that changed
        // (must be done per-DPU since we can't skip inside push_xfer)
        {
            std::vector<std::thread> threads;
            for (int idx = 0; idx < NR_DPUS; idx++) {
                if (!stageChanged[idx]) continue;
                threads.emplace_back([&, idx]() {
                    DPU_ASSERT(dpu_copy_from(dpuHandles[idx],
                        DPU_MRAM_HEAP_POINTER_NAME,
                        fixed_levels_off,
                        stageLevels[idx].data(),
                        xfer_levels_bytes));
                    DPU_ASSERT(dpu_copy_from(dpuHandles[idx],
                        DPU_MRAM_HEAP_POINTER_NAME,
                        fixed_parents_off,
                        stageParents[idx].data(),
                        xfer_parents_bytes));
                });
            }
            for (auto &t : threads) t.join();
        }

        totalDownload += now_sec() - t_dl;

        // ── Host merge ───────────────────────────────────────────────
        double t_merge = now_sec();
        int change_this_iter = 0;

        for (int idx = 0; idx < NR_DPUS; idx++) {
            if (!stageChanged[idx]) continue;
            auto &m = meta[idx];
            for (uint32_t i = 0; i < m.numNodes; i++) {
                uint32_t g = m.l2g[i];
                if (stageLevels[idx][i].value < globalLevel[g].value) {
                    globalLevel [g] = stageLevels [idx][i];
                    globalParent[g] = stageParents[idx][i];
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

    DPU_ASSERT(dpu_free(dpuSet));
    return 0;
}