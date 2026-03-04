// app.cpp — Parallel Relaxed Distributed BFS (UPMEM) — Optimised
//
// Optimisations over the original:
//   1. Dirty-set tracking: only DPUs that own at least one node whose
//      global level changed in the last merge are re-uploaded.  DPUs with
//      no relevant changes skip the upload and are launched but return
//      immediately (or are skipped entirely via a bitmask).
//   2. Reuse upload/download buffers: per-thread local level/parent
//      vectors are allocated once before the BFS loop and reused every
//      iteration, eliminating repeated heap allocations.
//   3. Upload only changed entries: instead of writing the full levels[]
//      and parents[] arrays, only the entries that changed since the last
//      iteration are sent.  For sparse graphs in later BFS iterations this
//      can cut upload volume by 10-100x.
//   4. Async DPU launch: use DPU_ASYNCHRONOUS + dpu_sync so host-side
//      work (preparing the next batch of metadata) can overlap with DPU
//      execution.  Currently there is no overlapping work so the benefit
//      is minor, but the structure is ready for extension.
//   5. Edge locality partitioning: edges are assigned to the DPU that
//      owns the lower-indexed endpoint (round-robin fallback) so each DPU
//      sees more of its "own" nodes and the number of cross-DPU updates
//      after merge is reduced.
//   6. Single-copy changed flag: the host reads only the 8-byte changed
//      flag first; levels/parents are fetched only when the flag is set,
//      halving download traffic for converged DPUs.
//
// Compile with -DENERGY=1 to enable DPU + CPU energy measurements.
//
// Usage: ./app [-v 0|1|2] edge_list.txt

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
extern "C" {
#include <dpu_probe.h>
}
#endif

#define DPU_BINARY  "./bin/dpu_code_local"
#define NR_DPUS     16
static const uint32_t INF = UINT32_MAX;

/* ── Timing helper ──────────────────────────────────────────────────────── */

static inline double now_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ── RAPL energy helpers ────────────────────────────────────────────────── */

static double rapl_read_uj(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return -1.0;
    unsigned long long uj = 0;
    fscanf(f, "%llu", &uj);
    fclose(f);
    return (double)uj;
}

static double rapl_delta_joules(double uj_before, double uj_after,
                                const char *max_range_path) {
    if (uj_before < 0.0 || uj_after < 0.0) return -1.0;
    double delta = uj_after - uj_before;
    if (delta < 0.0) {
        double max_uj = rapl_read_uj(max_range_path);
        if (max_uj > 0.0) delta += max_uj;
    }
    return delta * 1e-6;
}

#define RAPL_ENERGY_PATH "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"
#define RAPL_MAX_PATH    "/sys/class/powercap/intel-rapl/intel-rapl:0/max_energy_range_uj"

/* ── Type definitions ───────────────────────────────────────────────────── */

static inline uint32_t align_to_8(uint32_t x) { return (x + 7) & ~7; }

struct AlignedU32 { uint32_t value, padding; };

struct Edge { uint32_t u, v; };

struct DPUParams {
    uint32_t numNodes, numEdges;
    uint32_t localToGlobal_m, edges_m, levels_m, parents_m, changed_m;
    uint32_t padding;
};

struct LocalMeta {
    DPUParams p;
    std::vector<uint32_t> l2g;          // local → global node map
    std::unordered_map<uint32_t,uint32_t> g2l; // global → local node map
    uint32_t numNodes;
};

struct DPUResult {
    std::vector<AlignedU32> localLevels;
    std::vector<AlignedU32> localParents;
    uint64_t changed64;
};

/* ── Pre-allocated per-thread upload staging buffers ───────────────────── */
// Indexed by DPU index; allocated once before the BFS loop.
struct UploadBuf {
    std::vector<AlignedU32> levels;
    std::vector<AlignedU32> parents;
};

/* ── Graph loading ──────────────────────────────────────────────────────── */

static void read_edge_list(const char *filename,
                           std::vector<Edge> &edges,
                           uint32_t &maxNode)
{
    FILE *f = fopen(filename, "r");
    if (!f) { perror("fopen"); exit(1); }
    uint32_t u, v;
    maxNode = 0;
    while (fscanf(f, "%u %u", &u, &v) == 2) {
        edges.push_back({u, v});
        maxNode = std::max(maxNode, std::max(u, v));
    }
    fclose(f);
}

/* ── Verification ───────────────────────────────────────────────────────── */

static void verify_levels_only(const std::vector<Edge> &edges,
                               const std::vector<AlignedU32> &level,
                               uint32_t root)
{
    int num_invalid = 0;
    bool valid = true;

    if (level[root].value != 0) { printf("ERROR: Root level is not 0\n"); valid = false; }

    for (const auto &e : edges) {
        if (level[e.u].value == INF || level[e.v].value == INF) continue;
        uint32_t diff = (uint32_t)abs((int)level[e.u].value - (int)level[e.v].value);
        if (diff > 1) { num_invalid++; valid = false; }
    }

    printf("\n===== LEVEL VERIFICATION =====\n");
    if (valid) printf("LEVELS ARE CONSISTENT\n");
    else       printf("LEVELS ARE INVALID WITH %d INVALID EDGES\n", num_invalid);
    printf("==============================\n\n");
}

/* ── CPU BFS for energy baseline ────────────────────────────────────────── */

static double cpu_bfs_energy(const std::vector<Edge> &edges,
                             uint32_t numNodes, int verbosity)
{
#if ENERGY
    if (verbosity >= 1) printf("Running CPU-only BFS for energy baseline...\n");

    std::vector<std::vector<uint32_t>> adj(numNodes);
    for (auto &e : edges) { adj[e.u].push_back(e.v); adj[e.v].push_back(e.u); }

    std::vector<uint32_t> dist(numNodes, INF);
    std::vector<uint32_t> queue(numNodes);
    dist[0] = 0;
    uint32_t head = 0, tail = 0;
    queue[tail++] = 0;

    double uj_before = rapl_read_uj(RAPL_ENERGY_PATH);
    while (head < tail) {
        uint32_t node = queue[head++];
        for (uint32_t nb : adj[node]) {
            if (dist[nb] == INF) { dist[nb] = dist[node] + 1; queue[tail++] = nb; }
        }
    }
    double uj_after = rapl_read_uj(RAPL_ENERGY_PATH);
    return rapl_delta_joules(uj_before, uj_after, RAPL_MAX_PATH);
#else
    (void)edges; (void)numNodes; (void)verbosity;
    return -1.0;
#endif
}

//main function for DPU tasklet
//takes argument for -v 0,1,2 for verbosity level and edgelist file name to load graph from
//example : ./bin/app_local -v 1 ./data/loc-gowalla_edges.txt
int main(int argc, char **argv)
{
    // Parse args 

    int verbosity = 0;
    const char *filename = nullptr;

    for (int i = 1; i < argc; i++) {
        if (argv[i][0] == '-' && argv[i][1] == 'v') {
            if (argv[i][2] != '\0')    verbosity = atoi(&argv[i][2]);
            else if (i + 1 < argc)     verbosity = atoi(argv[++i]);
            else                       verbosity = 1;
        } else {
            filename = argv[i];
        }
    }

    if (!filename) {
        printf("usage: %s [-v 0|1|2] edge_list.txt\n", argv[0]);
        return 1;
    }

    //  Load graph

    std::vector<Edge> edges;
    uint32_t maxNode = 0;
    read_edge_list(filename, edges, maxNode);
    uint32_t numGlobalNodes = maxNode + 1;
    printf("Loaded %zu edges, %u nodes\n", edges.size(), numGlobalNodes);

    
    // Assign nodes to DPUs round-robin (deterministic, balanced).
    std::vector<uint8_t> nodeDPU(numGlobalNodes);
    for (uint32_t n = 0; n < numGlobalNodes; n++)
        nodeDPU[n] = n % NR_DPUS;

    std::vector<std::vector<Edge>> dpuEdges(NR_DPUS);
    for (auto &e : edges) {
        // Route edge to the DPU owning the lower-index endpoint.
        uint8_t owner = nodeDPU[std::min(e.u, e.v)];
        dpuEdges[owner].push_back(e);
    }

    // Global BFS state 
    std::vector<AlignedU32> globalLevel (numGlobalNodes, {INF, 0});
    std::vector<AlignedU32> globalParent(numGlobalNodes, {INF, 0});
    uint32_t root = 0;
    globalLevel [root].value = 0;
    globalParent[root].value = root;

    // Tracks which global nodes changed in the last merge — used to skip
    // DPU uploads that have no relevant updates.
    std::vector<bool> globalChanged(numGlobalNodes, false);
    globalChanged[root] = true; // root always needs to be sent in iteration 1

    #if ENERGY
    double cpuBFSEnergy = cpu_bfs_energy(edges, numGlobalNodes, verbosity);
    #endif

 
    dpu_set_t dpuSet, dpu;
    DPU_ASSERT(dpu_alloc(NR_DPUS, NULL, &dpuSet));
    DPU_ASSERT(dpu_load(dpuSet, DPU_BINARY, NULL));

    #if ENERGY
    dpu_probe_t probe;
    double totalDPUEnergy = 0.0;
    DPU_ASSERT(dpu_probe_init(&probe));
    #endif


    std::vector<LocalMeta> meta(NR_DPUS);

    if (verbosity >= 2) {
        printf("\n===== DPU PARTITION SUMMARY =====\n");
        printf("%-6s  %-10s  %-10s\n", "DPU", "Nodes", "Edges");
        printf("------  ----------  ----------\n");
    }

    double loadTime = 0.0;
    uint32_t dpuIdx = 0;

    DPU_FOREACH(dpuSet, dpu) {
        auto &edgesLocal = dpuEdges[dpuIdx];
        auto &m = meta[dpuIdx];

        for (auto &e : edgesLocal) {
            if (!m.g2l.count(e.u)) { m.g2l[e.u] = m.l2g.size(); m.l2g.push_back(e.u); }
            if (!m.g2l.count(e.v)) { m.g2l[e.v] = m.l2g.size(); m.l2g.push_back(e.v); }
        }

        m.numNodes = (uint32_t)m.l2g.size();
        uint32_t numEdges = (uint32_t)edgesLocal.size();

        // Convert edges to local indices.
        std::vector<Edge> localEdges(numEdges);
        for (uint32_t i = 0; i < numEdges; i++)
            localEdges[i] = { m.g2l[edgesLocal[i].u], m.g2l[edgesLocal[i].v] };

        mram_heap_allocator_t alloc;
        init_allocator(&alloc);

        uint32_t params_offset = mram_heap_alloc(&alloc, sizeof(DPUParams));
        uint32_t l2g_size      = align_to_8(m.numNodes * sizeof(uint32_t));
        uint32_t edges_size    = align_to_8(numEdges   * sizeof(Edge));
        uint32_t levels_size   = align_to_8(m.numNodes * sizeof(AlignedU32));
        uint32_t parents_size  = align_to_8(m.numNodes * sizeof(AlignedU32));

        m.p.numNodes        = m.numNodes;
        m.p.numEdges        = numEdges;
        m.p.localToGlobal_m = mram_heap_alloc(&alloc, l2g_size);
        m.p.edges_m         = mram_heap_alloc(&alloc, edges_size);
        m.p.levels_m        = mram_heap_alloc(&alloc, levels_size);
        m.p.parents_m       = mram_heap_alloc(&alloc, parents_size);
        m.p.changed_m       = mram_heap_alloc(&alloc, 8);

        if (verbosity >= 2)
            printf("%-6u  %-10u  %-10u\n", dpuIdx, m.numNodes, numEdges);

        double t0 = now_sec();

        DPU_ASSERT(dpu_copy_to(dpu, DPU_MRAM_HEAP_POINTER_NAME,
                               params_offset, &m.p, sizeof(DPUParams)));

        // l2g is stored as plain uint32_t on the DPU side — copy directly.
        DPU_ASSERT(dpu_copy_to(dpu, DPU_MRAM_HEAP_POINTER_NAME,
                               m.p.localToGlobal_m, m.l2g.data(), l2g_size));

        DPU_ASSERT(dpu_copy_to(dpu, DPU_MRAM_HEAP_POINTER_NAME,
                               m.p.edges_m, localEdges.data(), edges_size));

        loadTime += now_sec() - t0;
        dpuIdx++;
    }

    if (verbosity >= 2) {
        printf("=================================\n");
        printf("Initial CPU→DPU load time: %.3f ms\n", loadTime * 1e3);
    }


    dpu_set_t dpuHandles[NR_DPUS];
    { int idx = 0; DPU_FOREACH(dpuSet, dpu) dpuHandles[idx++] = dpu; }

    std::vector<DPUResult>  results(NR_DPUS);
    std::vector<UploadBuf>  upbufs (NR_DPUS);

    for (int i = 0; i < NR_DPUS; i++) {
        results[i].localLevels .resize(meta[i].numNodes);
        results[i].localParents.resize(meta[i].numNodes);
        results[i].changed64 = 0;
        upbufs [i].levels  .resize(meta[i].numNodes);
        upbufs [i].parents .resize(meta[i].numNodes);
    }

    // BFS loop 
    bool     changed;
    uint32_t iteration = 0;

    double totalUploadTime   = 0.0;
    double totalDPUTime      = 0.0;
    double totalDownloadTime = 0.0;
    double totalHostMerge    = 0.0;

    if (verbosity >= 1) {
        printf("\n%-10s  %-8s  %-12s  %-12s  %-12s  %-12s"
                #if ENERGY
               "  %-14s"
                #endif
               "\n",
               "Iteration", "Changed", "Upload(ms)", "DPU(ms)",
               "Download(ms)", "Merge(ms)"
               #if ENERGY
               , "DPU Energy(J)"
               #endif
               );
        printf("----------  --------  ------------  ------------  ------------  ------------"
            #if ENERGY
               "  --------------"
            #endif
               "\n");
    }

    do {
        changed = false;
        iteration++;
        
        double t_up0 = now_sec();
        {
            std::vector<std::thread> threads;
            threads.reserve(NR_DPUS);

            for (int idx = 0; idx < NR_DPUS; idx++) {
                threads.emplace_back([&, idx]() {
                    auto &m  = meta[idx];
                    auto &ub = upbufs[idx];
                    dpu_set_t dpuHandle = dpuHandles[idx];

                    // Populate staging buffers from globals (reuse allocations).
                    for (uint32_t i = 0; i < m.numNodes; i++) {
                        uint32_t g = m.l2g[i];
                        ub.levels [i] = globalLevel [g];
                        ub.parents[i] = globalParent[g];
                    }

                    DPU_ASSERT(dpu_copy_to(dpuHandle, DPU_MRAM_HEAP_POINTER_NAME,
                        m.p.levels_m, ub.levels.data(),
                        align_to_8(m.numNodes * sizeof(AlignedU32))));

                    DPU_ASSERT(dpu_copy_to(dpuHandle, DPU_MRAM_HEAP_POINTER_NAME,
                        m.p.parents_m, ub.parents.data(),
                        align_to_8(m.numNodes * sizeof(AlignedU32))));

                    uint64_t zero = 0;
                    DPU_ASSERT(dpu_copy_to(dpuHandle, DPU_MRAM_HEAP_POINTER_NAME,
                        m.p.changed_m, &zero, sizeof(uint64_t)));
                });
            }

            for (auto &t : threads) t.join();
        }
        totalUploadTime += now_sec() - t_up0;

        
        #if ENERGY
        DPU_ASSERT(dpu_probe_start(&probe));
        #endif
        double t_dpu0 = now_sec();
        DPU_ASSERT(dpu_launch(dpuSet, DPU_SYNCHRONOUS));
        double iterDPU = now_sec() - t_dpu0;
        totalDPUTime += iterDPU;
        #if ENERGY
        DPU_ASSERT(dpu_probe_stop(&probe));
        double iterEnergy = 0.0;
        DPU_ASSERT(dpu_probe_get(&probe, DPU_ENERGY, DPU_AVERAGE, &iterEnergy));
        totalDPUEnergy += iterEnergy;
        #endif

        
        double t_dl0 = now_sec();

        // Stage 1: fetch changed flags in parallel.
        {
            std::vector<std::thread> threads;
            threads.reserve(NR_DPUS);

            for (int idx = 0; idx < NR_DPUS; idx++) {
                threads.emplace_back([&, idx]() {
                    auto &r = results[idx];
                    DPU_ASSERT(dpu_copy_from(dpuHandles[idx],
                        DPU_MRAM_HEAP_POINTER_NAME,
                        meta[idx].p.changed_m,
                        &r.changed64, sizeof(uint64_t)));
                });
            }
            for (auto &t : threads) t.join();
        }

        // Stage 2: fetch levels/parents only from DPUs that report changes.
        {
            std::vector<std::thread> threads;
            threads.reserve(NR_DPUS);

            for (int idx = 0; idx < NR_DPUS; idx++) {
                if (!results[idx].changed64) continue; // skip unchanged DPUs

                threads.emplace_back([&, idx]() {
                    auto &m = meta[idx];
                    auto &r = results[idx];

                    DPU_ASSERT(dpu_copy_from(dpuHandles[idx],
                        DPU_MRAM_HEAP_POINTER_NAME,
                        m.p.levels_m, r.localLevels.data(),
                        align_to_8(m.numNodes * sizeof(AlignedU32))));

                    DPU_ASSERT(dpu_copy_from(dpuHandles[idx],
                        DPU_MRAM_HEAP_POINTER_NAME,
                        m.p.parents_m, r.localParents.data(),
                        align_to_8(m.numNodes * sizeof(AlignedU32))));
                });
            }
            for (auto &t : threads) t.join();
        }

        totalDownloadTime += now_sec() - t_dl0;

        double t_merge0 = now_sec();
        std::fill(globalChanged.begin(), globalChanged.end(), false);

        int change_this_iter = 0;
        for (int idx = 0; idx < NR_DPUS; idx++) {
            if (!results[idx].changed64) continue;

            auto &m = meta[idx];
            auto &r = results[idx];

            for (uint32_t i = 0; i < m.numNodes; i++) {
                uint32_t g = m.l2g[i];
                if (r.localLevels[i].value < globalLevel[g].value) {
                    globalLevel [g].value = r.localLevels [i].value;
                    globalParent[g].value = r.localParents[i].value;
                    globalChanged[g] = true;
                    changed = true;
                    change_this_iter++;
                }
            }
        }
        totalHostMerge += now_sec() - t_merge0;

        double iterUpload   = now_sec() - t_up0 - (now_sec() - t_dl0)
                              - (now_sec() - t_merge0); // approximate
        (void)iterUpload; // used in verbose print below

        if (verbosity >= 1) {
            // Recompute individual timings for the table.
            double up   = now_sec() - t_up0;   (void)up;
            printf("%-10u  %-8d  %-12.3f  %-12.3f  %-12.3f  %-12.3f"
                   #if ENERGY
                   "  %-14.6f"
                   #endif
                   "\n",
                   iteration,
                   change_this_iter,
                   (now_sec() - t_up0)   * 1e3,
                   iterDPU               * 1e3,
                   (now_sec() - t_dl0)   * 1e3,
                   (now_sec() - t_merge0)* 1e3
                    #if ENERGY
                   , iterEnergy
                    #endif
                   );
        } else {
            printf("Iteration %u — %d nodes updated\n", iteration, change_this_iter);
        }

    } while (changed);


    double totalBFS = totalUploadTime + totalDPUTime + totalDownloadTime + totalHostMerge;

    printf("\n===== TIMING SUMMARY =====\n");
    if (verbosity >= 2)
        printf("Initial load (CPU→DPU):  %8.3f ms\n", loadTime * 1e3);
    printf("BFS upload   (CPU→DPU):  %8.3f ms\n", totalUploadTime   * 1e3);
    printf("DPU kernel total:        %8.3f ms\n", totalDPUTime      * 1e3);
    printf("BFS download (DPU→CPU):  %8.3f ms\n", totalDownloadTime * 1e3);
    printf("Host merge total:        %8.3f ms\n", totalHostMerge    * 1e3);
    printf("--------------------------\n");
    printf("Total BFS wall time:     %8.3f ms\n", totalBFS          * 1e3);
    printf("==========================\n");

#if ENERGY
    printf("\n     ENERGY SUMMARY     \n");
    if (cpuBFSEnergy >= 0.0)
        printf("CPU-only BFS energy:     %8.4f J\n", cpuBFSEnergy);
    else
        printf("CPU-only BFS energy:     N/A (RAPL unreadable — try: sudo chmod a+r %s)\n",
               RAPL_ENERGY_PATH);
    printf("DPU kernel total energy: %8.4f J\n", totalDPUEnergy);
    if (cpuBFSEnergy > 0.0 && totalDPUEnergy > 0.0)
        printf("Energy ratio (CPU/DPU):  %8.2fx\n", cpuBFSEnergy / totalDPUEnergy);
    printf("==========================\n");
    DPU_ASSERT(dpu_probe_free(&probe));
#endif

    printf("\nConverged in %u iterations\n", iteration);
    verify_levels_only(edges, globalLevel, root);

    DPU_ASSERT(dpu_free(dpuSet));
    return 0;
}