// app.cpp — Parallel Relaxed Distributed BFS (UPMEM)
// Compile with -DENERGY=1 to enable DPU + CPU energy measurements.
//
// Usage: ./app [-v 0|1|2] edge_list.txt
//   -v 0  (default) summary only
//   -v 1            + per-iteration timing table
//   -v 2            + DPU partition table and load time
//
// CPU energy via Intel RAPL sysfs. If unreadable:
//   sudo chmod a+r /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj
//
// DPU energy via dpu_probe_t. Build with -DENERGY=1.

extern "C" {
#include <dpu.h>
#include <dpu_log.h>
}

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <unordered_map>
#include <vector>

#include "mram-management.h"

#if ENERGY
extern "C" {
#include <dpu_probe.h>
}
#endif

#define DPU_BINARY "./bin/dpu_code_local"
#define NR_DPUS 16
static const uint32_t INF = UINT32_MAX;

/* ================= TIMER ================= */

static inline double now_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ================= CPU ENERGY (Intel RAPL) ================= */

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

#define RAPL_ENERGY_PATH  "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"
#define RAPL_MAX_PATH     "/sys/class/powercap/intel-rapl/intel-rapl:0/max_energy_range_uj"

/* ================= TYPES ================= */

static inline uint32_t align_to_8(uint32_t x) {
    return (x + 7) & ~7;
}

struct AlignedU32 {
    uint32_t value;
    uint32_t padding;
};

struct Edge {
    uint32_t u;
    uint32_t v;
};

struct DPUParams {
    uint32_t numNodes;
    uint32_t numEdges;

    uint32_t localToGlobal_m;
    uint32_t edges_m;
    uint32_t levels_m;
    uint32_t parents_m;
    uint32_t changed_m;
    uint32_t padding;
};

struct LocalMeta {
    DPUParams p;
    std::vector<uint32_t> l2g;
    uint32_t numNodes;
};

/* ================= GRAPH LOADING ================= */

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

/* ================= VERIFY ================= */

static void verify_levels_only(
    const std::vector<Edge> &edges,
    const std::vector<AlignedU32> &level,
    uint32_t root)
{
    int num_invalid = 0;
    bool valid = true;

    if (level[root].value != 0) {
        printf("ERROR: Root level is not 0\n");
        valid = false;
    }

    for (const auto &e : edges) {
        if (level[e.u].value == UINT32_MAX || level[e.v].value == UINT32_MAX)
            continue;
        uint32_t difference = abs((int)level[e.u].value - (int)level[e.v].value);
        if (difference > 1) { num_invalid++; valid = false; }
    }

    printf("\n===== LEVEL VERIFICATION =====\n");
    if (valid) printf("LEVELS ARE CONSISTENT\n");
    else       printf("LEVELS ARE INVALID WITH %d INVALID EDGES\n", num_invalid);
    printf("==============================\n\n");
}

/* ================= CPU-ONLY BFS (energy baseline) ================= */

static double cpu_bfs_energy(
    const std::vector<Edge> &edges,
    uint32_t numNodes,
    int verbosity)
{
#if ENERGY
    if (verbosity >= 1)
        printf("Running CPU-only BFS for energy baseline...\n");

    std::vector<std::vector<uint32_t>> adj(numNodes);
    for (auto &e : edges) {
        adj[e.u].push_back(e.v);
        adj[e.v].push_back(e.u);
    }

    std::vector<uint32_t> dist(numNodes, INF);
    std::vector<uint32_t> queue(numNodes);
    dist[0] = 0;
    uint32_t head = 0, tail = 0;
    queue[tail++] = 0;

    double uj_before = rapl_read_uj(RAPL_ENERGY_PATH);
    while (head < tail) {
        uint32_t node = queue[head++];
        for (uint32_t nb : adj[node]) {
            if (dist[nb] == INF) {
                dist[nb] = dist[node] + 1;
                queue[tail++] = nb;
            }
        }
    }
    double uj_after = rapl_read_uj(RAPL_ENERGY_PATH);
    return rapl_delta_joules(uj_before, uj_after, RAPL_MAX_PATH);
#else
    (void)edges; (void)numNodes; (void)verbosity;
    return -1.0;
#endif
}

// main function takes two args -- 
// bin file 
// -v 0 ,1, 2 -- verbosity
// ./data/loc-gowalla_edges.txt -- data file
// example -- >  ./bin/app_local -v 1 ./data/loc-gowalla_edges.txt

int main(int argc, char **argv)
{
    //parse args
    int verbosity = 0;
    const char *filename = nullptr;

    for (int i = 1; i < argc; i++) {
        if (argv[i][0] == '-' && argv[i][1] == 'v') {
            if (argv[i][2] != '\0')       verbosity = atoi(&argv[i][2]);
            else if (i + 1 < argc)        verbosity = atoi(argv[++i]);
            else                          verbosity = 1;
        } else {
            filename = argv[i];
        }
    }

    if (!filename) {
        printf("usage: %s [-v 0|1|2] edge_list.txt\n", argv[0]);
        printf("  -v 0  (default) summary only\n");
        printf("  -v 1            + per-iteration timing table\n");
        printf("  -v 2            + DPU partition table and load time\n");
        return 1;
    }

    // Load Graph

    std::vector<Edge> edges;
    uint32_t maxNode = 0;
    read_edge_list(filename, edges, maxNode);
    uint32_t numGlobalNodes = maxNode + 1;
    printf("Loaded %zu edges, %u nodes\n", edges.size(), numGlobalNodes);

    std::vector<Edge> dpuEdges[NR_DPUS];
    for (size_t i = 0; i < edges.size(); i++)
        dpuEdges[i % NR_DPUS].push_back(edges[i]);

    std::vector<AlignedU32> globalLevel(numGlobalNodes, {INF, 0});
    std::vector<AlignedU32> globalParent(numGlobalNodes, {INF, 0});

    uint32_t root = 0;
    globalLevel[root].value  = 0;
    globalParent[root].value = root;

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

    DPU_FOREACH(dpuSet, dpu)
    {
        auto &edgesLocal = dpuEdges[dpuIdx];
        auto &m = meta[dpuIdx];

        std::unordered_map<uint32_t,uint32_t> g2l;

        for (auto &e : edgesLocal) {
            if (!g2l.count(e.u)) { g2l[e.u] = m.l2g.size(); m.l2g.push_back(e.u); }
            if (!g2l.count(e.v)) { g2l[e.v] = m.l2g.size(); m.l2g.push_back(e.v); }
        }

        m.numNodes = m.l2g.size();
        uint32_t numEdges = edgesLocal.size();

        std::vector<Edge> localEdges(numEdges);
        for (uint32_t i = 0; i < numEdges; i++)
            localEdges[i] = { g2l[edgesLocal[i].u], g2l[edgesLocal[i].v] };

        mram_heap_allocator_t alloc;
        init_allocator(&alloc);

        uint32_t params_offset = mram_heap_alloc(&alloc, sizeof(DPUParams));

        uint32_t l2g_size     = align_to_8(m.numNodes * sizeof(uint32_t));
        uint32_t edges_size   = align_to_8(numEdges * sizeof(Edge));
        uint32_t levels_size  = align_to_8(m.numNodes * sizeof(AlignedU32));
        uint32_t parents_size = align_to_8(m.numNodes * sizeof(AlignedU32));

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

    //BFS loop

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
        dpuIdx = 0;

        // do this in parallel 

        
        DPU_FOREACH(dpuSet, dpu)
        {
            auto &m = meta[dpuIdx];

            std::vector<AlignedU32> localLevels(m.numNodes, {INF, 0});
            std::vector<AlignedU32> localParents(m.numNodes, {INF, 0});

            for (uint32_t i = 0; i < m.numNodes; i++) {
                uint32_t g = m.l2g[i];
                localLevels[i].value  = globalLevel[g].value;
                localParents[i].value = globalParent[g].value;
            }

            DPU_ASSERT(dpu_copy_to(dpu, DPU_MRAM_HEAP_POINTER_NAME,
                m.p.levels_m, localLevels.data(),
                align_to_8(m.numNodes * sizeof(AlignedU32))));

            DPU_ASSERT(dpu_copy_to(dpu, DPU_MRAM_HEAP_POINTER_NAME,
                m.p.parents_m, localParents.data(),
                align_to_8(m.numNodes * sizeof(AlignedU32))));

            uint64_t zero = 0;
            DPU_ASSERT(dpu_copy_to(dpu, DPU_MRAM_HEAP_POINTER_NAME,
                m.p.changed_m, &zero, sizeof(uint64_t)));

            dpuIdx++;
        }
        double iterUpload = now_sec() - t_up0;
        totalUploadTime += iterUpload;

        //launch
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

        //pull 
        double t_dl0 = now_sec();
        dpuIdx = 0;
        int change_this_iter = 0;

        DPU_FOREACH(dpuSet, dpu)
        {
            auto &m = meta[dpuIdx];

            std::vector<AlignedU32> localLevels(m.numNodes);
            std::vector<AlignedU32> localParents(m.numNodes);
            uint64_t changed64 = 0;

            DPU_ASSERT(dpu_copy_from(dpu, DPU_MRAM_HEAP_POINTER_NAME,
                m.p.changed_m, &changed64, sizeof(uint64_t)));

            DPU_ASSERT(dpu_copy_from(dpu, DPU_MRAM_HEAP_POINTER_NAME,
                m.p.levels_m, localLevels.data(),
                align_to_8(m.numNodes * sizeof(AlignedU32))));

            DPU_ASSERT(dpu_copy_from(dpu, DPU_MRAM_HEAP_POINTER_NAME,
                m.p.parents_m, localParents.data(),
                align_to_8(m.numNodes * sizeof(AlignedU32))));

            double t_merge0 = now_sec();
            for (uint32_t i = 0; i < m.numNodes; i++) {
                uint32_t g = m.l2g[i];
                if (localLevels[i].value < globalLevel[g].value) {
                    globalLevel[g].value  = localLevels[i].value;
                    globalParent[g].value = localParents[i].value;
                    changed = true;
                    change_this_iter++;
                }
            }
            totalHostMerge += now_sec() - t_merge0;

            dpuIdx++;
        }
        double iterDownload = now_sec() - t_dl0;
        totalDownloadTime += iterDownload;

        if (verbosity >= 1) {
            printf("%-10u  %-8d  %-12.3f  %-12.3f  %-12.3f  %-12.3f"
#if ENERGY
                   "  %-14.6f"
#endif
                   "\n",
                   iteration,
                   change_this_iter,
                   iterUpload   * 1e3,
                   iterDPU      * 1e3,
                   iterDownload * 1e3,
                   totalHostMerge * 1e3
#if ENERGY
                   , iterEnergy
#endif
                   );
        } else {
            // always print iteration progress at v0 so user knows it's running
            printf("Iteration %u — %d nodes updated\n", iteration, change_this_iter);
        }

    } while (changed);

    //summary printing
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
    printf("\n===== ENERGY SUMMARY =====\n");
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