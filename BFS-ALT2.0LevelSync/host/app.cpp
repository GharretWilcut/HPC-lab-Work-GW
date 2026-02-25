// app.cpp — Level-Synchronous Distributed BFS (UPMEM)
// This is basically a hybrid CPU-DPU BFS implementation where the host CPU manages the global BFS state and each DPU is responsible 
// for processing its partition of the graph and updating local levels/parents. 
// The DPUs communicate back to the host using a "changed_flag" in DPU WRAM to indicate if they found any new nodes at the current level,
//  which tells the host whether another iteration is needed.

// This is a cross between the frontier approach that the example has and the complete partition approach that was in the original BFS-ALT2.0.
// Each DPU processes all its edges every iteration, 
// but only updates levels/parents for nodes that are at the current level.


extern "C" {
#include <dpu.h>
#include <dpu_log.h>
#if ENERGY
#include <dpu_probe.h>
#endif
}

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <unordered_map>
#include <vector>

#define DPU_BINARY "./bin/dpu_code_local"
#define NR_DPUS 16
static const uint32_t INF = UINT32_MAX;

// Timer

static inline double now_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Measures CPU energy using RAPL. Requires read permissions on the RAPL sysfs files 
// (e.g. sudo chmod a+r /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj) -- might not work
static double rapl_read_uj(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return -1.0;
    unsigned long long uj = 0;
    fscanf(f, "%llu", &uj);
    fclose(f);
    return (double)uj;
}

// Handles RAPL counter wraparound and converts microjoules to joules. Returns -1.0 on error.
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

//Types

struct AlignedU32 {
    uint32_t value;
    uint32_t padding;
};

struct Edge {
    uint32_t left;
    uint32_t right;
};

struct DPUParams {
    uint32_t numNodes;
    uint32_t numEdges;
    uint32_t currentLevel;
    uint32_t _pad;
};

// loading graphs from edge list files
static void read_edge_list(
    const char *filename,
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

// Verifies that the levels are consistent: root is level 0, and for every edge, the levels of the endpoints differ by at most 1.
static void verify_levels_only(
    const std::vector<Edge> &edges,
    const std::vector<AlignedU32> &level,
    uint32_t root)
{
    bool valid = true;
    int invalid = 0;
    if (level[root].value != 0) valid = false;
    for (const auto &e : edges) {
        if (level[e.left].value == INF || level[e.right].value == INF) continue;
        uint32_t diff = (uint32_t)std::abs((int)level[e.left].value -
                                           (int)level[e.right].value);
        if (diff > 1) { invalid++; valid = false; }
    }
    printf("\n===== LEVEL VERIFICATION =====\n");
    if (valid) printf("LEVELS ARE CONSISTENT\n");
    else       printf("LEVELS INVALID (%d bad edges)\n", invalid);
    printf("==============================\n\n");
}

//Cpu-only BFS to get an energy baseline for comparison. Not timed, 
// since the focus is on DPU energy, but it does measure energy via RAPL if available.
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
        adj[e.left].push_back(e.right);
        adj[e.right].push_back(e.left);
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

// Main application takes the edge list file as a command-line argument, 
// loads the graph, partitions it across DPUs, and runs the level-synchronous BFS loop until completion.
int main(int argc, char **argv)
{
    //parse command-line arguments for verbosity and input file
    int verbosity = 0;
    const char *filename = nullptr;

    for (int i = 1; i < argc; i++) {
        if (argv[i][0] == '-' && argv[i][1] == 'v') {
            // Accept -v N  or  -vN
            if (argv[i][2] != '\0') {
                verbosity = atoi(&argv[i][2]);
            } else if (i + 1 < argc) {
                verbosity = atoi(argv[++i]);
            } else {
                verbosity = 1; // -v with no argument → level 1
            }
        } else {
            filename = argv[i];
        }
    }

    if (!filename) {
        printf("usage: %s [-v 0|1|2] edge_list.txt\n", argv[0]);
        printf("  -v 0  (default) summary only\n");
        printf("  -v 1  + per-level timing table\n");
        printf("  -v 2  + DPU partition table\n");
        return 1;
    }

    //Loads graph

    std::vector<Edge> edges;
    uint32_t maxNode = 0;
    read_edge_list(filename, edges, maxNode);
    uint32_t numGlobalNodes = maxNode + 1;
    printf("Loaded %zu edges, %u nodes\n", edges.size(), numGlobalNodes);

    std::vector<Edge> dpuEdges[NR_DPUS];
    for (size_t i = 0; i < edges.size(); i++)
        dpuEdges[i % NR_DPUS].push_back(edges[i]);

    std::vector<AlignedU32> globalLevel(numGlobalNodes,  {INF, 0});
    std::vector<AlignedU32> globalParent(numGlobalNodes, {INF, 0});
    uint32_t root = 0;
    globalLevel[root].value  = 0;
    globalParent[root].value = root;


    #if ENERGY
    double cpuBFSEnergy = cpu_bfs_energy(edges, numGlobalNodes, verbosity);
    #endif

    //Initializes DPUs and uploads graph partitions and initial BFS state to each DPU. Each DPU gets its own portion of the edge list,
    dpu_set_t dpuSet, dpu;
    DPU_ASSERT(dpu_alloc(NR_DPUS, NULL, &dpuSet));
    DPU_ASSERT(dpu_load(dpuSet, DPU_BINARY, NULL));

    #if ENERGY
    struct dpu_probe_t probe;
    DPU_ASSERT(dpu_probe_init("energy_probe", &probe));
    double totalDPUEnergy = 0.0;
    #endif

    struct LocalMeta {
        DPUParams               p;
        std::vector<AlignedU32> l2g;
        std::vector<Edge>       localEdges;
    };
    std::vector<LocalMeta> meta(NR_DPUS);

    // Initial load of graph partitions and BFS state to DPUs. Also prints a summary table of the number of nodes/edges in each DPU partition, which gives insight into load balance.
    if (verbosity >= 2) {
        printf("\n===== DPU PARTITION SUMMARY =====\n");
        printf("%-6s  %-10s  %-10s\n", "DPU", "Nodes", "Edges");
        printf("------  ----------  ----------\n");
    }

    double loadTime = 0.0;
    uint32_t dpuIdx = 0;
    // For each DPU, we create a local-to-global mapping for the nodes in its partition, convert the edges to use local node IDs, 
    // and upload the edges, local-to-global mapping, and initial BFS state (levels/parents) to the DPU's MRAM.
    DPU_FOREACH(dpuSet, dpu)
    {
        auto &edgesLocal = dpuEdges[dpuIdx];
        auto &m          = meta[dpuIdx];

        std::unordered_map<uint32_t, uint32_t> g2l;
        for (auto &e : edgesLocal) {
            if (!g2l.count(e.left))  { g2l[e.left]  = (uint32_t)m.l2g.size(); m.l2g.push_back({e.left,  0}); }
            if (!g2l.count(e.right)) { g2l[e.right] = (uint32_t)m.l2g.size(); m.l2g.push_back({e.right, 0}); }
        }

        uint32_t numNodes = (uint32_t)m.l2g.size();
        uint32_t numEdges = (uint32_t)edgesLocal.size();

        m.localEdges.resize(numEdges);
        for (uint32_t i = 0; i < numEdges; i++)
            m.localEdges[i] = { g2l[edgesLocal[i].left], g2l[edgesLocal[i].right] };

        m.p = { numNodes, numEdges, 0, 0 };

        if (verbosity >= 2)
            printf("%-6u  %-10u  %-10u\n", dpuIdx, numNodes, numEdges);

        double t0 = now_sec();
        DPU_ASSERT(dpu_copy_to(dpu, "param",         0, &m.p,                sizeof(DPUParams)));
        DPU_ASSERT(dpu_copy_to(dpu, "localToGlobal", 0, m.l2g.data(),        numNodes * sizeof(AlignedU32)));
        DPU_ASSERT(dpu_copy_to(dpu, "edges_arr",     0, m.localEdges.data(), numEdges * sizeof(Edge)));
        AlignedU32 zero = {0, 0};
        DPU_ASSERT(dpu_copy_to(dpu, "changed_flag",  0, &zero,               sizeof(AlignedU32)));
        loadTime += now_sec() - t0;

        dpuIdx++;
    }

    if (verbosity >= 2) {
        printf("=================================\n");
        printf("Initial CPU→DPU load time: %.3f ms\n", loadTime * 1e3);
    }

    //BFS loop: The host CPU manages the global BFS state and iteratively launches the DPUs to process their partitions until no more nodes are found at the next level.
    
    uint32_t level           = 0;
    bool     changed;

    double totalUploadTime   = 0.0;
    double totalDPUTime      = 0.0;
    double totalDownloadTime = 0.0;
    double totalHostMerge    = 0.0;

    if (verbosity >= 1) {
        printf("\n%-8s  %-12s  %-12s  %-12s  %-12s"
        #if ENERGY
               "  %-14s"
        #endif
               "\n",
               "Level", "Upload(ms)", "DPU(ms)", "Download(ms)", "Merge(ms)"
        #if ENERGY
               , "DPU Energy(J)"
        #endif
               );
        printf("--------  ------------  ------------  ------------  ------------"
        #if ENERGY
               "  --------------"
        #endif
               "\n");
    }

    do {
        changed = false;

        double t_up0 = now_sec();
        dpuIdx = 0;
        DPU_FOREACH(dpuSet, dpu)
        {
            auto &m = meta[dpuIdx];
            uint32_t numNodes = m.p.numNodes;

            std::vector<AlignedU32> localLevels(numNodes);
            std::vector<AlignedU32> localParents(numNodes);
            for (uint32_t i = 0; i < numNodes; i++) {
                localLevels[i]  = globalLevel[m.l2g[i].value];
                localParents[i] = globalParent[m.l2g[i].value];
            }

            DPU_ASSERT(dpu_copy_to(dpu, "levels",  0, localLevels.data(),  numNodes * sizeof(AlignedU32)));
            DPU_ASSERT(dpu_copy_to(dpu, "parents", 0, localParents.data(), numNodes * sizeof(AlignedU32)));
            AlignedU32 zero = {0, 0};
            DPU_ASSERT(dpu_copy_to(dpu, "changed_flag", 0, &zero, sizeof(AlignedU32)));
            m.p.currentLevel = level;
            DPU_ASSERT(dpu_copy_to(dpu, "param", 0, &m.p, sizeof(DPUParams)));
            dpuIdx++;
        }
        double levelUpload = now_sec() - t_up0;
        totalUploadTime += levelUpload;

        //Launch
        #if ENERGY
        DPU_ASSERT(dpu_probe_start(&probe));
        #endif
        double t_dpu0 = now_sec();
        DPU_ASSERT(dpu_launch(dpuSet, DPU_SYNCHRONOUS));
        double levelDPU = now_sec() - t_dpu0;
        totalDPUTime += levelDPU;
        #if ENERGY
        DPU_ASSERT(dpu_probe_stop(&probe));
        double levelEnergy = 0.0;
        DPU_ASSERT(dpu_probe_get(&probe, DPU_ENERGY, DPU_AVERAGE, &levelEnergy));
        totalDPUEnergy += levelEnergy;
        #endif

        //pulls results from each DPU, merges into globalLevel/globalParent, and checks if we need another iteration
        double t_dl0 = now_sec();
        dpuIdx = 0;
        DPU_FOREACH(dpuSet, dpu)
        {
            auto &m = meta[dpuIdx];
            uint32_t numNodes = m.p.numNodes;
            //gets changed flag from DPU to see if we need another iteration
            AlignedU32 ch;
            DPU_ASSERT(dpu_copy_from(dpu, "changed_flag", 0, &ch, sizeof(AlignedU32)));
            if (ch.value == 1) changed = true;
            //pulls levels and parents from DPU and merges into global arrays
            std::vector<AlignedU32> localLevels(numNodes);
            DPU_ASSERT(dpu_copy_from(dpu, "levels", 0, localLevels.data(), numNodes * sizeof(AlignedU32)));

            std::vector<AlignedU32> localParents(numNodes);
            DPU_ASSERT(dpu_copy_from(dpu, "parents", 0, localParents.data(), numNodes * sizeof(AlignedU32)));
            //merges local results into global arrays
            double t_merge0 = now_sec();
            for (uint32_t i = 0; i < numNodes; i++) {
                uint32_t g = m.l2g[i].value;
                if (localLevels[i].value < globalLevel[g].value) {
                    globalLevel[g]  = localLevels[i];
                    globalParent[g] = localParents[i];
                }
            }
            totalHostMerge += now_sec() - t_merge0;
            dpuIdx++;
        }
        double levelDownload = now_sec() - t_dl0;
        totalDownloadTime += levelDownload;

        if (verbosity >= 1) {
            printf("%-8u  %-12.3f  %-12.3f  %-12.3f  %-12.3f"
            #if ENERGY
                   "  %-14.6f"
            #endif
                   "\n",
                   level,
                   levelUpload   * 1e3,
                   levelDPU      * 1e3,
                   levelDownload * 1e3,
                   totalHostMerge * 1e3
            #if ENERGY
                   , levelEnergy
            #endif
                   );
        }

        if (changed) level++;

    } while (changed);

    // always print the summary tables, even if verbose=0, since they include the total BFS time and energy

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
        printf("CPU-only BFS energy:     N/A (RAPL unreadable — try: sudo chmod a+r %s)\n", RAPL_ENERGY_PATH);
    printf("DPU kernel total energy: %8.4f J\n", totalDPUEnergy);
    if (cpuBFSEnergy > 0.0 && totalDPUEnergy > 0.0)
        printf("Energy ratio (CPU/DPU):  %8.2fx\n", cpuBFSEnergy / totalDPUEnergy);
    printf("==========================\n");
    DPU_ASSERT(dpu_probe_deinit(&probe));
    #endif

    printf("\nBFS completed at level %u\n", level);
    verify_levels_only(edges, globalLevel, root);

    DPU_ASSERT(dpu_free(dpuSet));
    return 0;
}