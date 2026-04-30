#pragma once

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
#include <unordered_set>
#include <vector>

static const uint32_t INF = UINT32_MAX;

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

#define RAPL_ENERGY_PATH "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"
#define RAPL_MAX_PATH    "/sys/class/powercap/intel-rapl/intel-rapl:0/max_energy_range_uj"

// -----------------------------------------------------------------------
// Core types shared between host (app.cpp) and DPU (task.c).
// ALL structs must be multiples of 8 bytes for MRAM alignment.
// -----------------------------------------------------------------------

// BFS level + parent packed into one 8-byte MRAM word.
// Field is named 'parent' (not 'padding') — carries the global parent node ID.
struct AlignedU32 {
    uint32_t value;   // BFS level (INF = undiscovered)
    uint32_t parent;  // global node ID of BFS parent (INF = none)
};

struct Edge { uint32_t u, v; };

// DPUParams — written once to each DPU's MRAM before the BFS loop.
// Must be exactly 32 bytes (8-byte aligned).
//
// Fields:
//   numNodes        – number of local nodes on this DPU
//   numEdges        – number of local edges on this DPU
//   localToGlobal_m – MRAM offset of the L2G lookup table  (uint32_t[numNodes], padded)
//   edges_m         – MRAM offset of the edge array        (Edge[numEdges], padded)
//   levels_m        – MRAM offset of the levels array      (AlignedU32[numNodes], padded)
//   changedNodes_m  – MRAM offset of sparse output array   (ChangedNode[NR_TASKLETS * MAX_CHANGED_PER_TASKLET])
//   changedCount_m  – MRAM offset of per-tasklet counts    (uint32_t[NR_TASKLETS], padded)
//   _pad            – padding to reach 32 bytes
typedef struct {
    uint32_t numNodes;       // filled by DPU after index build
    uint32_t numEdges;       // set by CPU before launch
    uint32_t localToGlobal_m;
    uint32_t edges_m;
    uint32_t levels_m;
    uint32_t changedNodes_m;
    uint32_t countOnly_m;
    uint32_t isIndexPass;    // 1 = build index, 0 = BFS iteration
} DPUParams;  // 32 bytes

static_assert(sizeof(DPUParams) == 32, "DPUParams must be 32 bytes");
static_assert(sizeof(DPUParams) % 8 == 0, "DPUParams must be 8-byte aligned");

// Per-DPU metadata kept on the host side only (never sent to DPU).
struct LocalMeta {
    DPUParams                             p;
    std::vector<uint32_t>                 l2g;   // local→global node ID
    std::unordered_map<uint32_t,uint32_t> g2l;   // global→local node ID
    uint32_t                              numNodes;
    uint32_t                              xfer_nodes;  // padded node count used for xfer rows
};

// -----------------------------------------------------------------------
// read_edge_list — load an undirected edge list from a text file.
// Format: one "u v" pair per line (0-based or 1-based, auto-detected).
// Populates `edges` and sets `maxNode` to the highest node ID seen.
// -----------------------------------------------------------------------
static void read_edge_list(const char *filename,
                           std::vector<Edge> &edges,
                           uint32_t &maxNode)
{
    FILE *f = fopen(filename, "r");
    if (!f) { perror(filename); exit(1); }

    maxNode = 0;
    uint32_t u, v;
    while (fscanf(f, "%u %u", &u, &v) == 2) {
        edges.push_back({u, v});
        if (u > maxNode) maxNode = u;
        if (v > maxNode) maxNode = v;
    }
    fclose(f);
}

// -----------------------------------------------------------------------
// verify_levels — checks that no two neighbours differ by more than 1.
// -----------------------------------------------------------------------
static void verify_levels(const std::vector<Edge> &edges,
                          const std::vector<AlignedU32> &level,
                          uint32_t root)
{
    int bad = 0;
    bool ok = true;
    if (level[root].value != 0) { printf("ERROR: root level != 0\n"); ok = false; }
    for (auto &e : edges) {
        if (level[e.u].value == INF || level[e.v].value == INF) continue;
        uint32_t diff = (level[e.u].value > level[e.v].value)
                      ? level[e.u].value - level[e.v].value
                      : level[e.v].value - level[e.u].value;
        if (diff > 1) { bad++; ok = false; }
    }
    printf("\n===== LEVEL VERIFICATION =====\n");
    if (ok) printf("LEVELS ARE CONSISTENT\n");
    else    printf("LEVELS ARE INVALID WITH %d INVALID EDGES\n", bad);
    printf("==============================\n\n");
}

// -----------------------------------------------------------------------
// cpu_bfs — reference single-threaded BFS on the host.
// -----------------------------------------------------------------------
static void cpu_bfs(const std::vector<Edge> &edges,
                    uint32_t numNodes, uint32_t root,
                    std::vector<uint32_t> &outLevel,
                    std::vector<uint32_t> &outParent)
{
    std::vector<std::vector<uint32_t>> adj(numNodes);
    for (auto &e : edges) {
        adj[e.u].push_back(e.v);
        adj[e.v].push_back(e.u);
    }

    outLevel.assign(numNodes, INF);
    outParent.assign(numNodes, INF);
    outLevel[root]  = 0;
    outParent[root] = root;

    std::vector<uint32_t> queue;
    queue.reserve(numNodes);
    queue.push_back(root);

    for (size_t head = 0; head < queue.size(); head++) {
        uint32_t u = queue[head];
        for (uint32_t v : adj[u]) {
            if (outLevel[v] == INF) {
                outLevel[v]  = outLevel[u] + 1;
                outParent[v] = u;
                queue.push_back(v);
            }
        }
    }
}

// -----------------------------------------------------------------------
// compare_bfs — diff DPU result against CPU reference.
// -----------------------------------------------------------------------
static void compare_bfs(const std::vector<AlignedU32> &dpuLevel,
                        const std::vector<uint32_t>   &cpuLevel,
                        uint32_t numNodes)
{
    uint32_t mismatches = 0, dpu_inf = 0, cpu_inf = 0, both_inf = 0;
    for (uint32_t n = 0; n < numNodes; n++) {
        bool du = (dpuLevel[n].value == INF);
        bool cu = (cpuLevel[n]       == INF);
        if (du && cu) { both_inf++; continue; }
        if (du)       { dpu_inf++;  mismatches++; continue; }
        if (cu)       { cpu_inf++;  mismatches++; continue; }
        if (dpuLevel[n].value != cpuLevel[n]) mismatches++;
    }
    printf("Mismatches: %u  DPU-only-INF: %u  CPU-only-INF: %u  Both-INF: %u\n",
           mismatches, dpu_inf, cpu_inf, both_inf);
}