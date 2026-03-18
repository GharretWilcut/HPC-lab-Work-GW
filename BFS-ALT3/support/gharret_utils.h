
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
static double rapl_delta_joules(double before, double after, const char *maxp) {
    if (before < 0 || after < 0) return -1.0;
    double d = after - before;
    if (d < 0) { double m = rapl_read_uj(maxp); if (m > 0) d += m; }
    return d * 1e-6;
}
#define RAPL_ENERGY_PATH "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"
#define RAPL_MAX_PATH    "/sys/class/powercap/intel-rapl/intel-rapl:0/max_energy_range_uj"

// need to redo aligned U32 struct to better utilize the 8-byte aligned xfers and avoid padding waste
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
    uint32_t                xfer_nodes; 
};


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

// only verifies that levels are consistent not bfs accurate (i.e. no level should differ by more than 1 from its neighbors)
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

static void cpu_bfs(const std::vector<Edge> &edges,
                    uint32_t numNodes, uint32_t root,
                    std::vector<uint32_t> &outLevel,
                    std::vector<uint32_t> &outParent)
{
    std::vector<std::vector<uint32_t>> adj(numNodes);
    for (auto &e : edges){
        adj[e.u].push_back(e.v);
        adj[e.v].push_back(e.u);
    }

    outLevel.assign(numNodes,INF);
    outParent.assign(numNodes,INF);
    outLevel[root] = 0;
    outParent[root] = root;

    std::vector<uint32_t> queue;
    queue.reserve(numNodes);
    queue.push_back(root);

    for (size_t head =0; head < queue.size();head++){
        uint32_t u = queue[head];
        for (uint32_t v : adj[u]){
            if (outLevel[v] == INF){
                outLevel[v] = outLevel[u]+1;
                outParent[v]=u;
                queue.push_back(v);
            }
        }
    }
} 

static void compare_bfs(const std::vector<AlignedU32> &dpuLevel,
                        const std::vector<uint32_t> &cpuLevel, uint32_t numNodes)
{
    uint32_t mismatches = 0,dpu_inf  = 0, cpu_inf =0, both_inf = 0;

    for (uint32_t n = 0; n < numNodes; n++){
        bool dpu_unreachable = (dpu_unreachable = (dpuLevel[n].value == INF));
        bool cpu_unreachable = (cpu_unreachable = (cpuLevel[n] == INF));
        if (dpu_unreachable && cpu_unreachable) { both_inf++; continue; }
        if (dpu_unreachable) {dpu_inf++; mismatches++; continue;}
        if (cpu_unreachable) {cpu_inf++; mismatches++; continue;}
        if (dpuLevel[n].value != cpuLevel[n]) mismatches++;
    }
    printf("Mismatches: %u, DPU Inf: %u, CPU Inf: %u, Both Inf: %u\n", mismatches, dpu_inf, cpu_inf, both_inf);
}