// task.c — DPU BFS kernel with self-built index
 

#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <mram.h>
#include <defs.h>
#include <barrier.h>
#include <mutex.h>

#define INF                      0xFFFFFFFF
#define EDGE_BATCH               96
#define MAX_CHANGED_PER_TASKLET  64    

#define RADIX_BUCKETS  256
#define SORT_BATCH     512
#define EDGE_BUF       256

BARRIER_INIT(barrier, NR_TASKLETS);
MUTEX_INIT(fallback_mutex);
MUTEX_INIT(cursor_mutex);    

typedef struct {
    uint32_t numNodes;
    uint32_t numEdges;
    uint32_t localToGlobal_m;
    uint32_t edges_m;
    uint32_t levels_m;
    uint32_t changedNodes_m;
    uint32_t countOnly_m;
    uint32_t isIndexPass;
} DPUParams;  // 32 bytes

typedef struct { uint32_t left;  uint32_t right; } Edge;
typedef struct { uint32_t value; uint32_t parent; } AlignedU32;

typedef struct {
    uint32_t globalID;
    uint32_t value;
    uint32_t parent;
    uint32_t _pad;
} ChangedNode;  // 16 bytes
 
static ChangedNode changedBuf[NR_TASKLETS][MAX_CHANGED_PER_TASKLET]
    __attribute__((aligned(8)));
static uint32_t changedCnt[NR_TASKLETS];
static uint32_t writeBase[NR_TASKLETS];

 
static uint32_t mram_write_base;

 
static uint32_t s_sort_buf [SORT_BATCH]    __attribute__((aligned(8)));
static uint32_t s_hist     [RADIX_BUCKETS] __attribute__((aligned(8)));
static uint32_t s_prefix   [RADIX_BUCKETS] __attribute__((aligned(8)));
static uint32_t s_dedup_w  [SORT_BATCH]    __attribute__((aligned(8)));
static Edge     s_edge_buf [EDGE_BUF]      __attribute__((aligned(8)));

 
static inline uint32_t mram_read_u32(
        __mram_ptr uint32_t *arr, uint32_t idx)
{
    uint32_t pair[2] __attribute__((aligned(8)));
    mram_read(&arr[idx & ~1u], pair, 8);
    return pair[idx & 1];
}

static inline void mram_write_u32(
        __mram_ptr uint32_t *arr, uint32_t idx, uint32_t val)
{
    uint32_t pair[2] __attribute__((aligned(8)));
    mram_read(&arr[idx & ~1u], pair, 8);
    pair[idx & 1] = val;
    mram_write(pair, &arr[idx & ~1u], 8);
}
 
static void flush_changed(
        uint32_t id, __mram_ptr ChangedNode *mram_changed)
{
    uint32_t base;
    mutex_lock(cursor_mutex);
    base = mram_write_base;
    mram_write_base += MAX_CHANGED_PER_TASKLET;
    mutex_unlock(cursor_mutex);

    mram_write(changedBuf[id],
               &mram_changed[base],
               MAX_CHANGED_PER_TASKLET * sizeof(ChangedNode));
    changedCnt[id] = 0;
}

static void radix_histogram(
        __mram_ptr uint32_t *src, uint32_t n, uint32_t shift)
{
    for (uint32_t b = 0; b < RADIX_BUCKETS; b++) s_hist[b] = 0;
    uint32_t i = 0;
    while (i < n) {
        uint32_t fetch = (n - i < SORT_BATCH) ? (n - i) : SORT_BATCH;
        uint32_t fa    = (fetch + 1) & ~1u;
        mram_read(&src[i], s_sort_buf, fa * sizeof(uint32_t));
        for (uint32_t k = 0; k < fetch; k++)
            s_hist[(s_sort_buf[k] >> shift) & 0xFF]++;
        i += fetch;
    }
}

static void radix_scatter(
        __mram_ptr uint32_t *src, __mram_ptr uint32_t *dst,
        uint32_t n, uint32_t shift)
{
    uint32_t i = 0;
    while (i < n) {
        uint32_t fetch = (n - i < SORT_BATCH) ? (n - i) : SORT_BATCH;
        uint32_t fa    = (fetch + 1) & ~1u;
        mram_read(&src[i], s_sort_buf, fa * sizeof(uint32_t));
        for (uint32_t k = 0; k < fetch; k++) {
            uint32_t val    = s_sort_buf[k];
            uint32_t bucket = (val >> shift) & 0xFF;
            uint32_t out    = s_prefix[bucket]++;
            mram_write_u32(dst, out, val);
        }
        i += fetch;
    }
}

static void mram_copy_u32(
        __mram_ptr uint32_t *dst, __mram_ptr uint32_t *src, uint32_t n)
{
    uint32_t i = 0;
    while (i < n) {
        uint32_t fetch = (n - i < SORT_BATCH) ? (n - i) : SORT_BATCH;
        uint32_t fa    = (fetch + 1) & ~1u;
        mram_read(&src[i],  s_sort_buf, fa * sizeof(uint32_t));
        mram_write(s_sort_buf, &dst[i], fa * sizeof(uint32_t));
        i += fetch;
    }
}

static void radix_sort_u32(
        __mram_ptr uint32_t *mram_a, __mram_ptr uint32_t *mram_b, uint32_t n)
{
    __mram_ptr uint32_t *s = mram_a;
    __mram_ptr uint32_t *d = mram_b;
    for (uint32_t pass = 0; pass < 3; pass++) {
        uint32_t shift = pass * 8;
        radix_histogram(s, n, shift);
        uint32_t acc = 0;
        for (uint32_t b = 0; b < RADIX_BUCKETS; b++) {
            s_prefix[b] = acc; acc += s_hist[b];
        }
        radix_scatter(s, d, n, shift);
        __mram_ptr uint32_t *tmp = s; s = d; d = tmp;
    }
    if (s != mram_a) mram_copy_u32(mram_a, s, n);
}

static uint32_t dedup_sorted_u32(__mram_ptr uint32_t *arr, uint32_t n)
{
    if (n == 0) return 0;
    uint32_t wcount = 0, wbase = 0, unique = 0;
    uint32_t prev = INF;
    uint32_t i = 0;
    while (i < n) {
        uint32_t fetch = (n - i < SORT_BATCH) ? (n - i) : SORT_BATCH;
        uint32_t fa    = (fetch + 1) & ~1u;
        mram_read(&arr[i], s_sort_buf, fa * sizeof(uint32_t));
        for (uint32_t k = 0; k < fetch; k++) {
            uint32_t v = s_sort_buf[k];
            if (v == prev) continue;
            prev = v;
            s_dedup_w[wcount++] = v;
            unique++;
            if (wcount == SORT_BATCH) {
                mram_write(s_dedup_w, &arr[wbase],
                           SORT_BATCH * sizeof(uint32_t));
                wbase += SORT_BATCH; wcount = 0;
            }
        }
        i += fetch;
    }
    if (wcount > 0) {
        uint32_t flush = (wcount + 1) & ~1u;
        if (flush > wcount) s_dedup_w[wcount] = s_dedup_w[wcount - 1];
        mram_write(s_dedup_w, &arr[wbase], flush * sizeof(uint32_t));
    }
    return unique;
}

static uint32_t g2l_search(
        __mram_ptr uint32_t *l2g, uint32_t numNodes, uint32_t g)
{
    uint32_t lo = 0, hi = numNodes;
    while (lo < hi) {
        uint32_t mid = (lo + hi) >> 1;
        uint32_t val = mram_read_u32(l2g, mid);
        if      (val == g) return mid;
        else if (val  < g) lo = mid + 1;
        else               hi = mid;
    }
    return INF;
}


static inline void record_changed(
        uint32_t id,
        uint32_t globalID, uint32_t value, uint32_t parent,
        __mram_ptr ChangedNode *mram_changed)
{
    changedBuf[id][changedCnt[id]].globalID = globalID;
    changedBuf[id][changedCnt[id]].value    = value;
    changedBuf[id][changedCnt[id]].parent   = parent;
    changedCnt[id]++;
    if (changedCnt[id] == MAX_CHANGED_PER_TASKLET)
        flush_changed(id, mram_changed);
}

int main()
{
    uint32_t id = me();

    DPUParams p;
    mram_read((__mram_ptr DPUParams *)DPU_MRAM_HEAP_POINTER,
              &p, sizeof(DPUParams));

    __mram_ptr Edge        *mram_edges   =
        (__mram_ptr Edge *)       (DPU_MRAM_HEAP_POINTER + p.edges_m);
    __mram_ptr uint32_t    *mram_l2g     =
        (__mram_ptr uint32_t *)   (DPU_MRAM_HEAP_POINTER + p.localToGlobal_m);
    __mram_ptr AlignedU32  *mram_levels  =
        (__mram_ptr AlignedU32 *) (DPU_MRAM_HEAP_POINTER + p.levels_m);
    __mram_ptr ChangedNode *mram_changed =
        (__mram_ptr ChangedNode *)(DPU_MRAM_HEAP_POINTER + p.changedNodes_m);

 
    if (p.isIndexPass) {
        if (id == 0) {
            uint32_t numEdges = p.numEdges;

            {
                uint32_t wcount = 0, wbase = 0;
                for (uint32_t i = 0; i < numEdges; ) {
                    uint32_t fetch = (numEdges - i < EDGE_BUF)
                                   ? (numEdges - i) : EDGE_BUF;
                    uint32_t fa    = (fetch + 1) & ~1u;
                    mram_read(&mram_edges[i], s_edge_buf, fa * sizeof(Edge));
                    for (uint32_t k = 0; k < fetch; k++) {
                        s_dedup_w[wcount++] = s_edge_buf[k].left;
                        if (wcount == SORT_BATCH) {
                            mram_write(s_dedup_w, &mram_l2g[wbase],
                                       SORT_BATCH * sizeof(uint32_t));
                            wbase += SORT_BATCH; wcount = 0;
                        }
                        s_dedup_w[wcount++] = s_edge_buf[k].right;
                        if (wcount == SORT_BATCH) {
                            mram_write(s_dedup_w, &mram_l2g[wbase],
                                       SORT_BATCH * sizeof(uint32_t));
                            wbase += SORT_BATCH; wcount = 0;
                        }
                    }
                    i += fetch;
                }
                if (wcount > 0) {
                    uint32_t flush = (wcount + 1) & ~1u;
                    if (flush > wcount) s_dedup_w[wcount] = s_dedup_w[wcount-1];
                    mram_write(s_dedup_w, &mram_l2g[wbase],
                               flush * sizeof(uint32_t));
                }
            }

            uint32_t rawCount = 2 * numEdges;

            __mram_ptr uint32_t *scratch =
                (__mram_ptr uint32_t *)(DPU_MRAM_HEAP_POINTER + p.levels_m);
            radix_sort_u32(mram_l2g, scratch, rawCount);

            uint32_t numNodes = dedup_sorted_u32(mram_l2g, rawCount);

            for (uint32_t i = 0; i < numEdges; ) {
                uint32_t fetch = (numEdges - i < EDGE_BUF)
                               ? (numEdges - i) : EDGE_BUF;
                uint32_t fa    = (fetch + 1) & ~1u;
                mram_read(&mram_edges[i], s_edge_buf, fa * sizeof(Edge));
                for (uint32_t k = 0; k < fetch; k++) {
                    s_edge_buf[k].left  = g2l_search(mram_l2g, numNodes,
                                                      s_edge_buf[k].left);
                    s_edge_buf[k].right = g2l_search(mram_l2g, numNodes,
                                                      s_edge_buf[k].right);
                }
                if (fa > fetch) {
                    s_edge_buf[fetch].left  = s_edge_buf[fetch-1].left;
                    s_edge_buf[fetch].right = s_edge_buf[fetch-1].right;
                }
                mram_write(s_edge_buf, &mram_edges[i], fa * sizeof(Edge));
                i += fetch;
            }

            {
                uint32_t header[2] __attribute__((aligned(8)));
                mram_read((__mram_ptr void *)DPU_MRAM_HEAP_POINTER, header, 8);
                header[0] = numNodes;
                mram_write(header,
                           (__mram_ptr void *)DPU_MRAM_HEAP_POINTER, 8);
            }
        }
        barrier_wait(&barrier);
        return 0;
    }

 
    if (id == 0) mram_write_base = 0;
    changedCnt[id] = 0;

    barrier_wait(&barrier);

    Edge       ebuf[EDGE_BATCH] __attribute__((aligned(8)));
    AlignedU32 lbuf[2]          __attribute__((aligned(8)));
    AlignedU32 wbuf[1]          __attribute__((aligned(8)));

    uint32_t i      = id * EDGE_BATCH;
    uint32_t stride = NR_TASKLETS * EDGE_BATCH;

    while (i < p.numEdges) {
        uint32_t fetch = (p.numEdges - i < EDGE_BATCH)
                       ? (p.numEdges - i) : EDGE_BATCH;

        mram_read(&mram_edges[i], ebuf, fetch * sizeof(Edge));

        for (uint32_t k = 0; k < fetch; k++) {
            uint32_t L = ebuf[k].left;
            uint32_t R = ebuf[k].right;

            mram_read(&mram_levels[L], &lbuf[0], sizeof(AlignedU32));
            mram_read(&mram_levels[R], &lbuf[1], sizeof(AlignedU32));
            uint32_t lvl_L = lbuf[0].value;
            uint32_t lvl_R = lbuf[1].value;

            // Relax L -> R
            if (lvl_L != INF && lvl_L + 1 < lvl_R) {
                mutex_lock(fallback_mutex);
                mram_read(&mram_levels[R], wbuf,     sizeof(AlignedU32));
                mram_read(&mram_levels[L], &lbuf[0], sizeof(AlignedU32));
                lvl_L = lbuf[0].value;
                if (lvl_L != INF && lvl_L + 1 < wbuf[0].value) {
                    uint32_t globalL = mram_read_u32(mram_l2g, L);
                    uint32_t globalR = mram_read_u32(mram_l2g, R);
                    wbuf[0].value  = lvl_L + 1;
                    wbuf[0].parent = globalL;
                    mram_write(wbuf, &mram_levels[R], sizeof(AlignedU32));
                    record_changed(id, globalR, wbuf[0].value, globalL,
                                   mram_changed);
                }
                mutex_unlock(fallback_mutex);
            }

            mram_read(&mram_levels[L], &lbuf[0], sizeof(AlignedU32));
            mram_read(&mram_levels[R], &lbuf[1], sizeof(AlignedU32));
            lvl_L = lbuf[0].value;
            lvl_R = lbuf[1].value;

            if (lvl_R != INF && lvl_R + 1 < lvl_L) {
                mutex_lock(fallback_mutex);
                mram_read(&mram_levels[L], wbuf,     sizeof(AlignedU32));
                mram_read(&mram_levels[R], &lbuf[1], sizeof(AlignedU32));
                lvl_R = lbuf[1].value;
                if (lvl_R != INF && lvl_R + 1 < wbuf[0].value) {
                    uint32_t globalR = mram_read_u32(mram_l2g, R);
                    uint32_t globalL = mram_read_u32(mram_l2g, L);
                    wbuf[0].value  = lvl_R + 1;
                    wbuf[0].parent = globalR;
                    mram_write(wbuf, &mram_levels[L], sizeof(AlignedU32));
                    record_changed(id, globalL, wbuf[0].value, globalR,
                                   mram_changed);
                }
                mutex_unlock(fallback_mutex);
            }
        }
        i += stride;
    }

    barrier_wait(&barrier);

    if (changedCnt[id] > 0) {
        uint32_t base;
        mutex_lock(cursor_mutex);
        base = mram_write_base;
        mram_write_base += changedCnt[id];
        mutex_unlock(cursor_mutex);

        mram_write(changedBuf[id],
                   &mram_changed[base],
                   changedCnt[id] * sizeof(ChangedNode));
    }

    barrier_wait(&barrier);

    if (id == 0) {
        uint64_t count_out __attribute__((aligned(8)));
        count_out = (uint64_t)mram_write_base;
        mram_write(&count_out,
                   (__mram_ptr void *)(DPU_MRAM_HEAP_POINTER + p.countOnly_m),
                   sizeof(uint64_t));
    }

    return 0;
}