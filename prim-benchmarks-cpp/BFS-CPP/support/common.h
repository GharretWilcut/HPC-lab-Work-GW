#ifndef _COMMON_H_
#define _COMMON_H_

#define ROUND_UP_TO_MULTIPLE_OF_2(x)    ((((x) + 1)/2)*2)
#define ROUND_UP_TO_MULTIPLE_OF_8(x)    ((((x) + 7)/8)*8)
#define ROUND_UP_TO_MULTIPLE_OF_64(x)   ((((x) + 63)/64)*64)

#define setBit(val, idx) ((val) |= (1ULL << (idx)))
#define isSet(val, idx)  ((val) &  (1ULL << (idx)))

struct DPUParams {
    uint32_t numNodes;
    uint32_t dpuStartNodeIdx;
    uint32_t dpuNodePtrsOffset;
    uint32_t level;

    uint32_t dpuNodePtrs_m;
    uint32_t dpuNeighborIdxs_m;
    uint32_t dpuNodeLevel_m;
    uint32_t dpuVisited_m;
    uint32_t dpuCurrentFrontier_m;
    uint32_t dpuNextFrontier_m;
    uint32_t globalRoot;

    uint32_t dpuNumNodes;
};


#endif

