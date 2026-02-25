#ifndef _COMMON_H_
#define _COMMON_H_

#define ROUND_UP_TO_MULTIPLE_OF_2(x)    ((((x) + 1)/2)*2)
#define ROUND_UP_TO_MULTIPLE_OF_8(x)    ((((x) + 7)/8)*8)
#define ROUND_UP_TO_MULTIPLE_OF_64(x)   ((((x) + 63)/64)*64)

#define setBit(val, idx) ((val) |= (1ULL << (idx)))
#define isSet(val, idx)  ((val) &  (1ULL << (idx)))

// this needs to be different since we are passing edge lists and doing traversals on node objects in DPUs
struct DPUParams {
    
};

#endif

