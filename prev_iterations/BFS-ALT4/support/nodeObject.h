#ifndef NODE_OBJECT_H
#define NODE_OBJECT_H

#include <stdint.h>
#include <limits.h>

// NodeObject: Represents a node in the BFS traversal with parent tracking
// NOTE: Size is 12 bytes, but MRAM operations must be 8-byte aligned
// Use 16-byte reads/writes for this structure
struct NodeObject {
    uint32_t nodeId;     // Global node ID
    uint32_t parentId;   // Parent node ID in BFS tree
    uint32_t level;      // BFS level (distance from source)
};

// Size helpers for MRAM alignment
#define NODEOBJECT_SIZE sizeof(struct NodeObject)
#define NODEOBJECT_ALIGNED_SIZE 16  // Round up to 8-byte multiple

// Initialize a NodeObject with default unvisited values
static inline void initializeNodeObject(struct NodeObject* nodeObj, uint32_t id) {
    nodeObj->nodeId = id;
    nodeObj->parentId = UINT32_MAX; // No parent
    nodeObj->level = UINT32_MAX;    // Unvisited
}

// Update or replace: Updates target if new node has lower level
// This is the primary function for updating node information
static inline void updateOrReplaceNodeObject(struct NodeObject* targetNodeObj, 
                                             const struct NodeObject* newNodeObj) {
    if (targetNodeObj->nodeId == newNodeObj->nodeId && 
        newNodeObj->level < targetNodeObj->level) {
        targetNodeObj->parentId = newNodeObj->parentId;
        targetNodeObj->level = newNodeObj->level;
    }
}

// Check if a node has been visited
static inline int isNodeVisited(const struct NodeObject* nodeObj) {
    return nodeObj->level != UINT32_MAX;
}

// Check if a node is unvisited
static inline int isNodeUnvisited(const struct NodeObject* nodeObj) {
    return nodeObj->level == UINT32_MAX;
}

// Compare two NodeObjects (for sorting/searching)
static inline int compareNodeObjects(const struct NodeObject* a, 
                                     const struct NodeObject* b) {
    if (a->nodeId < b->nodeId) return -1;
    if (a->nodeId > b->nodeId) return 1;
    return 0;
}

// Traversal list: Dynamic array of NodeObjects discovered by a DPU
struct TraversalList {
    struct NodeObject* nodes;  // Array of discovered nodes
    uint32_t count;            // Current number of nodes
    uint32_t capacity;         // Allocated capacity
};

// Initialize a traversal list
static inline void initTraversalList(struct TraversalList* list, uint32_t initialCapacity) {
    list->nodes = NULL;
    list->count = 0;
    list->capacity = initialCapacity;
}

// Add a node to the traversal list (simplified - no dynamic reallocation for MRAM)
static inline void addToTraversalList(struct TraversalList* list, 
                                      const struct NodeObject* nodeObj) {
    if (list->count < list->capacity) {
        list->nodes[list->count] = *nodeObj;
        list->count++;
    }
}

#endif // NODE_OBJECT_H