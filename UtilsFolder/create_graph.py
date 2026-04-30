#!/usr/bin/env python3
"""
Synthetic graph data generator.
Produces large graphs in the format:
    <num_nodes> <num_nodes> <num_edges>   ← header line
    <src>\t<dst>
    <src>\t<dst>
    ...

Usage:
    python3 create_graph.py [options]

Examples:
    python3 create_graph.py --nodes 1000000 --edges 10000000 --unreachable 0.1 --out graph.txt
    python3 create_graph.py --nodes 500000  --edges 50000000 --type scale_free --out big.txt
    python3 create_graph.py --nodes 200000  --edges 5000000  --type grid --out grid.txt
"""

import argparse
import random
import sys
import time
from collections import defaultdict

def gen_random(num_nodes, num_edges, unreachable_frac):
    
    reachable = max(1, int(num_nodes * (1.0 - unreachable_frac)))

    # Shuffled spine: guarantees full connectivity among reachable nodes
    edges_used = 0
    nodes = list(range(reachable))
    random.shuffle(nodes)

    for i in range(len(nodes) - 1):
        yield nodes[i], nodes[i + 1]
        edges_used += 1
        if edges_used >= num_edges:
            return

    # One random outgoing edge per node so every node has degree >= 2
    for node in range(reachable):
        dst = random.randrange(reachable)
        yield node, dst
        edges_used += 1
        if edges_used >= num_edges:
            return

    # Fill remaining budget
    for _ in range(num_edges - edges_used):
        yield random.randrange(reachable), random.randrange(reachable)


def gen_scale_free(num_nodes, num_edges, unreachable_frac):
   
    reachable = max(2, int(num_nodes * (1.0 - unreachable_frac)))
    m = max(1, num_edges // reachable)          # edges to attach per new node
    degree = [1] * reachable                    # seed degrees
    degree_sum = reachable

    edges_written = 0
    for new_node in range(1, reachable):
        # Pick m targets by preferential attachment (roulette wheel sampling)
        targets = set()
        attempts = 0
        while len(targets) < min(m, new_node) and attempts < m * 10:
            r = random.randrange(degree_sum)
            cumulative = 0
            for node, d in enumerate(degree[:new_node]):
                cumulative += d
                if cumulative > r:
                    targets.add(node)
                    break
            attempts += 1

        for t in targets:
            yield new_node, t
            degree[new_node] += 1
            degree[t] += 1
            degree_sum += 2
            edges_written += 1
            if edges_written >= num_edges:
                return

    # If we still need more edges after attachment phase, add random ones
    while edges_written < num_edges:
        yield random.randrange(reachable), random.randrange(reachable)
        edges_written += 1


def gen_grid(num_nodes, num_edges, unreachable_frac):
    import math
    reachable = max(4, int(num_nodes * (1.0 - unreachable_frac)))
    cols = max(2, int(math.sqrt(reachable)))
    rows = (reachable + cols - 1) // cols

    edges_written = 0
    for r in range(rows):
        for c in range(cols):
            node = r * cols + c
            if node >= reachable:
                break
            # right neighbour
            if c + 1 < cols and node + 1 < reachable:
                yield node, node + 1
                edges_written += 1
                if edges_written >= num_edges:
                    return
            # down neighbour
            if r + 1 < rows and node + cols < reachable:
                yield node, node + cols
                edges_written += 1
                if edges_written >= num_edges:
                    return

    # pad with random edges if grid exhausted before target
    while edges_written < num_edges:
        yield random.randrange(reachable), random.randrange(reachable)
        edges_written += 1


def gen_clustered(num_nodes, num_edges, unreachable_frac, num_clusters=None):
    
    reachable = max(2, int(num_nodes * (1.0 - unreachable_frac)))
    if num_clusters is None:
        num_clusters = max(2, reachable // 1000)

    cluster_size = reachable // num_clusters
    intra_prob = 0.7   # 70% of edges stay within a cluster

    edges_written = 0
    while edges_written < num_edges:
        if random.random() < intra_prob:
            # intra-cluster edge
            c = random.randrange(num_clusters)
            base = c * cluster_size
            end  = min(base + cluster_size, reachable)
            if end - base < 2:
                continue
            src = random.randrange(base, end)
            dst = random.randrange(base, end)
        else:
            # inter-cluster edge
            c1 = random.randrange(num_clusters)
            c2 = random.randrange(num_clusters)
            if c1 == c2:
                continue
            base1 = c1 * cluster_size
            base2 = c2 * cluster_size
            src = random.randrange(base1, min(base1 + cluster_size, reachable))
            dst = random.randrange(base2, min(base2 + cluster_size, reachable))

        yield src, dst
        edges_written += 1

def write_graph(out_file, num_nodes, num_edges, gen, dedup, verbose):
    t0 = time.time()

    # --- Phase 1: collect all edges into a per-source bucket ---
    if verbose:
        print("  Phase 1/2: collecting edges...", flush=True)

    seen = set() if dedup else None
    # Use a list-of-lists bucketed by source node for O(1) append
    buckets = defaultdict(list)
    collected = 0

    for src, dst in gen:
        if dedup:
            key = (src, dst)
            if key in seen:
                continue
            seen.add(key)

        buckets[src].append(dst)
        collected += 1

        if verbose and collected % 1_000_000 == 0:
            elapsed = time.time() - t0
            rate = collected / elapsed / 1e6
            print(f"    {collected:,} edges collected  ({rate:.1f}M/sec)", flush=True)

    if dedup and verbose and collected != num_edges:
        print(f"  (dedup removed {num_edges - collected:,} duplicate edges)")

    # --- Phase 2: write header then edges in ascending source-node order ---
    if verbose:
        print("  Phase 2/2: writing sorted output...", flush=True)

    actual_edges = 0
    with open(out_file, 'w') as f:
        # Header line: num_nodes num_nodes num_edges
        f.write(f"{num_nodes} {num_nodes} {collected}\n")

        for src in range(num_nodes):
            if src not in buckets:
                continue
            for dst in buckets[src]:
                f.write(f"{src}\t{dst}\n")
                actual_edges += 1

                if verbose and actual_edges % 1_000_000 == 0:
                    elapsed = time.time() - t0
                    rate = actual_edges / elapsed / 1e6
                    print(f"    {actual_edges:,} edges written  ({rate:.1f}M/sec)", flush=True)

    elapsed = time.time() - t0
    return actual_edges, elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic graph data for BFS / graph benchmarks."
    )
    parser.add_argument("--nodes",       type=int,   default=1_000_000,
                        help="Total number of nodes (default: 1,000,000)")
    parser.add_argument("--edges",       type=int,   default=10_000_000,
                        help="Target number of edges (default: 10,000,000)")
    parser.add_argument("--type",        type=str,   default="random",
                        choices=["random", "scale_free", "grid", "clustered"],
                        help="Graph topology (default: random)")
    parser.add_argument("--unreachable", type=float, default=0.05,
                        help="Fraction of nodes that are unreachable islands "
                             "(default: 0.05 = 5%%)")
    parser.add_argument("--seed",        type=int,   default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--out",         type=str,   default="graph.txt",
                        help="Output filename (default: graph.txt)")
    parser.add_argument("--dedup",       action="store_true",
                        help="Remove duplicate edges (slower, uses more RAM)")
    parser.add_argument("--quiet",       action="store_true",
                        help="Suppress progress output")

    args = parser.parse_args()

    random.seed(args.seed)
    verbose = not args.quiet

    if verbose:
        print(f"Generating graph:")
        print(f"  nodes       : {args.nodes:,}")
        print(f"  edges       : {args.edges:,}")
        print(f"  type        : {args.type}")
        print(f"  unreachable : {args.unreachable*100:.1f}% of nodes")
        print(f"  output      : {args.out}")
        print(f"  dedup       : {args.dedup}")
        print()

    # Pick generator
    if args.type == "random":
        gen = gen_random(args.nodes, args.edges, args.unreachable)
    elif args.type == "scale_free":
        gen = gen_scale_free(args.nodes, args.edges, args.unreachable)
    elif args.type == "grid":
        gen = gen_grid(args.nodes, args.edges, args.unreachable)
    elif args.type == "clustered":
        gen = gen_clustered(args.nodes, args.edges, args.unreachable)

    actual, elapsed = write_graph(
        args.out, args.nodes, args.edges, gen, args.dedup, verbose
    )

    if verbose:
        size_mb = __import__('os').path.getsize(args.out) / 1e6
        print(f"\nDone.")
        print(f"  edges written : {actual:,}")
        print(f"  time          : {elapsed:.1f}s")
        print(f"  file size     : {size_mb:.1f} MB")
        print(f"  output        : {args.out}")


if __name__ == "__main__":
    main()