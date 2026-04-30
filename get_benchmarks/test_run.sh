#!/bin/bash

# ===============================
# BFS Benchmark Runner (Single Graph, Terminal Output Only)
# Usage: ./run_benchmarks <graph_file>
# ===============================

# --- Check input ---
if [ $# -ne 1 ]; then
    echo "Usage: $0 <graph_file>"
    exit 1
fi

GRAPH=$(realpath "$1")

if [ ! -f "$GRAPH" ]; then
    echo "Error: File not found -> $GRAPH"
    exit 1
fi

# --- Paths ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BFS_DIR="$SCRIPT_DIR/BFS-CPP"
BFSALT_DIR="$SCRIPT_DIR/BFS-ALT5"

BFS_BIN="$BFS_DIR/bin/host_code"
BFSALT_BIN="$BFSALT_DIR/bin/app_local"

TMPOUT="/tmp/_bfs_out_$$"
TMPTIME="/tmp/_bfs_time_$$"

# --- Build ---
echo "[BUILD] BFS..."
make -C "$BFS_DIR" || exit 1

echo "[BUILD] BFS-ALT5..."
make -C "$BFSALT_DIR" || exit 1
echo ""

# --- Parse functions ---
parse_output() {
    local FILE="$1"
    CPU_DPU_MS=$(grep -oP 'CPU-DPU Time\(ms\):\s*\K[0-9.]+' "$FILE" | head -1)
    DPU_KERNEL_MS=$(grep -oP 'DPU Kernel Time \(ms\):\s*\K[0-9.]+' "$FILE" | head -1)
    INTER_DPU_MS=$(grep -oP 'Inter-DPU Time \(ms\):\s*\K[0-9.]+' "$FILE" | head -1)
    DPU_CPU_MS=$(grep -oP 'DPU-CPU Time \(ms\):\s*\K[0-9.]+' "$FILE" | head -1)

    CPU_DPU_MS="${CPU_DPU_MS:-N/A}"
    DPU_KERNEL_MS="${DPU_KERNEL_MS:-N/A}"
    INTER_DPU_MS="${INTER_DPU_MS:-N/A}"
    DPU_CPU_MS="${DPU_CPU_MS:-N/A}"
}

run_and_print() {
    local NAME="$1"
    local CMD="$2"

    echo "[RUN] $NAME"

    {
        TIMEFORMAT='%R'
        { time (eval "$CMD" 2>&1; echo $? > /tmp/_bfs_exit_$$) > "$TMPOUT"; } 2>"$TMPTIME"
        EXIT_CODE=$(cat /tmp/_bfs_exit_$$)
    }

    WALL_SEC=$(cat "$TMPTIME")
    WALL_MS=$(awk "BEGIN { printf \"%.2f\", $WALL_SEC * 1000 }")

    parse_output "$TMPOUT"

    echo "----------------------------------"
    echo "Wall time      : ${WALL_MS} ms"
    echo "CPU-DPU        : ${CPU_DPU_MS} ms"
    echo "DPU Kernel     : ${DPU_KERNEL_MS} ms"
    echo "Inter-DPU      : ${INTER_DPU_MS} ms"
    echo "DPU-CPU        : ${DPU_CPU_MS} ms"
    echo "Exit code      : ${EXIT_CODE}"
    echo "----------------------------------"

    if [ "$EXIT_CODE" -ne 0 ]; then
        echo "Program output:"
        cat "$TMPOUT" | sed 's/^/  /'
    fi

    echo ""
}

# --- Run both ---
run_and_print "BFS"      "cd $BFS_DIR && $BFS_BIN -v 0 -f $GRAPH"
run_and_print "BFS-ALT5" "cd $BFSALT_DIR && $BFSALT_BIN -v 0 $GRAPH"

# --- Cleanup ---
rm -f "$TMPOUT" "$TMPTIME"