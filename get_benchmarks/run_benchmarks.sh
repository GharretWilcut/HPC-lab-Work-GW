#!/bin/bash

# =============================================================================
#  BFS Benchmark Runner
#  Runs BFS (host_code) and BFS-ALT3 (app_local) against all graphs in data/
#  Captures program-reported timing breakdowns + wall-clock
#  Output: bfs_bench_results/results_<timestamp>.csv
#          bfs_bench_results/summary_<timestamp>.csv
# =============================================================================

# --- Paths (relative to this script's location) ------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BFS_DIR="$SCRIPT_DIR/BFS-CPP"
BFSALT_DIR="$SCRIPT_DIR/BFS-ALT5"
DATA_DIR="$SCRIPT_DIR/data"
RESULTS_DIR="$SCRIPT_DIR/bfs_bench_results"

# --- Confirmed binaries and invocation ---------------------------------------
#   BFS:      bin/host_code -f <graph>   (verbosity 0 prints timing breakdown)
#   BFS-ALT3: bin/app_local <graph>      (bare positional argument)
BFS_BIN="$BFS_DIR/bin/host_code"
BFSALT_BIN="$BFSALT_DIR/bin/app_local"

# --- How many times to run each (impl, graph) pair --------------------------
RUNS=3

# =============================================================================

mkdir -p "$RESULTS_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTFILE="$RESULTS_DIR/results_${TIMESTAMP}.csv"
SUMMARY_FILE="$RESULTS_DIR/summary_${TIMESTAMP}.csv"
TMPOUT="/tmp/_bfs_bench_out_$$"
TMPTIME="/tmp/_bfs_bench_time_$$"

# CSV header — BFS provides 4 internal timing fields; ALT3 fields may be N/A
echo "impl,graph,graph_size_mb,run,exit_code,wall_time_ms,cpu_dpu_ms,dpu_kernel_ms,inter_dpu_ms,dpu_cpu_ms" > "$OUTFILE"

echo "========================================"
echo "  BFS Benchmark Runner"
echo "  Data dir   : $DATA_DIR"
echo "  Results    : $OUTFILE"
echo "  Runs/config: $RUNS"
echo "========================================"
echo ""

# ---- Build both implementations ---------------------------------------------
echo "[BUILD] Building BFS..."
make -C "$BFS_DIR" 2>&1
if [ $? -ne 0 ]; then
    echo "  ERROR: BFS build failed. Aborting."
    exit 1
fi

echo ""
echo "[BUILD] Building BFS-ALT3..."
make -C "$BFSALT_DIR" 2>&1
if [ $? -ne 0 ]; then
    echo "  ERROR: BFS-ALT3 build failed. Aborting."
    exit 1
fi
echo ""

# ---- parse_bfs_output -------------------------------------------------------
# Parses the BFS verbosity-0 line:
#   "CPU-DPU Time(ms): X    DPU Kernel Time (ms): X    Inter-DPU Time (ms): X    DPU-CPU Time (ms): X"
# Sets: CPU_DPU_MS, DPU_KERNEL_MS, INTER_DPU_MS, DPU_CPU_MS  (or "N/A")
parse_bfs_output() {
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

# ---- parse_alt3_output ------------------------------------------------------
# BFS-ALT3 output format TBD — currently captures same fields if present,
# falls back to N/A. Update the grep patterns once output format is confirmed.
parse_alt3_output() {
    local FILE="$1"
    # Try same format as BFS first; update patterns here if ALT3 differs
    CPU_DPU_MS=$(grep -oP 'CPU-DPU Time\(ms\):\s*\K[0-9.]+' "$FILE" | head -1)
    DPU_KERNEL_MS=$(grep -oP 'DPU Kernel Time \(ms\):\s*\K[0-9.]+' "$FILE" | head -1)
    INTER_DPU_MS=$(grep -oP 'Inter-DPU Time \(ms\):\s*\K[0-9.]+' "$FILE" | head -1)
    DPU_CPU_MS=$(grep -oP 'DPU-CPU Time \(ms\):\s*\K[0-9.]+' "$FILE" | head -1)
    CPU_DPU_MS="${CPU_DPU_MS:-N/A}"
    DPU_KERNEL_MS="${DPU_KERNEL_MS:-N/A}"
    INTER_DPU_MS="${INTER_DPU_MS:-N/A}"
    DPU_CPU_MS="${DPU_CPU_MS:-N/A}"
}

# ---- run_bfs ----------------------------------------------------------------
run_bfs() {
    local GRAPH="$1"
    GRAPH=$(realpath "$GRAPH")
    local RUN_IDX="$2"
    local GRAPH_NAME GRAPH_SIZE_MB WALL_SEC WALL_MS EXIT_CODE

    GRAPH_NAME=$(basename "$GRAPH")
    GRAPH_SIZE_MB=$(du -m "$GRAPH" 2>/dev/null | cut -f1)

    printf "  [BFS]      %-22s run %d/%d ... " "$GRAPH_NAME" "$RUN_IDX" "$RUNS"

    {
        TIMEFORMAT='%R'
        { time (cd "$BFS_DIR" && "$BFS_BIN" -v 0 -f "$GRAPH" 2>&1; echo $? > /tmp/_bfs_exit_$$) > "$TMPOUT"; } 2>"$TMPTIME"
        EXIT_CODE=$(cat /tmp/_bfs_exit_$$)
    }
    WALL_SEC=$(cat "$TMPTIME")
    WALL_MS=$(awk "BEGIN { printf \"%.2f\", $WALL_SEC * 1000 }")

    parse_bfs_output "$TMPOUT"

    printf "wall=%sms  dpu_kernel=%sms  exit=%d\n" "$WALL_MS" "$DPU_KERNEL_MS" "$EXIT_CODE"

    if [ "$EXIT_CODE" -ne 0 ]; then
        echo "    --- Program output (exit=$EXIT_CODE) ---"
        cat "$TMPOUT" | sed 's/^/    /'
        echo "    ---"
    fi

    echo "BFS,${GRAPH_NAME},${GRAPH_SIZE_MB},${RUN_IDX},${EXIT_CODE},${WALL_MS},${CPU_DPU_MS},${DPU_KERNEL_MS},${INTER_DPU_MS},${DPU_CPU_MS}" >> "$OUTFILE"
}

# ---- run_alt3 ---------------------------------------------------------------
run_alt3() {
    local GRAPH="$1"
    GRAPH=$(realpath "$GRAPH")
    local RUN_IDX="$2"
    local GRAPH_NAME GRAPH_SIZE_MB WALL_SEC WALL_MS EXIT_CODE

    GRAPH_NAME=$(basename "$GRAPH")
    GRAPH_SIZE_MB=$(du -m "$GRAPH" 2>/dev/null | cut -f1)

    printf "  [BFS-ALT3] %-22s run %d/%d ... " "$GRAPH_NAME" "$RUN_IDX" "$RUNS"

    {
        TIMEFORMAT='%R'
        { time (cd "$BFSALT_DIR" && "$BFSALT_BIN" -v 0 "$GRAPH" 2>&1; echo $? > /tmp/_bfs_exit_$$) > "$TMPOUT"; } 2>"$TMPTIME"
        EXIT_CODE=$(cat /tmp/_bfs_exit_$$)
    }
    WALL_SEC=$(cat "$TMPTIME")
    WALL_MS=$(awk "BEGIN { printf \"%.2f\", $WALL_SEC * 1000 }")

    parse_alt3_output "$TMPOUT"

    printf "wall=%sms  dpu_kernel=%sms  exit=%d\n" "$WALL_MS" "$DPU_KERNEL_MS" "$EXIT_CODE"

    if [ "$EXIT_CODE" -ne 0 ]; then
        echo "    --- Program output (exit=$EXIT_CODE) ---"
        cat "$TMPOUT" | sed 's/^/    /'
        echo "    ---"
    fi

    echo "BFS-ALT3,${GRAPH_NAME},${GRAPH_SIZE_MB},${RUN_IDX},${EXIT_CODE},${WALL_MS},${CPU_DPU_MS},${DPU_KERNEL_MS},${INTER_DPU_MS},${DPU_CPU_MS}" >> "$OUTFILE"
}

# ---- Main loop --------------------------------------------------------------
# Sort numerically by the number in Graph_N.txt
mapfile -t GRAPHS < <(
    for f in "$DATA_DIR"/Graph_*.txt; do
        [[ -f "$f" ]] || continue
        num="${f##*_}"
        num="${num%.txt}"
        printf '%d\t%s\n' "$num" "$f"
    done | sort -n | cut -f2
)
if [ ${#GRAPHS[@]} -eq 0 ]; then
    echo "ERROR: No Graph_*.txt files found in $DATA_DIR"
    exit 1
fi

TOTAL_GRAPHS=${#GRAPHS[@]}
GRAPH_IDX=0

for GRAPH in "${GRAPHS[@]}"; do
    GRAPH_IDX=$((GRAPH_IDX + 1))
    GRAPH_NAME=$(basename "$GRAPH")
    GRAPH_SIZE=$(du -sh "$GRAPH" 2>/dev/null | cut -f1)

    echo "--- Graph ${GRAPH_IDX}/${TOTAL_GRAPHS}: ${GRAPH_NAME} (${GRAPH_SIZE}) ---"

    for RUN in $(seq 1 "$RUNS"); do
        run_bfs   "$GRAPH" "$RUN"
        run_alt3  "$GRAPH" "$RUN"
    done
    echo ""
done

rm -f "$TMPOUT" "$TMPTIME"

# ---- Summary (avg/min/max per impl+graph, all timing columns) ---------------
echo "impl,graph,graph_size_mb,runs,avg_wall_ms,min_wall_ms,max_wall_ms,avg_cpu_dpu_ms,avg_dpu_kernel_ms,avg_inter_dpu_ms,avg_dpu_cpu_ms" > "$SUMMARY_FILE"

awk -F',' '
NR==1 { next }
$5 != 0 { next }   # skip failed runs from averages
{
    key = $1 SUBSEP $2
    size[key]   = $3
    wall[key]  += $6+0;  wcount[key]++
    if (!(key in wmin) || $6+0 < wmin[key]) wmin[key] = $6+0
    if (!(key in wmax) || $6+0 > wmax[key]) wmax[key] = $6+0
    if ($7  != "N/A") { cdpu[key]  += $7+0;  cdpu_n[key]++ }
    if ($8  != "N/A") { kern[key]  += $8+0;  kern_n[key]++ }
    if ($9  != "N/A") { inter[key] += $9+0;  inter_n[key]++ }
    if ($10 != "N/A") { dcpu[key]  += $10+0; dcpu_n[key]++ }
}
END {
    for (key in wall) {
        split(key, p, SUBSEP)
        avg_w  = wcount[key]  ? wall[key]  / wcount[key]  : "N/A"
        avg_cd = cdpu_n[key]  ? cdpu[key]  / cdpu_n[key]  : "N/A"
        avg_k  = kern_n[key]  ? kern[key]  / kern_n[key]  : "N/A"
        avg_i  = inter_n[key] ? inter[key] / inter_n[key] : "N/A"
        avg_dc = dcpu_n[key]  ? dcpu[key]  / dcpu_n[key]  : "N/A"
        printf "%s,%s,%s,%d,%.2f,%.2f,%.2f,%s,%s,%s,%s\n",
            p[1], p[2], size[key], wcount[key],
            avg_w, wmin[key], wmax[key],
            (avg_cd == "N/A" ? "N/A" : sprintf("%.2f", avg_cd)),
            (avg_k  == "N/A" ? "N/A" : sprintf("%.2f", avg_k)),
            (avg_i  == "N/A" ? "N/A" : sprintf("%.2f", avg_i)),
            (avg_dc == "N/A" ? "N/A" : sprintf("%.2f", avg_dc))
    }
}
' "$OUTFILE" | sort -t',' -k2,2V -k1,1 | tee -a "$SUMMARY_FILE" | \
awk -F',' 'BEGIN {
    printf "\n========================================\n"
    printf "  Summary (successful runs only)\n"
    printf "========================================\n"
    printf "  %-12s %-18s %6s %10s %10s %12s\n", "impl","graph","size","avg_wall","dpu_kern","cpu_dpu"
    printf "  %s\n", "----------------------------------------------------------------------"
}
{
    printf "  %-12s %-18s %5sMB %9sms %9sms %11sms\n", $1,$2,$3,$5,$9,$8
}'

echo ""
echo "========================================"
echo "  Raw results : $OUTFILE"
echo "  Summary     : $SUMMARY_FILE"
echo "========================================"