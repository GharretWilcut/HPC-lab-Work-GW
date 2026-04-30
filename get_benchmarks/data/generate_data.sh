#!/bin/bash

# Graph dataset generator - 20 datasets from ~121MB to ~3GB
# Base: 200000 nodes, 10000000 edges = ~121MB
# Target: ~3GB (~25x scale)
# Scale factor per step: 25^(1/19) ≈ 1.197 (geometric progression)

SCRIPT="$HOME/HPC-lab-Work-GW/UtilsFolder/create_graph.py"
UNREACHABLE=0.0
OUTPUT_DIR="."

# Each entry: "nodes edges"
# Derived by multiplying base (200k nodes, 10M edges) by scale^i
# Scale factor ~1.197 per step, applied to both nodes and edges
DATASETS=(
    "200000 10000000"     #1
    "240000 12000000"     #2
    "288000 14400000"     #3
    "345600 17280000"     #4
    "414720 20736000"     #5
    "497664 24883200"     #6
    "597197 29859840"     #7
    "716636 35831808"     #8
    "859963 42998170"     #9
    "1031956 51597804"    #10
    "1238347 61917365"    #11
    "1486016 74300838"    #12
    "1783219 89161006"    #13
    "2139863 106993207"   #14
    "2567836 128391848"   #15
    "3081403 154070218"   #16
    "3697684 184884262"   #17
    "4437221 221861114"   #18
    "5324665 266233337"   #19
    "6389598 319480004"   #20
    "7667518 383376005"   #21
    "8720000 453000000"   #22
    "9450000 507000000"   #23
    "11340000 609000000" #24
    "13600000 730000000" #25
)

echo "========================================"
echo "  Graph Dataset Generator"
echo "  Generating 25 datasets (~121MB→~3GB)"
echo "========================================"
echo ""

TOTAL=${#DATASETS[@]}

for i in "${!DATASETS[@]}"; do
    IDX=$((i + 1))
    read -r NODES EDGES <<< "${DATASETS[$i]}"
    OUTFILE="${OUTPUT_DIR}/Graph_${IDX}.txt"

    echo "[${IDX}/${TOTAL}] Graph_${IDX}: nodes=${NODES}, edges=${EDGES} → ${OUTFILE}"

    python3 "$SCRIPT" \
        --nodes "$NODES" \
        --edges "$EDGES" \
        --unreachable "$UNREACHABLE" \
        --out "$OUTFILE"

    STATUS=$?
    if [ $STATUS -ne 0 ]; then
        echo "  ERROR: Graph_${IDX} failed with exit code $STATUS"
    else
        SIZE=$(du -sh "$OUTFILE" 2>/dev/null | cut -f1)
        echo "  Done. File size: ${SIZE}"
    fi
    echo ""
done

echo "========================================"
echo "  All graphs generated."
echo "========================================"