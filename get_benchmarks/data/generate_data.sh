#!/bin/bash

# Graph dataset generator - 20 datasets from ~121MB to ~3GB
# Base: 200000 nodes, 10000000 edges = ~121MB
# Target: ~3GB (~25x scale)
# Scale factor per step: 25^(1/19) ≈ 1.197 (geometric progression)

SCRIPT="$HOME/HPC-lab-Work-GW/UtilsFolder/create_graph.py"
UNREACHABLE=0.1
OUTPUT_DIR="."

# Each entry: "nodes edges"
# Derived by multiplying base (200k nodes, 10M edges) by scale^i
# Scale factor ~1.197 per step, applied to both nodes and edges
DATASETS=(
    "200000 10000000"       # Graph_1  ~121MB (base)
    "239000 11970000"       # Graph_2  ~145MB
    "286000 14320000"       # Graph_3  ~173MB
    "342000 17130000"       # Graph_4  ~207MB
    "409000 20490000"       # Graph_5  ~248MB
    "490000 24510000"       # Graph_6  ~297MB
    "586000 29320000"       # Graph_7  ~355MB
    "701000 35080000"       # Graph_8  ~425MB
    "839000 41970000"       # Graph_9  ~508MB
    "1004000 50210000"      # Graph_10 ~608MB
    "1201000 60080000"      # Graph_11 ~728MB
    "1437000 71880000"      # Graph_12 ~871MB
    "1719000 86000000"      # Graph_13 ~1.04GB
    "2057000 102870000"     # Graph_14 ~1.25GB
    "2461000 123080000"     # Graph_15 ~1.49GB
    "2945000 147280000"     # Graph_16 ~1.79GB
    "3523000 176190000"     # Graph_17 ~2.14GB
    "4215000 210790000"     # Graph_18 ~2.56GB
    "5044000 252210000"     # Graph_19 ~3.06GB  ← ~3GB
    "5044000 252210000"     # Graph_20 ~3.06GB  (duplicate cap if needed)
)

echo "========================================"
echo "  Graph Dataset Generator"
echo "  Generating 20 datasets (~121MB→~3GB)"
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