#!/bin/bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build}"
RESULTS_DIR="${RESULTS_DIR:-results}"
mkdir -p "$RESULTS_DIR"

METHODS=("ilu0" "asm" "ras" "twolevel_ras")
STRONG_RANKS=(1 2 4 8 16)
WEAK_DOFS=10000

echo "=== Strong Scaling ==="
for method in "${METHODS[@]}"; do
    for np in "${STRONG_RANKS[@]}"; do
        echo "Running strong_scaling: method=$method ranks=$np"
        mpirun -np "$np" \
            "$BUILD_DIR/bench/strong_scaling" \
            --nx 100 --ny 100 \
            --method "$method" \
            --overlap 1 \
            --output "$RESULTS_DIR/strong_${method}.jsonl" \
            2>&1 | tail -3
        echo "---"
    done
done

echo "=== Weak Scaling ==="
for method in "ras" "twolevel_ras"; do
    for np in "${STRONG_RANKS[@]}"; do
        echo "Running weak_scaling: method=$method ranks=$np"
        mpirun -np "$np" \
            "$BUILD_DIR/bench/weak_scaling" \
            --dofs "$WEAK_DOFS" \
            --method "$method" \
            --overlap 1 \
            --output "$RESULTS_DIR/weak_${method}.jsonl" \
            2>&1 | tail -3
        echo "---"
    done
done

echo "=== SuiteSparse matrices ==="
for mtx in data/*.mtx; do
    name=$(basename "$mtx" .mtx)
    for method in "ras" "twolevel_ras"; do
        echo "Running $name with $method"
        mpirun -np 4 \
            "$BUILD_DIR/bench/strong_scaling" \
            --matrix "$mtx" \
            --method "$method" \
            --overlap 1 \
            --output "$RESULTS_DIR/suite_${name}_${method}.jsonl" \
            2>&1 | tail -3
    done
done

echo "All benchmarks complete."
