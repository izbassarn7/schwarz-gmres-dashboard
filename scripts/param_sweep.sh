#!/bin/bash
set -euo pipefail

# Parameter sweep script for Schwarz preconditioner study.
# Runs solver over combinations of overlap and nparts parameters.

BINARY="${BINARY:-./build/schwarz_solve}"
DATA_DIR="${DATA_DIR:-./data}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/param_sweep}"
PRECOND="${PRECOND:-ras}"
RESTART="${RESTART:-30}"
MAXITER="${MAXITER:-500}"
TOL="${TOL:-1e-10}"

# Parse command-line options
while [[ $# -gt 0 ]]; do
    case "$1" in
        --binary)
            BINARY="$2"
            shift 2
            ;;
        --data)
            DATA_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --precond)
            PRECOND="$2"
            shift 2
            ;;
        --restart)
            RESTART="$2"
            shift 2
            ;;
        --maxiter)
            MAXITER="$2"
            shift 2
            ;;
        --tol)
            TOL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

mkdir -p "$OUTPUT_DIR"

# Collect all .mtx files
declare -a MATRICES
mapfile -t MATRICES < <(find "$DATA_DIR" -maxdepth 1 -name "*.mtx" -type f | sort)

if [[ ${#MATRICES[@]} -eq 0 ]]; then
    echo "[WARN] No .mtx files found in $DATA_DIR"
    exit 0
fi

echo "Found ${#MATRICES[@]} matrix file(s) in $DATA_DIR"

# Parameter ranges
OVERLAPS=(0 1 2 3)
NPARTS=(2 4 8 16)

# Loop over all combinations
for overlap in "${OVERLAPS[@]}"; do
    for nparts in "${NPARTS[@]}"; do
        for mtx_file in "${MATRICES[@]}"; do
            matrix_name=$(basename "$mtx_file" .mtx)
            out_file="$OUTPUT_DIR/${matrix_name}_pc${PRECOND}_ov${overlap}_np${nparts}.jsonl"

            if [[ -f "$out_file" ]]; then
                echo "[SKIP] $out_file already exists"
                continue
            fi

            echo "[RUN] overlap=$overlap nparts=$nparts matrix=$matrix_name"
            "$BINARY" \
                --matrix "$mtx_file" \
                --precond "$PRECOND" \
                --overlap "$overlap" \
                --nparts "$nparts" \
                --restart "$RESTART" \
                --tol "$TOL" \
                --maxiter "$MAXITER" \
                --output "$out_file" \
                || echo "[WARN] solver returned non-zero for $matrix_name ov=$overlap np=$nparts"
        done
    done
done

# Merge all results
cat "$OUTPUT_DIR"/*.jsonl > "$OUTPUT_DIR/all_results.jsonl" 2>/dev/null || true

echo "Done. Results in $OUTPUT_DIR/all_results.jsonl"
