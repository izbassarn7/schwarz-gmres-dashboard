#!/usr/bin/env bash
# ============================================================
#  GPU vs CPU — честное сравнение на больших матрицах
#  Показывает: при каком n GPU начинает выигрывать
#  Run from:  schwarz-gmres/build/
# ============================================================

BIN="./schwarz_solve"
SEP="━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

[[ ! -f "$BIN" ]] && echo "Build first. See demo_comparison.sh" && exit 1

mkdir -p results

echo ""
echo "$SEP"
echo "  GPU vs CPU  —  SpMV scaling by matrix size"
echo "  Same solver: GMRES(30) + RAS (4 subdomains, overlap=1)"
echo "$SEP"
echo ""

for N in 50 100 150 200 300; do
    n=$((N * N))
    echo "─── Poisson ${N}×${N}  (n = ${n} unknowns) ───"

    echo -n "  CPU: "
    $BIN --poisson $N --precond ras --nparts 4 --overlap 1 \
         --restart 30 --tol 1e-10 \
         --output results/cpu_scaling.jsonl 2>/dev/null \
    | grep -E "Iterations:|Solve time:"

    echo -n "  GPU: "
    $BIN --poisson $N --precond ras --nparts 4 --overlap 1 \
         --restart 30 --tol 1e-10 --gpu \
         --output results/gpu_scaling.jsonl 2>/dev/null \
    | grep -E "Iterations:|Solve time:"

    echo ""
done

echo "$SEP"
echo "  EXPECTED PATTERN:"
echo ""
echo "    n=2,500  (50²)   CPU: ~0.005s   GPU: ~0.08s   → CPU 16× faster"
echo "    n=10,000 (100²)  CPU: ~0.07s    GPU: ~0.99s   → CPU 14× faster"
echo "    n=22,500 (150²)  CPU: ~0.22s    GPU: ~0.60s   → CPU  3× faster"
echo "    n=40,000 (200²)  CPU: ~0.45s    GPU: ~0.55s   → ~equal"
echo "    n=90,000 (300²)  CPU: ~1.50s    GPU: ~0.90s   → GPU 1.6× faster"
echo ""
echo "  WHY: SpMV AI = 0.08 FLOP/byte (memory-bound)"
echo "       GPU wins ONLY when matrix doesn't fit in CPU L3 cache"
echo "       Cross-over point: n ≈ 40,000  (matrix ~25 MB)"
echo "$SEP"
echo ""
