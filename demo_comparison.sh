#!/usr/bin/env bash
# ============================================================
#  GPU vs CPU comparison demo  —  schwarz-gmres
#  Run from:  schwarz-gmres/build/
#  Usage:     bash ../demo_comparison.sh [--no-gpu]
# ============================================================

BIN="./schwarz_solve"
SEP="━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

NO_GPU=false
for arg in "$@"; do [[ "$arg" == "--no-gpu" ]] && NO_GPU=true; done

if [[ ! -f "$BIN" ]]; then
    echo "ERROR: $BIN not found. Build first:"
    echo "  mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j\$(nproc)"
    exit 1
fi

echo ""
echo "$SEP"
echo "  SCHWARZ-GMRES  —  CPU vs GPU Demonstration"
echo "  Poisson 2D · 100×100 = 10,000 unknowns"
echo "$SEP"

mkdir -p results

# ────────────────────────────────────────────────────────────
# 1. BASELINE  (CPU · no preconditioner)
# ────────────────────────────────────────────────────────────
echo ""
echo "► [1/6] CPU  |  GMRES(30)  |  no preconditioner  (baseline)"
echo "$SEP"
$BIN --poisson 100 --precond none --restart 30 --tol 1e-10 --maxiter 500 \
     --output results/demo.jsonl
echo ""

# ────────────────────────────────────────────────────────────
# 2. CPU · ILU0
# ────────────────────────────────────────────────────────────
echo "► [2/6] CPU  |  GMRES(30)  |  ILU0"
echo "$SEP"
$BIN --poisson 100 --precond ilu0 --restart 30 --tol 1e-10 \
     --output results/demo.jsonl
echo ""

# ────────────────────────────────────────────────────────────
# 3. CPU · ILUT
# ────────────────────────────────────────────────────────────
echo "► [3/6] CPU  |  GMRES(30)  |  ILUT  (τ=1e-4, ξ=10)"
echo "$SEP"
$BIN --poisson 100 --precond ilut --drop-tol 1e-4 --fill-factor 10 \
     --restart 30 --tol 1e-10 \
     --output results/demo.jsonl
echo ""

# ────────────────────────────────────────────────────────────
# 4. CPU · BiCGSTAB + ILUT  ← CHAMPION
# ────────────────────────────────────────────────────────────
echo "► [4/6] CPU  |  BiCGSTAB  |  ILUT  ⭐ CHAMPION"
echo "$SEP"
$BIN --poisson 100 --precond ilut --drop-tol 1e-4 --fill-factor 10 \
     --bicgstab --tol 1e-10 \
     --output results/demo.jsonl
echo ""

# ────────────────────────────────────────────────────────────
# 5. CPU · Two-Level Schwarz (4 subdomains, δ=1)
# ────────────────────────────────────────────────────────────
echo "► [5/6] CPU  |  GMRES(30)  |  Two-Level RAS  (4 subdomains, overlap=1)"
echo "$SEP"
$BIN --poisson 100 --precond twolevel_ras --nparts 4 --overlap 1 \
     --restart 30 --tol 1e-10 \
     --output results/demo.jsonl
echo ""

# ────────────────────────────────────────────────────────────
# 6. GPU · GMRES + RAS  (requires CUDA build)
# ────────────────────────────────────────────────────────────
if [[ "$NO_GPU" == "false" ]]; then
    echo "► [6/6] GPU  |  GMRES(30)  |  RAS  (RTX 4050 · 192 GB/s)"
    echo "$SEP"
    $BIN --poisson 100 --precond ras --nparts 4 --overlap 1 \
         --restart 30 --tol 1e-10 --gpu \
         --output results/demo.jsonl
    echo ""
else
    echo "► [6/6] GPU  — skipped (--no-gpu)"
fi

# ────────────────────────────────────────────────────────────
# 7. MPI · 4 ranks · distributed GMRES
# ────────────────────────────────────────────────────────────
echo ""
echo "$SEP"
echo "► [BONUS] MPI 4 ranks  |  dist_GMRES(30)  |  Two-Level RAS"
echo "$SEP"
mpirun -np 4 $BIN --poisson 100 --precond twolevel_ras \
       --nparts 4 --overlap 1 --restart 30 --tol 1e-10 \
       --output results/demo_mpi.jsonl
echo ""

# ────────────────────────────────────────────────────────────
# 8. SUMMARY TABLE
# ────────────────────────────────────────────────────────────
echo "$SEP"
echo "  SUMMARY"
echo "$SEP"
echo ""
echo "  Run   | Backend | Solver        | Precond      | Expected iters | Expected time"
echo "  ------|---------|---------------|--------------|----------------|---------------"
echo "  1     | CPU     | GMRES(30)     | none         | ~500+ (slow)   | ~0.20 s"
echo "  2     | CPU     | GMRES(30)     | ILU0         | ~141           | ~0.07 s"
echo "  3     | CPU     | GMRES(30)     | ILUT         | ~93            | ~0.04 s"
echo "  4 ⭐  | CPU     | BiCGSTAB      | ILUT         | ~43            | ~0.03 s"
echo "  5     | CPU     | GMRES(30)     | Two-Level    | ~134           | ~0.45 s"
echo "  6     | GPU     | GMRES(30)     | RAS          | ~194           | ~0.99 s"
echo "  MPI   | 4 ranks | dist_GMRES    | Two-Level    | ~189           | ~0.60 s"
echo ""
echo "  KEY TAKEAWAY:"
echo "    CPU BiCGSTAB+ILUT is fastest (43 iters, 0.03s)"
echo "    GPU RAS is SLOWER on this small matrix — memory-bound, AI=0.08 FLOP/byte"
echo "    GPU wins only at n > 100,000 (memory bandwidth fully utilized)"
echo "$SEP"
echo ""
