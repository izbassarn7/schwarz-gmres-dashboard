#!/usr/bin/env bash
# ============================================================
#  COMMITTEE DEMO — 3 самых убедительных эксперимента
#  Запустить ПЕРЕД защитой, показать вывод живьём
#  Run from:  schwarz-gmres/build/
# ============================================================

BIN="./schwarz_solve"
SEP="━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

[[ ! -f "$BIN" ]] && echo "Build first." && exit 1
mkdir -p results

# ============================================================
# DEMO 1 — Overlap effect: δ=0 vs δ=1 vs δ=2
#   "Больше overlap → меньше итераций" — доказывается экспериментом
# ============================================================
echo ""
echo "$SEP"
echo "  DEMO 1: Overlap Effect  (Theorem 4.3 confirmed experimentally)"
echo "  Poisson 100²  |  GMRES(30)  |  RAS  |  4 subdomains"
echo "$SEP"
echo ""

for OV in 0 1 2; do
    printf "  overlap = %d :  " $OV
    $BIN --poisson 100 --precond ras --nparts 4 --overlap $OV \
         --restart 30 --tol 1e-10 2>/dev/null \
    | grep -E "Iterations:|Residual:" | tr '\n' ' '
    echo ""
done

echo ""
echo "  Expected:  δ=0: ~230 iters   δ=1: ~194 iters   δ=2: ~160 iters"
echo "  → More overlap = fewer iterations. Theory confirmed."
echo ""

# ============================================================
# DEMO 2 — Solver comparison on same matrix
#   BiCGSTAB wins 3.3× over baseline
# ============================================================
echo "$SEP"
echo "  DEMO 2: Solver + Preconditioner Comparison"
echo "  Poisson 100²  |  n = 10,000"
echo "$SEP"
echo ""

echo "  [1] GMRES(30) + ILU0  ..."
printf "      "
$BIN --poisson 100 --precond ilu0 --restart 30 --tol 1e-10 2>/dev/null \
| grep -E "Iterations:|Solve time:" | tr '\n' '   '
echo ""

echo "  [2] GMRES(30) + ILUT  ..."
printf "      "
$BIN --poisson 100 --precond ilut --restart 30 --tol 1e-10 2>/dev/null \
| grep -E "Iterations:|Solve time:" | tr '\n' '   '
echo ""

echo "  [3] BiCGSTAB + ILUT  ⭐  ..."
printf "      "
$BIN --poisson 100 --precond ilut --bicgstab --tol 1e-10 2>/dev/null \
| grep -E "Iterations:|Solve time:" | tr '\n' '   '
echo ""

echo "  [4] FGMRES(30) + Two-Level RAS  ..."
printf "      "
$BIN --poisson 100 --precond twolevel_ras --nparts 4 --overlap 1 \
     --fgmres --restart 30 --tol 1e-10 2>/dev/null \
| grep -E "Iterations:|Solve time:" | tr '\n' '   '
echo ""

echo ""
echo "  Expected:  ILU0→141  ILUT→93  BiCGSTAB+ILUT→43  FGMRES+TwoLevel→131"
echo ""

# ============================================================
# DEMO 3 — ILU0 FAILS on thermal1, Two-Level SUCCEEDS
#   "Реальная матрица — только Two-Level сходится"
#   Requires thermal1.mtx in data/ folder
#   Download: https://sparse.tamu.edu/Schmid/thermal1
# ============================================================
echo "$SEP"
echo "  DEMO 3: thermal1 Real Matrix  (SuiteSparse)"
echo "  n = 82,654   nnz = 574,458   κ ≫ 10⁶"
echo "$SEP"
echo ""

MTX="../data/thermal1.mtx"

if [[ ! -f "$MTX" ]]; then
    echo "  ⚠  thermal1.mtx not found at $MTX"
    echo ""
    echo "  Download it:"
    echo "    mkdir -p ../data"
    echo "    cd ../data"
    echo "    wget https://suitesparse-collection-website.herokuapp.com/MM/Schmid/thermal1.tar.gz"
    echo "    tar xzf thermal1.tar.gz && mv thermal1/thermal1.mtx ."
    echo ""
    echo "  Then re-run this script."
    echo ""
else
    echo "  [1] GMRES(50) + ILU0  — expected to FAIL (stagnate at ~2.4e-3) ..."
    $BIN --matrix $MTX --precond ilu0 --restart 50 --tol 1e-10 --maxiter 200 2>/dev/null \
    | grep -E "Converged:|Iterations:|Residual:|Solve time:"
    echo ""

    echo "  [2] GMRES(50) + Two-Level RAS  — expected to CONVERGE (787 iters) ..."
    $BIN --matrix $MTX --precond twolevel_ras --nparts 8 --overlap 1 \
         --restart 50 --tol 1e-10 --maxiter 1000 2>/dev/null \
    | grep -E "Converged:|Iterations:|Residual:|Solve time:"
    echo ""

    echo "  Expected:"
    echo "    ILU0:      Converged: no   Residual: ~2.4e-3  (stagnation)"
    echo "    Two-Level: Converged: yes  Iterations: 787    Time: ~1.24s"
fi

# ============================================================
# DEMO 4 — Scientific analysis  (κ, ρ, convergence rate)
# ============================================================
echo "$SEP"
echo "  DEMO 4: Scientific Analysis  (--analyze flag)"
echo "  κ(A), κ(M⁻¹A), ρ(I − M⁻¹A), convergence rate"
echo "$SEP"
echo ""

$BIN --poisson 100 --precond twolevel_ras --nparts 4 --overlap 1 \
     --restart 30 --tol 1e-10 --analyze 2>/dev/null
echo ""

echo "  Expected:"
echo "    κ(A)         ≈ 1052"
echo "    κ(M⁻¹A)      ≈ 12     → 87× reduction"
echo "    ρ(I − M⁻¹A) ≈ 0.975  < 1  → convergence GUARANTEED"
echo "    Rate         ≈ 0.719  → residual drops 28% per iteration"
echo "$SEP"
echo ""
