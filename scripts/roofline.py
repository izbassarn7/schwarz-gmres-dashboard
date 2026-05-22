#!/usr/bin/env python3
"""
Roofline performance model for SpMV and GMRES kernels.

Computes arithmetic intensity (AI) and plots the roofline model showing
whether each kernel is compute-bound or memory-bandwidth-bound.

Usage:
  python roofline.py [results/scaling.jsonl] [--output roofline.pdf]

Theory:
  Roofline model (Williams et al., 2009):
    P_achievable = min(P_peak, beta * AI)
  where:
    P_peak = peak FP throughput [GFlop/s]
    beta   = memory bandwidth [GB/s]
    AI     = arithmetic intensity [FLop/byte]

SpMV arithmetic intensity:
  FLops  = 2 * nnz  (one multiply + one add per non-zero)
  Bytes  = nnz * (8 + 4) + n * 8  (values:double + col_idx:int + x:double)
         = nnz * 12 + n * 8
  AI_SpMV = (2 * nnz) / (nnz * 12 + n * 8)

For a typical sparse matrix:  AI_SpMV ≈ 0.17–0.25 [FLop/byte]
Theoretical memory bandwidth on RTX 4050:  ~192 GB/s
Peak FP64 throughput RTX 4050:              ~1.6 TFLop/s
→ Memory-bandwidth-bound regime expected.
"""

import sys
import json
import argparse
import math
import os

# ---- Hardware specs (RTX 4050 Laptop) --------------------------------
PEAK_FP64_GFLOPS  = 1600.0   # GFlop/s  (RTX 4050 Laptop, FP64)
PEAK_FP32_GFLOPS  = 9600.0   # GFlop/s  (FP32)
MEM_BW_CPU_GBS    = 51.2     # GB/s     (DDR5-5200 dual-channel laptop)
MEM_BW_GPU_GBS    = 192.0    # GB/s     (GDDR6 RTX 4050)

# CPU (Ryzen 7 or equivalent laptop)
CPU_PEAK_FP64     = 400.0    # GFlop/s  (approximate AVX2 peak, single socket)
CPU_MEM_BW        = MEM_BW_CPU_GBS

# ---- Arithmetic intensity formulas -----------------------------------

def ai_spmv(n: int, nnz: int) -> float:
    """SpMV arithmetic intensity [FLop/byte] (CSR double precision)."""
    flops = 2.0 * nnz
    bytes_ = nnz * (8 + 4) + n * 8   # vals(double) + col_idx(int) + x(double)
    return flops / bytes_ if bytes_ > 0 else 0.0

def ai_dot(n: int) -> float:
    """Dot product: 2n FLops, 2n*8 bytes."""
    return 2.0 * n / (2.0 * n * 8) if n > 0 else 0.0

def ai_axpy(n: int) -> float:
    """AXPY y = alpha*x + y: 2n FLops, 3n*8 bytes (read x,y write y)."""
    return 2.0 * n / (3.0 * n * 8) if n > 0 else 0.0

def roofline_ceiling(ai: float, peak: float, bw: float) -> float:
    """Roofline ceiling [GFlop/s] for given AI, peak, bandwidth."""
    return min(peak, bw * ai)

# ---- Plotting --------------------------------------------------------

def plot_roofline(results, output_path="results/roofline.pdf"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[WARN] matplotlib not available — writing text report only.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (hw_label, peak, bw) in zip(
        axes,
        [("CPU", CPU_PEAK_FP64, CPU_MEM_BW),
         ("GPU (RTX 4050)", PEAK_FP64_GFLOPS, MEM_BW_GPU_GBS)]):

        ai_range = np.logspace(-3, 3, 500)
        roof = np.minimum(peak, bw * ai_range)

        ax.loglog(ai_range, roof, 'k-', linewidth=2, label="Roofline")
        # Ridge point
        ridge = peak / bw
        ax.axvline(ridge, color='gray', linestyle='--', alpha=0.5,
                   label=f"Ridge: {ridge:.2f} FLop/byte")
        ax.fill_between(ai_range[ai_range < ridge], 1e-3, roof[ai_range < ridge],
                        alpha=0.07, color='blue', label="Memory-bound")
        ax.fill_between(ai_range[ai_range >= ridge], 1e-3, roof[ai_range >= ridge],
                        alpha=0.07, color='red', label="Compute-bound")

        # Plot measured kernels from JSONL results
        colours = {'spmv': 'blue', 'dot': 'green', 'axpy': 'orange',
                   'gmres': 'red', 'ilu': 'purple'}
        for row in results:
            n   = row.get("n",   10000)
            nnz = row.get("nnz", 5 * n)
            t   = row.get("time_solve_s", 0)
            itr = row.get("iterations", 1)
            label_r = row.get("method", "?")
            if t <= 0 or itr <= 0: continue

            # Estimate FLops per iteration: 2 SpMV + 2*restart dot + 2*restart axpy
            restart = row.get("restart", 30)
            flops_iter = (2 * nnz           # SpMV
                        + 2 * restart * n    # dot products (MGS)
                        + 2 * restart * n)   # AXPY updates
            total_flops_g = flops_iter * itr / 1e9
            total_time    = t
            perf_g = total_flops_g / total_time if total_time > 0 else 0

            # Arithmetic intensity: weighted average
            ai_val = (2 * nnz) / (nnz * 12 + n * 8)  # SpMV dominates

            colour = 'red' if 'GPU' in label_r else 'blue'
            ax.scatter(ai_val, perf_g, s=80, c=colour, zorder=5)
            ax.annotate(label_r[:15], (ai_val, perf_g),
                        textcoords="offset points", xytext=(5, 5), fontsize=7)

        # Mark standard SpMV AI for reference
        ref_ai = ai_spmv(10000, 5 * 10000)
        ax.axvline(ref_ai, color='blue', linestyle=':', alpha=0.7,
                   label=f"SpMV AI≈{ref_ai:.3f}")

        ax.set_xlabel("Arithmetic Intensity [FLop/byte]", fontsize=11)
        ax.set_ylabel("Performance [GFlop/s]", fontsize=11)
        ax.set_title(f"Roofline: {hw_label}\n"
                     f"Peak={peak:.0f} GFlop/s  BW={bw:.0f} GB/s", fontsize=11)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, which='both', alpha=0.3)
        ax.set_xlim(1e-3, 1e2)
        ax.set_ylim(1e-2, max(peak, bw) * 2)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Roofline plot saved to {output_path}")


def text_report(results):
    """Print a text roofline analysis table."""
    print("=" * 70)
    print("ROOFLINE ANALYSIS — SpMV & GMRES Kernels")
    print("=" * 70)
    print(f"{'Hardware':<25} {'Peak FP64':>12} {'Mem BW':>10} {'Ridge AI':>10}")
    print("-" * 60)
    for hw, peak, bw in [("CPU (laptop)", CPU_PEAK_FP64, CPU_MEM_BW),
                          ("GPU RTX 4050", PEAK_FP64_GFLOPS, MEM_BW_GPU_GBS)]:
        ridge = peak / bw
        print(f"  {hw:<23} {peak:>10.0f} GF {bw:>8.0f} GB/s {ridge:>8.3f}")
    print()

    print(f"{'Kernel':<15} {'AI (FLop/byte)':>16} {'Regime':>15}")
    print("-" * 50)
    # Representative problem
    n, nnz = 82654, 820000
    ridge_cpu = CPU_PEAK_FP64 / CPU_MEM_BW
    ridge_gpu = PEAK_FP64_GFLOPS / MEM_BW_GPU_GBS

    kernels = [
        ("SpMV",  ai_spmv(n, nnz)),
        ("DOT",   ai_dot(n)),
        ("AXPY",  ai_axpy(n)),
    ]
    for name, ai in kernels:
        regime = ("compute-bound" if ai > ridge_cpu else "memory-bound")
        print(f"  {name:<13} {ai:>14.4f}   {regime}")

    print()
    print("Conclusion: SpMV (AI≈0.17) is deeply memory-bandwidth-bound on both")
    print("  CPU and GPU. Optimisation target: maximise cache reuse.")
    print()

    if results:
        print(f"{'Method':<30} {'n':>8} {'nnz':>10} {'time':>8} {'iters':>6} {'GFlop/s':>10}")
        print("-" * 75)
        for row in results:
            n   = row.get("n",   0)
            nnz = row.get("nnz", 0)
            t   = row.get("time_solve_s", 0)
            itr = row.get("iterations", 1)
            mth = row.get("method", "?")
            restart = row.get("restart", 30)
            if t <= 0 or n == 0: continue
            flops_g = ((2*nnz + 4*restart*n) * itr) / 1e9
            gflops = flops_g / t
            print(f"  {mth:<28} {n:>8} {nnz:>10} {t:>8.3f} {itr:>6} {gflops:>10.3f}")


def main():
    parser = argparse.ArgumentParser(description="Roofline model for GMRES/SpMV")
    parser.add_argument("jsonl", nargs="?", default=None,
                        help="JSONL metrics file from benchmark run")
    parser.add_argument("--output", default="results/roofline.pdf",
                        help="Output PDF path")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip matplotlib plot, text only")
    args = parser.parse_args()

    results = []
    if args.jsonl and os.path.exists(args.jsonl):
        with open(args.jsonl) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    text_report(results)
    if not args.no_plot:
        plot_roofline(results, args.output)


if __name__ == "__main__":
    main()
