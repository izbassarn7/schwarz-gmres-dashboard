#!/usr/bin/env python3
"""Compare our solver against PETSc and Trilinos baselines.

Usage: python compare.py results/
"""

import json
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_all_results(results_dir):
    records = []
    for f in Path(results_dir).glob("*.jsonl"):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def group_by_matrix(records):
    groups = {}
    for r in records:
        mat = os.path.basename(r.get("matrix", "unknown"))
        groups.setdefault(mat, []).append(r)
    return groups


def print_comparison_table(records):
    groups = group_by_matrix(records)

    print(f"{'Matrix':<20} {'Method':<30} {'Iters':>6} {'Setup(s)':>10} "
          f"{'Solve(s)':>10} {'Total(s)':>10} {'Residual':>12}")
    print("-" * 100)

    for mat in sorted(groups.keys()):
        entries = sorted(groups[mat], key=lambda r: r["method"])
        for r in entries:
            total = r["time_setup_s"] + r["time_solve_s"]
            print(f"{mat:<20} {r['method']:<30} {r['iterations']:>6} "
                  f"{r['time_setup_s']:>10.4f} {r['time_solve_s']:>10.4f} "
                  f"{total:>10.4f} {r['residual_norm']:>12.2e}")
        print()


def plot_comparison(records, output_dir):
    groups = group_by_matrix(records)

    for mat, entries in groups.items():
        methods = [r["method"] for r in entries]
        times_setup = [r["time_setup_s"] for r in entries]
        times_solve = [r["time_solve_s"] for r in entries]
        iters = [r["iterations"] for r in entries]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        x = np.arange(len(methods))
        width = 0.35
        ax1.bar(x - width / 2, times_setup, width, label="Setup")
        ax1.bar(x + width / 2, times_solve, width, label="Solve")
        ax1.set_ylabel("Time (s)")
        ax1.set_title(f"{mat} - Time Comparison")
        ax1.set_xticks(x)
        ax1.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
        ax1.legend()
        ax1.grid(True, axis="y", alpha=0.3)

        ax2.bar(x, iters, color="steelblue")
        ax2.set_ylabel("Iterations")
        ax2.set_title(f"{mat} - Iteration Count")
        ax2.set_xticks(x)
        ax2.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
        ax2.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        out = os.path.join(output_dir, f"compare_{mat}.png")
        plt.savefig(out, dpi=150)
        print(f"Saved: {out}")
        plt.close()


def plot_summary(records, output_dir):
    """Single summary chart with normalized performance."""
    groups = group_by_matrix(records)

    # Find our methods vs library methods
    our_methods = [r for r in records if not r["method"].startswith(("PETSc", "Trilinos"))]
    petsc_methods = [r for r in records if r["method"].startswith("PETSc")]
    trilinos_methods = [r for r in records if r["method"].startswith("Trilinos")]

    if not petsc_methods and not trilinos_methods:
        print("No PETSc/Trilinos results for summary comparison.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    matrices = sorted(groups.keys())
    bar_data = {}

    for mat in matrices:
        mat_records = groups[mat]
        for r in mat_records:
            total_time = r["time_setup_s"] + r["time_solve_s"]
            key = r["method"]
            bar_data.setdefault(key, {})[mat] = total_time

    x = np.arange(len(matrices))
    n_methods = len(bar_data)
    width = 0.8 / max(n_methods, 1)

    for i, (method, mat_times) in enumerate(sorted(bar_data.items())):
        times = [mat_times.get(m, 0) for m in matrices]
        ax.bar(x + i * width, times, width, label=method)

    ax.set_xlabel("Matrix")
    ax.set_ylabel("Total Time (s)")
    ax.set_title("Performance Comparison: Ours vs PETSc vs Trilinos")
    ax.set_xticks(x + width * (n_methods - 1) / 2)
    ax.set_xticklabels(matrices, rotation=30, ha="right")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(output_dir, "summary_comparison.png")
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"

    records = load_all_results(results_dir)
    if not records:
        print(f"No results found in {results_dir}/")
        sys.exit(1)

    print(f"Loaded {len(records)} result records.\n")
    print_comparison_table(records)
    plot_comparison(records, results_dir)
    plot_summary(records, results_dir)
