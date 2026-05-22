#!/usr/bin/env python3
"""Plot strong/weak scaling results from JSONL metrics files."""

import json
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def plot_strong_scaling(results_dir, output_dir):
    """Strong scaling: speedup and efficiency vs. ranks."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    methods = {}
    for f in sorted(Path(results_dir).glob("strong_*.jsonl")):
        records = load_jsonl(f)
        if not records:
            continue
        method = records[0]["method"]
        ranks = [r["ranks"] for r in records]
        times = [r["time_solve_s"] for r in records]
        methods[method] = (ranks, times)

    for method, (ranks, times) in methods.items():
        t1 = times[0] if ranks[0] == 1 else times[0]
        speedup = [t1 / t for t in times]
        efficiency = [s / r for s, r in zip(speedup, ranks)]

        ax1.plot(ranks, speedup, "o-", label=method)
        ax2.plot(ranks, efficiency, "s-", label=method)

    max_rank = max(max(r) for r, _ in methods.values()) if methods else 16
    ax1.plot([1, max_rank], [1, max_rank], "k--", alpha=0.3, label="Ideal")
    ax1.set_xlabel("MPI Ranks")
    ax1.set_ylabel("Speedup")
    ax1.set_title("Strong Scaling - Speedup")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.axhline(y=1.0, color="k", linestyle="--", alpha=0.3)
    ax2.set_xlabel("MPI Ranks")
    ax2.set_ylabel("Efficiency")
    ax2.set_title("Strong Scaling - Efficiency")
    ax2.set_ylim(0, 1.2)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(output_dir, "strong_scaling.png")
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close()


def plot_weak_scaling(results_dir, output_dir):
    """Weak scaling: solve time vs. ranks (should stay flat)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for f in sorted(Path(results_dir).glob("weak_*.jsonl")):
        records = load_jsonl(f)
        if not records:
            continue
        method = records[0]["method"]
        ranks = [r["ranks"] for r in records]
        times = [r["time_solve_s"] for r in records]
        iters = [r["iterations"] for r in records]

        ax.plot(ranks, times, "o-", label=f"{method} (time)")

    ax.set_xlabel("MPI Ranks")
    ax.set_ylabel("Solve Time (s)")
    ax.set_title("Weak Scaling")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(output_dir, "weak_scaling.png")
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close()


def plot_iteration_comparison(results_dir, output_dir):
    """Bar chart comparing iteration counts across methods and matrices."""
    all_records = []
    for f in Path(results_dir).glob("suite_*.jsonl"):
        all_records.extend(load_jsonl(f))

    if not all_records:
        print("No SuiteSparse results found, skipping iteration comparison.")
        return

    matrices = sorted(set(r["matrix"] for r in all_records))
    methods = sorted(set(r["method"] for r in all_records))

    x = np.arange(len(matrices))
    width = 0.8 / len(methods)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, method in enumerate(methods):
        iters = []
        for mat in matrices:
            match = [r for r in all_records if r["matrix"] == mat and r["method"] == method]
            iters.append(match[0]["iterations"] if match else 0)
        ax.bar(x + i * width, iters, width, label=method)

    ax.set_xlabel("Matrix")
    ax.set_ylabel("Iterations")
    ax.set_title("Iteration Count by Method and Matrix")
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels([os.path.basename(m) for m in matrices], rotation=30, ha="right")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(output_dir, "iteration_comparison.png")
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else results_dir

    os.makedirs(output_dir, exist_ok=True)

    plot_strong_scaling(results_dir, output_dir)
    plot_weak_scaling(results_dir, output_dir)
    plot_iteration_comparison(results_dir, output_dir)
    print("All plots generated.")
