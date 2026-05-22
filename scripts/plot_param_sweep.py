#!/usr/bin/env python3
"""Heatmap plots of iteration count and solve time vs (overlap, nparts).

Usage: python plot_param_sweep.py [results/param_sweep/all_results.jsonl]
Produces PNG files in the same directory as the input file.
"""

import json
import sys
import os
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


def load_jsonl(path):
    """Load JSONL records from file."""
    records = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except FileNotFoundError:
        return []
    return records


def build_heatmaps(records):
    """Group records by matrix and build heatmap data.

    Returns:
        dict: {matrix_name -> {overlap_values, nparts_values, iters_grid, time_grid, precond_type}}
    """
    # Group by matrix
    by_matrix = defaultdict(list)
    for record in records:
        matrix = os.path.basename(record.get("matrix", "unknown"))
        by_matrix[matrix].append(record)

    # Build grids for each matrix
    heatmap_data = {}
    for matrix, matrix_records in by_matrix.items():
        if not matrix_records:
            continue

        # Extract unique overlaps and nparts
        overlaps = sorted(set(r.get("overlap", 0) for r in matrix_records))
        nparts_list = sorted(set(r.get("nparts", 2) for r in matrix_records))
        precond_type = matrix_records[0].get("method", "unknown")

        # Create index maps
        overlap_to_idx = {ov: i for i, ov in enumerate(overlaps)}
        nparts_to_idx = {np_: j for j, np_ in enumerate(nparts_list)}

        # Initialize grids with NaN
        iters_grid = np.full((len(overlaps), len(nparts_list)), np.nan)
        time_grid = np.full((len(overlaps), len(nparts_list)), np.nan)
        maxiter_default = 500

        # Fill grids
        for record in matrix_records:
            ov = record.get("overlap", 0)
            np_ = record.get("nparts", 2)
            if ov in overlap_to_idx and np_ in nparts_to_idx:
                i = overlap_to_idx[ov]
                j = nparts_to_idx[np_]
                iters = record.get("iterations", 0)
                time_s = record.get("time_solve_s", 0.0)
                maxiter = record.get("maxiter", maxiter_default)

                # Mark non-converged
                if iters >= maxiter:
                    iters_grid[i, j] = np.nan  # Will show as white
                else:
                    iters_grid[i, j] = iters
                time_grid[i, j] = time_s

        heatmap_data[matrix] = {
            "overlaps": overlaps,
            "nparts": nparts_list,
            "iters_grid": iters_grid,
            "time_grid": time_grid,
            "precond": precond_type,
        }

    return heatmap_data


def plot_heatmap_iterations(matrix_name, data, output_dir):
    """Plot iterations heatmap."""
    overlaps = data["overlaps"]
    nparts_list = data["nparts"]
    iters_grid = data["iters_grid"]
    precond = data["precond"]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot heatmap
    im = ax.imshow(iters_grid, cmap="viridis", aspect="auto", origin="lower")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Iterations", rotation=270, labelpad=15)

    # Annotate cells
    for i, ov in enumerate(overlaps):
        for j, np_ in enumerate(nparts_list):
            val = iters_grid[i, j]
            if np.isnan(val):
                text = "NC"
                color = "red"
            else:
                text = f"{int(val)}"
                color = "white" if val > iters_grid[~np.isnan(iters_grid)].mean() else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9, weight="bold")

    # Set ticks and labels
    ax.set_xticks(range(len(nparts_list)))
    ax.set_yticks(range(len(overlaps)))
    ax.set_xticklabels(nparts_list)
    ax.set_yticklabels(overlaps)
    ax.set_xlabel("Number of Partitions (nparts)")
    ax.set_ylabel("Overlap")
    ax.set_title(f"Iterations — {matrix_name} ({precond})")

    plt.tight_layout()
    out = os.path.join(output_dir, f"heatmap_iters_{matrix_name}.png")
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close()


def plot_heatmap_time(matrix_name, data, output_dir):
    """Plot solve time heatmap."""
    overlaps = data["overlaps"]
    nparts_list = data["nparts"]
    time_grid = data["time_grid"]
    precond = data["precond"]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot heatmap
    im = ax.imshow(time_grid, cmap="viridis", aspect="auto", origin="lower")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Time (s)", rotation=270, labelpad=15)

    # Annotate cells
    for i, ov in enumerate(overlaps):
        for j, np_ in enumerate(nparts_list):
            val = time_grid[i, j]
            if np.isnan(val):
                text = "—"
                color = "red"
            else:
                text = f"{val:.3f}s"
                color = "white" if val > time_grid[~np.isnan(time_grid)].mean() else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8, weight="bold")

    # Set ticks and labels
    ax.set_xticks(range(len(nparts_list)))
    ax.set_yticks(range(len(overlaps)))
    ax.set_xticklabels(nparts_list)
    ax.set_yticklabels(overlaps)
    ax.set_xlabel("Number of Partitions (nparts)")
    ax.set_ylabel("Overlap")
    ax.set_title(f"Solve Time — {matrix_name} ({precond})")

    plt.tight_layout()
    out = os.path.join(output_dir, f"heatmap_time_{matrix_name}.png")
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close()


def print_summary_table(records):
    """Print a summary table of all records."""
    if not records:
        print("No records to summarize.")
        return

    # Group and sort
    by_matrix_ov_np = []
    for record in records:
        matrix = os.path.basename(record.get("matrix", "unknown"))
        precond = record.get("method", "unknown")
        overlap = record.get("overlap", 0)
        nparts = record.get("nparts", 2)
        iters = record.get("iterations", 0)
        time_s = record.get("time_solve_s", 0.0)
        by_matrix_ov_np.append((matrix, precond, overlap, nparts, iters, time_s))

    by_matrix_ov_np.sort(key=lambda x: (x[0], x[2], x[3]))

    # Print table
    print("\n{:<20} {:<10} {:<8} {:<8} {:<7} {:<10}".format(
        "Matrix", "Precond", "Overlap", "NParts", "Iters", "Time(s)"))
    print("-" * 70)
    for matrix, precond, overlap, nparts, iters, time_s in by_matrix_ov_np:
        print("{:<20} {:<10} {:<8} {:<8} {:<7} {:<10.4f}".format(
            matrix, precond, overlap, nparts, iters, time_s))


if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else "results/param_sweep/all_results.jsonl"
    output_dir = os.path.dirname(input_file) if os.path.dirname(input_file) else "."

    records = load_jsonl(input_file)
    if not records:
        print(f"No records found in {input_file}. Have you run the parameter sweep?")
        sys.exit(0)

    # Build heatmaps
    heatmap_data = build_heatmaps(records)
    if not heatmap_data:
        print("No valid heatmap data found.")
        sys.exit(0)

    # Plot heatmaps
    for matrix_name, data in sorted(heatmap_data.items()):
        print(f"\nPlotting heatmaps for matrix: {matrix_name}")
        plot_heatmap_iterations(matrix_name, data, output_dir)
        plot_heatmap_time(matrix_name, data, output_dir)

    # Print summary table
    print_summary_table(records)
