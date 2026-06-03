#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Plot GMRES iteration counts comparing two matrix format variants.

Walks two benchmark result directory trees, matches JSON files by relative
path, detects the matrix format from the JSON content, and produces a grouped
bar chart with one pair of bars per matrix.

Usage
-----
    python plot_solver_iterations.py <base_tree> <amp_tree> [--output FILE]

Example
-------
    python plot_solver_iterations.py \\
        results-solver_fgs-csr \\
        results-solver_fgs-amp_csr \\
        --output gmres_iters.pdf
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _detect_format_label(json_path: Path) -> str:
    """Return a human-readable format label inferred from a benchmark JSON."""
    with open(json_path) as f:
        data = json.load(f)
    entry = data[0]
    spmv = entry.get("spmv", {})
    fmt = next(iter(spmv), None)
    if fmt == "amp":
        # prefer top-level amp_bins, fall back to spmv.amp.amp_bins
        amp_bins = entry.get("amp_bins") or spmv["amp"].get("amp_bins", {})
        base_type = amp_bins.get("base_type", "?")
        return f"AMP[{base_type}]"
    return fmt or "unknown"


def _extract_data(json_path: Path):
    """Return (matrix_name, solver_name, iteration_count, residual_norm) from a solver benchmark JSON.

    Values are None when absent.
    """
    with open(json_path) as f:
        data = json.load(f)
    entry = data[0]
    name = entry.get("problem", {}).get("name") or json_path.stem
    solver = entry.get("solver", {})
    if not solver:
        return name, None, None
    solver_key = next(iter(solver))
    solver_data = solver[solver_key]
    iters = solver_data.get("apply", {}).get("iterations")
    residual = solver_data.get("residual_norm")
    return name, solver_key, iters, residual


def collect_results(tree_root: str):
    """Walk *tree_root* and return ({rel_path: (name, iters, residual)}, format_label)."""
    root = Path(tree_root)
    results: dict[Path, tuple[str, int | None, float | None]] = {}
    fmt_label: str | None = None
    solver_name = ""

    for json_file in sorted(root.rglob("*.json")):
        if fmt_label is None:
            try:
                fmt_label = _detect_format_label(json_file)
            except Exception:
                pass
        try:
            name, solver_name, iters, residual = _extract_data(json_file)
            results[json_file.relative_to(root)] = (name, iters, residual)
        except Exception as exc:
            print(f"Warning: skipping {json_file}: {exc}", file=sys.stderr)

    return results, solver_name, fmt_label or "unknown"


def build_chart(
    matrix_names: list[str],
    base_iters: list[int],
    amp_iters: list[int],
    base_residuals: list[float],
    amp_residuals: list[float],
    base_label: str,
    amp_label: str,
    output_path: str,
) -> None:
    n = len(matrix_names)
    x = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, n * 1.4), 5))

    bars_base = ax.bar(x - width / 2, base_iters, width, label=base_label)
    bars_amp = ax.bar(x + width / 2, amp_iters, width, label=amp_label)

    #ax.set_xlabel("Matrix")
    ax.set_ylabel("GMRES iteration count")
    ax.set_xticks(x)
    ax.set_xticklabels(matrix_names, rotation=45, ha="right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    ax2 = ax.twinx()
    ax2.set_ylabel("True residual norm")
    ax2.set_yscale("log")

    base_color = bars_base.patches[0].get_facecolor()
    amp_color = bars_amp.patches[0].get_facecolor()

    ax2.scatter(
        x - width / 2,
        base_residuals,
        marker="o",
        color=base_color,
        edgecolors="black",
        linewidths=0.6,
        zorder=5,
        label=f"{base_label} residual",
    )
    ax2.scatter(
        x + width / 2,
        amp_residuals,
        marker="D",
        color=amp_color,
        edgecolors="black",
        linewidths=0.6,
        zorder=5,
        label=f"{amp_label} residual",
    )

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path)
    print(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot GMRES iteration counts and true residual norms for two matrix format variants.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("base_tree", help="Result directory tree for the base format")
    parser.add_argument("amp_tree", help="Result directory tree for the AMP format")
    parser.add_argument(
        "--output", "-o", default=None, help="Output PDF path (auto-named if omitted)"
    )
    args = parser.parse_args()

    base_results, base_solver_name, base_label = collect_results(args.base_tree)
    amp_results, solver_name, amp_label = collect_results(args.amp_tree)
    assert(base_solver_name == solver_name)
    solver_name = solver_name.replace('-','_')

    common_keys = sorted(set(base_results) & set(amp_results))
    if not common_keys:
        print(
            "Error: no matching JSON files found between the two trees.",
            file=sys.stderr,
        )
        sys.exit(1)

    matrix_names, base_iters, amp_iters = [], [], []
    base_residuals, amp_residuals = [], []
    for key in common_keys:
        name, b_iters, b_res = base_results[key]
        _, a_iters, a_res = amp_results[key]
        matrix_names.append(name)
        base_iters.append(b_iters if b_iters is not None else 0)
        amp_iters.append(a_iters if a_iters is not None else 0)
        base_residuals.append(b_res if b_res is not None else float("nan"))
        amp_residuals.append(a_res if a_res is not None else float("nan"))

    if args.output:
        output_path = args.output
    else:
        safe = lambda s: s.replace("[", "").replace("]", "")
        output_path = f"{safe(solver_name)}-iterations_trueres-{safe(base_label)}_vs_{safe(amp_label)}.pdf"

    build_chart(
        matrix_names,
        base_iters,
        amp_iters,
        base_residuals,
        amp_residuals,
        base_label,
        amp_label,
        output_path,
    )


if __name__ == "__main__":
    main()
