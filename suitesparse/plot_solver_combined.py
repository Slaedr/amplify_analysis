#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Combined solver benchmark summary: speedup chart, iterations/residuals chart,
and an ASCII table with times, speedups, iteration counts, and true residuals.

Usage
-----
    python plot_solver_combined.py <base_tree> <amp_tree> [--output-speedup FILE]
                                   [--output-iters FILE]

Example
-------
    python plot_solver_combined.py \\
        results-solver_fgs-csr \\
        results-solver_fgs-amp_csr
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _detect_format_label(json_path: Path) -> str:
    with open(json_path) as f:
        data = json.load(f)
    entry = data[0]
    spmv = entry.get("spmv", {})
    fmt = next(iter(spmv), None)
    if fmt == "amp":
        amp_bins = entry.get("amp_bins") or spmv["amp"].get("amp_bins", {})
        base_type = amp_bins.get("base_type", "?")
        return f"AMP[{base_type}]"
    return fmt or "unknown"


def _extract_all(json_path: Path):
    """Return (matrix_name, solver_key, time, iterations, residual_norm)."""
    with open(json_path) as f:
        data = json.load(f)
    entry = data[0]
    name = entry.get("problem", {}).get("name") or json_path.stem
    solver = entry.get("solver", {})
    if not solver:
        return name, "", None, None, None
    solver_key = next(iter(solver))
    solver_data = solver[solver_key]
    apply = solver_data.get("apply", {})
    time = apply.get("time")
    iters = apply.get("iterations")
    residual = solver_data.get("residual_norm")
    return name, solver_key, time, iters, residual


def collect_results(tree_root: str):
    """Walk *tree_root* and return ({rel_path: (name, time, iters, residual)}, solver_name, fmt_label)."""
    root = Path(tree_root)
    results: dict = {}
    fmt_label: str | None = None
    solver_name = ""

    for json_file in sorted(root.rglob("*.json")):
        if fmt_label is None:
            try:
                fmt_label = _detect_format_label(json_file)
            except Exception:
                pass
        try:
            name, solver_name, time, iters, residual = _extract_all(json_file)
            results[json_file.relative_to(root)] = (name, time, iters, residual)
        except Exception as exc:
            print(f"Warning: skipping {json_file}: {exc}", file=sys.stderr)

    return results, solver_name, fmt_label or "unknown"


def print_table(
    matrix_names: list,
    base_times: list,
    amp_times: list,
    speedups: list,
    base_iters: list,
    amp_iters: list,
    base_residuals: list,
    amp_residuals: list,
    base_label: str,
    amp_label: str,
) -> None:
    amp_col = f"AMP[{base_label}]"
    c0 = max(len("Matrix"), max(len(n) for n in matrix_names))
    c_bt = max(len(base_label), 10)
    c_at = max(len(amp_col), 10)
    c_sp = max(len("Speedup"), 7)
    c_bi = max(len(f"{base_label}_iters"), 10)
    c_ai = max(len(f"{amp_col}_iters"), 10)
    c_br = max(len(f"{base_label}_res"), 12)
    c_ar = max(len(f"{amp_col}_res"), 12)

    header = (
        f"{'Matrix':<{c0}}"
        f"  {base_label:>{c_bt}}"
        f"  {amp_col:>{c_at}}"
        f"  {'Speedup':>{c_sp}}"
        f"  {base_label + '_iters':>{c_bi}}"
        f"  {amp_col + '_iters':>{c_ai}}"
        f"  {base_label + '_res':>{c_br}}"
        f"  {amp_col + '_res':>{c_ar}}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for name, bt, at, sp, bi, ai, br, ar in zip(
        matrix_names, base_times, amp_times, speedups,
        base_iters, amp_iters, base_residuals, amp_residuals,
    ):
        br_str = f"{br:.3e}" if br == br else "N/A"  # nan check
        ar_str = f"{ar:.3e}" if ar == ar else "N/A"
        print(
            f"{name:<{c0}}"
            f"  {bt:>{c_bt}.3g}"
            f"  {at:>{c_at}.3g}"
            f"  {sp:>{c_sp}.3g}"
            f"  {bi:>{c_bi}}"
            f"  {ai:>{c_ai}}"
            f"  {br_str:>{c_br}}"
            f"  {ar_str:>{c_ar}}"
        )
    print(sep)


def build_speedup_chart(
    matrix_names: list,
    speedups: list,
    amp_label: str,
    base_label: str,
    output_path: str,
) -> None:
    n = len(matrix_names)
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(max(6, n * 1.2), 5))
    second_color = plt.rcParams["axes.prop_cycle"].by_key()["color"][1]
    ax.bar(x, speedups, 0.5, color=second_color, label=amp_label)
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel(f"Speedup over {base_label}")
    ax.set_xticks(x)
    ax.set_xticklabels(matrix_names, rotation=45, ha="right")
    ax.legend()
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_path)
    print(f"Saved: {output_path}")


def build_iters_chart(
    matrix_names: list,
    base_iters: list,
    amp_iters: list,
    base_residuals: list,
    amp_residuals: list,
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
        x - width / 2, base_residuals,
        marker="o", color=base_color, edgecolors="black",
        linewidths=0.6, zorder=5, label=f"{base_label} residual",
    )
    ax2.scatter(
        x + width / 2, amp_residuals,
        marker="D", color=amp_color, edgecolors="black",
        linewidths=0.6, zorder=5, label=f"{amp_label} residual",
    )

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path)
    print(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate solver speedup and iteration/residual plots plus a summary table.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("base_tree", help="Result directory tree for the base format")
    parser.add_argument("amp_tree", help="Result directory tree for the AMP format")
    parser.add_argument("--output-speedup", default=None, help="Output path for speedup chart")
    parser.add_argument("--output-iters", default=None, help="Output path for iterations/residuals chart")
    args = parser.parse_args()

    base_results, base_solver_name, base_label = collect_results(args.base_tree)
    amp_results, solver_name, amp_label = collect_results(args.amp_tree)
    assert base_solver_name == solver_name
    solver_name = solver_name.replace("-", "_")

    common_keys = sorted(set(base_results) & set(amp_results))
    if not common_keys:
        print("Error: no matching JSON files found between the two trees.", file=sys.stderr)
        sys.exit(1)

    matrix_names = []
    base_times, amp_times, speedups = [], [], []
    base_iters, amp_iters = [], []
    base_residuals, amp_residuals = [], []

    for key in common_keys:
        name, b_time, b_iters, b_res = base_results[key]
        _, a_time, a_iters, a_res = amp_results[key]
        if b_time is None or a_time is None:
            print(f"Warning: missing time for {name}, skipping.", file=sys.stderr)
            continue
        matrix_names.append(name)
        base_times.append(b_time)
        amp_times.append(a_time)
        speedups.append(b_time / a_time)
        base_iters.append(b_iters if b_iters is not None else 0)
        amp_iters.append(a_iters if a_iters is not None else 0)
        base_residuals.append(b_res if b_res is not None else float("nan"))
        amp_residuals.append(a_res if a_res is not None else float("nan"))

    print_table(
        matrix_names, base_times, amp_times, speedups,
        base_iters, amp_iters, base_residuals, amp_residuals,
        base_label, amp_label,
    )

    safe = lambda s: s.replace("[", "").replace("]", "")

    speedup_path = args.output_speedup or (
        f"{safe(solver_name)}-speedup-{safe(amp_label)}_over_{safe(base_label)}.pdf"
    )
    iters_path = args.output_iters or (
        f"{safe(solver_name)}-iterations_trueres-{safe(base_label)}_vs_{safe(amp_label)}.pdf"
    )

    build_speedup_chart(matrix_names, speedups, amp_label, base_label, speedup_path)
    build_iters_chart(
        matrix_names, base_iters, amp_iters, base_residuals, amp_residuals,
        base_label, amp_label, iters_path,
    )


if __name__ == "__main__":
    main()
