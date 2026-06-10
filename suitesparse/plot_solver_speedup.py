#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Plot GMRES speedup of one matrix format over another.

Walks two benchmark result directory trees, matches JSON files by relative
path, and produces a bar chart showing speedup (base_time / amp_time) per
matrix.

Usage
-----
    python plot_solver_speedup.py <base_tree> <amp_tree> [--output FILE]

Example
-------
    python plot_solver_speedup.py \\
        results-solver_fgs-csr \\
        results-solver_fgs-amp_csr \\
        --output gmres_speedup.pdf
"""

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

from plot_solver_time import collect_results


def print_table(
    matrix_names: list[str],
    base_times: list[float],
    amp_times: list[float],
    speedups: list[float],
    base_label: str,
    amp_label: str,
) -> None:
    amp_col_label = f"AMP[{base_label}]"
    col0 = max(len("Matrix"), max(len(n) for n in matrix_names))
    col1 = max(len(base_label), 12)
    col2 = max(len(amp_col_label), 12)
    col3 = len("Speedup")

    header = (
        f"{'Matrix':<{col0}}  {base_label:>{col1}}  {amp_col_label:>{col2}}  {'Speedup':>{col3}}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for name, b, a, s in zip(matrix_names, base_times, amp_times, speedups):
        print(f"{name:<{col0}}  {b:>{col1}.3g}  {a:>{col2}.3g}  {s:>{col3}.3g}")
    print(sep)


def build_speedup_chart(
    matrix_names: list[str],
    speedups: list[float],
    amp_label: str,
    base_label: str,
    output_path: str,
) -> None:
    n = len(matrix_names)
    x = np.arange(n)
    width = 0.5

    fig, ax = plt.subplots(figsize=(max(6, n * 1.2), 5))

    # Use the default color cycle's second color to match the second bar in
    # plot_solver_time.py (which also uses the default cycle).
    second_color = plt.rcParams["axes.prop_cycle"].by_key()["color"][1]
    ax.bar(x, speedups, width, color=second_color, label=amp_label)

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot GMRES speedup of the AMP format over the base format.",
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
    assert base_solver_name == solver_name
    solver_name = solver_name.replace("-", "_")

    common_keys = sorted(set(base_results) & set(amp_results))
    if not common_keys:
        print(
            "Error: no matching JSON files found between the two trees.",
            file=sys.stderr,
        )
        sys.exit(1)

    matrix_names, base_times, amp_times, speedups = [], [], [], []
    for key in common_keys:
        name, b = base_results[key]
        _, a = amp_results[key]
        if b and a:
            matrix_names.append(name)
            base_times.append(b)
            amp_times.append(a)
            speedups.append(b / a)
        else:
            print(f"Warning: missing time for {name}, skipping.", file=sys.stderr)

    print_table(matrix_names, base_times, amp_times, speedups, base_label, amp_label)

    if args.output:
        output_path = args.output
    else:
        safe = lambda s: s.replace("[", "").replace("]", "")
        output_path = f"{safe(solver_name)}-speedup-{safe(amp_label)}_over_{safe(base_label)}.pdf"

    build_speedup_chart(matrix_names, speedups, amp_label, base_label, output_path)


if __name__ == "__main__":
    main()
