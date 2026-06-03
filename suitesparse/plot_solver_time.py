#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Plot GMRES convergence time comparing two matrix format variants.

Walks two benchmark result directory trees, matches JSON files by relative
path, detects the matrix format from the JSON content, and produces a grouped
bar chart with one pair of bars per matrix.

Usage
-----
    python plot_solver_time.py <base_tree> <amp_tree> [--output FILE]

Example
-------
    python plot_solver_time.py \\
        results-solver_fgs-csr \\
        results-solver_fgs-amp_csr \\
        --output gmres_time.pdf
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
        amp_bins = entry.get("amp_bins") or spmv["amp"].get("amp_bins", {})
        base_type = amp_bins.get("base_type", "?")
        return f"AMP[{base_type}]"
    return fmt or "unknown"


def _extract_time(json_path: Path):
    """Return (matrix_name, solver_name, apply_time_seconds) from a solver benchmark JSON.

    apply_time is None when the field is absent.
    """
    with open(json_path) as f:
        data = json.load(f)
    entry = data[0]
    name = entry.get("problem", {}).get("name") or json_path.stem
    solver = entry.get("solver", {})
    if not solver:
        return name, None
    solver_key = next(iter(solver))
    t = solver[solver_key].get("apply", {}).get("time")
    return name, solver_key, t


def collect_results(tree_root: str):
    """Walk *tree_root* and return ({rel_path: (name, time)}, format_label)."""
    root = Path(tree_root)
    results: dict[Path, tuple[str, float | None]] = {}
    fmt_label: str | None = None
    solver_name = ""

    for json_file in sorted(root.rglob("*.json")):
        if fmt_label is None:
            try:
                fmt_label = _detect_format_label(json_file)
            except Exception:
                pass
        try:
            name, solver_name, t = _extract_time(json_file)
            results[json_file.relative_to(root)] = (name, t)
        except Exception as exc:
            print(f"Warning: skipping {json_file}: {exc}", file=sys.stderr)

    return results, solver_name, fmt_label or "unknown"


def build_chart(
    matrix_names: list[str],
    base_times: list[float],
    amp_times: list[float],
    base_label: str,
    amp_label: str,
    output_path: str,
) -> None:
    n = len(matrix_names)
    x = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, n * 1.4), 5))

    ax.bar(x - width / 2, base_times, width, label=base_label)
    ax.bar(x + width / 2, amp_times, width, label=amp_label)

    ax.set_ylabel("Convergence time (s)")
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
        description="Plot GMRES convergence time for two matrix format variants.",
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

    matrix_names, base_times, amp_times = [], [], []
    for key in common_keys:
        name, b = base_results[key]
        _, a = amp_results[key]
        matrix_names.append(name)
        base_times.append(b if b is not None else 0.0)
        amp_times.append(a if a is not None else 0.0)

    if args.output:
        output_path = args.output
    else:
        safe = lambda s: s.replace("[", "").replace("]", "")
        output_path = f"{safe(solver_name)}-time-{safe(base_label)}_vs_{safe(amp_label)}.pdf"

    build_chart(matrix_names, base_times, amp_times, base_label, amp_label, output_path)


if __name__ == "__main__":
    main()
