#!/usr/bin/env python3

"""Bar plot comparing relative SpMV error of a base format vs AMP."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 15,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
})


def load_results(results_dir, base_format):
    """Load JSON result files and extract max_relative_norm2 for both formats."""
    results_dir = Path(results_dir)
    records = []
    for json_path in sorted(results_dir.rglob("*.json")):
        with open(json_path) as f:
            data = json.load(f)
        for entry in data:
            spmv = entry.get("spmv", {})
            if base_format not in spmv or "amp" not in spmv:
                continue
            name = entry.get("problem", {}).get("name", json_path.stem)
            records.append({
                "name": name,
                "base_error": spmv[base_format]["max_relative_norm2"],
                "amp_error": spmv["amp"]["max_relative_norm2"],
            })
    records.sort(key=lambda r: r["name"])
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Plot base-format vs AMP SpMV relative error.")
    parser.add_argument(
        "--results-dir",
        default="results/GB10/cuda/SuiteSparse",
        help="Path to SuiteSparse results directory.")
    parser.add_argument(
        "--base-format", default="csr", choices=["csr", "ell"],
        help="Base sparse format to compare against AMP (default: csr).")
    parser.add_argument(
        "--output", default=None,
        help="Output image path (default: <results-dir>/spmv_error_<base>_vs_amp.png).")
    args = parser.parse_args()

    base_label = args.base_format.upper()
    records = load_results(args.results_dir, args.base_format)
    if not records:
        print(f"No results with both '{args.base_format}' and 'amp' "
              f"found in {args.results_dir}")
        raise SystemExit(1)

    names = [r["name"] for r in records]
    base_errors = np.array([r["base_error"] for r in records])
    amp_errors = np.array([r["amp_error"] for r in records])

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.8), 5))
    bars_base = ax.bar(x - width / 2, base_errors, width, label=base_label)
    bars_amp = ax.bar(x + width / 2, amp_errors, width,
                      label=f"AMP[{base_label}]")

    ax.set_ylabel("Max relative norm2 error")
    ax.set_yscale("log")
    #ax.set_title(f"{base_label} vs AMP[{base_label}] SpMV Relative Error")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.legend()
    ax.bar_label(bars_base, fmt="%.2e", fontsize=10, padding=2)
    ax.bar_label(bars_amp, fmt="%.2e", fontsize=10, padding=2)

    plt.grid('on')
    fig.tight_layout()
    out_path = (args.output if args.output else
                str(Path(args.results_dir) /
                    f"spmv_error_{args.base_format}_vs_amp.png"))
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")
    #plt.show()


if __name__ == "__main__":
    main()
