#!/usr/bin/env python3

"""Bar plot comparing a base format (CSR or ELL) vs AMP SpMV performance."""

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
    """Load JSON result files and extract base-format and AMP spmv times."""
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
                "base_time": spmv[base_format]["time"],
                "amp_time": spmv["amp"]["time"],
            })
    records.sort(key=lambda r: r["name"])
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Plot base-format vs AMP SpMV performance.")
    parser.add_argument(
        "--results-dir",
        default="results/GB10/cuda/SuiteSparse",
        help="Path to SuiteSparse results directory.")
    parser.add_argument(
        "--base-format", default="csrc", choices=["csrc", "ell", "cusparse_csr"],
        help="Base sparse format to compare against AMP (default: csrc).")
    parser.add_argument(
        "--output", default=None,
        help="Output image path (default: <results-dir>/spmv_<base>_vs_amp.png).")
    args = parser.parse_args()

    base_label = args.base_format.upper()
    records = load_results(args.results_dir, args.base_format)
    if not records:
        print(f"No results with both '{args.base_format}' and 'amp' "
              f"found in {args.results_dir}")
        raise SystemExit(1)

    names = [r["name"] for r in records]
    base_times = np.array([r["base_time"] for r in records])
    amp_times = np.array([r["amp_time"] for r in records])
    amp_relative = base_times / amp_times
    geometric_mean_speedup = np.exp(np.mean(np.log(amp_relative)))
    print(f"Geometric mean speedup: {geometric_mean_speedup:.2f}x")

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.8), 5))
    bars_base = ax.bar(x - width / 2, np.ones(len(names)), width,
                       label=base_label)
    base_label_p = base_label
    if "CUSPARSE" in base_label:
        base_label_p = base_label.split("_")[1]
    bars_amp = ax.bar(x + width / 2, amp_relative, width,
                      label=f"AMP[{base_label_p}]")

    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel(f"Speedup over {base_label}")
    #ax.set_title(f"AMP[{base_label}] SpMV Speedup over {base_label}")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.legend()
    speedup_label_kwargs = {
        "fmt": "%.2f",
        "fontsize": 12,
        "padding": -4,
        "color": "black",
    }
    ax.bar_label(bars_base, **speedup_label_kwargs)
    ax.bar_label(bars_amp, **speedup_label_kwargs)

    fig.tight_layout()
    out_path = (args.output if args.output else
                str(Path(args.results_dir) /
                    f"spmv_speedup_{args.base_format}_vs_amp.pdf"))
    fig.savefig(out_path)
    print(f"Saved plot to {out_path}")
    plt.grid('on')
    #plt.show()


if __name__ == "__main__":
    main()
