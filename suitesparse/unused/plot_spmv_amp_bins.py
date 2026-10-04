#!/usr/bin/env python3

"""Bar plot showing AMP bin nonzero distribution across test matrices."""

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


BIN_LABELS = {
    "bin_0": "FP64",
    "bin_1": "FP32",
    #"bin_2": "FP16",
    "bin_2": "BF16"
}


def load_results(results_dir):
    """Load JSON result files and extract AMP bin info."""
    results_dir = Path(results_dir)
    records = []
    for json_path in sorted(results_dir.rglob("*.json")):
        with open(json_path) as f:
            data = json.load(f)
        for entry in data:
            spmv = entry.get("spmv", {})
            amp = spmv.get("amp", {})
            amp_bins = amp.get("amp_bins")
            if not amp_bins:
                continue
            name = entry.get("problem", {}).get("name", json_path.stem)
            bins = {}
            for key in sorted(k for k in amp_bins if k.startswith("bin_")):
                bins[key] = amp_bins[key]
            records.append({
                "name": name,
                "bins": bins,
                "base_type": amp_bins.get("base_type", "ell"),
            })
    records.sort(key=lambda r: r["name"])
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Plot AMP bin nonzero distribution per matrix.")
    parser.add_argument(
        "--results-dir",
        default="results/GB10/cuda/SuiteSparse",
        help="Path to SuiteSparse results directory.")
    parser.add_argument(
        "--output", default=None,
        help="Output image path.")
    args = parser.parse_args()

    records = load_results(args.results_dir)
    if not records:
        print(f"No AMP bin data found in {args.results_dir}")
        raise SystemExit(1)

    # Collect all bin keys across all matrices
    all_bin_keys = sorted({k for r in records for k in r["bins"]})
    base_type = records[0]["base_type"].upper()
    ylabel = ("Total nonzeros per bin" if base_type == "CSR"
              else "Max nonzeros per row per bin")

    names = [r["name"] for r in records]
    x = np.arange(len(names))
    num_bins = len(all_bin_keys)
    width = 0.8 / num_bins

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 2), 5))
    for i, key in enumerate(all_bin_keys):
        values = [r["bins"].get(key, 0) for r in records]
        offset = (i - (num_bins - 1) / 2) * width
        label = BIN_LABELS.get(key, key)
        bars = ax.bar(x + offset, values, width, label=label)
        ax.bar_label(bars, fontsize=9, padding=2)

    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    #ax.set_title(f"AMP[{base_type}] Bin Distribution")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.legend()

    fig.tight_layout()
    out_path = (args.output if args.output else
                str(Path(args.results_dir) / f"spmv_amp_bins_{base_type}.png"))
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")
    #plt.show()


if __name__ == "__main__":
    main()
