#!/usr/bin/env python3

"""Bar plot comparing SpMV time for ELL, AMP[ELL], CSR, AMP[CSR]."""

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


def load_times(results_dir, base_format):
    """Load base-format and AMP spmv times from result files."""
    results_dir = Path(results_dir)
    records = {}
    for json_path in sorted(results_dir.rglob("*.json")):
        with open(json_path) as f:
            data = json.load(f)
        for entry in data:
            spmv = entry.get("spmv", {})
            if base_format not in spmv or "amp" not in spmv:
                continue
            name = entry.get("problem", {}).get("name", json_path.stem)
            records[name] = {
                "base_time": spmv[base_format]["time"],
                "amp_time": spmv["amp"]["time"],
            }
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Plot SpMV time for ELL, AMP[ELL], CSR, AMP[CSR].")
    parser.add_argument(
        "--csr-dir", required=True,
        help="Results directory for CSR / AMP[CSR] runs.")
    parser.add_argument(
        "--ell-dir", required=True,
        help="Results directory for ELL / AMP[ELL] runs.")
    parser.add_argument(
        "--output", default=None,
        help="Output image path (default: spmv_all_times.png).")
    args = parser.parse_args()

    csr_records = load_times(args.csr_dir, "csr")
    ell_records = load_times(args.ell_dir, "ell")

    common = sorted(set(csr_records) & set(ell_records))
    if not common:
        print("No common matrices found between the two result directories.")
        raise SystemExit(1)

    ell_times = np.array([ell_records[n]["base_time"] for n in common]) * 1e3
    amp_ell_times = np.array([ell_records[n]["amp_time"] for n in common]) * 1e3
    csr_times = np.array([csr_records[n]["base_time"] for n in common]) * 1e3
    amp_csr_times = np.array([csr_records[n]["amp_time"] for n in common]) * 1e3

    x = np.arange(len(common))
    width = 0.19

    fig, ax = plt.subplots(figsize=(max(7, len(common) * 2.5), 5))
    bars_ell = ax.bar(x - 1.5 * width, ell_times, width, label="ELL")
    bars_amp_ell = ax.bar(x - 0.5 * width, amp_ell_times, width,
                          label="AMP[ELL]")
    bars_csr = ax.bar(x + 0.5 * width, csr_times, width, label="CSR")
    bars_amp_csr = ax.bar(x + 1.5 * width, amp_csr_times, width,
                          label="AMP[CSR]")

    ax.set_ylabel("SpMV time (ms)")
    ax.set_yscale("log")
    #ax.set_title("SpMV Performance: ELL, AMP[ELL], CSR, AMP[CSR]")
    ax.set_xticks(x)
    ax.set_xticklabels(common, rotation=30, ha="right")
    ax.legend()
    ax.bar_label(bars_ell, fmt="%.3f", fontsize=7, padding=2)
    ax.bar_label(bars_amp_ell, fmt="%.3f", fontsize=7, padding=2)
    ax.bar_label(bars_csr, fmt="%.3f", fontsize=7, padding=2)
    ax.bar_label(bars_amp_csr, fmt="%.3f", fontsize=7, padding=2)

    fig.tight_layout()
    out_path = args.output if args.output else "spmv_all_times.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")
    #plt.show()


if __name__ == "__main__":
    main()
