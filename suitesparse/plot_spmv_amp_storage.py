#!/usr/bin/env python3

"""Bar plot comparing total stored NNZ: original vs AMP[CSR] vs AMP[ELL]."""

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


def load_amp_nnz(results_dir, base_type):
    """Load AMP bin data and compute total stored NNZ per matrix.

    For CSR: sum of bin NNZ.
    For ELL: sum of (bin max_nnz_per_row * nrows) across bins.
    """
    results_dir = Path(results_dir)
    records = {}
    for json_path in sorted(results_dir.rglob("*.json")):
        with open(json_path) as f:
            data = json.load(f)
        for entry in data:
            spmv = entry.get("spmv", {})
            amp = spmv.get("amp", {})
            amp_bins = amp.get("amp_bins")
            if not amp_bins:
                continue
            if amp_bins.get("base_type") != base_type:
                continue
            name = entry.get("problem", {}).get("name", json_path.stem)
            nrows = entry["rows"]
            total = 0
            for key, val in amp_bins.items():
                if not key.startswith("bin_"):
                    continue
                if base_type == "csr":
                    total += val
                else:
                    total += val * nrows
            records[name] = {"amp_nnz": total, "orig_nnz": entry["nonzeros"]}
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Plot total stored NNZ: original vs AMP[CSR] vs AMP[ELL].")
    parser.add_argument(
        "--csr-dir", required=True,
        help="Results directory for CSR / AMP[CSR] runs.")
    parser.add_argument(
        "--ell-dir", required=True,
        help="Results directory for ELL / AMP[ELL] runs.")
    parser.add_argument(
        "--output", default=None,
        help="Output image path (default: spmv_amp_storage.png).")
    args = parser.parse_args()

    csr_records = load_amp_nnz(args.csr_dir, "csr")
    ell_records = load_amp_nnz(args.ell_dir, "ell")

    common = sorted(set(csr_records) & set(ell_records))
    if not common:
        print("No common matrices found between the two result directories.")
        raise SystemExit(1)

    orig_nnz = np.array([csr_records[n]["orig_nnz"] for n in common])
    csr_nnz = np.array([csr_records[n]["amp_nnz"] for n in common])
    ell_nnz = np.array([ell_records[n]["amp_nnz"] for n in common])

    x = np.arange(len(common))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(6, len(common) * 2.2), 5))
    bars_orig = ax.bar(x - width, orig_nnz, width, label="Original")
    bars_csr = ax.bar(x, csr_nnz, width, label="AMP[CSR]")
    bars_ell = ax.bar(x + width, ell_nnz, width, label="AMP[ELL]")

    ax.set_ylabel("Total stored nonzeros")
    ax.set_yscale("log")
    #ax.set_title("Stored NNZ: Original vs AMP[CSR] vs AMP[ELL]")
    ax.set_xticks(x)
    ax.set_xticklabels(common, rotation=30, ha="right")
    ax.legend()
    ax.bar_label(bars_orig, fmt="%.2e", fontsize=8, padding=2)
    ax.bar_label(bars_csr, fmt="%.2e", fontsize=8, padding=2)
    ax.bar_label(bars_ell, fmt="%.2e", fontsize=8, padding=2)

    fig.tight_layout()
    out_path = args.output if args.output else "spmv_amp_storage.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")
    #plt.show()


if __name__ == "__main__":
    main()
