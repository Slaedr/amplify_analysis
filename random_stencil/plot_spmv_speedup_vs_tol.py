#!/usr/bin/env python3

"""AMP SpMV speedup over the base matrix type as a function of the AMP tolerance.

Reads the JSON files written by ``benchmark/amp/amp_benchmark_spmv`` (one run
per tolerance / strategy, as produced by ``sweep_spmv_tolerance.sh``) and draws

  * one group of bars per AMP tolerance (x-axis, decreasing left to right),
  * one bar per AMP SpMV strategy inside a group,
  * a horizontal line for the speedup of the base format in FP32,
  * optionally a second horizontal line for FP16 if the runs contain one,
  * the y = 1 reference line for the FP64 base format.

Every tolerance / strategy / base-format label is taken from the ``config``
block inside each JSON, so file names do not matter; the tolerance is not
encoded in the name the benchmark writes.

Usage:
    ./plot_spmv_speedup_vs_tol.py results-spmv-tol-ell-cuda
    ./plot_spmv_speedup_vs_tol.py results/ --base-format csr -o speedup.pdf
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({
    "font.size": 15,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 15,
    "legend.fontsize": 13,
})

# Colour-blind safe, ordered so the first two are the usual
# monolithic_classical / independent_buckets pair.
STRATEGY_COLORS = ["#2c7bb6", "#d7191c", "#fdae61", "#4daf4a", "#984ea3"]
FP32_COLOR = "#1a9850"
FP16_COLOR = "#756bb1"

PRETTY_STRATEGY = {
    "monolithic_classical": "AMP monolithic",
    "independent_buckets": "AMP independent buckets",
}


def find_row(rows, predicate):
    for row in rows:
        if predicate(row.get("format", "")):
            return row
    return None


def load_runs(results_dir):
    """Collect one record per benchmark JSON found under results_dir."""
    runs = []
    for path in sorted(Path(results_dir).rglob("*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  skipping {path}: {exc}")
            continue
        if not isinstance(data, dict) or "spmv" not in data:
            continue  # per-run config.json and other bystanders
        cfg = data.get("config", {})
        rows = data["spmv"]

        base = find_row(rows, lambda f: f.endswith("<double>")
                        and not f.startswith("AMP"))
        amp = find_row(rows, lambda f: f.startswith("AMP"))
        if base is None or amp is None:
            print(f"  skipping {path}: no base<double> / AMP entry")
            continue
        fp32 = find_row(rows, lambda f: f.endswith("<float>")
                        and not f.startswith("AMP"))
        fp16 = find_row(rows, lambda f: f.endswith("<half>")
                        and not f.startswith("AMP"))

        base_t = base["time_ms"]
        runs.append({
            "path": path,
            "base_format": cfg.get("amp_base_format", "?"),
            "base_label": base["format"],
            "executor": cfg.get("executor", "?"),
            "tolerance": float(cfg.get("amp_tolerance", "nan")),
            "strategy": cfg.get("amp_spmv_strategy", "unknown"),
            "amp_speedup": base_t / amp["time_ms"],
            "fp32_speedup": base_t / fp32["time_ms"] if fp32 else None,
            "fp16_speedup": base_t / fp16["time_ms"] if fp16 else None,
            "amp_error": amp.get("rel_error_vs_double"),
        })
    return runs


def mean(values):
    return sum(values) / len(values)


def main():
    parser = argparse.ArgumentParser(
        description="Plot AMP SpMV speedup vs AMP tolerance.")
    parser.add_argument(
        "results_dir", nargs="?", default="results",
        help="Directory holding the sweep results (searched recursively).")
    parser.add_argument(
        "--base-format", default=None, choices=["ell", "csr"],
        help="Which base format to plot; required only if the tree mixes both.")
    parser.add_argument(
        "--strategies", default=None,
        help="Comma separated strategy order, e.g. "
             "monolithic_classical,independent_buckets. "
             "Default: all found, monolithic first.")
    parser.add_argument(
        "--no-fp16", action="store_true",
        help="Do not draw the FP16 reference line even if the runs have one.")
    parser.add_argument(
        "--no-labels", action="store_true",
        help="Do not print the speedup value on top of each bar.")
    parser.add_argument(
        "--title", default=None, help="Optional figure title.")
    parser.add_argument(
        "--ymax", type=float, default=None, help="Force the y-axis maximum.")
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output file (default <results-dir>/spmv_speedup_vs_tolerance.pdf; "
             "extension picks the format).")
    parser.add_argument("--dpi", type=int, default=400)
    args = parser.parse_args()

    runs = load_runs(args.results_dir)
    if not runs:
        raise SystemExit(f"No SpMV result JSON found under {args.results_dir}")

    formats = sorted({r["base_format"] for r in runs})
    if args.base_format:
        runs = [r for r in runs if r["base_format"] == args.base_format]
        if not runs:
            raise SystemExit(f"No runs with base format '{args.base_format}' "
                             f"(found: {', '.join(formats)})")
    elif len(formats) > 1:
        raise SystemExit("Results mix base formats "
                         f"({', '.join(formats)}); pass --base-format.")

    base_label = runs[0]["base_label"]          # e.g. "ELL<double>"
    base_name = base_label.split("<")[0]        # e.g. "ELL"
    executor = runs[0]["executor"]

    # ---- aggregate: (tolerance, strategy) -> mean speedup -------------------
    by_key = defaultdict(list)
    for r in runs:
        by_key[(r["tolerance"], r["strategy"])].append(r["amp_speedup"])

    tolerances = sorted({t for t, _ in by_key}, reverse=True)  # 1e-4 ... 1e-14
    found_strategies = sorted({s for _, s in by_key},
                              key=lambda s: (s != "monolithic_classical", s))
    if args.strategies:
        strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
        missing = [s for s in strategies if s not in found_strategies]
        if missing:
            raise SystemExit(f"No runs for strategy/strategies: "
                             f"{', '.join(missing)} "
                             f"(found: {', '.join(found_strategies)})")
    else:
        strategies = found_strategies

    # ---- reference lines ----------------------------------------------------
    fp32_vals = [r["fp32_speedup"] for r in runs if r["fp32_speedup"]]
    fp16_vals = [r["fp16_speedup"] for r in runs if r["fp16_speedup"]]
    fp32 = mean(fp32_vals) if fp32_vals else None
    fp16 = mean(fp16_vals) if fp16_vals and not args.no_fp16 else None

    # ---- figure -------------------------------------------------------------
    x = np.arange(len(tolerances), dtype=float)
    n_str = len(strategies)
    width = min(0.8 / n_str, 0.35)

    fig_w = min(12.0, max(6.5, 0.85 * len(tolerances) + 2.0))
    fig, ax = plt.subplots(figsize=(fig_w, 4.8))

    all_vals = []
    for i, strategy in enumerate(strategies):
        offset = (i - (n_str - 1) / 2) * width
        vals, positions = [], []
        for xi, tol in enumerate(tolerances):
            samples = by_key.get((tol, strategy))
            if not samples:
                continue
            vals.append(mean(samples))
            positions.append(x[xi] + offset)
        if not vals:
            continue
        all_vals += vals
        bars = ax.bar(positions, vals, width,
                      label=PRETTY_STRATEGY.get(strategy, strategy),
                      color=STRATEGY_COLORS[i % len(STRATEGY_COLORS)],
                      edgecolor="black", linewidth=0.6, zorder=3)
        if not args.no_labels:
            ax.bar_label(bars, fmt="%.2f", fontsize=10, padding=2,
                         rotation=90 if len(tolerances) > 8 else 0)

    if fp32 is not None:
        all_vals.append(fp32)
        ax.axhline(fp32, color=FP32_COLOR, linestyle="--", linewidth=1.8,
                   zorder=4, label=f"{base_name}<float> (FP32), {fp32:.2f}x")
        if len(fp32_vals) > 1:
            lo, hi = min(fp32_vals), max(fp32_vals)
            if hi - lo > 0.01 * fp32:
                ax.axhspan(lo, hi, color=FP32_COLOR, alpha=0.12, zorder=1)
    if fp16 is not None:
        all_vals.append(fp16)
        ax.axhline(fp16, color=FP16_COLOR, linestyle=":", linewidth=1.8,
                   zorder=4, label=f"{base_name}<half> (FP16), {fp16:.2f}x")

    ax.axhline(1.0, color="black", linewidth=1.0, zorder=4)

    ax.set_xlabel("AMP tolerance")
    ax.set_ylabel(f"Speedup over {base_label}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"$10^{{{round(math.log10(t))}}}$" for t in tolerances])
    ax.set_xlim(-0.6, len(tolerances) - 0.4)
    top = args.ymax if args.ymax else max(all_vals) * 1.3
    ax.set_ylim(0.0, top)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    # Bars normally shrink towards tighter tolerances, so the right corner is
    # the least crowded place for the legend.
    ax.legend(loc="upper right", ncol=1, framealpha=0.9)
    if args.title:
        ax.set_title(args.title)

    out_path = Path(args.output) if args.output else (
        Path(args.results_dir) / "spmv_speedup_vs_tolerance.pdf")
    fig.tight_layout()
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")

    # ---- text summary -------------------------------------------------------
    print(f"\n{base_label} SpMV on '{executor}', {len(runs)} run(s)")
    if fp32 is not None:
        print(f"  FP32 reference speedup: {fp32:.3f}x")
    if fp16 is not None:
        print(f"  FP16 reference speedup: {fp16:.3f}x")
    head = "  tolerance  " + "".join(f"{s[:22]:>24}" for s in strategies)
    print(head)
    for tol in tolerances:
        line = f"  {tol:<11.0e}"
        for strategy in strategies:
            samples = by_key.get((tol, strategy))
            line += f"{mean(samples):>24.3f}" if samples else f"{'-':>24}"
        print(line)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
