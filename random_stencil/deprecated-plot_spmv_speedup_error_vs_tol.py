#!/usr/bin/env python3

"""AMP SpMV speedup *and* accuracy over the base matrix type vs AMP tolerance.

Same input and same bars as ``plot_spmv_speedup_vs_tol.py`` (left axis), with a
second, logarithmic axis on the right carrying the relative L2 error against the
FP64 base result:

  left axis  (bars + horizontal lines)   speedup over <base><double>
      * one bar per AMP SpMV strategy, grouped per AMP tolerance
      * horizontal line: <base><float> (FP32) speedup
      * horizontal line: <base><half>  (FP16) speedup, if present
      * y = 1 reference for the FP64 base format

  right axis (marked curves, log scale)  relative error vs FP64
      * AMP error vs tolerance -- a single curve, since the strategies only
        change the kernel, not the bin assignment, so they share an accuracy
      * <base><float> (FP32) error, constant in the tolerance
      * <base><half>  (FP16) error, constant in the tolerance

Speedup lines are unmarked and cool-toned; error curves are marked and
warm/neutral-toned, so the two families never read as each other.

Every tolerance / strategy / base-format label is taken from the ``config``
block inside each JSON, so file names do not matter; the tolerance is not
encoded in the name the benchmark writes.

Usage:
    ./plot_spmv_speedup_error_vs_tol.py results-spmv-tol-ell-cuda
    ./plot_spmv_speedup_error_vs_tol.py results/ --base-format csr -o fig.pdf
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
    "legend.fontsize": 12,
})

# --- speedup family: cool tones, no markers ------------------------------
STRATEGY_COLORS = ["#2c7bb6", "#d7191c", "#fdae61", "#4daf4a", "#984ea3"]
FP32_COLOR = "#1a9850"     # green, dashed
FP16_COLOR = "#756bb1"     # purple, dotted

# --- error family: neutral/warm tones, always marked ---------------------
AMP_ERR_COLOR = "#111111"    # near-black, solid + circles
FP32_ERR_COLOR = "#8c510a"   # brown, dash-dot + squares
FP16_ERR_COLOR = "#c51b7d"   # magenta, long dashes + triangles

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
            "fp32_error": fp32.get("rel_error_vs_double") if fp32 else None,
            "fp16_error": fp16.get("rel_error_vs_double") if fp16 else None,
        })
    return runs


def mean(values):
    return sum(values) / len(values)


def geomean(values):
    """Geometric mean of strictly positive values (errors live on a log axis)."""
    positive = [v for v in values if v is not None and v > 0]
    if not positive:
        return None
    return math.exp(mean([math.log(v) for v in positive]))


def main():
    parser = argparse.ArgumentParser(
        description="Plot AMP SpMV speedup and relative error vs AMP tolerance.")
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
        help="Drop the FP16 speedup and error lines even if the runs have them.")
    parser.add_argument(
        "--no-labels", action="store_true",
        help="Do not print the speedup value on top of each bar.")
    parser.add_argument(
        "--per-strategy-error", action="store_true",
        help="Draw one AMP error curve per strategy instead of a single "
             "averaged curve (use to check that they really do coincide).")
    parser.add_argument(
        "--title", default=None, help="Optional figure title.")
    parser.add_argument(
        "--ymax", type=float, default=None,
        help="Force the speedup (left) axis maximum.")
    parser.add_argument(
        "--err-ylim", nargs=2, type=float, default=None, metavar=("LO", "HI"),
        help="Force the error (right) axis limits, e.g. --err-ylim 1e-16 1e-1.")
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output file (default "
             "<results-dir>/spmv_speedup_error_vs_tolerance.png; "
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

    # ---- aggregate: (tolerance, strategy) -> speedup / error ----------------
    speedups = defaultdict(list)
    errors = defaultdict(list)
    for r in runs:
        speedups[(r["tolerance"], r["strategy"])].append(r["amp_speedup"])
        if r["amp_error"] is not None:
            errors[(r["tolerance"], r["strategy"])].append(r["amp_error"])

    tolerances = sorted({t for t, _ in speedups}, reverse=True)  # 1e-4 ... 1e-14
    found_strategies = sorted({s for _, s in speedups},
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

    # ---- reference values ---------------------------------------------------
    fp32_vals = [r["fp32_speedup"] for r in runs if r["fp32_speedup"]]
    fp16_vals = [r["fp16_speedup"] for r in runs if r["fp16_speedup"]]
    fp32 = mean(fp32_vals) if fp32_vals else None
    fp16 = mean(fp16_vals) if fp16_vals and not args.no_fp16 else None
    fp32_err = geomean([r["fp32_error"] for r in runs])
    fp16_err = None if args.no_fp16 else geomean([r["fp16_error"] for r in runs])

    # AMP accuracy: one value per tolerance, pooled over the strategies.
    amp_err_by_tol = {}
    for tol in tolerances:
        pooled = [e for s in strategies for e in errors.get((tol, s), [])]
        amp_err_by_tol[tol] = geomean(pooled)
    # Warn if the strategies do *not* agree -- that would mean the single
    # accuracy curve is hiding something.
    for tol in tolerances:
        per_s = [geomean(errors.get((tol, s), [])) for s in strategies]
        per_s = [e for e in per_s if e]
        if len(per_s) > 1 and max(per_s) > 1.05 * min(per_s):
            print(f"  note: AMP strategies differ in accuracy at tol={tol:.0e} "
                  f"({min(per_s):.3e} .. {max(per_s):.3e}); "
                  f"consider --per-strategy-error")

    # ---- figure -------------------------------------------------------------
    x = np.arange(len(tolerances), dtype=float)
    n_str = len(strategies)
    width = min(0.8 / n_str, 0.35)

    fig_w = min(12.0, max(7.0, 0.9 * len(tolerances) + 2.5))
    fig, ax = plt.subplots(figsize=(fig_w, 5.0))
    ax_err = ax.twinx()
    # The error curves must stay readable where they cross the bars, so the
    # twin axis is drawn on top (with a transparent background).
    ax_err.set_zorder(ax.get_zorder() + 1)
    ax_err.patch.set_visible(False)

    speedup_vals = []
    for i, strategy in enumerate(strategies):
        offset = (i - (n_str - 1) / 2) * width
        vals, positions = [], []
        for xi, tol in enumerate(tolerances):
            samples = speedups.get((tol, strategy))
            if not samples:
                continue
            vals.append(mean(samples))
            positions.append(x[xi] + offset)
        if not vals:
            continue
        speedup_vals += vals
        bars = ax.bar(positions, vals, width,
                      label=PRETTY_STRATEGY.get(strategy, strategy),
                      color=STRATEGY_COLORS[i % len(STRATEGY_COLORS)],
                      edgecolor="black", linewidth=0.6, zorder=3)
        if not args.no_labels:
            ax.bar_label(bars, fmt="%.2f", fontsize=10, padding=2,
                         rotation=90 if len(tolerances) > 8 else 0)

    if fp32 is not None:
        speedup_vals.append(fp32)
        ax.axhline(fp32, color=FP32_COLOR, linestyle="--", linewidth=2.0,
                   zorder=4,
                   label=f"speedup, {base_name}<float>  ({fp32:.2f}x)")
        if len(fp32_vals) > 1:
            lo, hi = min(fp32_vals), max(fp32_vals)
            if hi - lo > 0.01 * fp32:
                ax.axhspan(lo, hi, color=FP32_COLOR, alpha=0.12, zorder=1)
    if fp16 is not None:
        speedup_vals.append(fp16)
        ax.axhline(fp16, color=FP16_COLOR, linestyle=":", linewidth=2.2,
                   zorder=4,
                   label=f"speedup, {base_name}<half>  ({fp16:.2f}x)")

    ax.axhline(1.0, color="black", linewidth=1.0, zorder=4)

    # ---- error curves on the twin axis -------------------------------------
    error_vals = []
    if args.per_strategy_error:
        markers = ["o", "D", "v", "P", "X"]
        for i, strategy in enumerate(strategies):
            pts = [(xi, geomean(errors.get((tol, strategy), [])))
                   for xi, tol in enumerate(tolerances)]
            pts = [(xi, e) for xi, e in pts if e]
            if not pts:
                continue
            error_vals += [e for _, e in pts]
            ax_err.plot([p[0] for p in pts], [p[1] for p in pts],
                        color=AMP_ERR_COLOR, linestyle="-", linewidth=1.8,
                        marker=markers[i % len(markers)], markersize=7,
                        markerfacecolor="white", markeredgewidth=1.6,
                        zorder=6,
                        label=f"error, {PRETTY_STRATEGY.get(strategy, strategy)}")
    else:
        pts = [(xi, amp_err_by_tol[tol]) for xi, tol in enumerate(tolerances)]
        pts = [(xi, e) for xi, e in pts if e]
        if pts:
            error_vals += [e for _, e in pts]
            ax_err.plot([p[0] for p in pts], [p[1] for p in pts],
                        color=AMP_ERR_COLOR, linestyle="-", linewidth=1.8,
                        marker="o", markersize=7, markerfacecolor="white",
                        markeredgewidth=1.6, zorder=6, label="error, AMP")

    if fp32_err:
        error_vals.append(fp32_err)
        ax_err.plot(x, np.full_like(x, fp32_err), color=FP32_ERR_COLOR,
                    linestyle="-.", linewidth=1.8, marker="s", markersize=6,
                    markevery=max(1, len(x) // 5), markerfacecolor="white",
                    markeredgewidth=1.5, zorder=5,
                    label=f"error, {base_name}<float>")
    if fp16_err:
        error_vals.append(fp16_err)
        ax_err.plot(x, np.full_like(x, fp16_err), color=FP16_ERR_COLOR,
                    linestyle=(0, (7, 3)), linewidth=1.8, marker="^",
                    markersize=7, markevery=max(1, len(x) // 5),
                    markerfacecolor="white", markeredgewidth=1.5, zorder=5,
                    label=f"error, {base_name}<half>")

    # ---- axes ---------------------------------------------------------------
    ax.set_xlabel("AMP tolerance")
    ax.set_ylabel(f"Speedup over {base_label}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"$10^{{{round(math.log10(t))}}}$" for t in tolerances])
    ax.set_xlim(-0.6, len(tolerances) - 0.4)
    ax.set_ylim(0.0, args.ymax if args.ymax else max(speedup_vals) * 1.3)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    ax_err.set_yscale("log")
    ax_err.set_ylabel("Relative $\\ell_2$ error vs FP64")
    if args.err_ylim:
        ax_err.set_ylim(*args.err_ylim)
    elif error_vals:
        ax_err.set_ylim(min(error_vals) / 8.0, max(error_vals) * 8.0)
    ax_err.tick_params(axis="y", labelsize=13)

    # One legend for both axes; above the plot, where nothing competes with it.
    handles = ax.get_legend_handles_labels()[0] + \
        ax_err.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + \
        ax_err.get_legend_handles_labels()[1]
    ncol = 2 if len(handles) <= 4 else 3
    ax.legend(handles, labels, loc="lower center",
              bbox_to_anchor=(0.5, 1.01), ncol=ncol, frameon=False,
              columnspacing=1.4, handlelength=2.6)
    if args.title:
        ax.set_title(args.title, pad=48 if len(handles) > 4 else 34)

    out_path = Path(args.output) if args.output else (
        Path(args.results_dir) / "spmv_speedup_error_vs_tolerance.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")

    # ---- text summary -------------------------------------------------------
    print(f"\n{base_label} SpMV on '{executor}', {len(runs)} run(s)")
    if fp32 is not None:
        print(f"  FP32 reference: speedup {fp32:.3f}x, "
              f"error {fp32_err:.3e}" if fp32_err else
              f"  FP32 reference: speedup {fp32:.3f}x")
    if fp16 is not None:
        print(f"  FP16 reference: speedup {fp16:.3f}x, "
              f"error {fp16_err:.3e}" if fp16_err else
              f"  FP16 reference: speedup {fp16:.3f}x")
    head = "  tolerance  " + "".join(f"{s[:22]:>24}" for s in strategies) + \
        f"{'AMP error':>14}"
    print(head)
    for tol in tolerances:
        line = f"  {tol:<11.0e}"
        for strategy in strategies:
            samples = speedups.get((tol, strategy))
            line += f"{mean(samples):>24.3f}" if samples else f"{'-':>24}"
        err = amp_err_by_tol.get(tol)
        line += f"{err:>14.3e}" if err else f"{'-':>14}"
        print(line)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
