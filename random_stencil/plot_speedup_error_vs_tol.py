#!/usr/bin/env python3

"""AMP speedup *and* accuracy over the base matrix type vs AMP tolerance,
for the SpMV, FGS and GMRES benchmarks.

Generalizes ``plot_spmv_speedup_error_vs_tol.py``; the SpMV figure is
unchanged.  The benchmark is detected from the top-level key of each result
JSON ("spmv" / "fgs" / "gmres").

  left axis  (bars + horizontal lines)   speedup over <base><double>
      * one bar per AMP SpMV strategy, grouped per AMP tolerance
      * horizontal line: <base><float> (FP32) speedup
      * horizontal line: <base><half>  (FP16) speedup, if present
      * y = 1 reference for the FP64 base format

  right axis (marked curves, log scale)  relative error
      * AMP error vs tolerance -- a single curve, since the strategies only
        change the kernel, not the bin assignment, so they share an accuracy
      * <base><float> (FP32) error, constant in the tolerance
      * <base><half>  (FP16) error, constant in the tolerance
      * GMRES only: <base><double> error, constant in the tolerance

Per benchmark:

  spmv   time = time_ms  (one SpMV),        error vs the FP64 SpMV result
  fgs    time = time_ms  (one FGS sweep),   error vs the FP64 sweep result
  gmres  time = solve_ms (full solve),      error vs a 1e-14 reference solve;
         since the FP64 solve is itself only converged to gmres_tol, its error
         is non-zero and drawn as well.  A second panel below shows the GMRES
         iteration counts; bars / markers of runs that did not converge are
         hatched / drawn as red crosses.

Speedup lines are unmarked and cool-toned; error curves are marked and
warm/neutral-toned, so the two families never read as each other.

Every benchmark / tolerance / strategy / base-format label is taken from the
JSON itself, so file names do not matter; the tolerance is not encoded in the
name the benchmark writes.

Usage:
    ./plot_speedup_error_vs_tol.py results-tol-ell-cuda/gmres
    ./plot_speedup_error_vs_tol.py results-tol-ell-cuda --bench fgs
    ./plot_speedup_error_vs_tol.py results/ --base-format csr -o fig.pdf
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
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

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
FP64_ERR_COLOR = "#737373"   # grey, dotted + diamonds (GMRES only)
FP32_ERR_COLOR = "#8c510a"   # brown, dash-dot + squares
FP16_ERR_COLOR = "#c51b7d"   # magenta, long dashes + triangles

NOCONV_COLOR = "#e41a1c"     # red crosses for non-converged GMRES runs

PRETTY_STRATEGY = {
    "monolithic_classical": "AMP monolithic",
    "independent_buckets": "AMP independent buckets",
}

# Per-benchmark description of the result JSON.
BENCHES = {
    "spmv": {
        "name": "SpMV",
        "time_key": "time_ms",
        "error_key": "rel_error_vs_double",
        "speedup_label": "Speedup over {base}",
        "error_label": "Relative $\\ell_2$ error vs FP64",
        "iterative": False,
    },
    "fgs": {
        "name": "FGS",
        "time_key": "time_ms",
        "error_key": "rel_error_vs_double",
        "speedup_label": "Speedup over {base}",
        "error_label": "Relative $\\ell_2$ error vs FP64",
        "iterative": False,
    },
    "gmres": {
        "name": "GMRES",
        "time_key": "solve_ms",
        "error_key": "rel_error_vs_exact",
        "speedup_label": "Solve speedup over {base}",
        "error_label": "Relative $\\ell_2$ solution error",
        "iterative": True,
    },
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
        if not isinstance(data, dict):
            continue
        bench = next((b for b in BENCHES if b in data), None)
        if bench is None:
            continue  # per-run config.json and other bystanders
        spec = BENCHES[bench]
        tkey, ekey = spec["time_key"], spec["error_key"]
        cfg = data.get("config", {})
        rows = data[bench]

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

        def get(row, key):
            return row.get(key) if row else None

        base_t = base[tkey]
        runs.append({
            "path": path,
            "bench": bench,
            "base_format": cfg.get("amp_base_format", "?"),
            "base_label": base["format"],
            "executor": cfg.get("executor", "?"),
            "tolerance": float(cfg.get("amp_tolerance", "nan")),
            "strategy": cfg.get("amp_spmv_strategy", "unknown"),
            "amp_speedup": base_t / amp[tkey],
            "fp32_speedup": base_t / fp32[tkey] if fp32 else None,
            "fp16_speedup": base_t / fp16[tkey] if fp16 else None,
            "amp_error": get(amp, ekey),
            "fp64_error": get(base, ekey),
            "fp32_error": get(fp32, ekey),
            "fp16_error": get(fp16, ekey),
            # GMRES only (None elsewhere)
            "amp_iters": get(amp, "iters"),
            "fp64_iters": get(base, "iters"),
            "fp32_iters": get(fp32, "iters"),
            "fp16_iters": get(fp16, "iters"),
            "amp_converged": get(amp, "converged"),
            "fp64_converged": get(base, "converged"),
            "fp32_converged": get(fp32, "converged"),
            "fp16_converged": get(fp16, "converged"),
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


def all_converged(runs, key):
    """False if any run reports key == False; None if no run reports it."""
    flags = [r[key] for r in runs if r[key] is not None]
    if not flags:
        return None
    return all(flags)


def main():
    parser = argparse.ArgumentParser(
        description="Plot AMP speedup and relative error vs AMP tolerance "
                    "for the SpMV / FGS / GMRES benchmarks.")
    parser.add_argument(
        "results_dir", nargs="?", default="results",
        help="Directory holding the sweep results (searched recursively).")
    parser.add_argument(
        "--bench", default=None, choices=sorted(BENCHES),
        help="Which benchmark to plot; required only if the tree mixes several.")
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
        help="Draw one AMP error (and, for GMRES, iteration) curve per "
             "strategy instead of a single averaged curve (use to check that "
             "they really do coincide).")
    parser.add_argument(
        "--no-iters", action="store_true",
        help="GMRES: omit the iteration-count panel.")
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
             "<results-dir>/<bench>_speedup_error_vs_tolerance.png; "
             "extension picks the format).")
    parser.add_argument("--dpi", type=int, default=400)
    args = parser.parse_args()

    runs = load_runs(args.results_dir)
    if not runs:
        raise SystemExit(f"No SpMV/FGS/GMRES result JSON found under "
                         f"{args.results_dir}")

    benches = sorted({r["bench"] for r in runs})
    if args.bench:
        runs = [r for r in runs if r["bench"] == args.bench]
        if not runs:
            raise SystemExit(f"No runs for benchmark '{args.bench}' "
                             f"(found: {', '.join(benches)})")
    elif len(benches) > 1:
        raise SystemExit("Results mix benchmarks "
                         f"({', '.join(benches)}); pass --bench.")
    bench = runs[0]["bench"]
    spec = BENCHES[bench]

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
    iterative = spec["iterative"]
    show_iters = iterative and not args.no_iters

    # ---- aggregate: (tolerance, strategy) -> speedup / error / iters -------
    speedups = defaultdict(list)
    errors = defaultdict(list)
    iters = defaultdict(list)
    noconv = defaultdict(bool)
    for r in runs:
        key = (r["tolerance"], r["strategy"])
        speedups[key].append(r["amp_speedup"])
        if r["amp_error"] is not None:
            errors[key].append(r["amp_error"])
        if r["amp_iters"] is not None:
            iters[key].append(r["amp_iters"])
        if r["amp_converged"] is False:
            noconv[key] = True

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
    # The FP64 base is exactly the reference for SpMV/FGS (error 0, skipped by
    # geomean); for GMRES it is a converged-to-gmres_tol solve, error > 0.
    fp64_err = geomean([r["fp64_error"] for r in runs])
    fp32_err = geomean([r["fp32_error"] for r in runs])
    fp16_err = None if args.no_fp16 else geomean([r["fp16_error"] for r in runs])

    fp64_iters_vals = [r["fp64_iters"] for r in runs if r["fp64_iters"] is not None]
    fp32_iters_vals = [r["fp32_iters"] for r in runs if r["fp32_iters"] is not None]
    fp16_iters_vals = [r["fp16_iters"] for r in runs if r["fp16_iters"] is not None]
    fp64_iters = mean(fp64_iters_vals) if fp64_iters_vals else None
    fp32_iters = mean(fp32_iters_vals) if fp32_iters_vals else None
    fp16_iters = (mean(fp16_iters_vals)
                  if fp16_iters_vals and not args.no_fp16 else None)
    fp64_conv = all_converged(runs, "fp64_converged")
    fp32_conv = all_converged(runs, "fp32_converged")
    fp16_conv = all_converged(runs, "fp16_converged")

    def nc(conv):
        return ", not conv." if conv is False else ""

    # AMP accuracy / iterations: one value per tolerance, pooled over the
    # strategies.
    amp_err_by_tol = {}
    amp_iters_by_tol = {}
    amp_noconv_by_tol = {}
    for tol in tolerances:
        pooled = [e for s in strategies for e in errors.get((tol, s), [])]
        amp_err_by_tol[tol] = geomean(pooled)
        pooled_it = [i for s in strategies for i in iters.get((tol, s), [])]
        amp_iters_by_tol[tol] = mean(pooled_it) if pooled_it else None
        amp_noconv_by_tol[tol] = any(noconv[(tol, s)] for s in strategies)
    # Warn if the strategies do *not* agree -- that would mean the single
    # accuracy / iteration curve is hiding something.
    for tol in tolerances:
        per_s = [geomean(errors.get((tol, s), [])) for s in strategies]
        per_s = [e for e in per_s if e]
        if len(per_s) > 1 and max(per_s) > 1.05 * min(per_s):
            print(f"  note: AMP strategies differ in accuracy at tol={tol:.0e} "
                  f"({min(per_s):.3e} .. {max(per_s):.3e}); "
                  f"consider --per-strategy-error")
        per_s_it = [mean(iters[(tol, s)]) for s in strategies
                    if iters.get((tol, s))]
        if len(per_s_it) > 1 and max(per_s_it) != min(per_s_it):
            print(f"  note: AMP strategies differ in GMRES iterations at "
                  f"tol={tol:.0e} ({min(per_s_it):g} .. {max(per_s_it):g}); "
                  f"consider --per-strategy-error")

    # ---- figure -------------------------------------------------------------
    x = np.arange(len(tolerances), dtype=float)
    n_str = len(strategies)
    width = min(0.8 / n_str, 0.35)

    fig_w = min(12.0, max(7.0, 0.9 * len(tolerances) + 2.5))
    if show_iters:
        fig, (ax, ax_it) = plt.subplots(
            2, 1, sharex=True, figsize=(fig_w, 7.0),
            gridspec_kw={"height_ratios": [3.0, 1.25], "hspace": 0.08})
    else:
        fig, ax = plt.subplots(figsize=(fig_w, 5.0))
        ax_it = None
    ax_err = ax.twinx()
    # The error curves must stay readable where they cross the bars, so the
    # twin axis is drawn on top (with a transparent background).
    ax_err.set_zorder(ax.get_zorder() + 1)
    ax_err.patch.set_visible(False)

    speedup_vals = []
    any_noconv = False
    for i, strategy in enumerate(strategies):
        offset = (i - (n_str - 1) / 2) * width
        vals, positions, hatches = [], [], []
        for xi, tol in enumerate(tolerances):
            samples = speedups.get((tol, strategy))
            if not samples:
                continue
            vals.append(mean(samples))
            positions.append(x[xi] + offset)
            hatches.append("///" if noconv[(tol, strategy)] else None)
        if not vals:
            continue
        speedup_vals += vals
        bars = ax.bar(positions, vals, width,
                      label=PRETTY_STRATEGY.get(strategy, strategy),
                      color=STRATEGY_COLORS[i % len(STRATEGY_COLORS)],
                      edgecolor="black", linewidth=0.6, zorder=3)
        for bar, hatch in zip(bars, hatches):
            if hatch:
                bar.set_hatch(hatch)
                any_noconv = True
        if not args.no_labels:
            ax.bar_label(bars, fmt="%.2f", fontsize=10, padding=2,
                         rotation=90 if len(tolerances) > 8 else 0)

    if fp32 is not None:
        speedup_vals.append(fp32)
        ax.axhline(fp32, color=FP32_COLOR, linestyle="--", linewidth=2.0,
                   zorder=4,
                   label=f"speedup, {base_name}<float>  ({fp32:.2f}x"
                         f"{nc(fp32_conv)})")
        if len(fp32_vals) > 1:
            lo, hi = min(fp32_vals), max(fp32_vals)
            if hi - lo > 0.01 * fp32:
                ax.axhspan(lo, hi, color=FP32_COLOR, alpha=0.12, zorder=1)
    if fp16 is not None:
        speedup_vals.append(fp16)
        ax.axhline(fp16, color=FP16_COLOR, linestyle=":", linewidth=2.2,
                   zorder=4,
                   label=f"speedup, {base_name}<half>  ({fp16:.2f}x"
                         f"{nc(fp16_conv)})")

    ax.axhline(1.0, color="black", linewidth=1.0, zorder=4)

    # ---- error curves on the twin axis -------------------------------------
    markers = ["o", "D", "v", "P", "X"]
    error_vals = []
    if args.per_strategy_error:
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

    if fp64_err:
        error_vals.append(fp64_err)
        ax_err.plot(x, np.full_like(x, fp64_err), color=FP64_ERR_COLOR,
                    linestyle=":", linewidth=1.8, marker="d", markersize=7,
                    markevery=max(1, len(x) // 5), markerfacecolor="white",
                    markeredgewidth=1.5, zorder=5,
                    label=f"error, {base_name}<double>")
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

    # ---- GMRES iteration panel ----------------------------------------------
    if ax_it is not None:
        iter_vals = []

        def plot_amp_iters(pts, marker, label):
            ax_it.plot([p[0] for p in pts], [p[1] for p in pts],
                       color=AMP_ERR_COLOR, linestyle="-", linewidth=1.8,
                       marker=marker, markersize=7, markerfacecolor="white",
                       markeredgewidth=1.6, zorder=6, label=label)
            bad = [p for p in pts if p[2]]
            if bad:
                ax_it.plot([p[0] for p in bad], [p[1] for p in bad],
                           linestyle="none", marker="x", markersize=10,
                           markeredgewidth=2.2, color=NOCONV_COLOR, zorder=7)

        if args.per_strategy_error:
            for i, strategy in enumerate(strategies):
                pts = [(xi, mean(iters[(tol, strategy)]),
                        noconv[(tol, strategy)])
                       for xi, tol in enumerate(tolerances)
                       if iters.get((tol, strategy))]
                if pts:
                    iter_vals += [p[1] for p in pts]
                    plot_amp_iters(pts, markers[i % len(markers)],
                                   PRETTY_STRATEGY.get(strategy, strategy))
        else:
            pts = [(xi, amp_iters_by_tol[tol], amp_noconv_by_tol[tol])
                   for xi, tol in enumerate(tolerances)
                   if amp_iters_by_tol[tol] is not None]
            if pts:
                iter_vals += [p[1] for p in pts]
                plot_amp_iters(pts, "o", "AMP")
        if any(noconv[k] for k in noconv):
            any_noconv = True

        for val, conv, color, ls, lbl in (
                (fp64_iters, fp64_conv, "black", "-", f"{base_name}<double>"),
                (fp32_iters, fp32_conv, FP32_COLOR, "--", f"{base_name}<float>"),
                (fp16_iters, fp16_conv, FP16_COLOR, ":", f"{base_name}<half>")):
            if val is None:
                continue
            iter_vals.append(val)
            ax_it.axhline(val, color=color, linestyle=ls, linewidth=1.6,
                          zorder=4, label=f"{lbl} ({val:g}{nc(conv)})")

        ax_it.set_ylabel("GMRES iters")
        if iter_vals:
            lo, hi = min(iter_vals), max(iter_vals)
            # A reference that ran into gmres_max_iters would flatten the
            # AMP curve on a linear axis; switch to log for wide ranges.
            if lo > 0 and hi / lo > 4.0:
                ax_it.set_yscale("log")
                ax_it.set_ylim(lo / 1.6, hi * 1.6)
            else:
                pad = max(1.0, 0.15 * (hi - lo))
                ax_it.set_ylim(max(0.0, lo - pad), hi + pad)
        ax_it.yaxis.grid(True, linestyle="--", alpha=0.7, linewidth=0.5,
                         zorder=0)
        ax_it.set_axisbelow(True)
        ax_it.yaxis.grid(True, which="minor", linestyle=":", alpha=0.4,
                         linewidth=0.4, zorder=0)
        ax_it.tick_params(axis="y", labelsize=13)
        ax_it.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
                     frameon=False, fontsize=11, handlelength=2.2)

    # ---- axes ---------------------------------------------------------------
    ax_x = ax_it if ax_it is not None else ax
    ax_x.set_xlabel("AMP tolerance")
    ax_x.set_xticks(x)
    ax_x.set_xticklabels([f"$10^{{{round(math.log10(t))}}}$" for t in tolerances])
    ax.set_xlim(-0.6, len(tolerances) - 0.4)
    ax.set_ylabel(spec["speedup_label"].format(base=base_label))
    ax.set_ylim(0.0, args.ymax if args.ymax else max(speedup_vals) * 1.3)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    ax_err.set_yscale("log")
    ax_err.set_ylabel(spec["error_label"])
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
    if any_noconv:
        handles.append(Patch(facecolor="white", edgecolor="black",
                             hatch="///"))
        labels.append("not converged")
        if ax_it is not None:
            handles.append(Line2D([], [], linestyle="none", marker="x",
                                  markersize=9, markeredgewidth=2.0,
                                  color=NOCONV_COLOR))
            labels.append("not converged (iters)")
    ncol = 2 if len(handles) <= 4 else 3
    ax.legend(handles, labels, loc="lower center",
              bbox_to_anchor=(0.5, 1.01), ncol=ncol, frameon=False,
              columnspacing=1.4, handlelength=2.6)
    if args.title:
        n_rows = math.ceil(len(handles) / ncol)
        ax.set_title(args.title, pad=14 + 20 * n_rows)

    out_path = Path(args.output) if args.output else (
        Path(args.results_dir) / f"{bench}_speedup_error_vs_tolerance.png")
    if ax_it is None:
        fig.tight_layout()
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")

    # ---- text summary -------------------------------------------------------
    print(f"\n{base_label} {spec['name']} on '{executor}', {len(runs)} run(s)")
    for name, sp, err, it, conv in (
            ("FP64", 1.0 if iterative else None, fp64_err, fp64_iters,
             fp64_conv),
            ("FP32", fp32, fp32_err, fp32_iters, fp32_conv),
            ("FP16", fp16, fp16_err, fp16_iters, fp16_conv)):
        if sp is None:
            continue
        line = f"  {name} reference: speedup {sp:.3f}x"
        if err:
            line += f", error {err:.3e}"
        if it is not None:
            line += f", {it:g} iters"
        if conv is False:
            line += " (NOT converged)"
        print(line)
    head = "  tolerance  " + "".join(f"{s[:22]:>24}" for s in strategies) + \
        f"{'AMP error':>14}"
    if iterative:
        head += f"{'AMP iters':>12}"
    print(head)
    for tol in tolerances:
        line = f"  {tol:<11.0e}"
        for strategy in strategies:
            samples = speedups.get((tol, strategy))
            if not samples:
                line += f"{'-':>24}"
            elif iterative:
                mark = "*" if noconv[(tol, strategy)] else " "
                line += f"{mean(samples):>23.3f}{mark}"
            else:
                line += f"{mean(samples):>24.3f}"
        err = amp_err_by_tol.get(tol)
        line += f"{err:>14.3e}" if err else f"{'-':>14}"
        if iterative:
            it = amp_iters_by_tol.get(tol)
            line += f"{it:>12g}" if it is not None else f"{'-':>12}"
        print(line)
    if any(noconv.values()):
        print("  (* = AMP GMRES did not converge)")
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
