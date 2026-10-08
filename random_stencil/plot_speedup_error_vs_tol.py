#!/usr/bin/env python3

"""AMP speedup *and* accuracy over the base matrix type vs AMP tolerance,
for the SpMV, FGS and GMRES benchmarks.

Generalizes ``plot_spmv_speedup_error_vs_tol.py``; the SpMV figure is
unchanged.  The benchmark is detected from the top-level key of each result
JSON ("spmv" / "fgs" / "gmres"), and one figure is drawn for every benchmark
found under the results directory (so a tree with spmv, fgs and gmres gives
three figures, one with only spmv and fgs gives two, ...).  The benchmarks are
processed completely independently of each other: each has its own runs,
reference values, axes and output file.  To plot just one, point the script at
that benchmark's sub-directory.  Each figure is written to
<prefix>-<bench>.png (see -o).

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
         iteration counts of the AMP monolithic strategy (the first strategy
         found if monolithic was not run) against the base formats; bars /
         markers of runs that did not converge are hatched / drawn as red
         crosses.

Speedup lines are unmarked and cool-toned; error curves are marked and
warm/neutral-toned, so the two families never read as each other.

Every benchmark / tolerance / strategy / base-format label is taken from the
JSON itself, so file names do not matter; the tolerance is not encoded in the
name the benchmark writes.

Usage:
    ./plot_speedup_error_vs_tol.py results-tol-ell-cuda       # all benchmarks
    ./plot_speedup_error_vs_tol.py results-tol-ell-cuda/gmres # just GMRES
    ./plot_speedup_error_vs_tol.py results/ --base-format csr -o fig
        # -> fig-spmv.png, fig-fgs.png, fig-gmres.png
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
from matplotlib.transforms import blended_transform_factory  # noqa: E402

BASE_RCPARAMS = {
    "font.size": 15,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 15,
    "legend.fontsize": 16,
}
plt.rcParams.update(BASE_RCPARAMS)

# --large-text: thicker bars, a taller figure and bigger text and markers. All
# text sizes (the rcParams above and the explicit font sizes below) and marker
# sizes are multiplied by LARGE_TEXT_SCALE; the figure is LARGE_WIDTH_FACTOR
# times wider, the bars fill more of the space between tolerances
# (BAR_FILL / MAX_BAR_WIDTH), and the figure is LARGE_HEIGHT_FACTOR times
# taller.
BAR_FILL, MAX_BAR_WIDTH = 0.8, 0.35
LARGE_TEXT_SCALE = 1.4
LARGE_BAR_FILL, LARGE_MAX_BAR_WIDTH = 0.92, 0.46
LARGE_WIDTH_FACTOR = 1.4
LARGE_HEIGHT_FACTOR = 1.75
# With --large-text the speedup value on top of each bar and the tolerance
# labels on the x axis are emphasised beyond the general scale-up (font
# sizes in points; the defaults are 10 and 14, see BASE_RCPARAMS).
# Line widths, marker sizes and marker edge widths of the speedup lines and
# error/iteration curves are multiplied by this (bars, grid and the y = 1
# reference are not).
CURVE_SCALE = 1.2
LARGE_BAR_LABEL_SIZE = 18
LARGE_XTICK_SIZE = 22


def apply_layout(large_text):
    """Returns (text_scale, bar_fill, max_bar_width, width_factor,
    height_factor) and, for --large-text, scales the matplotlib font
    rcParams."""
    if not large_text:
        return 1.0, BAR_FILL, MAX_BAR_WIDTH, 1.0, 1.0
    plt.rcParams.update({k: v * LARGE_TEXT_SCALE
                         for k, v in BASE_RCPARAMS.items()})
    return (LARGE_TEXT_SCALE, LARGE_BAR_FILL, LARGE_MAX_BAR_WIDTH,
            LARGE_WIDTH_FACTOR, LARGE_HEIGHT_FACTOR)

# --- speedup family (left axis): blue/green/violet, no markers -----------
# Despite the name, two of these used to be red/orange -- squarely in the
# warm family the error curves below use, which defeats the point of having
# two families. Kept strictly cool/blue-green so a glance at hue alone says
# which axis a curve belongs to.
STRATEGY_COLORS = ["#1f78b4", "#33a02c", "#6a3d9a", "#a6cee3", "#b2df8a"]
FP32_COLOR = "#02818a"     # teal, dashed
FP16_COLOR = "#54278f"     # deep violet, dotted
LEFT_AXIS_COLOR = "#1c4e80"   # dark blue: tints the speedup axis label/ticks

# --- error family (right axis): neutral/warm tones, always marked --------
AMP_ERR_COLOR = "#a63603"    # burnt orange, solid + circles
FP64_ERR_COLOR = "#737373"   # grey, dotted + diamonds (GMRES only)
FP32_ERR_COLOR = "#8c510a"   # brown, dash-dot + squares
FP16_ERR_COLOR = "#c51b7d"   # magenta, long dashes + triangles
RIGHT_AXIS_COLOR = "#8c510a"  # brown: tints the error axis label/ticks

NOCONV_COLOR = "#e41a1c"     # red crosses for non-converged GMRES runs

PRETTY_STRATEGY = {
    "monolithic_classical": "AMP monolithic",
    "independent_buckets": "AMP independent",
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


def add_fitted_legend(fig, ax, handles, labels, **kwargs):
    """Legend horizontally centred on the figure, wrapped onto as few rows as
    it takes to be no wider than the figure, so it never sticks out past the
    axis labels. Extra keyword arguments go to ``ax.legend`` (``loc`` /
    ``bbox_to_anchor`` are interpreted with x in figure and y in axes
    coordinates of ``ax``). Returns (legend, number of columns)."""
    opts = dict(loc="lower center", bbox_to_anchor=(0.5, 1.01),
                bbox_transform=blended_transform_factory(fig.transFigure,
                                                         ax.transAxes))
    opts.update(kwargs)
    renderer = fig.canvas.get_renderer()
    max_width = 0.98 * fig.get_figwidth() * fig.dpi
    legend = None
    for ncol in range(len(handles), 0, -1):
        if legend is not None:
            legend.remove()
        legend = ax.legend(handles, labels, ncol=ncol, **opts)
        if legend.get_window_extent(renderer).width <= max_width:
            break
    return legend, ncol


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


def output_path(args, bench):
    """Output file of one benchmark's figure: <prefix>-<bench>.png, where the
    prefix is --output or, by default, <results-dir>/speedup_error_vs_tolerance."""
    prefix = args.output or str(
        Path(args.results_dir) / "speedup_error_vs_tolerance")
    return Path(f"{prefix}-{bench}.png")


def plot_bench(runs, args, layout, out_path):
    """Aggregate and plot one benchmark (all of ``runs`` must be of the same
    benchmark and base format) and save the figure to ``out_path``."""
    scale, bar_fill, max_bar_width, width_factor, height_factor = layout
    bench = runs[0]["bench"]
    spec = BENCHES[bench]

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

    tolerances = sorted({t for t, _ in speedups})  # 1e-14 ... 1e-2
    found_strategies = sorted({s for _, s in speedups},
                              key=lambda s: (s != "monolithic_classical", s))
    if args.strategies:
        strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
        missing = [s for s in strategies if s not in found_strategies]
        if missing:
            print(f"  skipping {bench}: no runs for strategy/strategies: "
                  f"{', '.join(missing)} "
                  f"(found: {', '.join(found_strategies)})")
            return
    else:
        strategies = found_strategies

    # ---- reference values ---------------------------------------------------
    fp32_vals = [r["fp32_speedup"] for r in runs if r["fp32_speedup"]]
    fp16_vals = [r["fp16_speedup"] for r in runs if r["fp16_speedup"]]
    fp32 = mean(fp32_vals) if fp32_vals else None
    fp16 = mean(fp16_vals) if fp16_vals and not args.no_fp16 else None

    def agg_by_tol(key, agg):
        """Per-tolerance aggregate of runs[key], skipping missing values."""
        out = {}
        for tol in tolerances:
            vals = [r[key] for r in runs
                    if r["tolerance"] == tol and r[key] is not None]
            out[tol] = agg(vals) if vals else None
        return out

    def noconv_by_tol(key):
        return {tol: any(r[key] is False for r in runs if r["tolerance"] == tol)
                for tol in tolerances}

    # GMRES only: gmres_tol is swept together with the AMP tolerance, so the
    # base formats' *speedup* over CSR<double> is not constant either (e.g.
    # once gmres_tol drops below what CSR<float> can resolve, it needs far
    # more iterations and its speedup collapses) -- tracked per tolerance and
    # drawn as a curve rather than a mean line + spread band.
    fp32_speedup_by_tol = agg_by_tol("fp32_speedup", mean)
    fp16_speedup_by_tol = {} if args.no_fp16 else agg_by_tol("fp16_speedup", mean)

    # The FP64 base is exactly the reference for SpMV/FGS (error 0, skipped by
    # geomean); for GMRES it is a converged-to-gmres_tol solve, error > 0.
    # sweep_tolerance.sh ties gmres_tol to the AMP tolerance for GMRES, so
    # there the base formats' error/iters genuinely change across the sweep
    # too -- tracked per tolerance below and drawn as curves, not one line.
    fp64_err = geomean([r["fp64_error"] for r in runs])
    fp32_err = geomean([r["fp32_error"] for r in runs])
    fp16_err = None if args.no_fp16 else geomean([r["fp16_error"] for r in runs])
    fp64_err_by_tol = agg_by_tol("fp64_error", geomean)
    fp32_err_by_tol = agg_by_tol("fp32_error", geomean)
    fp16_err_by_tol = {} if args.no_fp16 else agg_by_tol("fp16_error", geomean)

    fp64_iters_vals = [r["fp64_iters"] for r in runs if r["fp64_iters"] is not None]
    fp32_iters_vals = [r["fp32_iters"] for r in runs if r["fp32_iters"] is not None]
    fp16_iters_vals = [r["fp16_iters"] for r in runs if r["fp16_iters"] is not None]
    fp64_iters = mean(fp64_iters_vals) if fp64_iters_vals else None
    fp32_iters = mean(fp32_iters_vals) if fp32_iters_vals else None
    fp16_iters = (mean(fp16_iters_vals)
                  if fp16_iters_vals and not args.no_fp16 else None)
    fp64_iters_by_tol = agg_by_tol("fp64_iters", mean)
    fp32_iters_by_tol = agg_by_tol("fp32_iters", mean)
    fp16_iters_by_tol = {} if args.no_fp16 else agg_by_tol("fp16_iters", mean)
    fp64_conv = all_converged(runs, "fp64_converged")
    fp32_conv = all_converged(runs, "fp32_converged")
    fp16_conv = all_converged(runs, "fp16_converged")
    fp64_noconv_by_tol = noconv_by_tol("fp64_converged")
    fp32_noconv_by_tol = noconv_by_tol("fp32_converged")
    fp16_noconv_by_tol = {} if args.no_fp16 else noconv_by_tol("fp16_converged")

    def nc(conv):
        return ", not conv." if conv is False else ""

    # AMP accuracy: one value per tolerance, pooled over the strategies.
    # GMRES iterations: AMP monolithic only (first strategy if not present).
    iter_strategy = ("monolithic_classical"
                     if "monolithic_classical" in strategies else strategies[0])
    if iterative and iter_strategy != "monolithic_classical":
        print(f"  note: no monolithic_classical runs; iteration panel shows "
              f"'{iter_strategy}'")
    amp_err_by_tol = {}
    amp_iters_by_tol = {}
    amp_noconv_by_tol = {}
    for tol in tolerances:
        pooled = [e for s in strategies for e in errors.get((tol, s), [])]
        amp_err_by_tol[tol] = geomean(pooled)
        its = iters.get((tol, iter_strategy))
        amp_iters_by_tol[tol] = mean(its) if its else None
        amp_noconv_by_tol[tol] = noconv[(tol, iter_strategy)]
    # Warn if the strategies do *not* agree -- that would mean the single
    # accuracy / iteration curve is hiding something.
    for tol in tolerances:
        per_s = [geomean(errors.get((tol, s), [])) for s in strategies]
        per_s = [e for e in per_s if e]
        if len(per_s) > 1 and max(per_s) > 1.05 * min(per_s):
            print(f"  note: AMP strategies differ in accuracy at tol={tol:.0e} "
                  f"({min(per_s):.3e} .. {max(per_s):.3e}); "
                  f"consider --per-strategy-error")
    iter_diff = []
    for tol in tolerances:
        per_s_it = [mean(iters[(tol, s)]) for s in strategies
                    if iters.get((tol, s))]
        if len(per_s_it) > 1 and max(per_s_it) != min(per_s_it):
            iter_diff.append(f"{tol:.0e}")
    if iter_diff:
        print(f"  note: AMP strategies differ in GMRES iterations at "
              f"tol={', '.join(iter_diff)}; the iteration panel shows "
              f"'{iter_strategy}' only")

    # ---- figure -------------------------------------------------------------
    x = np.arange(len(tolerances), dtype=float)
    n_str = len(strategies)
    width = min(bar_fill / n_str, max_bar_width)

    fig_w = min(12.0, max(7.0, 0.9 * len(tolerances) + 2.5)) * width_factor
    if show_iters:
        fig, (ax, ax_it) = plt.subplots(
            2, 1, sharex=True, figsize=(fig_w, 7.0 * height_factor),
            gridspec_kw={"height_ratios": [3.0, 1.25], "hspace": 0.08})
    else:
        fig, ax = plt.subplots(figsize=(fig_w, 5.0 * height_factor))
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
            ax.bar_label(bars, fmt="%.2f", padding=2,
                         fontsize=(LARGE_BAR_LABEL_SIZE if args.large_text
                                   else 10),
                         rotation=90 if len(tolerances) > 8 else 0)

    if fp32 is not None:
        if iterative:
            # gmres_tol is swept together with the AMP tolerance, so the
            # base formats' speedup is not constant either -- draw the curve
            # instead of a mean line + spread band.
            pts = [(xi, fp32_speedup_by_tol.get(tol))
                   for xi, tol in enumerate(tolerances)]
            pts = [(xi, v) for xi, v in pts if v is not None]
            if pts:
                speedup_vals += [v for _, v in pts]
                ax.plot([p[0] for p in pts], [p[1] for p in pts],
                        color=FP32_COLOR, linestyle="--", linewidth=2.0 * CURVE_SCALE,
                        marker="s", markersize=6 * scale * CURVE_SCALE, markerfacecolor="white",
                        markeredgewidth=1.6 * CURVE_SCALE, zorder=4,
                        label=f"speedup, {base_name}<float>")
        else:
            speedup_vals.append(fp32)
            ax.axhline(fp32, color=FP32_COLOR, linestyle="--", linewidth=2.0 * CURVE_SCALE,
                       zorder=4,
                       label=f"speedup, {base_name}<float>  ({fp32:.2f}x"
                             f"{nc(fp32_conv)})")
            if len(fp32_vals) > 1:
                lo, hi = min(fp32_vals), max(fp32_vals)
                if hi - lo > 0.01 * fp32:
                    ax.axhspan(lo, hi, color=FP32_COLOR, alpha=0.12, zorder=1)
    if fp16 is not None:
        if iterative:
            pts = [(xi, fp16_speedup_by_tol.get(tol))
                   for xi, tol in enumerate(tolerances)]
            pts = [(xi, v) for xi, v in pts if v is not None]
            if pts:
                speedup_vals += [v for _, v in pts]
                ax.plot([p[0] for p in pts], [p[1] for p in pts],
                        color=FP16_COLOR, linestyle=":", linewidth=2.2 * CURVE_SCALE,
                        marker="^", markersize=7 * scale * CURVE_SCALE, markerfacecolor="white",
                        markeredgewidth=1.6 * CURVE_SCALE, zorder=4,
                        label=f"speedup, {base_name}<half>")
        else:
            speedup_vals.append(fp16)
            ax.axhline(fp16, color=FP16_COLOR, linestyle=":", linewidth=2.2 * CURVE_SCALE,
                       zorder=4,
                       label=f"speedup, {base_name}<half>  ({fp16:.2f}x"
                             f"{nc(fp16_conv)})")

    ax.axhline(1.0, color="gray", linewidth=0.8, alpha=0.5, zorder=4)

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
                        color=AMP_ERR_COLOR, linestyle="-", linewidth=1.8 * CURVE_SCALE,
                        marker=markers[i % len(markers)], markersize=7 * scale * CURVE_SCALE,
                        markerfacecolor="white", markeredgewidth=1.6 * CURVE_SCALE,
                        zorder=6,
                        label=f"error, {PRETTY_STRATEGY.get(strategy, strategy)}")
    else:
        pts = [(xi, amp_err_by_tol[tol]) for xi, tol in enumerate(tolerances)]
        pts = [(xi, e) for xi, e in pts if e]
        if pts:
            error_vals += [e for _, e in pts]
            ax_err.plot([p[0] for p in pts], [p[1] for p in pts],
                        color=AMP_ERR_COLOR, linestyle="-", linewidth=1.8 * CURVE_SCALE,
                        marker="o", markersize=7 * scale * CURVE_SCALE, markerfacecolor="white",
                        markeredgewidth=1.6 * CURVE_SCALE, zorder=6, label="error, AMP")

    def by_tol_points(d):
        pts = [(xi, d.get(tol)) for xi, tol in enumerate(tolerances)]
        return [(xi, e) for xi, e in pts if e]

    base_err_curves = (
        (fp64_err_by_tol, fp64_err, FP64_ERR_COLOR, ":", "d", 7,
         f"error, {base_name}<double>"),
        (fp32_err_by_tol, fp32_err, FP32_ERR_COLOR, "-.", "s", 6,
         f"error, {base_name}<float>"),
        (fp16_err_by_tol, fp16_err, FP16_ERR_COLOR, (0, (7, 3)), "^", 7,
         f"error, {base_name}<half>"),
    )
    for by_tol, pooled, color, ls, marker, ms, label in base_err_curves:
        if iterative:
            # gmres_tol is swept together with the AMP tolerance, so the
            # base formats' error is not constant here -- draw the curve.
            pts = by_tol_points(by_tol)
            if not pts:
                continue
            error_vals += [e for _, e in pts]
            ax_err.plot([p[0] for p in pts], [p[1] for p in pts],
                        color=color, linestyle=ls, linewidth=1.8 * CURVE_SCALE,
                        marker=marker, markersize=ms * scale * CURVE_SCALE, markerfacecolor="white",
                        markeredgewidth=1.5 * CURVE_SCALE, zorder=5, label=label)
        elif pooled:
            error_vals.append(pooled)
            ax_err.plot(x, np.full_like(x, pooled), color=color,
                        linestyle=ls, linewidth=1.8 * CURVE_SCALE, marker=marker,
                        markersize=ms * scale * CURVE_SCALE, markevery=max(1, len(x) // 5),
                        markerfacecolor="white", markeredgewidth=1.5 * CURVE_SCALE,
                        zorder=5, label=label)

    # ---- GMRES iteration panel ----------------------------------------------
    if ax_it is not None:
        iter_vals = []
        iters_noconv = False

        def plot_iters_curve(pts, color, marker, ls, label):
            nonlocal iters_noconv
            ax_it.plot([p[0] for p in pts], [p[1] for p in pts],
                       color=color, linestyle=ls, linewidth=1.8 * CURVE_SCALE,
                       marker=marker, markersize=7 * scale * CURVE_SCALE, markerfacecolor="white",
                       markeredgewidth=1.6 * CURVE_SCALE, zorder=6, label=label)
            bad = [p for p in pts if p[2]]
            if bad:
                iters_noconv = True
                ax_it.plot([p[0] for p in bad], [p[1] for p in bad],
                           linestyle="none", marker="x", markersize=10 * scale * CURVE_SCALE,
                           markeredgewidth=2.2 * CURVE_SCALE, color=NOCONV_COLOR, zorder=7)

        pts = [(xi, amp_iters_by_tol[tol], amp_noconv_by_tol[tol])
               for xi, tol in enumerate(tolerances)
               if amp_iters_by_tol[tol] is not None]
        if pts:
            iter_vals += [p[1] for p in pts]
            plot_iters_curve(pts, AMP_ERR_COLOR, "o", "-",
                             PRETTY_STRATEGY.get(iter_strategy, iter_strategy))

        # gmres_tol is swept together with the AMP tolerance here too, so
        # the base formats' iteration counts vary across the sweep as well.
        for by_tol, noconv_d, color, ls, marker, lbl in (
                (fp64_iters_by_tol, fp64_noconv_by_tol, "black", "-", "d",
                 f"{base_name}<double>"),
                (fp32_iters_by_tol, fp32_noconv_by_tol, FP32_COLOR, "--", "s",
                 f"{base_name}<float>"),
                (fp16_iters_by_tol, fp16_noconv_by_tol, FP16_COLOR, ":", "^",
                 f"{base_name}<half>")):
            pts = [(xi, by_tol[tol], noconv_d.get(tol, False))
                   for xi, tol in enumerate(tolerances)
                   if by_tol.get(tol) is not None]
            if not pts:
                continue
            iter_vals += [p[1] for p in pts]
            plot_iters_curve(pts, color, marker, ls, lbl)
        any_noconv = any_noconv or iters_noconv

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
        ax_it.tick_params(axis="y", labelsize=13 * scale)
        # Legend below the panel, under the "AMP tolerance" label, so the
        # iteration panel keeps the full width of the speedup panel.
        it_handles, it_labels = ax_it.get_legend_handles_labels()
        add_fitted_legend(fig, ax_it, it_handles, it_labels,
                          loc="upper center", bbox_to_anchor=(0.5, -0.42),
                          frameon=False, columnspacing=1.6, handlelength=2.4)

    # ---- axes ---------------------------------------------------------------
    ax_x = ax_it if ax_it is not None else ax
    ax_x.set_xlabel("AMP tolerance")
    ax_x.set_xticks(x)
    ax_x.set_xticklabels([f"$10^{{{round(math.log10(t))}}}$" for t in tolerances])
    if args.large_text:
        ax_x.tick_params(axis="x", labelsize=LARGE_XTICK_SIZE)
    ax.set_xlim(-0.6, len(tolerances) - 0.4)
    # Tint each y-axis (label, ticks, spine) to match its color family, so
    # which side a curve belongs to is legible from the axis alone, not just
    # the legend.
    ax.set_ylabel(spec["speedup_label"].format(base=base_label),
                  color=LEFT_AXIS_COLOR)
    ax.set_ylim(0.0, args.ymax if args.ymax else max(speedup_vals) * 1.3)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", colors=LEFT_AXIS_COLOR)
    ax.spines["left"].set_color(LEFT_AXIS_COLOR)

    ax_err.set_yscale("log")
    ax_err.set_ylabel(spec["error_label"], color=RIGHT_AXIS_COLOR)
    if args.err_ylim:
        ax_err.set_ylim(*args.err_ylim)
    elif error_vals:
        ax_err.set_ylim(min(error_vals) / 8.0, max(error_vals) * 8.0)
    ax_err.tick_params(axis="y", labelsize=13 * scale, colors=RIGHT_AXIS_COLOR)
    ax_err.spines["right"].set_color(RIGHT_AXIS_COLOR)
    ax_err.spines["left"].set_visible(False)

    # One legend for both axes; above the plot, where nothing competes with it.
    handles = ax.get_legend_handles_labels()[0] + \
        ax_err.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + \
        ax_err.get_legend_handles_labels()[1]
    if any_noconv:
        handles.append(Patch(facecolor="white", edgecolor="black",
                             hatch="///"))
        labels.append("not converged")
        if ax_it is not None and iters_noconv:
            handles.append(Line2D([], [], linestyle="none", marker="x",
                                  markersize=9 * scale * CURVE_SCALE, markeredgewidth=2.0 * CURVE_SCALE,
                                  color=NOCONV_COLOR))
            labels.append("not converged (iters)")
    _, ncol = add_fitted_legend(fig, ax, handles, labels, frameon=False,
                                columnspacing=1.4, handlelength=2.6)
    if args.title:
        n_rows = math.ceil(len(handles) / ncol)
        ax.set_title(args.title, pad=(14 + 20 * n_rows) * scale)

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
        head += f"{'mono iters' if iter_strategy == 'monolithic_classical' else 'AMP iters':>12}"
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



def main():
    parser = argparse.ArgumentParser(
        description="Plot AMP speedup and relative error vs AMP tolerance "
                    "for every SpMV / FGS / GMRES benchmark found in the "
                    "results directory, one figure per benchmark.")
    parser.add_argument(
        "results_dir", nargs="?", default="results",
        help="Directory holding the sweep results (searched recursively).")
    parser.add_argument(
        "--base-format", default=None, choices=["ell", "csr"],
        help="Which base format to plot; required only if a benchmark's "
             "results mix both.")
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
             "averaged curve (use to check that they really do coincide). "
             "The GMRES iteration panel always shows AMP monolithic only.")
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
        "-o", "--output", "--output-prefix", default=None, metavar="PREFIX",
        help="Output file prefix: each benchmark's figure is written to "
             "<PREFIX>-<bench>.png, e.g. '-o figs/random' gives "
             "figs/random-spmv.png, figs/random-fgs.png and "
             "figs/random-gmres.png (default PREFIX: "
             "<results-dir>/speedup_error_vs_tolerance).")
    parser.add_argument(
        "--large-text", action="store_true",
        help="Thicker bars, a taller figure and larger text and markers (for "
             "slides or small figures); the default layout is unchanged "
             "without it.")
    parser.add_argument("--dpi", type=int, default=400)
    args = parser.parse_args()
    layout = apply_layout(args.large_text)

    all_runs = load_runs(args.results_dir)
    if not all_runs:
        raise SystemExit(f"No SpMV/FGS/GMRES result JSON found under "
                         f"{args.results_dir}")

    # One independent set of runs per benchmark found, in BENCHES order.
    per_bench = {}
    for bench in BENCHES:
        runs = [r for r in all_runs if r["bench"] == bench]
        if not runs:
            continue
        formats = sorted({r["base_format"] for r in runs})
        if args.base_format:
            runs = [r for r in runs if r["base_format"] == args.base_format]
            if not runs:
                print(f"  skipping {bench}: no runs with base format "
                      f"'{args.base_format}' (found: {', '.join(formats)})")
                continue
        elif len(formats) > 1:
            raise SystemExit(f"{bench} results mix base formats "
                             f"({', '.join(formats)}); pass --base-format.")
        per_bench[bench] = runs
    if not per_bench:
        raise SystemExit(f"No runs with base format '{args.base_format}' "
                         f"under {args.results_dir}")
    print(f"Benchmarks found: {', '.join(per_bench)}")

    for bench, runs in per_bench.items():
        out_path = output_path(args, bench)
        plot_bench(runs, args, layout, out_path)


if __name__ == "__main__":
    main()
