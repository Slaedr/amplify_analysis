#!/usr/bin/env python3

"""AMP[CSR] vs AMP[ELL] speedup over FP64 vs AMP tolerance, side by side.

Companion of ``plot_speedup_error_vs_tol.py`` (same result JSONs, same loader,
same look), but instead of one bar per AMP *strategy* for a single base format
there is one bar per AMP *base format*:

  x axis      AMP tolerance (one group of bars per tolerance)
  y axis      speedup over the format's own FP64 baseline
                  AMP[CSR] over CSR<double>,  AMP[ELL] over ELL<double>
  bars        left: AMP[CSR], right: AMP[ELL]
  dashed line FP32 speedup over FP64 of each format (CSR<float> / ELL<float>),
              drawn in a darker shade of the format's bar colour.  For SpMV
              and FGS it is a horizontal line (the FP32 run does not depend on
              the AMP tolerance); for GMRES, where gmres_tol is swept together
              with the AMP tolerance, it is a marked curve.

No accuracy is plotted.  The benchmark is detected from the top-level key of
each result JSON ("spmv" / "fgs" / "gmres") and one figure is written for
every benchmark that is present for *both* formats, to <prefix>-<bench>.png.
A benchmark found for only one format is skipped with a message.

The AMP strategy (monolithic_classical / independent_buckets) is the same for
both formats; pick it with --strategy (default monolithic_classical).

Usage:
    ./plot_format_compare_speedup.py results-gh200-bf16-csr results-gh200-bf16-ell
    ./plot_format_compare_speedup.py results-gh200-bf16-* -o figs/gh200
        # -> figs/gh200-spmv.png, figs/gh200-fgs.png, ...
    ./plot_format_compare_speedup.py results-gh200-bf16-* --strategy independent_buckets
    ./plot_format_compare_speedup.py results-gh200-bf16-* --bench spmv --large-text
"""

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# Reuse the loader, benchmark table and styling of the speedup/error script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_speedup_error_vs_tol import (  # noqa: E402
    BENCHES,
    CURVE_SCALE,
    LARGE_BAR_LABEL_SIZE,
    LARGE_XTICK_SIZE,
    LEFT_AXIS_COLOR,
    PRETTY_STRATEGY,
    STRATEGY_COLORS,
    add_fitted_legend,
    apply_layout,
    load_runs,
    mean,
)

# base format -> (display name, bar colour, FP32 dashed-line colour).  The bar
# colours are the first two of the strategy palette of the sibling script; the
# dashed lines are darker shades of the same hues so each line reads as
# belonging to its bar series.
FORMATS = {
    "csr": ("CSR", STRATEGY_COLORS[0], "#08306b"),
    "ell": ("ELL", STRATEGY_COLORS[1], "#00441b"),
}
FORMAT_ORDER = ["csr", "ell"]


def tol_label(tol):
    return f"$10^{{{round(math.log10(tol))}}}$"


def output_path(args, bench):
    prefix = args.output or "format_compare_speedup"
    return Path(f"{prefix}-{bench}.png")


def plot_bench(bench, runs_by_fmt, args, layout, out_path):
    """``runs_by_fmt``: base format -> runs of this benchmark (already filtered
    to the chosen strategy).  Draws and saves one figure."""
    scale, bar_fill, max_bar_width, width_factor, height_factor = layout
    spec = BENCHES[bench]
    iterative = spec["iterative"]
    fmts = [f for f in FORMAT_ORDER if f in runs_by_fmt]

    tolerances = sorted({r["tolerance"] for rs in runs_by_fmt.values()
                         for r in rs})

    # (format, tolerance) -> mean AMP / FP32 speedup, and non-converged flags.
    amp = defaultdict(list)
    fp32 = defaultdict(list)
    noconv = defaultdict(bool)
    for f, rs in runs_by_fmt.items():
        for r in rs:
            amp[(f, r["tolerance"])].append(r["amp_speedup"])
            if r["fp32_speedup"] is not None:
                fp32[(f, r["tolerance"])].append(r["fp32_speedup"])
            if r["amp_converged"] is False:
                noconv[(f, r["tolerance"])] = True

    x = np.arange(len(tolerances), dtype=float)
    n_f = len(fmts)
    width = min(bar_fill / n_f, max_bar_width)
    fig_w = min(12.0, max(7.0, 0.9 * len(tolerances) + 2.5)) * width_factor
    fig, ax = plt.subplots(figsize=(fig_w, 5.0 * height_factor))

    vals_all = []
    any_noconv = False
    for i, f in enumerate(fmts):
        name, color, line_color = FORMATS[f]
        offset = (i - (n_f - 1) / 2) * width
        vals, pos, hatches = [], [], []
        for xi, tol in enumerate(tolerances):
            s = amp.get((f, tol))
            if not s:
                continue
            vals.append(mean(s))
            pos.append(x[xi] + offset)
            hatches.append("///" if noconv[(f, tol)] else None)
        if not vals:
            continue
        vals_all += vals
        bars = ax.bar(pos, vals, width, label=f"AMP[{name}]", color=color,
                      edgecolor="black", linewidth=0.6, zorder=3)
        for bar, hatch in zip(bars, hatches):
            if hatch:
                bar.set_hatch(hatch)
                any_noconv = True
        if not args.no_labels:
            ax.bar_label(bars, fmt="%.2f", padding=2,
                         fontsize=(LARGE_BAR_LABEL_SIZE if args.large_text
                                   else 10),
                         rotation=90 if len(tolerances) > 8 else 0,
                         zorder=6,
                         bbox=dict(boxstyle="square,pad=0.1", fc="white",
                                   ec="none", alpha=0.85))

    # FP32 speedup over FP64, one dashed line per format.
    for f in fmts:
        name, _, line_color = FORMATS[f]
        label = f"speedup, {name}<float>"
        if iterative:
            pts = [(xi, mean(fp32[(f, tol)])) for xi, tol in enumerate(tolerances)
                   if fp32.get((f, tol))]
            if not pts:
                continue
            vals_all += [p[1] for p in pts]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=line_color,
                    linestyle="--", linewidth=1.35 * CURVE_SCALE, marker="s",
                    markersize=6 * scale * CURVE_SCALE, markerfacecolor="white",
                    markeredgewidth=1.6 * CURVE_SCALE, zorder=4, label=label)
        else:
            vals = [v for tol in tolerances for v in fp32.get((f, tol), [])]
            if not vals:
                continue
            m = mean(vals)
            vals_all.append(m)
            ax.axhline(m, color=line_color, linestyle="--",
                       linewidth=1.35 * CURVE_SCALE, zorder=4,
                       label=f"{label}  ({m:.2f}x)")

    ax.axhline(1.0, color="gray", linewidth=0.8, alpha=0.5, zorder=4)

    ax.set_xlabel("AMP tolerance")
    ax.set_xticks(x)
    ax.set_xticklabels([tol_label(t) for t in tolerances])
    if args.large_text:
        ax.tick_params(axis="x", labelsize=LARGE_XTICK_SIZE)
    ax.set_xlim(-0.6, len(tolerances) - 0.4)
    ylabel = ("Solve speedup over FP64" if iterative else "Speedup over FP64")
    ax.set_ylabel(ylabel, color=LEFT_AXIS_COLOR)
    ax.set_ylim(0.0, args.ymax if args.ymax else max(vals_all) * 1.3)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", colors=LEFT_AXIS_COLOR)
    ax.spines["left"].set_color(LEFT_AXIS_COLOR)

    handles, labels = ax.get_legend_handles_labels()
    if any_noconv:
        handles.append(Patch(facecolor="white", edgecolor="black", hatch="///"))
        labels.append("not converged")
    _, ncol = add_fitted_legend(fig, ax, handles, labels, frameon=False,
                                columnspacing=1.4, handlelength=2.6)
    if args.title:
        n_rows = math.ceil(len(handles) / ncol)
        ax.set_title(args.title, pad=(14 + 20 * n_rows) * scale)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    # ---- text summary ------------------------------------------------------
    execs = sorted({r["executor"] for rs in runs_by_fmt.values() for r in rs})
    print(f"\n{spec['name']} on {', '.join(execs)}  "
          f"(speedup over each format's own FP64)")
    head = f"  {'tolerance':<11}"
    for f in fmts:
        n = FORMATS[f][0]
        head += f"{'AMP[' + n + ']':>12}{n + '<float>':>13}"
    if len(fmts) == 2:
        head += f"{'ELL/CSR':>10}"
    print(head)
    for tol in tolerances:
        line = f"  {tol:<11.0e}"
        got = {}
        for f in fmts:
            a, s = amp.get((f, tol)), fp32.get((f, tol))
            got[f] = mean(a) if a else None
            mark = "*" if noconv[(f, tol)] else " "
            line += (f"{mean(a):>11.3f}{mark}" if a else f"{'-':>12}")
            line += f"{mean(s):>13.3f}" if s else f"{'-':>13}"
        if len(fmts) == 2:
            c, e = got.get("csr"), got.get("ell")
            line += f"{e / c:>10.2f}" if c and e else f"{'-':>10}"
        print(line)
    if any(noconv.values()):
        print("  (* = AMP run did not converge)")
    print("  (ELL/CSR = AMP[ELL] speedup / AMP[CSR] speedup, each over its own "
          "FP64; not an absolute time ratio)")
    print(f"Saved plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot AMP[CSR] vs AMP[ELL] speedup over FP64 vs AMP "
                    "tolerance, one figure per SpMV / FGS / GMRES benchmark "
                    "found for both formats.")
    parser.add_argument(
        "results_dirs", nargs="+",
        help="Result trees (searched recursively); the CSR and ELL trees can "
             "be given in any order, the base format is read from each JSON.")
    parser.add_argument(
        "--strategy", default="monolithic_classical",
        help="AMP strategy to plot for both formats "
             "(default: monolithic_classical).")
    parser.add_argument(
        "--bench", default=None, choices=list(BENCHES),
        help="Plot only this benchmark (default: every one found).")
    parser.add_argument(
        "--no-labels", action="store_true",
        help="Do not print the speedup value on top of each bar.")
    parser.add_argument("--title", default=None, help="Optional figure title.")
    parser.add_argument(
        "--ymax", type=float, default=None,
        help="Force the speedup axis maximum.")
    parser.add_argument(
        "-o", "--output", "--output-prefix", default=None, metavar="PREFIX",
        help="Output file prefix: each benchmark is written to "
             "<PREFIX>-<bench>.png (default PREFIX: format_compare_speedup "
             "in the current directory).")
    parser.add_argument(
        "--large-text", action="store_true",
        help="Thicker bars, a taller figure and larger text (for slides).")
    parser.add_argument("--dpi", type=int, default=400)
    args = parser.parse_args()
    layout = apply_layout(args.large_text)

    all_runs = []
    for d in args.results_dirs:
        if not Path(d).is_dir():
            raise SystemExit(f"Not a directory: {d}")
        all_runs += load_runs(d)
    if not all_runs:
        raise SystemExit("No SpMV/FGS/GMRES result JSON found under "
                         + ", ".join(args.results_dirs))

    strategies = sorted({r["strategy"] for r in all_runs})
    if args.strategy not in strategies:
        raise SystemExit(f"No runs with strategy '{args.strategy}' "
                         f"(found: {', '.join(strategies)})")

    drew = False
    for bench in BENCHES:
        if args.bench and bench != args.bench:
            continue
        runs = [r for r in all_runs
                if r["bench"] == bench and r["strategy"] == args.strategy
                and r["base_format"] in FORMATS]
        by_fmt = {f: [r for r in runs if r["base_format"] == f]
                  for f in FORMAT_ORDER}
        by_fmt = {f: rs for f, rs in by_fmt.items() if rs}
        if not by_fmt:
            continue
        if len(by_fmt) < 2:
            only = next(iter(by_fmt)).upper()
            missing = "ELL" if only == "CSR" else "CSR"
            print(f"  skipping {bench}: only {only} runs found, no {missing} "
                  f"runs to compare against")
            continue
        plot_bench(bench, by_fmt, args, layout, output_path(args, bench))
        drew = True
    if not drew:
        raise SystemExit("Nothing to plot: no benchmark has runs for both CSR "
                         "and ELL with the chosen strategy.")


if __name__ == "__main__":
    main()
