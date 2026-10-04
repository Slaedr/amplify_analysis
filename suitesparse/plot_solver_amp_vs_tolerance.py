#!/usr/bin/env python3

"""Bar plot of solver speedup and solution error of AMP (at several
tolerances) and of single precision, relative to a fixed-precision FP64
solve, grouped by matrix.

Reads the ``solver_compare_results.json`` files written by ginkgo's
``benchmark/solver/solver_compare`` driver, laid out as
``submit_solver_amp_sweep.sh`` creates them:

    results-frontier-fgs-csrc/
        tol_1e-04/solver_compare_results.json
        tol_1e-06/solver_compare_results.json
        ...
        fp32/solver_compare_results.json

In every file, config_b is the FP64 baseline (<BASE><double>) and config_a
is either AMP[<BASE>] at one tolerance or <BASE><float>. For each matrix
(x-axis) this draws one bar per variant -- FP32 first, then the AMP
tolerances from loosest to tightest -- showing

    speedup = time(<BASE><double>) / time(config_a)

with the relative solution error of config_a with respect to FP64,
||x_a - x_FP64|| / ||x_FP64|| (the driver's ``rel_solution_diff_vs_b``), as
diamonds on a logarithmic right axis. Both numbers come from the same job,
so each speedup is measured against a baseline timed on the same node.

A second figure (disable with --no-iters-plot) shows how the solves
behaved: the same x-axis (matrix x variant), with bars for the number of
solver iterations of config_a, diamonds for the final *true* relative
residual ||b - A x|| / ||b|| of config_a on a logarithmic right axis (the
driver's ``residual_norm``, evaluated against the FP64 matrix, not the
residual in the possibly-reduced-precision storage format), a black tick
across each bar for the FP64 baseline's iterations in the same job (hence
with the same residual goal) and hollow diamonds for the baseline's final
relative residual. Runs that hit the iteration cap without converging are
marked with '*'.

The variant is always read from inside each JSON -- the precision of
config_a, and the ``amp_config.amp_tolerance`` the AMP matrix was actually
built with -- rather than from directory names, so the tree may be arranged
freely. Only if a result carries no AMP tolerance at all is it guessed from a
``tol_<value>`` directory name, with a warning.

Usage:
    ./plot_solver_amp_speedup_vs_tolerance.py --results-dir results-frontier-fgs-csrc
    ./plot_solver_amp_speedup_vs_tolerance.py --results-dir results-frontier-fgs-csrc \\
        --time solve --no-fp32
"""

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

plt.rcParams.update({
    "font.size": 15,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 15,
    "legend.fontsize": 13,
})

# Colour-blind safe palette, one colour per tolerance (cycles if there are
# more tolerances than colours) -- same as plot_spmv_speedup_vs_tolerance.py.
TOLERANCE_COLORS = [
    "#2c7bb6", "#d7191c", "#fdae61", "#4daf4a", "#984ea3", "#a6611a",
    "#01665e",
]
# FP32 gets its own colour, outside the tolerance palette, plus a hatch so it
# stays distinguishable in greyscale.
FP32_COLOR = "#8c8c8c"
FP32_HATCH = "//"

FP32 = "fp32"

# x-axis compactness: the bars of one matrix fill GROUP_FILL of the unit
# spacing between matrices; the figure grows by INCHES_PER_BAR per bar plus
# INCHES_GROUP_PAD per matrix group.
GROUP_FILL = 0.82
INCHES_PER_BAR = 0.36
INCHES_GROUP_PAD = 0.2
TOL_DIR_RE = re.compile(r"tol_([0-9.eE+-]+)")


def merged_config(data, which):
    """The effective settings of config_a/config_b: "common" overridden by
    the configuration's own block, as solver_compare applies them."""
    cfg = dict(data.get("common", {}))
    cfg.update(data.get(f"config_{which}", {}))
    return cfg


def config_precision(data, which):
    """"single" or "double"; newer driver versions record it directly."""
    recorded = data.get(f"precision_{which}")
    if recorded:
        return recorded
    value = merged_config(data, which).get("precision", "double")
    return "single" if value in ("single", "fp32", "float") else "double"


def round_tolerance(tol):
    """AMP stores its tolerance as a float (1e-6 comes back as
    9.99999997e-07); round to two significant digits so equal requests
    group together."""
    return float(f"{tol:.1e}")


def guess_tolerance_from_path(json_path):
    for part in reversed(json_path.parent.parts):
        m = TOL_DIR_RE.fullmatch(part)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
    return None


def matrix_name(row):
    problem = row.get("problem")
    if isinstance(problem, dict) and problem.get("name"):
        return problem["name"]
    name = Path(row.get("matrix", "?")).name
    return name[:-4] if name.endswith(".mtx") else name


def relative_residual(cfg_result, rhs_norm):
    """True final relative residual ||b - A x|| / ||b|| (``residual_norm`` is
    computed against the FP64 matrix), or None if unavailable/non-finite
    (e.g. a diverged FP32 run reports null)."""
    res = cfg_result.get("residual_norm")
    if res is None or not rhs_norm:
        return None
    rel = res / rhs_norm
    return rel if math.isfinite(rel) else None


def load_records(results_dir, pattern, time_key, amp_format_filter):
    """Returns (records, info). Each record is a dict with "matrix",
    "variant" (FP32 or an AMP tolerance), "speedup", "error", "base_time",
    "a_converged", "b_converged" and "source"."""
    results_dir = Path(results_dir)
    files = sorted(results_dir.rglob(pattern))
    info = {"base_formats": set(), "amp_formats": set(), "solvers": set(),
            "used_fallback_error": False, "files": 0}
    records = []
    for json_path in files:
        try:
            with open(json_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  skipping {json_path}: {exc}")
            continue
        if not isinstance(data, dict) or "results" not in data:
            print(f"  skipping {json_path}: not a solver_compare result file")
            continue

        cfg_a = merged_config(data, "a")
        cfg_b = merged_config(data, "b")
        prec_a = config_precision(data, "a")
        if config_precision(data, "b") != "double":
            print(f"  warning: {json_path}: config_b is not double precision; "
                  f"speedups and errors are not relative to FP64")
        format_a = cfg_a.get("formats", "")
        is_amp = format_a in ("amp", "ampib")
        if prec_a == "single":
            variant_kind = FP32
        elif is_amp:
            if amp_format_filter and format_a != amp_format_filter:
                continue
            variant_kind = "amp"
            info["amp_formats"].add(format_a)
        else:
            print(f"  skipping {json_path}: config_a ('{format_a}', "
                  f"{prec_a}) is neither AMP nor single precision")
            continue
        info["base_formats"].add(cfg_b.get("formats", "?"))
        info["solvers"].add((cfg_a.get("solvers", "?"),
                             cfg_a.get("preconditioners", "none")))
        info["files"] += 1

        for row in data["results"]:
            if "error" in row:
                print(f"  {json_path.parent.name}: {matrix_name(row)} "
                      f"failed: {row['error']}")
                continue
            a, b = row.get("a", {}), row.get("b", {})
            if not (a.get("completed") and b.get("completed")):
                for label, c in (("A", a), ("B", b)):
                    if not c.get("completed"):
                        print(f"  {json_path.parent.name}: {matrix_name(row)} "
                              f"[{label}] failed: {c.get('error', '?')}")
                continue
            if variant_kind == FP32:
                variant = FP32
            else:
                tol = a.get("amp_config", {}).get("amp_tolerance",
                                                  cfg_a.get("amp_tolerance"))
                if tol is None:
                    tol = guess_tolerance_from_path(json_path)
                    if tol is None:
                        print(f"  skipping {json_path}: no AMP tolerance "
                              f"recorded and none in the directory name")
                        break
                    print(f"  warning: {json_path} has no amp_tolerance; "
                          f"guessing {tol:.0e} from the directory name")
                variant = round_tolerance(float(tol))
            a_time, b_time = a.get(time_key), b.get(time_key)
            if not a_time or not b_time:
                print(f"  {json_path.parent.name}: {matrix_name(row)}: "
                      f"no {time_key}, skipping")
                continue
            error = row.get("rel_solution_diff_vs_b")
            if error is None and row.get("rel_solution_diff") is not None:
                # Older driver versions only normalized by ||x_a||.
                error = row["rel_solution_diff"]
                info["used_fallback_error"] = True
            rhs_norm = row.get("rhs_norm")
            records.append({
                "iters_a": a.get("iterations"),
                "iters_b": b.get("iterations"),
                "res_a": relative_residual(a, rhs_norm),
                "res_b": relative_residual(b, rhs_norm),
                "matrix": matrix_name(row),
                "variant": variant,
                "speedup": b_time / a_time,
                "error": error,
                "base_time": b_time,
                "a_converged": a.get("converged"),
                "b_converged": b.get("converged"),
                "source": json_path,
            })
    return records, info


def geomean(values):
    values = [v for v in values if v is not None and v > 0]
    if not values:
        return None
    return math.exp(sum(math.log(v) for v in values) / len(values))


def variant_label(variant, base_label):
    if variant == FP32:
        return f"FP32 ({base_label}<float>)"
    return f"$10^{{{round(math.log10(variant))}}}$"


def variant_short(variant):
    if variant == FP32:
        return "fp32"
    return f"1e{round(math.log10(variant))}"


def plot_iterations_residual(records, matrices, variants, base_label,
                             amp_label, args, out_path):
    """Second figure: solver iterations (bars, left axis) and final true
    relative residual (diamonds, log right axis) per matrix and variant, with
    the same-job FP64 baseline as a tick (iterations) / hollow diamond
    (residual)."""
    iters, iters_base, res, res_base = (defaultdict(list) for _ in range(4))
    not_converged = set()
    for r in records:
        key = (r["matrix"], r["variant"])
        if r["iters_a"] is not None:
            iters[key].append(r["iters_a"])
        if r["iters_b"] is not None:
            iters_base[key].append(r["iters_b"])
        if r["res_a"] is not None:
            res[key].append(r["res_a"])
        if r["res_b"] is not None:
            res_base[key].append(r["res_b"])
        if r["a_converged"] is False:
            not_converged.add(key)
    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    iters = {k: mean(v) for k, v in iters.items()}
    iters_base = {k: mean(v) for k, v in iters_base.items()}
    res = {k: geomean(v) for k, v in res.items()}
    res_base = {k: geomean(v) for k, v in res_base.items()}
    res = {k: v for k, v in res.items() if v}
    res_base = {k: v for k, v in res_base.items() if v}

    x = np.arange(len(matrices), dtype=float)
    n_var = len(variants)
    width = GROUP_FILL / n_var
    fig_w = max(7.0, len(matrices) * (INCHES_PER_BAR * n_var
                                    + INCHES_GROUP_PAD))
    fig, ax = plt.subplots(figsize=(fig_w, 6.0))
    ax2 = ax.twinx()

    tol_index = 0
    for i, variant in enumerate(variants):
        offset = (i - (n_var - 1) / 2) * width
        if variant == FP32:
            color, hatch = FP32_COLOR, FP32_HATCH
        else:
            color = TOLERANCE_COLORS[tol_index % len(TOLERANCE_COLORS)]
            hatch = None
            tol_index += 1
        pos, vals, texts = [], [], []
        base_pos, base_vals = [], []
        r_pos, r_vals, rb_pos, rb_vals = [], [], [], []
        for xi, matrix in enumerate(matrices):
            key = (matrix, variant)
            p = x[xi] + offset
            if key in iters:
                pos.append(p)
                vals.append(iters[key])
                texts.append(f"{iters[key]:.0f}"
                             + ("*" if key in not_converged else ""))
            if key in iters_base:
                base_pos.append(p)
                base_vals.append(iters_base[key])
            if key in res:
                r_pos.append(p)
                r_vals.append(res[key])
            if key in res_base:
                rb_pos.append(p)
                rb_vals.append(res_base[key])
        if vals:
            bars = ax.bar(pos, vals, width,
                          label=variant_label(variant, base_label),
                          color=color, hatch=hatch, edgecolor="black",
                          linewidth=0.6, zorder=3, alpha=0.55)
            if not args.no_labels:
                ax.bar_label(bars, labels=texts, fontsize=9, padding=2,
                             rotation=90 if len(matrices) * n_var > 10
                             else 0)
        if base_vals and not args.no_baseline:
            ax.hlines(base_vals, np.array(base_pos) - width / 2,
                      np.array(base_pos) + width / 2, color="black",
                      linewidth=2.2, zorder=4)
        if r_vals:
            ax2.plot(r_pos, r_vals, linestyle="none", marker="D",
                     markersize=7, markerfacecolor="black",
                     markeredgecolor="white", markeredgewidth=0.8, zorder=6)
        if rb_vals and not args.no_baseline:
            ax2.plot(rb_pos, rb_vals, linestyle="none", marker="o",
                     markersize=6, markerfacecolor="none",
                     markeredgecolor="black", markeredgewidth=1.3, zorder=6)

    all_iters = list(iters.values()) + list(iters_base.values())
    ax.set_yscale("log" if not args.linear_iters else "linear")
    if all_iters:
        top = max(all_iters)
        if args.linear_iters:
            ax.set_ylim(0.0, top * 1.25)
        else:
            ax.set_ylim(1.0, top * 12.0)
    ax.set_ylabel("Solver iterations")
    ax.set_xticks(x)
    ax.set_xticklabels(matrices, rotation=30, ha="right")
    ax.set_xlim(-0.5, len(matrices) - 0.5)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    ax2.set_yscale("log")
    ax2.set_ylabel("final relative residual "
                   r"$\|b - Ax\|/\|b\|$")
    all_res = list(res.values()) + list(res_base.values())
    if all_res:
        ax2.set_ylim(min(all_res) / 5.0, max(all_res) * 5.0)

    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], linestyle="none", marker="D",
                          markersize=7, markerfacecolor="black",
                          markeredgecolor="white", markeredgewidth=0.8))
    labels.append("final relative residual (right axis)")
    if not args.no_baseline:
        handles.append(Line2D([0], [0], color="black", linewidth=2.2))
        labels.append(f"{base_label}<double> iterations")
        handles.append(Line2D([0], [0], linestyle="none", marker="o",
                              markersize=6, markerfacecolor="none",
                              markeredgecolor="black", markeredgewidth=1.3))
        labels.append(f"{base_label}<double> residual")
    ncol = min(len(handles), 4) if len(handles) > 6 else len(handles)
    legend = ax.legend(handles, labels, title=f"{amp_label} tolerance",
                       ncol=ncol, loc="lower center",
                       bbox_to_anchor=(0.5, 1.02), framealpha=0.9)
    if not_converged and not args.no_labels:
        ax.annotate("* hit the iteration cap without converging",
                    xy=(1.0, -0.02), xycoords="axes fraction", ha="right",
                    va="top", fontsize=11, color="0.3",
                    xytext=(0, -56), textcoords="offset points")

    fig.tight_layout()
    if args.title:
        fig.canvas.draw()
        top = legend.get_window_extent().transformed(
            fig.transFigure.inverted()).y1
        fig.text(0.5, top + 0.01, args.title, ha="center", va="bottom",
                 fontsize=plt.rcParams["axes.titlesize"])
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")

    print(f"\niterations (* = did not converge; FP64 baseline in brackets)")
    head = "  matrix".ljust(18) + "".join(
        f"{variant_short(v):>16}" for v in variants)
    print(head)
    for matrix in matrices:
        line = f"  {matrix:<16}"
        for v in variants:
            k = (matrix, v)
            if k in iters:
                cell = (f"{iters[k]:.0f}" + ("*" if k in not_converged else "")
                        + (f" [{iters_base[k]:.0f}]" if k in iters_base
                           else ""))
            else:
                cell = "-"
            line += f"{cell:>16}"
        print(line)
    print("\nfinal relative residual (FP64 baseline in brackets)")
    print(head)
    for matrix in matrices:
        line = f"  {matrix:<16}"
        for v in variants:
            k = (matrix, v)
            cell = f"{res[k]:.1e}" if k in res else "-"
            if k in res_base:
                cell += f" [{res_base[k]:.0e}]"
            line += f"{cell:>16}"
        print(line)
    print(f"\nSaved iterations/residual plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot solver speedup and error of AMP (per tolerance) "
                    "and FP32 over a fixed-precision FP64 solve, one bar "
                    "group per matrix, from solver_compare results.")
    parser.add_argument(
        "--results-dir", required=True,
        help="Root directory written by submit_solver_amp_sweep.sh (searched "
             "recursively).")
    parser.add_argument(
        "--pattern", default="solver_compare_results.json",
        help="File name (glob) of the result files to read "
             "(default: solver_compare_results.json).")
    parser.add_argument(
        "--time", default="apply", choices=["apply", "solve", "generate"],
        help="Which time the speedup is computed from: the solver apply "
             "(default), generate+apply ('solve'), or the solver generation "
             "alone.")
    parser.add_argument(
        "--amp-format", default=None, choices=["amp", "ampib"],
        help="Only use AMP results with this SpMV strategy (default: all).")
    parser.add_argument(
        "--no-fp32", action="store_true",
        help="Leave out the single-precision bars.")
    parser.add_argument(
        "--no-labels", action="store_true",
        help="Do not print the speedup value on top of each bar.")
    parser.add_argument(
        "--title", default=None, help="Optional plot title.")
    parser.add_argument(
        "--output", default=None,
        help="Output image path (default: <results-dir>/"
             "<solver>[-<precond>]_speedup_<base>_by_tolerance.png).")
    parser.add_argument(
        "--output-iters", default=None,
        help="Output image path of the iterations/residual plot (default: "
             "<results-dir>/<solver>[-<precond>]_iterations_residual_<base>_"
             "by_tolerance.png).")
    parser.add_argument(
        "--no-iters-plot", action="store_true",
        help="Do not write the iterations/residual plot.")
    parser.add_argument(
        "--no-baseline", action="store_true",
        help="In the iterations/residual plot, leave out the FP64 baseline "
             "reference (iteration ticks and hollow residual markers).")
    parser.add_argument(
        "--linear-iters", action="store_true",
        help="Linear instead of logarithmic iterations axis.")
    parser.add_argument("--dpi", type=int, default=400)
    args = parser.parse_args()

    time_key = f"{args.time}_time"
    records, info = load_records(args.results_dir, args.pattern, time_key,
                                 args.amp_format)
    if args.no_fp32:
        records = [r for r in records if r["variant"] != FP32]
    if not records:
        print(f"No usable results found under {args.results_dir} "
              f"(pattern '{args.pattern}')")
        raise SystemExit(1)

    if len(info["base_formats"]) > 1:
        print(f"  warning: results mix baseline formats "
              f"{sorted(info['base_formats'])}; plotting them together")
    if len(info["amp_formats"]) > 1:
        print(f"  warning: results mix AMP strategies "
              f"{sorted(info['amp_formats'])}; use --amp-format to pick one")
    if len(info["solvers"]) > 1:
        print(f"  warning: results mix solver/preconditioner settings "
              f"{sorted(info['solvers'])}")
    if info["used_fallback_error"]:
        print("  note: some results predate rel_solution_diff_vs_b; using "
              "rel_solution_diff (normalized by ||x_a|| instead of ||x_FP64||)")

    base_format = "/".join(sorted(info["base_formats"]))
    base_label = base_format.upper()
    amp_label = "/".join(sorted(f.upper() for f in info["amp_formats"])) or "AMP"
    solver, precond = sorted(info["solvers"])[0]
    solver_label = solver.upper() + ("" if precond == "none"
                                     else f"+{precond.upper()}")

    matrices = sorted({r["matrix"] for r in records})
    tolerances = sorted({r["variant"] for r in records
                         if r["variant"] != FP32}, reverse=True)
    variants = ([FP32] if any(r["variant"] == FP32 for r in records) else [])
    variants += tolerances

    by_key = defaultdict(list)
    err_by_key = defaultdict(list)
    base_times = defaultdict(list)
    # (matrix, variant) pairs where the FP64 baseline converged but config_a
    # did not (e.g. FP32 asked for a residual below what it can reach): the
    # speedup then compares an iteration-capped run with a converged one.
    lost_convergence = set()
    for r in records:
        key = (r["matrix"], r["variant"])
        by_key[key].append(r["speedup"])
        if r["b_converged"] and r["a_converged"] is False:
            lost_convergence.add(key)
        if r["error"] is not None:
            err_by_key[key].append(r["error"])
        base_times[r["matrix"]].append(r["base_time"])
    for (matrix, variant), vals in by_key.items():
        if len(vals) > 1:
            print(f"  warning: {len(vals)} results for matrix={matrix!r}, "
                  f"variant={variant_short(variant)}; averaging them")
    # Every job re-times the FP64 baseline; a large spread between jobs means
    # the speedups compare against noticeably different baselines.
    for matrix, times in sorted(base_times.items()):
        if len(times) > 1 and min(times) > 0 and max(times) / min(times) > 1.10:
            print(f"  note: {base_label}<double> time for {matrix} varies "
                  f"{max(times) / min(times):.2f}x across jobs "
                  f"({min(times):.3e}..{max(times):.3e} s)")

    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    speedup = {k: mean(v) for k, v in by_key.items()}
    # Errors span orders of magnitude, so repeated samples are combined
    # geometrically; an exact 0 (AMP kept everything in FP64) stays 0.
    error = {}
    for k, v in err_by_key.items():
        error[k] = 0.0 if all(e == 0 for e in v) else geomean(v)
    have_errors = bool(error)

    x = np.arange(len(matrices), dtype=float)
    n_var = len(variants)
    width = GROUP_FILL / n_var
    fig_w = max(7.0, len(matrices) * (INCHES_PER_BAR * n_var
                                    + INCHES_GROUP_PAD))
    fig, ax = plt.subplots(figsize=(fig_w, 5.5))
    ax2 = ax.twinx() if have_errors else None

    nonzero_errors = [e for e in error.values() if e and e > 0]
    zero_floor = (min(nonzero_errors) / 10.0) if nonzero_errors else 1e-16
    have_zero_errors = any(e == 0 for e in error.values())

    all_vals = []
    tol_index = 0
    for i, variant in enumerate(variants):
        offset = (i - (n_var - 1) / 2) * width
        if variant == FP32:
            color, hatch = FP32_COLOR, FP32_HATCH
        else:
            color = TOLERANCE_COLORS[tol_index % len(TOLERANCE_COLORS)]
            hatch = None
            tol_index += 1
        vals, positions, bar_texts = [], [], []
        err_pos, err_vals, zero_pos = [], [], []
        for xi, matrix in enumerate(matrices):
            key = (matrix, variant)
            if key in speedup:
                vals.append(speedup[key])
                positions.append(x[xi] + offset)
                bar_texts.append(f"{speedup[key]:.2f}"
                                 + ("*" if key in lost_convergence else ""))
            if key in error:
                if error[key] > 0:
                    err_pos.append(x[xi] + offset)
                    err_vals.append(error[key])
                else:
                    zero_pos.append(x[xi] + offset)
        if vals:
            all_vals += vals
            bars = ax.bar(positions, vals, width,
                          label=variant_label(variant, base_label),
                          color=color, hatch=hatch, edgecolor="black",
                          linewidth=0.6, zorder=3)
            if not args.no_labels:
                ax.bar_label(bars, labels=bar_texts, fontsize=9, padding=2,
                             rotation=90 if len(matrices) * n_var > 10
                             else 0)
        if ax2 is not None:
            if err_vals:
                ax2.plot(err_pos, err_vals, linestyle="none", marker="D",
                         markersize=7, markerfacecolor=color,
                         markeredgecolor="black", markeredgewidth=0.6,
                         zorder=5)
            if zero_pos:
                ax2.plot(zero_pos, [zero_floor] * len(zero_pos),
                         linestyle="none", marker="v", markersize=8,
                         markerfacecolor=color, markeredgecolor="black",
                         markeredgewidth=0.6, zorder=5)

    # Faint dashed reference line at 1.0 (no speedup).
    ax.axhline(1.0, color="0.55", linewidth=1.0, linestyle="--", zorder=2)
    time_desc = {"apply": "", "solve": "\n(generate + apply time)",
                 "generate": "\n(generate time)"}[args.time]
    ax.set_ylabel(f"Speedup over {base_label}<double>{time_desc}")
    ax.set_xticks(x)
    ax.set_xticklabels(matrices, rotation=30, ha="right")
    ax.set_xlim(-0.5, len(matrices) - 0.5)
    if all_vals:
        ax.set_ylim(0.0, max(all_vals) * 1.25)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    handles, labels = ax.get_legend_handles_labels()
    if ax2 is not None:
        ax2.set_yscale("log")
        ax2.set_ylabel(f"relative error vs {base_label}<double>")
        marker_proxy = Line2D(
            [0], [0], linestyle="none", marker="D", markersize=7,
            markerfacecolor="0.75", markeredgecolor="black",
            markeredgewidth=0.6, label="relative error (right axis)")
        handles.append(marker_proxy)
        labels.append(marker_proxy.get_label())
        if have_zero_errors:
            zero_proxy = Line2D(
                [0], [0], linestyle="none", marker="v", markersize=8,
                markerfacecolor="0.75", markeredgecolor="black",
                markeredgewidth=0.6, label="error = 0 (at axis floor)")
            handles.append(zero_proxy)
            labels.append(zero_proxy.get_label())
    # Placed above the axes rather than in a corner: the error markers can
    # land anywhere vertically (their own log-scaled axis), so no inside
    # corner is reliably free of data.
    ncol = min(len(handles), 4) if len(handles) > 6 else len(handles)
    legend = ax.legend(handles, labels, title=f"{amp_label} tolerance",
                       ncol=ncol, loc="lower center",
                       bbox_to_anchor=(0.5, 1.02), framealpha=0.9)

    if lost_convergence and not args.no_labels:
        ax.annotate(f"* did not converge (the {base_label}<double> "
                    f"baseline did)", xy=(1.0, -0.02),
                    xycoords="axes fraction", ha="right", va="top",
                    fontsize=11, color="0.3",
                    xytext=(0, -56), textcoords="offset points")

    fig.tight_layout()
    if args.title:
        # Above the legend, which itself sits above the axes.
        fig.canvas.draw()
        top = legend.get_window_extent().transformed(
            fig.transFigure.inverted()).y1
        fig.text(0.5, top + 0.01, args.title, ha="center", va="bottom",
                 fontsize=plt.rcParams["axes.titlesize"])
    stem = solver + ("" if precond == "none" else f"-{precond}")
    out_path = Path(args.output) if args.output else (
        Path(args.results_dir) /
        f"{stem}_speedup_{base_format.replace('/', '-')}_by_tolerance.png")
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")

    print(f"\n{solver_label}: speedup over {base_label}<double> "
          f"({args.time} time), {len(matrices)} matrix(es), "
          f"{len(variants)} variant(s)")
    head = "  matrix".ljust(18) + "".join(
        f"{variant_short(v):>12}" for v in variants)
    print(head)
    for matrix in matrices:
        line = f"  {matrix:<16}"
        for v in variants:
            s = speedup.get((matrix, v))
            line += f"{s:>12.2f}" if s is not None else f"{'-':>12}"
        print(line)
    if have_errors:
        print(f"\nrelative error vs {base_label}<double>")
        print(head)
        for matrix in matrices:
            line = f"  {matrix:<16}"
            for v in variants:
                e = error.get((matrix, v))
                line += f"{e:>12.2e}" if e is not None else f"{'-':>12}"
            print(line)
    print()
    for v in variants:
        gm = geomean([speedup[(m, v)] for m in matrices if (m, v) in speedup])
        msg = f"  {variant_short(v):>6}: geometric mean speedup "
        msg += f"{gm:.2f}x" if gm else "-"
        if have_errors:
            gm_err = geomean([error[(m, v)] for m in matrices
                              if (m, v) in error])
            if gm_err:
                msg += f", geometric mean relative error {gm_err:.2e}"
            n_zero = sum(1 for m in matrices if error.get((m, v)) == 0)
            if n_zero:
                msg += f" (excluding {n_zero} exact)"
        print(msg)
    if lost_convergence:
        print("\n  did not converge although the FP64 baseline did: " +
              ", ".join(f"{m} @ {variant_short(v)}"
                        for m, v in sorted(lost_convergence,
                                           key=lambda k: (k[0], str(k[1])))))
    print(f"\nSaved plot to {out_path}")

    if not args.no_iters_plot:
        iters_path = Path(args.output_iters) if args.output_iters else (
            Path(args.results_dir) /
            f"{stem}_iterations_residual_{base_format.replace('/', '-')}"
            f"_by_tolerance.png")
        plot_iterations_residual(records, matrices, variants, base_label,
                                 amp_label, args, iters_path)


if __name__ == "__main__":
    main()
