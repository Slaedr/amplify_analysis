#!/usr/bin/env python3

"""Bar plot of SpMV speedup over a double-precision base format, grouped by
matrix, with one bar per AMP tolerance plus one for a single-precision base
format.

Reads the per-matrix JSON files written by ``benchmark/spmv/spmv`` (via
``benchmark/run_all_benchmarks.sh``, see ``submit_spmv_amp_sweep.sh``),
expecting this layout under ``--results-dir``:

    results-frontier-spmv-csrc-amp/
        tol_6/Janna/Serena.json        # spmv.csrc (FP64) + spmv.amp @ 1e-6
        tol_9/Janna/Serena.json        # spmv.csrc (FP64) + spmv.amp @ 1e-9
        ...
        double_precision/Janna/Serena.json   # spmv.csrc, FP64
        single_precision/Janna/Serena.json   # spmv.csrc, FP32

Matrices are matched across directories by ``problem.name`` (falling back to
the file name). For each matrix (x-axis) one bar is drawn per AMP tolerance
directory (``tol_*``) plus one for the single-precision run.

* AMP bar: ``csrc_time / amp_time`` taken from the *same* tolerance JSON, so
  both were timed back to back on the same node. If a tolerance JSON has no
  ``--base-format`` entry, the FP64 time from ``--double-dir`` is used instead.
* FP32 bar: ``double_precision time / single_precision time``. It needs both
  directories; it is omitted otherwise.

The AMP tolerance value itself is read from each JSON's
``spmv.<amp-format>.amp_config.amp_tolerance`` rather than parsed from the
directory name. If a JSON has no ``amp_tolerance``, it is guessed from a
``tol_<N>`` directory name (as ``1e-<N>``) as a fallback, with a warning.

Error markers (on by default; ``--no-show-error`` hides them) show ``max_relative_norm2`` of the AMP result
(recorded with DETAILED=1). The FP32 run has no error marker: its
``max_relative_norm2`` is measured against an FP32 reference, not FP64.

Usage:
    ./plot_spmv_speedup_vs_tolerance.py \\
        --results-dir results-frontier-spmv-csrc-amp
    ./plot_spmv_speedup_vs_tolerance.py --results-dir <dir> --amp-format ampib
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

# --large-text: thicker bars, a taller figure and bigger text and markers. All text sizes
# (the rcParams above, the explicit font sizes and the marker sizes) are multiplied by
# LARGE_TEXT_SCALE; the figure gets LARGE_BAR_WIDTH_FACTOR times more width
# per bar, bars fill LARGE_GROUP_FILL of the spacing between matrices, and the
# figure is LARGE_HEIGHT_FACTOR times taller.
LARGE_TEXT_SCALE = 1.8
LARGE_GROUP_FILL = 0.92
LARGE_BAR_WIDTH_FACTOR = 1.6
LARGE_HEIGHT_FACTOR = 1.75
# Font size (pt) of the speedup values printed above the bars under
# --large-text. Without --large-text they are 12 pt.
LARGE_BAR_LABEL_SIZE = 23


def apply_layout(large_text):
    """Returns (text_scale, group_fill, inches_per_bar, height_factor) and, for
    --large-text, scales the matplotlib font rcParams."""
    if not large_text:
        return 1.0, GROUP_FILL, INCHES_PER_BAR, 1.0
    plt.rcParams.update({k: v * LARGE_TEXT_SCALE
                         for k, v in BASE_RCPARAMS.items()})
    return (LARGE_TEXT_SCALE, LARGE_GROUP_FILL,
            INCHES_PER_BAR * LARGE_BAR_WIDTH_FACTOR, LARGE_HEIGHT_FACTOR)

# Colour-blind safe palette, one colour per tolerance (cycles if there are
# more tolerances than colours).
TOLERANCE_COLORS = [
    "#2c7bb6", "#d7191c", "#fdae61", "#4daf4a", "#984ea3", "#a6611a",
    "#01665e",
]

TOL_DIR_RE = re.compile(r"tol_(\d+)")

# Error level expected from single-precision arithmetic; drawn as a faint
# dashed line on the error axis when the FP32 bars are plotted.
FP32_EXPECTED_ERROR = 1e-7

# x-axis compactness: the bars of one matrix fill GROUP_FILL of the unit
# spacing between matrices; the figure grows by INCHES_PER_BAR per bar plus
# INCHES_GROUP_PAD per matrix group.
GROUP_FILL = 0.82
INCHES_PER_BAR = 0.36
INCHES_GROUP_PAD = 0.2


def add_fitted_legend(fig, ax, handles, labels, **kwargs):
    """Legend centred above the axes (and on the figure), wrapped onto as few
    rows as it takes to be no wider than the figure, so it never sticks out
    past the axis labels. Extra keyword arguments go to ``ax.legend``."""
    anchor = dict(
        loc="lower center", bbox_to_anchor=(0.5, 1.02), framealpha=0.9,
        bbox_transform=blended_transform_factory(fig.transFigure,
                                                 ax.transAxes))
    anchor.update(kwargs)
    renderer = fig.canvas.get_renderer()
    max_width = 0.98 * fig.get_figwidth() * fig.dpi
    legend = None
    for ncol in range(len(handles), 0, -1):
        if legend is not None:
            legend.remove()
        legend = ax.legend(handles, labels, ncol=ncol, **anchor)
        if legend.get_window_extent(renderer).width <= max_width:
            break
    return legend


def guess_tolerance_from_dirname(tol_dir_name):
    """Best-effort fallback: 'tol_9' -> 1e-9. Returns None if it doesn't
    match the convention."""
    m = TOL_DIR_RE.fullmatch(tol_dir_name)
    return 10.0 ** (-int(m.group(1))) if m else None


def get_amp_tolerance(amp_case, tol_dir_name, json_path):
    """Read amp_tolerance out of a format_case's amp_config, falling back to
    the tolerance subdirectory's name."""
    amp_config = amp_case.get("amp_config", {})
    if "amp_tolerance" in amp_config:
        return float(amp_config["amp_tolerance"])
    guess = guess_tolerance_from_dirname(tol_dir_name)
    if guess is not None:
        print(f"  warning: {json_path} has no amp_config.amp_tolerance; "
              f"guessing {guess:.0e} from directory name '{tol_dir_name}'")
        return guess
    return None


def matrix_name(entry, json_path):
    return entry.get("problem", {}).get("name", json_path.stem)


def read_json(json_path):
    try:
        with open(json_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  skipping {json_path}: {exc}")
        return []


def load_baseline(base_dir, base_format):
    """Returns {matrix: {"time", "error"}} for the `base_format` entry of
    every JSON under base_dir."""
    out = {}
    for json_path in sorted(Path(base_dir).rglob("*.json")):
        for entry in read_json(json_path):
            case = entry.get("spmv", {}).get(base_format)
            if case is None or not case.get("completed", True):
                continue
            name = matrix_name(entry, json_path)
            if name in out:
                print(f"  warning: duplicate baseline for {name!r} in "
                      f"{json_path}; keeping the first")
                continue
            out[name] = {"time": case["time"],
                         "error": case.get("max_relative_norm2")}
    return out


def load_amp_records(results_dir, double, base_format, amp_format, tol_glob):
    """Returns a list of {"matrix", "tolerance", "speedup", "error"} dicts.
    The baseline is the `base_format` entry that sits next to the AMP entry
    in the same JSON (same node, same run); if it is missing, the FP64 time
    from `double` (the {matrix: {...}} dict from load_baseline(), or None)
    is used."""
    results_dir = Path(results_dir)
    tol_dirs = sorted(d for d in results_dir.glob(tol_glob) if d.is_dir())
    if not tol_dirs:
        tol_dirs = [results_dir]

    records = []
    for tol_dir in tol_dirs:
        for json_path in sorted(tol_dir.rglob("*.json")):
            for entry in read_json(json_path):
                spmv = entry.get("spmv", {})
                amp_case = spmv.get(amp_format)
                if amp_case is None or not amp_case.get("completed", True):
                    continue
                name = matrix_name(entry, json_path)
                base_case = spmv.get(base_format)
                if base_case is not None and base_case.get("completed", True):
                    base_time = base_case["time"]
                elif double is not None and name in double:
                    base_time = double[name]["time"]
                else:
                    print(f"  skipping {json_path}: no FP64 '{base_format}' "
                          f"result for {name!r} (neither in the file nor in "
                          f"the double-precision directory)")
                    continue
                tolerance = get_amp_tolerance(amp_case, tol_dir.name,
                                              json_path)
                if tolerance is None:
                    print(f"  skipping {json_path}: no amp_tolerance found "
                          f"for '{amp_format}', and directory name "
                          f"'{tol_dir.name}' doesn't match 'tol_<N>' either")
                    continue
                records.append({
                    "matrix": name,
                    "tolerance": tolerance,
                    "speedup": base_time / amp_case["time"],
                    "error": amp_case.get("max_relative_norm2"),
                })
    return records


# Same grey and hatch as plot_solver_amp_vs_tolerance.py.
SINGLE_COLOR = "#8c8c8c"
SINGLE_HATCH = "//"


def fmt_tol(tol):
    return f"1e{round(math.log10(tol))}"


def mean(vals):
    return sum(vals) / len(vals)


def geomean(vals):
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def main():
    parser = argparse.ArgumentParser(
        description="Plot SpMV speedup over double-precision CSRC, with one "
                    "bar group per matrix and one bar per AMP tolerance plus "
                    "one for single precision.")
    parser.add_argument(
        "--results-dir",
        default="AMPLify/suitesparse/mi250x-spmv-csrc-amp/"
                "results-spmv-csrc-amp",
        help="Root directory holding the double_precision, single_precision "
             "and tol_* subdirectories.")
    parser.add_argument(
        "--double-dir", default="double_precision",
        help="Baseline subdirectory, relative to --results-dir (default: "
             "double_precision).")
    parser.add_argument(
        "--single-dir", default="single_precision",
        help="Single-precision subdirectory, relative to --results-dir "
             "(default: single_precision). Skipped if it doesn't exist.")
    parser.add_argument(
        "--tol-glob", default="tol_*",
        help="Glob, relative to --results-dir, matching one directory per "
             "AMP tolerance (default: 'tol_*').")
    parser.add_argument(
        "--base-format", default="csrc",
        choices=["csrc", "csr", "ell", "cusparse_csr"],
        help="Base sparse format in the double/single runs (default: csrc).")
    parser.add_argument(
        "--amp-format", default="amp", choices=["amp", "ampib"],
        help="Which AMP SpMV strategy to plot (default: amp).")
    parser.add_argument(
        "--no-labels", action="store_true",
        help="Do not print the speedup value on top of each bar.")
    parser.add_argument(
        "--show-error", action=argparse.BooleanOptionalAction, default=True,
        help="Overlay relative-error markers (max_relative_norm2, right "
             "axis); on by default, --no-show-error turns them off.")
    parser.add_argument(
        "--large-text", action="store_true",
        help="Thicker bars, a taller figure and larger text (for slides or "
             "small figures); the default layout is unchanged without it.")
    parser.add_argument(
        "--output", default=None,
        help="Output image path (default: <results-dir>/"
             "spmv_speedup_vs_double_<base-format>_<amp-format>.png).")
    parser.add_argument("--dpi", type=int, default=400)
    args = parser.parse_args()
    scale, group_fill, inches_per_bar, height_factor = \
        apply_layout(args.large_text)

    results_dir = Path(args.results_dir)
    double_dir = results_dir / args.double_dir
    single_dir = results_dir / args.single_dir

    double = None
    if double_dir.is_dir():
        double = load_baseline(double_dir, args.base_format)
        if not double:
            print(f"No '{args.base_format}' results found in {double_dir}")
            raise SystemExit(1)
    else:
        print(f"  note: {double_dir} not found; AMP baselines come from the "
              f"tolerance JSONs, and the FP32 bar is omitted")
    single = (load_baseline(single_dir, args.base_format)
              if single_dir.is_dir() else {})
    if double is not None and not single:
        print(f"  note: no single-precision results in {single_dir}; "
              f"omitting the single-precision bars")

    records = load_amp_records(results_dir, double, args.base_format,
                               args.amp_format, args.tol_glob)
    if not records:
        print(f"No '{args.amp_format}' results found under {results_dir} "
              f"(tolerance glob '{args.tol_glob}')")
        raise SystemExit(1)

    # (matrix, series) -> speedup / error, where series is a tolerance
    # (float) or the string "single".
    by_key = defaultdict(list)
    err_by_key = defaultdict(list)
    for r in records:
        key = (r["matrix"], r["tolerance"])
        by_key[key].append(r["speedup"])
        if r["error"] is not None:
            err_by_key[key].append(r["error"])
    for (matrix, tol), vals in by_key.items():
        if len(vals) > 1:
            print(f"  warning: {len(vals)} results for matrix={matrix!r}, "
                  f"tolerance={tol:.0e}; averaging them")
    if double is not None:
        for matrix, sp in single.items():
            if matrix in double:
                by_key[(matrix, "single")].append(
                    double[matrix]["time"] / sp["time"])
                # No error marker for FP32: its max_relative_norm2 is relative
                # to an FP32 reference, not to FP64.
            else:
                print(f"  skipping single-precision {matrix!r}: no "
                      f"double-precision baseline")

    matrices = sorted({m for m, _ in by_key})
    tolerances = sorted({t for _, t in by_key if t != "single"}, reverse=True)
    # FP32 first (leftmost bar of each group), then the tolerances from
    # loosest to tightest -- same order as plot_solver_amp_vs_tolerance.py.
    series = (["single"] if any(k[1] == "single" for k in by_key) else []) \
        + list(tolerances)
    have_errors = bool(err_by_key) and args.show_error
    if args.show_error and not err_by_key:
        print("  note: no max_relative_norm2 found in any result; skipping "
              "the relative-error axis")

    base_label = args.base_format.upper()
    amp_label = f"{'AMP' if args.amp_format == 'amp' else 'AMPIB'}" \
                f"[{base_label}]"

    x = np.arange(len(matrices), dtype=float)
    n_ser = len(series)
    width = group_fill / n_ser

    fig_w = max(7.0, len(matrices) * (inches_per_bar * n_ser
                                    + INCHES_GROUP_PAD))
    fig, ax = plt.subplots(figsize=(fig_w, 5.5 * height_factor))
    ax2 = ax.twinx() if have_errors else None

    all_vals = []
    tol_index = 0
    for i, ser in enumerate(series):
        offset = (i - (n_ser - 1) / 2) * width
        is_single = ser == "single"
        if is_single:
            color = SINGLE_COLOR
        else:
            # Index among the tolerances only, so FP32 being first does not
            # shift their colours.
            color = TOLERANCE_COLORS[tol_index % len(TOLERANCE_COLORS)]
            tol_index += 1
        if is_single:
            label = "FP32"
        else:
            label = f"$10^{{{round(math.log10(ser))}}}$"
        vals, positions, err_vals, err_positions = [], [], [], []
        for xi, matrix in enumerate(matrices):
            samples = by_key.get((matrix, ser))
            if samples:
                vals.append(mean(samples))
                positions.append(x[xi] + offset)
            err_samples = [e for e in err_by_key.get((matrix, ser), [])
                           if e > 0]  # exact zeros can't go on a log axis
            if err_samples:
                err_vals.append(mean(err_samples))
                err_positions.append(x[xi] + offset)
        if vals:
            all_vals += vals
            bars = ax.bar(
                positions, vals, width, label=label, color=color,
                edgecolor="black", linewidth=0.6, zorder=3,
                hatch=SINGLE_HATCH if is_single else None)
            if not args.no_labels:
                ax.bar_label(bars, fmt="%.2f",
                             fontsize=(LARGE_BAR_LABEL_SIZE if args.large_text
                                       else 12),
                             padding=2,
                             rotation=90 if len(matrices) * n_ser > 10
                             else 0)
        if ax2 is not None and err_vals:
            ax2.plot(err_positions, err_vals, linestyle="none", marker="D",
                     markersize=7 * scale, markerfacecolor=color,
                     markeredgecolor="black", markeredgewidth=0.6, zorder=5)

    ax.axhline(1.0, color="0.55", linewidth=1.0, linestyle="--", zorder=2)
    ax.set_ylabel("Speedup over FP64 " + base_label)
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
        ax2.set_ylabel("Relative error vs. FP64")
        proxy = Line2D([0], [0], linestyle="none", marker="D", markersize=7 * scale,
                       markerfacecolor="0.75", markeredgecolor="black",
                       markeredgewidth=0.6,
                       label="relative error (right axis)")
        handles.append(proxy)
        labels.append(proxy.get_label())
        if "single" in series:
            ax2.axhline(FP32_EXPECTED_ERROR, color="0.55", linewidth=1.0,
                        linestyle="--", zorder=1)
            # axhline does not autoscale: keep the line inside the axis.
            lo, hi = ax2.get_ylim()
            ax2.set_ylim(min(lo, FP32_EXPECTED_ERROR / 3.0),
                         max(hi, FP32_EXPECTED_ERROR * 3.0))
            ref = Line2D([0], [0], color="0.55", linewidth=1.0,
                         linestyle="--",
                         label="FP32 error level (1e%d)"
                               % round(math.log10(FP32_EXPECTED_ERROR)))
            handles.append(ref)
            labels.append(ref.get_label())
    add_fitted_legend(fig, ax, handles, labels)

    fig.tight_layout()
    out_path = Path(args.output) if args.output else (
        results_dir /
        f"spmv_speedup_vs_double_{args.base_format}_{args.amp_format}.png")
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")

    def name(ser):
        return "FP32" if ser == "single" else fmt_tol(ser)

    print(f"\n{amp_label} / FP32 SpMV speedup over FP64 {base_label}, "
          f"{len(matrices)} matrix(es)")
    print("  matrix".ljust(16) + "".join(f"{name(s):>12}" for s in series))
    for matrix in matrices:
        line = f"  {matrix:<14}"
        for ser in series:
            samples = by_key.get((matrix, ser))
            line += f"{mean(samples):>12.2f}" if samples else f"{'-':>12}"
        print(line)
    for ser in series:
        vals = [mean(by_key[(m, ser)]) for m in matrices
                if (m, ser) in by_key]
        if vals:
            print(f"  geometric mean speedup @ {name(ser)}: "
                  f"{geomean(vals):.2f}x")
        errs = [mean(err_by_key[(m, ser)]) for m in matrices
                if (m, ser) in err_by_key]
        n_zero = sum(1 for e in errs if e <= 0)
        errs = [e for e in errs if e > 0]
        if n_zero:
            print(f"  note: {n_zero} matrix(es) at {name(ser)} have exactly "
                  f"zero error; omitted from the error markers and mean")
        if errs:
            print(f"  geometric mean relative error @ {name(ser)}: "
                  f"{geomean(errs):.2e}")
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
