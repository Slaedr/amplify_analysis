#!/usr/bin/env python3

"""Bar plot of AMP SpMV speedup over a base format, grouped by matrix and by
AMP tolerance.

Reads the per-matrix JSON files written by ``benchmark/spmv/spmv``, expecting
one subdirectory per AMP-tolerance sweep under ``--results-dir``, e.g.:

    results-mi250x-spmv-csrc/
        tol_9/frontier/hip/SuiteSparse/Janna/CoupCons3D.json
        tol_9/frontier/hip/SuiteSparse/VLSI/ss1.json
        ...
        tol_12/frontier/hip/SuiteSparse/Janna/CoupCons3D.json
        tol_12/frontier/hip/SuiteSparse/VLSI/ss1.json
        ...

For each matrix (x-axis), draws one bar per tolerance subdirectory found,
showing base_time / amp_time.

The AMP tolerance value itself is always read from each JSON's
``spmv.<amp-format>.amp_config.amp_tolerance`` (written directly by
``benchmark/utils/formats.hpp``'s ``write_amp_info``, or hand-patched in for
older runs that predate it) rather than parsed from the subdirectory name, so
the ``tol_*`` naming convention is just an organizational convenience, not a
requirement. If a JSON has no ``amp_tolerance`` at all, the tolerance is
guessed from a ``tol_<N>`` -style directory name (as ``1e-<N>``) as a
fallback, with a warning, so older manually-patched trees still work.

Usage:
    ./plot_spmv_speedup_vs_tolerance.py --results-dir results-mi250x-spmv-csrc
    ./plot_spmv_speedup_vs_tolerance.py --results-dir results-mi250x-spmv-csrc \\
        --amp-format ampib --base-format csrc
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
# more tolerances than colours).
TOLERANCE_COLORS = [
    "#2c7bb6", "#d7191c", "#fdae61", "#4daf4a", "#984ea3", "#a6611a",
    "#01665e",
]

TOL_DIR_RE = re.compile(r"tol_(\d+)")


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


def load_records(results_dir, base_format, amp_format, tol_glob):
    """Returns a list of {"matrix", "tolerance", "speedup", "error"} dicts.
    "error" (the AMP relative error vs. the double baseline,
    max_relative_norm2) is None where a run didn't record it (e.g. it was
    run with --detailed=false)."""
    results_dir = Path(results_dir)
    tol_dirs = sorted(results_dir.glob(tol_glob))
    if not tol_dirs:
        # Not organized into tol_* subdirectories -- maybe --results-dir
        # already points at a single tolerance's own results.
        tol_dirs = [results_dir]

    records = []
    for tol_dir in tol_dirs:
        for json_path in sorted(tol_dir.rglob("*.json")):
            try:
                with open(json_path) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                print(f"  skipping {json_path}: {exc}")
                continue
            for entry in data:
                spmv = entry.get("spmv", {})
                if base_format not in spmv or amp_format not in spmv:
                    continue
                base_case = spmv[base_format]
                amp_case = spmv[amp_format]
                if not base_case.get("completed", True) or not amp_case.get(
                        "completed", True):
                    continue
                tolerance = get_amp_tolerance(amp_case, tol_dir.name,
                                              json_path)
                if tolerance is None:
                    print(f"  skipping {json_path}: no amp_tolerance found "
                          f"for '{amp_format}', and directory name "
                          f"'{tol_dir.name}' doesn't match 'tol_<N>' either")
                    continue
                name = entry.get("problem", {}).get("name", json_path.stem)
                records.append({
                    "matrix": name,
                    "tolerance": tolerance,
                    "speedup": base_case["time"] / amp_case["time"],
                    "error": amp_case.get("max_relative_norm2"),
                })
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Plot AMP SpMV speedup over a base format, with one "
                    "bar group per matrix and one bar per AMP tolerance.")
    parser.add_argument(
        "--results-dir", default="results-mi250x-spmv-csrc",
        help="Root directory holding one subdirectory per AMP-tolerance "
             "sweep (default: results-mi250x-spmv-csrc).")
    parser.add_argument(
        "--tol-glob", default="tol_*",
        help="Glob, relative to --results-dir, matching one directory per "
             "tolerance sweep (default: 'tol_*'). If nothing matches, "
             "--results-dir itself is treated as a single sweep.")
    parser.add_argument(
        "--base-format", default="csrc",
        choices=["csrc", "csr", "ell", "cusparse_csr"],
        help="Base sparse format to compare AMP against (default: csrc).")
    parser.add_argument(
        "--amp-format", default="amp", choices=["amp", "ampib"],
        help="Which AMP SpMV strategy to plot (default: amp).")
    parser.add_argument(
        "--no-labels", action="store_true",
        help="Do not print the speedup value on top of each bar.")
    parser.add_argument(
        "--output", default=None,
        help="Output image path (default: <results-dir>/"
             "spmv_speedup_<base-format>_vs_<amp-format>_by_tolerance.png).")
    parser.add_argument("--dpi", type=int, default=400)
    args = parser.parse_args()

    records = load_records(args.results_dir, args.base_format,
                           args.amp_format, args.tol_glob)
    if not records:
        print(f"No results with both '{args.base_format}' and "
              f"'{args.amp_format}' found under {args.results_dir} "
              f"(tolerance glob '{args.tol_glob}')")
        raise SystemExit(1)

    matrices = sorted({r["matrix"] for r in records})
    tolerances = sorted({r["tolerance"] for r in records}, reverse=True)

    by_key = defaultdict(list)
    err_by_key = defaultdict(list)
    for r in records:
        key = (r["matrix"], r["tolerance"])
        by_key[key].append(r["speedup"])
        if r["error"] is not None:
            err_by_key[key].append(r["error"])
    for (matrix, tolerance), vals in by_key.items():
        if len(vals) > 1:
            print(f"  warning: {len(vals)} results for matrix={matrix!r}, "
                  f"tolerance={tolerance:.0e}; averaging them")
    have_errors = bool(err_by_key)
    if not have_errors:
        print("  note: no max_relative_norm2 found in any result "
              "(runs with --detailed=false don't record it); skipping the "
              "relative-error axis")

    base_label = args.base_format.upper()
    amp_label = ("AMP" if args.amp_format == "amp" else "AMPIB")
    amp_label = f"{amp_label}[{base_label}]"

    x = np.arange(len(matrices), dtype=float)
    n_tol = len(tolerances)
    width = min(0.8 / n_tol, 0.28)

    fig_w = min(24.0, max(7.0, len(matrices) * max(1.8, 0.9 * n_tol)))
    fig, ax = plt.subplots(figsize=(fig_w, 5.5))
    ax2 = ax.twinx() if have_errors else None

    all_vals = []
    all_errs = []
    for i, tol in enumerate(tolerances):
        offset = (i - (n_tol - 1) / 2) * width
        color = TOLERANCE_COLORS[i % len(TOLERANCE_COLORS)]
        vals, positions = [], []
        err_vals, err_positions = [], []
        for xi, matrix in enumerate(matrices):
            samples = by_key.get((matrix, tol))
            if samples:
                vals.append(sum(samples) / len(samples))
                positions.append(x[xi] + offset)
            err_samples = err_by_key.get((matrix, tol))
            if err_samples:
                err_vals.append(sum(err_samples) / len(err_samples))
                err_positions.append(x[xi] + offset)
        if vals:
            all_vals += vals
            exponent = round(math.log10(tol))
            bars = ax.bar(
                positions, vals, width, label=f"$10^{{{exponent}}}$",
                color=color, edgecolor="black", linewidth=0.6, zorder=3)
            if not args.no_labels:
                ax.bar_label(bars, fmt="%.2f", fontsize=9, padding=2,
                             rotation=90 if len(matrices) * n_tol > 10
                             else 0)
        if ax2 is not None and err_vals:
            all_errs += err_vals
            ax2.plot(err_positions, err_vals, linestyle="none", marker="D",
                     markersize=7, markerfacecolor=color,
                     markeredgecolor="black", markeredgewidth=0.6, zorder=5)

    # Faint dashed reference line at 1.0 (no speedup) -- kept light so it
    # doesn't compete with the bars/markers.
    ax.axhline(1.0, color="0.55", linewidth=1.0, linestyle="--", zorder=2)
    ax.set_ylabel(f"Speedup over {base_label}")
    ax.set_xticks(x)
    ax.set_xticklabels(matrices, rotation=30, ha="right")
    ax.set_xlim(-0.6, len(matrices) - 0.4)
    if all_vals:
        ax.set_ylim(0.0, max(all_vals) * 1.25)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    handles, labels = ax.get_legend_handles_labels()
    if ax2 is not None:
        ax2.set_yscale("log")
        ax2.set_ylabel(f"{amp_label} relative error (max_relative_norm2)")
        marker_proxy = Line2D(
            [0], [0], linestyle="none", marker="D", markersize=7,
            markerfacecolor="0.75", markeredgecolor="black",
            markeredgewidth=0.6, label="relative error (right axis)")
        handles.append(marker_proxy)
        labels.append(marker_proxy.get_label())
    # Placed above the axes rather than in a corner: the error markers can
    # land anywhere vertically (their own log-scaled axis), so no inside
    # corner is reliably free of data.
    ax.legend(handles, labels, title="AMP tolerance", ncol=len(handles),
             loc="lower center", bbox_to_anchor=(0.5, 1.02), framealpha=0.9)

    fig.tight_layout()
    out_path = Path(args.output) if args.output else (
        Path(args.results_dir) /
        f"spmv_speedup_{args.base_format}_vs_{args.amp_format}_by_tolerance.png")
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")

    print(f"\n{amp_label} SpMV speedup over {base_label}, "
          f"{len(matrices)} matrix(es), {len(tolerances)} tolerance(s)")
    head = "  matrix".ljust(16) + "".join(
        f"{'1e' + str(round(math.log10(t))):>12}" for t in tolerances)
    print(head)
    for matrix in matrices:
        line = f"  {matrix:<14}"
        for tol in tolerances:
            samples = by_key.get((matrix, tol))
            line += (f"{(sum(samples) / len(samples)):>12.2f}"
                     if samples else f"{'-':>12}")
        print(line)
    for tol in tolerances:
        vals = [sum(by_key[(m, tol)]) / len(by_key[(m, tol)])
                for m in matrices if (m, tol) in by_key]
        if vals:
            gm = math.exp(sum(math.log(v) for v in vals) / len(vals))
            print(f"  geometric mean speedup @ tolerance "
                  f"1e{round(math.log10(tol))}: {gm:.2f}x")
        if have_errors:
            errs = [sum(err_by_key[(m, tol)]) / len(err_by_key[(m, tol)])
                    for m in matrices if (m, tol) in err_by_key]
            if errs:
                gm_err = math.exp(sum(math.log(e) for e in errs)
                                  / len(errs))
                print(f"  geometric mean relative error @ tolerance "
                      f"1e{round(math.log10(tol))}: {gm_err:.2e}")
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
