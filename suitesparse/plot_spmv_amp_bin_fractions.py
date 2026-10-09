#!/usr/bin/env python3

"""Stacked bar plot of the AMP bin distribution (fraction of nonzeros stored
in each precision) per matrix and AMP tolerance.

Reads the per-matrix JSON files written by ``benchmark/spmv/spmv`` (via
``submit_spmv_amp_sweep.sh``), in the same layout as
plot_spmv_speedup_vs_tolerance.py:

    results-frontier-bf16-spmv-csrc-amp/
        tol_6/Janna/Serena.json     # spmv.amp.amp_bins @ 1e-6
        tol_8/Janna/Serena.json
        ...

For each matrix (x-axis group) one stacked bar is drawn per AMP tolerance
(``tol_*``), loosest tolerance first. Each bar stacks the fraction of the
matrix' nonzeros held in every bin (``spmv.<amp-format>.amp_bins.bin_*``:
FP64 at the bottom, then FP32, then BF16) up to a total of 1.0. Only the
``amp`` format is read by default; ``ampib`` has the same bin counts.

By default the fractions are relative to the matrix' total number of nonzeros
(``nonzeros`` in the JSON). If the bins do not account for all of them -- the
bin counters of some matrices sum to less than ``nonzeros`` -- the remainder
is drawn as a grey "dropped" segment (entries AMP dropped) so every bar still reaches 1.0, and
a warning is printed. ``--normalize bins`` instead divides by the sum of the
bin counts.

The AMP tolerance is read from ``amp_config.amp_tolerance`` in each JSON (the
``tol_<N>`` directory name, as 1e-<N>, is only a fallback). Matrices that
have no ``amp`` entry, or where ``completed`` is false, are skipped.

Usage:
    ./plot_spmv_amp_bin_fractions.py \\
        --results-dir results-frontier-bf16-spmv-csrc-amp
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
from matplotlib.patches import Patch  # noqa: E402

BASE_RCPARAMS = {
    "font.size": 15,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 15,
    "legend.fontsize": 13,
}
plt.rcParams.update(BASE_RCPARAMS)

# Same bin naming as plot_spmv_amp_bins.py.
BIN_LABELS = {"bin_0": "FP64", "bin_1": "FP32", "bin_2": "BF16"}
# Colour-blind safe, one colour per precision (cycles for unknown bins).
BIN_COLORS = {"bin_0": "#2c7bb6", "bin_1": "#fdae61", "bin_2": "#d7191c"}
FALLBACK_COLORS = ["#4daf4a", "#984ea3", "#a6611a", "#01665e"]
OTHER_COLOR = "#d9d9d9"
OTHER_HATCH = "//"

# x-axis compactness: the bars of one matrix fill GROUP_FILL of the unit
# spacing between matrices; the figure grows by INCHES_PER_BAR per bar plus
# INCHES_GROUP_PAD per matrix group.
GROUP_FILL = 0.82
INCHES_PER_BAR = 0.36
INCHES_GROUP_PAD = 0.2

# --large-text: thicker bars, a taller figure and bigger text. All text sizes
# (the rcParams above and the explicit font sizes) are multiplied by
# LARGE_TEXT_SCALE; the figure gets LARGE_BAR_WIDTH_FACTOR times more width
# per bar, bars fill LARGE_GROUP_FILL of the spacing between matrices, and the
# figure is LARGE_HEIGHT_FACTOR times taller.
LARGE_TEXT_SCALE = 1.6
LARGE_GROUP_FILL = 0.92
LARGE_BAR_WIDTH_FACTOR = 1.6
LARGE_HEIGHT_FACTOR = 1.75
# Absolute font sizes of the in-bar percentages and the legend under
# --large-text (the defaults are 8 pt and legend.fontsize / 13 pt).
LARGE_BAR_LABEL_SIZE = 20
LARGE_LEGEND_SIZE = 26
LARGE_TOL_TICK_SIZE = 22      # tolerance labels under the bars (default 9 pt)
LARGE_MATRIX_NAME_SIZE = 18   # matrix names under each group (default 14 pt)
LARGE_MIN_FIG_WIDTH = 16.0


def apply_layout(large_text):
    """Returns (text_scale, group_fill, inches_per_bar, height_factor) and, for
    --large-text, scales the matplotlib font rcParams."""
    if not large_text:
        return 1.0, GROUP_FILL, INCHES_PER_BAR, 1.0
    plt.rcParams.update({k: v * LARGE_TEXT_SCALE
                         for k, v in BASE_RCPARAMS.items()})
    return (LARGE_TEXT_SCALE, LARGE_GROUP_FILL,
            INCHES_PER_BAR * LARGE_BAR_WIDTH_FACTOR, LARGE_HEIGHT_FACTOR)


TOL_DIR_RE = re.compile(r"tol_(\d+)")


def read_json(json_path):
    try:
        with open(json_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  skipping {json_path}: {exc}")
        return []


def tolerance_of(amp_case, tol_dir_name, json_path):
    tol = amp_case.get("amp_config", {}).get("amp_tolerance")
    if tol is not None:
        # AMP stores the tolerance as a float (1e-8 -> 9.99999994e-09).
        return float(f"{float(tol):.1e}")
    m = TOL_DIR_RE.fullmatch(tol_dir_name)
    if m:
        print(f"  warning: {json_path} has no amp_config.amp_tolerance; "
              f"guessing 1e-{m.group(1)} from the directory name")
        return 10.0 ** (-int(m.group(1)))
    return None


def load_records(results_dir, amp_format, tol_glob):
    """Returns a list of {"matrix", "tolerance", "bins", "nnz"} dicts, where
    "bins" maps bin_<i> to its nonzero count."""
    results_dir = Path(results_dir)
    tol_dirs = sorted(d for d in results_dir.glob(tol_glob) if d.is_dir())
    if not tol_dirs:
        print(f"No '{tol_glob}' directories under {results_dir}")
        return []
    records = []
    for tol_dir in tol_dirs:
        for json_path in sorted(tol_dir.rglob("*.json")):
            for entry in read_json(json_path):
                case = entry.get("spmv", {}).get(amp_format)
                if case is None or not case.get("completed", True):
                    continue
                amp_bins = case.get("amp_bins")
                if not amp_bins:
                    print(f"  {json_path}: no amp_bins, skipping")
                    continue
                tol = tolerance_of(case, tol_dir.name, json_path)
                if tol is None:
                    print(f"  {json_path}: no AMP tolerance, skipping")
                    continue
                bins = {k: v for k, v in amp_bins.items()
                        if k.startswith("bin_") and isinstance(v, (int, float))}
                name = entry.get("problem", {}).get("name", json_path.stem)
                records.append({
                    "matrix": name,
                    "tolerance": tol,
                    "bins": bins,
                    "nnz": entry.get("nonzeros",
                                     entry.get("problem", {}).get("nonzeros")),
                })
    return records


def tol_label(tol):
    return f"1e-{round(-math.log10(tol))}"


def main():
    parser = argparse.ArgumentParser(
        description="Stacked bar plot of the fraction of nonzeros in each AMP "
                    "bin, per matrix and AMP tolerance.")
    parser.add_argument(
        "--results-dir", required=True,
        help="Root directory with tol_*/<group>/<matrix>.json files.")
    parser.add_argument(
        "--amp-format", default="amp", choices=["amp", "ampib"],
        help="Which AMP format entry to read (default: amp; ampib has the "
             "same bin counts).")
    parser.add_argument(
        "--tol-glob", default="tol_*",
        help="Glob of the tolerance sub-directories (default: tol_*).")
    parser.add_argument(
        "--normalize", default="nnz", choices=["nnz", "bins"],
        help="Divide the bin counts by the matrix' total nonzeros (default; "
             "a grey segment shows dropped nonzeros, i.e. not in any bin) "
             "or by the sum of the bin counts.")
    parser.add_argument(
        "--no-labels", action="store_true",
        help="Do not print the segment percentages.")
    parser.add_argument("--title", default=None, help="Optional plot title.")
    parser.add_argument(
        "--large-text", action="store_true",
        help="Thicker bars, a taller figure and larger text (for slides or "
             "small figures); the default layout is unchanged without it.")
    parser.add_argument(
        "--output", default=None,
        help="Output image path (default: <results-dir>/"
             "spmv_amp_bin_fractions_<format>.png).")
    parser.add_argument("--dpi", type=int, default=400)
    args = parser.parse_args()
    scale, group_fill, inches_per_bar, height_factor = \
        apply_layout(args.large_text)

    records =load_records(args.results_dir, args.amp_format, args.tol_glob)
    if not records:
        print(f"No AMP bin data found under {args.results_dir}")
        raise SystemExit(1)

    matrices = sorted({r["matrix"] for r in records})
    tolerances = sorted({r["tolerance"] for r in records}, reverse=True)
    bin_keys = sorted({k for r in records for k in r["bins"]})

    by_key = {}
    for r in records:
        key = (r["matrix"], r["tolerance"])
        if key in by_key:
            print(f"  warning: several results for {r['matrix']!r} at "
                  f"{tol_label(r['tolerance'])}; keeping the first")
            continue
        by_key[key] = r

    # fractions[key] = {bin_key: fraction, "other": fraction}
    fractions = {}
    short = []
    for key, r in sorted(by_key.items()):
        total_bins = sum(r["bins"].values())
        denom = r["nnz"] if (args.normalize == "nnz" and r["nnz"]) \
            else total_bins
        if not denom:
            continue
        fr = {k: r["bins"].get(k, 0) / denom for k in bin_keys}
        other = max(0.0, 1.0 - sum(fr.values()))
        fr["other"] = other if other > 1e-9 else 0.0
        fractions[key] = fr
        if fr["other"] > 1e-6:
            short.append((key, fr["other"]))
        if sum(fr.values()) > 1.0 + 1e-6:
            print(f"  warning: bins of {key[0]} @ {tol_label(key[1])} exceed "
                  f"the total nonzeros (sum = {sum(fr.values()):.4f})")
    have_other = bool(short)
    if short:
        worst = max(short, key=lambda s: s[1])
        print(f"  warning: for {len(short)} matrix/tolerance pair(s) the bins "
              f"cover less than all nonzeros (up to "
              f"{worst[1] * 100:.1f}% missing, {worst[0][0]} @ "
              f"{tol_label(worst[0][1])}); shown as a grey segment")

    n_tol = len(tolerances)
    # Bars of one matrix sit next to each other; groups are 1 unit apart.
    bar_w = group_fill / n_tol
    # The enlarged legend sits above the axes in a single row; keep the figure
    # at least as wide as it so tight_layout does not squeeze the bars.
    fig_w = max(LARGE_MIN_FIG_WIDTH if args.large_text else 8.0,
                len(matrices) * (inches_per_bar * n_tol
                                      + INCHES_GROUP_PAD))
    fig, ax = plt.subplots(figsize=(fig_w, 6.0 * height_factor))

    fallback = iter(FALLBACK_COLORS)
    colors = {}
    for k in bin_keys:
        colors[k] = BIN_COLORS.get(k) or next(fallback, "0.5")

    centres, tick_pos, tick_lab = [], [], []
    for mi, matrix in enumerate(matrices):
        base = float(mi)
        centres.append(base)
        for ti, tol in enumerate(tolerances):
            fr = fractions.get((matrix, tol))
            if fr is None:
                continue
            xpos = base + (ti - (n_tol - 1) / 2) * bar_w
            tick_pos.append(xpos)
            tick_lab.append(tol_label(tol))
            bottom = 0.0
            for k in bin_keys + ["other"]:
                v = fr.get(k, 0.0)
                if v <= 0:
                    continue
                ax.bar(xpos, v, bar_w * 0.92, bottom=bottom,
                       color=OTHER_COLOR if k == "other" else colors[k],
                       hatch=OTHER_HATCH if k == "other" else None,
                       edgecolor="black", linewidth=0.5, zorder=3)
                if not args.no_labels and v >= 0.06:
                    ax.text(xpos, bottom + v / 2, f"{v * 100:.0f}",
                            ha="center", va="center",
                            fontsize=(LARGE_BAR_LABEL_SIZE if args.large_text
                                      else 8),
                            color="black", zorder=4)
                bottom += v

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Fraction of nonzeros")
    ax.set_xlim(-0.5, centres[-1] + 0.5)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Two tick levels: tolerance under every bar, matrix name under the group.
    ax.set_xticks(tick_pos)
    tol_fs = LARGE_TOL_TICK_SIZE if args.large_text else 9
    name_fs = LARGE_MATRIX_NAME_SIZE if args.large_text else 14
    ax.set_xticklabels(tick_lab, rotation=90, fontsize=tol_fs)
    ax.tick_params(axis="x", length=2, pad=2)
    for c, matrix in zip(centres, matrices):
        # Offset below the axes grows with the tolerance label size so the
        # matrix name stays clear of the rotated tolerance labels.
        ax.annotate(matrix, xy=(c, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -(42 / 9) * tol_fs), textcoords="offset points",
                    ha="right", va="top", rotation=30,
                    rotation_mode="anchor", fontsize=name_fs)

    handles = [Patch(facecolor=colors[k], edgecolor="black",
                     label=BIN_LABELS.get(k, k)) for k in bin_keys]
    if have_other:
        handles.append(Patch(facecolor=OTHER_COLOR, edgecolor="black",
                             hatch=OTHER_HATCH, label="dropped"))
    legend_kw = ({"fontsize": LARGE_LEGEND_SIZE,
                  "title_fontsize": LARGE_LEGEND_SIZE}
                 if args.large_text else {})
    ax.legend(handles=handles, title=f"{args.amp_format.upper()} precision",
              ncol=len(handles), loc="lower center",
              bbox_to_anchor=(0.5, 1.02), framealpha=0.9, **legend_kw)
    if args.title:
        ax.set_title(args.title, pad=48 * scale)

    fig.tight_layout()
    out_path = Path(args.output) if args.output else (
        Path(args.results_dir) /
        f"spmv_amp_bin_fractions_{args.amp_format}.png")
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")

    labels = {k: BIN_LABELS.get(k, k) for k in bin_keys}
    print(f"\nfraction of nonzeros per AMP bin ({args.amp_format}, "
          f"normalized by {'nonzeros' if args.normalize == 'nnz' else 'bins'}"
          f"; %, columns = tolerance)")
    for matrix in matrices:
        print(f"  {matrix}")
        for k in bin_keys + (["other"] if have_other else []):
            line = f"    {labels.get(k, 'dropped'):<10}"
            for tol in tolerances:
                fr = fractions.get((matrix, tol))
                line += (f"{fr[k] * 100:>9.1f}" if fr else f"{'-':>9}")
            print(line)
    print("    " + " " * 10 + "".join(f"{tol_label(t):>9}" for t in tolerances))
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
