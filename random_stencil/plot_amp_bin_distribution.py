#!/usr/bin/env python3

"""Stacked bar plot of the AMP bin distribution (fraction of nonzeros stored
in each precision) vs AMP tolerance, for the random-stencil SpMV, FGS and
GMRES benchmarks.

Companion of ``plot_speedup_error_vs_tol.py`` (same results tree, same
benchmark / base-format detection) and of
``../suitesparse/plot_spmv_amp_bin_fractions.py`` (same stacked-bar style).

One stacked bar is drawn per AMP tolerance, tightest tolerance on the left
like in the speedup/error plot.  Each bar stacks the fraction of the matrix'
nonzeros held in every AMP bin (``amp_details[*].nnz`` of the AMP entry in
the result JSON: FP64 at the bottom, then FP32, then FP16) up to 1.0.  Only
the AMP bins are drawn -- there are no bars for the uniform-precision
reference formats (CSR<float> / CSR<half>).

If the bins do not account for all nonzeros, the remainder is drawn as a
hatched grey "dropped" segment on top, so every bar still reaches 1.0 (and a
warning is printed).  ``--normalize bins`` instead divides by the sum of the
bin counts, in which case nothing is shown as dropped.

The AMP strategies (monolithic / independent buckets) only change the kernel,
not the bin assignment, so they report identical bins.  The bins are taken
from the first strategy found (or ``--strategy``); a warning is printed if
two strategies disagree.

The benchmark is detected from the top-level key of each result JSON
("spmv" / "fgs" / "gmres"); ``--bench`` selects which one is plotted and
defaults to spmv.  The tolerance, base format and total nonzeros are read
from the JSON itself, so file and directory names do not matter.

Usage:
    ./plot_amp_bin_distribution.py results-mi250x-tol-csr
    ./plot_amp_bin_distribution.py results-mi250x-tol-csr --bench fgs \\
        --strategy independent_buckets -o fgs_bins.pdf
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

plt.rcParams.update({
    "font.size": 15,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 15,
    "legend.fontsize": 13,
})

BENCHES = ("spmv", "fgs", "gmres")

# Default precision of AMP bin 0, 1, 2 (override with --bin-labels).
DEFAULT_BIN_LABELS = ["FP64", "FP32", "FP16"]
# Colour-blind safe, one colour per precision (same as
# plot_spmv_amp_bin_fractions.py; cycles for unknown bins).
BIN_COLORS = ["#2c7bb6", "#fdae61", "#d7191c"]
FALLBACK_COLORS = ["#4daf4a", "#984ea3", "#a6611a", "#01665e"]
OTHER_COLOR = "#d9d9d9"
OTHER_HATCH = "//"

BAR_WIDTH = 0.8
INCHES_PER_BAR = 0.62


def load_runs(results_dir, bench_filter, base_filter):
    """One record per benchmark JSON found under results_dir."""
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
        cfg = data.get("config", {})
        amp = next((row for row in data[bench]
                    if str(row.get("format", "")).startswith("AMP")), None)
        if amp is None:
            print(f"  skipping {path}: no AMP entry")
            continue

        # amp_details is a list of {"bin": i, "nnz": n} dicts, followed by an
        # ["amp_setup_ms", value] pair; keep only the dicts.
        bins = {}
        for d in amp.get("amp_details", []):
            if isinstance(d, dict) and "bin" in d and "nnz" in d:
                bins[int(d["bin"])] = int(d["nnz"])
        if not bins:
            print(f"  skipping {path}: no amp_details bins")
            continue

        tol = cfg.get("amp_tolerance")
        if tol is None:
            print(f"  skipping {path}: no amp_tolerance")
            continue
        # AMP stores the tolerance as a float (1e-8 -> 9.99999994e-09).
        tol = float(f"{float(tol):.1e}")
        nnz = cfg.get("global_nnz", cfg.get("nnz"))

        runs.append({
            "path": path,
            "bench": bench,
            "base_format": cfg.get("amp_base_format", "?"),
            "executor": cfg.get("executor", "?"),
            "strategy": cfg.get("amp_spmv_strategy", "unknown"),
            "tolerance": tol,
            "bins": bins,
            "nnz": nnz,
        })

    if not runs:
        raise SystemExit(f"No SpMV/FGS/GMRES result JSON with AMP bins found "
                         f"under {results_dir}")

    benches = sorted({r["bench"] for r in runs})
    if bench_filter:
        runs = [r for r in runs if r["bench"] == bench_filter]
        if not runs:
            raise SystemExit(f"No runs for benchmark '{bench_filter}' "
                             f"(found: {', '.join(benches)})")
    elif len(benches) > 1:
        raise SystemExit(f"Results mix benchmarks ({', '.join(benches)}); "
                         f"pass --bench.")

    formats = sorted({r["base_format"] for r in runs})
    if base_filter:
        runs = [r for r in runs if r["base_format"] == base_filter]
        if not runs:
            raise SystemExit(f"No runs with base format '{base_filter}' "
                             f"(found: {', '.join(formats)})")
    elif len(formats) > 1:
        raise SystemExit(f"Results mix base formats ({', '.join(formats)}); "
                         f"pass --base-format.")
    return runs


def tol_label(tol):
    return f"1e-{round(-math.log10(tol))}"


def tol_tick(tol):
    return f"$10^{{-{round(-math.log10(tol))}}}$"


def main():
    parser = argparse.ArgumentParser(
        description="Stacked bar plot of the fraction of nonzeros in each "
                    "AMP bin vs AMP tolerance (random-stencil sweeps).")
    parser.add_argument(
        "results_dir", nargs="?", default="results",
        help="Directory holding the sweep results (searched recursively).")
    parser.add_argument(
        "--bench", default="spmv", choices=BENCHES,
        help="Which benchmark to plot (default: %(default)s).")
    parser.add_argument(
        "--base-format", default=None, choices=["ell", "csr"],
        help="Which base format to plot; required only if the tree mixes both.")
    parser.add_argument(
        "--strategy", default=None,
        help="Take the bins from this AMP strategy (e.g. independent_buckets). "
             "Default: the first found, monolithic first.")
    parser.add_argument(
        "--bin-labels", default=",".join(DEFAULT_BIN_LABELS),
        help="Comma separated precision names of bin 0,1,2,... "
             "(default: %(default)s; use FP64,FP32,BF16 for a BF16 build).")
    parser.add_argument(
        "--normalize", default="nnz", choices=["nnz", "bins"],
        help="Divide the bin counts by the total nonzeros of the matrix "
             "(default; nonzeros in no bin are drawn as a hatched grey "
             "'dropped' segment) or by the sum of the bin counts.")
    parser.add_argument(
        "--no-labels", action="store_true",
        help="Do not print the segment percentages.")
    parser.add_argument("--title", default=None, help="Optional plot title.")
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output file (default "
             "<results-dir>/<bench>_amp_bin_distribution.png; extension "
             "picks the format).")
    parser.add_argument("--dpi", type=int, default=400)
    args = parser.parse_args()

    runs = load_runs(args.results_dir, args.bench, args.base_format)
    bench = runs[0]["bench"]
    bin_names = [s.strip() for s in args.bin_labels.split(",") if s.strip()]

    # ---- pick one bin set per tolerance --------------------------------------
    strategies = sorted({r["strategy"] for r in runs},
                        key=lambda s: (s != "monolithic_classical", s))
    strategy = args.strategy or strategies[0]
    if strategy not in strategies:
        raise SystemExit(f"No runs for strategy '{strategy}' "
                         f"(found: {', '.join(strategies)})")

    by_tol = defaultdict(list)
    for r in runs:
        by_tol[r["tolerance"]].append(r)
    chosen = {}
    for tol, rs in by_tol.items():
        picked = [r for r in rs if r["strategy"] == strategy]
        if not picked:
            print(f"  warning: no '{strategy}' run at {tol_label(tol)}; "
                  f"skipping that tolerance")
            continue
        if len(picked) > 1:
            print(f"  warning: several '{strategy}' runs at {tol_label(tol)}; "
                  f"keeping {picked[0]['path']}")
        chosen[tol] = picked[0]
        for other in rs:
            if other is not picked[0] and other["bins"] != picked[0]["bins"]:
                print(f"  warning: bins at {tol_label(tol)} differ between "
                      f"{picked[0]['strategy']} and {other['strategy']}")

    tolerances = sorted(chosen)  # 1e-14 ... 1e-2, like plot_speedup_error_vs_tol
    bin_ids = sorted({b for r in chosen.values() for b in r["bins"]})

    def bin_name(b):
        return bin_names[b] if b < len(bin_names) else f"bin {b}"

    colors, fallback = {}, iter(FALLBACK_COLORS)
    for b in bin_ids:
        colors[b] = BIN_COLORS[b] if b < len(BIN_COLORS) \
            else next(fallback, "0.5")

    # ---- fractions per tolerance ---------------------------------------------
    fractions = {}
    for tol in tolerances:
        r = chosen[tol]
        total_bins = sum(r["bins"].values())
        denom = r["nnz"] if (args.normalize == "nnz" and r["nnz"]) \
            else total_bins
        if not denom:
            continue
        fr = {b: r["bins"].get(b, 0) / denom for b in bin_ids}
        other = max(0.0, 1.0 - sum(fr.values()))
        fr["other"] = other if other > 1e-9 else 0.0
        if sum(fr.values()) > 1.0 + 1e-6:
            print(f"  warning: bins at {tol_label(tol)} exceed the total "
                  f"nonzeros (sum = {sum(fr.values()):.4f})")
        fractions[tol] = fr
    tolerances = [t for t in tolerances if t in fractions]

    short = [(t, fractions[t]["other"]) for t in tolerances
             if fractions[t]["other"] > 1e-6]
    if short:
        worst = max(short, key=lambda s: s[1])
        print(f"  warning: for {len(short)} tolerance(s) the bins cover less "
              f"than all nonzeros (up to {worst[1] * 100:.2f}% missing at "
              f"{tol_label(worst[0])}); shown as a hatched grey segment")
    have_other = bool(short)

    # ---- plot ----------------------------------------------------------------
    n = len(tolerances)
    fig, ax = plt.subplots(figsize=(max(7.0, n * INCHES_PER_BAR + 1.5), 5.0))
    for i, tol in enumerate(tolerances):
        fr = fractions[tol]
        bottom = 0.0
        for k in bin_ids + ["other"]:
            v = fr.get(k, 0.0)
            if v <= 0:
                continue
            ax.bar(i, v, BAR_WIDTH, bottom=bottom,
                   color=OTHER_COLOR if k == "other" else colors[k],
                   hatch=OTHER_HATCH if k == "other" else None,
                   edgecolor="black", linewidth=0.5, zorder=3)
            if not args.no_labels and v >= 0.06:
                ax.text(i, bottom + v / 2, f"{v * 100:.0f}",
                        ha="center", va="center", fontsize=10,
                        color="black", zorder=4)
            bottom += v

    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylabel("Fraction of nonzeros")
    ax.set_xlabel("AMP tolerance")
    ax.set_xticks(range(n))
    ax.set_xticklabels([tol_tick(t) for t in tolerances], rotation=45,
                       ha="right", rotation_mode="anchor")
    ax.yaxis.grid(True, linestyle="--", alpha=0.7, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    handles = [Patch(facecolor=colors[b], edgecolor="black",
                     label=bin_name(b)) for b in bin_ids]
    if have_other:
        handles.append(Patch(facecolor=OTHER_COLOR, edgecolor="black",
                             hatch=OTHER_HATCH, label="dropped"))
    ax.legend(handles=handles, title="AMP precision", ncol=len(handles),
              loc="lower center", bbox_to_anchor=(0.5, 1.02), framealpha=0.9)
    if args.title:
        ax.set_title(args.title, pad=48)

    fig.tight_layout()
    out_path = Path(args.output) if args.output else (
        Path(args.results_dir) / f"{bench}_amp_bin_distribution.png")
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")

    # ---- text table ----------------------------------------------------------
    print(f"\nfraction of nonzeros per AMP bin ({bench}, "
          f"{runs[0]['base_format'].upper()} on {runs[0]['executor']}, "
          f"strategy {strategy}, normalized by "
          f"{'nonzeros' if args.normalize == 'nnz' else 'bins'}; %)")
    print("    " + " " * 10 + "".join(f"{tol_label(t):>8}" for t in tolerances))
    for k in bin_ids + (["other"] if have_other else []):
        name = "dropped" if k == "other" else bin_name(k)
        print(f"    {name:<10}" +
              "".join(f"{fractions[t].get(k, 0.0) * 100:>8.1f}"
                      for t in tolerances))
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
