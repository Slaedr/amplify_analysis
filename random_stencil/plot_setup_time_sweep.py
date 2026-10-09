#!/usr/bin/env python3

"""FGS setup time vs AMP tolerance, as stacked bars.

Reads the FGS results of an AMP tolerance sweep (``<results>/fgs/tol_*/
<run>/fgs_*_results.json``) and plots, for every AMP tolerance, the overall
FGS setup time of the AMP<double> run split into two stacked parts:

  bottom   AMP setup      ``amp_setup_ms``  (AMP matrix generation only)
  top      other setup    ``setup_ms - amp_setup_ms``  (reading the matrix
                          into the base format; the FGS solver construction
                          is negligible)

The total bar height is therefore the overall ``setup_ms`` of the AMP<double>
row.  This relies on the timing in ginkgo-amp/benchmark/amp/fgs.cpp:

    t0 .. t1   create + read the base matrix
    t1 .. t2   AMP::generate           -> amp_setup_ms
    t2 .. t3   FwdGaussSeidel::generate
    setup_ms     = t3 - t0   (contains amp_setup_ms)

FwdGaussSeidel::generate only stores the system matrix, combines the stopping
criteria and copies the host-side ``color_ptrs`` vector; it performs no matrix
analysis or coloring, so the "other" part is dominated by the matrix read.

``amp_setup_ms`` is stored in the JSON as the pair ["amp_setup_ms", value]
inside ``amp_details``.

The setup time does not depend on the SpMV kernel used, so only one run per
tolerance is plotted.

By default (``--spmv_units``) the times are expressed in units of one AMP<double>
SpMV at the same AMP tolerance, i.e. every setup time is divided by the
``time_ms`` of the AMP<double> row of the SpMV benchmark run at that tolerance
(``<results>/spmv/tol_*/<run>/spmv_*_results.json``).  Use ``--no-spmv_units``
to plot milliseconds instead.

Usage:
    ./plot_setup_time_sweep.py results-mi250x-bf16-csr
    ./plot_setup_time_sweep.py results-mi250x-bf16-csr -o figs/mi250x-setup.png
    ./plot_setup_time_sweep.py results-mi250x-bf16-csr --no-spmv_units   # ms
"""

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RUN_DIR = "monolithic_classical"  # setup time is kernel-independent
AMP_COLOR = "#1f77b4"
OTHER_COLOR = "#aec7e8"


def tol_label(tol):
    return f"$10^{{{round(math.log10(tol))}}}$"


def amp_setup_ms(amp_row):
    """Extract amp_setup_ms from the ["amp_setup_ms", v] pair in amp_details."""
    for item in amp_row.get("amp_details", []):
        if isinstance(item, (list, tuple)) and len(item) == 2 and item[0] == "amp_setup_ms":
            return float(item[1])
    return None


def load_run(path):
    """Return (tolerance, overall setup_ms, amp setup_ms) of one FGS result."""
    with open(path) as f:
        data = json.load(f)
    tol = float(data["config"]["amp_tolerance"])
    rows = data["fgs"]
    amp = next((r for r in rows if r["format"].startswith("AMP")), None)
    if amp is None:
        raise ValueError("no AMP row")
    a = amp_setup_ms(amp)
    if a is None:
        raise ValueError("no amp_setup_ms in amp_details")
    return tol, float(amp["setup_ms"]), a


def load_sweep(results_dir):
    """Sorted list of (tol, total_ms, amp_ms), loosest tolerance first."""
    runs = []
    for path in sorted(results_dir.glob(f"fgs/tol_*/{RUN_DIR}/*results.json")):
        try:
            tol, total, amp = load_run(path)
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            print(f"skipping {path}: {e}", file=sys.stderr)
            continue
        if amp > total:
            print(f"warning: amp_setup_ms > setup_ms in {path}", file=sys.stderr)
        runs.append((round(tol, 15), total, amp))
    return sorted(runs, key=lambda r: -r[0])


def load_spmv_times(results_dir):
    """tolerance -> time_ms of one AMP<double> SpMV, from the SpMV sweep."""
    times = {}
    for path in sorted(results_dir.glob(f"spmv/tol_*/{RUN_DIR}/*results.json")):
        try:
            with open(path) as f:
                data = json.load(f)
            tol = round(float(data["config"]["amp_tolerance"]), 15)
            amp = next(r for r in data["spmv"] if r["format"].startswith("AMP"))
            times[tol] = float(amp["time_ms"])
        except (StopIteration, KeyError, ValueError, json.JSONDecodeError) as e:
            print(f"skipping {path}: {e}", file=sys.stderr)
    return times


def to_spmv_units(runs, spmv_ms):
    """Divide every setup time by the AMP SpMV time at the same tolerance."""
    out = []
    for tol, total, amp in runs:
        t = spmv_ms.get(tol)
        if t is None:
            sys.exit(f"no SpMV result for AMP tolerance {tol:g}; "
                     "use --no-spmv_units to plot milliseconds")
        out.append((tol, total / t, amp / t))
    return out


def draw_plot(ax, runs, spmv_units):
    tols = [r[0] for r in runs]
    total = np.array([r[1] for r in runs])
    amp = np.array([r[2] for r in runs])
    other = total - amp
    x = np.arange(len(runs))

    ax.bar(x, amp, 0.7, color=AMP_COLOR, label="AMP setup", zorder=3)
    ax.bar(x, other, 0.7, bottom=amp, color=OTHER_COLOR,
           label="Other setup (matrix read)", zorder=3)
    fmt = "{:.1f}" if spmv_units else "{:.0f}"
    for xi, t in zip(x, total):
        ax.text(xi, t, fmt.format(t), ha="center", va="bottom", fontsize=7)
    for xi, a in zip(x, amp):
        ax.text(xi, a, fmt.format(a), ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([tol_label(t) for t in tols])
    ax.set_xlabel("AMP tolerance")
    ax.set_ylabel("Setup time [# AMP SpMVs]" if spmv_units else "Setup time [ms]")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path,
                    help="sweep results directory containing fgs/tol_*/")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="output image (default: <results>-fgs-setup-time.png "
                         "next to the results directory)")
    ap.add_argument("--spmv_units", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="express times as the number of AMP<double> SpMVs at "
                         "the same AMP tolerance, using the SpMV results in "
                         "the same results directory (default: on; "
                         "--no-spmv_units for ms)")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    runs = load_sweep(args.results)
    if not runs:
        sys.exit(f"no FGS results found under {args.results}/fgs")

    if args.spmv_units:
        spmv_ms = load_spmv_times(args.results)
        if not spmv_ms:
            sys.exit(f"no SpMV results found under {args.results}/spmv; "
                     "use --no-spmv_units to plot milliseconds")
        runs = to_spmv_units(runs, spmv_ms)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    draw_plot(ax, runs, args.spmv_units)
    ax.legend(frameon=False, loc="upper left")
    ax.set_ylim(0, max(r[1] for r in runs) * 1.3)
    fig.tight_layout()

    out = args.output or args.results.with_name(args.results.name + "-fgs-setup-time.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
