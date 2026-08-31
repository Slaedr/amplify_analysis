#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""
amp_bin_predict.py -- static, GPU-free model of what AMP[CSR] SpMV *can*
achieve on a given matrix, and where the current kernel throws it away.

It replays the exact binning rule from
  common/unified/matrix/amp_algorithms.hpp
    lbs[k]   = tol * ||a_i||_1 / eps(next_narrower_type)   for k < q-1
    lbs[q-1] = tol * ||a_i||_1
    bin(v)   = first k with |v| > lbs[k], else dropped
    diagonal -> bin 0 unconditionally
    underflow: promote while |v| < min_normal(bin)
and then answers four questions per matrix, before you burn a node-hour:

  1. Is there anything to win?  Per-bin nnz split, bytes moved vs plain CSR
     fp64, and the resulting bandwidth-bound speed-up ceiling.  Remember the
     int32 column index rides along with every nonzero regardless of value
     precision, so an all-fp32 matrix caps out at 12/8 = 1.5x, not 2x.

  2. How much of the 64-wide wavefront is actually busy?  The kernel gives one
     full wavefront to one row *per bin*, so the inner loop over bin k runs
     ceil(nnz[i,k]/64) iterations with the tail masked.  Reports the exact
     lane-utilisation this produces, plus what a merged (non-split) loop and
     narrower subwarp tiles would give.

  3. How much of the work is pure overhead?  Fraction of (row, bin) pairs that
     are empty but still cost two row-pointer loads and a loop setup.

  4. Where does x locality go?  Per (row, bin) the column indices are a sparse
     subset of the row's columns, so the gather stride inside one bin loop is
     wider than in plain CSR.  Reports mean column-index gap per bin.

Usage:
  python3 amp_bin_predict.py A.mtx [B.mtx ...] --tol 1e-9 [--tol 1e-10 ...]
                             [--half fp16|bf16] [--tile 64,32,16]
                             [--json out.json] [--csv out.csv]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Scalar type table.  epsilon = 2^-(p-1) with p = significand bits incl. the
# implicit one; min = smallest positive *normal*, matching numeric_limits::min.
# ---------------------------------------------------------------------------
TYPES = {
    "double": dict(eps=2.0 ** -52, min=2.2250738585072014e-308, bytes=8),
    "float":  dict(eps=2.0 ** -23, min=1.1754943508222875e-38, bytes=4),
    "fp16":   dict(eps=2.0 ** -10, min=6.103515625e-05, bytes=2),
    "bf16":   dict(eps=2.0 ** -7,  min=1.1754943508222875e-38, bytes=2),
}
WAVE = 64


def read_mtx(path):
    """Return (nrows, ncols, row, col, val) as 0-based COO with symmetry
    expanded.  Uses scipy when available, otherwise a numpy fallback."""
    try:
        from scipy.io import mmread
        import scipy.sparse as sp
        A = mmread(path)
        A = sp.coo_matrix(A)
        v = np.abs(A.data.astype(np.float64))
        return A.shape[0], A.shape[1], A.row.astype(np.int64), \
            A.col.astype(np.int64), v
    except Exception:
        print("!! Scipy not available! Using Numpy to read mtx.")
        pass
    sym, field = "general", "real"
    with open(path, "r", errors="replace") as fh:
        first = fh.readline()
        if first.startswith("%%MatrixMarket"):
            p = first.lower().split()
            if len(p) >= 5:
                field, sym = p[3], p[4]
        line = fh.readline()
        while line.startswith("%"):
            line = fh.readline()
        nr, nc, nnz = (int(x) for x in line.split()[:3])
        data = np.loadtxt(fh, max_rows=nnz, ndmin=2)
    r = data[:, 0].astype(np.int64) - 1
    c = data[:, 1].astype(np.int64) - 1
    if data.shape[1] >= 4 and field == "complex":
        v = np.hypot(data[:, 2], data[:, 3])
    elif data.shape[1] >= 3:
        v = np.abs(data[:, 2].astype(np.float64))
    else:                                    # pattern matrix
        v = np.ones(len(r))
    if sym in ("symmetric", "hermitian", "skew-symmetric"):
        off = r != c
        r, c, v = (np.concatenate([r, c[off]]), np.concatenate([c, r[off]]),
                   np.concatenate([v, v[off]]))
    return nr, nc, r, c, v


def bin_assign(nrows, row, col, aval, tol, narrow):
    """Vectorised replay of get_adjusted_bin over all nonzeros.
    narrow: list of type names, index 0 = highest precision.
    Returns int array of bin ids (-1 = dropped)."""
    q = len(narrow)
    rn = np.bincount(row, weights=aval, minlength=nrows)      # row 1-norms
    rn_nz = rn[row]

    # lbs[k] for k < q-1 uses eps of the *next narrower* type; lbs[q-1] = tol*rn
    lbs = np.empty((q, len(aval)), dtype=np.float64)
    for k in range(q):
        if k == q - 1:
            lbs[k] = tol * rn_nz
        else:
            lbs[k] = tol * rn_nz / TYPES[narrow[k + 1]]["eps"]

    ibin = np.full(len(aval), -1, dtype=np.int64)
    assigned = np.zeros(len(aval), dtype=bool)
    for k in range(q):                                        # first k with |v|>lbs[k]
        take = (~assigned) & (aval > lbs[k])
        ibin[take] = k
        assigned |= take

    # underflow promotion: while ibin>0 and |v| < min_normal(type[ibin])
    for _ in range(q):
        mins = np.array([TYPES[t]["min"] for t in narrow])
        cur = np.where(ibin > 0, ibin, 0)
        need = (ibin > 0) & (aval < mins[cur])
        if not need.any():
            break
        ibin[need] -= 1

    ibin[row == col] = 0                                      # diagonal override
    return ibin, rn


def lane_util(counts_per_row_bin, tile):
    """counts: 2-D (nrows, q) int array of nnz per (row, bin).
    Returns useful/issued lane-slots for the given tile width."""
    issued = tile * np.ceil(counts_per_row_bin / tile).sum()
    useful = counts_per_row_bin.sum()
    return (useful / issued) if issued else float("nan")


def analyse(path, tol, half, tiles):
    nrows, ncols, row, col, aval = read_mtx(path)
    narrow = ["double", "float", half]
    q = len(narrow)
    ibin, rn = bin_assign(nrows, row, col, aval, tol, narrow)

    nnz = len(aval)
    kept = ibin >= 0
    dropped = int((~kept).sum())

    counts = np.zeros((nrows, q), dtype=np.int64)
    for k in range(q):
        m = ibin == k
        counts[:, k] = np.bincount(row[m], minlength=nrows)
    row_nnz = counts.sum(axis=1)

    per_bin_nnz = counts.sum(axis=0)
    val_bytes = np.array([TYPES[t]["bytes"] for t in narrow], dtype=np.float64)

    # ---- traffic model (per-GCD HBM compulsory bytes, perfect x reuse) -----
    idx_b, y_b, x_b = 4.0, 8.0, 8.0
    amp_mat = float((per_bin_nnz * (val_bytes + idx_b)).sum())
    amp_rowptr = q * (nrows + 1) * idx_b
    amp_ideal = amp_mat + amp_rowptr + nrows * y_b + nrows * x_b
    csr_mat = nnz * (8.0 + idx_b)
    csr_ideal = csr_mat + (nrows + 1) * idx_b + nrows * y_b + nrows * x_b
    # a hypothetical uniform-fp32 CSR, for reference
    csr32_ideal = nnz * (4.0 + idx_b) + (nrows + 1) * idx_b + nrows * y_b + nrows * x_b

    # ---- wavefront utilisation -------------------------------------------
    util = {}
    for t in tiles:
        util[t] = dict(
            split=lane_util(counts, t),                       # what the kernel does
            merged=lane_util(row_nnz.reshape(-1, 1), t),      # one loop per row
        )
    empty_binrows = int((counts == 0).sum())
    # rows where >1 bin is non-empty: these are the ones the split actually hurts
    active_bins = (counts > 0).sum(axis=1)
    waves_split = int(np.ceil(counts / WAVE).sum())
    waves_merged = int(np.ceil(row_nnz / max(WAVE, 1)).sum())

    # ---- x locality: mean column gap within a (row, bin) segment ----------
    order = np.lexsort((col, ibin, row))
    r_s, c_s, b_s = row[order], col[order], ibin[order]
    same_seg = (r_s[1:] == r_s[:-1]) & (b_s[1:] == b_s[:-1]) & (b_s[1:] >= 0)
    gaps_split = np.abs(np.diff(c_s))[same_seg]
    order2 = np.lexsort((col, row))
    r2, c2 = row[order2], col[order2]
    same_row = r2[1:] == r2[:-1]
    gaps_merged = np.abs(np.diff(c2))[same_row]

    return dict(
        matrix=os.path.basename(path), tol=tol, half=half,
        rows=int(nrows), cols=int(ncols), nnz=int(nnz),
        nnz_per_row_mean=float(nnz / nrows) if nrows else 0.0,
        nnz_per_row_max=int(row_nnz.max()) if nrows else 0,
        nnz_per_row_p50=float(np.percentile(row_nnz, 50)) if nrows else 0.0,
        nnz_per_row_p99=float(np.percentile(row_nnz, 99)) if nrows else 0.0,
        dropped_nnz=dropped, dropped_frac=dropped / nnz if nnz else 0.0,
        bin_nnz=[int(x) for x in per_bin_nnz],
        bin_frac=[float(x / nnz) for x in per_bin_nnz] if nnz else [],
        bin_types=narrow,
        empty_binrow_frac=empty_binrows / (nrows * q) if nrows else 0.0,
        mean_active_bins_per_row=float(active_bins.mean()) if nrows else 0.0,
        wave_launches_split=waves_split, wave_launches_merged=waves_merged,
        wave_launch_inflation=(waves_split / waves_merged) if waves_merged else 0.0,
        lane_util=util,
        bytes_amp_ideal=amp_ideal, bytes_csr64_ideal=csr_ideal,
        bytes_csr32_ideal=csr32_ideal,
        bytes_amp_rowptr_frac=amp_rowptr / amp_ideal if amp_ideal else 0.0,
        bw_bound_speedup_vs_csr64=csr_ideal / amp_ideal if amp_ideal else 0.0,
        bw_bound_speedup_csr32_vs_csr64=csr_ideal / csr32_ideal,
        mean_col_gap_split=float(gaps_split.mean()) if gaps_split.size else 0.0,
        mean_col_gap_merged=float(gaps_merged.mean()) if gaps_merged.size else 0.0,
    )


def render(r):
    q = len(r["bin_types"])
    print(f"\n=== {r['matrix']}   tol={r['tol']:g}   narrow=({', '.join(r['bin_types'])})")
    print(f"  {r['rows']:,} x {r['cols']:,}   nnz {r['nnz']:,}   "
          f"nnz/row mean {r['nnz_per_row_mean']:.1f}  p50 {r['nnz_per_row_p50']:.0f}  "
          f"p99 {r['nnz_per_row_p99']:.0f}  max {r['nnz_per_row_max']:,}")

    print("\n  bin split")
    for i, t in enumerate(r["bin_types"]):
        print(f"    bin {i} ({t:>6}): {r['bin_nnz'][i]:>12,}  "
              f"{100*r['bin_frac'][i]:5.1f}%")
    if r["dropped_nnz"]:
        print(f"    dropped      : {r['dropped_nnz']:>12,}  "
              f"{100*r['dropped_frac']:5.1f}%")
    print(f"    empty (row,bin) pairs: {100*r['empty_binrow_frac']:.1f}% of "
          f"{r['rows']*q:,}   mean active bins/row {r['mean_active_bins_per_row']:.2f}")

    print("\n  bandwidth ceiling (compulsory bytes, perfect x reuse)")
    print(f"    CSR fp64 {r['bytes_csr64_ideal']/1e6:10.2f} MB   "
          f"CSR fp32 {r['bytes_csr32_ideal']/1e6:10.2f} MB   "
          f"AMP {r['bytes_amp_ideal']/1e6:10.2f} MB")
    print(f"    AMP speed-up ceiling vs CSR fp64 : "
          f"{r['bw_bound_speedup_vs_csr64']:.3f}x     "
          f"(uniform fp32 would give {r['bw_bound_speedup_csr32_vs_csr64']:.3f}x)")
    print(f"    row-pointer arrays are {100*r['bytes_amp_rowptr_frac']:.1f}% of "
          f"AMP's compulsory traffic ({q} copies)")

    print("\n  wavefront lane utilisation (useful lanes / issued lane-slots)")
    print(f"    {'tile':>6}  {'split (current)':>16}  {'merged (1 loop/row)':>20}")
    for t in sorted(r["lane_util"], reverse=True):
        u = r["lane_util"][t]
        print(f"    {t:>6}  {100*u['split']:>15.1f}%  {100*u['merged']:>19.1f}%")
    print(f"    wave-iterations: split {r['wave_launches_split']:,} vs "
          f"merged {r['wave_launches_merged']:,}  "
          f"({r['wave_launch_inflation']:.2f}x inflation from the bin split)")

    print("\n  x-gather locality (mean |column gap| between consecutive gathers)")
    print(f"    within a (row,bin) segment: {r['mean_col_gap_split']:.1f}   "
          f"within a whole row: {r['mean_col_gap_merged']:.1f}")

    # ---- verdict ---------------------------------------------------------
    v = []
    if r["bw_bound_speedup_vs_csr64"] < 1.10:
        v.append(f"no headroom: the bytes model says at most "
                 f"{r['bw_bound_speedup_vs_csr64']:.2f}x even at 100% efficiency. "
                 f"AMP cannot win on this matrix at tol={r['tol']:g}.")
    if r["bin_frac"] and r["bin_frac"][0] > 0.9:
        v.append(f"{100*r['bin_frac'][0]:.0f}% of nonzeros stay in bin 0 (fp64): "
                 f"the tolerance is too tight, or the in-row dynamic range is "
                 f"too small, for demotion to happen.")
    u64 = r["lane_util"].get(64, {}).get("split")
    if u64 is not None and u64 < 0.35:
        # widest tile that still keeps >=60% of lanes busy
        ok = [t for t in r["lane_util"] if r["lane_util"][t]["split"] >= 0.60]
        best = max(ok) if ok else min(r["lane_util"])
        v.append(f"only {100*u64:.0f}% of the 64 lanes do work. A subwarp tile "
                 f"of {best} would give {100*r['lane_util'][best]['split']:.0f}% "
                 f"and {64//best}x more rows in flight per wavefront.")
    if u64 is not None and r["lane_util"].get(64, {}).get("merged"):
        um = r["lane_util"][64]["merged"]
        if um > 1.25 * u64:
            v.append(f"merging the {len(r['bin_types'])} per-bin loops into one "
                     f"would lift 64-wide lane utilisation from {100*u64:.0f}% "
                     f"to {100*um:.0f}% on its own.")
    if r["wave_launch_inflation"] > 1.5:
        v.append(f"the per-bin loop split costs {r['wave_launch_inflation']:.2f}x "
                 f"more masked wave-iterations than one merged loop would.")
    if r["mean_col_gap_split"] > 2 * max(r["mean_col_gap_merged"], 1e-9):
        v.append(f"gathers inside a bin segment are "
                 f"{r['mean_col_gap_split']/max(r['mean_col_gap_merged'],1e-9):.1f}x "
                 f"farther apart than in plain CSR -- expect worse L1 hit rate "
                 f"on x than the fp64 baseline.")
    if v:
        print("\n  verdict")
        for m in v:
            print(f"    * {m}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("matrices", nargs="+")
    ap.add_argument("--tol", action="append", type=float, default=None,
                    help="AMP tolerance (repeatable); default 1e-9")
    ap.add_argument("--half", choices=["fp16", "bf16"], default="bf16",
                    help="narrowest bin type; bf16 if GINKGO_ENABLE_BFLOAT16=ON")
    ap.add_argument("--tile", default="64,32",
                    help="subwarp widths to evaluate [64,32]")
    ap.add_argument("--json", default=None)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    tols = args.tol or [1e-9]
    tiles = [int(t) for t in args.tile.split(",")]
    recs = []
    for m in args.matrices:
        if not os.path.exists(m):
            print(f"skip (missing): {m}", file=sys.stderr)
            continue
        for tol in tols:
            try:
                r = analyse(m, tol, args.half, tiles)
            except Exception as exc:                       # noqa: BLE001
                print(f"skip {m} (tol={tol}): {exc}", file=sys.stderr)
                continue
            recs.append(r)
            render(r)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(recs, fh, indent=2)
        print(f"\nwrote {args.json}")
    if args.csv:
        import csv as _csv
        flat = []
        for r in recs:
            d = {k: v for k, v in r.items() if not isinstance(v, (dict, list))}
            for i, t in enumerate(r["bin_types"]):
                d[f"bin{i}_{t}_frac"] = r["bin_frac"][i] if r["bin_frac"] else None
            for t, u in r["lane_util"].items():
                d[f"laneutil_split_{t}"] = u["split"]
                d[f"laneutil_merged_{t}"] = u["merged"]
            flat.append(d)
        if flat:
            with open(args.csv, "w", newline="") as fh:
                w = _csv.DictWriter(fh, fieldnames=sorted(flat[0]))
                w.writeheader()
                w.writerows(flat)
            print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
