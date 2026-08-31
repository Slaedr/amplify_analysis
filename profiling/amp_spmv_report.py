#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""
amp_spmv_report.py -- turn the raw rocprofv3 CSVs from amp_spmv_profile.sh into
the handful of numbers that actually decide what is limiting AMP SpMV.

For every (matrix, format) it reports:

  time / bandwidth
    kernel time, effective GFLOP/s, HBM bytes moved, achieved HBM BW,
    % of peak and % of an achievable (STREAM-like) ceiling

  wavefront utilisation                       <- "is the 64-wide wave idle?"
    VALUUtilization = SQ_THREAD_CYCLES_VALU / (SQ_ACTIVE_INST_VALU * 64)
    VALU / SALU / VMEM / SMEM instruction counts normalised per nonzero

  cache behaviour
    L1 (TCP) hit rate, L2 (TCC) hit rate, scalar D-cache hit rate,
    L1->L2 and L2->HBM request counts per nonzero

  the AMP-specific diagnostic
    x_reread_factor = measured HBM read bytes / bytes that a perfect-x-reuse
    run would need.  1.0 means every x element was fetched from HBM once;
    3.0 means the bin split (or poor locality) is re-fetching x three times.

  occupancy
    VGPR/SGPR/LDS/scratch, waves in flight per CU, and the occupancy the
    register allocation permits.

Usage:
  python3 amp_spmv_report.py <outdir> [-o report_prefix] [--peak-bw auto|BYTES/s]
                             [--achievable-frac 0.85] [--skip-dispatches N]
                             [--json] [--markdown]
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import statistics
import sys
from collections import defaultdict

# ---------------------------------------------------------------------------
# Device constants.  Peak HBM bandwidth per *GCD* (rocprof counters are always
# per-agent, and one MI250X exposes two agents).
#   MI250X : 220 CU / 2 GCD = 110 CU per GCD, 3.2 TB/s / 2 = 1.6 TB/s per GCD
#   MI210  : 104 CU, 1.6 TB/s, single die -- a good proxy for one MI250X GCD
# FP64 vector peak per GCD: MI250X 23.9 TFLOP/s, MI210 22.6 TFLOP/s.
# ---------------------------------------------------------------------------
DEVICE_TABLE = {
    # (gfx, cu_hint): dict
    "gfx90a": {
        110: dict(name="MI250X (1 GCD)", peak_bw=1.6e12, peak_fp64=23.9e12,
                  l2_bytes=8 << 20, l1_bytes=16 << 10, clock_hz=1.7e9),
        104: dict(name="MI210", peak_bw=1.6e12, peak_fp64=22.6e12,
                  l2_bytes=8 << 20, l1_bytes=16 << 10, clock_hz=1.7e9),
    },
    "gfx942": {
        304: dict(name="MI300X", peak_bw=5.3e12, peak_fp64=81.7e12,
                  l2_bytes=4 << 20, l1_bytes=32 << 10, clock_hz=2.1e9),
    },
}
WAVE = 64  # gfx9 wavefront width


# --------------------------------------------------------------- csv helpers


def _find(root, pattern):
    hits = glob.glob(os.path.join(root, "**", pattern), recursive=True)
    return sorted(hits)


def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def read_counter_csv(path):
    """rocprofv3 counter_collection.csv -> {kernel: {counter: [per-dispatch]}}
    plus {kernel: {'VGPR_Count': .., 'SGPR_Count': .., ...}} static info."""
    per_disp = defaultdict(lambda: defaultdict(dict))  # kern -> disp -> ctr -> v
    static = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            kern = (row.get("Kernel_Name") or row.get("Kernel_Name ") or "").strip()
            if not kern:
                continue
            disp = row.get("Dispatch_Id") or row.get("Correlation_Id") or "0"
            ctr = (row.get("Counter_Name") or "").strip()
            val = _num(row.get("Counter_Value"))
            if ctr and val is not None:
                per_disp[kern][disp][ctr] = val
            st = static.setdefault(kern, {})
            for k in ("Grid_Size", "Workgroup_Size", "LDS_Block_Size",
                      "Scratch_Size", "VGPR_Count", "Accum_VGPR_Count",
                      "SGPR_Count"):
                v = _num(row.get(k))
                if v is not None:
                    st[k] = v
    out = defaultdict(lambda: defaultdict(list))
    for kern, disps in per_disp.items():
        for _, ctrs in sorted(disps.items(), key=lambda kv: _num(kv[0]) or 0):
            for c, v in ctrs.items():
                out[kern][c].append(v)
    return out, static


def read_kernel_trace(path):
    """kernel_trace.csv -> {kernel: [durations_ns]} + grid info."""
    dur = defaultdict(list)
    info = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            kern = (row.get("Kernel_Name") or "").strip()
            if not kern:
                continue
            s, e = _num(row.get("Start_Timestamp")), _num(row.get("End_Timestamp"))
            if s is not None and e is not None and e > s:
                dur[kern].append(e - s)
            gx = _num(row.get("Grid_Size_X")) or _num(row.get("Grid_Size"))
            wx = _num(row.get("Workgroup_Size_X"))
            if gx:
                info.setdefault(kern, {})["grid_x"] = gx
            if wx:
                info.setdefault(kern, {})["wg_x"] = wx
            for k, tgt in (("Private_Segment_Size", "scratch"),
                           ("Group_Segment_Size", "lds")):
                v = _num(row.get(k))
                if v is not None:
                    info.setdefault(kern, {})[tgt] = v
    return dur, info


# --------------------------------------------------------- matrix meta data


def read_bench_json(path):
    """Pull nnz, rows, cols, per-bin nnz and Ginkgo's own timing out of the
    benchmark/spmv result JSON."""
    try:
        with open(path) as fh:
            data = json.load(fh)
    except Exception:
        return {}
    if isinstance(data, list):
        data = data[0] if data else {}
    out = {"rows": data.get("rows"), "cols": data.get("cols"),
           "nonzeros": data.get("nonzeros")}
    spmv = data.get("spmv", {}) or {}
    fmts = {}
    for fmt, rec in spmv.items():
        if not isinstance(rec, dict):
            continue
        fmts[fmt] = {
            "time_s": rec.get("time"),
            "storage": rec.get("storage"),
            "amp_bins": rec.get("amp_bins"),
            "max_relative_norm2": rec.get("max_relative_norm2"),
        }
    out["formats"] = fmts
    return out


def mtx_header(path):
    """Cheap read of rows/cols/nnz from a MatrixMarket file."""
    try:
        with open(path, "r", errors="replace") as fh:
            sym = "general"
            for line in fh:
                if line.startswith("%%MatrixMarket"):
                    parts = line.lower().split()
                    if len(parts) >= 5:
                        sym = parts[4]
                    continue
                if line.startswith("%"):
                    continue
                r, c, n = (int(x) for x in line.split()[:3])
                if sym in ("symmetric", "hermitian", "skew-symmetric"):
                    n = 2 * n  # upper bound; diagonal double-counted
                return dict(rows=r, cols=c, nonzeros=n, symmetry=sym)
    except Exception:
        pass
    return {}


# ---------------------------------------------------------- derived metrics


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def safe_div(a, b):
    if a is None or b in (None, 0):
        return None
    return a / b


def derive(ctr, dev):
    """ctr: {counter_name: median_value}.  Returns derived metric dict."""
    d = {}
    g = ctr.get

    # ---- wavefront / VALU lane utilisation -------------------------------
    # SQ_THREAD_CYCLES_VALU counts per-lane active cycles; SQ_ACTIVE_INST_VALU
    # counts wave-cycles in which a VALU instruction was active.  The ratio to
    # 64 is the fraction of lanes doing useful work.
    # Prefer rocprofv3's own derived metric when it was collected: on releases
    # where a raw SQ counter has been renamed or dropped it is the only way to
    # get this number, and where both exist they agree.
    d["VALUUtilization_pct"] = g("VALUUtilization")
    if d["VALUUtilization_pct"] is None:
        d["VALUUtilization_pct"] = safe_div(
            100.0 * (g("SQ_THREAD_CYCLES_VALU") or 0),
            (g("SQ_ACTIVE_INST_VALU") or 0) * WAVE)
    else:
        d["VALUUtilization_source"] = "rocprofv3-derived"
    d["VALUBusy_pct"] = g("VALUBusy")
    if d["VALUBusy_pct"] is None:
        d["VALUBusy_pct"] = safe_div(100.0 * (g("SQ_ACTIVE_INST_VALU") or 0),
                                     g("SQ_BUSY_CYCLES"))
    d["waves"] = g("SQ_WAVES")
    d["valu_insts_per_wave"] = safe_div(g("SQ_INSTS_VALU"), g("SQ_WAVES"))
    d["salu_insts_per_wave"] = safe_div(g("SQ_INSTS_SALU"), g("SQ_WAVES"))
    d["vmem_rd_per_wave"] = safe_div(g("SQ_INSTS_VMEM_RD"), g("SQ_WAVES"))
    d["smem_per_wave"] = safe_div(g("SQ_INSTS_SMEM"), g("SQ_WAVES"))
    d["salu_to_valu"] = safe_div(g("SQ_INSTS_SALU"), g("SQ_INSTS_VALU"))

    # ---- latency hiding ---------------------------------------------------
    d["wave_wait_frac_pct"] = safe_div(100.0 * (g("SQ_WAIT_ANY") or 0),
                                       g("SQ_WAVE_CYCLES"))
    # rocprof's MeanOccupancyPerCU convention
    d["mean_occupancy_per_cu"] = g("MeanOccupancyPerCU")
    if d["mean_occupancy_per_cu"] is None and \
            g("SQ_ACCUM_PREV_HIRES") is not None and g("GRBM_GUI_ACTIVE"):
        d["mean_occupancy_per_cu"] = (g("SQ_ACCUM_PREV_HIRES")
                                      / g("GRBM_GUI_ACTIVE"))
    d["gpu_busy_pct"] = safe_div(100.0 * (g("GRBM_GUI_ACTIVE") or 0), g("GRBM_COUNT"))

    # ---- vector L1 (TCP) --------------------------------------------------
    acc = g("TCP_TOTAL_CACHE_ACCESSES_sum")
    miss = (g("TCP_TCC_READ_REQ_sum") or 0) + (g("TCP_TCC_WRITE_REQ_sum") or 0)
    if acc:
        d["L1_hit_pct"] = 100.0 * (1.0 - miss / acc)
        d["L1_accesses"] = acc
        d["L1_to_L2_requests"] = miss
    d["L1_stall_cycles"] = g("TCP_PENDING_STALL_CYCLES_sum")

    # ---- L2 (TCC) and HBM -------------------------------------------------
    hit, ms = g("TCC_HIT_sum"), g("TCC_MISS_sum")
    if hit is not None and ms is not None and (hit + ms) > 0:
        d["L2_hit_pct"] = 100.0 * hit / (hit + ms)
        d["L2_requests"] = hit + ms
    elif g("L2CacheHit") is not None:
        d["L2_hit_pct"] = g("L2CacheHit")
    rd, rd32 = g("TCC_EA_RDREQ_sum"), g("TCC_EA_RDREQ_32B_sum")
    if rd is not None:
        rd32 = rd32 or 0.0
        d["hbm_read_bytes"] = rd32 * 32 + (rd - rd32) * 64
    elif g("FetchSize") is not None:
        d["hbm_read_bytes"] = g("FetchSize") * 1024.0     # rocprof reports KB
    wr, wr64 = g("TCC_EA_WRREQ_sum"), g("TCC_EA_WRREQ_64B_sum")
    if wr is not None:
        wr64 = wr64 or 0.0
        d["hbm_write_bytes"] = wr64 * 64 + (wr - wr64) * 32
    elif g("WriteSize") is not None:
        d["hbm_write_bytes"] = g("WriteSize") * 1024.0
    d["MemUnitStalled_pct"] = g("MemUnitStalled")
    if d.get("hbm_read_bytes") is not None:
        d["hbm_bytes"] = d["hbm_read_bytes"] + (d.get("hbm_write_bytes") or 0.0)
    d["L2_tag_stall"] = g("TCC_TAG_STALL_sum")

    # ---- scalar cache -----------------------------------------------------
    sreq, shit = g("SQC_DCACHE_REQ"), g("SQC_DCACHE_HITS")
    if sreq:
        d["sL1D_hit_pct"] = 100.0 * (shit or 0) / sreq
        d["sL1D_requests"] = sreq

    # ---- FP mix -----------------------------------------------------------
    f64 = sum(g(k) or 0 for k in ("SQ_INSTS_VALU_ADD_F64", "SQ_INSTS_VALU_MUL_F64",
                                  "SQ_INSTS_VALU_FMA_F64"))
    f32 = sum(g(k) or 0 for k in ("SQ_INSTS_VALU_ADD_F32", "SQ_INSTS_VALU_MUL_F32",
                                  "SQ_INSTS_VALU_FMA_F32"))
    if f64 or f32:
        d["fp64_valu_insts"] = f64
        d["fp32_valu_insts"] = f32
    d["cvt_insts"] = g("SQ_INSTS_VALU_CVT")
    if g("SQ_INSTS_VALU"):
        d["cvt_frac_of_valu_pct"] = safe_div(100.0 * (g("SQ_INSTS_VALU_CVT") or 0),
                                             g("SQ_INSTS_VALU"))
    return d


def traffic_model(rows, nnz, bins, index_bytes=4, y_bytes=8, x_bytes=8):
    """Compulsory HBM bytes assuming *perfect* reuse of x (each x element read
    exactly once) and no reuse at all, for the AMP[CSR] layout.

    bins: list of (nnz_in_bin, value_bytes).  For plain CSR fp64 pass
          [(nnz, 8)].
    """
    if not rows or not nnz:
        return {}
    q = len(bins)
    mat = sum(n * (vb + index_bytes) for n, vb in bins)
    rowptr = q * (rows + 1) * index_bytes
    y = rows * y_bytes
    ideal = mat + rowptr + y + rows * x_bytes          # perfect x reuse
    worst = mat + rowptr + y + nnz * x_bytes           # zero x reuse
    return dict(matrix_bytes=mat, rowptr_bytes=rowptr, y_bytes=y,
                x_bytes_ideal=rows * x_bytes, x_bytes_worst=nnz * x_bytes,
                ideal_bytes=ideal, worst_bytes=worst)


# ------------------------------------------------------------------ walking


def collect(outdir, skip_dispatches, kernel_hint=None):
    results = []
    dev_meta = {}
    dj = os.path.join(outdir, "device.json")
    if os.path.exists(dj):
        with open(dj) as fh:
            dev_meta = json.load(fh)
    expected = None
    try:
        expected = int(dev_meta.get("warmup", 0)) + int(dev_meta.get("repetitions", 0))
    except (TypeError, ValueError):
        pass

    for mdir in sorted(d for d in glob.glob(os.path.join(outdir, "*"))
                       if os.path.isdir(d)):
        matrix = os.path.basename(mdir)
        for fdir in sorted(d for d in glob.glob(os.path.join(mdir, "*"))
                           if os.path.isdir(d)):
            fmt = os.path.basename(fdir)
            rec = {"matrix": matrix, "format": fmt, "counters": {},
                   "static": {}, "kernel": None}

            # ---- timing: keep EVERY kernel, choose afterwards
            dur_all, kinfo_all = {}, {}
            for tr in _find(os.path.join(fdir, "timing"), "*kernel_trace.csv"):
                dur, kinfo = read_kernel_trace(tr)
                dur_all.update(dur)
                kinfo_all.update(kinfo)

            bj = os.path.join(fdir, "timing", "bench.json")
            if os.path.exists(bj):
                rec["bench"] = read_bench_json(bj)

            # ---- counters: merge every CSV under every pmc_* directory.
            # A group that failed as a whole and was retried counter-by-counter
            # leaves one CSV per counter in pmc_<set>/single_<counter>/; those
            # are just as good as one combined file.
            ctr_all = defaultdict(dict)
            static_all = defaultdict(dict)
            for pdir in sorted(glob.glob(os.path.join(fdir, "pmc_*"))):
                setname = os.path.basename(pdir)[4:]
                got = False
                for cf in _find(pdir, "*counter_collection.csv"):
                    per_kern, static = read_counter_csv(cf)
                    for kern, ctrs in per_kern.items():
                        for c, vals in ctrs.items():
                            v = vals[skip_dispatches:] or vals
                            ctr_all[kern][c] = med(v)
                            got = True
                        static_all[kern].update(static.get(kern, {}))
                if got:
                    rec.setdefault("pmc_sets", []).append(setname)

            # ---- which kernels make up ONE SpMV?
            group = _select_kernel_group(dur_all, ctr_all, fmt, kernel_hint,
                                         expected)
            rec["kernel"] = " + ".join(group) if group else None
            rec["kernel_group"] = group

            # Some strategies need more than one kernel per SpMV: merge-path is
            # abstract_merge_path_spmv + abstract_reduce (the latter a single
            # block that finishes the partial sums).  Load-balance
            # (abstract_spmv) and classical (abstract_classical_spmv) are one
            # kernel each.  Whatever the group turns out to be, its times and
            # counters add.
            times, disp = [], []
            for k in group:
                ds = (dur_all.get(k) or [])[skip_dispatches:] \
                     or dur_all.get(k) or []
                if ds:
                    times.append(statistics.median(ds))
                    disp.append(len(dur_all[k]))
                for c, v in ctr_all.get(k, {}).items():
                    rec["counters"][c] = rec["counters"].get(c, 0.0) + (v or 0.0)
                rec["static"].update(static_all.get(k, {}))
                rec["static"].update(kinfo_all.get(k, {}))
            if times:
                rec["time_ns_median"] = sum(times)
                rec["dispatches"] = max(disp) if disp else None

            # full breakdown, so a multi-kernel baseline is never hidden
            rec["all_kernels"] = sorted(
                ({"name": k, "dispatches": len(v),
                  "median_ns": med(v[skip_dispatches:] or v),
                  "total_ns": sum(v)} for k, v in dur_all.items()),
                key=lambda e: -(e["total_ns"] or 0))[:12]
            results.append(rec)
    return dev_meta, results


def _select_kernel_group(dur_all, ctr_all, fmt, hint, expected):
    """Names of the kernels that together constitute one SpMV.

    The discriminator that actually works is dispatch count: benchmark/spmv
    launches the SpMV (warmup + repetitions) times, while setup, conversion and
    RHS kernels run once or twice.  This is strategy-agnostic, which matters
    because --formats=csr uses Ginkgo's 'automatical' strategy and so the
    kernel depends on the matrix:

        load-balance : abstract_spmv                                (1 kernel)
        classical    : abstract_classical_spmv                      (1 kernel)
        merge-path   : abstract_merge_path_spmv + abstract_reduce   (2 kernels)

    (There is no 'abstract_load_balance_spmv'.)  Only merge-path needs the
    trailing abstract_reduce; when a strategy does not launch it, it simply
    never appears in the trace and the group is a single kernel.
    """
    names = [n for n in (dur_all or ctr_all) if n]
    if not names:
        return []
    if hint:
        m = [n for n in names if re.search(hint, n, re.I)]
        if m:
            return sorted(m)

    counts = {n: len(dur_all.get(n, [])) for n in names}
    cands = names
    if expected and dur_all:
        lo = max(2, int(expected * 0.7))
        rep = [n for n in names if counts.get(n, 0) >= lo]
        if rep:
            cands = rep

    if fmt == "amp":
        amp = [n for n in cands if re.search(r"amp_.*spmv|amp.*spmv", n, re.I)]
        if amp:
            return sorted(amp)
    spmvish = [n for n in cands
               if re.search(r"spmv|abstract_reduce", n, re.I)]
    if spmvish:
        return sorted(spmvish)
    return sorted(cands)


# ------------------------------------------------------------------ report


def fmt_num(v, unit="", nd=2):
    if v is None:
        return "-"
    if unit == "%":
        return f"{v:.1f}%"
    if abs(v) >= 1e12:
        return f"{v/1e12:.{nd}f}T{unit}"
    if abs(v) >= 1e9:
        return f"{v/1e9:.{nd}f}G{unit}"
    if abs(v) >= 1e6:
        return f"{v/1e6:.{nd}f}M{unit}"
    if abs(v) >= 1e3:
        return f"{v/1e3:.{nd}f}k{unit}"
    return f"{v:.{nd}f}{unit}"


def analyse(dev_meta, results, peak_bw, achievable_frac, peak_fp64, dev_name):
    rows_out = []
    for rec in results:
        d = derive(rec["counters"], dev_meta)
        b = rec.get("bench") or {}
        nnz = b.get("nonzeros")
        nrows = b.get("rows")
        t_s = (rec.get("time_ns_median") or 0) * 1e-9 or None

        # per-bin nnz from write_amp_bin_info(), when present
        bins = None
        fmt_rec = (b.get("formats") or {}).get(rec["format"]) or {}
        amp_bins = fmt_rec.get("amp_bins")
        if isinstance(amp_bins, (list, tuple)) and amp_bins:
            vb = [8, 4, 2]
            bins = []
            for i, e in enumerate(amp_bins):
                n = e.get("nnz") if isinstance(e, dict) else e
                if n is None:
                    continue
                bins.append((float(n), vb[i] if i < len(vb) else 2))
        if bins is None and nnz:
            bins = [(float(nnz), 8.0)]

        tm = traffic_model(nrows, nnz, bins) if (nrows and nnz and bins) else {}

        row = dict(rec_matrix=rec["matrix"], rec_format=rec["format"],
                   kernel=rec.get("kernel"),
                   kernel_group=rec.get("kernel_group"),
                   all_kernels=rec.get("all_kernels"),
                   pmc_sets=rec.get("pmc_sets"),
                   dispatches=rec.get("dispatches"),
                   rows=nrows, nnz=nnz,
                   time_us=(t_s * 1e6) if t_s else None)
        row.update(d)
        row.update({f"model_{k}": v for k, v in tm.items()})

        if t_s and d.get("hbm_bytes"):
            row["achieved_bw_Bps"] = d["hbm_bytes"] / t_s
            row["pct_of_peak_bw"] = 100.0 * row["achieved_bw_Bps"] / peak_bw
            row["pct_of_achievable_bw"] = (100.0 * row["achieved_bw_Bps"]
                                           / (peak_bw * achievable_frac))
        if t_s and nnz:
            row["gflops"] = 2.0 * nnz / t_s / 1e9
            row["pct_of_fp64_peak"] = 100.0 * (row["gflops"] * 1e9) / peak_fp64
            row["ns_per_nnz"] = t_s * 1e9 / nnz
        if d.get("hbm_bytes") and tm.get("ideal_bytes"):
            row["bytes_per_nnz_measured"] = d["hbm_bytes"] / nnz if nnz else None
            row["bytes_per_nnz_ideal"] = tm["ideal_bytes"] / nnz if nnz else None
            row["traffic_inflation"] = d["hbm_bytes"] / tm["ideal_bytes"]
            # how many times was x pulled from HBM, relative to reading it once?
            non_x = tm["ideal_bytes"] - tm["x_bytes_ideal"]
            extra = d["hbm_read_bytes"] - (non_x - tm["y_bytes"])
            if tm["x_bytes_ideal"] > 0:
                row["x_reread_factor"] = max(0.0, extra) / tm["x_bytes_ideal"]
        # register-limited occupancy (gfx90a: 512 VGPRs/SIMD, 8 waves/SIMD max)
        v = rec["static"].get("VGPR_Count")
        av = rec["static"].get("Accum_VGPR_Count") or 0
        if v:
            tot = math.ceil((v + av) / 8) * 8
            row["vgpr"] = v
            row["accum_vgpr"] = av
            row["occupancy_waves_per_simd"] = min(8, int(512 // max(tot, 8)))
        row["sgpr"] = rec["static"].get("SGPR_Count")
        row["lds"] = rec["static"].get("LDS_Block_Size") or rec["static"].get("lds")
        row["scratch"] = (rec["static"].get("Scratch_Size")
                          or rec["static"].get("scratch"))
        # per-nonzero instruction accounting
        if nnz and rec["counters"].get("SQ_INSTS_VALU"):
            row["valu_per_nnz"] = rec["counters"]["SQ_INSTS_VALU"] * WAVE / nnz
        if nnz and rec["counters"].get("SQ_INSTS_VMEM_RD"):
            row["vmem_rd_per_nnz"] = rec["counters"]["SQ_INSTS_VMEM_RD"] * WAVE / nnz
        if nrows and rec["counters"].get("SQ_INSTS_SMEM"):
            row["smem_per_row"] = rec["counters"]["SQ_INSTS_SMEM"] / nrows
        rows_out.append(row)
    return rows_out


HEADLINE = [
    ("rec_matrix", "matrix", ""),
    ("rec_format", "fmt", ""),
    ("nnz", "nnz", ""),
    ("time_us", "t[us]", ""),
    ("gflops", "GF/s", ""),
    ("achieved_bw_Bps", "BW", "B/s"),
    ("pct_of_peak_bw", "%peak", "%"),
    ("VALUUtilization_pct", "laneUtil", "%"),
    ("mean_occupancy_per_cu", "occ/CU", ""),
    ("L1_hit_pct", "L1hit", "%"),
    ("L2_hit_pct", "L2hit", "%"),
    ("bytes_per_nnz_measured", "B/nnz", ""),
    ("traffic_inflation", "inflate", ""),
    ("x_reread_factor", "x-reread", ""),
]


def print_table(rows, stream=sys.stdout):
    hdr = [h for _, h, _ in HEADLINE]
    body = []
    for r in rows:
        line = []
        for key, _, unit in HEADLINE:
            v = r.get(key)
            if isinstance(v, str) or v is None:
                line.append(v if v else "-")
            else:
                line.append(fmt_num(v, unit))
        body.append(line)
    widths = [max(len(hdr[i]), *(len(b[i]) for b in body)) if body else len(hdr[i])
              for i in range(len(hdr))]
    sep = "  "
    print(sep.join(h.rjust(w) for h, w in zip(hdr, widths)), file=stream)
    print(sep.join("-" * w for w in widths), file=stream)
    for b in body:
        print(sep.join(c.rjust(w) for c, w in zip(b, widths)), file=stream)


def diagnose(rows):
    """Short, opinionated verdicts.  Thresholds are deliberately blunt."""
    out = []
    by_matrix = defaultdict(dict)
    for r in rows:
        by_matrix[r["rec_matrix"]][r["rec_format"]] = r
    for matrix, fmts in sorted(by_matrix.items()):
        amp = fmts.get("amp")
        csr = fmts.get("csr")
        msgs = []
        for r in (amp, csr):
            if r and (r.get("pct_of_peak_bw") or 0) > 105:
                msgs.append(
                    f"[{r['rec_format']}] measured {r['pct_of_peak_bw']:.0f}% of "
                    f"peak HBM BW -- impossible. Either the TCC_* counters were "
                    f"summed across dispatches instead of taken per-dispatch "
                    f"(check --skip-dispatches and that the CSV has one row per "
                    f"Dispatch_Id), or the timing pass and the counter pass saw "
                    f"different kernels. Treat the bandwidth column as invalid.")
        if amp:
            lu = amp.get("VALUUtilization_pct")
            if lu is not None and lu < 35:
                msgs.append(
                    f"lane utilisation {lu:.0f}% -- the 64-wide wave-per-row is "
                    f"mostly masked off; try a narrower subwarp tile")
            pb = amp.get("pct_of_peak_bw")
            if pb is not None:
                if pb > 60:
                    msgs.append(f"HBM-bound: {pb:.0f}% of peak BW; only a bytes "
                                f"reduction will help")
                elif pb < 25:
                    msgs.append(f"NOT bandwidth bound ({pb:.0f}% of peak) -- "
                                f"latency/occupancy/issue is the limiter")
            xr = amp.get("x_reread_factor")
            if xr is not None and xr > 2.0:
                msgs.append(f"x re-read {xr:.1f}x from HBM -- the bin split is "
                            f"destroying x locality; consider bin-interleaved "
                            f"or column-sorted layout")
            occ = amp.get("occupancy_waves_per_simd")
            if occ is not None and occ <= 4:
                msgs.append(f"register-limited to {occ} waves/SIMD "
                            f"(VGPR={amp.get('vgpr')}) -- the q-way constexpr_for "
                            f"unroll is expensive; try __launch_bounds__ tuning")
            l1 = amp.get("L1_hit_pct")
            if l1 is not None and l1 < 30:
                msgs.append(f"L1 hit {l1:.0f}% -- x gather is not hitting the "
                            f"16 KB vector L1")
            sm = amp.get("smem_per_row")
            if sm is not None and sm > 6:
                msgs.append(f"{sm:.1f} scalar loads/row -- q row-pointer pairs "
                            f"are not being amortised")
        if amp and csr and amp.get("time_us") and csr.get("time_us"):
            sp = csr["time_us"] / amp["time_us"]
            bideal = None
            if amp.get("model_ideal_bytes") and csr.get("model_ideal_bytes"):
                bideal = csr["model_ideal_bytes"] / amp["model_ideal_bytes"]
            m = f"AMP vs CSR: {sp:.2f}x"
            if bideal:
                m += (f" measured, {bideal:.2f}x is the bytes-model ceiling"
                      f" -> {100*sp/bideal:.0f}% of the achievable win")
            msgs.insert(0, m)
        if msgs:
            out.append((matrix, msgs))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outdir")
    ap.add_argument("-o", "--out-prefix", default=None)
    ap.add_argument("--peak-bw", default="auto",
                    help="peak HBM bytes/s per GCD, or 'auto'")
    ap.add_argument("--peak-fp64", default="auto")
    ap.add_argument("--achievable-frac", type=float, default=0.85,
                    help="STREAM-achievable fraction of peak BW [0.85]")
    ap.add_argument("--skip-dispatches", type=int, default=2,
                    help="drop this many leading dispatches (warmup) [2]")
    ap.add_argument("--kernel", default=None, help="kernel-name regex override")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()

    dev_meta, results = collect(args.outdir, args.skip_dispatches, args.kernel)
    gfx = dev_meta.get("gfx_arch", "gfx90a")
    ncu = int(dev_meta.get("num_compute_units") or 0)
    tbl = DEVICE_TABLE.get(gfx, {})
    best = None
    if tbl:
        best = tbl.get(ncu) or tbl[min(tbl, key=lambda k: abs(k - ncu) if ncu else k)]
    best = best or dict(name=gfx, peak_bw=1.6e12, peak_fp64=23.9e12)

    peak_bw = best["peak_bw"] if args.peak_bw == "auto" else float(args.peak_bw)
    peak_fp64 = (best["peak_fp64"] if args.peak_fp64 == "auto"
                 else float(args.peak_fp64))

    rows = analyse(dev_meta, results, peak_bw, args.achievable_frac,
                   peak_fp64, best["name"])

    print(f"\ndevice: {best['name']}  ({gfx}, {ncu} CU)"
          f"   peak HBM {peak_bw/1e12:.2f} TB/s"
          f"   peak FP64 {peak_fp64/1e12:.1f} TF/s\n")
    print_table(rows)

    print("\n--- diagnosis " + "-" * 60)
    for matrix, msgs in diagnose(rows):
        print(f"\n{matrix}")
        for m in msgs:
            print(f"  * {m}")
    print()

    prefix = args.out_prefix or os.path.join(args.outdir, "report")
    with open(prefix + ".json", "w") as fh:
        json.dump({"device": {**best, "gfx": gfx, "num_cu": ncu},
                   "meta": dev_meta, "rows": rows}, fh, indent=2, default=str)
    print(f"wrote {prefix}.json")

    if args.markdown:
        with open(prefix + ".md", "w") as fh:
            fh.write(f"# AMP SpMV profile -- {best['name']}\n\n```\n")
            print_table(rows, stream=fh)
            fh.write("```\n\n## Diagnosis\n\n")
            for matrix, msgs in diagnose(rows):
                fh.write(f"### {matrix}\n\n")
                for m in msgs:
                    fh.write(f"- {m}\n")
                fh.write("\n")
        print(f"wrote {prefix}.md")


if __name__ == "__main__":
    main()
