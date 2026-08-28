#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
#
# amp_spmv_profile.sh -- collect AMD hardware counters for Ginkgo/AMPLify SpMV
#                        kernels on MI250X (Frontier compute node) or MI210
#                        (login/dev node).
#
# Drives ./benchmark/spmv from the Ginkgo build tree under rocprofv3 and/or
# rocprof-compute, one matrix at a time, one format at a time, with the counter
# sets split into single-pass groups so that the numbers within a group are
# self-consistent (no counter multiplexing / kernel replay inside a group).
#
# Output layout:
#   <outdir>/device.json
#   <outdir>/<matrix>/<format>/timing/{*kernel_trace.csv, bench.json, stdout.log}
#   <outdir>/<matrix>/<format>/pmc_<set>/*counter_collection.csv
#   <outdir>/<matrix>/<format>/roc-compute/...          (if --tool includes it)
#
# Post-process with amp_spmv_report.py.

set -uo pipefail
exec 9>&2          # keep a handle on the real stderr for --dry-run listings

die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }
warn() { printf '\033[33mwarn:\033[0m %s\n'  "$*" >&2; }
info() { printf '\033[36m==>\033[0m %s\n'    "$*" >&2; }

# ---------------------------------------------------------------- defaults ---
# Every default can also come from the environment, using the same variable
# names as the other job scripts in this repo (see ../README.md), so this can
# be driven either way:
#     GINKGO_BUILD_DIR=... EXECUTOR=hip ./amp_spmv_profile.sh --matrix-list ...
#     ./amp_spmv_profile.sh --build-dir ... --executor hip --matrix-list ...
BUILD_DIR="${GINKGO_BUILD_DIR:-}"
SYSTEM_NAME="${SYSTEM_NAME:-unspecified}"
FORMATS="${FORMATS:-csr,amp}"
AMP_BASE="${AMP_BASE_TYPE:-csr}"
AMP_TOL="${AMP_TOLERANCE:-1e-9}"
AMP_TOL_TYPE="${AMP_TOLERANCE_TYPE:-componentwise}"
EXECUTOR="${EXECUTOR:-hip}"
DEVICE_ID=0
WARMUP="${WARMUP:-3}"
REPS="${REPETITIONS:-20}"
OUTDIR=""          # filled in after argument parsing, from RESULTS_DIR
KERNEL_RE=""            # empty -> auto per format
MATRICES=()
MATRIX_LIST=""
TOOL="auto"             # auto | rocprofv3 | rocprof-compute | both
LAUNCH="auto"           # auto | srun | local
SRUN_EXTRA="-n1 -c7 --gpus-per-task=1 --gpu-bind=closest"
PMC_SETS="all"
VALIDATE_COUNTERS=1
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage: amp_spmv_profile.sh (--matrix FILE.mtx | --matrix-list FILE) [options]

Environment (same names as the other job scripts in this repo; flags override):
  GINKGO_BUILD_DIR   Ginkgo build tree containing benchmark/spmv   (required)
  EXECUTOR           hip | cuda                                    [hip]
  RESULTS_DIR        Where to put the run directory                [$PWD]
  SYSTEM_NAME        Tag for the results directory name            [unspecified]
  AMP_BASE_TYPE / AMP_TOLERANCE / AMP_TOLERANCE_TYPE / FORMATS / REPETITIONS

Required:
  --matrix FILE          MatrixMarket file to profile (repeatable)
  --matrix-list FILE     File with one .mtx path per line (# comments ok)

Workload:
  --build-dir DIR        Ginkgo build tree           [$GINKGO_BUILD_DIR]
  --formats LIST         Comma list for benchmark/spmv    [csr,amp]
  --amp-base-type T      ell | csr                        [csr]
  --amp-tolerance T      AMP tolerance                    [1e-9]
  --amp-tolerance-type T componentwise | normwise         [componentwise]
  --executor E           hip | cuda                       [hip]
  --device-id N          Ginkgo device id                 [0]
  --warmup N             Warmup reps                      [3]
  --repetitions N        Timed reps                       [20]
  --system-name NAME     Tag used in the results directory name
  --kernel REGEX         Kernel name filter; default is auto-chosen per format

Profiling:
  --tool T               auto | rocprofv3 | rocprof-compute | both   [auto]
  --pmc-sets LIST        all | comma list of: wave,stall,l1,l2,scalar,fp  [all]
  --no-validate          Skip checking counters against `rocprofv3 --list-avail`
  --outdir DIR           Output directory
                         [$RESULTS_DIR/results-profile-spmv-<base>-<system>]

Launch:
  --launch M             auto | srun | local   [auto: srun if SLURM_JOB_ID or sbatch env]
  --srun-extra "..."     Extra srun flags  [-n1 -c7 --gpus-per-task=1 --gpu-bind=closest]
  --dry-run              Print commands, run nothing
  -h, --help             This help

Counter collection needs exclusive use of one GCD.  On Frontier always profile
with a single rank bound to a single GCD; this script sets ROCR_VISIBLE_DEVICES
and GPU_MAX_HW_QUEUES=1 for you.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-dir)           BUILD_DIR="$2"; shift 2 ;;
        --matrix)              MATRICES+=("$2"); shift 2 ;;
        --matrix-list)         MATRIX_LIST="$2"; shift 2 ;;
        --formats)             FORMATS="$2"; shift 2 ;;
        --amp-base-type)       AMP_BASE="$2"; shift 2 ;;
        --amp-tolerance)       AMP_TOL="$2"; shift 2 ;;
        --amp-tolerance-type)  AMP_TOL_TYPE="$2"; shift 2 ;;
        --executor)            EXECUTOR="$2"; shift 2 ;;
        --device-id)           DEVICE_ID="$2"; shift 2 ;;
        --warmup)              WARMUP="$2"; shift 2 ;;
        --repetitions)         REPS="$2"; shift 2 ;;
        --system-name)         SYSTEM_NAME="$2"; shift 2 ;;
        --kernel)              KERNEL_RE="$2"; shift 2 ;;
        --tool)                TOOL="$2"; shift 2 ;;
        --pmc-sets)            PMC_SETS="$2"; shift 2 ;;
        --no-validate)         VALIDATE_COUNTERS=0; shift ;;
        --outdir)              OUTDIR="$2"; shift 2 ;;
        --launch)              LAUNCH="$2"; shift 2 ;;
        --srun-extra)          SRUN_EXTRA="$2"; shift 2 ;;
        --dry-run)             DRY_RUN=1; shift ;;
        -h|--help)             usage; exit 0 ;;
        *) die "unknown argument: $1 (try --help)" ;;
    esac
done

[[ -n "$BUILD_DIR" ]] || { usage; die "set GINKGO_BUILD_DIR or pass --build-dir"; }
[[ -n "$OUTDIR" ]] || \
    OUTDIR="${RESULTS_DIR:-$PWD}/results-profile-spmv-${AMP_BASE}-${SYSTEM_NAME}"
SPMV="$BUILD_DIR/benchmark/spmv"
[[ -x "$SPMV" ]] || die "not executable: $SPMV"

if [[ -n "$MATRIX_LIST" ]]; then
    [[ -r "$MATRIX_LIST" ]] || die "cannot read --matrix-list $MATRIX_LIST"
    while IFS= read -r line; do
        line="${line%%#*}"; line="${line//[$'\t\r\n']/}"
        [[ -n "${line// /}" ]] && MATRICES+=("$line")
    done < "$MATRIX_LIST"
fi
[[ ${#MATRICES[@]} -gt 0 ]] || die "no matrices given (--matrix / --matrix-list)"

mkdir -p "$OUTDIR" || die "cannot create $OUTDIR"
OUTDIR="$(cd "$OUTDIR" && pwd)"

# ------------------------------------------------------------------ launch ---
if [[ "$LAUNCH" == "auto" ]]; then
    if [[ -n "${SLURM_STEP_ID:-}${SLURM_PROCID:-}" ]]; then
        # already inside an srun step -- nesting srun would deadlock
        LAUNCH="local"
    elif [[ -n "${SLURM_JOB_ID:-}" ]] && command -v srun >/dev/null 2>&1; then
        LAUNCH="srun"
    else
        LAUNCH="local"
    fi
fi
info "launch mode: $LAUNCH"

# run CMD...  -- honours $LAUNCH and $DRY_RUN
run() {
    local -a cmd=()
    if [[ "$LAUNCH" == "srun" ]]; then
        # shellcheck disable=SC2206
        cmd=(srun $SRUN_EXTRA "$@")
    else
        cmd=("$@")
    fi
    if [[ $DRY_RUN -eq 1 ]]; then
        # fd 9 is the script's original stderr, so the per-command redirections
        # below do not swallow the dry-run listing
        { printf '   '; printf '%q ' "${cmd[@]}"; printf '\n'; } >&9
        return 0
    fi
    "${cmd[@]}"
}

# ------------------------------------------------------------- tool probes ---
HAVE_RPV3=0; HAVE_RPV1=0; HAVE_RCOMP=0
command -v rocprofv3       >/dev/null 2>&1 && HAVE_RPV3=1
command -v rocprof         >/dev/null 2>&1 && HAVE_RPV1=1
command -v rocprof-compute >/dev/null 2>&1 && HAVE_RCOMP=1
command -v omniperf        >/dev/null 2>&1 && HAVE_RCOMP=2   # legacy name

if [[ "$TOOL" == "auto" ]]; then
    if   [[ $HAVE_RPV3 -eq 1 && $HAVE_RCOMP -ne 0 ]]; then TOOL="both"
    elif [[ $HAVE_RPV3 -eq 1 ]];                     then TOOL="rocprofv3"
    elif [[ $HAVE_RCOMP -ne 0 ]];                    then TOOL="rocprof-compute"
    else die "neither rocprofv3 nor rocprof-compute found on PATH; module load rocm/... first"
    fi
fi
RCOMP_BIN="rocprof-compute"; [[ $HAVE_RCOMP -eq 2 ]] && RCOMP_BIN="omniperf"
info "profiling tool(s): $TOOL"

# rocprofv3 output flag spelling differs between ROCm minor versions.
RPV3_DIR_FLAG="--output-directory"; RPV3_FILE_FLAG="--output-file"
if [[ $HAVE_RPV3 -eq 1 ]]; then
    RPV3_HELP="$(rocprofv3 --help 2>&1 || true)"
    grep -q -- '--output-directory' <<<"$RPV3_HELP" || RPV3_DIR_FLAG="-d"
    grep -q -- '--output-file'      <<<"$RPV3_HELP" || RPV3_FILE_FLAG="-o"
    RPV3_KFILTER=""
    if   grep -q -- '--kernel-include-regex' <<<"$RPV3_HELP"; then RPV3_KFILTER="--kernel-include-regex"
    elif grep -q -- '--kernel-filter-include' <<<"$RPV3_HELP"; then RPV3_KFILTER="--kernel-filter-include"
    fi
fi

# ------------------------------------------------------------ device record --
GFX="unknown"; NUM_CU=0
if command -v rocminfo >/dev/null 2>&1; then
    RI="$(rocminfo 2>/dev/null || true)"
    GFX="$(grep -oE 'gfx[0-9a-f]+' <<<"$RI" | head -1)"
    NUM_CU="$(awk '/Compute Unit:/ {print $3; exit}' <<<"$RI")"
fi
cat > "$OUTDIR/device.json" <<EOF
{
  "gfx_arch": "${GFX:-unknown}",
  "num_compute_units": ${NUM_CU:-0},
  "system_name": "$SYSTEM_NAME",
  "executor": "$EXECUTOR",
  "hostname": "$(hostname)",
  "rocm_path": "${ROCM_PATH:-}",
  "slurm_job_id": "${SLURM_JOB_ID:-}",
  "amp_tolerance": "$AMP_TOL",
  "amp_base_type": "$AMP_BASE",
  "warmup": $WARMUP,
  "repetitions": $REPS
}
EOF
info "device: ${GFX:-unknown}, ${NUM_CU:-?} CUs"

# ---------------------------------------------------------------- counters ---
# Six single-pass groups.  Kept small enough that gfx90a can retire each group
# in one pass; if rocprofiler still reports multiplexing, split further.
declare -A PMC
# (a) Wavefront / VALU lane utilisation -- tests the "64-wide wave per row is
#     mostly idle" hypothesis directly.  VALUUtilization is the headline number.
PMC[wave]="SQ_WAVES SQ_INSTS_VALU SQ_ACTIVE_INST_VALU SQ_THREAD_CYCLES_VALU SQ_INSTS_SALU SQ_INSTS_VMEM_RD SQ_INSTS_SMEM SQ_BUSY_CYCLES"
# (b) Latency hiding: how much of a wave's life is spent waiting, and occupancy.
PMC[stall]="SQ_WAVES SQ_WAVE_CYCLES SQ_WAIT_ANY SQ_BUSY_CYCLES SQ_ACCUM_PREV_HIRES GRBM_GUI_ACTIVE GRBM_COUNT"
# (c) Vector L1 (TCP, 16 KB/CU).  Hit rate on the x-gather lives here.
PMC[l1]="TCP_TOTAL_CACHE_ACCESSES_sum TCP_TCC_READ_REQ_sum TCP_TCC_WRITE_REQ_sum TCP_PENDING_STALL_CYCLES_sum TCP_TCP_TA_DATA_STALL_CYCLES"
# (d) L2 (TCC, 8 MB/GCD) + HBM bytes.  FetchSize/WriteSize are derived from EA.
PMC[l2]="TCC_HIT_sum TCC_MISS_sum TCC_EA_RDREQ_sum TCC_EA_RDREQ_32B_sum TCC_EA_WRREQ_sum TCC_EA_WRREQ_64B_sum TCC_BUSY_sum TCC_TAG_STALL_sum"
# (e) Scalar caches.  AMP[CSR] loads q row-pointer pairs per row through SMEM;
#     if the compiler scalarised them they show up here, not in TCP.
PMC[scalar]="SQC_DCACHE_REQ SQC_DCACHE_HITS SQC_ICACHE_REQ SQC_ICACHE_HITS SQ_INSTS_SMEM"
# (f) FP mix -- confirms how much real f64/f32 work is issued vs conversions.
PMC[fp]="SQ_INSTS_VALU_ADD_F64 SQ_INSTS_VALU_MUL_F64 SQ_INSTS_VALU_FMA_F64 SQ_INSTS_VALU_ADD_F32 SQ_INSTS_VALU_MUL_F32 SQ_INSTS_VALU_FMA_F32 SQ_INSTS_VALU_CVT SQ_INSTS_VALU"

if [[ "$PMC_SETS" == "all" ]]; then
    SETS=(wave stall l1 l2 scalar fp)
else
    IFS=',' read -r -a SETS <<< "$PMC_SETS"
fi

AVAIL=""
if [[ $VALIDATE_COUNTERS -eq 1 && $HAVE_RPV3 -eq 1 && $DRY_RUN -eq 0 ]]; then
    info "querying available counters (rocprofv3 --list-avail)"
    AVAIL="$(rocprofv3 --list-avail 2>/dev/null || true)"
fi

filter_counters() {  # $1 = space separated list -> echoes surviving list
    local out=() c
    for c in $1; do
        if [[ -z "$AVAIL" ]] || grep -qw -- "$c" <<<"$AVAIL"; then
            out+=("$c")
        else
            warn "counter not available on this agent, dropping: $c"
        fi
    done
    echo "${out[*]}"
}

# --------------------------------------------------------------- benchmark ---
default_kernel_re() {
    case "$1" in
        amp) [[ "$AMP_BASE" == "csr" ]] && echo "csr_amp_.*spmv" || echo "ell_amp_.*spmv" ;;
        csr) echo "abstract_(classical|load_balance|merge_path)_spmv|csr_spmv" ;;
        ell) echo "spmv_kernel|abstract_spmv" ;;
        *)   echo "spmv" ;;
    esac
}

bench_argv() {  # $1 format, $2 matrix
    local fmt="$1" mtx="$2"
    printf '%s\0' "$SPMV" \
        "--executor=$EXECUTOR" "--device_id=$DEVICE_ID" \
        "--formats=$fmt" \
        "--amp_base_type=$AMP_BASE" \
        "--amp_tolerance=$AMP_TOL" \
        "--amp_tolerance_type=$AMP_TOL_TYPE" \
        "--input_matrix=$mtx" \
        "--warmup=$WARMUP" "--repetitions=$REPS" \
        "--detailed=false" "--profiler_hook=none"
}

export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-$DEVICE_ID}"
export GPU_MAX_HW_QUEUES=1
export HSA_ENABLE_SDMA="${HSA_ENABLE_SDMA:-0}"
# benchmark/spmv addresses the *visible* device, so it is always index 0 here.
DEVICE_ID=0

IFS=',' read -r -a FMT_ARR <<< "$FORMATS"

total=$(( ${#MATRICES[@]} * ${#FMT_ARR[@]} )); done_n=0

for mtx in "${MATRICES[@]}"; do
    [[ -r "$mtx" ]] || { warn "skipping unreadable matrix: $mtx"; continue; }
    mname="$(basename "${mtx%.mtx}")"
    for fmt in "${FMT_ARR[@]}"; do
        done_n=$((done_n+1))
        kre="${KERNEL_RE:-$(default_kernel_re "$fmt")}"
        base="$OUTDIR/$mname/$fmt"
        mkdir -p "$base"
        info "[$done_n/$total] $mname :: $fmt  (kernel filter: $kre)"

        mapfile -d '' -t ARGV < <(bench_argv "$fmt" "$mtx")

        # ---- pass 0: timing + kernel trace, no counters (undisturbed clocks)
        tdir="$base/timing"; mkdir -p "$tdir"
        if [[ $HAVE_RPV3 -eq 1 ]]; then
            run rocprofv3 --kernel-trace --output-format csv \
                "$RPV3_DIR_FLAG" "$tdir" "$RPV3_FILE_FLAG" "trace" \
                -- "${ARGV[@]}" > "$tdir/stdout.log" 2> "$tdir/stderr.log"
        else
            run "${ARGV[@]}" > "$tdir/stdout.log" 2> "$tdir/stderr.log"
        fi
        # benchmark/spmv prints the result JSON on stdout; keep it for nnz,
        # per-bin sizes (amp_bins) and Ginkgo's own wall-clock number.
        sed -n '/^[[{]/,$p' "$tdir/stdout.log" > "$tdir/bench.json" 2>/dev/null || true

        # ---- passes 1..n: counters, one single-pass group at a time
        if [[ "$TOOL" == "rocprofv3" || "$TOOL" == "both" ]]; then
            [[ $HAVE_RPV3 -eq 1 ]] || warn "rocprofv3 requested but not found"
            for s in "${SETS[@]}"; do
                [[ -n "${PMC[$s]:-}" ]] || { warn "unknown pmc set '$s'"; continue; }
                cnt="$(filter_counters "${PMC[$s]}")"
                [[ -n "$cnt" ]] || { warn "pmc set '$s' empty after filtering"; continue; }
                pdir="$base/pmc_$s"; mkdir -p "$pdir"
                # shellcheck disable=SC2086
                if [[ -n "${RPV3_KFILTER:-}" ]]; then
                    run rocprofv3 --pmc $cnt "$RPV3_KFILTER" "$kre" \
                        --output-format csv \
                        "$RPV3_DIR_FLAG" "$pdir" "$RPV3_FILE_FLAG" "$s" \
                        -- "${ARGV[@]}" > "$pdir/stdout.log" 2> "$pdir/stderr.log"
                else
                    run rocprofv3 --pmc $cnt --output-format csv \
                        "$RPV3_DIR_FLAG" "$pdir" "$RPV3_FILE_FLAG" "$s" \
                        -- "${ARGV[@]}" > "$pdir/stdout.log" 2> "$pdir/stderr.log"
                fi
                if [[ $DRY_RUN -eq 0 ]] && ! compgen -G "$pdir/**/*counter_collection.csv" >/dev/null \
                   && ! compgen -G "$pdir/*counter_collection.csv" >/dev/null; then
                    warn "no counter CSV produced for set '$s' -- see $pdir/stderr.log"
                fi
            done
        fi

        # ---- rocprof-compute: full memory chart + roofline for this kernel
        if [[ "$TOOL" == "rocprof-compute" || "$TOOL" == "both" ]]; then
            if [[ $HAVE_RCOMP -eq 0 ]]; then
                warn "rocprof-compute requested but not found"
            else
                cdir="$base/roc-compute"; mkdir -p "$cdir"
                run "$RCOMP_BIN" profile -n "${mname}_${fmt}" \
                    -k "$kre" --path "$cdir" \
                    -- "${ARGV[@]}" > "$cdir/profile.log" 2>&1
                # analyze: speed-of-light, wavefront, instr mix, L1, L2, L2-fabric
                wl="$(find "$cdir" -maxdepth 3 -type d -name 'MI*' 2>/dev/null | head -1)"
                if [[ -n "$wl" ]]; then
                    run "$RCOMP_BIN" analyze -p "$wl" \
                        > "$cdir/analyze_full.txt" 2>&1
                    run "$RCOMP_BIN" analyze -p "$wl" -b 2 7 10 11 14 15 16 17 18 \
                        > "$cdir/analyze_blocks.txt" 2>&1
                elif [[ $DRY_RUN -eq 0 ]]; then
                    warn "no rocprof-compute workload dir under $cdir"
                fi
            fi
        fi
    done
done

info "done. results in $OUTDIR"
info "next: python3 amp_spmv_report.py $OUTDIR --peak-bw auto -o $OUTDIR/report"
