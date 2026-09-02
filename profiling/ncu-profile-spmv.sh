#!/usr/bin/env bash
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
FORMATS="${FORMATS:-csrc,amp}"
AMP_BASE="${AMP_BASE_TYPE:-csr}"
AMP_TOL="${AMP_TOLERANCE:-1e-9}"
AMP_TOL_TYPE="${AMP_TOLERANCE_TYPE:-componentwise}"
EXECUTOR="${EXECUTOR:-cuda}"
BENCH_PRECISION="${BENCHMARK_PRECISION:-double}"
DEVICE_ID=0
WARMUP="${WARMUP:-3}"
REPS="${REPETITIONS:-10}"
OUTDIR=""          # filled in after argument parsing, from RESULTS_DIR
KERNEL_RE=""            # empty -> auto per format
MATRICES=()
MATRIX_LIST=""
LAUNCH="auto"           # auto | srun | local
SRUN_EXTRA="-n1 -c7 --gpus-per-task=1 --gpu-bind=closest"
NCU_SETS="${NCU_SETS:-detailed}"   # ncu --set value
DRY_RUN=0

CSR_CLASSICAL_KERNEL_NAME=abstract_classical_spmv

usage() {
    cat <<'EOF'
Usage: amp_spmv_profile.sh (--matrix FILE.mtx | --matrix-list FILE) [options]

Environment (same names as the other job scripts in this repo; flags override):
  GINKGO_BUILD_DIR   Ginkgo build tree containing benchmark/spmv   (required)
  RESULTS_DIR        Where to put the run directory                [$PWD]
  SYSTEM_NAME        Tag for the results directory name            [unspecified]
  BENCHMARK_PRECISION  double | single | dcomplex | scomplex        [double]
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
  --precision P          double | single | dcomplex | scomplex, selecting
                         benchmark/spmv/spmv[_suffix]     [double]
  --device-id N          Ginkgo device id                 [0]
  --warmup N             Warmup reps                      [3]
  --repetitions N        Timed reps                       [20]
  --system-name NAME     Tag used in the results directory name

Profiling:
  --ncu-sets SET         ncu --set value: basic | launch | memory | source |
                         roofline | full | detailed                  [detailed]
  --outdir DIR           Output directory
                         [$RESULTS_DIR/results-profile-spmv-<base>-<system>]

Launch:
  --launch M             auto | srun | local   [auto: srun if SLURM_JOB_ID or sbatch env]
  --srun-extra "..."     Extra srun flags  [-n1 -c7 --gpus-per-task=1 --gpu-bind=closest]
  --dry-run              Print commands, run nothing
  -h, --help             This help

Counter collection needs exclusive use of one GPU; this script sets
CUDA_VISIBLE_DEVICES for you based on --device-id.
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
        --precision)           BENCH_PRECISION="$2"; shift 2 ;;
        --device-id)           DEVICE_ID="$2"; shift 2 ;;
        --warmup)              WARMUP="$2"; shift 2 ;;
        --repetitions)         REPS="$2"; shift 2 ;;
        --system-name)         SYSTEM_NAME="$2"; shift 2 ;;
        --ncu-sets)            NCU_SETS="$2"; shift 2 ;;
        --outdir)              OUTDIR="$2"; shift 2 ;;
        --launch)              LAUNCH="$2"; shift 2 ;;
        --srun-extra)          SRUN_EXTRA="$2"; shift 2 ;;
        --dry-run)             DRY_RUN=1; shift ;;
        -h|--help)             usage; exit 0 ;;
        *) die "unknown argument: $1 (try --help)" ;;
    esac
done

# Same BENCHMARK_PRECISION -> suffix mapping as benchmark/run_all_benchmarks.sh
case "$BENCH_PRECISION" in
    double)   BENCH_SUFFIX="" ;;
    single)   BENCH_SUFFIX="_single" ;;
    dcomplex) BENCH_SUFFIX="_dcomplex" ;;
    scomplex) BENCH_SUFFIX="_scomplex" ;;
    *) die "BENCHMARK_PRECISION is set to the not supported \"$BENCH_PRECISION\"."\
"  Supported: double, single, dcomplex, scomplex" ;;
esac

[[ -n "$BUILD_DIR" ]] || { usage; die "set GINKGO_BUILD_DIR or pass --build-dir"; }
[[ -n "$OUTDIR" ]] || \
    OUTDIR="${RESULTS_DIR:-$PWD}/results-profile-spmv-${AMP_BASE}-${SYSTEM_NAME}"

SPMV_EXEC="$BUILD_DIR/benchmark/spmv/spmv$BENCH_SUFFIX"
[[ -f "$SPMV_EXEC" && -x "$SPMV_EXEC" ]] || die "spmv benchmark binary not found or not executable: $SPMV_EXEC
  Check GINKGO_BUILD_DIR ($BUILD_DIR) and that benchmarks were built."

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

bench_argv() {  # $1 format, $2 matrix
    local fmt="$1" mtx="$2"
    printf '%s\0' "$SPMV_EXEC" \
        "--executor=$EXECUTOR" "--device_id=$DEVICE_ID" \
        "--formats=$fmt" \
        "--amp_base_type=$AMP_BASE" \
        "--amp_tolerance=$AMP_TOL" \
        "--amp_tolerance_type=$AMP_TOL_TYPE" \
        "--input_matrix=$mtx" \
        "--warmup=$WARMUP" "--repetitions=$REPS" \
        "--detailed=false" "--profiler_hook=nvtx"
}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$DEVICE_ID}"

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

IFS=',' read -r -a FMT_ARR <<< "$FORMATS"

total=$(( ${#MATRICES[@]} * ${#FMT_ARR[@]} )); done_n=0

for mtx in "${MATRICES[@]}"; do
    [[ -r "$mtx" ]] || { warn "skipping unreadable matrix: $mtx"; continue; }
    mname="$(basename "${mtx%.mtx}")"
    for fmt in "${FMT_ARR[@]}"; do
        done_n=$((done_n+1))
        kre="regex:spmv"
		if [ "$fmt" = "csrc" ]; then
			kre="regex:$CSR_CLASSICAL_KERNEL_NAME"
		fi
        base="$OUTDIR/$mname/$fmt"
        mkdir -p "$base"
        info "[$done_n/$total] $mname :: $fmt  (kernel filter: ${kre}, set: ${NCU_SETS})"

        mapfile -d '' -t ARGV < <(bench_argv "$fmt" "$mtx")

        run ncu --devices "$DEVICE_ID" -k "$kre" --set "$NCU_SETS" \
            -o "$base/$NCU_SETS" --page details "${ARGV[@]}" \
            > >(tee "$base/console_out.log") 2>&1
	done
done
