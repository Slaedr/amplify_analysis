#!/usr/bin/env bash
#
# Submits Slurm jobs that run ginkgo's benchmark/run_all_benchmarks.sh with
# BENCHMARK=spmv to measure the speedup and accuracy of AMP[<BASE_FORMAT>] with
# respect to <BASE_FORMAT> in FP64, at several AMP tolerances, plus the speedup
# of <BASE_FORMAT> in FP32 with respect to <BASE_FORMAT> in FP64.
#
# One job is submitted per AMP tolerance, and one more ("fp32") for the
# single-precision comparison:
#
#   job        passes (each pass = one run_all_benchmarks.sh call, own RESULTS_DIR)
#   tol_<N>    tol_<N>/         FORMATS=<BASE_FORMAT>,<AMP_FORMAT>  (FP64)
#              -> every JSON holds the FP64 base result and the AMP result at
#                 that tolerance, timed back to back on the same node
#   fp32       double_precision/ FORMATS=<BASE_FORMAT>   BENCHMARK_PRECISION=double
#              single_precision/ FORMATS=<BASE_FORMAT>   BENCHMARK_PRECISION=single
#              -> FP32 and FP64 of the base format, same node, same job
#
# Directory tree produced (RESULTS_DIR of each pass is one of the first-level
# directories below, so run_all_benchmarks.sh writes <RESULTS_DIR>/<group>/<name>.json):
#
#   $RESULTS_ROOT/
#     sweep_manifest.txt            settings + one line per submitted job
#     tol_9/                        (N = -log10 of the tolerance)
#       job.sbatch
#       slurm-<jobid>.out
#       Janna/Serena.json           spmv.<base> and spmv.<amp> (amp_config has
#       VLSI/nv2.json               the tolerance), matrix stats, ...
#     tol_10/ ...
#     double_precision/Janna/Serena.json   spmv.<base>, FP64
#     single_precision/Janna/Serena.json   spmv.<base>, FP32
#     fp32_job/                     job.sbatch + slurm-<jobid>.out of the fp32 job
#
# run_all_benchmarks.sh never overwrites: a matrix whose result file already
# exists in a pass's RESULTS_DIR is skipped. To resume an interrupted sweep,
# just run this script again with the same RESULTS_ROOT.
#
# Plot with plot_spmv_speedup_vs_tolerance.py --results-dir $RESULTS_ROOT.
#
# Run this on a login node: SuiteSparse matrices are resolved/extracted with
# ssget *before* submission (compute nodes usually have no internet). This
# script never calls `ssget -c`, and the jobs run with KEEP_MATRICES=true so
# concurrent jobs never delete each other's matrices.
#
# Required:
#   GINKGO_BUILD_DIR   Ginkgo build dir containing benchmark/spmv/spmv (and
#                      benchmark/spmv/spmv_single and
#                      benchmark/matrix_statistics/matrix_statistics{,_single}
#                      for the fp32 job)
#   MATRIX_LIST_FILE   Text file, one SuiteSparse id, name or group/name per
#                      line. No default: it must be set explicitly.
#   SLURM_ACCOUNT      project to charge (not needed with DRY_RUN=1)
#
# Experiment (optional; defaults in brackets):
#   AMP_TOLERANCES     space-separated AMP tolerances     ["1e-6 1e-8 1e-9 1e-10"]
#   INCLUDE_FP32       1 = also submit the fp32 job                          [1]
#   BASE_FORMAT        csrc | ell -- base format, also the storage under AMP [csrc]
#   AMP_FORMAT         amp | ampib                                          [amp]
#   AMP_TOLERANCE_TYPE componentwise | normwise                [componentwise]
#   AMP_CSR_STRATEGY   automatical|classical|load_balance|merge_path
#                                                                   [automatical]
#   AMP_HIGH_PRECISION_DIAGONAL   unset = the AMP class's default
#   AMP_BIN_FOLDUP_NNZ_RATIO      unset = the AMP class's default
#   REPETITIONS        timed repetitions                                    [10]
#   DETAILED           1 = also record max_relative_norm2 (error vs. the
#                      default format in the same precision); needed for the
#                      error markers in the plot                             [1]
#   EXECUTOR           hip | cuda | omp | reference                        [hip]
#   SYSTEM_NAME        only used for naming                          [frontier]
#   DEVICE_ID          passed through                                       [0]
#   RUN_ALL_BENCHMARKS path to run_all_benchmarks.sh
#                      [$GINKGO_BUILD_DIR/benchmark/run_all_benchmarks.sh]
#   RESULTS_ROOT       output root
#                      [./results-<system>-spmv-<base>-<amp>]
#   PREFETCH           1 = resolve and extract matrices with ssget on this node
#                      before submitting                                    [1]
#   SSGET              ssget executable                                [ssget]
#   DRY_RUN            1 = write everything, submit nothing                 [0]
#
# Slurm (optional):
#   SLURM_PARTITION [batch]   SLURM_TIME [02:00:00]   SLURM_QOS [unset]
#   SLURM_NODES [1]           SBATCH_EXTRA_ARGS extra sbatch arguments
#   SRUN_ARGS ["-N1 -n1 -c7 --gpus-per-task=1 --gpu-bind=closest"]
#   JOB_SETUP shell commands run at the top of every job, e.g. "module load rocm"
#
# Example:
#   GINKGO_BUILD_DIR=$HOME/ginkgo/build SLURM_ACCOUNT=abc123 \
#   JOB_SETUP="module load rocm" MATRIX_LIST_FILE=matrices.txt \
#       ./submit_spmv_amp_sweep.sh

set -uo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

die() { echo "ERROR: $*" >&2; exit 1; }
warn() { echo "WARNING: $*" >&2; }

# --- Settings -----------------------------------------------------------------

AMP_TOLERANCES=${AMP_TOLERANCES:-"1e-6 1e-8 1e-9 1e-10"}
INCLUDE_FP32=${INCLUDE_FP32:-1}
BASE_FORMAT=${BASE_FORMAT:-csrc}
AMP_FORMAT=${AMP_FORMAT:-amp}
AMP_TOLERANCE_TYPE=${AMP_TOLERANCE_TYPE:-componentwise}
AMP_CSR_STRATEGY=${AMP_CSR_STRATEGY:-automatical}
AMP_HIGH_PRECISION_DIAGONAL=${AMP_HIGH_PRECISION_DIAGONAL:-}
AMP_BIN_FOLDUP_NNZ_RATIO=${AMP_BIN_FOLDUP_NNZ_RATIO:-}
REPETITIONS=${REPETITIONS:-10}
DETAILED=${DETAILED:-1}
EXECUTOR=${EXECUTOR:-hip}
SYSTEM_NAME=${SYSTEM_NAME:-frontier}
DEVICE_ID=${DEVICE_ID:-0}
MATRIX_LIST_FILE=${MATRIX_LIST_FILE:-}
SSGET=${SSGET:-ssget}
PREFETCH=${PREFETCH:-1}
DRY_RUN=${DRY_RUN:-0}
PYTHON=${PYTHON:-python3}
RUN_ALL_BENCHMARKS=${RUN_ALL_BENCHMARKS:-${GINKGO_BUILD_DIR:-}/benchmark/run_all_benchmarks.sh}

SLURM_ACCOUNT=${SLURM_ACCOUNT:-}
SLURM_PARTITION=${SLURM_PARTITION:-batch}
SLURM_TIME=${SLURM_TIME:-02:00:00}
SLURM_QOS=${SLURM_QOS:-}
SLURM_NODES=${SLURM_NODES:-1}
SBATCH_EXTRA_ARGS=${SBATCH_EXTRA_ARGS:-}
SRUN_ARGS=${SRUN_ARGS:-"-N1 -n1 -c7 --gpus-per-task=1 --gpu-bind=closest"}
JOB_SETUP=${JOB_SETUP:-}

RESULTS_ROOT=${RESULTS_ROOT:-${PWD}/results-${SYSTEM_NAME}-spmv-${BASE_FORMAT}-${AMP_FORMAT}}

# --- Validation ---------------------------------------------------------------

command -v "${PYTHON}" >/dev/null 2>&1 || die "need python3 (or set PYTHON=)"
[ -n "${GINKGO_BUILD_DIR:-}" ] || die "set GINKGO_BUILD_DIR"
[ -d "${GINKGO_BUILD_DIR}/benchmark" ] ||
    die "'${GINKGO_BUILD_DIR}/benchmark' is not a directory"
[ -n "${MATRIX_LIST_FILE}" ] ||
    die "set MATRIX_LIST_FILE (there is no default matrix list)"
[ -f "${MATRIX_LIST_FILE}" ] || die "matrix list '${MATRIX_LIST_FILE}' not found"
[ -f "${RUN_ALL_BENCHMARKS}" ] ||
    die "run_all_benchmarks.sh not found at '${RUN_ALL_BENCHMARKS}' (defaults to \$GINKGO_BUILD_DIR/benchmark/run_all_benchmarks.sh; set RUN_ALL_BENCHMARKS to override)"
case "${BASE_FORMAT}" in csrc|ell) ;; *) die "BASE_FORMAT must be csrc or ell, got '${BASE_FORMAT}'" ;; esac
case "${AMP_FORMAT}" in amp|ampib) ;; *) die "AMP_FORMAT must be amp or ampib, got '${AMP_FORMAT}'" ;; esac
case "${AMP_TOLERANCE_TYPE}" in componentwise|normwise) ;; *) die "AMP_TOLERANCE_TYPE must be componentwise or normwise" ;; esac
case "${DETAILED}" in 0|1) ;; *) die "DETAILED must be 0 or 1" ;; esac

for tol in ${AMP_TOLERANCES}; do
    "${PYTHON}" -c 'import sys; v = float(sys.argv[1]); assert v > 0' "${tol}" \
        2>/dev/null || die "invalid AMP tolerance '${tol}'"
done
[ -n "${AMP_TOLERANCES}" ] || [ "${INCLUDE_FP32}" = "1" ] ||
    die "nothing to run: AMP_TOLERANCES is empty and INCLUDE_FP32 != 1"
if [ "${DRY_RUN}" != "1" ]; then
    [ -n "${SLURM_ACCOUNT}" ] || die "set SLURM_ACCOUNT (or DRY_RUN=1)"
    command -v sbatch >/dev/null 2>&1 || die "sbatch not found"
fi

# Make every path absolute: the jobs run from a different directory.
abspath() { (cd -- "$(dirname -- "$1")" && printf '%s/%s\n' "$(pwd)" "$(basename -- "$1")"); }
MATRIX_LIST_FILE=$(abspath "${MATRIX_LIST_FILE}")
RUN_ALL_BENCHMARKS=$(abspath "${RUN_ALL_BENCHMARKS}")
GINKGO_BUILD_DIR=$(cd -- "${GINKGO_BUILD_DIR}" && pwd)
mkdir -p "${RESULTS_ROOT}" || die "cannot create ${RESULTS_ROOT}"
RESULTS_ROOT=$(cd -- "${RESULTS_ROOT}" && pwd)

# Binaries each kind of pass needs (checked here so a typo doesn't waste queue time).
need_bins() {
    local suffix=$1 exe missing=0
    for exe in spmv/spmv matrix_statistics/matrix_statistics; do
        if [ ! -x "${GINKGO_BUILD_DIR}/benchmark/${exe}${suffix}" ]; then
            warn "missing executable: ${GINKGO_BUILD_DIR}/benchmark/${exe}${suffix}"
            missing=1
        fi
    done
    return ${missing}
}
if [ "${DRY_RUN}" != "1" ]; then
    [ -n "${AMP_TOLERANCES}" ] && { need_bins "" || die "build the double-precision benchmarks first"; }
    [ "${INCLUDE_FP32}" = "1" ] && { need_bins "_single" || die "build the single-precision benchmarks first (or INCLUDE_FP32=0)"; }
fi

# --- Prefetch matrices (never cleans up anything) --------------------------------

if [ "${PREFETCH}" = "1" ]; then
    if command -v "${SSGET}" >/dev/null 2>&1; then
        echo "Resolving/extracting matrices from ${MATRIX_LIST_FILE} with ${SSGET} ..."
        n_ok=0
        while read -r entry _; do
            entry=${entry%%#*}
            [ -n "${entry}" ] || continue
            if [[ "${entry}" =~ ^[0-9]+$ ]]; then
                id=${entry}
            elif [[ "${entry}" =~ ^([A-Za-z0-9_-]+)/([A-Za-z0-9_-]+)$ ]]; then
                id=$("${SSGET}" -s "[ @name == ${BASH_REMATCH[2]} ] && [ @group == ${BASH_REMATCH[1]} ]")
            elif [[ "${entry}" =~ ^[A-Za-z0-9_-]+$ ]]; then
                id=$("${SSGET}" -s "[ @name == ${entry} ]")
            else
                warn "skipping unrecognized entry '${entry}'"
                continue
            fi
            ids=(${id})
            if [ ${#ids[@]} -eq 0 ] || [ "${ids[0]}" = "0" ]; then
                warn "no match for '${entry}' in the SuiteSparse index"
                continue
            fi
            path=$("${SSGET}" -i "${ids[0]}" -e)
            if [ -s "${path}" ]; then
                echo "  ${entry}: ${path}"
                n_ok=$((n_ok + 1))
            else
                warn "ssget did not produce a matrix file for '${entry}'"
            fi
        done < "${MATRIX_LIST_FILE}"
        [ "${n_ok}" -gt 0 ] || die "no matrices could be resolved from ${MATRIX_LIST_FILE}"
        echo "Prefetched ${n_ok} matri$( [ "${n_ok}" -eq 1 ] && echo x || echo ces)."
        echo
    else
        warn "'${SSGET}' not found; skipping the prefetch (jobs will call ssget themselves)"
    fi
fi

# --- Jobs ---------------------------------------------------------------------

# Each job: "name|job_dir|pass;pass;..." with pass = "results_dir,formats,precision,amp_tol".
# (no ',' or '|' occurs in any of the fields)
jobs=()
declare -A seen_dirs=()
for tol in ${AMP_TOLERANCES}; do
    # tol_<N> when the tolerance is 10^-N (as in the old results tree),
    # otherwise tol_<value>.
    tag=$("${PYTHON}" -c '
import math, sys
v = float(sys.argv[1]); n = round(-math.log10(v))
print(n if n > 0 and abs(v - 10.0 ** -n) < 1e-12 * v else f"{v:.0e}".replace("e-0", "e-"))
' "${tol}")
    d="tol_${tag}"
    [ -z "${seen_dirs[${d}]:-}" ] || die "tolerance '${tol}' maps to the same directory '${d}' as an earlier one"
    seen_dirs[${d}]=1
    jobs+=("${d}|${RESULTS_ROOT}/${d}|${RESULTS_ROOT}/${d},${BASE_FORMAT}:${AMP_FORMAT},double,${tol}")
done
if [ "${INCLUDE_FP32}" = "1" ]; then
    jobs+=("fp32|${RESULTS_ROOT}/fp32_job|${RESULTS_ROOT}/double_precision,${BASE_FORMAT},double,;${RESULTS_ROOT}/single_precision,${BASE_FORMAT},single,")
fi

# --- Manifest -----------------------------------------------------------------

manifest="${RESULTS_ROOT}/sweep_manifest.txt"
{
    echo "# AMP SpMV speedup/accuracy sweep"
    echo "# date           : $(date -Is)"
    echo "# host           : $(hostname)"
    echo "# ginkgo build   : ${GINKGO_BUILD_DIR}"
    echo "# run_all script : ${RUN_ALL_BENCHMARKS}"
    echo "# executor       : ${EXECUTOR} (${SYSTEM_NAME})"
    echo "# base format    : ${BASE_FORMAT}; AMP format ${AMP_FORMAT}"
    echo "# amp tolerances : ${AMP_TOLERANCES} (${AMP_TOLERANCE_TYPE}, csr strategy ${AMP_CSR_STRATEGY})"
    echo "# amp diag / fold: ${AMP_HIGH_PRECISION_DIAGONAL:-<default>} / ${AMP_BIN_FOLDUP_NNZ_RATIO:-<default>}"
    echo "# fp32 job       : ${INCLUDE_FP32}"
    echo "# repetitions    : ${REPETITIONS} (detailed=${DETAILED})"
    echo "# matrix list    : ${MATRIX_LIST_FILE}"
    echo "#"
    echo "# job job_id directory"
} > "${manifest}"

# --- Batch scripts and submission ------------------------------------------------

gpu_timer=false
case "${EXECUTOR}" in cuda|hip) gpu_timer=true ;; esac

q() { printf '%q' "$1"; }

n_submitted=0
for j in "${jobs[@]}"; do
    name=${j%%|*}
    rest=${j#*|}
    jdir=${rest%%|*}
    passes=${rest#*|}
    mkdir -p "${jdir}" || die "cannot create ${jdir}"
    job="${jdir}/job.sbatch"

    {
        echo "#!/bin/bash"
        echo "#SBATCH -J spmv-amp-${BASE_FORMAT}-${name}"
        [ -n "${SLURM_ACCOUNT}" ] && echo "#SBATCH -A ${SLURM_ACCOUNT}"
        echo "#SBATCH -p ${SLURM_PARTITION}"
        [ -n "${SLURM_QOS}" ] && echo "#SBATCH -q ${SLURM_QOS}"
        echo "#SBATCH -t ${SLURM_TIME}"
        echo "#SBATCH -N ${SLURM_NODES}"
        echo "#SBATCH --gpus-per-task=1"
        echo "#SBATCH -o ${jdir}/slurm-%j.out"
        echo ""
        echo "# Generated by $(basename -- "$0") on $(date -Is)"
        echo "set -uo pipefail"
        if [ -n "${JOB_SETUP}" ]; then
            echo ""
            echo "${JOB_SETUP}"
        fi
        echo ""
        echo "# Settings shared by all passes (read by run_all_benchmarks.sh)"
        echo "export BENCHMARK=spmv"
        echo "export GINKGO_BUILD_DIR=$(q "${GINKGO_BUILD_DIR}")"
        echo "export MATRIX_LIST_FILE=$(q "${MATRIX_LIST_FILE}")"
        echo "export EXECUTOR=$(q "${EXECUTOR}")"
        echo "export SYSTEM_NAME=$(q "${SYSTEM_NAME}")"
        echo "export DEVICE_ID=$(q "${DEVICE_ID}")"
        echo "export GPU_TIMER=${gpu_timer}"
        echo "export REPETITIONS=$(q "${REPETITIONS}")"
        echo "export DETAILED=${DETAILED}"
        echo "export KEEP_MATRICES=true"
        echo "export AMP_BASE_TYPE=$(q "${BASE_FORMAT}")"
        echo "export AMP_TOLERANCE_TYPE=$(q "${AMP_TOLERANCE_TYPE}")"
        echo "export AMP_CSR_STRATEGY=$(q "${AMP_CSR_STRATEGY}")"
        [ -n "${AMP_HIGH_PRECISION_DIAGONAL}" ] &&
            echo "export AMP_HIGH_PRECISION_DIAGONAL=$(q "${AMP_HIGH_PRECISION_DIAGONAL}")"
        [ -n "${AMP_BIN_FOLDUP_NNZ_RATIO}" ] &&
            echo "export AMP_BIN_FOLDUP_NNZ_RATIO=$(q "${AMP_BIN_FOLDUP_NNZ_RATIO}")"
        [ -n "${SSGET_ARCHIVE:-}" ] && echo "export SSGET_ARCHIVE=$(q "${SSGET_ARCHIVE}")"
        echo ""
        echo "status=0"
        echo "echo \"host: \$(hostname)  job: \${SLURM_JOB_ID:-?}  start: \$(date -Is)\""
        echo ""
        echo "# run_pass <results_dir> <formats> <double|single> [amp_tolerance]"
        echo "run_pass() {"
        echo "    echo \"=== pass: RESULTS_DIR=\$1 FORMATS=\$2 BENCHMARK_PRECISION=\$3 AMP_TOLERANCE=\${4:-<n/a>}\""
        echo "    RESULTS_DIR=\"\$1\" FORMATS=\"\$2\" BENCHMARK_PRECISION=\"\$3\" \\"
        echo "    AMP_TOLERANCE=\"\${4:-1e-14}\" \\"
        echo "        srun ${SRUN_ARGS} bash $(q "${RUN_ALL_BENCHMARKS}") || status=\$?"
        echo "}"
        echo ""
        IFS=';' read -ra plist <<< "${passes}"
        for p in "${plist[@]}"; do
            IFS=',' read -r rdir fmts prec atol <<< "${p}"
            fmts=${fmts//:/,}
            echo "run_pass $(q "${rdir}") ${fmts} ${prec}${atol:+ ${atol}}"
        done
        echo ""
        echo "echo \"end: \$(date -Is)  exit status: \${status}\""
        echo "exit \${status}"
    } > "${job}"
    chmod +x "${job}"

    if [ "${DRY_RUN}" = "1" ]; then
        echo "[dry run] sbatch ${SBATCH_EXTRA_ARGS} ${job}"
        echo "${name} DRYRUN ${jdir}" >> "${manifest}"
        continue
    fi
    # shellcheck disable=SC2086
    if job_id=$(sbatch --parsable ${SBATCH_EXTRA_ARGS} "${job}"); then
        job_id=${job_id%%;*}
        echo "submitted ${name}: job ${job_id}"
        echo "${name} ${job_id} ${jdir}" >> "${manifest}"
        n_submitted=$((n_submitted + 1))
    else
        warn "sbatch failed for ${name}"
        echo "${name} SUBMIT_FAILED ${jdir}" >> "${manifest}"
    fi
done

echo
echo "Results root: ${RESULTS_ROOT} (manifest: ${manifest})"
[ "${DRY_RUN}" = "1" ] || echo "Submitted ${n_submitted}/${#jobs[@]} job(s)."
echo "Plot with:"
echo "  ${PYTHON} ${script_dir}/plot_spmv_speedup_vs_tolerance.py --results-dir ${RESULTS_ROOT} --base-format ${BASE_FORMAT} --amp-format ${AMP_FORMAT}"
