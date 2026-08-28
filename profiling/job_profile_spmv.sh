#!/usr/bin/env bash
# Hardware-counter profiling run for AMP SpMV, in the same style as
# suitesparse/job_spmv.sh.
#
#   GINKGO_BUILD_DIR=... EXECUTOR=hip SYSTEM_NAME=frontier \
#   RESULTS_DIR=$PWD ./job_profile_spmv.sh
#
# Unlike suitesparse/job_spmv.sh this does NOT go through
# run_all_benchmarks.sh -- hardware counters have to be attached to a single
# benchmark/spmv process with exclusive use of one GCD, which that driver
# cannot arrange.  Matrices are therefore given as explicit .mtx paths.

set -euo pipefail

if [ ! -n "${GINKGO_BUILD_DIR:-}" ]; then
    echo Please set GINKGO_BUILD_DIR.
	exit -1
fi
if [ ! -n "${EXECUTOR:-}" ]; then
    echo Please set EXECUTOR.
	exit -1
fi
if [ ! -n "${RESULTS_DIR:-}" ]; then
    echo RESULTS_DIR not set.
	echo Setting it to the current directory: `pwd`
	export RESULTS_DIR=`pwd`
fi
if [ ! -n "${SYSTEM_NAME:-}" ]; then
    echo SYSTEM_NAME not set. Setting it to "unspecified".
	export SYSTEM_NAME=unspecified
fi
if [ ! -n "${MATRIX_PATH_FILE:-}" ]; then
    echo MATRIX_PATH_FILE not set. Setting it to matrix_paths.txt
	export MATRIX_PATH_FILE=matrix_paths.txt
fi
if [ ! -r "$MATRIX_PATH_FILE" ]; then
    echo "Cannot read $MATRIX_PATH_FILE."
	echo "It needs one absolute .mtx path per line -- these are the files"
	echo "ssget left under \$GINKGO_BUILD_DIR/benchmark/matrices, not the"
	echo "bare SuiteSparse names used in suitesparse/matrices.txt."
	exit -1
fi

export AMP_BASE_TYPE="${AMP_BASE_TYPE:-csr}"
export AMP_TOLERANCE="${AMP_TOLERANCE:-1e-9}"
export AMP_TOLERANCE_TYPE="${AMP_TOLERANCE_TYPE:-componentwise}"
# csr as well as amp: nearly every derived metric in the report is a
# comparison against the uniform-fp64 baseline on the same matrix.
export FORMATS="${FORMATS:-csr,amp}"
export REPETITIONS="${REPETITIONS:-20}"
export WARMUP="${WARMUP:-3}"
export BENCHMARK_PRECISION="${BENCHMARK_PRECISION:-double}"
export PROFILE_TOOL="${PROFILE_TOOL:-auto}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$HERE/amp_spmv_profile.sh" \
    --matrix-list "$MATRIX_PATH_FILE" \
    --tool "$PROFILE_TOOL" \
    --launch "${PROFILE_LAUNCH:-auto}"

OUTDIR="$RESULTS_DIR/results-profile-spmv-${AMP_BASE_TYPE}-${SYSTEM_NAME}"
python3 "$HERE/amp_spmv_report.py" "$OUTDIR" --markdown

# Static bin/utilisation model over the same matrices, for the same tolerance.
python3 "$HERE/amp_bin_predict.py" $(grep -v '^\s*#' "$MATRIX_PATH_FILE") \
    --tol "$AMP_TOLERANCE" \
    --json "$OUTDIR/bin_model.json" --csv "$OUTDIR/bin_model.csv" \
    > "$OUTDIR/bin_model.txt"

echo "Results in $OUTDIR"
