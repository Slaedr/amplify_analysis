#!/usr/bin/env bash
#
# Sweep the AMP tolerance for the AMPLify distributed SpMV benchmark.
#
# For every (tolerance, amp_spmv_strategy) pair this runs
# benchmark/amp/amp_benchmark_spmv once, with a generated config whose
# output_file_prefix points into a per-run directory.  The benchmark names its
# own output file
#
#   <output_file_prefix>spmv_<base_format><strategy_suffix>_<executor>_results.json
#
# so giving each run its own directory is what keeps the runs from overwriting
# each other (the tolerance is *not* part of that name).  Each result directory
# ends up holding exactly one JSON plus the config that produced it; the
# companion plot script recovers the tolerance and the strategy from the
# "config" block inside the JSON, so it never has to parse file names.
#
# Layout produced:
#
#   $RESULTS_DIR/
#     sweep_manifest.txt
#     tol_1e-04/monolithic_classical/{config.json,spmv_..._results.json,run.log}
#     tol_1e-04/independent_buckets/{...}
#     ...
#
# Required:
#   GINKGO_BUILD_DIR   Ginkgo build directory (must contain
#                      benchmark/amp/amp_benchmark_spmv), unless BENCH_BIN is set.
#
# Optional:
#   BENCH_BIN     benchmark binary   (default $GINKGO_BUILD_DIR/benchmark/amp/amp_benchmark_spmv)
#   BASE_CONFIG   template config    (default ./config.json next to this script)
#   RESULTS_DIR   output root        (default ./results-spmv-tol-<format>-<executor>)
#   LAUNCHER      MPI launcher       (default empty; e.g. "srun -n 4" or "mpirun -np 4")
#   TOLERANCES    space separated    (default 1e-4 .. 1e-14, one per decade)
#   STRATEGIES    space separated    (default "monolithic_classical independent_buckets")
#   DRY_RUN       1 = print only
#
# The base matrix format, executor, grid size, rep counts, CSR strategies etc.
# are taken from BASE_CONFIG unchanged -- one base format per invocation.
#
# Example:
#   GINKGO_BUILD_DIR=$HOME/ginkgo/build LAUNCHER="srun -n 4" \
#       ./sweep_spmv_tolerance.sh
#

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

BENCH_BIN=${BENCH_BIN:-${GINKGO_BUILD_DIR:-}/benchmark/amp/amp_benchmark_spmv}
LAUNCHER=${LAUNCHER:-}
TOLERANCES=${TOLERANCES:-"1e-4 1e-5 1e-6 1e-7 1e-8 1e-9 1e-10 1e-11 1e-12 1e-13 1e-14"}
STRATEGIES=${STRATEGIES:-"monolithic_classical independent_buckets"}
DRY_RUN=${DRY_RUN:-0}

PYTHON=${PYTHON:-python3}

die() { echo "ERROR: $*" >&2; exit 1; }

command -v "${PYTHON}" >/dev/null 2>&1 ||
    die "need python3 (or set PYTHON=) to generate the per-run configs"
[ -f "${BASE_CONFIG}" ] ||
    die "base config '${BASE_CONFIG}' not found (set BASE_CONFIG)"
[ -x "${BENCH_BIN}" ] ||
    die "benchmark binary '${BENCH_BIN}' not found or not executable
     (set GINKGO_BUILD_DIR, or BENCH_BIN directly)"

# Pull the fields we need for naming out of the template config.
read -r base_format executor < <(
    "${PYTHON}" - "${BASE_CONFIG}" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
print(cfg.get("amp_base_format", "ell"), cfg.get("executor", "cuda"))
PY
)

RESULTS_DIR=${RESULTS_DIR:-${PWD}/results-spmv-tol-${base_format}-${executor}}
mkdir -p "${RESULTS_DIR}"

manifest="${RESULTS_DIR}/sweep_manifest.txt"
{
    echo "# AMP SpMV tolerance sweep"
    echo "# date        : $(date -Is)"
    echo "# host        : $(hostname)"
    echo "# binary      : ${BENCH_BIN}"
    echo "# base config : ${BASE_CONFIG}"
    echo "# launcher    : ${LAUNCHER:-<none>}"
    echo "# base format : ${base_format}"
    echo "# executor    : ${executor}"
    echo "# tolerances  : ${TOLERANCES}"
    echo "# strategies  : ${STRATEGIES}"
    echo "#"
    echo "# tolerance strategy status run_dir"
} > "${manifest}"

echo "AMP SpMV tolerance sweep"
echo "  binary      : ${BENCH_BIN}"
echo "  base config : ${BASE_CONFIG}  (format=${base_format}, executor=${executor})"
echo "  launcher    : ${LAUNCHER:-<none>}"
echo "  results dir : ${RESULTS_DIR}"
echo

n_ok=0
n_fail=0

for tol in ${TOLERANCES}; do
    # 1e-4 -> 1e-04 so that lexical and numerical order agree in the listing
    tol_tag=$("${PYTHON}" -c 'import sys; print(f"{float(sys.argv[1]):.0e}")' "${tol}")
    for strategy in ${STRATEGIES}; do
        run_dir="${RESULTS_DIR}/tol_${tol_tag}/${strategy}"
        run_cfg="${run_dir}/config.json"
        mkdir -p "${run_dir}"

        # Per-run config: template + this tolerance/strategy, output written
        # into this run's own directory.
        "${PYTHON}" - "${BASE_CONFIG}" "${run_cfg}" "${tol}" "${strategy}" \
                    "${run_dir}/" <<'PY'
import json, sys
src, dst, tol, strategy, prefix = sys.argv[1:6]
cfg = json.load(open(src))
cfg["amp_tolerance"] = float(tol)
cfg["amp_spmv_strategy"] = strategy
cfg["output_file_prefix"] = prefix
with open(dst, "w") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")
PY

        echo "=== tol=${tol_tag}  strategy=${strategy} ==="
        if [ "${DRY_RUN}" = "1" ]; then
            echo "    (dry run) ${LAUNCHER} ${BENCH_BIN} ${run_cfg}"
            echo "${tol} ${strategy} DRYRUN ${run_dir}" >> "${manifest}"
            continue
        fi

        # shellcheck disable=SC2086
        if ${LAUNCHER} "${BENCH_BIN}" "${run_cfg}" 2>&1 |
                tee "${run_dir}/run.log"; then
            status=OK
            n_ok=$((n_ok + 1))
        else
            status=FAILED
            n_fail=$((n_fail + 1))
            echo "    !! run failed, see ${run_dir}/run.log" >&2
        fi
        echo "${tol} ${strategy} ${status} ${run_dir}" >> "${manifest}"
        echo
    done
done

echo "Sweep finished: ${n_ok} ok, ${n_fail} failed."
echo "Results under ${RESULTS_DIR} (manifest: ${manifest})"
echo
echo "Plot with:"
echo "  ${PYTHON} ${script_dir}/plot_spmv_speedup_vs_tol.py ${RESULTS_DIR}"
[ "${n_fail}" -eq 0 ]
