#!/usr/bin/env bash
#
# Sweep the AMP tolerance for the AMPLify SpMV, FGS and/or GMRES benchmarks.
#
# For every (benchmark, tolerance, amp_spmv_strategy) triple this runs
# benchmark/amp/amp_benchmark_<bench> once, with a generated config whose
# output_file_prefix points into a per-run directory.  Each benchmark names its
# own output file
#
#   <output_file_prefix><bench>_<base_format><strategy_suffix>_<executor>_results.json
#
# so giving each run its own directory is what keeps the runs from overwriting
# each other (the tolerance is *not* part of that name).  Each result directory
# ends up holding exactly one JSON plus the config that produced it; the
# companion plot script recovers the benchmark (from the top-level "spmv" /
# "fgs" / "gmres" key), the tolerance and the strategy (from the "config"
# block) from inside the JSON, so it never has to parse file names.
#
# Layout produced:
#
#   $RESULTS_DIR/
#     sweep_manifest.txt
#     spmv/tol_1e-04/monolithic_classical/{config.json,spmv_..._results.json,run.log}
#     spmv/tol_1e-04/independent_buckets/{...}
#     fgs/tol_1e-04/...
#     gmres/tol_1e-04/...
#     ...
#
# Each $RESULTS_DIR/<bench> subtree can be handed to the plot script directly.
#
# Required:
#   BASE_CONFIG        template config (no default; the sweep refuses to run
#                      without it)
#   GINKGO_BUILD_DIR   Ginkgo build directory (must contain
#                      benchmark/amp/amp_benchmark_<bench>), unless BENCH_DIR
#                      is set.
#
# Optional:
#   BENCHES       space separated    (default "spmv fgs gmres")
#   BENCH_DIR     binary directory   (default $GINKGO_BUILD_DIR/benchmark/amp)
#   RESULTS_DIR   output root        (default ./results-tol-<format>-<executor>)
#   LAUNCHER      MPI launcher for spmv and gmres
#                                    (default empty; e.g. "srun -n 4")
#   FGS_LAUNCHER  launcher for fgs   (default empty; amp_benchmark_fgs refuses
#                                    to run on more than one MPI rank, so this
#                                    must be a 1-rank launcher if set, e.g.
#                                    "srun -n 1")
#   TOLERANCES    space separated    (default 1e-4 .. 1e-14, one per decade)
#   STRATEGIES    space separated    (default "monolithic_classical independent_buckets")
#   DRY_RUN       1 = print only
#
# The base matrix format, executor, grid size, rep counts, GMRES settings, CSR
# strategies etc. are taken from BASE_CONFIG unchanged -- one base format per
# invocation.  Note that nx/ny/nz are *per rank* for spmv and gmres, so with
# LAUNCHER="srun -n 4" those runs solve a 4x larger global problem than fgs.
#
# Examples:
#   BASE_CONFIG=./config.json GINKGO_BUILD_DIR=$HOME/ginkgo/build \
#       LAUNCHER="srun -n 4" FGS_LAUNCHER="srun -n 1" ./sweep_tolerance.sh
#
#   BENCHES=gmres TOLERANCES="1e-4 1e-8 1e-12" BASE_CONFIG=./config.json \
#       GINKGO_BUILD_DIR=$HOME/ginkgo/build ./sweep_tolerance.sh
#

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

BENCHES=${BENCHES:-"spmv fgs gmres"}
BENCH_DIR=${BENCH_DIR:-${GINKGO_BUILD_DIR:-}/benchmark/amp}
LAUNCHER=${LAUNCHER:-}
FGS_LAUNCHER=${FGS_LAUNCHER:-}
TOLERANCES=${TOLERANCES:-"1e-4 1e-5 1e-6 1e-7 1e-8 1e-9 1e-10 1e-11 1e-12 1e-13 1e-14"}
STRATEGIES=${STRATEGIES:-"monolithic_classical independent_buckets"}
DRY_RUN=${DRY_RUN:-0}

PYTHON=${PYTHON:-python3}

die() { echo "ERROR: $*" >&2; exit 1; }

bench_bin() { echo "${BENCH_DIR}/amp_benchmark_$1"; }
bench_launcher() {
    if [ "$1" = "fgs" ]; then echo "${FGS_LAUNCHER}"; else echo "${LAUNCHER}"; fi
}

command -v "${PYTHON}" >/dev/null 2>&1 ||
    die "need python3 (or set PYTHON=) to generate the per-run configs"
[ -n "${BASE_CONFIG:-}" ] ||
    die "BASE_CONFIG is not set; pass the template config explicitly"
[ -f "${BASE_CONFIG}" ] ||
    die "base config '${BASE_CONFIG}' not found (set BASE_CONFIG)"
for bench in ${BENCHES}; do
    case "${bench}" in
        spmv|fgs|gmres) ;;
        *) die "unknown benchmark '${bench}' in BENCHES (spmv, fgs, gmres)" ;;
    esac
    [ -x "$(bench_bin "${bench}")" ] ||
        die "benchmark binary '$(bench_bin "${bench}")' not found or not executable
     (set GINKGO_BUILD_DIR, or BENCH_DIR directly)"
done

# Pull the fields we need for naming out of the template config.
read -r base_format executor < <(
    "${PYTHON}" - "${BASE_CONFIG}" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
print(cfg.get("amp_base_format", "ell"), cfg.get("executor", "cuda"))
PY
)

RESULTS_DIR=${RESULTS_DIR:-${PWD}/results-tol-${base_format}-${executor}}
mkdir -p "${RESULTS_DIR}"

manifest="${RESULTS_DIR}/sweep_manifest.txt"
{
    echo "# AMP tolerance sweep"
    echo "# date         : $(date -Is)"
    echo "# host         : $(hostname)"
    echo "# benchmarks   : ${BENCHES}"
    echo "# binary dir   : ${BENCH_DIR}"
    echo "# base config  : ${BASE_CONFIG}"
    echo "# launcher     : ${LAUNCHER:-<none>}"
    echo "# fgs launcher : ${FGS_LAUNCHER:-<none>}"
    echo "# base format  : ${base_format}"
    echo "# executor     : ${executor}"
    echo "# tolerances   : ${TOLERANCES}"
    echo "# strategies   : ${STRATEGIES}"
    echo "#"
    echo "# bench tolerance strategy status run_dir"
} > "${manifest}"

echo "AMP tolerance sweep"
echo "  benchmarks   : ${BENCHES}"
echo "  binary dir   : ${BENCH_DIR}"
echo "  base config  : ${BASE_CONFIG}  (format=${base_format}, executor=${executor})"
echo "  launcher     : ${LAUNCHER:-<none>}"
echo "  fgs launcher : ${FGS_LAUNCHER:-<none>}"
echo "  results dir  : ${RESULTS_DIR}"
echo

n_ok=0
n_fail=0

for bench in ${BENCHES}; do
    bin=$(bench_bin "${bench}")
    launcher=$(bench_launcher "${bench}")
    for tol in ${TOLERANCES}; do
        # 1e-4 -> 1e-04 so that lexical and numerical order agree in the listing
        tol_tag=$("${PYTHON}" -c 'import sys; print(f"{float(sys.argv[1]):.0e}")' "${tol}")
        for strategy in ${STRATEGIES}; do
            run_dir="${RESULTS_DIR}/${bench}/tol_${tol_tag}/${strategy}"
            run_cfg="${run_dir}/config.json"
            mkdir -p "${run_dir}"

            # Per-run config: template + this tolerance/strategy, output
            # written into this run's own directory.
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

            echo "=== ${bench}  tol=${tol_tag}  strategy=${strategy} ==="
            if [ "${DRY_RUN}" = "1" ]; then
                echo "    (dry run) ${launcher} ${bin} ${run_cfg}"
                echo "${bench} ${tol} ${strategy} DRYRUN ${run_dir}" >> "${manifest}"
                continue
            fi

            # shellcheck disable=SC2086
            if ${launcher} "${bin}" "${run_cfg}" 2>&1 |
                    tee "${run_dir}/run.log"; then
                status=OK
                n_ok=$((n_ok + 1))
            else
                status=FAILED
                n_fail=$((n_fail + 1))
                echo "    !! run failed, see ${run_dir}/run.log" >&2
            fi
            echo "${bench} ${tol} ${strategy} ${status} ${run_dir}" >> "${manifest}"
            echo
        done
    done
done

echo "Sweep finished: ${n_ok} ok, ${n_fail} failed."
echo "Results under ${RESULTS_DIR} (manifest: ${manifest})"
echo
echo "Plot with:"
for bench in ${BENCHES}; do
    echo "  ${PYTHON} ${script_dir}/plot_speedup_error_vs_tol.py ${RESULTS_DIR}/${bench}"
done
[ "${n_fail}" -eq 0 ]
