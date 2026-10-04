#!/usr/bin/env bash
#
# Slurm/multi-GPU-node variant of sweep_tolerance.sh: same sweep over the AMP
# tolerance for the AMPLify SpMV, FGS and/or GMRES benchmarks, but instead of
# running every (benchmark, tolerance, strategy) combination one after
# another, it statically partitions the GPUs of the current Slurm
# allocation into "lanes" and runs one combination per lane concurrently,
# each lane pinned to its own disjoint set of GPUs and host cores so that
# the runs never contend for resources (this matters because we are
# measuring running time).
#
# Meant to be launched from inside an sbatch script, e.g. on Frontier
# (8 GPUs/node, 64 cores split into 8 L3 domains of 8 cores each, one core
# per domain reserved by the system -- so 7 usable cores per GPU):
#
#   #!/bin/bash
#   #SBATCH -N 1
#   #SBATCH -t 02:00:00
#   #SBATCH -q debug
#
#   module load rocm craype-accel-amd-gfx90a
#   BASE_CONFIG=./config.json GINKGO_BUILD_DIR=$HOME/ginkgo/build \
#       CPUS_PER_GPU=7 ./sweep_tolerance_slurm.sh
#
# For a multi-rank benchmark (RANKS>1 below), each lane becomes an
# `srun -N.. -n RANKS ..` spanning that many GPUs (within one node if
# RANKS <= GPUS_PER_NODE, or across whole nodes if RANKS is a multiple of
# GPUS_PER_NODE); FGS is always single-rank regardless of RANKS, since
# amp_benchmark_fgs refuses to run on more than one MPI rank.
#
# For GMRES, note that the solver tolerance is set to be the same as the AMP
# tolerance; we sweep both of them together.
#
# For every (benchmark, tolerance, amp_spmv_strategy) triple this runs
# benchmark/amp/amp_benchmark_<bench> once, with a generated config whose
# output_file_prefix points into a per-run directory.  Each benchmark names
# its own output file
#
#   <output_file_prefix><bench>_<base_format><strategy_suffix>_<executor>_results.json
#
# so giving each run its own directory is what keeps the runs from
# overwriting each other (the tolerance is *not* part of that name).  Each
# result directory ends up holding exactly one JSON plus the config that
# produced it (plus run.log and, unless CHECK_BINDING=0, binding.txt); the
# companion plot script recovers the benchmark, the tolerance and the
# strategy from inside the JSON, so it never has to parse file names.
#
# Layout produced (identical to sweep_tolerance.sh):
#
#   $RESULTS_DIR/
#     sweep_manifest.txt
#     lanes/lane_<id>.runs, lane_<id>.log      (bookkeeping; not needed to plot)
#     spmv/tol_1e-02/monolithic_classical/{config.json,spmv_..._results.json,run.log,binding.txt}
#     spmv/tol_1e-02/independent_buckets/{...}
#     fgs/tol_1e-02/...
#     gmres/tol_1e-02/...
#     ...
#
# Each $RESULTS_DIR/<bench> subtree can be handed to the plot script
# directly, exactly like sweep_tolerance.sh's output.
#
# Required:
#   BASE_CONFIG        template config (no default; the sweep refuses to run
#                      without it)
#   GINKGO_BUILD_DIR   Ginkgo build directory (must contain
#                      benchmark/amp/amp_benchmark_<bench>), unless BENCH_DIR
#                      is set.
#
# Optional, matching sweep_tolerance.sh:
#   BENCHES       space separated    (default "spmv fgs gmres")
#   BENCH_DIR     binary directory   (default $GINKGO_BUILD_DIR/benchmark/amp)
#   RESULTS_DIR   output root        (default ./results-tol-<format>-<executor>)
#   TOLERANCES    space separated    (default 1e-2 .. 1e-14, one per decade)
#   STRATEGIES    space separated    (default "monolithic_classical independent_buckets")
#   DRY_RUN       1 = print only
#
# Optional, Slurm/placement specific (each is auto-detected from the current
# allocation if left unset; see resource discovery below):
#   NODES           space-separated node hostnames to use
#                   (default: scontrol show hostnames "$SLURM_JOB_NODELIST")
#   GPUS_PER_NODE   GPUs per node available to this job
#                   (default: $SLURM_GPUS_ON_NODE / $SLURM_GPUS_PER_NODE)
#   CPUS_PER_GPU    host cores dedicated to each GPU's lane, i.e. -c for that
#                   lane's srun (default: $SLURM_CPUS_ON_NODE / GPUS_PER_NODE)
#   RANKS           MPI ranks (= GPUs) per spmv/gmres run; fgs is always 1
#                   rank regardless of this               (default 1)
#   SRUN_EXTRA_ARGS extra args appended to every srun, e.g. "--threads-per-core=1"
#   SRUN_MEM        if set, passed as --mem-per-gpu=<val> to every srun (use
#                   this if concurrent runs appear to serialize on memory --
#                   by default a step may be granted the whole node's memory)
#   CHECK_BINDING   1 (default) = each run records its actual CPU/GPU
#                   placement to binding.txt and flags mismatches;
#                   0 = skip (slightly less srun-wrapper overhead)
#   STRICT_BINDING  1 = abort a run outright on a binding mismatch instead of
#                   just flagging it                     (default 0)
#
# The base matrix format, executor, grid size, rep counts, GMRES settings,
# CSR strategies etc. are taken from BASE_CONFIG unchanged -- one base format
# per invocation.  Note that nx/ny/nz are *per rank* for spmv and gmres, so
# with RANKS=4 those runs solve a 4x larger global problem than fgs.
#
# Examples:
#   # 1 node, 8 GPUs, default RANKS=1: up to 8 runs at once, one GPU each.
#   BASE_CONFIG=./config.json GINKGO_BUILD_DIR=$HOME/ginkgo/build \
#       ./sweep_tolerance_slurm.sh
#
#   # 2 nodes, 4-rank spmv/gmres runs (fgs still runs single-rank, 8-wide).
#   BASE_CONFIG=./config.json GINKGO_BUILD_DIR=$HOME/ginkgo/build \
#       RANKS=4 ./sweep_tolerance_slurm.sh
#
#   BENCHES=gmres TOLERANCES="1e-4 1e-8 1e-12" BASE_CONFIG=./config.json \
#       GINKGO_BUILD_DIR=$HOME/ginkgo/build ./sweep_tolerance_slurm.sh
#

set -uo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WRAPPER="${script_dir}/slurm_binding_wrapper.sh"
HELPER="${script_dir}/slurm_sweep_helpers.py"

BENCHES=${BENCHES:-"spmv fgs gmres"}
BENCH_DIR=${BENCH_DIR:-${GINKGO_BUILD_DIR:-}/benchmark/amp}
TOLERANCES=${TOLERANCES:-"1e-2 1e-3 1e-4 1e-5 1e-6 1e-7 1e-8 1e-9 1e-10 1e-11 1e-12 1e-13 1e-14"}
STRATEGIES=${STRATEGIES:-"monolithic_classical independent_buckets"}
DRY_RUN=${DRY_RUN:-0}

RANKS=${RANKS:-1}
SRUN_EXTRA_ARGS=${SRUN_EXTRA_ARGS:-}
SRUN_MEM=${SRUN_MEM:-}
CHECK_BINDING=${CHECK_BINDING:-1}
STRICT_BINDING=${STRICT_BINDING:-0}

PYTHON=${PYTHON:-python3}

export PYTHON HELPER CHECK_BINDING STRICT_BINDING

die() { echo "ERROR: $*" >&2; exit 1; }

bench_bin() { echo "${BENCH_DIR}/amp_benchmark_$1"; }

command -v "${PYTHON}" >/dev/null 2>&1 ||
    die "need python3 (or set PYTHON=) to generate the per-run configs"
[ -n "${BASE_CONFIG:-}" ] ||
    die "BASE_CONFIG is not set; pass the template config explicitly"
[ -f "${BASE_CONFIG}" ] ||
    die "base config '${BASE_CONFIG}' not found (set BASE_CONFIG)"
[ -x "${WRAPPER}" ] ||
    die "binding wrapper '${WRAPPER}' not found or not executable (should ship next to this script)"
for bench in ${BENCHES}; do
    case "${bench}" in
        spmv|fgs|gmres) ;;
        *) die "unknown benchmark '${bench}' in BENCHES (spmv, fgs, gmres)" ;;
    esac
    [ -x "$(bench_bin "${bench}")" ] ||
        die "benchmark binary '$(bench_bin "${bench}")' not found or not executable
     (set GINKGO_BUILD_DIR, or BENCH_DIR directly)"
done
case "${RANKS}" in
    ''|*[!0-9]*) die "RANKS must be a positive integer, got '${RANKS}'" ;;
esac
[ "${RANKS}" -ge 1 ] || die "RANKS must be >= 1"

# --- Resource discovery -----------------------------------------------------
# Everything here can be overridden by exporting the variable before calling
# this script; auto-detection only fills in what's left unset.

if [ -n "${NODES:-}" ]; then
    read -r -a nodes_array <<< "${NODES}"
else
    [ -n "${SLURM_JOB_NODELIST:-}" ] || [ "${DRY_RUN}" = "1" ] ||
        die "not inside a Slurm allocation (SLURM_JOB_NODELIST unset) and NODES not set"
    if [ -n "${SLURM_JOB_NODELIST:-}" ]; then
        mapfile -t nodes_array < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
    else
        nodes_array=(localhost)
    fi
fi
N_NODES=${#nodes_array[@]}
[ "${N_NODES}" -ge 1 ] || die "no nodes found (empty NODES / SLURM_JOB_NODELIST)"

if [ -z "${GPUS_PER_NODE:-}" ]; then
    if [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
        GPUS_PER_NODE=${SLURM_GPUS_ON_NODE}
    elif [ -n "${SLURM_GPUS_PER_NODE:-}" ]; then
        GPUS_PER_NODE=${SLURM_GPUS_PER_NODE##*:}
    elif [ -n "${SLURM_JOB_GPUS:-}" ]; then
        GPUS_PER_NODE=$(tr ',' '\n' <<< "${SLURM_JOB_GPUS}" | grep -c .)
    else
        die "cannot determine GPUS_PER_NODE from the Slurm allocation; set it explicitly"
    fi
fi
case "${GPUS_PER_NODE}" in
    ''|*[!0-9]*) die "GPUS_PER_NODE must be a positive integer, got '${GPUS_PER_NODE}'" ;;
esac
[ "${GPUS_PER_NODE}" -ge 1 ] || die "GPUS_PER_NODE must be >= 1"

if [ -z "${CPUS_PER_GPU:-}" ]; then
    [ -n "${SLURM_CPUS_ON_NODE:-}" ] ||
        die "cannot determine CPUS_PER_GPU (SLURM_CPUS_ON_NODE unset); set CPUS_PER_GPU explicitly"
    CPUS_PER_GPU=$(( SLURM_CPUS_ON_NODE / GPUS_PER_NODE ))
fi
case "${CPUS_PER_GPU}" in
    ''|*[!0-9]*) die "CPUS_PER_GPU must be a positive integer, got '${CPUS_PER_GPU}'" ;;
esac
[ "${CPUS_PER_GPU}" -ge 1 ] ||
    die "computed CPUS_PER_GPU < 1 (SLURM_CPUS_ON_NODE=${SLURM_CPUS_ON_NODE:-?} / GPUS_PER_NODE=${GPUS_PER_NODE}); set CPUS_PER_GPU explicitly"

# Pull the fields we need for naming out of the template config.
read -r base_format executor < <(
    "${PYTHON}" - "${BASE_CONFIG}" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
print(cfg.get("amp_base_format", "ell"), cfg.get("executor", "cuda"))
PY
)

RESULTS_DIR=${RESULTS_DIR:-${PWD}/results-tol-${base_format}-${executor}}
mkdir -p "${RESULTS_DIR}/lanes"
rm -f "${RESULTS_DIR}"/lanes/lane_*.runs "${RESULTS_DIR}"/lanes/lane_*.log

manifest="${RESULTS_DIR}/sweep_manifest.txt"
{
    echo "# AMP tolerance sweep (Slurm, multi-GPU)"
    echo "# date          : $(date -Is)"
    echo "# host          : $(hostname)"
    echo "# job id        : ${SLURM_JOB_ID:-<none>}"
    echo "# nodes         : ${nodes_array[*]}"
    echo "# gpus/node     : ${GPUS_PER_NODE}"
    echo "# cpus/gpu      : ${CPUS_PER_GPU}"
    echo "# ranks/run     : ${RANKS} (fgs always 1)"
    echo "# benchmarks    : ${BENCHES}"
    echo "# binary dir    : ${BENCH_DIR}"
    echo "# base config   : ${BASE_CONFIG}"
    echo "# base format   : ${base_format}"
    echo "# executor      : ${executor}"
    echo "# tolerances    : ${TOLERANCES}"
    echo "# strategies    : ${STRATEGIES}"
    echo "# check binding : ${CHECK_BINDING} (strict=${STRICT_BINDING})"
    echo "#"
    echo "# bench tolerance strategy status run_dir"
} > "${manifest}"

echo "AMP tolerance sweep (Slurm, multi-GPU)"
echo "  nodes        : ${nodes_array[*]} (gpus/node=${GPUS_PER_NODE}, cpus/gpu=${CPUS_PER_GPU})"
echo "  ranks/run    : ${RANKS} (fgs always 1)"
echo "  benchmarks   : ${BENCHES}"
echo "  binary dir   : ${BENCH_DIR}"
echo "  base config  : ${BASE_CONFIG}  (format=${base_format}, executor=${executor})"
echo "  results dir  : ${RESULTS_DIR}"
echo

# --- Lane geometry -----------------------------------------------------------
# A lane is a fixed, disjoint slice of the allocation (one or more whole
# GPUs, plus their closest cores) that runs its assigned runs one after
# another. Lanes of different widths never coexist on the same GPU: all
# lanes for a given rank count are built (and run) together as one "phase",
# and phases run strictly one after another via `wait`.

declare -a lane_nodelist=() lane_nnodes=() lane_width=() lane_extra=()

build_lanes() {
    local width=$1
    if [ "${width}" -le "${GPUS_PER_NODE}" ]; then
        [ $(( GPUS_PER_NODE % width )) -eq 0 ] ||
            die "GPUS_PER_NODE (${GPUS_PER_NODE}) is not divisible by RANKS (${width})"
        local lanes_per_node=$(( GPUS_PER_NODE / width ))
        local node
        for node in "${nodes_array[@]}"; do
            local i
            for ((i = 0; i < lanes_per_node; i++)); do
                lane_nodelist+=("${node}")
                lane_nnodes+=(1)
                lane_extra+=("")
                lane_width+=("${width}")
            done
        done
    else
        [ $(( width % GPUS_PER_NODE )) -eq 0 ] ||
            die "RANKS (${width}) must be a multiple of GPUS_PER_NODE (${GPUS_PER_NODE}) to span nodes"
        local nodes_per_lane=$(( width / GPUS_PER_NODE ))
        [ $(( N_NODES % nodes_per_lane )) -eq 0 ] ||
            die "allocation has ${N_NODES} node(s), not divisible by ${nodes_per_lane} nodes/lane needed for RANKS=${width}"
        local idx=0
        while [ "${idx}" -lt "${N_NODES}" ]; do
            local chunk=("${nodes_array[@]:idx:nodes_per_lane}")
            local joined
            joined=$(IFS=,; echo "${chunk[*]}")
            lane_nodelist+=("${joined}")
            lane_nnodes+=("${nodes_per_lane}")
            lane_extra+=("--ntasks-per-node=${GPUS_PER_NODE}")
            lane_width+=("${width}")
            idx=$(( idx + nodes_per_lane ))
        done
    fi
}

# Group benchmarks by how many ranks their runs use: fgs is always 1 rank;
# spmv/gmres use RANKS (which collapses to the same group as fgs when
# RANKS=1, so on the common single-rank case everything is one phase).
group1_benches=()
groupR_benches=()
for bench in ${BENCHES}; do
    if [ "${bench}" = "fgs" ] || [ "${RANKS}" -eq 1 ]; then
        group1_benches+=("${bench}")
    else
        groupR_benches+=("${bench}")
    fi
done

group1_lo=0; group1_hi=0
groupR_lo=0; groupR_hi=0

if [ ${#group1_benches[@]} -gt 0 ]; then
    build_lanes 1
    group1_hi=${#lane_nodelist[@]}
fi
groupR_lo=${group1_hi}
if [ ${#groupR_benches[@]} -gt 0 ]; then
    build_lanes "${RANKS}"
    groupR_hi=${#lane_nodelist[@]}
fi

echo "  lanes        : $(( group1_hi - group1_lo )) x 1-rank$( [ "${groupR_hi}" -gt "${groupR_lo}" ] && echo ", $(( groupR_hi - groupR_lo )) x ${RANKS}-rank" )"
echo

# --- Assign runs to lanes, round robin --------------------------------------

assign_group() {
    local -n benches_ref=$1
    local lo=$2 hi=$3
    local nlanes=$(( hi - lo ))
    [ "${nlanes}" -gt 0 ] ||
        die "internal error: no lanes for benches '${benches_ref[*]}'"
    local i=0
    local bench tol strategy
    for bench in "${benches_ref[@]}"; do
        for tol in ${TOLERANCES}; do
            for strategy in ${STRATEGIES}; do
                local lane=$(( lo + (i % nlanes) ))
                echo "${bench}|${tol}|${strategy}" >> "${RESULTS_DIR}/lanes/lane_${lane}.runs"
                i=$(( i + 1 ))
            done
        done
    done
}

[ ${#group1_benches[@]} -gt 0 ] && assign_group group1_benches "${group1_lo}" "${group1_hi}"
[ ${#groupR_benches[@]} -gt 0 ] && assign_group groupR_benches "${groupR_lo}" "${groupR_hi}"

# --- Per-lane worker ---------------------------------------------------------

lane_worker() {
    local lane_id=$1
    local runs_file="${RESULTS_DIR}/lanes/lane_${lane_id}.runs"
    local log_file="${RESULTS_DIR}/lanes/lane_${lane_id}.log"
    : > "${log_file}"
    [ -f "${runs_file}" ] || return 0

    local nodelist=${lane_nodelist[${lane_id}]}
    local nnodes=${lane_nnodes[${lane_id}]}
    local ranks=${lane_width[${lane_id}]}
    local extra=${lane_extra[${lane_id}]}

    local bench tol strategy
    while IFS='|' read -r bench tol strategy; do
        [ -n "${bench}" ] || continue

        local bin; bin=$(bench_bin "${bench}")
        local tol_tag
        tol_tag=$("${PYTHON}" -c 'import sys; print(f"{float(sys.argv[1]):.0e}")' "${tol}")
        local run_dir="${RESULTS_DIR}/${bench}/tol_${tol_tag}/${strategy}"
        local run_cfg="${run_dir}/config.json"
        mkdir -p "${run_dir}"

        # Set diagonal policy based on bench - only SpMV uses unrestricted diagonal
        local high_prec_diag="true"
        [ "${bench}" = "spmv" ] && high_prec_diag="false"

        # Per-run config: template + this tolerance/strategy, output written
        # into this run's own directory.
        "${PYTHON}" - "${BASE_CONFIG}" "${run_cfg}" "${tol}" "${strategy}" \
            "${high_prec_diag}" "${run_dir}/" <<'PY'
import json, sys
src, dst, tol, strategy, hpdiag, prefix = sys.argv[1:7]
cfg = json.load(open(src))
cfg["amp_tolerance"] = float(tol)
cfg["amp_spmv_strategy"] = strategy
cfg["amp_high_precision_diagonal"] = bool(hpdiag == "true")
cfg["gmres_tol"] = float(tol)
cfg["output_file_prefix"] = prefix
with open(dst, "w") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")
PY

        local srun_cmd=(srun --exact -N "${nnodes}" -n "${ranks}" -w "${nodelist}"
            -c "${CPUS_PER_GPU}" --gpus-per-task=1 --gpu-bind=closest --cpu-bind=threads)
        [ -n "${extra}" ] && srun_cmd+=(${extra})
        [ -n "${SRUN_MEM}" ] && srun_cmd+=(--mem-per-gpu="${SRUN_MEM}")
        [ -n "${SRUN_EXTRA_ARGS}" ] && srun_cmd+=(${SRUN_EXTRA_ARGS})
        srun_cmd+=("${WRAPPER}" "${run_dir}" "${bin}" "${run_cfg}")

        local t0 t1 status
        t0=$(date +%s)
        if [ "${DRY_RUN}" = "1" ]; then
            echo "[lane ${lane_id} ${nodelist}] (dry run) ${srun_cmd[*]}" >> "${log_file}"
            status=DRYRUN
        elif "${srun_cmd[@]}" > "${run_dir}/run.log" 2>&1; then
            status=OK
        else
            status=FAILED
            echo "[lane ${lane_id} ${nodelist}] !! ${bench} tol=${tol_tag} ${strategy} FAILED, see ${run_dir}/run.log" >&2
        fi
        t1=$(date +%s)

        if [ -f "${run_dir}/binding.txt" ]; then
            echo "end=${t1}" >> "${run_dir}/binding.txt"
            if [ "${status}" = "OK" ] && grep -q '^badbind=1$' "${run_dir}/binding.txt"; then
                status=OK_BADBIND
            fi
        fi

        echo "${bench} ${tol} ${strategy} ${status} ${run_dir}" >> "${log_file}"
        echo "[lane ${lane_id} ${nodelist}] ${bench} tol=${tol_tag} ${strategy} ${status} ($(( t1 - t0 ))s)"
    done < "${runs_file}"
}

run_phase() {
    local lo=$1 hi=$2
    [ "${hi}" -gt "${lo}" ] || return 0
    local lane
    for (( lane = lo; lane < hi; lane++ )); do
        lane_worker "${lane}" &
    done
    wait
}

run_phase "${group1_lo}" "${group1_hi}"
run_phase "${groupR_lo}" "${groupR_hi}"

# --- Combine per-lane logs into the manifest, in the same column format as
# sweep_tolerance.sh, sorted for readability -------------------------------

combined="${RESULTS_DIR}/lanes/combined.log"
: > "${combined}"
for f in "${RESULTS_DIR}"/lanes/lane_*.log; do
    [ -f "${f}" ] && cat "${f}" >> "${combined}"
done
sort -k1,1 -k2,2g -k3,3 "${combined}" >> "${manifest}"

n_ok=$(awk '$4=="OK" || $4=="OK_BADBIND" {c++} END{print c+0}' "${combined}")
n_fail=$(awk '$4=="FAILED" {c++} END{print c+0}' "${combined}")
n_badbind=$(awk '$4=="OK_BADBIND" {c++} END{print c+0}' "${combined}")

echo
if [ "${DRY_RUN}" != "1" ] && [ "${CHECK_BINDING}" = "1" ]; then
    echo "Checking recorded run placement for cross-run resource conflicts..."
    if ! "${PYTHON}" "${HELPER}" crosscheck "${RESULTS_DIR}"; then
        echo "!! placement crosscheck found conflicts (see above) -- runs may have shared resources" >&2
    fi
    echo
fi

echo "Sweep finished: ${n_ok} ok, ${n_fail} failed${n_badbind:+, ${n_badbind} ok but off the GPUs local cores}."
echo "Results under ${RESULTS_DIR} (manifest: ${manifest})"
echo
echo "Plot with:"
for bench in ${BENCHES}; do
    echo "  ${PYTHON} ${script_dir}/plot_speedup_error_vs_tol.py ${RESULTS_DIR}/${bench}"
done
[ "${n_fail}" -eq 0 ]
