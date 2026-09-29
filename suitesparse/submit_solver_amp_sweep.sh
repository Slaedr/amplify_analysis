#!/usr/bin/env bash
#
# Submits one Slurm job per AMP tolerance (plus one single-precision job) that
# runs ginkgo's benchmark/solver/solver_compare on a list of SuiteSparse
# matrices, always comparing against the same fixed-precision FP64 baseline:
#
#   variant      config_a                                  config_b
#   tol_<tol>    AMP[<BASE_FORMAT>] at that tolerance       <BASE_FORMAT><double>
#   fp32         <BASE_FORMAT><float> (whole solve in FP32) <BASE_FORMAT><double>
#
# Everything else (solver, preconditioner, reordering, iteration limit,
# residual goal, right-hand side, ...) is shared through the "common" block,
# so each job measures the speedup and the solution error of config_a with
# respect to FP64. Each job re-times its own FP64 baseline on the node it runs
# on, so every speedup is a same-node ratio.
#
# The companion plot script is plot_solver_amp_speedup_vs_tolerance.py.
#
# Run this on a login node: matrices are resolved (and, if needed,
# downloaded) with ssget *before* anything is submitted, since compute nodes
# usually have no internet access. ssget is only ever asked to look up and
# extract matrices -- this script never calls `ssget -c` and never deletes a
# .mtx file. Matrices that ssget unpacked into node-local /tmp are *copied*
# into MATRIX_CACHE_DIR so the compute nodes can read them.
#
# Directory tree produced:
#
#   $RESULTS_ROOT/
#     sweep_manifest.txt          settings + one line per submitted job
#     matrices_resolved.txt       name, group, ssget id, path of every matrix
#     matrices/<group>/<name>.mtx only for matrices copied out of /tmp
#     tol_1e-04/
#       config.json               solver_compare input
#       job.sbatch                the submitted batch script
#       solver_compare_results.json   (written by the job; updated after
#                                      every matrix)
#       slurm-<jobid>.out
#     tol_1e-06/ ...
#     fp32/ ...
#
# Required:
#   GINKGO_BUILD_DIR   Ginkgo build directory containing
#                      benchmark/solver/solver_compare (or set
#                      SOLVER_COMPARE_BIN to the binary directly)
#   SLURM_ACCOUNT      project to charge (not needed with DRY_RUN=1)
#
# Experiment (all optional; defaults in brackets):
#   AMP_TOLERANCES     space-separated AMP tolerances  ["1e-4 1e-6 1e-8 1e-10"]
#   INCLUDE_FP32       1 = also submit the single-precision job      [1]
#   BASE_FORMAT        csrc | ell -- fixed-precision format, and the
#                      storage underneath AMP                       [csrc]
#   AMP_FORMAT         amp (monolithic_classical) | ampib
#                      (independent_buckets)                         [amp]
#   AMP_TOLERANCE_TYPE componentwise | normwise              [componentwise]
#   AMP_HIGH_PRECISION_DIAGONAL  true | false                         [true]
#   AMP_BIN_FOLDUP_NNZ_RATIO     unset = solver_compare's default
#   SOLVER             solver_compare "solvers" value, e.g. fgs, gmres  [fgs]
#   PRECOND            "preconditioners" value, e.g. none, fgs, jacobi [none]
#   MAX_ITERS          iteration limit (for SOLVER=fgs: number of sweeps) [1]
#   REL_RES_GOAL       relative residual goal                        [1e-10]
#   REORDER            reordering; fgs (solver or preconditioner) needs
#                      multicolor                              [multicolor]
#   FGS_SWEEPS         sweeps per FGS preconditioner application      [1]
#   GMRES_RESTART      GMRES Krylov dimension                        [100]
#   RHS_GENERATION     1 | random | sinus                              [1]
#   INITIAL_GUESS      0 | random | rhs                                [0]
#   REPETITIONS        timed repetitions per config ("auto" allowed)  [10]
#   WARMUP             warmup repetitions per config                   [2]
#   EXECUTOR           hip | cuda | omp | reference                  [hip]
#   SYSTEM_NAME        only used for naming                     [frontier]
#   MATRIX_LIST_FILE   one SuiteSparse name, group/name, ssget id or .mtx
#                      path per line; '#' starts a comment
#                      [<this dir>/structurally_symmetric.txt]
#   RESULTS_ROOT       output root
#                      [./results-<system>-<solver>[-<precond>]-<base>]
#   MATRIX_CACHE_DIR   where matrices found under /tmp are copied
#                      [$RESULTS_ROOT/matrices]
#   COPY_TMP_MATRICES  1 = copy matrices that ssget left under /tmp or
#                      /var/tmp (node-local on most clusters) into
#                      MATRIX_CACHE_DIR; 0 = use them in place          [1]
#   SSGET              ssget executable                            [ssget]
#   OVERWRITE          1 = allow reusing a variant directory that already
#                      holds results                                   [0]
#   DRY_RUN            1 = resolve matrices and write every config and
#                      batch script, but do not submit                 [0]
#
# Slurm (optional):
#   SLURM_PARTITION    [batch]      SLURM_TIME   [02:00:00]
#   SLURM_QOS          [unset]      SLURM_NODES  [1]
#   SBATCH_EXTRA_ARGS  extra sbatch arguments, e.g. "--exclusive"
#   SRUN_ARGS          ["-N1 -n1 -c7 --gpus-per-task=1 --gpu-bind=closest"]
#   JOB_SETUP          shell commands run at the top of every job, e.g.
#                      "module load rocm"                              [none]
#
# Examples:
#   # GMRES + multicolor-FGS preconditioner, solved to 1e-10, on CSR (classical)
#   GINKGO_BUILD_DIR=$HOME/ginkgo/build SLURM_ACCOUNT=abc123 \
#   JOB_SETUP="module load rocm" SOLVER=gmres PRECOND=fgs MAX_ITERS=10000 \
#       ./submit_solver_amp_sweep.sh
#
#   # 10 standalone FGS sweeps on ELL, three tolerances, preview only
#   DRY_RUN=1 GINKGO_BUILD_DIR=$HOME/ginkgo/build BASE_FORMAT=ell \
#   MAX_ITERS=10 AMP_TOLERANCES="1e-4 1e-8 1e-12" ./submit_solver_amp_sweep.sh
#

set -uo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

die() { echo "ERROR: $*" >&2; exit 1; }
warn() { echo "WARNING: $*" >&2; }

# --- Settings -----------------------------------------------------------------

AMP_TOLERANCES=${AMP_TOLERANCES:-"1e-4 1e-6 1e-8 1e-10"}
INCLUDE_FP32=${INCLUDE_FP32:-1}
BASE_FORMAT=${BASE_FORMAT:-csrc}
AMP_FORMAT=${AMP_FORMAT:-amp}
AMP_TOLERANCE_TYPE=${AMP_TOLERANCE_TYPE:-componentwise}
AMP_HIGH_PRECISION_DIAGONAL=${AMP_HIGH_PRECISION_DIAGONAL:-true}
AMP_BIN_FOLDUP_NNZ_RATIO=${AMP_BIN_FOLDUP_NNZ_RATIO:-}
SOLVER=${SOLVER:-fgs}
PRECOND=${PRECOND:-none}
MAX_ITERS=${MAX_ITERS:-1}
REL_RES_GOAL=${REL_RES_GOAL:-1e-10}
REORDER=${REORDER:-multicolor}
FGS_SWEEPS=${FGS_SWEEPS:-1}
GMRES_RESTART=${GMRES_RESTART:-100}
RHS_GENERATION=${RHS_GENERATION:-1}
INITIAL_GUESS=${INITIAL_GUESS:-0}
REPETITIONS=${REPETITIONS:-10}
WARMUP=${WARMUP:-2}
EXECUTOR=${EXECUTOR:-hip}
SYSTEM_NAME=${SYSTEM_NAME:-frontier}
MATRIX_LIST_FILE=${MATRIX_LIST_FILE:-${script_dir}/structurally_symmetric.txt}
SSGET=${SSGET:-ssget}
COPY_TMP_MATRICES=${COPY_TMP_MATRICES:-1}
OVERWRITE=${OVERWRITE:-0}
DRY_RUN=${DRY_RUN:-0}
PYTHON=${PYTHON:-python3}

SLURM_ACCOUNT=${SLURM_ACCOUNT:-}
SLURM_PARTITION=${SLURM_PARTITION:-batch}
SLURM_TIME=${SLURM_TIME:-02:00:00}
SLURM_QOS=${SLURM_QOS:-}
SLURM_NODES=${SLURM_NODES:-1}
SBATCH_EXTRA_ARGS=${SBATCH_EXTRA_ARGS:-}
SRUN_ARGS=${SRUN_ARGS:-"-N1 -n1 -c7 --gpus-per-task=1 --gpu-bind=closest"}
JOB_SETUP=${JOB_SETUP:-}

if [ -n "${SOLVER_COMPARE_BIN:-}" ]; then
    BIN=${SOLVER_COMPARE_BIN}
else
    [ -n "${GINKGO_BUILD_DIR:-}" ] ||
        die "set GINKGO_BUILD_DIR (or SOLVER_COMPARE_BIN)"
    BIN=${GINKGO_BUILD_DIR}/benchmark/solver/solver_compare
fi

precond_tag=""
[ "${PRECOND}" != "none" ] && precond_tag="-${PRECOND}"
RESULTS_ROOT=${RESULTS_ROOT:-${PWD}/results-${SYSTEM_NAME}-${SOLVER}${precond_tag}-${BASE_FORMAT}}
MATRIX_CACHE_DIR=${MATRIX_CACHE_DIR:-${RESULTS_ROOT}/matrices}

# --- Validation ---------------------------------------------------------------

command -v "${PYTHON}" >/dev/null 2>&1 ||
    die "need python3 (or set PYTHON=) to write the configs"
[ -x "${BIN}" ] || die "solver_compare binary '${BIN}' not found or not executable"
[ -f "${MATRIX_LIST_FILE}" ] || die "matrix list '${MATRIX_LIST_FILE}' not found"
case "${BASE_FORMAT}" in
    csrc) AMP_BASE_TYPE=csr ;;
    ell) AMP_BASE_TYPE=ell ;;
    *) die "BASE_FORMAT must be csrc or ell, got '${BASE_FORMAT}'" ;;
esac
case "${AMP_FORMAT}" in
    amp|ampib) ;;
    *) die "AMP_FORMAT must be amp or ampib, got '${AMP_FORMAT}'" ;;
esac
case "${AMP_TOLERANCE_TYPE}" in
    componentwise|normwise) ;;
    *) die "AMP_TOLERANCE_TYPE must be componentwise or normwise" ;;
esac
case "${MAX_ITERS}" in
    ''|*[!0-9]*) die "MAX_ITERS must be a positive integer, got '${MAX_ITERS}'" ;;
esac
if [ "${SOLVER}" = "fgs" ] || [ "${PRECOND}" = "fgs" ]; then
    [ "${REORDER}" = "multicolor" ] ||
        die "SOLVER/PRECOND=fgs requires REORDER=multicolor (got '${REORDER}')"
fi
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

# --- Variants -----------------------------------------------------------------

variants=()
for tol in ${AMP_TOLERANCES}; do
    tag=$("${PYTHON}" -c 'import sys; print(f"{float(sys.argv[1]):.0e}")' "${tol}")
    variants+=("tol_${tag}|${tol}")
done
[ "${INCLUDE_FP32}" = "1" ] && variants+=("fp32|")

for v in "${variants[@]}"; do
    vdir="${RESULTS_ROOT}/${v%%|*}"
    if [ -s "${vdir}/solver_compare_results.json" ] && [ "${OVERWRITE}" != "1" ]; then
        die "${vdir} already holds results; use a new RESULTS_ROOT or set OVERWRITE=1"
    fi
done

mkdir -p "${RESULTS_ROOT}"

# --- Resolve matrices with ssget (never cleans up anything) --------------------

resolved="${RESULTS_ROOT}/matrices_resolved.txt"
: > "${resolved}"
echo "# name group ssget_id path" >> "${resolved}"

have_ssget=0
command -v "${SSGET}" >/dev/null 2>&1 && have_ssget=1

echo "Resolving matrices from ${MATRIX_LIST_FILE} ..."
n_ok=0
declare -A seen_names=()
# Records one resolved matrix, skipping a name that was already listed (the
# plot script identifies matrices by name).
add_matrix() {
    local name=$1 group=$2 id=$3 path=$4
    if [ -n "${seen_names[${name}]:-}" ]; then
        warn "skipping duplicate matrix name '${name}' (${path}); already have ${seen_names[${name}]}"
        return
    fi
    seen_names[${name}]=${path}
    echo "${name} ${group} ${id} ${path}" >> "${resolved}"
    echo "  ${name} [${group}, id ${id}]: ${path}"
    n_ok=$((n_ok + 1))
}
while read -r entry _; do
    entry=${entry%%#*}
    [ -n "${entry}" ] || continue

    # A direct path to a Matrix Market file is used as-is.
    if [[ "${entry}" == *.mtx ]]; then
        [ -s "${entry}" ] || { warn "skipping '${entry}': file not found"; continue; }
        path=$(cd -- "$(dirname -- "${entry}")" && pwd)/$(basename -- "${entry}")
        name=$(basename -- "${entry}" .mtx)
        add_matrix "${name}" - - "${path}"
        continue
    fi

    [ "${have_ssget}" = "1" ] ||
        die "'${SSGET}' not found, needed to resolve '${entry}' (or list .mtx paths instead)"
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
    # ssget -s prints one id per match; names are not unique across groups
    ids=(${id})
    if [ ${#ids[@]} -eq 0 ] || [ "${ids[0]}" = "0" ]; then
        warn "skipping '${entry}': no match in the SuiteSparse index"
        continue
    fi
    if [ ${#ids[@]} -gt 1 ]; then
        warn "'${entry}' matches ids ${ids[*]}; using ${ids[0]} (write group/name to choose)"
    fi
    id=${ids[0]}

    name=$("${SSGET}" -i "${id}" -pname)
    group=$("${SSGET}" -i "${id}" -pgroup)
    if [ "$("${SSGET}" -i "${id}" -preal)" = "0" ]; then
        warn "skipping ${group}/${name}: not a real-valued matrix"
        continue
    fi
    if [ "$("${SSGET}" -i "${id}" -prows)" != "$("${SSGET}" -i "${id}" -pcols)" ]; then
        warn "skipping ${group}/${name}: not square"
        continue
    fi

    # Downloads and extracts if not already present; prints the .mtx path.
    path=$("${SSGET}" -i "${id}" -e)
    [ -s "${path}" ] || { warn "skipping ${group}/${name}: ssget returned '${path}', which is missing or empty"; continue; }

    # Node-local /tmp is not visible from the compute nodes: copy (never move)
    # the file somewhere shared, keeping any copy already there.
    case "${COPY_TMP_MATRICES}:${path}" in
        1:/tmp/*|1:/var/tmp/*)
            dest="${MATRIX_CACHE_DIR}/${group}/${name}.mtx"
            if [ ! -s "${dest}" ]; then
                mkdir -p "$(dirname -- "${dest}")"
                cp -- "${path}" "${dest}.partial" && mv -f -- "${dest}.partial" "${dest}" ||
                    die "could not copy ${path} to ${dest}"
            fi
            path=${dest}
            ;;
    esac

    add_matrix "${name}" "${group}" "${id}" "${path}"
done < "${MATRIX_LIST_FILE}"

[ "${n_ok}" -gt 0 ] || die "no usable matrices resolved from ${MATRIX_LIST_FILE}"
echo "Resolved ${n_ok} matri$( [ "${n_ok}" -eq 1 ] && echo x || echo ces)."
echo

# --- Manifest -----------------------------------------------------------------

manifest="${RESULTS_ROOT}/sweep_manifest.txt"
{
    echo "# AMP vs FP64 solver_compare sweep"
    echo "# date           : $(date -Is)"
    echo "# host           : $(hostname)"
    echo "# binary         : ${BIN}"
    echo "# executor       : ${EXECUTOR}"
    echo "# base format    : ${BASE_FORMAT} (AMP base type ${AMP_BASE_TYPE}, ${AMP_FORMAT})"
    echo "# amp tolerances : ${AMP_TOLERANCES} (${AMP_TOLERANCE_TYPE}, high-precision diagonal ${AMP_HIGH_PRECISION_DIAGONAL})"
    echo "# fp32 variant   : ${INCLUDE_FP32}"
    echo "# solver         : ${SOLVER}, preconditioner ${PRECOND}, reorder ${REORDER}"
    echo "# max iters      : ${MAX_ITERS}, rel_res_goal ${REL_RES_GOAL}"
    echo "# rhs / x0       : ${RHS_GENERATION} / ${INITIAL_GUESS}"
    echo "# reps / warmup  : ${REPETITIONS} / ${WARMUP}"
    echo "# matrix list    : ${MATRIX_LIST_FILE} (${n_ok} resolved)"
    echo "#"
    echo "# variant job_id directory"
} > "${manifest}"

# --- Configs, batch scripts, submission ----------------------------------------

gpu_timer=false
case "${EXECUTOR}" in cuda|hip) gpu_timer=true ;; esac

n_submitted=0
for v in "${variants[@]}"; do
    variant=${v%%|*}
    tol=${v#*|}
    vdir="${RESULTS_ROOT}/${variant}"
    mkdir -p "${vdir}"

    CFG_EXECUTOR=${EXECUTOR} CFG_GPU_TIMER=${gpu_timer} CFG_SOLVER=${SOLVER} \
    CFG_PRECOND=${PRECOND} CFG_REORDER=${REORDER} CFG_MAX_ITERS=${MAX_ITERS} \
    CFG_REL_RES_GOAL=${REL_RES_GOAL} CFG_GMRES_RESTART=${GMRES_RESTART} \
    CFG_FGS_SWEEPS=${FGS_SWEEPS} CFG_RHS=${RHS_GENERATION} \
    CFG_X0=${INITIAL_GUESS} CFG_REPS=${REPETITIONS} CFG_WARMUP=${WARMUP} \
    CFG_BASE=${BASE_FORMAT} CFG_AMP_FORMAT=${AMP_FORMAT} \
    CFG_AMP_BASE=${AMP_BASE_TYPE} CFG_AMP_TOL_TYPE=${AMP_TOLERANCE_TYPE} \
    CFG_AMP_HPD=${AMP_HIGH_PRECISION_DIAGONAL} \
    CFG_AMP_FOLDUP=${AMP_BIN_FOLDUP_NNZ_RATIO} \
    "${PYTHON}" - "${vdir}/config.json" "${resolved}" "${variant}" "${tol}" <<'PY' ||
import json, os, sys

out_path, resolved_path, variant, tol = sys.argv[1:5]
env = os.environ

matrices = []
with open(resolved_path) as f:
    for line in f:
        if line.startswith("#") or not line.strip():
            continue
        name, group, ssget_id, path = line.rstrip("\n").split(" ", 3)
        problem = {"name": name}
        if group != "-":
            problem["group"] = group
        if ssget_id != "-":
            problem["ssget_id"] = int(ssget_id)
        matrices.append({"filename": path, "problem": problem})

max_iters = int(env["CFG_MAX_ITERS"])
common = {
    "executor": env["CFG_EXECUTOR"],
    "gpu_timer": env["CFG_GPU_TIMER"] == "true",
    "solvers": env["CFG_SOLVER"],
    "preconditioners": env["CFG_PRECOND"],
    "reorder": env["CFG_REORDER"],
    "max_iters": max_iters,
    "warmup_max_iters": max_iters,
    "rel_res_goal": float(env["CFG_REL_RES_GOAL"]),
    "gmres_restart": int(env["CFG_GMRES_RESTART"]),
    "fgs_sweeps": int(env["CFG_FGS_SWEEPS"]),
    "rhs_generation": env["CFG_RHS"],
    "initial_guess_generation": env["CFG_X0"],
    "repetitions": env["CFG_REPS"],
    "warmup": int(env["CFG_WARMUP"]),
}

base = env["CFG_BASE"]
base_upper = base.upper()
if variant == "fp32":
    config_a = {"formats": base, "precision": "single"}
    label_a = f"{base_upper}<float>"
else:
    config_a = {
        "formats": env["CFG_AMP_FORMAT"],
        "amp_base_type": env["CFG_AMP_BASE"],
        "amp_tolerance": float(tol),
        "amp_tolerance_type": env["CFG_AMP_TOL_TYPE"],
        "amp_high_precision_diagonal": env["CFG_AMP_HPD"] == "true",
    }
    if env.get("CFG_AMP_FOLDUP"):
        config_a["amp_bin_foldup_nnz_ratio"] = float(env["CFG_AMP_FOLDUP"])
    label_a = (f"{env['CFG_AMP_FORMAT'].upper()}[{base_upper}] "
               f"tol={float(tol):.0e}")

cfg = {
    "label_a": label_a,
    "label_b": f"{base_upper}<double>",
    "output_file": "solver_compare_results.json",
    "common": common,
    "config_a": config_a,
    "config_b": {"formats": base, "precision": "double"},
    "matrices": matrices,
}
with open(out_path, "w") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")
PY
        die "failed to write ${vdir}/config.json"

    job="${vdir}/job.sbatch"
    {
        echo "#!/bin/bash"
        echo "#SBATCH -J amp-${SOLVER}${precond_tag}-${BASE_FORMAT}-${variant}"
        [ -n "${SLURM_ACCOUNT}" ] && echo "#SBATCH -A ${SLURM_ACCOUNT}"
        echo "#SBATCH -p ${SLURM_PARTITION}"
        [ -n "${SLURM_QOS}" ] && echo "#SBATCH -q ${SLURM_QOS}"
        echo "#SBATCH -t ${SLURM_TIME}"
        echo "#SBATCH -N ${SLURM_NODES}"
        echo "#SBATCH -o ${vdir}/slurm-%j.out"
        echo ""
        echo "# Generated by $(basename -- "$0") on $(date -Is)"
        echo "set -uo pipefail"
        if [ -n "${JOB_SETUP}" ]; then
            echo ""
            echo "${JOB_SETUP}"
        fi
        echo ""
        echo "cd \"${vdir}\" || exit 1"
        echo "echo \"host: \$(hostname)  job: \${SLURM_JOB_ID:-?}  start: \$(date -Is)\""
        echo "srun ${SRUN_ARGS} \"${BIN}\" config.json"
        echo "status=\$?"
        echo "echo \"end: \$(date -Is)  exit status: \${status}\""
        echo "exit \${status}"
    } > "${job}"
    chmod +x "${job}"

    if [ "${DRY_RUN}" = "1" ]; then
        # shellcheck disable=SC2086
        echo "[dry run] sbatch ${SBATCH_EXTRA_ARGS} ${job}"
        echo "${variant} DRYRUN ${vdir}" >> "${manifest}"
        continue
    fi
    # shellcheck disable=SC2086
    if job_id=$(sbatch --parsable ${SBATCH_EXTRA_ARGS} "${job}"); then
        job_id=${job_id%%;*}
        echo "submitted ${variant}: job ${job_id}"
        echo "${variant} ${job_id} ${vdir}" >> "${manifest}"
        n_submitted=$((n_submitted + 1))
    else
        warn "sbatch failed for ${variant}"
        echo "${variant} SUBMIT_FAILED ${vdir}" >> "${manifest}"
    fi
done

echo
echo "Results root: ${RESULTS_ROOT} (manifest: ${manifest})"
[ "${DRY_RUN}" = "1" ] || echo "Submitted ${n_submitted}/${#variants[@]} job(s)."
echo "Plot with:"
echo "  ${PYTHON} ${script_dir}/plot_solver_amp_speedup_vs_tolerance.py --results-dir ${RESULTS_ROOT}"
