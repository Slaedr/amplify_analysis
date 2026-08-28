#!/bin/bash
#SBATCH -A <PROJECT_ID>
#SBATCH -J amp-spmv-prof
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH -p batch
#SBATCH -o amp-spmv-prof-%j.out
#
# Frontier: profile AMP SpMV on ONE MI250X GCD.
#
# Hardware counters cannot be collected while another process shares the GCD,
# so this uses a single rank bound to a single GCD.  A Frontier node has 4
# MI250X = 8 GCDs; we deliberately use one and leave the rest idle.
#
#   sbatch --export=ALL,GINKGO_BUILD_DIR=...,MATRIX_PATH_FILE=...,AMP_TOLERANCE=1e-6 \
#          frontier_profile.sbatch

set -euo pipefail

module load rocm            # or a specific version, e.g. rocm/6.2.4
module list 2>&1 || true

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SYSTEM_NAME="${SYSTEM_NAME:-frontier}"
export EXECUTOR="${EXECUTOR:-hip}"
export RESULTS_DIR="${RESULTS_DIR:-$SLURM_SUBMIT_DIR}"
export MATRIX_PATH_FILE="${MATRIX_PATH_FILE:-$HERE/matrix_paths.txt}"

# One rank, one GCD, closest CPU cores.  --gpu-bind=closest hands this rank a
# single GCD; amp_spmv_profile.sh then pins to it with ROCR_VISIBLE_DEVICES.
srun -N1 -n1 -c7 --gpus-per-task=1 --gpu-bind=closest \
    bash "$HERE/job_profile_spmv.sh"
