#!/usr/bin/env bash
#
# Records this run's CPU affinity, visible GPU and that GPU's local CPU list
# into <run_dir>/binding.txt (rank 0 of the step only), then execs the real
# benchmark. Invoked by sweep_tolerance_slurm.sh as the srun program, so that
# concurrent runs' actual placement can be checked after the fact instead of
# just trusted.
#
# Usage: slurm_binding_wrapper.sh <run_dir> <bin> [args...]
#
# Honors, from the environment (set by the caller):
#   CHECK_BINDING    1 = record + verify (default 1), 0 = skip entirely
#   STRICT_BINDING   1 = abort this run on a binding mismatch (default 0)
#   PYTHON           interpreter for the subset check (default python3)
#   HELPER           path to slurm_sweep_helpers.py (subset check skipped if unset)

set -uo pipefail

run_dir=$1; shift
bin=$1; shift

if [ "${CHECK_BINDING:-1}" = "1" ] && [ "${SLURM_PROCID:-0}" = "0" ]; then
    python=${PYTHON:-python3}
    helper=${HELPER:-}

    cpuset=""
    if command -v taskset >/dev/null 2>&1; then
        cpuset=$(taskset -cp $$ 2>/dev/null | sed -E 's/.*: *//')
    fi

    gpu_bus=""
    if command -v rocm-smi >/dev/null 2>&1; then
        gpu_bus=$(rocm-smi --showbus 2>/dev/null |
            grep -oE '[0-9a-fA-F]{4,8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9]' | head -1)
    fi
    if [ -z "$gpu_bus" ] && command -v nvidia-smi >/dev/null 2>&1; then
        gpu_bus=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null | head -1)
    fi
    # Normalize to the 12-char "0000:bb:dd.f" form sysfs uses, lowercase.
    if [ -n "$gpu_bus" ]; then
        gpu_bus=$(echo "${gpu_bus: -12}" | tr 'A-Z' 'a-z')
    fi

    gpu_local_cpus=""
    if [ -n "$gpu_bus" ] && [ -f "/sys/bus/pci/devices/${gpu_bus}/local_cpulist" ]; then
        gpu_local_cpus=$(cat "/sys/bus/pci/devices/${gpu_bus}/local_cpulist")
    fi

    {
        echo "host=$(hostname -s 2>/dev/null || hostname)"
        echo "cpuset=${cpuset}"
        echo "gpu_bus=${gpu_bus}"
        echo "gpu_local_cpus=${gpu_local_cpus}"
        echo "start=$(date +%s)"
    } > "${run_dir}/binding.txt"

    if [ -n "$cpuset" ] && [ -n "$gpu_local_cpus" ] && [ -n "$helper" ]; then
        if ! "$python" "$helper" subset "$cpuset" "$gpu_local_cpus" >/dev/null 2>&1; then
            echo "badbind=1" >> "${run_dir}/binding.txt"
            msg="BINDING_MISMATCH in ${run_dir}: cpuset=${cpuset} not within gpu_local_cpus=${gpu_local_cpus} (gpu_bus=${gpu_bus})"
            if [ "${STRICT_BINDING:-0}" = "1" ]; then
                echo "$msg" >&2
                exit 97
            else
                echo "WARNING: $msg" >&2
            fi
        fi
    fi
fi

exec "$bin" "$@"
