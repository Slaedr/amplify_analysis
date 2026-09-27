#!/usr/bin/env python3
"""Helpers for sweep_tolerance_slurm.sh: Linux cpu-list parsing, a subset
check used by slurm_binding_wrapper.sh, and a post-hoc cross-check that no
two concurrently running benchmark runs shared CPUs or a GPU.

Usage:
    slurm_sweep_helpers.py subset <cpuset> <gpu_local_cpus>
        Exit 0 if every cpu in <cpuset> is also in <gpu_local_cpus>, else 1.
        Both are Linux cpu-list strings, e.g. "0-6,64-70".

    slurm_sweep_helpers.py crosscheck <results_dir>
        Scan <results_dir> recursively for binding.txt files (written by
        slurm_binding_wrapper.sh, with a "end=<epoch>" line appended by
        sweep_tolerance_slurm.sh once the run finishes), and report any pair
        of runs whose [start, end) intervals overlap and which also share a
        CPU or a GPU PCI bus id -- which should never happen given Slurm's
        default step exclusivity, so a hit here means the static
        distribution actually let two runs collide. Exit 0 if none found (or
        nothing to check), 1 otherwise; details go to stdout either way.
"""
import sys
import glob
import os


def parse_cpulist(s):
    cpus = set()
    s = s.strip()
    if not s:
        return cpus
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            cpus.update(range(int(lo), int(hi) + 1))
        else:
            cpus.add(int(part))
    return cpus


def cmd_subset(argv):
    cpuset = parse_cpulist(argv[0])
    local_cpus = parse_cpulist(argv[1])
    return 0 if cpuset and cpuset <= local_cpus else 1


def parse_binding_file(path):
    fields = {}
    with open(path) as f:
        for line in f:
            if "=" in line:
                k, v = line.rstrip("\n").split("=", 1)
                fields[k] = v
    return fields


def cmd_crosscheck(argv):
    results_dir = argv[0]
    runs = []
    for path in glob.glob(os.path.join(results_dir, "**", "binding.txt"), recursive=True):
        f = parse_binding_file(path)
        try:
            start = int(f.get("start", ""))
            end = int(f.get("end", ""))
        except ValueError:
            continue  # run still in flight, or the wrapper never got to record it
        runs.append({
            "run_dir": os.path.dirname(path),
            "start": start,
            "end": end,
            "cpuset": parse_cpulist(f.get("cpuset", "")),
            "gpu_bus": f.get("gpu_bus", ""),
        })

    conflicts = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            a, b = runs[i], runs[j]
            overlaps_in_time = a["start"] < b["end"] and b["start"] < a["end"]
            if not overlaps_in_time:
                continue
            shares_cpu = bool(a["cpuset"] & b["cpuset"])
            shares_gpu = bool(a["gpu_bus"] and a["gpu_bus"] == b["gpu_bus"])
            if shares_cpu or shares_gpu:
                conflicts.append((a, b, shares_cpu, shares_gpu))

    if not conflicts:
        print(f"crosscheck: no conflicts found among {len(runs)} timed runs")
        return 0

    print(f"crosscheck: {len(conflicts)} conflict(s) among {len(runs)} timed runs")
    for a, b, shares_cpu, shares_gpu in conflicts:
        why = []
        if shares_cpu:
            why.append(f"shared cpus {sorted(a['cpuset'] & b['cpuset'])}")
        if shares_gpu:
            why.append(f"shared gpu_bus {a['gpu_bus']}")
        print(f"  CONFLICT: {a['run_dir']}  <->  {b['run_dir']}  ({', '.join(why)})")
    return 1


def main():
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "subset":
        return cmd_subset(rest)
    if cmd == "crosscheck":
        return cmd_crosscheck(rest)
    print(f"unknown subcommand '{cmd}'", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
