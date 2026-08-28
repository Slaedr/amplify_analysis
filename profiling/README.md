# AMP SpMV performance tooling (MI250X / MI210)

Hardware-counter profiling of the AMP SpMV kernels in the Ginkgo fork. This is
a third experiment kind alongside `random_stencil/` and `suitesparse/`, and it
follows the same environment-variable conventions (`GINKGO_BUILD_DIR`,
`EXECUTOR`, `RESULTS_DIR`, `SYSTEM_NAME`, `AMP_*`).

Three tools, meant to be run in this order:

| | tool | needs a GPU? | answers |
|---|---|---|---|
| 1 | `amp_bin_predict.py` | no | *Is there anything to win on this matrix at this tolerance, and how much does the current kernel shape throw away?* |
| 2 | `amp_spmv_profile.sh` | yes | collects the counters |
| 3 | `amp_spmv_report.py` | no | reduces them to the ten numbers that matter |

`job_profile_spmv.sh` runs all three in sequence; `frontier_profile.sbatch`
wraps that in a single-GCD Slurm step.

Run step 1 first. It is a few hundred milliseconds per matrix and it will tell
you, before you spend a node-hour, which matrices are even capable of showing a
speed-up.

## Why this does not go through `run_all_benchmarks.sh`

`suitesparse/job_spmv.sh` drives Ginkgo's `run_all_benchmarks.sh`, which loops
over matrix *names* and fetches them with `ssget`. Counter collection needs a
single `benchmark/spmv` process with exclusive use of one GCD, which that
driver cannot arrange, so these scripts invoke `benchmark/spmv` directly and
take explicit `.mtx` **paths** — the files `ssget` left under
`$GINKGO_BUILD_DIR/benchmark/matrices`, not the bare names in
`suitesparse/matrices.txt`. Put one absolute path per line in
`matrix_paths.txt` (or point `MATRIX_PATH_FILE` elsewhere).

Two other deliberate differences from `suitesparse/job_spmv.sh`:

* **`FORMATS=csr,amp`, not `amp` alone.** Almost every derived metric in the
  report — `x_reread_factor`, traffic inflation, "% of the achievable win" — is
  a comparison between AMP and the uniform-fp64 baseline *on the same matrix
  and the same GPU*. Without the `csr` run there is nothing to attribute a
  slowdown against.
* **`REPETITIONS` is per counter pass.** The script runs one clean
  kernel-trace pass for timing and then one `rocprofv3` invocation per counter
  group, so wall-clock is roughly (number of groups + 1) times a normal
  benchmark run.

---

## The short version of what to look at

Ordered by how likely each is to be the actual limiter for `AMP[CSR]` SpMV as
the kernel stands in `common/cuda_hip/matrix/amp_kernels.cpp`.

### 1. The bytes ceiling — check this before anything else

SpMV is bandwidth bound, so the *only* thing that makes AMP faster is moving
fewer bytes. Per nonzero, CSR fp64 moves 8 (value) + 4 (int32 column index) =
12 B. Demoting a value to fp32 gives 8 B, to bf16/fp16 gives 6 B. So:

* all-fp32 caps at **1.5×**, all-fp16 at **2.0×** — the column index rides
  along regardless of value precision and never shrinks.
* `AMP[CSR]` also stores **q separate row-pointer arrays** (q = 3 with
  bfloat16 enabled), i.e. `3·(nrows+1)·4` bytes instead of `(nrows+1)·4`. On a
  short-row matrix that is a real fraction of total traffic — the predictor
  prints it.

`amp_bin_predict.py` prints `AMP speed-up ceiling vs CSR fp64`. If that number
is below ~1.1 there is nothing to optimise; the kernel is not the problem, the
format is. **At the default `--amp_tolerance=1e-14` this is the common case.**

Why: the thresholds in `amp_algorithms.hpp` are
`lbs[0] = τ·‖a_i‖₁ / ε_float`, so with τ=1e-14 anything above
`8.4e-8·‖a_i‖₁` stays in bin 0. Ordinary well-scaled rows put essentially every
entry in fp64, and you pay the q-way loop overhead for zero storage benefit.
Confirmed on `ani4.mtx`: 100 % bin 0 at τ=1e-14, 95 % at τ=1e-10, and only at
τ=1e-6 does it become 15/83/2 % — where the ceiling finally rises to 1.18×.

### 2. Wavefront lane utilisation — your hypothesis, and it is worse than one wave per row

`csr_amp_basic_spmv` gives one **64-wide** wavefront to one row, and then loops
over the q bins *inside* that wavefront. Each bin's segment of the row is a
*fraction* of the row's nonzeros, so the utilisation is not `nnz_row/64`, it is
`nnz_row/(64·q_active)`. The bin split makes the short-row problem q times
worse.

Measured metric: `VALUUtilization = SQ_THREAD_CYCLES_VALU / (SQ_ACTIVE_INST_VALU · 64)`.
Predicted metric: `amp_bin_predict.py`'s `lane_util[64]['split']`. They should
agree within a few points; if they do not, something else (the reduction, the
`x_stride` multiply, address arithmetic) is generating VALU work.

The predictor also prints what a **merged** loop (one pass over the row, bins
interleaved) and **narrower subwarp tiles** would give, which is exactly the
design decision to make. For `ani4` at τ=1e-6: 5 % at tile 64 split, 11 %
merged, 58 % at tile 4. Ginkgo's own classical CSR SpMV already picks its
subwarp size from `nnz/row` — `csr_amp_basic_spmv` hardcodes `config::warp_size`
and does not.

### 3. Where the L1/L2 misses come from

Only `x` can miss usefully — `values`, `col_idxs` and `row_ptrs` are streamed
and will miss by construction. So the question is only ever *how many times did
we pull `x` out of HBM*. The report calls this **`x_reread_factor`**:

```
x_reread = (measured HBM read bytes − compulsory matrix+rowptr bytes) / (nrows · 8)
```

1.0 = perfect reuse, ≫1 = the gather pattern is defeating the 8 MB L2. This is
the number to compare between `csr` and `amp` on the same matrix, because it
isolates the AMP-specific damage: within a bin segment the column indices are a
*sparse subset* of the row's columns, so consecutive gathers are farther apart
than in plain CSR. `amp_bin_predict.py` prints exactly that
(`mean |column gap| within a (row,bin) segment` vs `within a whole row`).

Raw hit rates are still worth having, but read them in that light:
* L1 (TCP, 16 KB/CU) hit rate — mostly tells you about intra-wave x reuse.
* L2 (TCC, 8 MB/GCD) hit rate — tells you whether x fits and stays resident.
  If `nrows·8` < 8 MB (about 1 M rows) then a good kernel should hit near 100 %
  on x and `x_reread_factor` should be ≈1.

### 4. Things that are easy to overlook and are cheap to fix

* **q× scalar row-pointer loads.** `bin_row_ptrs[k][irow]`/`[irow+1]` for all
  k, and `irow` is wavefront-uniform so these should compile to `s_load`. The
  `scalar` counter set (`SQC_DCACHE_*`, `SQ_INSTS_SMEM`) tells you whether they
  did. `smem_per_row` much above 2q means the compiler kept them vector.
* **`__launch_bounds__(512)` and the q-way `constexpr_for` unroll.** Three
  instantiations with three different value types in one register live range.
  The report prints `VGPR_Count` and the resulting `occupancy_waves_per_simd`
  (gfx90a: 512 VGPRs/SIMD, 8 waves max). At ≤4 waves/SIMD a latency-bound
  gather kernel cannot hide anything. 512 threads/block is also a large block
  for a load-imbalanced kernel.
* **Lane-0-only store.** Each wavefront writes 8 bytes of `y`; 16 wavefronts
  must cooperate to fill a 128 B line. Visible as poor `WriteUnitStalled` /
  low `TCC_EA_WRREQ_64B_sum` relative to `TCC_EA_WRREQ_sum`.
* **`x[acols[iz] * x_stride + irhs]`.** `x_stride` is a runtime `uint32`, so
  every gather pays a `v_mul` the compiler cannot fold. A specialisation for
  `nrhs == 1 && x_stride == 1` is free.
* **Conversion instructions.** `static_cast<mult_type>` per nonzero for the
  narrow bins. The `fp` counter set separates `SQ_INSTS_VALU_CVT` from real
  FMA work; if conversions are a large share of VALU, packed math (`v_pk_*`,
  or 2×bf16 per lane) is on the table.
* **Load imbalance.** One wave per row with `default_block_size = 512` means 8
  waves per block wait for the longest row in the block. Look at
  `nnz_per_row_p99` vs `mean` in the predictor output.

### 5. What *not* to chase

FLOPs. At 12 B/nnz and 2 FLOP/nnz the arithmetic intensity is ~0.17 FLOP/B; on
a 1.6 TB/s GCD that caps you at ~270 GFLOP/s against a 23.9 TFLOP/s FP64 peak.
`%peak FP64` will always look terrible and it means nothing.

---

## Usage

### Step 1 — static model (no GPU)

```sh
python3 amp_bin_predict.py \
    $(cat matrix_paths.txt) \
    --tol 1e-9 --tol 1e-6 --tol 1e-4 \
    --half bf16 \
    --tile 64,32,16,8,4 \
    --json bins.json --csv bins.csv
```

`--half bf16` if the build has `GINKGO_ENABLE_BFLOAT16=ON`, else `fp16`.
numpy and scipy only — both already in `../requirements.txt`.

Start the tolerance sweep at whatever `suitesparse/job_spmv.sh` currently uses
(`AMP_TOLERANCE=1e-9`) and walk it looser. With
`lbs[0] = tau * ||a_i||_1 / eps_float`, tau=1e-9 only demotes entries below
`8.4e-3 * ||a_i||_1`, so on well-scaled PDE matrices most nonzeros are expected
to stay in bin 0 and the bytes ceiling to sit near or below 1.0x. If that is
what the sweep shows, the finding is about the tolerance and the format, not
the kernel.

### Step 2 — counters

Everything at once, in the repo's usual style:

```sh
export GINKGO_BUILD_DIR=/path/to/ginkgo-amp/build-rele
export EXECUTOR=hip
export SYSTEM_NAME=frontier          # or mi210-login
export RESULTS_DIR=$PWD
export AMP_TOLERANCE=1e-6
export MATRIX_PATH_FILE=$PWD/matrix_paths.txt

./job_profile_spmv.sh                # profile + report + static model
```

On Frontier, submit it instead — one rank on one GCD:

```sh
sbatch --export=ALL,GINKGO_BUILD_DIR=...,MATRIX_PATH_FILE=...,AMP_TOLERANCE=1e-6 \
       frontier_profile.sbatch
```

Or drive the profiler directly for one-off runs:

```sh
./amp_spmv_profile.sh --matrix-list matrix_paths.txt \
    --formats csr,amp --amp-base-type csr --amp-tolerance 1e-6 \
    --system-name mi210-login --pmc-sets wave,l2
```

Counter collection needs **exclusive use of one GCD** — one rank, one GCD, no
other process on it. The script sets `ROCR_VISIBLE_DEVICES`,
`GPU_MAX_HW_QUEUES=1` and `HSA_ENABLE_SDMA=0` for you. It runs a clean
kernel-trace pass for timing first (counters perturb clocks), then one
rocprofv3 invocation per counter group so nothing inside a group is
multiplexed. Add `--tool rocprof-compute` (or `both`) for the full memory
chart + roofline.

`--dry-run` prints every command without running anything — worth doing once on
the login node before submitting.

### Step 3 — report

`job_profile_spmv.sh` already does this; to re-run it on an existing directory:

```sh
python3 amp_spmv_report.py \
    $RESULTS_DIR/results-profile-spmv-csr-$SYSTEM_NAME --markdown
```

Prints the table plus a per-matrix diagnosis, and writes `report.json` /
`report.md`. Override the roofline constants with `--peak-bw` /
`--peak-fp64` if you want to compare against a measured BabelStream number
instead of the 1.6 TB/s spec figure (BabelStream typically lands around
1.3–1.4 TB/s per GCD, which is why `--achievable-frac` defaults to 0.85).

---

## Device notes

Counters are always **per agent**, and one MI250X exposes two agents. Every
number the report produces is therefore per-GCD:

| | CUs | HBM BW | FP64 vector | L2 |
|---|---|---|---|---|
| MI250X (1 GCD) | 110 | 1.6 TB/s | 23.9 TF/s | 8 MB |
| MI210 | 104 | 1.6 TB/s | 22.6 TF/s | 8 MB |

MI210 is a good proxy for one MI250X GCD for everything above — same gfx90a
ISA, same wavefront width, same cache sizes, 6 % fewer CUs. Develop on the
login node, confirm on Frontier.
