AMPLify analysis scripts
========================

Job running and postprocessing scripts for AMPLify Ginkgo project experiments. There are three kinds of experiments:
- Random stencil matrices (SpMV, FGS and GMRES+FGS)
- Suitesparse matrices
  + original list of matrices: SpMV
  + structurally symmetric matrices: GMRES+FGS
- Hardware-counter profiling of the AMP SpMV kernels on AMD GPUs (MI250X, MI210)

# Input settings for the experiments

`GINKGO_BUILD_DIR`: The build directory where the Ginkgo build directory exists.

## Random stencil experiments
Change the "`executor`" and "`amp_base_format`" fields in the input file config.json.

## Suitesparse experiments
Environment variables:
- `RESULTS_DIR`: The directory to move results to.
- `SYSTEM_NAME`: Name of the system. Matters only for the reults directory structure.
- `EXECUTOR`: hip,cuda,omp or reference.

## Profiling experiments
See `profiling/README.md`. Uses the same `GINKGO_BUILD_DIR`, `RESULTS_DIR`,
`SYSTEM_NAME`, `EXECUTOR` and `AMP_*` variables as above, plus:
- `MATRIX_PATH_FILE`: file with one absolute `.mtx` path per line. Counter
  collection needs a single `benchmark/spmv` process with exclusive use of one
  GCD, so these jobs bypass `run_all_benchmarks.sh` and cannot resolve matrix
  names through `ssget`.

`profiling/amp_bin_predict.py` needs no GPU and should be run first: it models
the per-bin split, the bandwidth ceiling and the wavefront lane utilisation
straight from the matrix, and says whether a given matrix and tolerance can
show a speed-up at all.
