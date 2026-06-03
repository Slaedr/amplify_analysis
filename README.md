AMPLify analysis scripts
========================

Job running and postprocessing scripts for AMPLify Ginkgo project experiments. There are two kinds of experiments:
- Random stencil matrices (SpMV, FGS and GMRES+FGS)
- Suitesparse matrices
  + original list of matrices: SpMV
  + structurally symmetric matrices: GMRES+FGS

# Input settings for the experiments

`GINKGO_BUILD_DIR`: The build directory where the Ginkgo build directory exists.

## Random stencil experiments
Change the "`executor`" and "`amp_base_format`" in the input file config.json.

## Suitesparse experiments
`RESULTS_DIR`: The directory to move results to.
`SYSTEM_NAME`: Name of the system. Matters only for the reults directory structure.
`EXECUTOR`: hip,cuda,omp or reference.
