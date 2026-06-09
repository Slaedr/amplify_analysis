if [ $# -lt 1 ]; then
    echo "Usage: $0 <config.json>"
    exit 1
fi

${GINKGO_BUILD_DIR}/benchmark/amp/amp_benchmark_spmv $1

${GINKGO_BUILD_DIR}/benchmark/amp/amp_benchmark_fgs $1

${GINKGO_BUILD_DIR}/benchmark/amp/amp_benchmark_gmres $1
