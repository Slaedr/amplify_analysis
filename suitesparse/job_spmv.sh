if [ ! -n "$GINKGO_BUILD_DIR" ]; then
    echo Please set GINKGO_BUILD_DIR.
	exit -1
fi
if [ ! -n "$EXECUTOR" ]; then
    echo Please set EXECUTOR.
	exit -1
fi
if [ ! -n "$RESULTS_DIR" ]; then
    echo RESULTS_DIR not set.
	echo Setting it to the current directory: `pwd`
	export RESULTS_DIR=`pwd`
fi
if [ ! -n "$SYSTEM_NAME" ]; then
    echo SYSTEM_NAME not set. Setting it to "unspecified".
	export SYSTEM_NAME=unspecified
fi

export BENCHMARK=spmv
export MATRIX_LIST_FILE=matrices.txt
export AMP_BASE_TYPE=csr
export FORMATS=amp
export AMP_TOLERANCE_TYPE=componentwise
export AMP_TOLERANCE=1e-9
export BENCHMARK_PRECISION=double
export PRECONDS=jacobi

export REPETITIONS=20
export SOLVER_REPETITIONS=3
#export SYSTEM_NAME=<set before calling>
#export EXECUTOR=<set before calling>
export SOLVERS=gmres
export SOLVERS_GMRES_RESTART=80
export SOLVERS_JACOBI_MAX_BS=1
export SOLVER_RHS=1

cd ${GINKGO_BUILD_DIR}/benchmark
./run_all_benchmarks.sh
mv results ${RESULTS_DIR}/results-${BENCHMARK}-${AMP_BASE_TYPE}
