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

BENCH_DIR=$GINKGO_BUILD_DIR/benchmark

export BENCHMARK=solver
export MATRIX_LIST_FILE=structurally_symmetric_subset.txt
export BENCHMARK_PRECISION=double
#export EXECUTOR=<set>
#export SYSTEM_NAME=<set>

export SOLVERS=gmres
export SOLVERS_GMRES_RESTART=80
export PRECONDS=fgs
export SOLVERS_REORDER=multicolor
export SOLVERS_FGS_SWEEPS=1

export SOLVER_REPETITIONS=3
export SOLVERS_PRECISION=1e-10
export SOLVERS_MAX_ITERATIONS=10000
export SOLVERS_RHS=1
export SOLVERS_INITIAL_GUESS=0

export AMP_BASE_TYPE=csr
export AMP_TOLERANCE_TYPE=componentwise
export AMP_TOLERANCE=1e-9

cp $MATRIX_LIST_FILE $BENCH_DIR/$MATRIX_LIST_FILE

cd $BENCH_DIR

export FORMATS=${AMP_BASE_TYPE}
./run_all_benchmarks.sh
mv results $RESULTS_DIR/results-solver_fgs-${AMP_BASE_TYPE}

export FORMATS=amp
./run_all_benchmarks.sh
mv results $RESULTS_DIR/results-solver_fgs-amp_${AMP_BASE_TYPE}

rm $BENCH_DIR/$MATRIX_LIST_FILE
