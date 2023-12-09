PETSC_DIR := /home/yizhang/petsc
PETSC_ARCH := arch-linux-c-debug
PETSCDM_PATH := /home/yizhang/petsc/include/petsc/finclude
MPI_COMPILER := mpicc

# Run in shell when operation fail
# start oneAPI environment when you use oneAPI mpi: ource /opt/intel/oneapi/setvar.sh
# export PETSC_DIR=/home/harribelwww/download/petsc
# export PETSC_ARCH=arch-linux-c-debug   /   export PETSC_ARCH=arch-linux-c-opt
# export LD_LIBRARY_PATH=${PETSC_DIR}/${PETSC_ARCH}/lib:$LD_LIBRARY_PATH


solve_mpivff:solve_mpivff.cpp
	${MPI_COMPILER} -shared solve_mpivff.cpp -o solve_mpivff.so -I${PETSC_DIR}/include -I${PETSC_DIR}/${PETSC_ARCH}/include -L${PETSC_DIR}/${PETSC_ARCH}/lib -lpetsc -fPIC -O3

all: solve_mpivff

clean:
	rm -f solve_mpivff.so
	