#!/bin/bash
set -e

export NCCL_HOME=$HOME/local/nccl
export CPATH=$NCCL_HOME/include:$CPATH
export LIBRARY_PATH=$NCCL_HOME/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$NCCL_HOME/lib:$LD_LIBRARY_PATH

export PATH=$HOME/local/openmpi/bin:$PATH
export LD_LIBRARY_PATH=$HOME/local/openmpi/lib:$LD_LIBRARY_PATH
export MANPATH=$HOME/local/openmpi/share/man:$MANPATH
MPI_INC=$HOME/local/openmpi/include
MPI_LIB=$HOME/local/openmpi/lib
which mpicc

CUDA_HOME=/usr/local/cuda-12.9
NVCC=${CUDA_HOME}/bin/nvcc
NCCL_ROOT=$HOME/local/nccl2.29

NUMPY_INC=$(python3 -c "import numpy; print(numpy.get_include())")

DIST_DIR=./build/
mkdir -p ${DIST_DIR}

INCLUDE="-I${CUDA_HOME}/include \
         -I${NUMPY_INC} \
         -I${NCCL_ROOT}/include \
         -I${MPI_INC}"

# ===================================================
# for various GPU architectures
# sm_70: V100, sm_75: T4, sm_80: A100, sm_86: RTX3090,
# sm_89: RTX4090, sm_90: H100
# compute_90: PTX (for JIT GPU)
# ===================================================
GENCODE="-gencode arch=compute_70,code=sm_70 \
         -gencode arch=compute_75,code=sm_75 \
         -gencode arch=compute_80,code=sm_80 \
         -gencode arch=compute_86,code=sm_86 \
         -gencode arch=compute_89,code=sm_89 \
         -gencode arch=compute_90,code=sm_90 \
         -gencode arch=compute_90,code=compute_90"


echo "=== Compiling fci_contract.c ==="
gcc -O2 -fPIC -c fci_contract.c -o fci_contract.o ${INCLUDE}

echo "=== Compiling fci.cu ==="
${NVCC} -O2 -Xcompiler -fPIC \
	${GENCODE} \
        -rdc=true \
        ${INCLUDE} \
        -c fci.cu -o fci.o

echo "=== Linking libfci.so ==="
${NVCC} -shared ${GENCODE} \
    fci_contract.o fci.o \
    -L${CUDA_HOME}/lib64 \
    -L${NCCL_ROOT}/lib -lnccl \
    -L${MPI_LIB} -lmpi \
    -lcublas -lcusolver \
    -Xcompiler -static-libstdc++ \
    -Xcompiler -static-libgcc \
    -Xlinker -rpath,'$ORIGIN' \
    -o ${DIST_DIR}/libfci.so

echo ""
echo "=== Done! ==="
echo "Contents of ${DIST_DIR}:"
ls -la ${DIST_DIR}/
