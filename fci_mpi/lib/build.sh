#!/bin/bash
set -e

CUDA_HOME=/usr/local/cuda-12.9
NVCC=${CUDA_HOME}/bin/nvcc
NCCL_ROOT=$HOME/local/nccl2.29
MPI_INC=/usr/lib/x86_64-linux-gnu/openmpi/include
MPI_LIB=/usr/lib/x86_64-linux-gnu/openmpi/lib
NUMPY_INC=$(python3 -c "import numpy; print(numpy.get_include())")
which mpicc
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

# ===================================================
# Copy NCCL libraries to dist directory (default: disabled)
# Usage: COPY_NCCL=1 ./build.sh
# ===================================================
COPY_NCCL=${COPY_NCCL:-0}

if [ "${COPY_NCCL}" = "1" ]; then
    echo "=== Collecting NCCL libraries ==="
    for lib in ${NCCL_ROOT}/lib/libnccl.so.2*; do
        if [ -f "$lib" ]; then
            cp $lib ${DIST_DIR}/
            echo "  Copied: $(basename $lib)"
        fi
    done

    # Create symbolic links
    cd ${DIST_DIR}
    for lib in libnccl.so.2*; do
        base=$(echo $lib | sed 's/\.so\.\([0-9]*\)\..*/\.so\.\1/')
        base2=$(echo $lib | sed 's/\.so\..*/\.so/')
        [ "$base" != "$lib" ] && [ ! -f "$base" ] && ln -sf $lib $base && echo "  Symlink: $base -> $lib"
        [ ! -f "$base2" ] && ln -sf $lib $base2 && echo "  Symlink: $base2 -> $lib"
    done
    cd -
else
    echo "=== Skipping NCCL copy (COPY_NCCL=0) ==="
fi


echo ""
echo "=== Done! ==="
echo "Contents of ${DIST_DIR}:"
ls -la ${DIST_DIR}/
