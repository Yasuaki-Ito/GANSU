#!/bin/bash


source /etc/profile.d/modules.sh
module load cuda/12.9/12.9.1
export NCCL_HOME=$HOME/local/nccl-2.29
export CPATH=$NCCL_HOME/include:$CPATH
export LIBRARY_PATH=$NCCL_HOME/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$NCCL_HOME/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HDF5_ROOT/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=$NCCL_HOME/lib/libnccl.so:$LD_PRELOAD
module load hpc_sdk/25.5 nvhpc-hpcx-cuda12/25.5
module load hpcx
which mpicc
module load python/3.10/3.10.16
source /python/env/path
export UCX_TLS=rc,cuda_copy,cuda_ipc
export NCCL_IB_DISABLE=0
export NCCL_P2P_LEVEL=NVL
