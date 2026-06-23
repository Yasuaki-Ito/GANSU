#!/bin/bash

source ./env.sh
NCORES=96
LOCAL_SIZE=${OMPI_COMM_WORLD_LOCAL_SIZE}
LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK}

ncores_rank=$((${NCORES}/${LOCAL_SIZE}))
first_core=$((${ncores_rank}*${LOCAL_RANK}))
cores="${first_core}-$((${first_core}+${ncores_rank}-1))"

echo "rank${OMPI_COMM_WORLD_RANK} on $(hostname): cores=${cores}, gpu=${LOCAL_RANK}"
export OMP_NUM_THREADS=${ncores_rank}
export CUDA_VISIBLE_DEVICES=${LOCAL_RANK}
if [ $LOCAL_RANK -lt 2 ]; then
    export UCX_NET_DEVICES="mlx5_0:1"
else
    export UCX_NET_DEVICES="mlx5_1:1"
fi

taskset -c ${cores} $@
