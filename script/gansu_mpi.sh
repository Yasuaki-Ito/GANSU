#!/usr/bin/env bash
#
# GANSU: GPU Accelerated Numerical Simulation Utility
# Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
# BSD 3-Clause License.
#
# MPI launcher wrapper (MPI + NCCL hybrid scale-out, Step 0 — see MPI_DESIGN.md).
#
# Pins each MPI rank to exactly ONE GPU via CUDA_VISIBLE_DEVICES = node-local
# rank. After pinning, every rank sees its GPU as device 0, so the existing
# device-0 code paths (cudaSetDevice(0), native operators) run per-rank
# unchanged. This is the core trick that lets MPI scale-out avoid an
# operator device-parameterization refactor.
#
# Usage:
#   mpirun -np 8 --bind-to none ./script/gansu_mpi.sh ./gansu -x ../xyz/H2O.xyz -g cc-pvdz ...
#
# Notes:
#   --bind-to none lets each rank use all CPU cores it needs (DLPNO host loops
#   are OpenMP-parallel); GPU binding is what we control here, not CPU binding.
#
# Resolve this rank's node-local index from whichever launcher set it.
if   [ -n "${OMPI_COMM_WORLD_LOCAL_RANK}" ]; then
    LOCAL_RANK="${OMPI_COMM_WORLD_LOCAL_RANK}"     # Open MPI
elif [ -n "${MV2_COMM_WORLD_LOCAL_RANK}" ]; then
    LOCAL_RANK="${MV2_COMM_WORLD_LOCAL_RANK}"      # MVAPICH2
elif [ -n "${SLURM_LOCALID}" ]; then
    LOCAL_RANK="${SLURM_LOCALID}"                  # Slurm srun
elif [ -n "${PMI_LOCAL_RANK}" ]; then
    LOCAL_RANK="${PMI_LOCAL_RANK}"                 # MPICH / Intel MPI
else
    echo "[gansu_mpi.sh] WARNING: could not determine node-local rank; defaulting GPU 0" >&2
    LOCAL_RANK=0
fi

export CUDA_VISIBLE_DEVICES="${LOCAL_RANK}"

exec "$@"
