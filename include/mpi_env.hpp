/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 */

/**
 * @file mpi_env.hpp
 * @brief RAII wrapper for the MPI runtime (MPI + NCCL hybrid scale-out).
 *
 * Step 0 of the MPI scale-out plan (see MPI_DESIGN.md). This header only
 * provides process/rank scaffolding — no collectives, no GPU pinning logic.
 * GPU pinning is done by the launcher (script/gansu_mpi.sh) via
 * CUDA_VISIBLE_DEVICES=$LOCAL_RANK, so each rank sees its GPU as device 0
 * and the existing device-0 code paths run per-rank unchanged.
 *
 * When GANSU_MPI is NOT defined, every accessor degrades to a single-rank
 * stub (world_rank()==0, world_size()==1, is_mpi()==false) so the rest of
 * the code can call these unconditionally and the non-MPI build is
 * byte-identical to before.
 *
 * Usage (HF_main.cu):
 *   int main(int argc, char* argv[]) {
 *       gansu::MpiEnv mpi(argc, argv);   // MPI_Init here, MPI_Finalize on scope exit
 *       ...
 *   }
 */

#pragma once

#ifdef GANSU_MPI
#include <mpi.h>
#endif

namespace gansu {

/**
 * @brief RAII handle for the MPI runtime.
 *
 * Constructing the object initializes MPI (once) and caches the world rank,
 * world size, and node-local rank. Destruction finalizes MPI. Because it is
 * instantiated at the top of main(), every return path out of main() (normal
 * or exception) finalizes MPI exactly once.
 *
 * In a non-MPI build this is an empty object reporting a single rank.
 */
class MpiEnv {
public:
    MpiEnv(int& argc, char**& argv) {
#ifdef GANSU_MPI
        int already = 0;
        MPI_Initialized(&already);
        if (!already) {
            MPI_Init(&argc, &argv);
            owns_ = true;
        }
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

        // Node-local rank: split the world by shared-memory domain (one node),
        // then take this process's rank within that node. This is what the
        // launcher maps to CUDA_VISIBLE_DEVICES, and what later steps use to
        // pick the per-rank GPU when not relying on the launcher.
        MPI_Comm node_comm = MPI_COMM_NULL;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank_,
                            MPI_INFO_NULL, &node_comm);
        MPI_Comm_rank(node_comm, &local_rank_);
        MPI_Comm_size(node_comm, &local_size_);
        MPI_Comm_free(&node_comm);
#else
        (void)argc;
        (void)argv;
#endif
    }

    ~MpiEnv() {
#ifdef GANSU_MPI
        if (owns_) {
            int finalized = 0;
            MPI_Finalized(&finalized);
            if (!finalized) MPI_Finalize();
        }
#endif
    }

    MpiEnv(const MpiEnv&) = delete;
    MpiEnv& operator=(const MpiEnv&) = delete;

    /// This process's rank in MPI_COMM_WORLD (0 if non-MPI build).
    int world_rank() const { return world_rank_; }
    /// Total number of ranks (1 if non-MPI build).
    int world_size() const { return world_size_; }
    /// This process's rank within its node (0 if non-MPI build).
    int local_rank() const { return local_rank_; }
    /// Number of ranks on this node (1 if non-MPI build).
    int local_size() const { return local_size_; }
    /// True when running under MPI with more than one rank.
    bool is_mpi() const { return world_size_ > 1; }
    /// True on the world-root rank (always true in a non-MPI build).
    bool is_root() const { return world_rank_ == 0; }

private:
    int world_rank_ = 0;
    int world_size_ = 1;
    int local_rank_ = 0;
    int local_size_ = 1;
#ifdef GANSU_MPI
    bool owns_ = false;
#endif
};

} // namespace gansu
