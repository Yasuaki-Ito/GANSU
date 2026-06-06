/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <array>
#include <vector>

#include "types.hpp"

namespace gansu {

/**
 * @brief GPU helper for the dominant per-triple ERI build step in
 *        build_eri_in_tno: per-Q B_TTQ[a,b,Q] = Q_tno^T · B_ao_ao[Q] · Q_tno.
 *
 * For TEOS-class molecules (nao~280, naux~1500), the CPU per-Q gather + DGEMM
 * loop dominates the DLPNO-(T) CPU prep. Moving it to GPU via cuBLAS strided
 * batched DGEMM gives a >50× per-triple speedup.
 *
 * Usage (per OMP thread, after cudaSetDevice):
 *   EriBuildGpu eri_gpu(B_ao_ao_host, nao, naux);
 *   // ...inside per-triple loop...
 *   eri_gpu.build_b_ttq(Q_tno_host, n_tno, B_TTQ_host_out);
 *
 * Falls back to active()=false if GPU is unavailable or alloc fails; caller
 * should use the CPU path (build_eri_in_tno) in that case.
 */
class EriBuildGpu {
public:
    /// Upload B_ao_ao, B_lmo_ao, B_lmo_lmo to device.
    /// max_n_tno is the upper bound on TNO size for per-triple scratch.
    EriBuildGpu(const real_t* B_ao_ao_host,
                const real_t* B_lmo_ao_host,
                const real_t* B_lmo_lmo_host,
                int nao, int nocc, int naux, int max_n_tno);

    ~EriBuildGpu();
    EriBuildGpu(const EriBuildGpu&)            = delete;
    EriBuildGpu& operator=(const EriBuildGpu&) = delete;

    bool active() const { return active_; }

    /// Compute K_iadc and the 6 off-diagonal M tensors for one triple, all
    /// on GPU. K and M are returned to host buffers.
    ///   K_iadc layout:  3 × n × n × n  (row-major flat).
    ///   M[slot] layout: nocc × n       (row-major), slot = sp*3+sq, p≠q.
    bool build_eri_and_m(const real_t* Q_tno_host,
                         int n_tno,
                         const int triple_lmos[3],
                         std::vector<real_t>& K_iadc_out,
                         std::array<std::vector<real_t>, 9>& M_out,
                         bool download = true);

    /// Device-resident batch path (GANSU_DLPNO_T_DEVICE_PACK): run the same
    /// DGEMMs but leave K/M on device (skip the D2H) and record `ev` on the
    /// internal stream so a consumer on another stream can cudaStreamWaitEvent.
    /// Returns false if inactive. Read the results via device_K()/device_M().
    bool build_eri_and_m_device(const real_t* Q_tno_host,
                                int n_tno,
                                const int triple_lmos[3],
                                void* ev /*cudaEvent_t*/);

    /// Device pointers to the most recent build's K_iadc (3·n³, row-major
    /// (s,a,b,d) stride max_n_tno-padded? NO — contiguous stride n) and M
    /// (9·nocc·n, (slot,l,a) stride n; only 6 off-diag slots valid).
    real_t* device_K() const { return d_K_iadc_; }
    real_t* device_M() const { return d_M_; }

private:
    bool  active_ = false;
    int   nao_    = 0;
    int   nocc_   = 0;
    int   naux_   = 0;
    int   max_n_  = 0;

    void* stream_ = nullptr;   // cudaStream_t
    void* cublas_ = nullptr;   // cublasHandle_t

    // Once-uploaded global RI tensors:
    real_t* d_B_ao_ao_Qmunu_ = nullptr;   // naux × nao × nao  (Q-major)
    real_t* d_B_lmo_ao_      = nullptr;   // nocc × nao × naux (native (l, ν, Q))
    real_t* d_B_lmo_lmo_     = nullptr;   // nocc × nocc × naux

    // Per-triple scratch (sized to max_n):
    real_t* d_Q_tno_         = nullptr;   // nao × max_n
    real_t* d_MQ_T_          = nullptr;   // naux × nao × max_n
    real_t* d_B_TTQ_         = nullptr;   // naux × max_n × max_n  (Q-major)
    real_t* d_B_TTQ_abQ_     = nullptr;   // max_n × max_n × naux  (a,b,Q layout)
    real_t* d_B_lTQ_         = nullptr;   // nocc × max_n × naux   (l, a, Q)
    real_t* d_K_iadc_        = nullptr;   // 3 × max_n³
    real_t* d_M_             = nullptr;   // 9 × nocc × max_n (off-diag slots)

    // Pinned host staging for downloads:
    real_t* h_pinned_K_      = nullptr;   // 3 × max_n³
    real_t* h_pinned_M_      = nullptr;   // 9 × nocc × max_n
};

} // namespace gansu
