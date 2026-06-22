/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * You may obtain a copy of the license in the LICENSE file
 * located in the root directory of this source tree or at:
 * https://opensource.org/licenses/BSD-3-Clause
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef GANSU_EOM_CHAIN_CONTEXT_HPP
#define GANSU_EOM_CHAIN_CONTEXT_HPP

#include <vector>
#include <string>

#include "types.hpp"
#include "device_host_memory.hpp"
#include "cis_nto_active_space.hpp"   // CISNTOResult
#include "ip_eom_result.hpp"          // IPEOMResult
#include "ea_eom_result.hpp"          // EAEOMResult
#include "steom_result.hpp"           // STEOMResult
#include "steom_barh_cache.hpp"       // SteomBarHCache

namespace gansu {

/**
 * @brief Standalone electronic state + inter-stage scratch for the
 *        CIS-NTO → IP-EOM → EA-EOM → STEOM composite chain (DMET-STEOM, §3 of
 *        AQUA/DMET_STEOM.md).
 *
 * The composite STEOM chain is normally bound to a global `RHF&`: the four
 * drivers (`compute_cis_impl` / `compute_ip_eom_ccsd_impl` /
 * `compute_ea_eom_ccsd_impl` / `compute_steom_ccsd_impl`) read the reference
 * electronic state (C, ε, nmo, n_elec, n_frozen) from the RHF and pass the
 * inter-stage results (CIS-NTO active space, IP/EA roots, shared bar-H) between
 * stages through result members on the same RHF object.
 *
 * For an embedded DMET cluster there is no `RHF` whose `num_basis` /
 * `num_electrons` equal the cluster MO count (those are `const` in HF, fixed at
 * the AO basis size — see hf.hpp). Instead of a fake RHF view, the chain is
 * parameterised on this context: when a driver is handed a non-null
 * `EOMChainContext*` it sources the cluster electronic state and the inter-stage
 * results FROM THE CONTEXT, while the `RHF&` it still receives supplies only
 * read-only configuration (thresholds / verbose / spin / num_gpus) and the
 * AO-basis metadata + integral engine (the cluster MOs are genuine LCAO over the
 * full-molecule AO basis, so the full-molecule ERI engine transforms them
 * correctly via build_mo_eri / build_B_mo).
 *
 * Owning the inter-stage results here (rather than on a shared RHF) makes the
 * chain re-entrant: independent fragments / clusters can run concurrently
 * without clobbering each other's CIS-NTO/IP/EA state, and a per-fragment
 * cluster CIS-NTO never collides with a full-system CIS-NTO stored on the RHF
 * (DMET_STEOM.md §Step B).
 *
 * Phase-0 contract: when every driver is passed `ctx == nullptr` the code path
 * is byte-identical to the legacy RHF-bound chain (plain STEOM). The DMET-STEOM
 * reduction test (fragment = whole molecule) then matches plain STEOM bit-exact.
 */
struct EOMChainContext {
    // -------------------------------------------------------------------------
    // Cluster electronic state (NOT owned — the standalone entry point owns the
    // backing DeviceHostMatrix/Memory and keeps them alive for the chain).
    // -------------------------------------------------------------------------
    DeviceHostMatrix<real_t>* C   = nullptr;   ///< cluster MO coefficients [nao_ao × nmo] (LCAO over the real AO basis)
    DeviceHostMemory<real_t>* eps = nullptr;   ///< cluster orbital energies [nmo] (level-shifted)

    int nmo       = 0;   ///< cluster MO count (= n_emb); stands in for rhf.get_num_basis()
    int n_elec    = 0;   ///< cluster electron count (= 2·n_emb_occ); stands in for rhf.get_num_electrons()
    int n_frozen  = 0;   ///< cluster frozen-core count; stands in for rhf.get_num_frozen_core()
    int nao_ao    = 0;   ///< real AO count, for CIS oscillator-strength dipole integrals.
                         ///< Equals `nmo` on the full path / Phase-0 (square C); for a true
                         ///< rectangular-C cluster it is the full-molecule AO count (Phase 1).

    // -------------------------------------------------------------------------
    // Inter-stage results — OWNED here (re-entrant; isolated from the RHF's).
    // -------------------------------------------------------------------------
    CISNTOResult   cis_nto_result;   ///< stage 1 → consumed by IP/EA/STEOM for active-space routing
    IPEOMResult    ip_eom_result;    ///< stage 2 → consumed by STEOM (Ŝ^IP)
    EAEOMResult    ea_eom_result;    ///< stage 3 → consumed by STEOM (Ŝ^EA)
    STEOMResult    steom_result;     ///< final STEOM excited states
    SteomBarHCache barh;             ///< shared dressed bar-H across IP→EA→STEOM (build_dressed de-dup)
    bool share_barh = false;         ///< stands in for rhf.steom_share_barh()

    // Reporting outputs (kept local so a cluster run does not overwrite the
    // full-molecule RHF's report / excitation-energy / oscillator-strength
    // vectors). The standalone entry surfaces these after the chain returns.
    std::vector<real_t>      excitation_energies;
    std::vector<real_t>      oscillator_strengths;
    std::string              excited_state_report;
    real_t                   post_hf_energy = 0.0;   ///< CCSD ground re-solve energy (informational)

    // -------------------------------------------------------------------------
    // Path flags — the cluster runs the CANONICAL chain, never the DLPNO/native
    // EOM branches (which assume per-pair PNO state + square C + overlap). These
    // are forced false so the drivers take the canonical code paths regardless
    // of how the full-molecule RHF was configured.
    // -------------------------------------------------------------------------
    bool use_dlpno_amplitudes = false;
    bool use_dlpno_native_eom = false;
    bool use_dlpno_projected_eom = false;

    // ----- convenience accessors mirroring the RHF getters used by the chain ---
    int get_num_basis()        const { return nmo; }
    int get_num_electrons()    const { return n_elec; }
    int get_num_frozen_core()  const { return n_frozen; }
};

class RHF;  // config + AO-basis source (full molecule); see EOMChainContext doc.

// -----------------------------------------------------------------------------
// Cluster (DMET-STEOM) composite-chain stages — the canonical CIS-NTO → IP-EOM →
// EA-EOM sequence run over a PRECOMPUTED cluster MO-ERI tensor (`d_eri_mo`,
// device, [nmo⁴]). Each reads the cluster electronic state + writes its result
// into `ctx`; `cfg` is the full-molecule RHF supplying read-only config + AO
// basis. Defined alongside their RHF-bound counterparts:
//   compute_cluster_cis_nto      → src/eri_stored_cis.cu
//   compute_cluster_ip_eom_ccsd  → src/eri_stored_ip_eom_ccsd.cu
//   compute_cluster_ea_eom_ccsd  → src/eri_stored_ea_eom_ccsd.cu
// The STEOM driver (src/eri_stored_steom_ccsd.cu) invokes these directly on the
// cluster path instead of the polymorphic eri_method.compute_* dispatch, so no
// ERI engine is needed (the integrals are already in MO basis).
// -----------------------------------------------------------------------------
void compute_cluster_cis_nto(RHF& cfg, EOMChainContext& ctx,
                             real_t* d_eri_mo, int n_states_cis);
void compute_cluster_ip_eom_ccsd(RHF& cfg, EOMChainContext& ctx,
                                 real_t* d_eri_mo, int n_states);
void compute_cluster_ea_eom_ccsd(RHF& cfg, EOMChainContext& ctx,
                                 real_t* d_eri_mo, int n_states);

class ERI;  // integral engine (full molecule); only its type is needed here.

// -----------------------------------------------------------------------------
// DMET-STEOM standalone cluster entry — the STEOM analogue of
// `ccsd_spatial_orbital` (dmet.cu). Runs the full canonical CIS-NTO → IP-EOM →
// EA-EOM → STEOM chain on an embedded cluster described purely by raw arrays:
//   d_C_can  [nao × n_emb]  cluster canonical MO coefficients (LCAO over real AO)
//   d_eps    [n_emb]        cluster orbital energies (level-shifted)
//   d_eri_mo [n_emb⁴]       precomputed cluster MO-ERI (e.g. eri.build_mo_eri)
// `cfg` is the full-molecule RHF (read-only config + AO basis); `eri_method` is
// only carried for the legacy dispatch signature — the cluster path never uses
// it (integrals are already in MO basis). Returns the cluster STEOM result.
// Defined in src/eri_stored_steom_ccsd.cu.
STEOMResult steom_spatial_orbital(RHF& cfg, ERI& eri_method,
                                  const real_t* d_C_can,
                                  const real_t* d_eps,
                                  real_t* d_eri_mo,
                                  int nao, int n_emb, int n_emb_occ,
                                  int n_states, int n_frozen);

} // namespace gansu

#endif // GANSU_EOM_CHAIN_CONTEXT_HPP
