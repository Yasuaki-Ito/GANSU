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

/**
 * @file dmet_memory_policy.hpp
 * @brief Automatic memory-layout policy for large DMET cluster solves.
 *
 * The DMET-STEOM / DMET-ADC(2) chain has a family of switches that change only
 * HOW tensors are laid out (blocked-from-B instead of a dense cluster MO-ERI,
 * host staging, per-tile rebuilds, multi-GPU sharding). They are numerically
 * inert — the same integrals reach the same contractions — but without them a
 * large cluster asks for a tensor that cannot fit and the run dies with an OOM
 * that says nothing about which switch was missing.
 *
 * Historically these lived only in hand-written scratch scripts as a dozen
 * environment variables, so reproducing a published number meant reproducing an
 * undocumented env list (and getting it wrong meant a 247 GB allocation for a
 * cluster of n_emb = 427). This module decides them from the cluster dimensions
 * and the free device memory, so the published command line is just the method:
 *
 *   ./gansu -x mol.xyz -g cc-pvdz --eri_method ri -ag aux.gbs \
 *           --post_hf_method dmet_steom --dmet_steom_auto_fragment 1
 *
 * Every switch is applied with setenv(..., overwrite=0): an explicitly exported
 * variable always wins, so existing scripts keep their exact behaviour.
 */

#pragma once

namespace gansu {

/**
 * @brief Enable the memory-layout switches this cluster size requires.
 *
 * Call once per cluster solve, after the embedding dimensions are known and
 * BEFORE anything reads the layout environment variables.
 *
 * @param n_emb      Cluster orbital count (fragment + bath, after augmentation)
 * @param n_emb_occ  Occupied cluster orbitals (virtuals = n_emb - n_emb_occ)
 * @param verbose    >0 prints the decisions and the reason for each
 */
void dmet_apply_memory_policy(int n_emb, int n_emb_occ, int verbose = 1);

} // namespace gansu
