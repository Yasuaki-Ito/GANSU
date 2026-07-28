#!/usr/bin/env bash
# GANSU: GPU Accelerated Numerical Simulation Utility
# Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
# SPDX-License-Identifier: BSD-3-Clause
#
# Solve the doxorubicin group-0 fragment (19 atoms) with the CHEAPER ADC(2)
# cluster solver, to reach a fragment that was too large for the STEOM-CCSD
# cluster solve here (it OOM'd; see the paper's Section 4.5).
# -------------------------------------------------------------------------
# This is the practical payoff of solver independence (paper Section 4.7): the
# automatic selection produces a standard embedded cluster, so when the cluster
# is too big for STEOM-CCSD one can hand the SAME cluster to a lighter
# excited-state method. ADC(2) has no cluster CCSD, no IP/EA-EOM and no STEOM
# dressing, so its memory footprint is well below the STEOM chain that ran out.
#
# If this solves, the paper's "too large for the GPU memory available here"
# sentence becomes an actual DMET-ADC(2) energy for a driver-produced fragment,
# and Section 4.7 gains a concrete reason solver independence is useful.
#
# group-0 = manual-18 anthraquinone PLUS atom 10 (strict superset). 0-based,
# verbatim from figures/make_figures.py DOX_GROUP0.
#
# Runs on remote GPU box (s177: H200x4); needs 4 GPUs free. cd ~/GANSU/build.
set -euo pipefail

GANSU=./gansu
XYZ=../xyz/large_molecular/Doxorubicin.xyz
AUX=../auxiliary_basis/cc-pvdz-rifit.gbs
FRAG="{4,5,8,9,10,16,17,23,25,26,27,30,31,32,33,34,35,36,37}"   # dox group-0, 19 atoms

export GANSU_DMET_LEVEL_SHIFT_DENOM_ONLY=1   # ADC(2) is fed the physical (un-shifted)
                                             # epsilon in its denominators (handled
                                             # internally); keep this on for parity
export GANSU_DMET_STEOM_BATH_DIAG=1          # print the gauge (expect SUFFICIENT)

LOG=/tmp/dox_group0_ccpvdz_adc2.log
echo ">>> DMET-ADC(2) doxorubicin group-0 / cc-pVDZ -> $LOG"
$GANSU -x $XYZ -g cc-pvdz --eri_method ri -ag $AUX \
  --post_hf_method dmet_steom \
  --dmet_excited_method adc2 \
  --dmet_fragments "$FRAG" \
  --n_excited_states 5 \
  --frozen_core auto --initial_guess sad --num_gpus 4 \
  2>&1 | tee "$LOG"

echo
echo "== record: the DMET-ADC(2) excitation energies (eV) for dox group-0 =="
echo "   NB: ADC(2) energies are not directly comparable to the STEOM numbers in"
echo "   Table 1 (different excited-state method); the point is that the"
echo "   driver-produced fragment now SOLVES, under the cheaper solver."
echo "   For an apples-to-apples anchor, the whole-molecule DMET-ADC(2) vs"
echo "   standard ADC(2) match is already established for naphthalene (Table 6)."

# If this still OOMs on 4 GPUs:
#   --num_gpus 8   on the H200x8 box (ADC(2) should fit there with room to spare),
# and if it fits, that is still a clean result to report.
