#!/usr/bin/env bash
# GANSU: GPU Accelerated Numerical Simulation Utility
# Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
# SPDX-License-Identifier: BSD-3-Clause
#
# Solve the DRIVER-PRODUCED doxorubicin group-0 fragment (19 atoms) so the paper
# can replace "the oxygen-rich cluster exceeded the memory of the dense
# diagonalization path" with an actual energy.
# -------------------------------------------------------------------------
# The earlier attempt OOM'd because it used the DENSE STEOM diagonalization
# (GANSU_STEOM_DENSE_DIAG=2), which is only needed for complex-pair counting.
# For a single reported energy the default ITERATIVE diagonalization is far
# lighter, so this run simply omits dense diag. The doxorubicin auto-19 fragment
# already solves cleanly with the DLPNO cluster solver (Table 1, 3.811 eV; gauge
# SUFFICIENT 0.0053, Table 2), so group-0 -- a strict superset of the manual 18
# and of comparable size -- should solve on the same 4-GPU box.
#
# group-0 = manual-18 anthraquinone PLUS atom 10 (a strict superset). 0-based,
# verbatim from figures/make_figures.py DOX_GROUP0.
#
# NB: NTO-bath augmentation is OFF here. It is (a) unnecessary -- the dox bare
# Schmidt bath is already SUFFICIENT -- and (b) incompatible with the DLPNO
# cluster solver (goes NaN in the PNO construction). BATH_DIAG is on only to
# print the gauge, to confirm it reads SUFFICIENT as for auto-19.
#
# Runs on remote GPU box (s177: H200x4); needs 4 GPUs free. cd ~/GANSU/build.
set -euo pipefail

GANSU=./gansu
XYZ=../xyz/large_molecular/Doxorubicin.xyz
AUX=../auxiliary_basis/cc-pvdz-rifit.gbs
FRAG="{4,5,8,9,10,16,17,23,25,26,27,30,31,32,33,34,35,36,37}"   # dox group-0, 19 atoms

export GANSU_DMET_LEVEL_SHIFT_DENOM_ONLY=1   # CORRECTNESS-CRITICAL epsilon un-shift
export GANSU_CCSD_CONV=1e-7
export GANSU_DMET_STEOM_BATH_DIAG=1          # print the gauge (expect SUFFICIENT)
# NO GANSU_STEOM_DENSE_DIAG  -> default iterative diag (this is what avoids OOM)
# NO GANSU_DMET_STEOM_NTO_BATH* -> bare Schmidt bath (sufficient for dox; also
#    required so the DLPNO solver does not hit the NTO-bath+DLPNO NaN)

LOG=/tmp/dox_group0_ccpvdz_dlpno_iter.log
echo ">>> DMET-STEOM doxorubicin group-0 / cc-pVDZ / DLPNO / iterative diag -> $LOG"
$GANSU -x $XYZ -g cc-pvdz --eri_method ri -ag $AUX \
  --post_hf_method dmet_steom \
  --dmet_fragments "$FRAG" \
  --dmet_cluster_solver dlpno \
  --n_excited_states 5 --steom_n_root_cis 14 \
  --frozen_core auto --initial_guess sad --num_gpus 4 \
  2>&1 | tee "$LOG"

echo
echo "== compare to the doxorubicin references (cc-pVDZ) =="
echo "   manual 18-atom lowest = 3.7640 eV ; auto 19-atom (B=700) lowest = 3.811 eV"
echo "   group-0 = manual-18 + atom 10, so its lowest should sit near 3.76-3.81 eV."
echo "== record: lowest DeltaE (eV), per-root eta, gauge verdict =="

# FALLBACKS if this OOMs or the DLPNO solver misbehaves:
#   1) canonical solver, still iterative diag (heavier CCSD, lighter than dense diag):
#        --dmet_cluster_solver canonical           (drop the dlpno line)
#   2) more memory: --num_gpus 8  on the H200x8 box, plus the memory-safety env
#        the completed canonical runs used:
#        export GANSU_DMET_CCSD_BNATIVE=1
#        export GANSU_STEOM_OPERATOR_DEVICE_BALANCING=1
#        export GANSU_EA_RI_LADDER=1 ; export GANSU_EA_W_HOST=1
#   Only turn dense diag back on if a complex-pair count is specifically wanted;
#   it is not needed for the single energy this run is after.
