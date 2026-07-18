#!/usr/bin/env bash
# Taxol benzamide / cc-pVDZ — gauge (in-domain) diagnostic + NTO-bath augmentation.
# Hypothesis (log214/216): the benzamide fragment is NOT in-domain — the cluster
# CIS lowest root sits at ~7.95 eV in BOTH bases, ~2-3 eV above benzamide's true
# lowest pi->pi* (~4.6-5.5 eV), so the low state is outside fragment+bath and STEOM
# throws spurious low-eta / near-defective roots.
#  BATH_DIAG=1 prints the CIS-NTO gauge (uncaptured weight -> SUFFICIENT/INSUFFICIENT).
#  NTO_BATH (tau_vir) / NTO_BATH_OCC (tau_occ) then augment the bath to recover the
#  environment-delocalised character (the 2-butanone 8 eV recovery mechanism).
# The gauge prints in the CIS-NTO stage (~1-2 min in) — you may Ctrl-C after it if
# you only want the diagnostic, or let it run to see if the roots stabilise.
# Run on remote GPU box (s177: H200x4); needs 4 GPUs free.
set -euo pipefail

GANSU=./gansu
XYZ=../xyz/large_molecular/Taxol.xyz
AUX=../auxiliary_basis/cc-pvdz-rifit.gbs
FRAG="{47,48,49,56,57,58,59,60,61}"

export GANSU_DMET_LEVEL_SHIFT_DENOM_ONLY=1
export GANSU_CCSD_CONV=1e-7
export GANSU_STEOM_DENSE_DIAG=2
export GANSU_DMET_STEOM_BATH_DIAG=1       # print gauge (in-domain check)
export GANSU_DMET_STEOM_NTO_BATH=0.01     # tau_vir  (virtual NTO-bath augmentation)
export GANSU_DMET_STEOM_NTO_BATH_OCC=0.5  # tau_occ  (occupied NTO-bath augmentation)

LOG=/tmp/taxol_steom_ccpvdz_diag.log
echo ">>> DMET-STEOM Taxol benzamide / cc-pVDZ / gauge+NTO-bath  -> $LOG"
$GANSU -x $XYZ -g cc-pvdz \
  --eri_method ri -ag $AUX \
  --post_hf_method dmet_steom \
  --dmet_fragments "$FRAG" \
  --dmet_cluster_solver dlpno --dlpno_bt_polish 3 \
  --n_excited_states 5 --steom_n_root_cis 14 \
  --initial_guess sad \
  --num_gpus 4 \
  2>&1 | tee $LOG
# Look for: "gauge ... uncaptured = X  -> SUFFICIENT / INSUFFICIENT" in the CIS-NTO
# stage, and whether the augmented run lowers the STEOM roots toward ~5 eV with
# healthy eta.  INSUFFICIENT + recovery => fragment was under-sized (redesign needed).
