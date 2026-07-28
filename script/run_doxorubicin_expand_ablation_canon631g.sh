#!/usr/bin/env bash
# GANSU: GPU Accelerated Numerical Simulation Utility
# Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
# SPDX-License-Identifier: BSD-3-Clause
#
# POSITIVE example of gauge-triggered fragment expansion (Phase B) --
# CANONICAL / 6-31G route. (Separate from run_doxorubicin_expand_ablation.sh,
# which is the cc-pVDZ/DLPNO route; DLPNO is still incomplete, so this uses the
# pure canonical STEOM solver on a smaller basis that fits in H200 memory.)
# -------------------------------------------------------------------------
# Goal: the constructive counterpart to naphthalene (where Phase B improves the
# gauge but NOT the energy). Here the chromophore (anthraquinone) is genuinely
# LOCALIZED, so under-seeding the fragment and letting Phase B walk it back
# should improve BOTH the gauge AND the lowest excitation, all in pure canonical
# DMET-STEOM (no DLPNO approximation).
#
# Why this fits where cc-pVDZ canonical OOM'd: 6-31G is ~0.64x the basis of
# cc-pVDZ, so n_emb ~210 (vs 333) and the cluster 4-index MO ERI drops ~6x
# (~15 GB vs 91.6 GB) -- back inside one H200.
#
# ISOLATION: NTO-bath augmentation is OFF (tau not set) so the ONLY correction
# mechanism is atom expansion. Both arms (A/B) share identical settings; the
# only difference is --dmet_steom_auto_max_expand, so the A->B change is Phase B
# and nothing else.
#
# Runs on remote GPU box (s177: H200x4); needs 4 GPUs free. cd ~/GANSU/build.
set -euo pipefail

GANSU=./gansu
XYZ=../xyz/large_molecular/Doxorubicin.xyz
AUX=../auxiliary_basis/cc-pvdz-rifit.gbs   # 6-31G orbital basis + cc-pVDZ RI-fit,
                                           # the same convention the paper's 6-31G
                                           # ubiquinone/paclitaxel runs used (no
                                           # 6-31g-specific fit set ships here).

# Correctness + memory. NO GANSU_DMET_STEOM_NTO_BATH* here (augmentation OFF).
export GANSU_DMET_LEVEL_SHIFT_DENOM_ONLY=1   # CORRECTNESS-CRITICAL epsilon un-shift
export GANSU_CCSD_CONV=1e-7
export GANSU_DMET_STEOM_BATH_DIAG=1          # report the bath-sufficiency gauge
# Memory-safety helpers from the completed canonical runs (harmless if unused):
export GANSU_DMET_CCSD_BNATIVE=1             # cluster CCSD from RI B (no dense ne^4 on device)
export GANSU_STEOM_OPERATOR_DEVICE_BALANCING=1
export GANSU_EA_RI_LADDER=1
export GANSU_EA_W_HOST=1

BASIS="-g 6-31g --eri_method ri -ag $AUX"
COMMON="-x $XYZ $BASIS --post_hf_method dmet_steom \
  --n_excited_states 5 --steom_n_root_cis 14 --dmet_cluster_solver canonical \
  --frozen_core auto --initial_guess sad --num_gpus 4"

OUT=/tmp/dox_expand_canon631g
mkdir -p $OUT

lowest() {  # $1 = log -> lowest STEOM root eV (field 3) + eta (field 4)
  awk '/STEOM excited-state energies/{f=1;next} /active-space health/{f=0}
       f && /^ +[0-9]+ +[0-9]/{print $3, $4; exit}' "$1"
}
verdict() { grep -oE "SUFFICIENT|MARGINAL|INSUFFICIENT" "$1" | tail -1; }
natoms()  { grep -oE "selected [0-9]+ atom" "$1" | grep -oE "[0-9]+" | head -1; }

# ---- [0] REFERENCE: full 18-atom manual anthraquinone (the target E_ref) ----
# Same basis/solver/augmentation-off as the ablation, so E_ref is the canonical
# 6-31G answer the expansion should converge toward.
DOX_MANUAL="{4,5,8,9,16,17,23,25,26,27,30,31,32,33,34,35,36,37}"
ref=$OUT/dox_ref_manual18.log
echo ">>> [0] reference: manual 18-atom anthraquinone / 6-31g / canonical -> $ref"
$GANSU $COMMON --dmet_fragments "$DOX_MANUAL" > "$ref" 2>&1 \
  || { echo "    reference run FAILED (see $ref) -- fix this before trusting A/B"; }
EREF=$(lowest "$ref" | awk '{print $1}')
echo "    E_ref (lowest, 6-31g canonical, 18-atom) = ${EREF:-<none>} eV"
echo "======================================================================"

# ---- [A/B] under-seeded auto fragment, expansion OFF vs ON ------------------
COVS=${1:-"0.70 0.80"}
for cov in $COVS; do
  for xp in 0 8; do
    tag="cov${cov}_expand${xp}"
    log=$OUT/dox_${tag}.log
    echo ">>> $tag -> $log"
    $GANSU $COMMON --dmet_steom_auto_fragment 1 --dmet_steom_auto_n_cis 14 \
      --dmet_steom_auto_coverage $cov --dmet_steom_auto_max_expand $xp \
      > "$log" 2>&1 || { echo "    (run failed; see $log)"; continue; }
  done
  a=$OUT/dox_cov${cov}_expand0.log ; b=$OUT/dox_cov${cov}_expand8.log
  echo "---- coverage $cov  (E_ref=${EREF:-?} eV) --------------------------"
  printf "  A (no expansion): atoms=%-3s gauge=%-12s  lowest= %s\n" \
         "$(natoms $a)" "$(verdict $a)" "$(lowest $a)"
  printf "  B (Phase B on)  : atoms=%-3s gauge=%-12s  lowest= %s\n" \
         "$(natoms $b)" "$(verdict $b)" "$(lowest $b)"
  echo "  -> POSITIVE iff B's gauge is better than A AND B's lowest is closer to E_ref, eta>=0.96"
  echo
done

# What to record for the paper if a coverage gives a clean win:
#   A: seed size / gauge / lowest DeltaE / eta        (max_expand 0)
#   B: final size / gauge / lowest DeltaE / eta        (max_expand 8)
#   atoms Phase B appended:  grep "\[DMET-STEOM Phase B\] expand += atom" <log>
#   per-round gauge:         grep "\[DMET-STEOM Phase B\] round"          <log>
# This is the constructive sibling of the naphthalene negative result and of
# Table 3 (paclitaxel augmentation): auto-correction that improves BOTH gauge
# and energy, in pure canonical DMET-STEOM.
