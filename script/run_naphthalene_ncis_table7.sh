#!/usr/bin/env bash
# GANSU: GPU Accelerated Numerical Simulation Utility
# Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
# SPDX-License-Identifier: BSD-3-Clause
#
# Naphthalene: make Table 7 (ADC(2)) consistent with the n_cis=12 selection used
# throughout the paper, and confirm the n_cis-sensitivity of the selection with
# the CURRENT code.
#
#   (A) whole-molecule DMET-ADC(2)          -> Table 7 reference column
#   (B) auto DMET-ADC(2) at n_cis=12 (14 at)-> Table 7 fragment column (replaces
#                                              the old n_cis=7 / 10-carbon 5.6755)
#   (C) auto STEOM at n_cis=7 (10 carbons)  -> confirms the low-n_cis under-capture
#                                              (expected ~6.72 eV, gauge MARGINAL)
#
# All ADC(2) runs pin --adc2_solver schur_omega (the exact omega-iterated Schur;
# naphthalene's singles counts stay < 10^4 for both the whole molecule and the
# 14-atom cluster, so the exact solver fits and the two columns are uniform --
# avoiding the omega=0 solver-switch that had to be fixed for Reichardt).
#
# Small system; the tensor-layout memory policy is auto in v2026.8.3, so no hand
# env is needed beyond correctness + the STEOM dense diag for (C).
#
#   cd ~/GANSU/build && bash ../script/run_naphthalene_ncis_table7.sh
set -uo pipefail

GANSU=./gansu
AUX=../auxiliary_basis/cc-pvdz-rifit.gbs
XYZ=../xyz/Naphthalene.xyz
OUT=/tmp/nap_ncis; mkdir -p "$OUT"

export GANSU_DMET_LEVEL_SHIFT_DENOM_ONLY=1
export GANSU_CCSD_CONV=1e-7
export GANSU_DMET_STEOM_BATH_DIAG=1

HEAD="-x $XYZ -g cc-pvdz --eri_method ri -ag $AUX --frozen_core auto --num_gpus 4 --initial_guess sad --n_excited_states 3"

echo ">>> (A) whole-molecule DMET-ADC(2) (schur_omega) -> $OUT/nap_full_adc2.log"
$GANSU $HEAD --post_hf_method dmet_steom --dmet_excited_method adc2 \
  --adc2_solver schur_omega \
  > "$OUT/nap_full_adc2.log" 2>&1 && echo "   done" || { echo "   FAILED"; tail -4 "$OUT/nap_full_adc2.log"; }

echo ">>> (B) auto DMET-ADC(2), n_cis=12 (expect 14 atoms) -> $OUT/nap_auto12_adc2.log"
$GANSU $HEAD --post_hf_method dmet_steom --dmet_excited_method adc2 \
  --adc2_solver schur_omega \
  --dmet_steom_auto_fragment 1 --dmet_steom_auto_n_cis 12 \
  > "$OUT/nap_auto12_adc2.log" 2>&1 && echo "   done" || { echo "   FAILED"; tail -4 "$OUT/nap_auto12_adc2.log"; }

echo ">>> (C) auto STEOM, n_cis=7 (expect 10 carbons, ~6.72) -> $OUT/nap_auto7_steom.log"
GANSU_STEOM_DENSE_DIAG=2 \
$GANSU $HEAD --post_hf_method dmet_steom \
  --dmet_steom_auto_fragment 1 --dmet_steom_auto_n_cis 7 \
  > "$OUT/nap_auto7_steom.log" 2>&1 && echo "   done" || { echo "   FAILED"; tail -4 "$OUT/nap_auto7_steom.log"; }

echo
echo "============================ SUMMARY ============================"
for tag in "nap_full_adc2:(A) whole-molecule ADC(2)" \
           "nap_auto12_adc2:(B) auto ADC(2) n_cis=12" \
           "nap_auto7_steom:(C) auto STEOM n_cis=7"; do
  f=${tag%%:*}; label=${tag##*:}; L="$OUT/$f.log"
  echo "===================== $label ====================="
  grep -hE "selected [0-9]+ atom|coverage=|bath (SUFFICIENT|MARGINAL|INSUFFICIENT)" "$L" 2>/dev/null | head -2
  awk '/(STEOM|ADC\(2\)) excited-state energies/{p=1} p{print} /active-space health/{if(p){print;exit}}' "$L" 2>/dev/null | head -8
  echo
done
echo ">>> expected: (B) auto-14 ~ (A) whole molecule to sub-meV; (C) ~6.72/7.82/8.00, MARGINAL"