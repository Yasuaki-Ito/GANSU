#!/usr/bin/env bash
# GANSU: GPU Accelerated Numerical Simulation Utility
# Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
# SPDX-License-Identifier: BSD-3-Clause
#
# Complete the excitation-energy table with the CHEAPER ADC(2) cluster solver
# for the systems where STEOM-CCSD was not run to completion (reviewer minor
# point: do not leave "not solved" -- use ADC(2) to fill the table).
# ---------------------------------------------------------------------------
#   ubiquinone-10 : manual benzoquinone (10) and driver group-0 (15) fragments,
#                   6-31G -- localized quinone, ADC(2) is meaningful here.
#   Reichardt     : automatic 22-atom fragment, cc-pVDZ. NB: this is a charge-
#                   transfer state; ADC(2) (like CIS/STEOM) overestimates CT
#                   energies, so report the number WITH that caveat -- the point
#                   is that the table entry exists and the fragment solves, not
#                   that the CT energy is quantitative.
#
# Runs on the remote GPU box (H200); cd ~/GANSU/build first.
set -uo pipefail

GANSU=./gansu
AUX=../auxiliary_basis/cc-pvdz-rifit.gbs
OUT=/tmp/adc2_energytable
mkdir -p "$OUT"

export GANSU_DMET_LEVEL_SHIFT_DENOM_ONLY=1   # ADC(2) fed physical (un-shifted) epsilon
export GANSU_CCSD_CONV=1e-7
export GANSU_DMET_STEOM_BATH_DIAG=1

UBQ=../xyz/large_molecular/ubiquinone10.xyz
REI=../xyz/large_molecular/Reichardt_dye.xyz
UBQ_MANUAL="{0,1,2,3,46,50,51,53,54,55}"                       # benzoquinone, 10 atoms
UBQ_GROUP0="{0,1,2,3,24,26,32,34,43,46,50,51,53,54,55}"        # driver group-0, 15 atoms

echo ">>> ubiquinone-10 benzoquinone (manual 10) / 6-31g / DMET-ADC(2)"
$GANSU -x $UBQ -g 6-31g --eri_method ri -ag $AUX --post_hf_method dmet_steom \
  --dmet_excited_method adc2 --dmet_fragments "$UBQ_MANUAL" \
  --n_excited_states 5 --frozen_core auto --initial_guess sad --num_gpus 4 \
  > "$OUT/ubq_manual_adc2.log" 2>&1 || echo "   (failed)"

echo ">>> ubiquinone-10 group-0 (driver 15) / 6-31g / DMET-ADC(2)"
$GANSU -x $UBQ -g 6-31g --eri_method ri -ag $AUX --post_hf_method dmet_steom \
  --dmet_excited_method adc2 --dmet_fragments "$UBQ_GROUP0" \
  --n_excited_states 5 --frozen_core auto --initial_guess sad --num_gpus 4 \
  > "$OUT/ubq_group0_adc2.log" 2>&1 || echo "   (failed)"

echo ">>> Reichardt automatic fragment / cc-pVDZ / DMET-ADC(2)  (CT: caveat)"
$GANSU -x $REI -g cc-pvdz --eri_method ri -ag $AUX --post_hf_method dmet_steom \
  --dmet_excited_method adc2 --dmet_steom_auto_fragment 1 --dmet_steom_auto_n_cis 12 \
  --n_excited_states 5 --frozen_core auto --initial_guess sad --num_gpus 4 \
  > "$OUT/reichardt_auto_adc2.log" 2>&1 || echo "   (failed)"

echo
echo "=== ADC(2) excitation energies (eV) ==="
for L in ubq_manual_adc2 ubq_group0_adc2 reichardt_auto_adc2; do
  echo "-- $L --"
  awk '/excited-state energies/{f=1;next} /Device Memory|END:/{f=0}
       f&&/^ +[0-9]+ +[0-9]/{print "   k="$1"  "$3" eV"}' "$OUT/$L.log" 2>/dev/null | head -6
done
