#!/usr/bin/env bash
# GANSU: GPU Accelerated Numerical Simulation Utility
# Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
# SPDX-License-Identifier: BSD-3-Clause
#
# M8: external-program validation of the GANSU STEOM-CCSD and ADC(2)
# implementations on small molecules, to be tabulated in the SI against ORCA
# (STEOM-CCSD) and, for ADC(2), against a program that implements it
# (PySCF+adcc / Turbomole / Q-Chem; ORCA has no ADC(2)).
#
# Whole-molecule (no DMET fragmentation), cc-pVDZ/RI, frozen core, 5 states.
# STEOM uses the exact dense geev; ADC(2) uses the exact omega-iterated Schur.
# These are code-vs-code checks, so the basis only needs to match ORCA's.
#
#   cd ~/GANSU/build && bash ../script/run_orca_validation_gansu.sh
set -uo pipefail

GANSU=./gansu
AUX=../auxiliary_basis/cc-pvdz-rifit.gbs
OUT=/tmp/orca_val; mkdir -p "$OUT"

export GANSU_DMET_LEVEL_SHIFT_DENOM_ONLY=1
export GANSU_CCSD_CONV=1e-7

HEAD="-g cc-pvdz --eri_method ri -ag $AUX --frozen_core auto --num_gpus 4 --initial_guess sad --n_excited_states 5"

for m in Formaldehyde C2H4 H2O; do
  XYZ=../xyz/$m.xyz
  echo "=============================================================="
  echo ">>> $m  STEOM-CCSD (whole molecule, dense geev)"
  GANSU_STEOM_DENSE_DIAG=2 \
  $GANSU -x $XYZ $HEAD --post_hf_method dmet_steom --steom_n_root_cis 10 \
    > "$OUT/${m}_steom.log" 2>&1 \
    && awk '/STEOM excited-state energies/{p=1} p && /^   [0-4] /{print} /active-space health/{if(p)exit}' "$OUT/${m}_steom.log" \
    || { echo "   STEOM FAILED"; tail -3 "$OUT/${m}_steom.log"; }

  echo ">>> $m  ADC(2) (whole molecule, exact schur_omega)"
  $GANSU -x $XYZ $HEAD --post_hf_method dmet_steom --dmet_excited_method adc2 \
    --adc2_solver schur_omega \
    > "$OUT/${m}_adc2.log" 2>&1 \
    && awk '/(STEOM|ADC\(2\)) excited-state energies/{p=1} p && /^ *[0-4] /{print} /active-space health/{if(p)exit}' "$OUT/${m}_adc2.log" \
    || { echo "   ADC(2) FAILED"; tail -3 "$OUT/${m}_adc2.log"; }
done
echo
echo ">>> GANSU side done. Logs in $OUT. Now run the matching ORCA STEOM-CCSD"
echo "    (template: script/orca_val_template.inp) and paste both sets of energies."
