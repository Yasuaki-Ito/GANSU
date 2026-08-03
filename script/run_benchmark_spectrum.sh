#!/usr/bin/env bash
# GANSU: GPU Accelerated Numerical Simulation Utility
# Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
# SPDX-License-Identifier: BSD-3-Clause
#
# Re-run of run_benchmark_localized.sh with the STEOM G-spectrum diagnostic
# dump enabled (GANSU_STEOM_DUMP_SPECTRUM=1). This is behaviour-inert: the
# excitation energies are IDENTICAL to the recorded benchmark (deterministic
# dense geev). The ONLY new information is, per root, the real AND imaginary
# part of the STEOM G eigenvalue, so we can see exactly WHICH reported state is
# a near-defective complex-conjugate pair (|Im|>0) and how large |Im| is.
#
# Purpose: the recorded *_full.log / *_auto.log report only a COUNT of complex
# roots and max|Im| (e.g. acetophenone 1/5, max|Im|=1.999 eV) but not which
# root. This run resolves that, for both the FULL reference and the AUTO
# fragment, on every benchmark molecule.
#
# Runs on the remote GPU box (H200); cd ~/GANSU/build first, then:
#   bash ../script/run_benchmark_spectrum.sh
set -uo pipefail

GANSU=./gansu
AUX=../auxiliary_basis/cc-pvdz-rifit.gbs
OUT=/tmp/bench_spectrum
mkdir -p "$OUT"

export GANSU_DMET_LEVEL_SHIFT_DENOM_ONLY=1   # CORRECTNESS-CRITICAL epsilon un-shift
export GANSU_CCSD_CONV=1e-7
export GANSU_STEOM_DENSE_DIAG=2              # dense geev: exact roots + complex-pair flag (small systems)
export GANSU_DMET_STEOM_BATH_DIAG=1          # report the bath gauge on the auto runs
export GANSU_STEOM_DUMP_SPECTRUM=1           # NEW: dump per-root Re/Im of the STEOM G spectrum

# name | xyz (relative to build dir) | basis | n_excited_states | n_cis
# All benchmark molecules enabled (identical settings to run_benchmark_localized.sh).
MOLS=(
  "butanone|../xyz/2-butanone.xyz|cc-pvdz|5|12"
  "octanone|../xyz/2-octanone_opt.xyz|cc-pvdz|5|12"
  "hexadecanone|../xyz/2-hexadecanone.xyz|cc-pvdz|5|12"        # auto-only (full too large)
  "butylbenzene|../xyz/butylbenzene.xyz|cc-pvdz|5|12"
  "butylnaphthalene|../xyz/2-butylnaphthalene.xyz|cc-pvdz|5|12" # auto-only (reduction)
  "acetophenone|../xyz/acetophenone.xyz|cc-pvdz|5|12"
  "cyclohexanone|../xyz/cyclohexanone.xyz|cc-pvdz|5|12"
  "anisaldehyde|../xyz/anisaldehyde.xyz|cc-pvdz|5|12"
  "styrene|../xyz/styrene.xyz|cc-pvdz|5|12"
)

for entry in "${MOLS[@]}"; do
  IFS='|' read -r name xyz basis nstate ncis <<< "$entry"
  common="-x $xyz -g $basis --eri_method ri -ag $AUX --post_hf_method dmet_steom \
    --n_excited_states $nstate --steom_n_root_cis $ncis \
    --frozen_core auto --initial_guess sad --num_gpus 4"

  echo ">>> [$name] FULL reference (whole-molecule STEOM) -> $OUT/${name}_full.log"
  $GANSU $common > "$OUT/${name}_full.log" 2>&1 \
    || echo "    (full reference failed/too large for $name -- auto-only)"

  echo ">>> [$name] AUTO fragment DMET-STEOM -> $OUT/${name}_auto.log"
  $GANSU $common --dmet_steom_auto_fragment 1 --dmet_steom_auto_n_cis $ncis \
    --dmet_steom_auto_json "$OUT/${name}_states.json" \
    --dmet_steom_auto_xyz  "$OUT/${name}_frag.xyz" \
    > "$OUT/${name}_auto.log" 2>&1 \
    || echo "    (auto run failed for $name)"
done

echo
echo ">>> per-root complex spectrum summary (full + auto) ..."
for f in "$OUT"/*_full.log "$OUT"/*_auto.log; do
  [ -f "$f" ] || continue
  echo "===== $(basename "$f") ====="
  # print the spectrum dump table if present
  awk '/STEOM spectrum dump/{p=1} p{print} /^$/{if(p)p=0}' "$f" | head -12
  grep -h "complex-root recovery" "$f" | sed -E 's/.*recovery\] //'
done

echo
echo ">>> analysing (should reproduce the recorded MAE/RMSE/max exactly) ..."
python3 ../script/analyze_benchmark.py "$OUT" "${MOLS[@]}"
