#!/usr/bin/env bash
# GANSU: GPU Accelerated Numerical Simulation Utility
# Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
# SPDX-License-Identifier: BSD-3-Clause
#
# Quantitative validation benchmark for the excitation-driven auto-fragment
# DMET-STEOM method: compare the AUTOMATIC fragment against the FULL (whole-
# molecule) STEOM-CCSD reference on small "localized chromophore in an inert
# scaffold" molecules where the full calculation is still tractable.
# ---------------------------------------------------------------------------
# For each molecule we run twice with identical settings:
#   (ref)  whole-molecule DMET-STEOM with NO --dmet_fragments  == full STEOM
#   (auto) --dmet_steom_auto_fragment 1                         == auto DMET-STEOM
# The analyzer (analyze_benchmark.py) then reports, per state, the reference
# energy, the auto energy, their difference, the bath gauge, and the STEOM
# active character eta -- giving MAE/RMSE/max deviation (reviewer point 1) and
# the gauge / eta / error correlation (reviewer point 6).
#
# The 2-alkanone series (same carbonyl chromophore, alkyl scaffold of growing
# length) is the core test: the auto fragment should stay the carbonyl region
# and reproduce the full STEOM lowest excitation as the chain -- and the full
# cost -- grows. 2-hexadecanone is included as the case where the FULL reference
# is expected to be intractable (auto-only; comment it out if it will not run).
#
# ADD aromatic pi->pi* cases for chromophore variety by appending to MOLS,
# e.g. butylbenzene / 2-butylnaphthalene / 4-methylacetophenone (provide the
# geometries under ../xyz/); the harness is molecule-agnostic.
#
# Runs on the remote GPU box (H200); cd ~/GANSU/build first.
set -uo pipefail

GANSU=./gansu
AUX=../auxiliary_basis/cc-pvdz-rifit.gbs
OUT=/tmp/bench_localized
mkdir -p "$OUT"

export GANSU_DMET_LEVEL_SHIFT_DENOM_ONLY=1   # CORRECTNESS-CRITICAL epsilon un-shift
export GANSU_CCSD_CONV=1e-7
export GANSU_STEOM_DENSE_DIAG=2              # dense geev: exact roots + complex-pair flag (small systems)
export GANSU_DMET_STEOM_BATH_DIAG=1          # report the bath gauge on the auto runs

# name | xyz (relative to build dir) | basis | n_excited_states | n_cis
# Molecules that already have data are commented out; the active block below
# solves the NEW cases that broaden the benchmark to 5-8 examples and add
# intermediate (non-sub-meV) chromophores, per reviewer request. Build each XYZ
# (RDKit ETKDG + a light opt, or Avogadro) and drop it under ../xyz/; SMILES are
# given. Whole-molecule STEOM stays tractable for <~26-atom molecules.
MOLS=(
  # --- already recorded (uncomment only to reproduce) ---
  # "butanone|../xyz/2-butanone.xyz|cc-pvdz|5|12"
  # "octanone|../xyz/2-octanone_opt.xyz|cc-pvdz|5|12"
  # "hexadecanone|../xyz/2-hexadecanone.xyz|cc-pvdz|5|12"      # auto-only (reduction)
  # "butylbenzene|../xyz/butylbenzene.xyz|cc-pvdz|5|12"
  # "butylnaphthalene|../xyz/2-butylnaphthalene.xyz|cc-pvdz|5|12"  # auto-only (reduction)
  # --- NEW (build XYZ first). SMILES:
  #   acetophenone   CC(=O)c1ccccc1    aryl ketone, localized  -> expect small error
  #   cyclohexanone  O=C1CCCCC1        aliphatic n->pi*         -> expect small error
  #   anisaldehyde   COc1ccc(C=O)cc1   push-pull OMe...CHO      -> INTERMEDIATE error
  #   styrene        C=Cc1ccccc1       ring+vinyl conjugation   -> INTERMEDIATE error
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
echo ">>> analysing ..."
python3 ../script/analyze_benchmark.py "$OUT" "${MOLS[@]}"
