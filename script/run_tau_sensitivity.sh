#!/usr/bin/env bash
# GANSU: GPU Accelerated Numerical Simulation Utility
# Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
# SPDX-License-Identifier: BSD-3-Clause
#
# Sensitivity of the NTO bath augmentation to its thresholds tau_vir, tau_occ
# (reviewer point: justify the default tau_vir=0.01, tau_occ=0.5).
# ---------------------------------------------------------------------------
# The fragment is FIXED (the hand-picked chromophore) so only the augmentation
# thresholds change; we record the cluster size n_emb, how many NTOs are
# appended, the lowest excitation, the complex-pair count, and eta, as tau is
# swept around the default. This shows how the result behaves near the default
# and whether that default sits on a stable plateau.
#
# paclitaxel benzamide is the informative case (its bare bath is INSUFFICIENT,
# so augmentation actually matters); doxorubicin is included because the
# reviewer named it (its bare bath is already SUFFICIENT, so the result should
# be nearly tau-independent -- itself a useful robustness statement).
#
# canonical solver is mandatory: NTO-bath + DLPNO is incompatible (NaN).
# Runs on the remote GPU box (H200); cd ~/GANSU/build first.
set -uo pipefail

GANSU=./gansu
AUX=../auxiliary_basis/cc-pvdz-rifit.gbs
OUT=/tmp/tau_sensitivity
mkdir -p "$OUT"

export GANSU_DMET_LEVEL_SHIFT_DENOM_ONLY=1
export GANSU_CCSD_CONV=1e-7
export GANSU_STEOM_DENSE_DIAG=2               # complex-pair counts
export GANSU_DMET_STEOM_BATH_DIAG=1

# name | xyz | fragment (0-based) | basis | n_states | n_cis
SYS=(
  "paclitaxel_benzamide|../xyz/large_molecular/paclitaxel.xyz|{47,48,49,56,57,58,59,60,61}|6-31g|5|14"
  "doxorubicin_anthraq|../xyz/large_molecular/Doxorubicin.xyz|{4,5,8,9,16,17,23,25,26,27,30,31,32,33,34,35,36,37}|6-31g|5|14"
)

# (tau_vir, tau_occ) grid: sweep vir at occ=0.5, then occ at vir=0.01. Default is 0.01/0.5.
TAUS=( "0.005 0.5" "0.01 0.5" "0.02 0.5" "0.05 0.5" "0.01 0.3" "0.01 0.7" )

for entry in "${SYS[@]}"; do
  IFS='|' read -r name xyz frag basis nstate ncis <<< "$entry"
  for tau in "${TAUS[@]}"; do
    read -r tvir tocc <<< "$tau"
    tag="${name}_v${tvir}_o${tocc}"; log="$OUT/${tag}.log"
    echo ">>> $tag -> $log"
    GANSU_DMET_STEOM_NTO_BATH=$tvir GANSU_DMET_STEOM_NTO_BATH_OCC=$tocc \
    $GANSU -x $xyz -g $basis --eri_method ri -ag $AUX --post_hf_method dmet_steom \
      --dmet_fragments "$frag" --dmet_cluster_solver canonical \
      --n_excited_states $nstate --steom_n_root_cis $ncis \
      --initial_guess sad --num_gpus 4 > "$log" 2>&1 \
      || echo "    (failed; see $log)"
  done
done

echo
echo "=== summary: tau -> n_emb, appended NTOs, gauge, lowest dE, complex pairs ==="
for entry in "${SYS[@]}"; do
  IFS='|' read -r name xyz frag basis nstate ncis <<< "$entry"
  echo "-- $name --"
  for tau in "${TAUS[@]}"; do
    read -r tvir tocc <<< "$tau"; log="$OUT/${name}_v${tvir}_o${tocc}.log"
    nemb=$(grep -oE "n_emb=[0-9]+" "$log" 2>/dev/null | tail -1)
    aug=$(grep -oE "added [0-9]+ virtual \+ [0-9]+ occupied" "$log" 2>/dev/null | tail -1)
    verdict=$(grep -oE "INSUFFICIENT|MARGINAL|SUFFICIENT" "$log" 2>/dev/null | head -1)
    low=$(awk '/STEOM excited-state energies/{f=1;next} /active-space health/{f=0} f&&/^ +[0-9]+ +[0-9]/{print $3; exit}' "$log" 2>/dev/null)
    printf "   vir=%-5s occ=%-3s : %-10s  %-28s  gauge=%-12s  low=%s eV\n" \
           "$tvir" "$tocc" "$nemb" "${aug:-no augmentation}" "$verdict" "${low:-NA}"
  done
done
# Report against the default (vir=0.01, occ=0.5): a flat lowest-dE across the
# grid is the robustness argument for the chosen default.
