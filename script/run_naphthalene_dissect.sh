#!/usr/bin/env bash
# GANSU: GPU Accelerated Numerical Simulation Utility
# Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
# SPDX-License-Identifier: BSD-3-Clause
#
# M4: dissect WHY the naphthalene auto fragment (all ten carbons) blue-shifts by
# +2.46 eV under STEOM-CCSD but only +0.49 eV under ADC(2) on the SAME cluster.
# The suspicion is that the excess is STEOM-specific: STEOM builds its active
# space from truncated IP/EA-EOM roots inside the cluster, and a delocalized
# pi* manifold is exactly what that truncation cannot hold. ADC(2) uses the full
# cluster virtual space and has no such active-space step.
#
# This run records, for the whole molecule AND the auto fragment:
#   - the STEOM active space (n_act_occ, n_act_vir) and total_dim
#   - the IP/EA-EOM active-root counts that seed it
#   - eta per root (active character)
#   - the full G spectrum with |Im| (GANSU_STEOM_DUMP_SPECTRUM)
# Compare against the existing ADC(2) logs (energies/nap_dmet_adc2.log,
# nap_frag_adc2.log) to attribute the 2.46 vs 0.49 eV gap.
#
# Small system (10 C, cc-pVDZ ~180 functions) -> dense geev is fine; the
# production memory set is NOT needed here.
#
# Runs on the remote GPU box (H200); cd ~/GANSU/build first, then:
#   bash ../script/run_naphthalene_dissect.sh
set -uo pipefail

GANSU=./gansu
AUX=../auxiliary_basis/cc-pvdz-rifit.gbs
OUT=/tmp/nap_dissect
mkdir -p "$OUT"

export GANSU_DMET_LEVEL_SHIFT_DENOM_ONLY=1     # correctness-critical epsilon un-shift
export GANSU_CCSD_CONV=1e-7
export GANSU_STEOM_DENSE_DIAG=2                 # exact roots + complex-pair flag (small)
export GANSU_DMET_STEOM_BATH_DIAG=1
export GANSU_STEOM_DUMP_SPECTRUM=1             # per-root Re/Im of the STEOM G

XYZ=../xyz/Naphthalene.xyz     # note the capital N (case-sensitive on Linux)
BAS=cc-pvdz
NSTATE=5
NCIS=12
common="-x $XYZ -g $BAS --eri_method ri -ag $AUX --post_hf_method dmet_steom \
  --n_excited_states $NSTATE --steom_n_root_cis $NCIS \
  --frozen_core auto --initial_guess sad --num_gpus 4"

echo ">>> [naphthalene] FULL (whole molecule) STEOM -> $OUT/naphthalene_full.log"
$GANSU $common > "$OUT/naphthalene_full.log" 2>&1 && echo "    full DONE" || echo "    (full failed)"

echo ">>> [naphthalene] AUTO fragment (all ten carbons) STEOM -> $OUT/naphthalene_auto.log"
$GANSU $common --dmet_steom_auto_fragment 1 --dmet_steom_auto_n_cis $NCIS \
  --dmet_steom_auto_json "$OUT/naphthalene_states.json" \
  --dmet_steom_auto_xyz  "$OUT/naphthalene_frag.xyz" \
  > "$OUT/naphthalene_auto.log" 2>&1 && echo "    auto DONE" || echo "    (auto failed)"

echo
echo ">>> KEY DIAGNOSTICS (paste these back):"
for tag in full auto; do
  L="$OUT/naphthalene_${tag}.log"
  echo "===================== naphthalene $tag ====================="
  echo "-- STEOM / IP / EA active-space headers --"
  grep -nE "STEOM-CCSD ----|IP-EOM-CCSD ----|EA-EOM-CCSD ----" "$L"
  echo "-- active-root counts routed to STEOM --"
  grep -nE "Active root assignment|Auxiliary (IP|EA) roots" "$L"
  echo "-- STEOM energies + eta --"
  awk '/STEOM excited-state energies/{p=1} p{print} /active-space health/{if(p){print;exit}}' "$L"
  echo "-- G spectrum dump (lowest roots, Re/Im) --"
  awk '/STEOM spectrum dump/{p=1} p{print} /^ *0007/{if(p)exit}' "$L" | head -12
  echo
done
echo ">>> For the ADC(2) side, compare energies/nap_dmet_adc2.log and nap_frag_adc2.log"
echo "    (whole-molecule 5.1897 eV / auto 5.6755 eV => +0.49 eV, vs STEOM +2.46 eV)."
