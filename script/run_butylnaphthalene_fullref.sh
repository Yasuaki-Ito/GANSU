#!/usr/bin/env bash
# GANSU: GPU Accelerated Numerical Simulation Utility
# Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
# SPDX-License-Identifier: BSD-3-Clause
#
# M6: whole-molecule STEOM-CCSD reference for 2-butylnaphthalene (30 atoms,
# ~330 cc-pVDZ functions). The validation benchmark left this molecule auto-only
# because the full run OOM'd under the small-system settings (dense geev). Here
# we use the production "n_emb~490 memory-ceiling" set (host-staged EA
# intermediates, RI ladder, NO dense diag) that already carried doxorubicin
# (68 atoms), so the whole molecule is tractable. This gives the first
# whole-molecule ground truth in the "fragment < half the molecule" regime.
#
# full and auto are run with IDENTICAL settings so the deviation is a clean
# state-by-state comparison (Davidson STEOM in both; no dense complex-root flag,
# but butylnaphthalene's only near-defective root had |Im|=0.07 eV, negligible).
#
# Runs on the remote GPU box (H200); cd ~/GANSU/build first, then:
#   bash ../script/run_butylnaphthalene_fullref.sh
set -uo pipefail

GANSU=./gansu
AUX=../auxiliary_basis/cc-pvdz-rifit.gbs
OUT=/tmp/bnap_fullref
mkdir -p "$OUT"

export GANSU_DMET_LEVEL_SHIFT_DENOM_ONLY=1     # correctness-critical epsilon un-shift
export GANSU_CCSD_CONV=1e-7
export GANSU_DMET_STEOM_BATH_DIAG=1
# Full "n_emb~490 memory-ceiling" production set. NB: NO GANSU_STEOM_DENSE_DIAG.
export GANSU_EA_W_HOST=1                        # EA W intermediate staged to host
export GANSU_STEOM_OPERATOR_DEVICE_BALANCING=1  # spread STEOM operator build over GPUs
export GANSU_EA_RI_LADDER=1                     # EA via RI ladder (no dense nvir^4)
export GANSU_EA_WVVVO_HOST_ASM=1                # Wvvvo host assembly (the missing piece)
export GANSU_DMET_CCSD_BNATIVE=1                # cluster CCSD from RI B (no dense ne^4)

XYZ=../xyz/2-butylnaphthalene.xyz
BAS=cc-pvdz
NSTATE=5
NCIS=12
common="-x $XYZ -g $BAS --eri_method ri -ag $AUX --post_hf_method dmet_steom \
  --n_excited_states $NSTATE --steom_n_root_cis $NCIS \
  --frozen_core auto --initial_guess sad --num_gpus 4"

echo ">>> [2-butylnaphthalene] FULL reference (whole-molecule STEOM) -> $OUT/butylnaphthalene_full.log"
$GANSU $common > "$OUT/butylnaphthalene_full.log" 2>&1 \
  && echo "    full reference DONE" \
  || echo "    (full reference still failed -- inspect the log tail)"

echo ">>> [2-butylnaphthalene] AUTO fragment (same settings) -> $OUT/butylnaphthalene_auto.log"
$GANSU $common --dmet_steom_auto_fragment 1 --dmet_steom_auto_n_cis $NCIS \
  --dmet_steom_auto_json "$OUT/butylnaphthalene_states.json" \
  --dmet_steom_auto_xyz  "$OUT/butylnaphthalene_frag.xyz" \
  > "$OUT/butylnaphthalene_auto.log" 2>&1 \
  && echo "    auto DONE" || echo "    (auto failed)"

echo
echo ">>> STEOM excitation energies (full vs auto):"
for tag in full auto; do
  echo "--- $tag ---"
  awk '/STEOM excited-state energies/{p=1} p{print} /active-space health/{if(p){print;exit}}' "$OUT/butylnaphthalene_${tag}.log"
done
echo ">>> tail of full log (in case of failure):"; tail -5 "$OUT/butylnaphthalene_full.log"
