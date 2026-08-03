#!/usr/bin/env bash
# GANSU: GPU Accelerated Numerical Simulation Utility
# Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
# SPDX-License-Identifier: BSD-3-Clause
#
# INTEGRITY VERIFICATION of the showcase molecules with the CURRENT code.
#
# The naphthalene re-run revealed that the automatic SELECTION now includes
# hydrogens that were below the floor when the paper's showcase runs were made
# (naphthalene: old auto = 10 C -> 6.72 eV; current auto = 14 atoms = 10 C + 4 H
# -> 4.25 eV, matching the whole molecule). 6.7157 is NOT a bug -- the current
# code reproduces it for the manual 10-carbon cluster -- but the current AUTO
# picks a different, larger fragment. This script checks whether the other
# showcase molecules' reported AUTO fragments and energies still hold under the
# current code, using the exact recorded commands (SI). Compare each run's
# "selected N atoms / coverage / gauge / STEOM energies" against the PAPER
# values quoted in the comments.
#
# Runs on the remote GPU box (H200); cd ~/GANSU/build first, then:
#   bash ../script/verify_showcase.sh
set -uo pipefail

GANSU=./gansu
AUX=../auxiliary_basis/cc-pvdz-rifit.gbs
OUT=/tmp/showcase_verify
mkdir -p "$OUT"

export GANSU_DMET_LEVEL_SHIFT_DENOM_ONLY=1   # correctness-critical
export GANSU_CCSD_CONV=1e-7
export GANSU_DMET_STEOM_BATH_DIAG=1          # report the bath gauge

# --- memory kit REQUIRED at showcase cluster sizes (n_emb 390-480) ----------
# Established on dox group-0 (2026-07-30, see script/run_doxorubicin_group0_ccpvdz.sh
# and paper energies/dox_group0_steom.md). These are memory-layout switches only:
# every path is bit-exact, so the reported energies/selection are unaffected.
# Without them the 2026-07-30 showcase re-run died with dense n_emb^4 requests that
# match the cluster size exactly (dox 271.72 GB = 437^4, ubiquinone 177.73 = 393^4,
# paclitaxel 188.83 = 399^4) and Reichardt OOM'd later in the STEOM chain.
export GANSU_DMET_STEOM_RI_BLOCK=1           # MO-ERI blocks from cluster B, never n_emb^4
export GANSU_CCSD_RI_BNATIVE=1 GANSU_CCSD_RI_LADDER_TILE=1
export GANSU_CCSD_OCCI=1 GANSU_CCSD_VR_TILE=1
export GANSU_EA_RI_LADDER=1 GANSU_EA_W_HOST=1 GANSU_EA_WVVVO_HOST_ASM=1
export GANSU_STEOM_OPERATOR_DEVICE_BALANCING=1
export GANSU_DMET_STEOM_CLUSTER_GPU=1 GANSU_STEOM_BARH_GPU=3
export GANSU_STEOM_SHARE_BARH=1 GANSU_IP_SIGMA_GEMM=1
export GANSU_EOM_MAX_SUB_PER_ROOT=7
export OMP_NUM_THREADS=64

report () {  # $1 = log path, $2 = label
  local L="$1" tag="$2"
  echo "===================== $tag ====================="
  grep -hE "include_h=|selected [0-9]+ atom|coverage=|Phase B|n_emb=|bath (SUFFICIENT|INSUFFICIENT|MARGINAL)|region|budget-capped|RIGHT-SIZED|OVER-SELECTED|charge transfer|donor|acceptor" "$L" | head -20
  echo "-- STEOM energies + eta --"
  awk '/STEOM excited-state energies/{p=1} p{print} /active-space health/{if(p){print;exit}}' "$L"
  echo
}

# ---------------------------------------------------------------------------
# 1) DOXORUBICIN  (auto; DLPNO cluster solver, budget 700, NTO augmentation)
#    PAPER (SI S3.1): auto selects 19 atoms {6,8,9,10,16,17,23,24,25,26,27,30-37},
#      coverage 0.9314, gauge SUFFICIENT 0.0053,
#      energies 3.811 / 4.681 / 5.349 / 5.657 / 5.880 eV.
#      Hand-tuned 18-atom manual: 3.7640 ... => headline |3.811-3.764| = 0.05 eV.
# ---------------------------------------------------------------------------
echo ">>> [doxorubicin] AUTO -> $OUT/doxorubicin_auto.log"
GANSU_DMET_STEOM_DLPNO=2 GANSU_DMET_STEOM_NTO_BATH=0.01 GANSU_DMET_STEOM_NTO_BATH_OCC=0.5 \
$GANSU -x ../xyz/large_molecular/Doxorubicin.xyz -g cc-pvdz --eri_method ri \
  -ag $AUX --post_hf_method dmet_steom --frozen_core auto --num_gpus 4 \
  --initial_guess sad --n_excited_states 5 \
  --dmet_steom_auto_fragment 1 --dmet_steom_auto_n_cis 14 \
  --dmet_steom_auto_json "$OUT/dox_states.json" --dmet_steom_auto_xyz "$OUT/dox_frag.xyz" \
  > "$OUT/doxorubicin_auto.log" 2>&1 && echo "   done" || { echo "   FAILED (tail):"; tail -5 "$OUT/doxorubicin_auto.log"; }

# ---------------------------------------------------------------------------
# 2) REICHARDT'S DYE  (auto; CT detection)
#    PAPER (SI S2.4): 22 atoms incl phenolate O (atom 0) + pyridinium N (atom 1),
#      coverage 0.829, budget-capped, 3-region note; lowest is a CT state.
# ---------------------------------------------------------------------------
echo ">>> [reichardt] AUTO -> $OUT/reichardt_auto.log"
GANSU_DMET_STEOM_DLPNO=2 GANSU_DMET_STEOM_RI_BLOCK=1 \
$GANSU -x ../xyz/large_molecular/Reichardt_dye.xyz -g cc-pvdz \
  --eri_method ri -ag $AUX --post_hf_method dmet_steom \
  --frozen_core auto --num_gpus 4 --initial_guess sad --n_excited_states 5 \
  --dmet_steom_auto_fragment 1 --dmet_steom_auto_n_cis 12 \
  --dmet_steom_auto_json "$OUT/reich_states.json" --dmet_steom_auto_xyz "$OUT/reich_frag.xyz" \
  > "$OUT/reichardt_auto.log" 2>&1 && echo "   done" || { echo "   FAILED (tail):"; tail -5 "$OUT/reichardt_auto.log"; }

# ---------------------------------------------------------------------------
# 3) NAPHTHALENE  (auto, default AND max_expand 0)  -- confirm the selection shift
#    PAPER (SI S3.2 (ii)): max_expand 0 -> 10 C, coverage 0.972, MARGINAL 0.0232,
#      6.7157 / 7.8140 / 8.0019 eV.  Current default already shown: 14 atoms, 4.25.
# ---------------------------------------------------------------------------
echo ">>> [naphthalene] AUTO max_expand 0 -> $OUT/naphthalene_mx0.log"
GANSU_STEOM_DENSE_DIAG=2 GANSU_STEOM_DUMP_SPECTRUM=1 \
$GANSU -x ../xyz/Naphthalene.xyz -g cc-pvdz --eri_method ri -ag $AUX \
  --post_hf_method dmet_steom --frozen_core auto --num_gpus 4 --initial_guess sad \
  --n_excited_states 3 --dmet_steom_auto_fragment 1 --dmet_steom_auto_n_cis 12 \
  --dmet_steom_auto_max_expand 0 \
  > "$OUT/naphthalene_mx0.log" 2>&1 && echo "   done" || { echo "   FAILED"; tail -5 "$OUT/naphthalene_mx0.log"; }

# ---------------------------------------------------------------------------
# 4) UBIQUINONE-10  (auto selection + grouping; 6-31G)
#    PAPER: group 0 contains the benzoquinone + 5 tail carbons.
# 5) PACLITAXEL     (auto selection + grouping; 6-31G)
#    PAPER: grouping (sim 0.8) yields group 4 = benzamide {47,48,49,56,57,58,59,60,61}.
#    NOTE: confirm the 6-31G ERI setting (the manual energy runs used -ag $AUX with
#    --eri_method ri; the selection runs were 6-31G). Adjust -ag if your build
#    needs a 6-31G-matched aux, or drop -ag for conventional ERIs.
# ---------------------------------------------------------------------------
for m in ubiquinone10:ubiquinone10.xyz paclitaxel:paclitaxel.xyz; do
  name=${m%%:*}; xyz=${m##*:}
  echo ">>> [$name] AUTO (6-31G) -> $OUT/${name}_auto.log"
  $GANSU -x "../xyz/large_molecular/$xyz" -g 6-31g --eri_method ri -ag $AUX \
    --post_hf_method dmet_steom --frozen_core auto --num_gpus 4 --initial_guess sad \
    --n_excited_states 5 --dmet_steom_auto_fragment 1 --dmet_steom_auto_n_cis 14 \
    --dmet_steom_auto_json "$OUT/${name}_states.json" \
    --dmet_steom_auto_xyz  "$OUT/${name}_frag.xyz" \
    > "$OUT/${name}_auto.log" 2>&1 && echo "   selection done" || { echo "   FAILED (tail):"; tail -5 "$OUT/${name}_auto.log"; }
  if [ -f "$OUT/${name}_states.json" ]; then
    echo "   grouping (sim 0.8) ->"
    python3 ../script/dmet_steom_group_states.py --json "$OUT/${name}_states.json" \
      --xyz "../xyz/large_molecular/$xyz" --outdir "$OUT/${name}_groups" --sim 0.8 \
      > "$OUT/${name}_group.log" 2>&1 && tail -20 "$OUT/${name}_group.log" || echo "   (grouping failed)"
  fi
done

echo
echo "============================ SUMMARY ============================"
report "$OUT/doxorubicin_auto.log"   "DOXORUBICIN (paper: 19 atoms; 3.811/4.681/5.349/5.657/5.880 eV)"
report "$OUT/reichardt_auto.log"     "REICHARDT (paper: 22 atoms incl O0 + N1, cov 0.829, CT)"
report "$OUT/naphthalene_mx0.log"    "NAPHTHALENE max_expand0 (paper: 10 C, 6.7157/7.8140/8.0019 eV)"
report "$OUT/ubiquinone10_auto.log"  "UBIQUINONE-10 (paper: group0 = benzoquinone + tail)"
report "$OUT/paclitaxel_auto.log"    "PACLITAXEL (paper: group4 = benzamide {47..61})"
echo ">>> selected-fragment geometries in $OUT/*_frag.xyz ; grouping in $OUT/*_groups/"
