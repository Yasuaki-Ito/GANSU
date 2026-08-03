#!/usr/bin/env bash
# GANSU: GPU Accelerated Numerical Simulation Utility
# Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
# SPDX-License-Identifier: BSD-3-Clause
#
# DEFINITIVE showcase dataset, recomputed with the current code under ONE recipe.
#
# Why this replaces verify_showcase.sh
# -----------------------------------
# The earlier showcase numbers were produced by hand-assembled command lines that
# each carried a different set of ~13 environment variables. Reproducing them
# meant reproducing an undocumented env list, and getting it wrong changed the
# answer silently (a missing GANSU_DMET_LEVEL_SHIFT_DENOM_ONLY shifts every
# excitation energy up by ~4 eV; a missing GANSU_STEOM_DENSE_DIAG handed the
# diagonalisation to a Davidson that misses roots). Both of those are now decided
# by the code: the tensor-layout switches come from the cluster size and the free
# memory (dmet_memory_policy), the dense/Davidson choice comes from a memory probe,
# and the denominator-only level shift is the default. So every run below is a
# plain command line, and the log records the layout it used.
#
# The one recipe
# --------------
#   --dmet_cluster_solver dlpno    cluster ground state (production mode)
#   bt-polish                      DEFAULT = polish to convergence. NOT capped:
#                                  the capped (3-iteration) polish is a speed
#                                  setting, and on the auto fragment it moves the
#                                  lowest root by 0.18 eV, so results quoted from
#                                  it are tied to the cap. Full polish costs ~3x.
#   --dmet_nto_bath 0.01 --dmet_nto_bath_occ 0.5     NTO-bath augmentation
#   --frozen_core auto --initial_guess sad --num_gpus 4
#   n_cis: per molecule (how many CIS states drive the selection) — the only
#          knob that legitimately differs between systems.
#
# Runs on the remote GPU box (H200x4); cd ~/GANSU/build first, then:
#   bash ../script/showcase_final.sh
# Expect several hours: full bt-polish dominates (~40 min per large cluster).
set -uo pipefail

GANSU=./gansu
AUX=../auxiliary_basis/cc-pvdz-rifit.gbs
OUT=/tmp/showcase_final
mkdir -p "$OUT"

# Shared, molecule-independent part of the recipe.
RECIPE="--post_hf_method dmet_steom --dmet_cluster_solver dlpno \
  --dmet_nto_bath 0.01 --dmet_nto_bath_occ 0.5 \
  --frozen_core auto --initial_guess sad --num_gpus 4"

run () {  # $1 = tag, rest = molecule-specific flags
  local tag="$1"; shift
  local log="$OUT/${tag}.log"
  if grep -aq "STEOM excited-state energies" "$log" 2>/dev/null; then
    echo ">>> $tag: already complete, skipping"; return
  fi
  echo ">>> $tag -> $log"
  $GANSU $RECIPE "$@" > "$log" 2>&1 \
    && echo "    done" \
    || { echo "    FAILED (tail):"; tail -4 "$log"; }
}

# --- doxorubicin: the headline auto-vs-manual comparison, same recipe both ways.
run dox_auto -x ../xyz/large_molecular/Doxorubicin.xyz -g cc-pvdz --eri_method ri -ag $AUX \
    --n_excited_states 5 --dmet_steom_auto_fragment 1 --dmet_steom_auto_n_cis 14 \
    --dmet_steom_auto_json "$OUT/dox_states.json" --dmet_steom_auto_xyz "$OUT/dox_frag.xyz"

run dox_manual -x ../xyz/large_molecular/Doxorubicin.xyz -g cc-pvdz --eri_method ri -ag $AUX \
    --n_excited_states 5 --steom_n_root_cis 14 \
    --dmet_fragments "{4,5,8,9,16,17,23,25,26,27,30,31,32,33,34,35,36,37}"

# --- Reichardt's dye: charge-transfer selection (both ends in one fragment).
run reichardt -x ../xyz/large_molecular/Reichardt_dye.xyz -g cc-pvdz --eri_method ri -ag $AUX \
    --n_excited_states 5 --dmet_steom_auto_fragment 1 --dmet_steom_auto_n_cis 12 \
    --dmet_steom_auto_json "$OUT/reich_states.json" --dmet_steom_auto_xyz "$OUT/reich_frag.xyz"

# --- naphthalene: delocalised negative control. n_cis is left at the DEFAULT
#     (n_excited+4 = 7): that is what selects the 10-carbon fragment. The old
#     script pinned n_cis 12, which selects 14 atoms (10 C + 4 H) — a different
#     fragment, and the source of the apparent "the code changed" discrepancy.
run naphthalene -x ../xyz/Naphthalene.xyz -g cc-pvdz --eri_method ri -ag $AUX \
    --n_excited_states 3 --dmet_steom_auto_fragment 1 --dmet_steom_auto_max_expand 0

# --- ubiquinone-10 / paclitaxel: selection + per-state grouping (6-31G).
for m in ubiquinone10:ubiquinone10.xyz paclitaxel:paclitaxel.xyz; do
  name=${m%%:*}; xyz=${m##*:}
  run "$name" -x "../xyz/large_molecular/$xyz" -g 6-31g --eri_method ri -ag $AUX \
      --n_excited_states 5 --dmet_steom_auto_fragment 1 --dmet_steom_auto_n_cis 14 \
      --dmet_steom_auto_json "$OUT/${name}_states.json" \
      --dmet_steom_auto_xyz  "$OUT/${name}_frag.xyz"
  [ -f "$OUT/${name}_states.json" ] && \
    python3 ../script/dmet_steom_group_states.py --json "$OUT/${name}_states.json" \
      --xyz "../xyz/large_molecular/$xyz" --outdir "$OUT/${name}_groups" --sim 0.8 \
      > "$OUT/${name}_group.log" 2>&1
done

echo
echo "======================= FINAL SHOWCASE TABLE ======================="
for f in "$OUT"/*.log; do
  case "$f" in *_group.log) continue;; esac
  echo "===== $(basename "$f" .log) ====="
  grep -haE "selected [0-9]+ atom|coverage=|bath (SUFFICIENT|INSUFFICIENT|MARGINAL)|memory policy\] auto|STEOM diag\]" "$f" | head -6
  awk '/STEOM excited-state energies/{p=1} p{print} /active-space health/{if(p){print;exit}}' "$f"
  echo
done
echo ">>> logs + fragment xyz + grouping in $OUT/"
