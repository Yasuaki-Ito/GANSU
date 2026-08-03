#!/usr/bin/env bash
# GANSU: GPU Accelerated Numerical Simulation Utility
# Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
# SPDX-License-Identifier: BSD-3-Clause
#
# Systematic regeneration of the showcase / SI ENERGIES with the CURRENT code,
# so every number in Tables 2, 4, 5, SI S7 and the Reichardt CT section is a
# current-code value.
#
# Resumable: a run whose log already contains the energies block is skipped, and
# every run keeps a checkpoint. Safe to Ctrl-C and re-launch.
#
#   cd ~/GANSU/build && bash ../script/regen_showcase_energies.sh
#
# ---------------------------------------------------------------------------
# THREE THINGS THIS SCRIPT PINS ON PURPOSE (read before trusting the output)
# ---------------------------------------------------------------------------
# 1. Fragment lists are passed through "$@", never through eval. A previous
#    revision ran the command through `eval "$KIT ... $*"`, and bash then brace-
#    expanded the fragment set:
#        --dmet_fragments "{4,5,8,9,...}"  ->  --dmet_fragments 4 5 8 9 ...
#    which silently broke every manual-fragment run. Do not reintroduce eval.
#
# 2. bt-polish depth is EXPLICIT (POLISH below). It is not a free choice: the
#    published doxorubicin numbers were produced with the 3-iteration cap, and
#    polishing to convergence moves them (measured 2026-08-03, same code, one
#    variable changed):
#        manual-18 : cap3 3.7640 eV   full 3.8037 eV   (+0.040)
#        auto-19   : cap3 3.8109 eV   full 3.6351 eV   (-0.176, spectrum reorders)
#    so the headline |auto - manual| agreement is 0.05 eV under cap3 and 0.17 eV
#    under full polish. Default here = cap3, which reproduces the published
#    values to 8 digits (verified 2026-08-03). Set POLISH="" for full polish if
#    the manuscript is to quote converged-polish numbers instead.
#
# 3. The ADC(2) solver is pinned per run. `auto` switches at singles = 10^4
#    (<= exact omega-iterated Schur, > matrix-free Davidson at omega = 0, which
#    carries a ~0.005-0.02 Ha bias), so leaving it on auto makes the solver a
#    function of cluster size — the existing Reichardt CT logs already mix the
#    two (donor/acceptor 7,875 singles = schur_omega; both/auto 21,420 =
#    schur_davidson). Pin it so a table is internally uniform.
#
# Note on the environment: the tensor-layout switches this used to export by
# hand (RI_BLOCK, CCSD tiling, EA host staging, ...) are now chosen by the code
# from the cluster size and free memory, and the denominator-only level shift is
# the default, so they are deliberately NOT set here. Letting the policy decide
# also keeps small clusters (e.g. the paclitaxel benzamide, n_emb = 184) on the
# direct layout they were originally computed with.
set -uo pipefail

GANSU=./gansu
AUX=../auxiliary_basis/cc-pvdz-rifit.gbs
OUT=/tmp/regen; mkdir -p "$OUT"

POLISH="--dlpno_bt_polish 3"     # see (2) above; "" = polish to convergence
ADC2_SOLVER="schur_davidson"     # see (3); matches the existing large-cluster logs

# Correctness / reporting only. DENSE_DIAG=2 forces the dense geev for the final
# STEOM diagonalisation: the auto rule now also picks dense whenever it fits, but
# forcing it guarantees the converged answer rather than an iterative one. If a
# run dies at the diagonalisation with an OOM, drop it FOR THAT MOLECULE and
# treat the resulting Davidson roots as provisional (the log says so too).
COMMON_ENV="GANSU_CCSD_CONV=1e-7 GANSU_DMET_STEOM_BATH_DIAG=1"
STEOM_DENSE="GANSU_STEOM_DENSE_DIAG=2"

DOXXYZ=../xyz/large_molecular/Doxorubicin.xyz
PACXYZ=../xyz/large_molecular/paclitaxel.xyz
UBQXYZ=../xyz/large_molecular/ubiquinone10.xyz

HEADccpvdz="-g cc-pvdz --eri_method ri -ag $AUX --post_hf_method dmet_steom --frozen_core auto --num_gpus 4 --initial_guess sad --n_excited_states 5"
HEAD631="-g 6-31g --eri_method ri -ag $AUX --post_hf_method dmet_steom --frozen_core auto --num_gpus 4 --initial_guess sad --n_excited_states 5"

# run <name> <extra-env> <gansu args...>
#   extra-env is a space-separated VAR=VAL list (word-split on purpose);
#   the gansu arguments keep their quoting via "$@".
run () {
  local name="$1"; shift
  local xenv="$1"; shift
  local log="$OUT/${name}.log"
  if grep -qaE "STEOM excited-state energies|ADC\(2\) excited-state" "$log" 2>/dev/null; then
    echo ">>> [$name] already done -> skip"; return 0
  fi
  echo ">>> [$name] running -> $log"
  env $COMMON_ENV $xenv GANSU_DMET_STEOM_CKPT="$OUT/${name}.ckpt" \
      $GANSU "$@" > "$log" 2>&1 \
    && echo "   done" \
    || { echo "   FAILED (tail):"; tail -4 "$log"; }
}

DLPNO="--dmet_cluster_solver dlpno $POLISH"
AUGON="--dmet_nto_bath 0.01 --dmet_nto_bath_occ 0.5"

# ===========================================================================
# A) DOXORUBICIN  (Table 2, cc-pVDZ, STEOM) -- settings from SI S3.1
#    auto-19 uses the DLPNO cluster solver (budget 700); auto-14 is the
#    canonical-budget (460) selection, so it deliberately omits --dmet_cluster_solver.
# ===========================================================================
run dox_manual18 "$STEOM_DENSE" \
  -x $DOXXYZ $HEADccpvdz $DLPNO $AUGON --steom_n_root_cis 14 \
  --dmet_fragments "{4,5,8,9,16,17,23,25,26,27,30,31,32,33,34,35,36,37}"

run dox_auto19_B700 "$STEOM_DENSE" \
  -x $DOXXYZ $HEADccpvdz $DLPNO $AUGON \
  --dmet_steom_auto_fragment 1 --dmet_steom_auto_n_cis 14

run dox_auto14_B460 "$STEOM_DENSE" \
  -x $DOXXYZ $HEADccpvdz $AUGON \
  --dmet_steom_auto_fragment 1 --dmet_steom_auto_n_cis 14

# ===========================================================================
# B) PACLITAXEL benzamide  (Tables 4 & 5, 6-31G, canonical cluster CCSD)
#    pac_aug is the paper's augmentation demonstration (gauge INSUFFICIENT ->
#    +14 vir/+13 occ -> n_emb 184 -> 4.7122 eV); pac_bare is the same fragment
#    with the bare Schmidt bath, i.e. the before/after pair.
# ===========================================================================
run pac_bare "$STEOM_DENSE" \
  -x $PACXYZ $HEAD631 --dmet_cluster_solver canonical --steom_n_root_cis 14 \
  --dmet_fragments "{47,48,49,56,57,58,59,60,61}"

run pac_aug "$STEOM_DENSE" \
  -x $PACXYZ $HEAD631 --dmet_cluster_solver canonical --steom_n_root_cis 14 $AUGON \
  --dmet_fragments "{47,48,49,56,57,58,59,60,61}"

# ===========================================================================
# C) UBIQUINONE-10 group 0  (SI S7, 6-31G, ADC(2))
#    ?? The atom list below is what the current driver (sim 0.8) produces for
#    group 0 (15 atoms). The paper describes group 0 as "benzoquinone + 5 tail
#    carbons" -- confirm the two agree before quoting this number.
#    DENSE_DIAG does not apply to ADC(2); the solver is pinned instead.
# ===========================================================================
run ubq_group0_adc2 "" \
  -x $UBQXYZ $HEAD631 --dmet_excited_method adc2 --adc2_solver $ADC2_SOLVER \
  --dmet_fragments "{0,1,2,3,24,26,32,34,43,46,50,51,53,54,55}"

# ===========================================================================
# D) REICHARDT CT  (main text 4.5 / SI S2) -- separate script, ADC(2), four
#    fragments. Re-run it with --adc2_solver pinned to the SAME value as above,
#    otherwise donor/acceptor (7,875 singles) silently use the exact solver while
#    both/auto (21,420) use the omega=0 one and the four rows are not comparable.
# ===========================================================================
echo ">>> Reichardt CT: re-run ../script/run_reichardt_ct.sh with"
echo "    --adc2_solver $ADC2_SOLVER added to its COMMON, if its logs are stale."

echo
echo "============================ SUMMARY ============================"
echo "polish: ${POLISH:-full (converged)}   adc2_solver: $ADC2_SOLVER"
echo
for n in dox_manual18 dox_auto19_B700 dox_auto14_B460 pac_bare pac_aug ubq_group0_adc2; do
  L="$OUT/${n}.log"; [ -f "$L" ] || { printf "%-18s (not run)\n" "$n"; continue; }
  echo "===================== $n ====================="
  # Record what the run actually decided, not just what was asked for.
  grep -haE "selected [0-9]+ atom|coverage=|bath (SUFFICIENT|MARGINAL|INSUFFICIENT)|memory policy\] auto|STEOM diag\]|solver=|bath aug\]" "$L" | head -6
  awk '/STEOM excited-state energies|ADC\(2\) excited-state/{p=1} p{print} /active-space health/{if(p){print;exit}}' "$L" | head -14
  echo
done
echo ">>> logs + checkpoints in $OUT/"
