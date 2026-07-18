#!/usr/bin/env bash
# Taxol benzamide / 6-31g — PLAIN CANONICAL DMET-STEOM (the validated path).
# This is the baseline that should have been run FIRST: canonical cluster CCSD
# (the ORCA-validated DMET-STEOM solver), NO DLPNO, NO bt-polish, NO NTO-bath.
# n_emb=157 is small -> canonical is cheap.  Establishes the clean reference before
# any approximation (DLPNO) or augmentation (NTO-bath) is layered on.
# Run on remote GPU box (s177: H200x4); needs 4 GPUs free.
set -euo pipefail

GANSU=./gansu
XYZ=../xyz/large_molecular/Taxol.xyz
AUX=../auxiliary_basis/cc-pvdz-rifit.gbs
FRAG="{47,48,49,56,57,58,59,60,61}"

export GANSU_DMET_LEVEL_SHIFT_DENOM_ONLY=1   # correctness (un-shift), unchanged
export GANSU_CCSD_CONV=1e-7
export GANSU_STEOM_DENSE_DIAG=2

LOG=/tmp/taxol_steom_631g_canonical.log
echo ">>> DMET-STEOM Taxol benzamide / 6-31g / CANONICAL (validated baseline)  -> $LOG"
$GANSU -x $XYZ -g 6-31g \
  --eri_method ri -ag $AUX \
  --post_hf_method dmet_steom \
  --dmet_fragments "$FRAG" \
  --dmet_cluster_solver canonical \
  --n_excited_states 5 --steom_n_root_cis 14 \
  --initial_guess sad \
  --num_gpus 4 \
  2>&1 | tee $LOG
# Compare directly to the DLPNO run log212 (5.30/6.01/7.15/7.18/7.72, 1/5 complex):
#  - if canonical gives a stable, healthy-eta spectrum -> DLPNO was degrading it;
#  - if canonical reproduces the same instability -> the benzamide fragment/bath is
#    the issue (gauge INSUFFICIENT) and augmentation is genuinely needed.
