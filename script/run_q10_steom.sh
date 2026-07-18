#!/usr/bin/env bash
# DMET-STEOM electronic excitation of Coenzyme Q10 (ubiquinone, C59H90O4, 153 atoms)
# Target chromophore = 2,3-dimethoxy-5-methyl-1,4-BENZOQUINONE head group
#   ring {46,50,51,53,54,55} + carbonyls C51=O2 / C53=O3 + methoxy O0 / O1
# Cut bonds: C46-C40 (polyprenyl tail), C50-C57 (ring-methyl),
#            O0-C61 / O1-C62 (methoxy O kept in fragment, CH3 -> bath; dox-style)
# Run on remote GPU box (s177: H200x4).  cd ~/GANSU/build first.
set -euo pipefail

GANSU=./gansu
XYZ=../xyz/large_molecular/CoenzymeQ10.xyz
AUX=../auxiliary_basis/cc-pvdz-rifit.gbs
FRAG="{0,1,2,3,46,50,51,53,54,55}"           # benzoquinone chromophore (0-based)

# CORRECTNESS-CRITICAL (see run_taxol_steom.sh): when the cluster gap < 0.5 Ha a
# level shift +s is applied to virtual ε.  DENOM_ONLY=1 converges CCSD to the TRUE
# energy AND fires the IP/EA/STEOM "ε un-shifted (−s)" correction, else roots are
# biased ~+s (≈ +2 eV) too high.  Always keep this on for DMET-STEOM.
export GANSU_DMET_LEVEL_SHIFT_DENOM_ONLY=1
export GANSU_CCSD_CONV=1e-7          # CCSD tail cutoff: iter ~4x fewer, root impact < 1e-4 eV

# ---- Stage 1: 6-31g smoke test (fast, proven money-shot path like dox) ----
BASIS=6-31g
LOG=/tmp/q10_steom_631g.log
echo ">>> Stage 1: DMET-STEOM CoenzymeQ10 quinone / $BASIS  -> $LOG"
$GANSU -x $XYZ -g $BASIS \
  --eri_method ri -ag $AUX \
  --post_hf_method dmet_steom \
  --dmet_fragments "$FRAG" \
  --dmet_cluster_solver dlpno --dlpno_bt_polish 3 \
  --n_excited_states 5 \
  --initial_guess sad \
  --num_gpus 4 \
  2>&1 | tee $LOG
# Sanity: log MUST show "ε un-shifted (−s=...)" (IP/EA/STEOM) + "denominator-only
# level shift ON" if a shift fired.  If a shift line appears WITHOUT un-shift, re-run.

# ---- Stage 2 (optional): cc-pVDZ production once 6-31g is clean ----
# 10-atom fragment (~n_emb 280) is well under the dox 427 ceiling.
# BASIS=cc-pvdz
# LOG=/tmp/q10_steom_ccpvdz.log
# GANSU_STEOM_DENSE_DIAG=2 \
# GANSU_DMET_LEVEL_SHIFT_DENOM_ONLY=1 GANSU_CCSD_CONV=1e-7 \
# $GANSU -x $XYZ -g $BASIS \
#   --eri_method ri -ag $AUX \
#   --post_hf_method dmet_steom \
#   --dmet_fragments "$FRAG" \
#   --dmet_cluster_solver dlpno --dlpno_bt_polish 3 \
#   --n_excited_states 5 --steom_n_root_cis 14 \
#   --initial_guess sad --num_gpus 4 \
#   2>&1 | tee $LOG
