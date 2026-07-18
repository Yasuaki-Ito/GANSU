#!/usr/bin/env bash
# DMET-DLPNO-STEOM-CCSD excited states of Taxol / cc-pVDZ (DZP) — production.
# Benzamide chromophore fragment {47,48,49,56,57,58,59,60,61} (see run_taxol_steom.sh).
# Run on remote GPU box (s177: H200x4).  cd ~/GANSU/build first.
# ⚠ Needs the 4 GPUs free — do NOT launch while the Q10 job is still running
#   (s177 GPUs are Exclusive_Process; a held GPU makes cuSOLVER handle init fail).
set -euo pipefail

GANSU=./gansu
XYZ=../xyz/large_molecular/Taxol.xyz
AUX=../auxiliary_basis/cc-pvdz-rifit.gbs
FRAG="{47,48,49,56,57,58,59,60,61}"

# correctness-critical (see run_taxol_steom.sh): denominator-only shift + ε un-shift
export GANSU_DMET_LEVEL_SHIFT_DENOM_ONLY=1
export GANSU_CCSD_CONV=1e-7
export GANSU_STEOM_DENSE_DIAG=2          # force deterministic dense geev (avoid Davidson artifact)

LOG=/tmp/taxol_steom_ccpvdz.log
echo ">>> DMET-STEOM Taxol benzamide / cc-pVDZ  -> $LOG"
$GANSU -x $XYZ -g cc-pvdz \
  --eri_method ri -ag $AUX \
  --post_hf_method dmet_steom \
  --dmet_fragments "$FRAG" \
  --dmet_cluster_solver dlpno --dlpno_bt_polish 3 \
  --n_excited_states 5 --steom_n_root_cis 14 \
  --initial_guess sad \
  --num_gpus 4 \
  2>&1 | tee $LOG

# After it finishes, check the log has (else re-run):
#   "denominator-only level shift ON (s=...)"   +   3x "ε un-shifted (−s=...)"
#   "active-space health: 0/5 root(s) below η = 0.96"
# Then for an n_cis convergence check, rerun once with --steom_n_root_cis 20 and
# confirm the roots move < ~40 meV (the dox/490 convergence protocol).
