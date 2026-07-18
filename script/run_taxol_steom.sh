#!/usr/bin/env bash
# DMET-STEOM electronic excitation of Taxol (C47H51NO14, 113 atoms)
# Target chromophore = BENZAMIDE (N-benzoyl): phenyl {56-61} + amide C48=O49 + N47
# Cut bond = N47-C46 (amide-N to sp3 C-3', clean single bond)
# Run on remote GPU box (s177: H200x4).  cd ~/GANSU/build first.
set -euo pipefail

GANSU=./gansu
XYZ=../xyz/large_molecular/Taxol.xyz
AUX=../auxiliary_basis/cc-pvdz-rifit.gbs
FRAG="{47,48,49,56,57,58,59,60,61}"          # benzamide chromophore (0-based)

# CORRECTNESS-CRITICAL env:
#  The DMET cluster gets a level shift (+s on virtual ε) whenever the cluster
#  HOMO-LUMO gap < 0.5 Ha, to stabilise the small-gap CCSD.  DENOM_ONLY=1 makes
#  the CCSD converge to the TRUE (unshifted) energy AND fires the IP/EA/STEOM
#  "ε un-shifted (−s)" correction so the excitation roots are NOT biased by ~+s.
#  WITHOUT it, roots come out ~+s (≈ +2 eV) too high (verified: log211 benzamide
#  6.97-9.67 eV had NO un-shift line; s was 0.0736 Ha).  Always keep this on.
export GANSU_DMET_LEVEL_SHIFT_DENOM_ONLY=1
export GANSU_CCSD_CONV=1e-7          # CCSD tail cutoff: iter ~4x fewer, root impact < 1e-4 eV

# ---- Stage 1: 6-31g smoke test (fast, proven money-shot path like dox) ----
BASIS=6-31g
LOG=/tmp/taxol_steom_631g.log
echo ">>> Stage 1: DMET-STEOM Taxol benzamide / $BASIS  -> $LOG"
$GANSU -x $XYZ -g $BASIS \
  --eri_method ri -ag $AUX \
  --post_hf_method dmet_steom \
  --dmet_fragments "$FRAG" \
  --dmet_cluster_solver dlpno --dlpno_bt_polish 3 \
  --n_excited_states 5 \
  --initial_guess sad \
  --num_gpus 4 \
  2>&1 | tee $LOG
# Sanity check after run: the log MUST contain 3 lines
#   "[DMET-STEOM] IP-EOM operator ε un-shifted (−s=0.0736) ..."  (and EA, STEOM)
# and "denominator-only level shift ON (s=0.0736) ...".  If absent, roots are biased.

# ---- Stage 2 (optional): cc-pVDZ production once 6-31g is clean ----
# Benzamide fragment is only 9 atoms (~n_emb 250, well under the dox 427 ceiling),
# so cc-pVDZ should fit on one H200 without the heavy EA-host env recipe.
# Uncomment to run:
# BASIS=cc-pvdz
# LOG=/tmp/taxol_steom_ccpvdz.log
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
