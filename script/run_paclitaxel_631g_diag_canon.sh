#!/usr/bin/env bash
# paclitaxel benzamide / 6-31g — NTO-bath augmentation with CANONICAL cluster solver.
# log218 established: (1) gauge = bath INSUFFICIENT (uncaptured 0.636 vir / 0.450 occ)
# -> the bare benzamide fragment is under-domain (root cause of the unstable spectrum);
# (2) augmentation added 14 vir + 13 occ leaking NTOs (n_emb 157->184) = the right fix,
# BUT the DLPNO cluster solver then produced NaN (NTO-bath + DLPNO is an untested/
# incompatible combo; the augmented bath goes near-singular in the PNO construction).
# NTO-bath was validated historically with the CANONICAL solver, so drop --dmet_cluster_solver
# dlpno.  At n_emb=184 canonical cluster CCSD is cheap and robust.
# Run on remote GPU box (s177: H200x4); needs 4 GPUs free.
set -euo pipefail

GANSU=./gansu
XYZ=../xyz/large_molecular/paclitaxel.xyz
AUX=../auxiliary_basis/cc-pvdz-rifit.gbs
FRAG="{47,48,49,56,57,58,59,60,61}"

export GANSU_DMET_LEVEL_SHIFT_DENOM_ONLY=1
export GANSU_CCSD_CONV=1e-7
export GANSU_STEOM_DENSE_DIAG=2
export GANSU_DMET_STEOM_BATH_DIAG=1
export GANSU_DMET_STEOM_NTO_BATH=0.01
export GANSU_DMET_STEOM_NTO_BATH_OCC=0.5

LOG=/tmp/paclitaxel_steom_631g_diag_canon.log
echo ">>> DMET-STEOM paclitaxel benzamide / 6-31g / NTO-bath / CANONICAL solver  -> $LOG"
$GANSU -x $XYZ -g 6-31g \
  --eri_method ri -ag $AUX \
  --post_hf_method dmet_steom \
  --dmet_fragments "$FRAG" \
  --dmet_cluster_solver canonical \
  --n_excited_states 5 --steom_n_root_cis 14 \
  --initial_guess sad \
  --num_gpus 4 \
  2>&1 | tee $LOG
# Success = no NaN, and the STEOM roots move well below the plain 6-31g run
# (log212: 5.30/6.01/...) toward benzamide's ~4.6-5.5 eV band with healthy eta and
# fewer/no complex pairs.  That would confirm: bath was insufficient, augment fixes it.
