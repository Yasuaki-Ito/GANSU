#!/usr/bin/env bash
# paclitaxel benzamide / 6-31g — gauge (in-domain) diagnostic + NTO-bath augmentation.
# The cc-pVDZ diagnostic (log217) OOM'd: BATH_DIAG builds the FULL-molecule CIS-NTO
# (nocc=226, nvir=959, dim=216734 at cc-pVDZ -> 50.8 GB alloc on top of 172 GB).
# 6-31g shrinks the full-system CIS (nvir 434, dim ~98k) so the leakage analysis fits.
# The gauge verdict (SUFFICIENT/INSUFFICIENT) is what we need to confirm whether the
# benzamide fragment is in-domain; NTO-bath then tests recovery.
# Run on remote GPU box (s177: H200x4); needs 4 GPUs free.
set -euo pipefail

GANSU=./gansu
XYZ=../xyz/large_molecular/paclitaxel.xyz
AUX=../auxiliary_basis/cc-pvdz-rifit.gbs
FRAG="{47,48,49,56,57,58,59,60,61}"

export GANSU_DMET_LEVEL_SHIFT_DENOM_ONLY=1
export GANSU_CCSD_CONV=1e-7
export GANSU_STEOM_DENSE_DIAG=2
export GANSU_DMET_STEOM_BATH_DIAG=1       # print gauge (in-domain check)
export GANSU_DMET_STEOM_NTO_BATH=0.01     # tau_vir  (virtual NTO-bath augmentation)
export GANSU_DMET_STEOM_NTO_BATH_OCC=0.5  # tau_occ  (occupied NTO-bath augmentation)

LOG=/tmp/paclitaxel_steom_631g_diag.log
echo ">>> DMET-STEOM paclitaxel benzamide / 6-31g / gauge+NTO-bath  -> $LOG"
$GANSU -x $XYZ -g 6-31g \
  --eri_method ri -ag $AUX \
  --post_hf_method dmet_steom \
  --dmet_fragments "$FRAG" \
  --dmet_cluster_solver dlpno --dlpno_bt_polish 3 \
  --n_excited_states 5 --steom_n_root_cis 14 \
  --initial_guess sad \
  --num_gpus 4 \
  2>&1 | tee $LOG
# Look for the gauge line (uncaptured -> SUFFICIENT/INSUFFICIENT) in the CIS-NTO
# stage, then whether the augmented STEOM roots differ from the plain 6-31g run
# (log212: 5.30/6.01/7.15/7.18/7.72) — augmentation changing them a lot = the
# bare benzamide bath was under-describing the states.
