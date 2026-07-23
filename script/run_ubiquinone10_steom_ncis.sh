#!/usr/bin/env bash
# ubiquinone10 DMET-DLPNO-STEOM — n_cis-enlarged rerun.
# log213 (default n_cis) gave 3/5 roots with η<0.96: k2/k3/k4 had low %act_o
# (occupied active space too small).  Raise steom_n_root_cis to enlarge the
# state-averaged CIS-NTO density -> larger occupied active space (dox/490 protocol).
# Run on remote GPU box (s177: H200x4); needs 4 GPUs free.
set -euo pipefail

GANSU=./gansu
XYZ=../xyz/large_molecular/ubiquinone10.xyz
AUX=../auxiliary_basis/cc-pvdz-rifit.gbs
FRAG="{0,1,2,3,46,50,51,53,54,55}"

export GANSU_DMET_LEVEL_SHIFT_DENOM_ONLY=1
export GANSU_CCSD_CONV=1e-7
export GANSU_STEOM_DENSE_DIAG=2

LOG=/tmp/ubiquinone10_steom_631g_ncis20.log
echo ">>> DMET-STEOM ubiquinone10 quinone / 6-31g / n_cis=20  -> $LOG"
$GANSU -x $XYZ -g 6-31g \
  --eri_method ri -ag $AUX \
  --post_hf_method dmet_steom \
  --dmet_fragments "$FRAG" \
  --dmet_cluster_solver dlpno --dlpno_bt_polish 3 \
  --n_excited_states 5 --steom_n_root_cis 20 \
  --initial_guess sad \
  --num_gpus 4 \
  2>&1 | tee $LOG
# Compare vs log213 (default n_cis): k0/k1 should be stable (~40 meV);
# k2/k3/k4 η should rise toward >=0.96.  If some root stays low-η, the state is
# environment-delocalised -> add NTO-bath (GANSU_DMET_STEOM_NTO_BATH tau_occ).
