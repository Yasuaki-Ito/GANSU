#!/usr/bin/env bash
# paclitaxel benzamide / cc-pVDZ — n_cis=20 convergence probe.
# log214 (n_cis=14) STEOM roots sat +2.2 eV above the 6-31g run while the CIS
# reference agreed (~7.95 eV both) => the STEOM G is near-defective (2/5 complex
# pairs, max|Im|=2.80 eV) and the absolute roots are NOT converged.
# Raise n_cis: if the complex pairs vanish and roots stabilise (<~40 meV vs a
# later n_cis), the spectrum is trustworthy (p-DDPA 490 behaviour); if they
# persist, the near-defectiveness is intrinsic to this cluster's G (dox behaviour)
# and the reported real parts carry an inherent +-0.1-0.15 eV uncertainty.
# Run on remote GPU box (s177: H200x4); needs 4 GPUs free.
set -euo pipefail

GANSU=./gansu
XYZ=../xyz/large_molecular/paclitaxel.xyz
AUX=../auxiliary_basis/cc-pvdz-rifit.gbs
FRAG="{47,48,49,56,57,58,59,60,61}"

export GANSU_DMET_LEVEL_SHIFT_DENOM_ONLY=1
export GANSU_CCSD_CONV=1e-7
export GANSU_STEOM_DENSE_DIAG=2

LOG=/tmp/paclitaxel_steom_ccpvdz_ncis20.log
echo ">>> DMET-STEOM paclitaxel benzamide / cc-pVDZ / n_cis=20  -> $LOG"
$GANSU -x $XYZ -g cc-pvdz \
  --eri_method ri -ag $AUX \
  --post_hf_method dmet_steom \
  --dmet_fragments "$FRAG" \
  --dmet_cluster_solver dlpno --dlpno_bt_polish 3 \
  --n_excited_states 5 --steom_n_root_cis 20 \
  --initial_guess sad \
  --num_gpus 4 \
  2>&1 | tee $LOG
# Check the STEOM block for "complex-root recovery": is it 0/5 now?  And do the
# roots move <~40 meV vs log214?  Both -> converged.  Else -> intrinsic G defect.
