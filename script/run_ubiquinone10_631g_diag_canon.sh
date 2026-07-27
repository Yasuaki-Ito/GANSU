#!/usr/bin/env bash
# ubiquinone-10 benzoquinone / 6-31g — NTO-bath augmentation with CANONICAL solver.
# Fills the ubiquinone-10 row of the paper's Table 4, whose original run log was
# not retained.  Mirrors run_paclitaxel_631g_diag_canon.sh exactly so the two
# rows of Table 4 become directly comparable.
#
# Why canonical and not dlpno: NTO-bath + DLPNO is incompatible (the augmented
# bath goes near-singular in the PNO construction -> NaN); see the paclitaxel
# script.  The retained ubiquinone scripts used dlpno + bt_polish WITHOUT the
# NTO bath, which is why they do not reproduce the Table 4 conditions.
#
# Why steom_n_root_cis 20: at the default (n_excited_states+4 = 9) three of the
# five roots came out with eta < 0.96 (k2/k3/k4, low %act_o = occupied active
# space too small) -- see run_ubiquinone10_steom_ncis.sh.
#
# Memory note: BATH_DIAG builds the FULL-molecule CIS-NTO.  At 6-31g this system
# is nocc ~238 x nvir ~509 = ~1.2e5, about 1.2x the paclitaxel 6-31g case that
# fit; cc-pVDZ would not fit (the paclitaxel cc-pVDZ diagnostic OOM'd).
#
# Run on remote GPU box (s177: H200x4); needs 4 GPUs free.  cd ~/GANSU/build first.
set -euo pipefail

GANSU=./gansu
XYZ=../xyz/large_molecular/ubiquinone10.xyz
AUX=../auxiliary_basis/cc-pvdz-rifit.gbs
FRAG="{0,1,2,3,46,50,51,53,54,55}"           # benzoquinone chromophore (0-based)

export GANSU_DMET_LEVEL_SHIFT_DENOM_ONLY=1   # CORRECTNESS-CRITICAL: fires the
                                             # IP/EA/STEOM epsilon un-shift; without
                                             # it roots are high by the level shift
export GANSU_CCSD_CONV=1e-7
export GANSU_STEOM_DENSE_DIAG=2              # dense geev; all reported complex-pair
                                             # counts use this path
export GANSU_DMET_STEOM_BATH_DIAG=1          # report the bath-sufficiency gauge
export GANSU_DMET_STEOM_NTO_BATH=0.01        # tau_vir
export GANSU_DMET_STEOM_NTO_BATH_OCC=0.5     # tau_occ

NCIS=${1:-${NCIS:-20}}       # 1st positional arg wins, then $NCIS, then 20
LOG=/tmp/ubiquinone10_steom_631g_diag_canon_ncis${NCIS}.log
echo ">>> DMET-STEOM ubiquinone10 quinone / 6-31g / NTO-bath / CANONICAL / n_cis=$NCIS  -> $LOG"
$GANSU -x $XYZ -g 6-31g \
  --eri_method ri -ag $AUX \
  --post_hf_method dmet_steom \
  --dmet_fragments "$FRAG" \
  --dmet_cluster_solver canonical \
  --n_excited_states 5 --steom_n_root_cis $NCIS \
  --initial_guess sad \
  --num_gpus 4 \
  2>&1 | tee $LOG

# What to record from the log, for the paper:
#   - gauge verdict + uncaptured (vir / occ) BEFORE augmentation
#   - number of NTOs appended (+n vir, +n occ) and n_emb before -> after
#   - the five excitation energies (eV)
#   - per-root eta, and how many roots are near-defective complex pairs
# Reference (current Table 4, provenance unknown): 4.48 / 4.89 / 6.12 / 6.52 / 6.78 eV,
# eta >= 0.97, one complex-pair root.
#
# Convergence check required by the project protocol (DMET_STEOM.md 4.5):
#   NCIS=14 bash run_ubiquinone10_631g_diag_canon.sh
# and compare.  All roots within ~50 meV and eta healthy => converged; report
# either value.  If they differ, n_cis must go up until they stop moving.
