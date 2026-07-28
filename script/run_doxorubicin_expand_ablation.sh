#!/usr/bin/env bash
# GANSU: GPU Accelerated Numerical Simulation Utility
# Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
# SPDX-License-Identifier: BSD-3-Clause
#
# POSITIVE example of BATH AUGMENTATION: when the Schmidt bath is insufficient,
# do the augmented (NTO) bath orbitals improve -- or fix -- the excitation?
# ---------------------------------------------------------------------------
# The claim under test is the augmentation itself, so the fragment is held FIXED
# and the ONLY thing that varies is the NTO-bath channel:
#
#   A) tau_vir = tau_occ = 0        -> plain Schmidt bath (dmet.cu default = OFF)
#   B) tau_vir = 0.01, tau_occ = 0  -> particle-leak NTOs only (extra virtuals)
#   C) tau_vir = 0.01, tau_occ = 0.5 -> + hole-leak NTOs (extra occupieds)
#                                       = the reference production recipe
#
# Lowering tau admits more environment-leaking NTOs, so the cluster grows toward
# the full molecule and the excitation error must decrease monotonically. The
# datum we want is: gauge verdict INSUFFICIENT/MARGINAL -> SUFFICIENT together
# with the lowest root moving toward the converged value and eta recovering.
#
# Fragment expansion (Phase B, --dmet_steom_auto_max_expand) is deliberately
# DISABLED here: it changes the atom list and would confound the bath effect.
#
# REFERENCE (converged, manual 18-atom anthraquinone): 3.7640 eV
#            (auto 19-atom, same recipe):              3.811  eV
#
# Cluster ground state: CANONICAL (exact cluster CCSD). The DLPNO cluster solver
# is still being shaken out, so it is kept OUT of this measurement.
#
# Runs on remote GPU box (s177: H200x4); needs 4 GPUs free. cd ~/GANSU/build.
set -euo pipefail

GANSU=./gansu
XYZ=../xyz/large_molecular/Doxorubicin.xyz
AUX=../auxiliary_basis/cc-pvdz-rifit.gbs

export GANSU_DMET_LEVEL_SHIFT_DENOM_ONLY=1   # CORRECTNESS-CRITICAL epsilon un-shift
export GANSU_CCSD_CONV=1e-7
export GANSU_DMET_STEOM_BATH_DIAG=1          # report the bath-sufficiency gauge in EVERY run
                                             # (it is auto-on only when augmentation runs,
                                             #  and arm A has none -- we need its gauge too)

# NB: do NOT set GANSU_DMET_STEOM_DLPNO. That env OVERRIDES --dmet_cluster_solver
# (eri_stored_steom_ccsd.cu), so setting it would silently switch the cluster
# ground back to DLPNO. The solver is chosen on the command line below.
#
# MEMORY: a cc-pVDZ doxorubicin cluster is large (n_emb ~330 at 13 atoms) and the
# canonical cluster CCSD materialises a dense 4-index MO ERI (~92 GB at n_emb=333).
# The first version of this script OOM'd because that block sat on GPU 0 while
# GPU 1-3 held 4-7 GB and the IP-EOM stage then asked for another 137.9 GB on the
# SAME device. These are the knobs that spread and shrink that footprint.
# Augmentation makes the cluster BIGGER, so they matter more here, not less.
export GANSU_DMET_CCSD_BNATIVE=1                # cluster CCSD from RI B — no dense ne^4 on device
export GANSU_STEOM_OPERATOR_DEVICE_BALANCING=1  # move IP/EA operators to the freest GPU
export GANSU_EA_RI_LADDER=1                     # EA ladder from RI factors (no dense nvir^4)
export GANSU_EA_W_HOST=1                        # host-stage the two nvir^3*nocc EA tensors
export GANSU_EA_WVVVO_HOST_ASM=1                # assemble Wvvvo on host (a-slabs)

COMMON="-x $XYZ -g cc-pvdz --eri_method ri -ag $AUX --post_hf_method dmet_steom \
  --n_excited_states 5 --steom_n_root_cis 14 --dmet_cluster_solver canonical \
  --frozen_core auto --initial_guess sad --num_gpus 4 \
  --dmet_steom_auto_fragment 1 --dmet_steom_auto_n_cis 14 \
  --dmet_steom_auto_max_expand 0"

# Coverage seed. We WANT an under-sized fragment so the un-augmented bath is
# genuinely insufficient — that is the premise of the experiment. 0.92 (default)
# already gives the full chromophore and leaves nothing for the bath to fix.
COVS=${1:-"0.70"}
OUT=/tmp/dox_bath_ablation
mkdir -p $OUT

lowest() {  # $1 = log -> lowest STEOM root in eV (field 3) and its eta (field 4)
  awk '/STEOM excited-state energies/{f=1;next} /active-space health/{f=0}
       f && /^ +[0-9]+ +[0-9]/{print $3, $4; exit}' "$1"
}
verdict()   { grep -oE "SUFFICIENT|MARGINAL|INSUFFICIENT" "$1" | tail -1; }
uncapt()    { grep -oE "uncaptured=[0-9.]+" "$1" | tail -1; }
nemb()      { grep -oE "n_emb=[0-9]+" "$1" | head -1; }
natoms()    { grep -oE "selected [0-9]+ atom" "$1" | grep -oE "[0-9]+" | head -1; }

# arm name | tau_vir | tau_occ
ARMS="A:0:0 B:0.01:0 C:0.01:0.5"

echo "REFERENCE (converged): manual 18-atom 3.7640 eV | auto 19-atom 3.811 eV"
echo "======================================================================"
for cov in $COVS; do
  for arm in $ARMS; do
    name=${arm%%:*} ; rest=${arm#*:} ; tv=${rest%%:*} ; to=${rest#*:}
    log=$OUT/dox_cov${cov}_${name}.log
    echo ">>> cov=$cov arm=$name (tau_vir=$tv tau_occ=$to) -> $log"
    # Augmentation is per-run: unset for arm A (0 means OFF in dmet.cu anyway,
    # but keep the environment literally clean so the log is unambiguous).
    if [ "$tv" = "0" ] && [ "$to" = "0" ]; then
      env -u GANSU_DMET_STEOM_NTO_BATH -u GANSU_DMET_STEOM_NTO_BATH_OCC \
        $GANSU $COMMON --dmet_steom_auto_coverage $cov > "$log" 2>&1 \
        || { echo "    (run failed; see $log)"; continue; }
    else
      GANSU_DMET_STEOM_NTO_BATH=$tv GANSU_DMET_STEOM_NTO_BATH_OCC=$to \
        $GANSU $COMMON --dmet_steom_auto_coverage $cov > "$log" 2>&1 \
        || { echo "    (run failed; see $log)"; continue; }
    fi
  done
  echo "---- coverage $cov ------------------------------------------------"
  for arm in $ARMS; do
    name=${arm%%:*} ; rest=${arm#*:} ; tv=${rest%%:*} ; to=${rest#*:}
    l=$OUT/dox_cov${cov}_${name}.log
    [ -f "$l" ] || continue
    printf "  %s (tau_vir=%-5s tau_occ=%-3s): atoms=%-3s %-10s gauge=%-12s %-22s lowest= %s\n" \
           "$name" "$tv" "$to" "$(natoms $l)" "$(nemb $l)" "$(verdict $l)" "$(uncapt $l)" "$(lowest $l)"
  done
  echo "  -> positive example iff the gauge improves AND the lowest root moves"
  echo "     toward 3.76-3.81 eV monotonically from A -> B -> C, eta >= 0.96."
  echo
done

# What to record for the paper if this gives a clean win:
#   per arm: n_emb / gauge verdict + uncaptured / lowest DeltaE / eta
#   the monotone A -> B -> C trend IS the result (bath augmentation converges the
#   excitation), and C reproducing 3.76-3.81 is the "solved" end point.
