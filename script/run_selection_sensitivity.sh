#!/usr/bin/env bash
# GANSU: GPU Accelerated Numerical Simulation Utility
# Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
# SPDX-License-Identifier: BSD-3-Clause
#
# Sensitivity of the AUTOMATIC selection to its two thresholds: the cumulative
# coverage target T and the per-atom floor f (reviewer point: justify T=0.92,
# f=0.01). Runs the real auto-fragment code on doxorubicin, so both the
# resulting fragment size AND the excitation energies are reported per setting.
# ---------------------------------------------------------------------------
# doxorubicin / 6-31G / DLPNO cluster solver. 6-31G (not cc-pVDZ) because the
# embedded cluster of a fragment of the 68-atom molecule has a large Schmidt
# bath (n_emb ~ 440 at cc-pVDZ), and the STEOM EOM dense intermediates then
# exceed the H200 memory (a 247 GB single allocation); 6-31G shrinks n_emb to
# ~280 and the intermediate to ~40 GB, which fits. The dox bare bath is
# SUFFICIENT so no NTO augmentation is needed, keeping the DLPNO path valid.
# Sweep T at f=0.01, then f at T=0.92.
#
# Runs on the remote GPU box (H200); cd ~/GANSU/build first.
set -uo pipefail

GANSU=./gansu
XYZ=../xyz/large_molecular/Doxorubicin.xyz
AUX=../auxiliary_basis/cc-pvdz-rifit.gbs
OUT=/tmp/selection_sensitivity
mkdir -p "$OUT"

export GANSU_DMET_LEVEL_SHIFT_DENOM_ONLY=1    # correctness-critical epsilon un-shift
export GANSU_CCSD_CONV=1e-7
export GANSU_DMET_STEOM_BATH_DIAG=1
# Full "n_emb~490 memory-ceiling" set (production). NB: NO GANSU_STEOM_DENSE_DIAG
# here -- dense geev is what OOM'd the large STEOM clusters; the iterative
# (default) diagonalization is used, which is correct for the single energies we
# need (dense diag is only for complex-pair counting).
export GANSU_EA_W_HOST=1                       # EA W intermediate staged to host
export GANSU_STEOM_OPERATOR_DEVICE_BALANCING=1 # spread STEOM operator build over GPUs
export GANSU_EA_RI_LADDER=1                    # EA via RI ladder (no dense nvir^4)
export GANSU_EA_WVVVO_HOST_ASM=1               # Wvvvo host assembly (the piece that was missing)
export GANSU_DMET_CCSD_BNATIVE=1               # cluster CCSD from RI B (no dense ne^4)

# ADC(2) as the cluster solver: the selection is solver-independent (same
# fragments as STEOM), and ADC(2) is not bound by the STEOM dense-intermediate
# ceiling that OOM'd the 19-20 atom clusters, so energies come out at every
# setting. (Switch back to STEOM by dropping --dmet_excited_method for the
# smaller fragments if a STEOM energy is wanted.)
COMMON="-x $XYZ -g 6-31g --eri_method ri -ag $AUX --post_hf_method dmet_steom \
  --dmet_excited_method adc2 \
  --n_excited_states 5 --steom_n_root_cis 14 \
  --dmet_cluster_solver dlpno --dlpno_bt_polish 3 \
  --frozen_core auto --initial_guess sad --num_gpus 4 \
  --dmet_steom_auto_fragment 1 --dmet_steom_auto_n_cis 14"

run () {  # $1=tag  $2..=extra flags
  local tag="$1"; shift
  local log="$OUT/${tag}.log"
  echo ">>> $tag -> $log"
  $GANSU $COMMON "$@" > "$log" 2>&1 || echo "    (failed; see $log)"
}

# coverage sweep (f = 0.01 default)
for T in 0.85 0.90 0.92 0.95; do run "cov_$T" --dmet_steom_auto_coverage $T --dmet_steom_auto_atom_floor 0.01; done
# floor sweep (T = 0.92 default)
for F in 0.005 0.01 0.02; do run "flr_$F" --dmet_steom_auto_coverage 0.92 --dmet_steom_auto_atom_floor $F; done

echo
echo "=== summary: setting -> selected atoms, coverage, lowest dE, eta ==="
for tag in cov_0.85 cov_0.90 cov_0.92 cov_0.95 flr_0.005 flr_0.01 flr_0.02; do
  log="$OUT/${tag}.log"
  sel=$(grep -oE "selected [0-9]+ atom" "$log" 2>/dev/null | grep -oE "[0-9]+" | head -1)
  cov=$(grep -oE "coverage=0\.[0-9]+" "$log" 2>/dev/null | head -1)
  read -r low eta < <(awk '/excited-state energies/{f=1;next} /active-space health|Device Memory|END:/{f=0} f&&/^ +[0-9]+ +[0-9]/{print $3,$4; exit}' "$log" 2>/dev/null)
  printf "   %-10s : %3s atoms  %-14s  low=%s eV  eta=%s\n" "$tag" "${sel:-NA}" "${cov:-}" "${low:-NA}" "${eta:-NA}"
done
# The coverage plateau (T=0.90-0.95 giving the same fragment and energy) and the
# floor's monotone size control are the robustness statement for the defaults.
