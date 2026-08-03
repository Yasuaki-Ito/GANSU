#!/usr/bin/env bash
# GANSU: GPU Accelerated Numerical Simulation Utility
# Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
# SPDX-License-Identifier: BSD-3-Clause
#
# PILOT: doxorubicin auto-fragment STEOM, scanning the CIS-NTO population knob
# --dmet_steom_auto_n_cis over {14,16,18,20}, with NTO-bath augmentation on.
#
# Context: the current code reproduces the paper's doxorubicin auto SELECTION
# exactly (19 atoms, gauge 0.0053) but at n_cis=14 the STEOM active character is
# unhealthy (4/5 roots eta<0.96) and the energies (k0=3.635 eV) differ from the
# paper's n_cis=14 numbers (3.811). Established project knowledge: n_cis, not the
# DLPNO thresholds, is the primary knob for eta health (490 series: n_cis=9 was
# under-converged; n_cis=20 gave eta>=0.987, zero complex pairs). The dox
# diagnostic showed requested==effective (14->14->14 / 9->9->9), i.e. the active
# space is simply small -> raise n_cis. This pilot finds the n_cis at which dox
# becomes eta-healthy, which then sets the converged reference for the rewrite.
#
# Memory-layout kit kept (bit-exact); the convergence-affecting
# GANSU_EOM_MAX_SUB_PER_ROOT cap is intentionally NOT set.
#
# Runs on the remote GPU box (H200); cd ~/GANSU/build first, then:
#   bash ../script/run_dox_ncis_scan.sh
set -uo pipefail

GANSU=./gansu
AUX=../auxiliary_basis/cc-pvdz-rifit.gbs
OUT=/tmp/dox_ncis
mkdir -p "$OUT"

# memory-layout (bit-exact) + correctness + DLPNO + NTO-bath augmentation
export GANSU_DMET_STEOM_RI_BLOCK=1 GANSU_CCSD_RI_BNATIVE=1 GANSU_CCSD_RI_LADDER_TILE=1
export GANSU_CCSD_OCCI=1 GANSU_CCSD_VR_TILE=1
export GANSU_EA_RI_LADDER=1 GANSU_EA_W_HOST=1 GANSU_EA_WVVVO_HOST_ASM=1
export GANSU_STEOM_OPERATOR_DEVICE_BALANCING=1
export GANSU_DMET_STEOM_CLUSTER_GPU=1 GANSU_STEOM_BARH_GPU=3 GANSU_STEOM_SHARE_BARH=1
export GANSU_IP_SIGMA_GEMM=1 OMP_NUM_THREADS=64
export GANSU_DMET_LEVEL_SHIFT_DENOM_ONLY=1 GANSU_CCSD_CONV=1e-7 GANSU_DMET_STEOM_BATH_DIAG=1
export GANSU_DMET_STEOM_DLPNO=2 GANSU_DMET_STEOM_NTO_BATH=0.01 GANSU_DMET_STEOM_NTO_BATH_OCC=0.5

for NC in 14 16 18 20; do
  echo "=============================================================="
  echo ">>> doxorubicin auto, n_cis = $NC -> $OUT/dox_ncis${NC}.log"
  echo "=============================================================="
  $GANSU -x ../xyz/large_molecular/Doxorubicin.xyz -g cc-pvdz --eri_method ri \
    -ag $AUX --post_hf_method dmet_steom --frozen_core auto --num_gpus 4 \
    --initial_guess sad --n_excited_states 5 \
    --dmet_steom_auto_fragment 1 --dmet_steom_auto_n_cis $NC \
    > "$OUT/dox_ncis${NC}.log" 2>&1 || { echo "   FAILED (tail):"; tail -5 "$OUT/dox_ncis${NC}.log"; continue; }

  L="$OUT/dox_ncis${NC}.log"
  echo "-- selection / gauge --"
  grep -hE "selected [0-9]+ atom|coverage=|bath (SUFFICIENT|MARGINAL|INSUFFICIENT)|uncaptured=|bath aug\] added" "$L" | head -6
  echo "-- active-space requested -> routed -> effective --"
  grep -hiE "requested|routed|effective|n_act_occ|n_act_vir" "$L" | grep -iE "occ|vir|route|effective|request" | head -8
  echo "-- STEOM energies + eta health --"
  awk '/STEOM excited-state energies/{p=1} p{print} /active-space health/{if(p){print;exit}}' "$L"
  echo
done

echo "=============================================================="
echo ">>> SUMMARY: lowest root and eta-health vs n_cis"
for NC in 14 16 18 20; do
  L="$OUT/dox_ncis${NC}.log"; [ -f "$L" ] || continue
  k0=$(awk '/STEOM excited-state energies/{p=1} p && /^   0 /{print $3; exit}' "$L")
  health=$(grep -hE "active-space health" "$L" | head -1 | sed -E 's/^ *//')
  printf "n_cis=%-3s  k0=%-8s  %s\n" "$NC" "${k0:-NA}" "${health:-NA}"
done
echo ">>> paper (n_cis=14): k0=3.811 eV.  Want: n_cis where health = 0/5 below eta=0.96."
