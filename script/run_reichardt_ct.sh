#!/usr/bin/env bash
# GANSU: GPU Accelerated Numerical Simulation Utility
# Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
# SPDX-License-Identifier: BSD-3-Clause
#
# Reichardt's dye: does the SELECTION (taking BOTH charge-transfer ends) matter?
# Separates the fragmentation choice from the solver's CT error by running the
# SAME ADC(2) solver on four fragments:
#   donor-only    : phenolate hole atoms only          -> cannot host the CT
#   acceptor-only : pyridinium particle atoms only      -> cannot host the CT
#   both          : donor U acceptor (14 atoms)         -> hosts the CT
#   auto          : the method's own selection (22 atoms) -> hosts the CT
# Only fragments containing both ends can produce the low-lying CT state; a
# donor- or acceptor-only fragment leaves only a high-energy local excitation.
# Any residual absolute error is the ADC(2)/CT limitation, common to all four,
# not the fragmentation. Atom sets are derived from the state-0 hole/particle
# NTO scores in per_state/reich_states.json (donor = top hole, acceptor = top
# particle; the two are disjoint). STEOM OOMs on this 72-atom system, so ADC(2)
# is used throughout (cf. run_adc2_energytable.sh).
#
# Runs on the remote GPU box (H200); cd ~/GANSU/build first.
set -uo pipefail

GANSU=./gansu
AUX=../auxiliary_basis/cc-pvdz-rifit.gbs
REI=../xyz/large_molecular/Reichardt_dye.xyz
OUT=/tmp/reichardt_ct
mkdir -p "$OUT"

export GANSU_DMET_LEVEL_SHIFT_DENOM_ONLY=1
export GANSU_CCSD_CONV=1e-7
export GANSU_DMET_STEOM_BATH_DIAG=1

DONOR="{0,4,8,9,12,13,14}"                     # phenolate donor (hole), 7 atoms
ACCEPTOR="{1,2,3,5,6,7,15}"                    # pyridinium acceptor (particle), 7 atoms
BOTH="{0,1,2,3,4,5,6,7,8,9,12,13,14,15}"       # donor U acceptor, 14 atoms

COMMON="-x $REI -g cc-pvdz --eri_method ri -ag $AUX --post_hf_method dmet_steom \
  --dmet_excited_method adc2 --n_excited_states 5 \
  --frozen_core auto --initial_guess sad --num_gpus 4"

echo ">>> donor-only    -> $OUT/donor.log";    $GANSU $COMMON --dmet_fragments "$DONOR"    > "$OUT/donor.log"    2>&1 || echo "    (failed)"
echo ">>> acceptor-only -> $OUT/acceptor.log"; $GANSU $COMMON --dmet_fragments "$ACCEPTOR" > "$OUT/acceptor.log" 2>&1 || echo "    (failed)"
echo ">>> both          -> $OUT/both.log";     $GANSU $COMMON --dmet_fragments "$BOTH"     > "$OUT/both.log"     2>&1 || echo "    (failed)"
echo ">>> auto          -> $OUT/auto.log";     $GANSU $COMMON --dmet_steom_auto_fragment 1 --dmet_steom_auto_n_cis 12 > "$OUT/auto.log" 2>&1 || echo "    (failed)"

echo
echo "=== lowest DMET-ADC(2) excitation energies (eV) ==="
for L in donor acceptor both auto; do
  echo "-- $L --"
  awk '/excited-state energies/{f=1;next} /active-space health|Device Memory|END:/{f=0}
       f&&/^ +[0-9]+ +[0-9]/{print "   k="$1"  "$3" eV"}' "$OUT/$L.log" 2>/dev/null | head -5
done
# Expected: donor-only and acceptor-only give only high-energy local excitations;
# both and auto reproduce the low CT state. The selection, not the solver, is
# what makes the CT state accessible.
