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
OUT=${OUT:-/tmp/reichardt_ct}
mkdir -p "$OUT"

# The four fragments differ in size by 3x, and the ADC(2) `auto` solver switches
# at singles = 10^4: the single-end clusters (7,875 singles) get the exact
# omega-iterated Schur solver while donor-U-acceptor and auto (21,420) get the
# matrix-free Davidson at omega = 0, which carries a ~0.005-0.02 Ha bias. Four
# rows compared against each other must not be computed by two different solvers,
# so pin it. schur_davidson is the only choice available to ALL four: the exact
# solver dense-diagonalises a singles x singles M_eff once per root per omega
# iteration, which is not tractable at 21,420. The bias is common to every row
# and is far smaller than the effect being measured (single-end fragments sit
# ~3 eV above the charge-transfer state), but the absolute numbers are ~0.1-0.5 eV
# high and should be quoted as such.
ADC2_SOLVER=${ADC2_SOLVER:-schur_davidson}

export GANSU_DMET_LEVEL_SHIFT_DENOM_ONLY=1
export GANSU_CCSD_CONV=1e-7
export GANSU_DMET_STEOM_BATH_DIAG=1

DONOR="{0,4,8,9,12,13,14}"                     # phenolate donor (hole), 7 atoms
ACCEPTOR="{1,2,3,5,6,7,15}"                    # pyridinium acceptor (particle), 7 atoms
BOTH="{0,1,2,3,4,5,6,7,8,9,12,13,14,15}"       # donor U acceptor, 14 atoms

COMMON="-x $REI -g cc-pvdz --eri_method ri -ag $AUX --post_hf_method dmet_steom \
  --dmet_excited_method adc2 --adc2_solver $ADC2_SOLVER --n_excited_states 5 \
  --frozen_core auto --initial_guess sad --num_gpus 4"

# GANSU can hang at process exit AFTER the energies are already written for these
# small single-fragment ADC(2) runs. Run each in the background, poll the log for
# the completed 5-root energy block, then reap the process so the loop proceeds.
NROOT=5
POLL_MAX=2400   # poll up to ~200 min: the memory-lean schur_omega solver is slow
                # (one root at a time) for the larger single-end clusters; this is
                # only a backstop, each job is reaped as soon as its 5 roots appear.
roots_in () {  # count completed ADC(2) roots already in a log (0 if absent)
  [ -f "$1" ] || { echo 0; return; }
  local n
  n=$(awk '/excited-state energies/{f=1;next} /active-space health|Device Memory|END:/{f=0}
           f&&/^ +[0-9]+ +[0-9]/{c++} END{print c+0}' "$1" 2>/dev/null)
  echo "${n:-0}"
}
run_one () {  # $1=tag ; $2..=extra flags
  local tag="$1"; shift
  local log="$OUT/${tag}.log"
  if [ "$(roots_in "$log")" -ge "$NROOT" ]; then
    echo ">>> $tag: already complete ($(roots_in "$log") roots), skipping (no recompute)"; return
  fi
  echo ">>> $tag -> $log"
  $GANSU $COMMON "$@" > "$log" 2>&1 &                          # run in background
  local pid=$!
  local i
  for i in $(seq 1 "$POLL_MAX"); do                           # poll (see POLL_MAX)
    kill -0 "$pid" 2>/dev/null || { wait "$pid" 2>/dev/null; return; }   # exited cleanly
    if [ "$(roots_in "$log")" -ge "$NROOT" ]; then            # energies written; reap exit-hang
      sleep 5; kill -TERM "$pid" 2>/dev/null; sleep 2; kill -9 "$pid" 2>/dev/null
      wait "$pid" 2>/dev/null; return
    fi
    sleep 5
  done
  kill -9 "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  echo "    (timed out without $NROOT roots; see $log)"
}

# Tags match the existing /tmp logs: donor.log is already complete and is skipped.
run_one donor    --dmet_fragments "$DONOR"
run_one acceptor --dmet_fragments "$ACCEPTOR"
run_one both     --dmet_fragments "$BOTH"
run_one auto     --dmet_steom_auto_fragment 1 --dmet_steom_auto_n_cis 12

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
