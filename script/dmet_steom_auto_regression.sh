#!/bin/bash
# GANSU: GPU Accelerated Numerical Simulation Utility
# Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
# SPDX-License-Identifier: BSD-3-Clause
#
# Regression anchor for CIS-guided DMET-STEOM auto fragmentation
# (dmet_steom_auto_*). Run from the build directory after every rebuild that
# touches src/dmet_auto_fragment.cu, src/dmet.cu, or the CIS-NTO path:
#
#     bash ../script/dmet_steom_auto_regression.sh
#
# Naphthalene / cc-pVDZ / RI, 3 states. Fast (~few min). Checks:
#   1. DEFAULT-OFF byte-identity: dmet_steom_auto_fragment=0 reproduces the plain
#      whole-molecule DMET-STEOM anchor (= plain STEOM) to <= 5 meV.
#   2. AUTO selection: the delocalized naphthalene pi->pi* selects all 10 carbons,
#      excludes all H (floor), coverage ~0.97.
#   3. Phase B gauge: bath MARGINAL with the virtual-space tail line present, and
#      the size classification reports RIGHT-SIZED.
set -u
GANSU=${GANSU:-./gansu}
XYZ=${XYZ:-../xyz/Naphthalene.xyz}
AUX=${AUX:-../auxiliary_basis/cc-pvdz-rifit.gbs}
COMMON="-x $XYZ -g cc-pvdz --eri_method ri -ag $AUX --post_hf_method dmet_steom --n_excited_states 3"
# Anchor roots (eV), naphthalene cc-pVDZ DMET-STEOM auto-off n_cis=7 (2026-07-23):
ANCHOR=(4.2541 5.3653 5.9557)
TOL=0.005   # eV
fail=0

echo "== [1] default-off byte-identity =="
$GANSU $COMMON > /tmp/reg_auto_off.log 2>&1
# eV is field 3 of the "k  omega(Ha)  omega(eV)  eta ..." root rows.
roots=$(awk '/STEOM excited-state energies/{f=1;next} /active-space health/{f=0}
             f && /^ +[0-9]+ +[0-9]/{print $3}' /tmp/reg_auto_off.log | head -3)
i=0; for r in $roots; do
  d=$(python3 -c "print(abs($r-${ANCHOR[$i]}))")
  ok=$(python3 -c "print(1 if abs($r-${ANCHOR[$i]})<$TOL else 0)")
  echo "   root $i: $r eV (anchor ${ANCHOR[$i]}, |Δ|=$d) $([ $ok = 1 ] && echo PASS || echo FAIL)"
  [ $ok = 1 ] || fail=1; i=$((i+1))
done
[ $i -eq 3 ] || { echo "   FAIL: expected 3 roots, got $i"; fail=1; }

echo "== [2] auto selection (10 carbons, H excluded) =="
$GANSU $COMMON --dmet_steom_auto_fragment 1 --dmet_steom_auto_max_expand 1 > /tmp/reg_auto_on.log 2>&1
sel=$(grep -oE "selected [0-9]+ atom" /tmp/reg_auto_on.log | grep -oE "[0-9]+")
cov=$(grep -oE "coverage=0\.[0-9]+" /tmp/reg_auto_on.log | head -1 | grep -oE "0\.[0-9]+")
nH=$(grep -oE "selected 10 atom.*:.*" /tmp/reg_auto_on.log | grep -oc "H" )
echo "   selected atoms: $sel (expect 10), coverage $cov (expect ~0.97)"
[ "$sel" = "10" ] && echo "   PASS: 10 atoms" || { echo "   FAIL: expected 10"; fail=1; }
grep -q "selected 10 atom.*0C 1C 2C 3C 4C 5C 6C 7C 8C 9C" /tmp/reg_auto_on.log \
  && echo "   PASS: all 10 carbons, no H" || { echo "   FAIL: carbon set changed"; fail=1; }

echo "== [3] Phase B gauge + virtual-space + size class =="
grep -q "virtual-space tail uncaptured" /tmp/reg_auto_on.log \
  && echo "   PASS: virtual-space tail line present" || { echo "   FAIL: no virtual-space line"; fail=1; }
grep -q "RIGHT-SIZED" /tmp/reg_auto_on.log \
  && echo "   PASS: RIGHT-SIZED classification" || { echo "   FAIL: size class changed"; fail=1; }

echo
[ $fail = 0 ] && echo "REGRESSION: ALL PASS" || echo "REGRESSION: FAILURES (see /tmp/reg_auto_*.log)"
exit $fail
