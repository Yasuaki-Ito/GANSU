# STEOM-CCSD formula validation against ORCA (canonical STEOM)

Why: the harness `script/steom_fulltest_frozen.py` assumed STEOM(complete active)
== EOM-EE-CCSD bit-exact. That is FALSE (literature: STEOM singles are only
*partially* decoupled from higher excitations). So EOM is the wrong reference and
cannot distinguish "formula bug" from "inherent STEOM-EOM difference". ORCA's
canonical STEOM-CCSD is the trusted reference instead.

Two independent correct-looking Python formulas (Nooijen `build_g_canonical_full`
and the faithful CFOUR port `script/steom_cfour_weff.py`) agree at ~53 mHa
(triplet) / ~66 mHa (singlet) vs EOM on H2O sto-3g — consistent with either a
shared missing term OR the inherent STEOM-EOM gap. ORCA decides which.

## Run (remote ORCA box)
```bash
cd bench/steom_validation
for f in ch2o_sto3g_steom h2o_sto3g_steom ch2o_ccpvdz_steom; do
  $ORCA/orca $f.inp > $f.out 2>&1
done
grep -A20 "STEOM-CCSD RESULTS\|TRANSITION ENERGIES\|STATE " ch2o_sto3g_steom.out
```
In each output, also confirm the frozen-core count line ("Number of frozen core
orbitals ... 2") and that the active space = complete (NActIP/NActEA as set).

## Decisive test = ch2o_sto3g_steom.out (lowest singlet, n->pi*)
| ORCA STEOM lowest singlet | conclusion |
|---|---|
| ~4.0 eV (== EOM-EE)        | Python/GANSU STEOM **formula is BUGGED** (under-corrects ~1.4 eV). Debug the W^eff dressing against ORCA. |
| ~2.6 eV (== Python STEOM)  | STEOM != EOM inherently; Python formula is **correct**. The 1.48 eV GANSU number is then a DLPNO/active-space issue, not the W^eff core. |

Reference numbers (my Python complete-active STEOM, same geometry/basis/frozen):
- **ch2o sto-3g** (frozen=2, NActIP6/NActEA4): Python singlet
  [2.625, 7.374, 12.666, 13.906, 14.814, 17.391, 18.017, 18.263] eV;
  EOM-EE [4.034, 9.992, 12.744, 13.513, 14.652, 15.334, 17.610, 17.679] eV.
- **h2o sto-3g** (frozen=2, NActIP3/NActEA2): Python singlet
  [10.804, 12.473, 16.105, 16.478, 22.352, 26.533] eV;
  EOM-EE [11.727, 13.761, 16.192, 18.285, 21.171, 27.565] eV.
- **ch2o cc-pVDZ** (default active): expect ORCA n->pi* ~3.9-4.0 eV (= GANSU bug target).

Paste the ORCA lowest-few singlet excitation energies back and I'll diagnose.
