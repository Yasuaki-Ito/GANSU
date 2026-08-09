#!/usr/bin/env bash
# M8: ORCA STEOM-CCSD reference for the GANSU comparison.
# Generates one ORCA input per molecule (same cc-pVDZ / frozen core / 5 roots as
# GANSU) and runs them, then prints the excitation energies.
#
#   cd ~/GANSU/build          # or wherever; adjust ORCA path below
#   bash ../script/run_orca_validation.sh
#
# Requires `orca` on PATH (or set ORCA=/full/path/to/orca). ORCA must be called
# with its FULL path for parallel runs; here we keep it serial (small molecules).
set -uo pipefail

ORCA=${ORCA:-$(command -v orca)}
[ -x "$ORCA" ] || { echo "ORCA not found. Set ORCA=/path/to/orca"; exit 1; }
XYZDIR=../xyz
OUT=/tmp/orca_val; mkdir -p "$OUT"

for m in Formaldehyde C2H4 H2O; do
  # ORCA's xyz reader chokes on tab-separated columns; GANSU tolerates them.
  # Detab on copy so identical coordinates reach ORCA.
  sed 's/\t/ /g' "$XYZDIR/$m.xyz" > "$OUT/$m.xyz"
  # Conventional (non-RI) integrals: cc-pVDZ has no /J Coulomb-fitting set in
  # ORCA, and these molecules are tiny, so the exact STEOM-CCSD is trivial and
  # gives a clean reference (GANSU uses RI; the ~0.01 eV residual IS the RI err).
  cat > "$OUT/$m.inp" <<EOF
! STEOM-CCSD cc-pVDZ TightSCF
%mdci
  nroots 5
end
* xyzfile 0 1 $m.xyz
EOF
  echo ">>> ORCA STEOM-CCSD: $m"
  ( cd "$OUT" && "$ORCA" "$m.inp" > "$m.out" 2>&1 )
  # ORCA prints per-root lines containing "IROOT" and an "eV" column.
  grep -iE "IROOT|eV" "$OUT/$m.out" | grep -iE "eV" | head -12 \
    || echo "   (parse failed -- open $OUT/$m.out and read the STEOM-CCSD block)"
  echo
done
echo ">>> ORCA done. Outputs in $OUT/*.out"
