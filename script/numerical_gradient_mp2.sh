#!/bin/bash
# Numerical gradient of MP2 energy for H2O/STO-3G
# Usage: cd build && bash ../script/numerical_gradient_mp2.sh

EXEC="./HF_main"
BASIS="../basis/sto-3g.gbs"
DELTA_BOHR=0.001
BOHR2ANG=0.52917721090300
DELTA_ANG=$(python3 -c "print(f'{$DELTA_BOHR * $BOHR2ANG:.15f}')")

# Reference geometry (Angstrom)
COORDS=(
    0.000  0.000  0.127    # O
    0.000  0.758 -0.509    # H
    0.000 -0.758 -0.509    # H
)
LABELS=("O" "H1" "H2")
DIRS=("x" "y" "z")

TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

make_xyz() {
    printf "3\nWater\nO  %.15f  %.15f  %.15f\nH  %.15f  %.15f  %.15f\nH  %.15f  %.15f  %.15f\n" \
        "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" > "${10}"
}

run_mp2() {
    local out=$($EXEC -x "$1" -g "$BASIS" -m RHF --post_hf_method MP2 -r energy 2>&1)
    echo "$out" | grep "Total Energy (including post-HF" | grep -oP '[-0-9]+\.[0-9]+'
}

echo "Delta = $DELTA_BOHR Bohr = $DELTA_ANG Angstrom"
echo ""

# Reference energy
make_xyz ${COORDS[@]} "$TMPDIR/ref.xyz"
EREF=$(run_mp2 "$TMPDIR/ref.xyz")
echo "E_ref = $EREF"
echo ""

printf "%-5s %-3s  %22s  %22s  %20s\n" "Atom" "Dir" "E(+delta)" "E(-delta)" "dE/dX (Ha/Bohr)"
printf "%-5s %-3s  %22s  %22s  %20s\n" "----" "---" "---------" "---------" "---------------"

for iatom in 0 1 2; do
    for idir in 0 1 2; do
        idx=$((iatom * 3 + idir))

        # Plus displacement
        C_PLUS=("${COORDS[@]}")
        C_PLUS[$idx]=$(python3 -c "print(f'{${COORDS[$idx]} + $DELTA_ANG:.15f}')")
        make_xyz "${C_PLUS[@]}" "$TMPDIR/plus.xyz"
        EP=$(run_mp2 "$TMPDIR/plus.xyz")

        # Minus displacement
        C_MINUS=("${COORDS[@]}")
        C_MINUS[$idx]=$(python3 -c "print(f'{${COORDS[$idx]} - $DELTA_ANG:.15f}')")
        make_xyz "${C_MINUS[@]}" "$TMPDIR/minus.xyz"
        EM=$(run_mp2 "$TMPDIR/minus.xyz")

        GRAD=$(python3 -c "print(f'{($EP - $EM) / (2.0 * $DELTA_BOHR):.12f}')")
        printf "%-5s %-3s  %22.15f  %22.15f  %20s\n" "${LABELS[$iatom]}" "${DIRS[$idir]}" "$EP" "$EM" "$GRAD"
    done
done

echo ""
echo "Compare with analytical gradient from: $EXEC -x ../xyz/H2O.xyz -g $BASIS -r gradient --post_hf_method MP2"
