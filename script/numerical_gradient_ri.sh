#!/bin/bash
# Numerical gradient for RI-HF by central difference
# Usage: cd build && bash ../script/numerical_gradient_ri.sh

EXEC="./HF_main"
BASIS="../basis/sto-3g.gbs"
AUX="../auxiliary_basis/cc-pvdz-rifit.gbs"
XYZ="../xyz/H2O.xyz"
DELTA=0.0001  # Angstrom

# Read original coordinates
# H2O: O(0, 0, 0.127), H(0, 0.758, -0.509), H(0, -0.758, -0.509)
atoms=("O" "H" "H")
x=(0.000 0.000 0.000)
y=(0.000 0.758 -0.758)
z=(0.127 -0.509 -0.509)

get_energy() {
    local tmpxyz=$(mktemp /tmp/mol_XXXXXX.xyz)
    echo "3" > "$tmpxyz"
    echo "perturbed" >> "$tmpxyz"
    for i in 0 1 2; do
        printf "%s %16.10f %16.10f %16.10f\n" "${atoms[$i]}" "${1[$i]}" "${2[$i]}" "${3[$i]}" >> "$tmpxyz"
    done
    E=$($EXEC -x "$tmpxyz" -g "$BASIS" --eri_method ri -ag "$AUX" 2>&1 | grep "Total Energy:" | tail -1 | awk '{print $3}')
    rm -f "$tmpxyz"
    echo "$E"
}

echo "Numerical RI-HF gradient (central difference, delta=$DELTA Ang)"
echo "============================================================"

for iatom in 0 1 2; do
    for idir in 0 1 2; do
        dir_name=("x" "y" "z")

        # +delta
        xp=("${x[@]}"); yp=("${y[@]}"); zp=("${z[@]}")
        if [ $idir -eq 0 ]; then xp[$iatom]=$(echo "${x[$iatom]} + $DELTA" | bc -l); fi
        if [ $idir -eq 1 ]; then yp[$iatom]=$(echo "${y[$iatom]} + $DELTA" | bc -l); fi
        if [ $idir -eq 2 ]; then zp[$iatom]=$(echo "${z[$iatom]} + $DELTA" | bc -l); fi

        tmpxyz_p=$(mktemp /tmp/mol_XXXXXX.xyz)
        echo "3" > "$tmpxyz_p"
        echo "perturbed" >> "$tmpxyz_p"
        for i in 0 1 2; do
            printf "%s %16.10f %16.10f %16.10f\n" "${atoms[$i]}" "${xp[$i]}" "${yp[$i]}" "${zp[$i]}" >> "$tmpxyz_p"
        done
        Ep=$($EXEC -x "$tmpxyz_p" -g "$BASIS" --eri_method ri -ag "$AUX" 2>&1 | grep "Total Energy:" | tail -1 | awk '{print $3}')
        rm -f "$tmpxyz_p"

        # -delta
        xm=("${x[@]}"); ym=("${y[@]}"); zm=("${z[@]}")
        if [ $idir -eq 0 ]; then xm[$iatom]=$(echo "${x[$iatom]} - $DELTA" | bc -l); fi
        if [ $idir -eq 1 ]; then ym[$iatom]=$(echo "${y[$iatom]} - $DELTA" | bc -l); fi
        if [ $idir -eq 2 ]; then zm[$iatom]=$(echo "${z[$iatom]} - $DELTA" | bc -l); fi

        tmpxyz_m=$(mktemp /tmp/mol_XXXXXX.xyz)
        echo "3" > "$tmpxyz_m"
        echo "perturbed" >> "$tmpxyz_m"
        for i in 0 1 2; do
            printf "%s %16.10f %16.10f %16.10f\n" "${atoms[$i]}" "${xm[$i]}" "${ym[$i]}" "${zm[$i]}" >> "$tmpxyz_m"
        done
        Em=$($EXEC -x "$tmpxyz_m" -g "$BASIS" --eri_method ri -ag "$AUX" 2>&1 | grep "Total Energy:" | tail -1 | awk '{print $3}')
        rm -f "$tmpxyz_m"

        # dE/dR = (E+ - E-) / (2*delta_bohr)
        # delta in Angstrom → bohr: * 1.8897259886
        grad=$(python3 -c "
Ep=$Ep; Em=$Em; delta=$DELTA
bohr_per_ang = 1.8897259886
dR = 2.0 * delta * bohr_per_ang
print(f'{(Ep - Em) / dR:.10e}')
")
        echo "  Atom $iatom (${atoms[$iatom]}) d${dir_name[$idir]}: $grad   (E+=$Ep  E-=$Em)"
    done
done
