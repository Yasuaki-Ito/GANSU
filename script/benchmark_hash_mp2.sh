#!/bin/bash
# Benchmark: Stored ERI vs Hash ERI vs Direct SCF for MP2/Hessian
# Usage: cd build && bash ../script/benchmark_hash_mp2.sh

EXEC="./HF_main"
BASIS_DIR="../basis"
AUX_DIR="../auxiliary_basis"
XYZ_DIR="../xyz"

echo "================================================================"
echo "  Hash ERI Benchmark: MP2 and Hessian"
echo "================================================================"
echo ""

# Molecules and basis sets to test
declare -a TESTS=(
    # xyz:basis:label
    "$XYZ_DIR/H2O.xyz:$BASIS_DIR/sto-3g.gbs:H2O/STO-3G"
    "$XYZ_DIR/H2O.xyz:$BASIS_DIR/cc-pvdz.gbs:H2O/cc-pVDZ"
    "$XYZ_DIR/NH3.xyz:$BASIS_DIR/cc-pvdz.gbs:NH3/cc-pVDZ"
    "$XYZ_DIR/Benzene.xyz:$BASIS_DIR/sto-3g.gbs:Benzene/STO-3G"
    "$XYZ_DIR/Benzene.xyz:$BASIS_DIR/cc-pvdz.gbs:Benzene/cc-pVDZ"
)

echo "=== MP2 Energy ==="
printf "%-25s %8s %12s %12s %12s %12s\n" "System" "nao" "Stored(ms)" "Hash(ms)" "Direct(ms)" "E_MP2"
printf "%-25s %8s %12s %12s %12s %12s\n" "-------" "---" "----------" "--------" "----------" "-----"

for test in "${TESTS[@]}"; do
    IFS=':' read -r xyz basis label <<< "$test"

    # Stored ERI MP2
    t_stored=$($EXEC -x "$xyz" -g "$basis" --post_hf_method MP2 2>&1 | grep "Computing time" | grep -oP '[\d.]+')
    e_stored=$($EXEC -x "$xyz" -g "$basis" --post_hf_method MP2 2>&1 | grep "Post-HF energy" | grep -oP '[-\d.]+')

    # Hash ERI MP2
    t_hash=$($EXEC -x "$xyz" -g "$basis" --eri_method hash --post_hf_method MP2 2>&1 | grep "Computing time" | grep -oP '[\d.]+')

    # Direct SCF MP2
    t_direct=$($EXEC -x "$xyz" -g "$basis" --eri_method direct --post_hf_method MP2 2>&1 | grep "Computing time" | grep -oP '[\d.]+')

    printf "%-25s %8s %12s %12s %12s %12s\n" "$label" "-" "${t_stored:-N/A}" "${t_hash:-N/A}" "${t_direct:-N/A}" "${e_stored:-N/A}"
done

echo ""
echo "=== Hessian ==="
printf "%-25s %12s %12s %12s\n" "System" "Stored(ms)" "Hash(ms)" "Direct(ms)"
printf "%-25s %12s %12s %12s\n" "-------" "----------" "--------" "----------"

for test in "${TESTS[@]}"; do
    IFS=':' read -r xyz basis label <<< "$test"

    # Only small systems for hessian (expensive)
    case "$label" in
        *Benzene/cc-pVDZ*) continue ;;
    esac

    t_stored=$($EXEC -x "$xyz" -g "$basis" -r hessian 2>&1 | grep "Computing time" | grep -oP '[\d.]+')
    t_hash=$($EXEC -x "$xyz" -g "$basis" --eri_method hash -r hessian 2>&1 | grep "Computing time" | grep -oP '[\d.]+')
    t_direct=""  # Direct SCF hessian not implemented

    printf "%-25s %12s %12s %12s\n" "$label" "${t_stored:-N/A}" "${t_hash:-N/A}" "${t_direct:-N/A}"
done

echo ""
echo "Done."
