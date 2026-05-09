#!/bin/bash
# Benchmark: Stored vs Direct (half-transform) vs Hash (half-transform) for MP3
# Usage: cd build && bash ../script/benchmark_half_transform_mp3.sh

EXEC="./HF_main"
BASIS_DIR="../basis"
XYZ_DIR="../xyz"
LARGE_DIR="../xyz/large_molecular"

echo "================================================================"
echo "  Half-Transform MP3 Benchmark"
echo "  Stored (full nao^4) vs Direct (half-transform) vs Hash (half-transform)"
echo "================================================================"
echo ""

declare -a TESTS=(
    # Small
    "$XYZ_DIR/H2O.xyz:$BASIS_DIR/sto-3g.gbs:H2O/STO-3G"
    "$XYZ_DIR/H2O.xyz:$BASIS_DIR/cc-pvdz.gbs:H2O/cc-pVDZ"
    "$XYZ_DIR/Benzene.xyz:$BASIS_DIR/sto-3g.gbs:Benzene/STO-3G"
    # Medium
    "$XYZ_DIR/Ethanol.xyz:$BASIS_DIR/cc-pvdz.gbs:Ethanol/cc-pVDZ"
    "$XYZ_DIR/Naphthalene.xyz:$BASIS_DIR/sto-3g.gbs:Naphthalene/STO-3G"
    "$XYZ_DIR/Benzene.xyz:$BASIS_DIR/cc-pvdz.gbs:Benzene/cc-pVDZ"
    # Large (Stored may OOM, half-transform should work)
    "$LARGE_DIR/Cholesterol.xyz:$BASIS_DIR/sto-3g.gbs:Cholesterol/STO-3G"
    "$LARGE_DIR/VitaminE.xyz:$BASIS_DIR/sto-3g.gbs:VitaminE/STO-3G"
    "$LARGE_DIR/Doxorubicin.xyz:$BASIS_DIR/sto-3g.gbs:Doxorubicin/STO-3G"
    "$XYZ_DIR/Naphthalene.xyz:$BASIS_DIR/cc-pvdz.gbs:Naphthalene/cc-pVDZ"
    "$XYZ_DIR/Anthracene.xyz:$BASIS_DIR/cc-pvdz.gbs:Anthracene/cc-pVDZ"
)

# Run a single benchmark. Sets: _time, _mem, _energy, _nao, _nocc
run_bench() {
    local out
    out=$($EXEC "$@" --post_hf_method MP3 2>&1)
    local rc=$?

    # Always try to extract nao/nocc (printed before OOM)
    local n=$(echo "$out" | grep "Number of basis functions:" | awk '{print $NF}')
    local o=$(echo "$out" | grep "Number of alpha-spin electrons:" | awk '{print $NF}')
    [[ -n "$n" ]] && _nao="$n"
    [[ -n "$o" ]] && _nocc="$o"

    # Check for OOM or error
    if echo "$out" | grep -qi "out of memory\|cudaMalloc failed\|Not enough GPU memory\|CUDA error\|cudaErrorMemoryAllocation"; then
        _time="OOM"; _mem="OOM"; _energy="OOM"
        return 1
    fi
    if [[ $rc -ne 0 ]] || ! echo "$out" | grep -q "Post-HF energy correction"; then
        _time="FAIL"; _mem="FAIL"; _energy="FAIL"
        return 1
    fi

    _energy=$(echo "$out" | grep "Post-HF energy correction:" | head -1 | awk '{print $(NF-1)}')
    _time=$(echo "$out" | grep "compute_mp3_energy" | grep "microseconds" | head -1 | awk -F': ' '{print $2}' | awk '{print $1}')

    local peak_val peak_unit
    peak_val=$(echo "$out" | grep "Peak usage:" | head -1 | awk '{print $(NF-1)}')
    peak_unit=$(echo "$out" | grep "Peak usage:" | head -1 | awk '{print $NF}')
    if [[ "$peak_unit" == "KB" ]]; then
        _mem=$(echo "$peak_val" | awk '{printf "%.1f", $1/1024}')
    elif [[ "$peak_unit" == "GB" ]]; then
        _mem=$(echo "$peak_val" | awk '{printf "%.0f", $1*1024}')
    else
        _mem="$peak_val"
    fi
    return 0
}

# Header
printf "%-22s %4s %4s %4s  %8s %7s  %8s %7s  %8s %7s  %10s\n" \
    "System" "nao" "nocc" "nvir" "Stor(ms)" "Mem MB" "Dir(ms)" "Mem MB" "Hash(ms)" "Mem MB" "max|dE|"
printf "%s\n" "--------------------------------------------------------------------------------------------------------------"

for test in "${TESTS[@]}"; do
    IFS=':' read -r xyz basis label <<< "$test"

    if [[ ! -f "$xyz" ]]; then
        printf "%-22s  SKIPPED (file not found)\n" "$label"
        continue
    fi

    # Reset nao/nocc for each system
    _nao="?"; _nocc="?"

    echo -n "  Running $label ..." >&2

    # --- Stored ---
    echo -n " stored" >&2
    run_bench -x "$xyz" -g "$basis"
    nao=$_nao; nocc=$_nocc
    nvir="?"
    [[ "$nao" != "?" && "$nocc" != "?" ]] && nvir=$((nao - nocc))
    e_stored=$_energy; t_stored=$_time; m_stored=$_mem

    # --- Direct ---
    echo -n " direct" >&2
    run_bench -x "$xyz" -g "$basis" --eri_method Direct
    [[ "$nao" == "?" ]] && nao=$_nao
    [[ "$nocc" == "?" ]] && nocc=$_nocc
    [[ "$nao" != "?" && "$nocc" != "?" ]] && nvir=$((nao - nocc))
    e_direct=$_energy; t_direct=$_time; m_direct=$_mem

    # --- Hash ---
    echo -n " hash" >&2
    run_bench -x "$xyz" -g "$basis" --eri_method hash
    [[ "$nao" == "?" ]] && nao=$_nao
    [[ "$nocc" == "?" ]] && nocc=$_nocc
    [[ "$nao" != "?" && "$nocc" != "?" ]] && nvir=$((nao - nocc))
    e_hash=$_energy; t_hash=$_time; m_hash=$_mem

    echo " done" >&2

    # Compute max |diff| vs Stored (or between Direct/Hash)
    max_diff="-"
    if [[ "$e_stored" != "OOM" && "$e_stored" != "FAIL" ]]; then
        vals=""
        [[ "$e_direct" != "OOM" && "$e_direct" != "FAIL" ]] && vals="abs($e_direct-($e_stored))"
        [[ "$e_hash" != "OOM" && "$e_hash" != "FAIL" ]] && vals="${vals:+$vals,}abs($e_hash-($e_stored))"
        [[ -n "$vals" ]] && max_diff=$(python3 -c "print(f'{max($vals):.1e}')" 2>/dev/null || echo "N/A")
    elif [[ "$e_direct" != "OOM" && "$e_direct" != "FAIL" && "$e_hash" != "OOM" && "$e_hash" != "FAIL" ]]; then
        max_diff=$(python3 -c "print(f'{abs($e_direct-($e_hash)):.1e}')" 2>/dev/null || echo "N/A")
    fi

    printf "%-22s %4s %4s %4s  %8s %7s  %8s %7s  %8s %7s  %10s\n" \
        "$label" "$nao" "$nocc" "$nvir" \
        "${t_stored:-N/A}" "${m_stored:-N/A}" \
        "${t_direct:-N/A}" "${m_direct:-N/A}" \
        "${t_hash:-N/A}" "${m_hash:-N/A}" \
        "${max_diff}"
done

echo ""
echo "Notes:"
echo "  Stored = full nao^4 MO ERI, Direct/Hash = half-transform (no nao^4)"
echo "  Times = MP3 only (excl. SCF), Mem = GPU peak (incl. SCF + ERI)"
echo "  OOM = out of GPU memory"
echo ""
echo "Done."
