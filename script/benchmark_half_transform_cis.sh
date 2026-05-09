#!/bin/bash
# Benchmark: Stored vs Direct (half-transform) vs Hash (half-transform) for CIS
# Usage: cd build && bash ../script/benchmark_half_transform_cis.sh

EXEC="./HF_main"
BASIS_DIR="../basis"
XYZ_DIR="../xyz"

echo "================================================================"
echo "  Half-Transform CIS Benchmark"
echo "  Stored (full nao^4) vs Direct (half-transform) vs Hash (half-transform)"
echo "================================================================"
echo ""

declare -a TESTS=(
    "$XYZ_DIR/H2O.xyz:$BASIS_DIR/sto-3g.gbs:H2O/STO-3G:3"
    "$XYZ_DIR/Benzene.xyz:$BASIS_DIR/sto-3g.gbs:Benzene/STO-3G:5"
    "$XYZ_DIR/Benzene.xyz:$BASIS_DIR/cc-pvdz.gbs:Benzene/cc-pVDZ:5"
    "$XYZ_DIR/Naphthalene.xyz:$BASIS_DIR/sto-3g.gbs:Naphthalene/STO-3G:5"
    "$XYZ_DIR/Naphthalene.xyz:$BASIS_DIR/cc-pvdz.gbs:Naphthalene/cc-pVDZ:5"
)

extract_peak() {
    local val unit
    val=$(echo "$2" | grep "Peak usage:" | head -1 | awk '{print $(NF-1)}')
    unit=$(echo "$2" | grep "Peak usage:" | head -1 | awk '{print $NF}')
    if [[ "$unit" == "KB" ]]; then
        echo "$val" | awk '{printf "%.1f", $1/1024}'
    elif [[ "$unit" == "GB" ]]; then
        echo "$val" | awk '{printf "%.0f", $1*1024}'
    else
        echo "$val"
    fi
}

run_bench() {
    local out
    out=$($EXEC "$@" 2>&1)
    local rc=$?

    local n=$(echo "$out" | grep "Number of basis functions:" | awk '{print $NF}')
    local o=$(echo "$out" | grep "Number of alpha-spin electrons:" | awk '{print $NF}')
    [[ -n "$n" ]] && _nao="$n"
    [[ -n "$o" ]] && _nocc="$o"

    if echo "$out" | grep -qi "out of memory\|cudaMalloc failed\|Not enough GPU memory\|CUDA error"; then
        _time="OOM"; _mem="OOM"; _e1="OOM"
        return 1
    fi
    if [[ $rc -ne 0 ]] || ! echo "$out" | grep -q "State"; then
        _time="FAIL"; _mem="FAIL"; _e1="FAIL"
        return 1
    fi

    _time=$(echo "$out" | grep "compute_cis" | grep "microseconds" | head -1 | awk -F': ' '{print $2}' | awk '{print $1}')
    _mem=$(extract_peak "" "$out")
    _e1=$(echo "$out" | grep "    1 " | head -1 | awk '{print $2}')
    _e2=$(echo "$out" | grep "    2 " | head -1 | awk '{print $2}')
    _e3=$(echo "$out" | grep "    3 " | head -1 | awk '{print $2}')
    return 0
}

printf "%-22s %4s %4s %4s  %8s %7s  %8s %7s  %8s %7s  %10s %10s %10s\n" \
    "System" "nao" "nocc" "nvir" "Stor(ms)" "Mem MB" "Dir(ms)" "Mem MB" "Hash(ms)" "Mem MB" "E1(Ha)" "E2(Ha)" "E3(Ha)"
printf "%s\n" "--------------------------------------------------------------------------------------------------------------------------------------"

for test in "${TESTS[@]}"; do
    IFS=':' read -r xyz basis label nst <<< "$test"

    if [[ ! -f "$xyz" ]]; then
        printf "%-22s  SKIPPED\n" "$label"
        continue
    fi

    _nao="?"; _nocc="?"
    echo -n "  Running $label ..." >&2

    echo -n " stored" >&2
    run_bench -x "$xyz" -g "$basis" --post_hf_method CIS --n_excited_states "$nst"
    nao=$_nao; nocc=$_nocc; nvir="?"
    [[ "$nao" != "?" && "$nocc" != "?" ]] && nvir=$((nao - nocc))
    t_stored=$_time; m_stored=$_mem; e1s=$_e1; e2s=$_e2; e3s=$_e3

    echo -n " direct" >&2
    run_bench -x "$xyz" -g "$basis" --eri_method Direct --post_hf_method CIS --n_excited_states "$nst"
    [[ "$nao" == "?" ]] && nao=$_nao
    [[ "$nocc" == "?" ]] && nocc=$_nocc
    [[ "$nao" != "?" && "$nocc" != "?" ]] && nvir=$((nao - nocc))
    t_direct=$_time; m_direct=$_mem

    echo -n " hash" >&2
    run_bench -x "$xyz" -g "$basis" --eri_method hash --post_hf_method CIS --n_excited_states "$nst"
    [[ "$nao" == "?" ]] && nao=$_nao
    [[ "$nocc" == "?" ]] && nocc=$_nocc
    [[ "$nao" != "?" && "$nocc" != "?" ]] && nvir=$((nao - nocc))
    t_hash=$_time; m_hash=$_mem

    echo " done" >&2

    printf "%-22s %4s %4s %4s  %8s %7s  %8s %7s  %8s %7s  %10s %10s %10s\n" \
        "$label" "$nao" "$nocc" "$nvir" \
        "${t_stored:-N/A}" "${m_stored:-N/A}" \
        "${t_direct:-N/A}" "${m_direct:-N/A}" \
        "${t_hash:-N/A}" "${m_hash:-N/A}" \
        "${e1s:-N/A}" "${e2s:-N/A}" "${e3s:-N/A}"
done

echo ""
echo "Notes:"
echo "  Stored = full nao^4 MO ERI + CIS A-matrix, Direct/Hash = half-transform OVOV+OOVV"
echo "  Times = CIS only (excl. SCF), Mem = GPU peak (incl. SCF + ERI)"
echo "  OOM = out of GPU memory"
echo ""
echo "Done."
