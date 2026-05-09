#!/bin/bash
# Full benchmark: Stored / Hash(3) / Direct for MP2
# Usage: cd build && bash ../script/benchmark_hash_mp2_full.sh

EXEC="./HF_main"
B="../basis"
X="../xyz"

run_mp2() {
    local label=$1; shift
    local result=$($EXEC "$@" 2>&1)
    local peak=$(echo "$result" | grep "Peak usage" | grep -oP '[\d.]+\s*[GMKB]+' | head -1)
    local e_mp2=$(echo "$result" | grep "Post-HF energy correction" | grep -oP '[-\d.]+')
    echo "$label|$peak|$e_mp2"
}

echo "================================================================"
echo "  Hash ERI MP2 Benchmark (time + memory)"
echo "================================================================"
echo ""

for mol in "H2O:$X/H2O.xyz:$B/sto-3g.gbs" \
           "H2O:$X/H2O.xyz:$B/cc-pvdz.gbs" \
           "Benzene:$X/Benzene.xyz:$B/sto-3g.gbs" \
           "Benzene:$X/Benzene.xyz:$B/cc-pvdz.gbs" \
           "Naphthalene:$X/Naphthalene.xyz:$B/sto-3g.gbs" \
           "Naphthalene:$X/Naphthalene.xyz:$B/cc-pvdz.gbs" \
           "Indigo:$X/Indigo.xyz:$B/sto-3g.gbs" \
           "Indigo:$X/Indigo.xyz:$B/cc-pvdz.gbs"; do

    IFS=':' read -r name xyz basis <<< "$mol"
    bname=$(basename $basis .gbs)
    sys="$name/$bname"

    echo "--- $sys ---"
    printf "%-20s %10s %12s %s\n" "Method" "Time(s)" "Peak Mem" "E_MP2"

    for method in "Stored:--post_hf_method MP2" \
                  "Hash/Compact:--eri_method hash --hash_fock_method compact --post_hf_method MP2" \
                  "Hash/Indexed:--eri_method hash --hash_fock_method indexed --post_hf_method MP2" \
                  "Hash/Fullscan:--eri_method hash --hash_fock_method fullscan --post_hf_method MP2" \
                  "Direct:--eri_method direct --post_hf_method MP2"; do

        IFS=':' read -r mlabel mopts <<< "$method"

        start=$(date +%s%N)
        result=$($EXEC -x "$xyz" -g "$basis" $mopts 2>&1)
        end=$(date +%s%N)
        elapsed=$(echo "scale=2; ($end - $start) / 1000000000" | bc)

        peak=$(echo "$result" | grep "Peak usage" | grep -oP '[\d.]+\s*[A-Z]+' | head -1)
        e_mp2=$(echo "$result" | grep "Post-HF energy correction" | grep -oP '[-\d.]+' | head -1)

        if [ -z "$e_mp2" ]; then
            e_mp2="FAILED/OOM"
            peak="N/A"
        fi

        printf "%-20s %10s %12s %s\n" "$mlabel" "${elapsed}s" "$peak" "$e_mp2"
    done
    echo ""
done
