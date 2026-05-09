#!/bin/bash
# Benchmark: Stored RI / Hash RI / Semi-Direct RI / Direct RI for MP2
# Usage: cd build && bash ../script/benchmark_ri_mp2.sh

EXEC="./HF_main"
B="../basis"
A="../auxiliary_basis"
X="../xyz"

echo "================================================================"
echo "  RI MP2 Benchmark (4 methods)"
echo "================================================================"
echo ""

for mol in "H2O:$X/H2O.xyz:$B/sto-3g.gbs:$A/cc-pvdz-rifit.gbs" \
           "H2O:$X/H2O.xyz:$B/cc-pvdz.gbs:$A/cc-pvdz-rifit.gbs" \
           "Benzene:$X/Benzene.xyz:$B/sto-3g.gbs:$A/cc-pvdz-rifit.gbs" \
           "Benzene:$X/Benzene.xyz:$B/cc-pvdz.gbs:$A/cc-pvdz-rifit.gbs" \
           "Naphthalene:$X/Naphthalene.xyz:$B/sto-3g.gbs:$A/cc-pvdz-rifit.gbs" \
           "Naphthalene:$X/Naphthalene.xyz:$B/cc-pvdz.gbs:$A/cc-pvdz-rifit.gbs" \
           "Indigo:$X/Indigo.xyz:$B/sto-3g.gbs:$A/cc-pvdz-rifit.gbs" \
           "Indigo:$X/Indigo.xyz:$B/cc-pvdz.gbs:$A/cc-pvdz-rifit.gbs"; do

    IFS=':' read -r name xyz basis aux <<< "$mol"
    bname=$(basename $basis .gbs)
    sys="$name/$bname"

    echo "--- $sys ---"
    printf "%-20s %10s %12s %s\n" "Method" "Time(s)" "Peak Mem" "E_MP2"

    for method in "Stored RI:ri" "Hash RI:hash_ri" "Semi-Direct RI:semi_direct_ri" "Direct RI:direct_ri"; do
        IFS=':' read -r mlabel mopt <<< "$method"

        start=$(date +%s%N)
        result=$($EXEC -x "$xyz" -g "$basis" --eri_method "$mopt" -ag "$aux" --post_hf_method MP2 2>&1)
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
