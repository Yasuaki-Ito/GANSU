#!/bin/bash
# =============================================================================
# Benchmark: Excited state calculations (GANSU GPU/CPU)
#
# Measures wall time and excitation energies for CIS, ADC(2), EOM-CCSD
# across the acene series for scaling analysis.
#
# Usage:
#   cd build
#   bash ../script/benchmark_excited_states.sh gpu   # GPU mode
#   bash ../script/benchmark_excited_states.sh cpu   # CPU mode
# =============================================================================

# Do not exit on error — some calculations may fail (OOM, timeout) and we want to continue
# set -e

MODE=${1:-gpu}
NSTATES=3
TIMEOUT=300   # 5 minutes per calculation
OUTDIR="../benchmark_results"
mkdir -p "$OUTDIR"

if [ "$MODE" = "cpu" ]; then
    CPU_FLAG="--cpu"
    TAG="cpu"
else
    CPU_FLAG=""
    TAG="gpu"
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_FILE="$OUTDIR/excited_states_${TAG}_${TIMESTAMP}.csv"

echo "mode,molecule,basis,method,solver,natoms,nbasis,n_states,time_total_sec,time_posthf_sec,excitation_energies_eV" > "$RESULT_FILE"

# =============================================================================
# Helper: run GANSU and extract timing + excitation energies
# =============================================================================
run_gansu() {
    local mol_name="$1"
    local xyz_file="$2"
    local basis="$3"
    local method="$4"
    local solver="${5:-}"   # optional solver (schur_static, schur_omega, full)

    local natoms=$(head -1 "$xyz_file")

    local solver_label="${solver:-default}"
    local solver_flag=""
    if [ -n "$solver" ]; then
        # Map method to solver parameter name
        case "$method" in
            adc2)      solver_flag="--adc2_solver $solver" ;;
            eom_mp2)   solver_flag="--eom_mp2_solver $solver" ;;
            eom_cc2)   solver_flag="--eom_cc2_solver $solver" ;;
        esac
    fi

    echo "  $mol_name / $basis / $method ($solver_label) ..."

    local tmpout=$(mktemp)
    local start_time=$(date +%s%N)

    # Run with timeout
    timeout $TIMEOUT ./HF_main -x "$xyz_file" -g "../basis/${basis}.gbs" \
         --post_hf_method "$method" --n_excited_states "$NSTATES" \
         $solver_flag $CPU_FLAG > "$tmpout" 2>&1
    local exit_code=$?

    if [ $exit_code -eq 124 ]; then
        echo "    -> TIMEOUT (>${TIMEOUT}s)"
        echo "$TAG,$mol_name,$basis,$method,$solver_label,$natoms,,$NSTATES,TIMEOUT,," >> "$RESULT_FILE"
    elif [ $exit_code -eq 0 ]; then
        local end_time=$(date +%s%N)
        local elapsed=$(echo "scale=3; ($end_time - $start_time) / 1000000000" | bc)

        # Extract number of basis functions
        local nbasis=$(grep "Number of basis functions:" "$tmpout" | head -1 | awk '{print $NF}')

        # Extract excitation energies (eV) from the excited state table
        local energies=$(awk '/^[[:space:]]+[0-9]+[[:space:]]+[0-9]/ {printf "%s;", $3}' "$tmpout" | head -c -1)

        # Extract post-HF time from output (ADC(2) time, EOM-MP2 time, etc.)
        local posthf_time=$(grep -oP '(?:ADC\(2\)|EOM-MP2|EOM-CC2|EOM-CCSD|CIS)\s+time:\s+\K[\d.]+' "$tmpout" | head -1)
        # Fallback: try "Computing time: NNN [ms]" and convert to seconds
        if [ -z "$posthf_time" ]; then
            local computing_ms=$(grep -oP 'Computing time:\s+\K\d+' "$tmpout" | tail -1)
            if [ -n "$computing_ms" ]; then
                posthf_time=$(echo "scale=3; $computing_ms / 1000" | bc)
            fi
        fi
        posthf_time=${posthf_time:-$elapsed}

        echo "$TAG,$mol_name,$basis,$method,$solver_label,$natoms,$nbasis,$NSTATES,$elapsed,$posthf_time,$energies" >> "$RESULT_FILE"
        echo "    -> total=${elapsed}s, posthf=${posthf_time}s, E = $energies eV"
    else
        echo "    -> FAILED (exit=$exit_code)"
        echo "$TAG,$mol_name,$basis,$method,$solver_label,$natoms,,$NSTATES,FAILED,," >> "$RESULT_FILE"
    fi

    rm -f "$tmpout"
}

# =============================================================================
# Part 1: Accuracy (H2O with multiple methods and bases)
# =============================================================================
echo "=== Part 1: Accuracy benchmark (H2O) ==="

for basis in sto-3g cc-pvdz; do
    for method in cis adc2 eom_ccsd; do
        run_gansu "H2O" "../xyz/H2O.xyz" "$basis" "$method"
    done
done

# =============================================================================
# Part 2: Solver comparison (ADC(2), EOM-MP2, EOM-CC2)
# =============================================================================
echo ""
echo "=== Part 2: Solver comparison (Schur vs Full) ==="

SOLVERS="schur_static schur_omega full"

for basis in sto-3g cc-pvdz; do
    echo "--- ADC(2) / $basis ---"
    for solver in $SOLVERS; do
        run_gansu "H2O" "../xyz/H2O.xyz" "$basis" "adc2" "$solver"
    done

    echo "--- EOM-MP2 / $basis ---"
    for solver in $SOLVERS; do
        run_gansu "H2O" "../xyz/H2O.xyz" "$basis" "eom_mp2" "$solver"
    done

    echo "--- EOM-CC2 / $basis ---"
    for solver in $SOLVERS; do
        run_gansu "H2O" "../xyz/H2O.xyz" "$basis" "eom_cc2" "$solver"
    done
done

# Solver scaling: larger molecules (skip schur_omega — too slow for large systems)
SOLVERS_FAST="schur_static full"
for mol in Benzene Naphthalene; do
    xyz="../xyz/${mol}.xyz"
    [ -f "$xyz" ] || continue
    echo "--- ADC(2) solver comparison: $mol ---"
    for solver in $SOLVERS_FAST; do
        run_gansu "$mol" "$xyz" "sto-3g" "adc2" "$solver"
    done
    echo "--- EOM-MP2 solver comparison: $mol ---"
    for solver in $SOLVERS_FAST; do
        run_gansu "$mol" "$xyz" "sto-3g" "eom_mp2" "$solver"
    done
done

# =============================================================================
# Part 3: Scaling (acene series)
# =============================================================================
echo ""
echo "=== Part 3: Scaling benchmark (acene series) ==="

# --- CIS (cheapest, largest molecules) ---
echo "--- CIS ---"
for mol in Benzene Naphthalene Anthracene Tetracene Pentacene; do
    xyz="../xyz/${mol}.xyz"
    [ -f "$xyz" ] || continue
    run_gansu "$mol" "$xyz" "sto-3g" "cis"
done
# cc-pVDZ: limit to Naphthalene (Anthracene+ exceeds stored ERI memory)
for mol in Benzene Naphthalene; do
    xyz="../xyz/${mol}.xyz"
    [ -f "$xyz" ] || continue
    run_gansu "$mol" "$xyz" "cc-pvdz" "cis"
done

# --- ADC(2) (moderate cost) ---
echo "--- ADC(2) ---"
for mol in Benzene Naphthalene Anthracene Tetracene; do
    xyz="../xyz/${mol}.xyz"
    [ -f "$xyz" ] || continue
    run_gansu "$mol" "$xyz" "sto-3g" "adc2"
done
for mol in Benzene Naphthalene; do
    xyz="../xyz/${mol}.xyz"
    [ -f "$xyz" ] || continue
    run_gansu "$mol" "$xyz" "cc-pvdz" "adc2"
done

# --- EOM-MP2 ---
echo "--- EOM-MP2 ---"
for mol in Benzene Naphthalene; do
    xyz="../xyz/${mol}.xyz"
    [ -f "$xyz" ] || continue
    run_gansu "$mol" "$xyz" "sto-3g" "eom_mp2"
done
run_gansu "Benzene" "../xyz/Benzene.xyz" "cc-pvdz" "eom_mp2"

# --- EOM-CCSD (expensive, small molecules only) ---
echo "--- EOM-CCSD ---"
for mol in Benzene Naphthalene; do
    xyz="../xyz/${mol}.xyz"
    [ -f "$xyz" ] || continue
    run_gansu "$mol" "$xyz" "sto-3g" "eom_ccsd"
done
run_gansu "Benzene" "../xyz/Benzene.xyz" "cc-pvdz" "eom_ccsd"

echo ""
echo "Results saved to: $RESULT_FILE"
