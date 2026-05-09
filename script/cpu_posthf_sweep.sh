#!/bin/bash
# CPU post-HF sweep: verify that every (eri_method, post_hf_method) combo
# that is declared to support post-HF actually runs under --cpu.
#
# Usage: cd build && bash ../script/cpu_posthf_sweep.sh
#
# Each test prints a single line:
#   eri_method/post_hf_method: <status> <energy | error>
# where <status> is ✅ (success) or ❌ (failure).

set -u

EXEC="./HF_main"
B="../basis"
A="../auxiliary_basis"
X="../xyz"
MOL="$X/H2O.xyz"
# Use STO-3G to keep the full sweep tractable on CPU.  Heavy methods such as
# EOM-MP2/EOM-CCSD have O(N^7) inner kernels in their σ build, so even
# H2O/cc-pvdz (doubles_dim ≈ 9000) takes >20 minutes per Davidson iter.
# STO-3G (nao=7, doubles_dim=100) keeps the whole sweep under a few minutes
# and still exercises every CPU code path.
BAS="$B/sto-3g.gbs"
AUX="$A/cc-pvdz-rifit.gbs"
COMMON_OPTS="--cpu -x $MOL -g $BAS -m RHF --convergence_method Damping"

# Temporary log directory
LOGDIR=$(mktemp -d /tmp/gansu_cpu_posthf_sweep.XXXXXX)
trap 'rm -rf "$LOGDIR"' EXIT

# ------------------------------------------------------------
#  Helpers
# ------------------------------------------------------------
run_one() {
    local eri="$1"
    local post="$2"
    local label="${eri}/${post}"
    local log="$LOGDIR/${eri}_${post}.log"

    # Build command
    local cmd="$EXEC $COMMON_OPTS --eri_method $eri"
    case "$eri" in
        RI|direct_ri|semi_direct_ri|hash_ri)
            cmd="$cmd -ag $AUX"
            ;;
    esac
    cmd="$cmd --post_hf_method $post"

    # 5-minute wall clock per test is plenty for sto-3g across all methods.
    local timeout_s=300
    local start_time=$(date +%s)
    if timeout "$timeout_s" $cmd > "$log" 2>&1; then
        local end_time=$(date +%s)
        local dt=$((end_time - start_time))

        # Extract total energy including post-HF correction
        local energy
        energy=$(grep -E "Total Energy \(including post-HF correction\)" "$log" | tail -1 | awk '{print $(NF-1)}')

        if [ -z "$energy" ]; then
            # Excited-state methods may print differently; try first excited state line
            energy=$(grep -E "^\s*1\s+[0-9]" "$log" | head -1 | awk '{print $2" Ha ("$3" eV)"}')
        fi

        if [ -z "$energy" ]; then
            printf "%-28s  ⚠️   ran OK but no energy line  [%ds]\n" "$label" "$dt"
        else
            printf "%-28s  ✅  %s  [%ds]\n" "$label" "$energy" "$dt"
        fi
    else
        local rc=$?
        # Grab the last non-empty error-ish line for diagnostics
        local errline
        errline=$(grep -iE "error|throw|exception|assert|segfault|abort" "$log" | tail -1)
        if [ -z "$errline" ]; then
            errline=$(tail -5 "$log" | tr '\n' ' ' | cut -c1-120)
        fi
        if [ $rc -eq 124 ]; then
            printf "%-28s  ⏱   TIMEOUT (%ds)\n" "$label" "$timeout_s"
        else
            printf "%-28s  ❌  %s\n" "$label" "$errline"
        fi
    fi
}

echo "================================================================"
echo "  CPU post-HF sweep  (H2O / cc-pvdz, --cpu, Damping)"
echo "  Logs in: $LOGDIR"
echo "================================================================"

# ------------------------------------------------------------
#  Phase 1: MP2 on every eri_method that supports post-HF
# ------------------------------------------------------------
echo ""
echo "--- Phase 1: MP2 on every eri_method ---"
for eri in stored RI direct hash direct_ri semi_direct_ri hash_ri; do
    run_one "$eri" "mp2"
done

# ------------------------------------------------------------
#  Phase 2: full post-HF sweep on eri_methods that support all
# ------------------------------------------------------------
# stored: full list (including MP4, FCI)
# RI    : full list except MP4
# direct: full list
# hash  : full list
FULL_POSTHF_STORED="mp3 mp4 cc2 ccsd ccsd_t cis adc2 adc2x eom_mp2 eom_cc2 eom_ccsd"
FULL_POSTHF_RI="mp3 cc2 ccsd ccsd_t cis adc2 adc2x eom_mp2 eom_cc2 eom_ccsd"
FULL_POSTHF_DIRECT="mp3 mp4 cc2 ccsd ccsd_t cis adc2 adc2x eom_mp2 eom_cc2 eom_ccsd"
FULL_POSTHF_HASH="mp3 mp4 cc2 ccsd ccsd_t cis adc2 adc2x eom_mp2 eom_cc2 eom_ccsd"

echo ""
echo "--- Phase 2a: stored full post-HF ---"
for post in $FULL_POSTHF_STORED; do
    run_one "stored" "$post"
done

echo ""
echo "--- Phase 2b: RI full post-HF (no MP4) ---"
for post in $FULL_POSTHF_RI; do
    run_one "RI" "$post"
done

echo ""
echo "--- Phase 2c: direct full post-HF ---"
for post in $FULL_POSTHF_DIRECT; do
    run_one "direct" "$post"
done

echo ""
echo "--- Phase 2d: hash full post-HF ---"
for post in $FULL_POSTHF_HASH; do
    run_one "hash" "$post"
done

# FCI on each ERI method (sto-3g is already the global basis above)
echo ""
echo "--- Phase 3: FCI (stored/RI/direct/hash) ---"
for eri in stored RI direct hash; do
    run_one "$eri" "fci"
done

echo ""
echo "================================================================"
echo "  Done.  Full logs in $LOGDIR (kept until process exit)"
echo "================================================================"

# Keep logs around for inspection by disarming the trap
trap - EXIT
echo "Logs retained at: $LOGDIR"
