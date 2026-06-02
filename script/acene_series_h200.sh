#!/bin/bash
# ============================================================================
# Acene series DLPNO-STEOM-CCSD timing reference run on H200×4
# Ship 11-15 + GPUHandle::reset() integrated (2026-06-01)
#
# Runs Benzene → Naphthalene → Anthracene → Tetracene → Pentacene in
# sequence (~1 hr 45 min total). SSH-safe: re-execs itself via nohup +
# setsid so the parent shell can disconnect freely. PID stored to
# logs/acene_series.pid for monitoring/kill.
#
# Usage:
#   ./script/acene_series_h200.sh                # detach + return prompt
#   tail -f logs/acene_series_<TS>_summary.log   # follow progress
#   kill $(cat logs/acene_series.pid)            # abort
# ============================================================================

set -u
mkdir -p logs

# --------------------------------------------------------------------------
# SSH-safe self-detach: if not already detached, re-exec under nohup+setsid
# so the script keeps running after the SSH session closes.
# --------------------------------------------------------------------------
if [ "${ACENE_DETACHED:-0}" != "1" ]; then
    TS=$(date +%Y%m%d_%H%M%S)
    SUMMARY="logs/acene_series_${TS}_summary.log"
    PIDFILE="logs/acene_series.pid"

    echo "[launcher] Detaching to background via nohup + setsid..."
    echo "[launcher] Summary log: $SUMMARY"
    echo "[launcher] PID file   : $PIDFILE"

    # Re-exec self detached. setsid creates a new session so SIGHUP from SSH
    # disconnect won't reach us; nohup is belt-and-suspenders.
    ACENE_DETACHED=1 ACENE_TS="$TS" ACENE_SUMMARY="$SUMMARY" \
        nohup setsid "$0" "$@" > /dev/null 2>&1 < /dev/null &
    CHILD_PID=$!
    echo "$CHILD_PID" > "$PIDFILE"
    disown "$CHILD_PID" 2>/dev/null || true

    echo "[launcher] Started PID=$CHILD_PID"
    echo "[launcher] Monitor: tail -f $SUMMARY"
    echo "[launcher] Kill   : kill $CHILD_PID"
    echo "[launcher] You can safely disconnect SSH now."
    exit 0
fi

# --------------------------------------------------------------------------
# Detached body: from here on we're running detached from the original TTY.
# --------------------------------------------------------------------------

TS="${ACENE_TS:-$(date +%Y%m%d_%H%M%S)}"
SUMMARY="${ACENE_SUMMARY:-logs/acene_series_${TS}_summary.log}"
PIDFILE="logs/acene_series.pid"

# Validated env chain (29 envs) — Pentacene 36-atom production set
ENV_CHAIN="\
OPENBLAS_NUM_THREADS=64 \
GANSU_EA_VVVV_NSLAB=4 \
GANSU_STEOM_OPERATOR_DEVICE_BALANCING=1 \
GANSU_DLPNO_CANONICAL_SKIP=1 \
GANSU_STEOM_BUILD_GPUS=4 \
GANSU_DLPNO_NATIVE_EOM=1 \
GANSU_DLPNO_NATIVE_BARE=1 \
GANSU_DLPNO_NATIVE_RING=1 \
GANSU_DLPNO_NATIVE_DRESSED=1 \
GANSU_DLPNO_NATIVE_GPU=1 \
GANSU_DLPNO_NATIVE_GPU_PROJ=1 \
GANSU_DLPNO_NATIVE_GPU_LIFT=1 \
GANSU_DLPNO_NATIVE_GPU_XPAIR=1 \
GANSU_DLPNO_NATIVE_GPU_RESIDENT=1 \
GANSU_DLPNO_NATIVE_GPU_T1=1 \
GANSU_DLPNO_NATIVE_GPU_T8=1 \
GANSU_DLPNO_NATIVE_GPU_PHL=1 \
GANSU_DLPNO_NATIVE_GPU_S1LOO=1 \
GANSU_DLPNO_NATIVE_GPU_S1FOV=1 \
GANSU_DLPNO_NATIVE_GPU_S1WOOOV=1 \
GANSU_DLPNO_NATIVE_GPU_PH1=1 \
GANSU_DLPNO_NATIVE_GPU_PH2=1 \
GANSU_DLPNO_NATIVE_GPU_PH3=1 \
GANSU_DLPNO_NATIVE_GPU_TMP=1 \
GANSU_DLPNO_NATIVE_GPU_TLVV=1 \
GANSU_DLPNO_NATIVE_GPU_TR1=1 \
GANSU_DLPNO_NATIVE_GPU_S1LVV=1 \
GANSU_DLPNO_NATIVE_GPU_S1WVOVV=1"

# Acenes (atoms ascending) — adjust the list if you want to skip some
MOLECULES=(
    "Benzene:12"
    "Naphthalene:18"
    "Anthracene:24"
    "Tetracene:30"
    "Pentacene:36"
)

{
echo "=========================================================="
echo "Acene series H200x4 DLPNO-STEOM-CCSD timing reference"
echo "Started : $(date)"
echo "Host    : $(hostname)"
echo "PID     : $$"
echo "Ship 11-15 + GPUHandle::reset() integrated"
echo "=========================================================="
echo ""
} > "$SUMMARY"

SERIES_START=$(date +%s)

for entry in "${MOLECULES[@]}"; do
    MOL=${entry%:*}
    ATOMS=${entry#*:}
    LOG="logs/acene_${MOL}_${TS}.log"

    {
        echo "----------------------------------------------------------"
        echo "[$(date +%H:%M:%S)] Starting $MOL ($ATOMS atoms)"
        echo "  log: $LOG"
    } >> "$SUMMARY"

    START=$(date +%s)
    eval "$ENV_CHAIN ./build/gansu \
        -x xyz/${MOL}.xyz \
        -g cc-pvdz \
        --eri_method ri \
        -ag auxiliary_basis/cc-pvdz-rifit.gbs \
        --post_hf_method dlpno_steom_ccsd \
        --num_gpus 4" \
        > "$LOG" 2>&1
    RC=$?
    END=$(date +%s)
    WALL=$((END - START))

    if [ $RC -eq 0 ]; then
        STATUS="OK"
    else
        STATUS="FAILED (rc=$RC)"
    fi

    HF=$(grep   "Total Energy:"               "$LOG" | tail -1 | awk '{print $3}')
    CORR=$(grep "E(total CCSD corr)"          "$LOG" | tail -1 | awk '{print $NF}')
    IPR0=$(grep -E "^\s+0\s+0\.[0-9]+\s+[0-9]+\.[0-9]+\s+0\.[0-9]+\s+0\.[0-9]+\s+0$" "$LOG" | head -1 | awk '{print $3}')
    EARR0=$(grep -E "^\s+0\s+0\.[0-9]+\s+[0-9]+\.[0-9]+\s+0\.[0-9]+\s+0\.[0-9]+\s+0$" "$LOG" | sed -n '2p' | awk '{print $3}')
    STEOM=$(grep -A 6 "STEOM excited-state energies" "$LOG" | tail -5 | awk '{printf "%s ", $3}')

    {
        echo "[$(date +%H:%M:%S)] $MOL done: wall=${WALL}s status=${STATUS}"
        echo "  HF total energy : $HF Ha"
        echo "  DLPNO corr      : $CORR Ha"
        echo "  IP root 0 (eV)  : $IPR0"
        echo "  EA root 0 (eV)  : $EARR0"
        echo "  STEOM 5 (eV)    : $STEOM"
        echo ""
    } >> "$SUMMARY"

    # Stop the series early on hard failure to save GPU time
    if [ $RC -ne 0 ]; then
        echo "[$(date +%H:%M:%S)] ABORTING series: $MOL failed (rc=$RC)" >> "$SUMMARY"
        break
    fi
done

SERIES_END=$(date +%s)
TOTAL=$((SERIES_END - SERIES_START))

{
echo "=========================================================="
echo "[$(date +%H:%M:%S)] Series complete: total wall = ${TOTAL}s ($(printf '%d:%02d' $((TOTAL/60)) $((TOTAL%60))))"
echo "  Summary: $SUMMARY"
echo "=========================================================="
} >> "$SUMMARY"

rm -f "$PIDFILE"
