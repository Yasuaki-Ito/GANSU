#!/bin/bash
# ============================================================================
# Tetracene + Pentacene dense-geev STEOM rerun on H200×4
#
# Adds GANSU_STEOM_DENSE_DIAG=1 to the validated 29-env Pentacene chain to
# force the dense non-Hermitian eigensolver (cusolverDnXgeev) instead of
# the auto-Davidson that kicks in for total_dim > 12000.  Davidson on the
# 23871-dim Pentacene G^{1h1p} converged to spurious negative roots due to
# D2h near-degeneracy + non-Hermitian nondeterminism — dense geev is
# deterministic and returns ALL eigenvalues sorted, so we can pick the
# lowest physical roots reliably.
#
# Extra device memory: dim^2 * 8B * ~3 transient buffers
#   Tetracene  total_dim = 16200  → 6.3 GB
#   Pentacene  total_dim = 23871  → 13.7 GB
# Both fit on the Ship 15 W_eff redirect target (~110 GB free).
#
# Wall overhead: STEOM solve 0.28s → ~10-30s (geev N^3).  Build phase
# unchanged (Ship 14 vvvv-slab + Ship 15 W_eff redirect untouched).
#
# SSH-safe: re-execs under nohup + setsid so disconnection is fine.
# Usage:
#   ./script/acene_dense_steom_h200.sh
#   tail -f logs/acene_dense_steom_<TS>_summary.log
# ============================================================================

set -u
mkdir -p logs

# --------------------------------------------------------------------------
# Self-detach via nohup + setsid (same pattern as acene_series_h200.sh)
# --------------------------------------------------------------------------
if [ "${ACENE_DENSE_DETACHED:-0}" != "1" ]; then
    TS=$(date +%Y%m%d_%H%M%S)
    SUMMARY="logs/acene_dense_steom_${TS}_summary.log"
    PIDFILE="logs/acene_dense_steom.pid"

    echo "[launcher] Detaching to background via nohup + setsid..."
    echo "[launcher] Summary log: $SUMMARY"
    echo "[launcher] PID file   : $PIDFILE"

    ACENE_DENSE_DETACHED=1 ACENE_DENSE_TS="$TS" ACENE_DENSE_SUMMARY="$SUMMARY" \
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
# Detached body
# --------------------------------------------------------------------------

TS="${ACENE_DENSE_TS:-$(date +%Y%m%d_%H%M%S)}"
SUMMARY="${ACENE_DENSE_SUMMARY:-logs/acene_dense_steom_${TS}_summary.log}"
PIDFILE="logs/acene_dense_steom.pid"

# 29-env Pentacene production chain + GANSU_STEOM_DENSE_DIAG=1 (30 total)
ENV_CHAIN="\
OPENBLAS_NUM_THREADS=64 \
GANSU_STEOM_DENSE_DIAG=1 \
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

# Just the two molecules that auto-Davidson misbehaved on
MOLECULES=(
    "Tetracene:30"
    "Pentacene:36"
)

{
echo "=========================================================="
echo "Tetracene + Pentacene dense-geev STEOM rerun on H200x4"
echo "Started : $(date)"
echo "Host    : $(hostname)"
echo "PID     : $$"
echo "Extra env: GANSU_STEOM_DENSE_DIAG=1 (forces deterministic geev)"
echo "=========================================================="
echo ""
} > "$SUMMARY"

SERIES_START=$(date +%s)

for entry in "${MOLECULES[@]}"; do
    MOL=${entry%:*}
    ATOMS=${entry#*:}
    LOG="logs/acene_dense_${MOL}_${TS}.log"

    {
        echo "----------------------------------------------------------"
        echo "[$(date +%H:%M:%S)] Starting $MOL ($ATOMS atoms) — dense geev"
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
    STEOM=$(grep -A 6 "STEOM excited-state energies" "$LOG" | tail -5 | awk '{printf "%s ", $3}')
    STEOM_TIME=$(grep "STEOM-CCSD solve time"  "$LOG" | tail -1 | awk '{print $4}')
    SOLVER=$(grep -E "dense non-Hermitian geev|iterative non-Hermitian Davidson" "$LOG" | tail -1 | sed 's/^ *//' | cut -c1-80)

    {
        echo "[$(date +%H:%M:%S)] $MOL done: wall=${WALL}s status=${STATUS}"
        echo "  HF total energy : $HF Ha"
        echo "  DLPNO corr      : $CORR Ha"
        echo "  STEOM solver    : $SOLVER"
        echo "  STEOM 5 (eV)    : $STEOM"
        echo "  STEOM solve     : ${STEOM_TIME} s"
        echo ""
    } >> "$SUMMARY"

    if [ $RC -ne 0 ]; then
        echo "[$(date +%H:%M:%S)] ABORTING series: $MOL failed (rc=$RC)" >> "$SUMMARY"
        break
    fi
done

SERIES_END=$(date +%s)
TOTAL=$((SERIES_END - SERIES_START))

{
echo "=========================================================="
echo "[$(date +%H:%M:%S)] Dense-geev rerun complete: total wall = ${TOTAL}s ($(printf '%d:%02d' $((TOTAL/60)) $((TOTAL%60))))"
echo "  Summary: $SUMMARY"
echo "=========================================================="
} >> "$SUMMARY"

rm -f "$PIDFILE"
