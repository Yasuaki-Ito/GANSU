#!/bin/bash
# ============================================================================
# Acene series ORCA 6.1.1 DLPNO-STEOM-CCSD — TightPNO accuracy preset
#
# Per ORCA 6 official documentation (DLPNO-STEOM-CCSD section):
#   "Use TightPNO for all molecules"
#   Combine with: OThresh 0.005, VThresh 0.005, TCutPNOSingles 1e-11
#   TightSCF is a must for any CCSD calculation
#
# This config drives ORCA toward canonical-limit STEOM accuracy, making the
# wall-time vs GANSU bt-PNO-STEOM (canonical dressing) a more meaningful
# "same algorithm class" comparison. NormalPNO results (acene_orca_h200.sh)
# remain as the speed-optimized baseline.
#
# Expected wall increase: ~3-5× over NormalPNO (Pentacene from 9.5 min to
# 30-60 min estimated). Still likely faster than GANSU 91 min, but the gap
# narrows as ORCA approaches canonical limit.
#
# SSH-safe: re-execs under nohup + setsid.
# Usage:
#   ./script/acene_orca_tight_h200.sh
#   tail -f logs/orca_acene_tight_<TS>_summary.log
# ============================================================================

set -u

GANSU_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$GANSU_ROOT/logs"
mkdir -p "$LOG_DIR"

ORCA_HOME="${ORCA_HOME:-$HOME/orca_6_1_1}"
ORCA_BIN="$ORCA_HOME/orca"

# --------------------------------------------------------------------------
# Self-detach via nohup + setsid
# --------------------------------------------------------------------------
if [ "${ORCA_TIGHT_DETACHED:-0}" != "1" ]; then
    TS=$(date +%Y%m%d_%H%M%S)
    SUMMARY="$LOG_DIR/orca_acene_tight_${TS}_summary.log"
    PIDFILE="$LOG_DIR/orca_acene_tight.pid"

    if [ ! -x "$ORCA_BIN" ]; then
        echo "[launcher] ERROR: ORCA binary not found: $ORCA_BIN"
        exit 1
    fi

    echo "[launcher] ORCA binary: $ORCA_BIN"
    echo "[launcher] Accuracy   : TightPNO + recommended STEOM settings"
    echo "[launcher] Detaching to background via nohup + setsid..."
    echo "[launcher] Summary log: $SUMMARY"
    echo "[launcher] PID file   : $PIDFILE"

    ORCA_TIGHT_DETACHED=1 ORCA_TIGHT_TS="$TS" ORCA_TIGHT_SUMMARY="$SUMMARY" \
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

TS="${ORCA_TIGHT_TS:-$(date +%Y%m%d_%H%M%S)}"
SUMMARY="${ORCA_TIGHT_SUMMARY:-$LOG_DIR/orca_acene_tight_${TS}_summary.log}"
PIDFILE="$LOG_DIR/orca_acene_tight.pid"

export PATH="$ORCA_HOME:$PATH"
export LD_LIBRARY_PATH="$ORCA_HOME:${LD_LIBRARY_PATH:-}"

NPROCS=64
MAXCORE=4000
NROOTS=5

SCRATCH_BASE="${ORCA_SCRATCH_BASE:-/tmp/orca_acene_tight_${TS}}"
mkdir -p "$SCRATCH_BASE"

MOLECULES=(
    "Benzene:12"
    "Naphthalene:18"
    "Anthracene:24"
    "Tetracene:30"
    "Pentacene:36"
)

XYZ_DIR="$GANSU_ROOT/xyz"

{
echo "=========================================================="
echo "Acene series ORCA 6.1.1 DLPNO-STEOM-CCSD TightPNO"
echo "Started : $(date)"
echo "Host    : $(hostname)"
echo "PID     : $$"
echo "ORCA    : $ORCA_BIN"
echo "Settings: TightPNO + TCutPNOSingles 1e-11 + OThresh/VThresh 0.005"
echo "          (ORCA-recommended STEOM-CCSD accuracy config)"
echo "Cores   : $NPROCS MPI × $MAXCORE MB"
echo "Scratch : $SCRATCH_BASE"
echo "=========================================================="
echo ""
} > "$SUMMARY"

SERIES_START=$(date +%s)

for entry in "${MOLECULES[@]}"; do
    MOL=${entry%:*}
    ATOMS=${entry#*:}
    LOG="$LOG_DIR/orca_tight_${MOL}_${TS}.log"
    SCRATCH="$SCRATCH_BASE/$MOL"
    mkdir -p "$SCRATCH"

    XYZ_PATH="$XYZ_DIR/${MOL}.xyz"
    if [ ! -f "$XYZ_PATH" ]; then
        echo "[$(date +%H:%M:%S)] ERROR: $XYZ_PATH not found, skipping $MOL" >> "$SUMMARY"
        continue
    fi

    # Normalize XYZ for ORCA
    NORM_XYZ="$SCRATCH/${MOL}.xyz"
    awk 'NR<=2 {print; next}
         NF>=4 {printf "%-2s   %12.8f   %12.8f   %12.8f\n", $1, $2+0, $3+0, $4+0; next}
         {print}' "$XYZ_PATH" > "$NORM_XYZ"

    # ORCA input file — TightPNO + STEOM-recommended accuracy
    INP="$SCRATCH/${MOL}.inp"
    cat > "$INP" <<EOF
! DLPNO-STEOM-CCSD cc-pVDZ cc-pVDZ/C TightSCF TightPNO

%pal nprocs $NPROCS end
%maxcore $MAXCORE

%mdci
  nroots $NROOTS
  TCutPNOSingles 1e-11
  OThresh 0.005
  VThresh 0.005
end

* xyzfile 0 1 $NORM_XYZ
EOF

    {
        echo "----------------------------------------------------------"
        echo "[$(date +%H:%M:%S)] Starting $MOL ($ATOMS atoms) — ORCA TightPNO"
        echo "  inp: $INP"
        echo "  log: $LOG"
    } >> "$SUMMARY"

    START=$(date +%s)
    (
        cd "$SCRATCH"
        "$ORCA_BIN" "$INP" > "$LOG" 2>&1
    )
    RC=$?
    END=$(date +%s)
    WALL=$((END - START))

    if [ $RC -eq 0 ]; then
        STATUS="OK"
    else
        STATUS="FAILED (rc=$RC)"
    fi

    HF=$(grep -E "^Total Energy       :"  "$LOG" | head -1 | awk '{print $4}')
    DLPNO=$(grep -E "E\(CORR\)\(corrected\)" "$LOG" | tail -1 | awk '{print $NF}')
    STEOM=$(grep -A 80 "STEOM-CCSD RESULTS (SINGLETS)" "$LOG" \
            | grep -E "^IROOT=" | head -$NROOTS | awk '{print $4}' | tr '\n' ' ')
    ORCA_WALL=$(grep "TOTAL RUN TIME:" "$LOG" | tail -1 | sed 's/^.*TIME://')

    {
        echo "[$(date +%H:%M:%S)] $MOL done: wall=${WALL}s status=${STATUS}"
        echo "  HF total energy : $HF Ha"
        echo "  DLPNO corr      : $DLPNO Ha"
        echo "  STEOM 5 (eV)    : $STEOM"
        echo "  ORCA-reported   :$ORCA_WALL"
        echo ""
    } >> "$SUMMARY"

    if [ $RC -ne 0 ]; then
        echo "[$(date +%H:%M:%S)] ABORTING series: $MOL failed (rc=$RC). See $LOG for details." >> "$SUMMARY"
        break
    fi
done

SERIES_END=$(date +%s)
TOTAL=$((SERIES_END - SERIES_START))

{
echo "=========================================================="
echo "[$(date +%H:%M:%S)] ORCA TightPNO acene series complete: total wall = ${TOTAL}s ($(printf '%d:%02d' $((TOTAL/60)) $((TOTAL%60))))"
echo "  Summary: $SUMMARY"
echo "  Scratch: $SCRATCH_BASE"
echo "=========================================================="
} >> "$SUMMARY"

rm -f "$PIDFILE"
