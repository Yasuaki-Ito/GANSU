#!/bin/bash
# ============================================================================
# Acene series DLPNO-STEOM-CCSD timing — ORCA 6.1.1 reference run
#
# Mirror of script/acene_series_h200.sh (GANSU) using ORCA 6 to provide a
# direct CPU-baseline comparison for the H200×4 multi-GPU GANSU timings.
#
# Settings match GANSU as closely as possible:
#   - DLPNO-STEOM-CCSD with cc-pVDZ AO + cc-pVDZ/C correlation auxiliary
#   - DLPNO defaults = "Normal" (TCutPNO 3.33e-7, TCutPairs 1e-4, TCutDO 1e-2)
#   - n_excited_states = 5  (matches GANSU's STEOM nroots = 5)
#   - TightSCF (1e-8 SCF convergence — note: GANSU uses 1e-6, ORCA can't go
#     to 1e-6 with TightSCF, but the energy difference is well below relevant)
#
# Hardware: ORCA is CPU-MPI only (no GPU acceleration in STEOM path).
# Uses 64 MPI procs × 4 GB = 256 GB RAM. Adjust if box is differently sized.
#
# SSH-safe: re-execs under nohup + setsid.
# Usage:
#   ./script/acene_orca_h200.sh
#   tail -f logs/orca_acene_<TS>_summary.log
# ============================================================================

set -u

# Absolute paths so the per-molecule "cd $SCRATCH" doesn't break logging.
GANSU_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$GANSU_ROOT/logs"
mkdir -p "$LOG_DIR"

ORCA_HOME="${ORCA_HOME:-$HOME/orca_6_1_1}"
ORCA_BIN="$ORCA_HOME/orca"

# Sanity check (deferred to detached phase so launcher prints nice error)
# --------------------------------------------------------------------------
# Self-detach via nohup + setsid
# --------------------------------------------------------------------------
if [ "${ORCA_DETACHED:-0}" != "1" ]; then
    TS=$(date +%Y%m%d_%H%M%S)
    SUMMARY="$LOG_DIR/orca_acene_${TS}_summary.log"
    PIDFILE="$LOG_DIR/orca_acene.pid"

    if [ ! -x "$ORCA_BIN" ]; then
        echo "[launcher] ERROR: ORCA binary not found or not executable: $ORCA_BIN"
        echo "[launcher] Set ORCA_HOME env or fix the path in this script (line 17)."
        exit 1
    fi

    echo "[launcher] ORCA binary: $ORCA_BIN"
    echo "[launcher] Detaching to background via nohup + setsid..."
    echo "[launcher] Summary log: $SUMMARY"
    echo "[launcher] PID file   : $PIDFILE"

    ORCA_DETACHED=1 ORCA_TS="$TS" ORCA_SUMMARY="$SUMMARY" \
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

TS="${ORCA_TS:-$(date +%Y%m%d_%H%M%S)}"
SUMMARY="${ORCA_SUMMARY:-$LOG_DIR/orca_acene_${TS}_summary.log}"
PIDFILE="$LOG_DIR/orca_acene.pid"

# ORCA needs its own dir + bundled OpenMPI on PATH
export PATH="$ORCA_HOME:$PATH"
export LD_LIBRARY_PATH="$ORCA_HOME:${LD_LIBRARY_PATH:-}"

# ORCA settings
NPROCS=64
MAXCORE=4000       # MB per MPI process (4 GB × 64 = 256 GB total)
NROOTS=5

# Per-mol scratch base — ORCA writes many temp files
SCRATCH_BASE="${ORCA_SCRATCH_BASE:-/tmp/orca_acene_${TS}}"
mkdir -p "$SCRATCH_BASE"

# Acene series (atoms ascending) — same as GANSU acene_series
MOLECULES=(
    "Benzene:12"
    "Naphthalene:18"
    "Anthracene:24"
    "Tetracene:30"
    "Pentacene:36"
)

# XYZ files location (absolute path so ORCA scratch dir can find them)
XYZ_DIR="$GANSU_ROOT/xyz"

{
echo "=========================================================="
echo "Acene series ORCA 6.1.1 DLPNO-STEOM-CCSD reference"
echo "Started : $(date)"
echo "Host    : $(hostname)"
echo "PID     : $$"
echo "ORCA    : $ORCA_BIN"
echo "Cores   : $NPROCS MPI × $MAXCORE MB"
echo "Scratch : $SCRATCH_BASE"
echo "=========================================================="
echo ""
} > "$SUMMARY"

SERIES_START=$(date +%s)

for entry in "${MOLECULES[@]}"; do
    MOL=${entry%:*}
    ATOMS=${entry#*:}
    LOG="$LOG_DIR/orca_${MOL}_${TS}.log"
    SCRATCH="$SCRATCH_BASE/$MOL"
    mkdir -p "$SCRATCH"

    XYZ_PATH="$XYZ_DIR/${MOL}.xyz"
    if [ ! -f "$XYZ_PATH" ]; then
        echo "[$(date +%H:%M:%S)] ERROR: $XYZ_PATH not found, skipping $MOL" >> "$SUMMARY"
        continue
    fi

    # Normalize XYZ for ORCA: TAB → space, ensure all coords are float with
    # decimal (Naphthalene/Anthracene have TAB separators; Naphthalene also
    # has bare integer "0" which ORCA's parser rejects as "y coord 0").
    NORM_XYZ="$SCRATCH/${MOL}.xyz"
    awk 'NR<=2 {print; next}
         NF>=4 {printf "%-2s   %12.8f   %12.8f   %12.8f\n", $1, $2+0, $3+0, $4+0; next}
         {print}' "$XYZ_PATH" > "$NORM_XYZ"

    # ORCA input file.  Uses ORCA's default cc-pVDZ convention (spherical d,
    # 14 BF per C, contraction pattern {881/31/1}).  GANSU uses Cartesian-d
    # cc-pVDZ (15 BF per C, ~5 mHa lower HF) — accepted as basis convention
    # difference for this wall-time comparison study.
    INP="$SCRATCH/${MOL}.inp"
    cat > "$INP" <<EOF
! DLPNO-STEOM-CCSD cc-pVDZ cc-pVDZ/C TightSCF

%pal nprocs $NPROCS end
%maxcore $MAXCORE

%mdci
  nroots $NROOTS
end

* xyzfile 0 1 $NORM_XYZ
EOF

    {
        echo "----------------------------------------------------------"
        echo "[$(date +%H:%M:%S)] Starting $MOL ($ATOMS atoms) — ORCA"
        echo "  inp: $INP"
        echo "  log: $LOG"
    } >> "$SUMMARY"

    START=$(date +%s)
    (
        cd "$SCRATCH"
        # ORCA needs to be called with full path so MPI children find each other
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

    # ORCA output extraction (best-effort regex; ORCA 6 STEOM section format)
    HF=$(grep -E "^Total Energy       :"  "$LOG" | head -1 | awk '{print $4}')
    DLPNO=$(grep -E "E\(CORR\)\(corrected\)" "$LOG" | tail -1 | awk '{print $NF}')
    # ORCA STEOM-CCSD prints a "STEOM-CCSD RESULTS" section with eV column
    STEOM=$(awk '/STEOM-CCSD RESULTS/,/----/' "$LOG" \
            | grep -E "^\s+[0-9]+\s+-?[0-9]+\.[0-9]+\s+-?[0-9]+\.[0-9]+" \
            | head -$NROOTS | awk '{printf "%s ", $3}')
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
echo "[$(date +%H:%M:%S)] ORCA acene series complete: total wall = ${TOTAL}s ($(printf '%d:%02d' $((TOTAL/60)) $((TOTAL%60))))"
echo "  Summary: $SUMMARY"
echo "  Scratch: $SCRATCH_BASE  (manual cleanup if needed)"
echo "=========================================================="
} >> "$SUMMARY"

rm -f "$PIDFILE"
