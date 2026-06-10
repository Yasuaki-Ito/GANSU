#!/usr/bin/env bash
# One-shot ORCA runner: give it an xyz file + basis set and it builds the ORCA
# input, runs it, and reports the wall / ORCA-reported run time. Companion to the
# GANSU vs ORCA comparison (see run_orca.sh for the size-series sweep).
#
#   ORCA=/path/to/orca_root  bash bench/run_orca_one.sh -x H2O.xyz -g cc-pVDZ
#
# Required:
#   -x <xyz>      molecular geometry (GANSU/standard xyz; TAB or space separated)
#   -g <basis>    orbital basis set name (e.g. cc-pVDZ, def2-TZVP)
#   ORCA env      directory containing the `orca` binary (needed for ORCA's MPI
#                 launch); if unset, falls back to `orca` found in PATH.
#
# Optional:
#   -m <method>   rihf | rimp2 | dlpno_ccsd | dlpno_ccsd_t | dlpno_steom | ccsd | ccsd_t
#                 (default: dlpno_ccsd)
#   -n <nprocs>   MPI ranks / threads (%pal). Default 64.
#   -c <maxcore>  per-rank memory in MB (%maxcore). Default 3000.
#   -a <aux>      RI correlation-fitting aux basis. Default "<basis>/C"; use
#                 "AutoAux" for basis sets without a built-in /C fit.
#   -q <charge>   molecular charge. Default 0.
#   -s <mult>     spin multiplicity. Default 1.
#   -o <file>     ORCA input path to write. Default bench/orca/inp/<method>__<xyzstem>.inp
#   --no-run      only write the .inp, do not launch ORCA.
set -u

METHOD="dlpno_ccsd"; NPROCS=64; MAXCORE=3000; AUX=""; CHARGE=0; MULT=1
XYZ=""; BASIS=""; OUTINP=""; RUN=1
while [ $# -gt 0 ]; do
  case "$1" in
    -x) XYZ="$2"; shift 2;;
    -g) BASIS="$2"; shift 2;;
    -m) METHOD="$2"; shift 2;;
    -n) NPROCS="$2"; shift 2;;
    -c) MAXCORE="$2"; shift 2;;
    -a) AUX="$2"; shift 2;;
    -q) CHARGE="$2"; shift 2;;
    -s) MULT="$2"; shift 2;;
    -o) OUTINP="$2"; shift 2;;
    --no-run) RUN=0; shift;;
    -h|--help) sed -n '2,30p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[ -n "$XYZ" ]   || { echo "error: -x <xyz> required" >&2; exit 2; }
[ -n "$BASIS" ] || { echo "error: -g <basis> required" >&2; exit 2; }
[ -f "$XYZ" ]   || { echo "error: xyz not found: $XYZ" >&2; exit 2; }
[ -n "$AUX" ]   || AUX="${BASIS}/C"     # default correlation-fitting aux

HERE="$(cd "$(dirname "$0")" && pwd)"

# Method → ORCA simple-input header (frozen core = ORCA default = matches GANSU
# --frozen_core auto). DLPNO/MP2/CCSD correlation methods need the /C aux; RIHF
# uses RIJCOSX and ignores AUX.
case "$METHOD" in
  rihf)         HEAD="! RHF ${BASIS} RIJCOSX TightSCF";              EXTRA="";;
  rimp2)        HEAD="! RI-MP2 ${BASIS} ${AUX} TightSCF";            EXTRA="";;
  dlpno_ccsd)   HEAD="! DLPNO-CCSD ${BASIS} ${AUX} TightSCF";        EXTRA="";;
  dlpno_ccsd_t) HEAD="! DLPNO-CCSD(T) ${BASIS} ${AUX} TightSCF";     EXTRA="";;
  dlpno_steom)  HEAD="! STEOM-DLPNO-CCSD ${BASIS} ${AUX} TightSCF";  EXTRA=$'%mdci\n  nroots 5\nend\n';;
  ccsd)         HEAD="! CCSD ${BASIS} ${AUX} TightSCF";              EXTRA="";;
  ccsd_t)       HEAD="! CCSD(T) ${BASIS} ${AUX} TightSCF";           EXTRA="";;
  *) echo "error: unknown method '$METHOD'" >&2; exit 2;;
esac

STEM="$(basename "${XYZ%.*}")"
[ -n "$OUTINP" ] || { mkdir -p "$HERE/orca/inp"; OUTINP="$HERE/orca/inp/${METHOD}__${STEM}.inp"; }

# Re-emit the geometry space-separated. GANSU xyz files are TAB-separated, which
# ORCA's coordinate scanner rejects; split on any whitespace and rewrite "sym x y z".
GEOM="$(awk 'NR>2 && NF>=4 {printf "%-3s %18s %18s %18s\n",$1,$2,$3,$4}' "$XYZ")"
[ -n "$GEOM" ] || { echo "error: no atoms parsed from $XYZ" >&2; exit 2; }

{
  echo "$HEAD"
  echo "%pal nprocs ${NPROCS} end"
  echo "%maxcore ${MAXCORE}"
  printf '%s' "$EXTRA"
  echo "* xyz ${CHARGE} ${MULT}"
  echo "$GEOM"
  echo "*"
} > "$OUTINP"
echo "[run_orca_one] wrote $OUTINP  (method=$METHOD basis=$BASIS aux=$AUX nprocs=$NPROCS maxcore=${MAXCORE}MB)"

[ "$RUN" -eq 1 ] || { echo "[run_orca_one] --no-run: skipping execution."; exit 0; }

# Locate the orca binary (full path required so ORCA can launch its MPI children).
if [ -n "${ORCA:-}" ]; then ORCA_BIN="$ORCA/orca"
else ORCA_BIN="$(command -v orca || true)"; fi
[ -n "$ORCA_BIN" ] && [ -x "$ORCA_BIN" ] || {
  echo "error: orca binary not found. Set ORCA=/path/to/orca_root (dir with the orca binary)." >&2
  exit 3; }

OUT="${OUTINP%.inp}.out"
echo "[run_orca_one] running: $ORCA_BIN $OUTINP  →  $OUT"
START=$SECONDS
"$ORCA_BIN" "$OUTINP" > "$OUT" 2>&1 < /dev/null
RC=$?
WALL=$((SECONDS-START))

if   grep -qi "not enough memory\|insufficient memory\|out of memory\|aborting the run\|ORCA finished by error" "$OUT"; then STATUS="OOM_or_ERR"
elif grep -q "ORCA TERMINATED NORMALLY" "$OUT"; then STATUS="OK"
else STATUS="FAIL_rc$RC"; fi
RUNTIME="$(grep -oE 'TOTAL RUN TIME:.*' "$OUT" | tail -1)"
FINAL="$(grep -E 'FINAL SINGLE POINT ENERGY' "$OUT" | tail -1)"

echo "[run_orca_one] status=$STATUS  wall=${WALL}s  ${RUNTIME:-}"
[ -n "$FINAL" ] && echo "[run_orca_one] $FINAL"
echo "[run_orca_one] full output: $OUT"
[ "$STATUS" = "OK" ] && exit 0 || exit 1
