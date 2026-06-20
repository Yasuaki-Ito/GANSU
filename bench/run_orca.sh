#!/usr/bin/env bash
# ORCA size-scaling + max-size runner (CPU reference for the GANSU GPU benchmark).
# Processes molecules in series.tsv order (increasing size) for one method and
# STOPS at the first OOM / timeout / error — never wastes time on a larger molecule
# once the current size already failed.
#
#   ORCA=/path/to/orca_root  bash bench/run_orca.sh <method> [timeout_min]
#     method  : rihf | rimp2 | dlpno_ccsd | dlpno_steom
#     ORCA env: directory containing the `orca` binary (needed for ORCA's MPI launch)
# Inputs come from bench/orca/inp/<method>__<molecule>.inp (run make_orca_inputs.py first).
# Output: bench/orca/inp/<...>.out  + bench/orca/results_<method>.tsv
set -u
METHOD="${1:?method: rihf|rimp2|dlpno_ccsd|dlpno_steom}"
TIMEOUT_MIN="${2:-240}"
: "${ORCA:?set ORCA=/path/to/orca_root (dir with the orca binary)}"
HERE="$(cd "$(dirname "$0")" && pwd)"
SERIES="${SERIES:-$HERE/series.tsv}"        # override e.g. SERIES=bench/series_aldehyde.tsv
SERIES_TAG="$(basename "$SERIES" .tsv)"
INP="$HERE/orca/inp"
if [ "$SERIES_TAG" = "series" ]; then
  RES="$HERE/orca/results_${METHOD}.tsv"
else
  RES="$HERE/orca/results_${METHOD}_${SERIES_TAG}.tsv"
fi
mkdir -p "$HERE/orca"
printf 'name\tnatoms\tstatus\twall_s\torca_runtime\n' > "$RES"

echo "# (Ctrl-C once: kills the ORCA MPI tree + exits the whole sweep — no run-on.)"
# One Ctrl-C kills the running ORCA *process group* (ORCA spawns MPI children that a
# plain kill would orphan) and exits the whole script. setsid puts ORCA in its own
# session/pgroup so we can kill the entire tree; the loop reads via process
# substitution so the trap runs in the main shell.
CHILD=""
on_int() {
  echo
  echo "[run_orca] interrupted → killing ORCA (process group) and exiting (results: $RES)."
  if [ -n "$CHILD" ]; then
    kill -TERM -- -"$CHILD" 2>/dev/null   # CHILD = its own pgroup leader (setsid)
    sleep 2; kill -KILL -- -"$CHILD" 2>/dev/null
  fi
  exit 130
}
trap on_int INT TERM

while IFS=$'\t' read -r NAME XYZ NAT TAGS; do
  [ -z "${NAME:-}" ] && continue
  case ",$TAGS," in *",$METHOD,"*) : ;; *) continue ;; esac
  F="$INP/${METHOD}__${NAME}.inp"
  [ -f "$F" ] || { echo "  SKIP $NAME (no $F)"; continue; }
  OUT="${F%.inp}.out"
  echo "  RUN  $NAME ($NAT atoms) → $OUT    (Ctrl-C to stop)"
  START=$SECONDS
  # < /dev/null: keep the background job from reading the loop's stdin (the
  # process-substitution fd) — otherwise ORCA/mpirun swallows the rest of series.tsv
  # and the while-read loop ends after the first molecule.
  setsid timeout "${TIMEOUT_MIN}m" "$ORCA/orca" "$F" > "$OUT" 2>&1 < /dev/null &
  CHILD=$!
  wait "$CHILD"; RC=$?
  CHILD=""
  WALL=$((SECONDS-START))
  if   [ $RC -eq 124 ]; then STATUS="TIMEOUT"
  elif grep -qi "not enough memory\|insufficient memory\|out of memory\|aborting the run\|ORCA finished by error" "$OUT"; then STATUS="OOM_or_ERR"
  elif grep -q "ORCA TERMINATED NORMALLY" "$OUT"; then STATUS="OK"
  else STATUS="FAIL_rc$RC"; fi
  RUNTIME=$(grep -oE 'TOTAL RUN TIME:.*' "$OUT" | tail -1)
  printf '%s\t%s\t%s\t%s\t%s\n' "$NAME" "$NAT" "$STATUS" "$WALL" "${RUNTIME:-NA}" >> "$RES"
  echo "       $STATUS  wall=${WALL}s"
  if [ "$STATUS" != "OK" ]; then
    echo "  ↳ ORCA ceiling reached for $METHOD at $NAME ($NAT atoms); stopping (no larger tries)."
    break
  fi
done < <(grep -v '^#' "$SERIES")
echo "done. results: $RES"
