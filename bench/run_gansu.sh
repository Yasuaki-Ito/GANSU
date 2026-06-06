#!/usr/bin/env bash
# GANSU size-scaling + max-size benchmark runner.
# Run from the build/ directory (where ./gansu lives):
#     bash ../bench/run_gansu.sh <method> <num_gpus> [timeout_min]
#   method  : rihf | rimp2 | dlpno_ccsd | dlpno_steom
#   num_gpus: 1 2 4 ...   (H200x4 → use 4; pass 1 for the single-GPU reference column)
#   timeout : per-job wall cap in minutes (default 120). A job that hits the cap
#             or OOMs is recorded as the size ceiling for that method.
#
# Output: bench/logs/<method>_g<N>_<mol>.log  (full stdout/stderr)
#         bench/results_<method>_g<N>.tsv      (one row/molecule: status + wall + key phase times)
# Parse all results into a CSV with: python3 ../bench/parse_gansu.py
set -u

METHOD="${1:?method: rihf|rimp2|dlpno_ccsd|dlpno_steom}"
NG="${2:?num_gpus}"
TIMEOUT_MIN="${3:-120}"
BENCH_DIR="$(cd "$(dirname "$0")" && pwd)"
SERIES="$BENCH_DIR/series.tsv"
LOGDIR="$BENCH_DIR/logs"; mkdir -p "$LOGDIR"
AUX="../auxiliary_basis/cc-pvdz-rifit.gbs"
BASIS="cc-pvdz"
RES="$BENCH_DIR/results_${METHOD}_g${NG}.tsv"

# Common env for the native EOM / large-system paths (auto-scale handles the rest).
export GANSU_DLPNO_NATIVE_EOM=1
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-64}"

# SCF initial guess: SAD (superposition of atomic densities) by default so the SCF
# wall reflects the per-iteration GPU cost rather than the convergence path (the
# default `core` guess takes many more, run-dependent iterations). Override with
# GUESS=core|gwh|minao|... if needed. (Parser also reports iters + per-iter Fock
# time, which is iteration-count-independent.)
GUESS="${GUESS:-sad}"

# method → gansu args (post-HF + RI + aux). Spherical (5D/7F, --use_spherical) +
# SAD guess are applied commonly below — cc-pVDZ is defined spherical and ORCA uses
# spherical for cc-pVnZ by default, so this keeps the GANSU/ORCA comparison fair.
gansu_args() {
  case "$1" in
    rihf)        echo "--eri_method ri -ag $AUX -m RHF" ;;
    rimp2)       echo "--eri_method ri -ag $AUX --post_hf_method MP2" ;;
    dlpno_ccsd)  echo "--eri_method ri -ag $AUX --post_hf_method dlpno_ccsd --frozen_core auto" ;;
    dlpno_steom) echo "--eri_method ri -ag $AUX --post_hf_method dlpno_steom_ccsd --frozen_core auto" ;;
    *) echo "UNKNOWN_METHOD" ;;
  esac
}
ARGS="$(gansu_args "$METHOD")"
[ "$ARGS" = "UNKNOWN_METHOD" ] && { echo "unknown method '$METHOD'"; exit 2; }

printf 'name\tnatoms\tstatus\twall_s\tscf_ms\tpostproc_ms\textra\n' > "$RES"
echo "# method=$METHOD num_gpus=$NG timeout=${TIMEOUT_MIN}min  → $RES"
echo "# (Ctrl-C once: kills the running job + exits the whole sweep — no run-on.)"

# One Ctrl-C kills the running gansu and exits the WHOLE script (never continues to
# the next molecule). The loop reads via process substitution so it runs in the MAIN
# shell — a `grep | while` pipe runs the loop in a subshell where the trap could not
# stop it. gansu is single-process multi-GPU, so killing it frees all GPUs cleanly.
CHILD=""
on_int() {
  echo
  echo "[run_gansu] interrupted → killing current job and exiting (results so far: $RES)."
  if [ -n "$CHILD" ]; then kill -TERM "$CHILD" 2>/dev/null; sleep 1; kill -KILL "$CHILD" 2>/dev/null; fi
  exit 130
}
trap on_int INT TERM

while IFS=$'\t' read -r NAME XYZ NAT TAGS; do
  [ -z "${NAME:-}" ] && continue
  # only molecules tagged for this method
  case ",$TAGS," in *",$METHOD,"*) : ;; *) continue ;; esac
  [ -f "$XYZ" ] || { echo "  SKIP $NAME (missing $XYZ)"; continue; }

  LOG="$LOGDIR/${METHOD}_g${NG}_${NAME}.log"
  echo "  RUN  $NAME ($NAT atoms) → $LOG"
  START=$SECONDS
  # < /dev/null: stop the background job from consuming the loop's stdin (the
  # process-substitution fd) — else it eats the rest of series.tsv and the loop
  # ends after the first molecule.
  timeout "${TIMEOUT_MIN}m" ./gansu -x "$XYZ" -g "$BASIS" --num_gpus "$NG" \
          --use_spherical 1 --initial_guess "$GUESS" $ARGS > "$LOG" 2>&1 < /dev/null &
  CHILD=$!
  wait "$CHILD"; RC=$?
  CHILD=""
  WALL=$((SECONDS-START))

  if   [ $RC -eq 124 ]; then STATUS="TIMEOUT"
  elif grep -qi "out of memory\|cudaMalloc failed\|bad_alloc" "$LOG"; then STATUS="OOM"
  elif [ $RC -ne 0 ]; then STATUS="FAIL_rc$RC"
  else STATUS="OK"; fi

  SCF=$(grep -oE 'Computing time: [0-9]+' "$LOG" | tail -1 | grep -oE '[0-9]+'); SCF="${SCF:-NA}"
  PP=$(grep -oE 'post_process_after_scf after [0-9.eE+]+ ms' "$LOG" | tail -1 | grep -oE '[0-9.eE+]+' | head -1); PP="${PP:-NA}"
  # method-specific extra (corr energy / first excitation) for a sanity column
  EXTRA=""
  case "$METHOD" in
    dlpno_ccsd)  EXTRA=$(grep -oE 'E\(total CCSD corr\)  = -[0-9.]+' "$LOG" | tail -1) ;;
    dlpno_steom) EXTRA=$(grep -A6 'STEOM excited-state energies' "$LOG" | grep -oE '\-?[0-9]+\.[0-9]+ *$' | head -1) ;;
    rimp2)       EXTRA=$(grep -oiE 'MP2 correlation[^-]*-[0-9.]+' "$LOG" | tail -1) ;;
    rihf)        EXTRA=$(grep -oE 'Total Energy: -[0-9.]+' "$LOG" | tail -1) ;;
  esac
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$NAME" "$NAT" "$STATUS" "$WALL" "$SCF" "$PP" "$EXTRA" >> "$RES"
  echo "       $STATUS  wall=${WALL}s"

  # stop pushing size once we hit the ceiling for this method (OOM/timeout)
  if [ "$STATUS" = "OOM" ] || [ "$STATUS" = "TIMEOUT" ]; then
    echo "  ↳ size ceiling reached for $METHOD at $NAME ($NAT atoms); stopping."
    break
  fi
done < <(grep -v '^#' "$SERIES")
echo "done. results: $RES"
