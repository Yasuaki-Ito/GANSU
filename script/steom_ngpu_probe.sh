#!/usr/bin/env bash
# steom_ngpu_probe.sh — characterize full-native DLPNO-bt-STEOM behaviour vs
# num_gpus on ONE small molecule (benzene), with a short fail-fast timeout, so we
# can pick the fastest viable --num_gpus BEFORE committing to the scaling ladder.
#
# For each num_gpus it records: completion status, total wall time, per-device
# peak memory, and the LAST timestamped phase reached (so a hang/slowdown is
# pinned to a stage: stage1 DLPNO-CCSD ground / IP-EOM / EA-EOM / STEOM).
#
# USAGE (from the build dir containing ./gansu):
#   bash ../script/steom_ngpu_probe.sh                 # sweeps 1 2 4 8, timeout 600s
#   bash ../script/steom_ngpu_probe.sh "1 2 4" 900     # custom list + timeout
#   bash ../script/steom_ngpu_probe.sh "1" 600 ../xyz/Naphthalene.xyz   # other mol

set -u
NGPU_LIST="${1:-1 2 4 8}"
TIMEOUT="${2:-600}"
XYZ="${3:-../xyz/Benzene.xyz}"

GANSU=./gansu
BASIS=cc-pvdz
AUX=../auxiliary_basis/cc-pvdz-rifit.gbs
NSTATES=5
NAME=$(basename "$XYZ" .xyz)
OUTDIR="ngpu_probe_${NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"
SUMMARY="$OUTDIR/SUMMARY.txt"

# --- dense-free native path ---
export GANSU_DLPNO_NATIVE_EOM=1
export GANSU_DLPNO_NATIVE_DRESSED=1
export GANSU_DLPNO_NATIVE_RING=1
export GANSU_DLPNO_NATIVE_BARE=1
# --- FULL device-only matvec (the 184.5s config). These are strictly nested
#     AND-gated; RESIDENT requires EVERY term flag on. Omitting any one drops the
#     native operator back to the un-accelerated (host / tiny-GEMV) path = >>10x
#     slower (the original 4899s-class IP solve). ---
export GANSU_DLPNO_NATIVE_GPU=1
export GANSU_DLPNO_NATIVE_GPU_PROJ=1
export GANSU_DLPNO_NATIVE_GPU_LIFT=1
export GANSU_DLPNO_NATIVE_GPU_XPAIR=1
# IP-sector terms
export GANSU_DLPNO_NATIVE_GPU_T1=1
export GANSU_DLPNO_NATIVE_GPU_T8=1
export GANSU_DLPNO_NATIVE_GPU_PHL=1
export GANSU_DLPNO_NATIVE_GPU_S1LOO=1
export GANSU_DLPNO_NATIVE_GPU_S1WOOOV=1
# EA-sector terms
export GANSU_DLPNO_NATIVE_GPU_PH1=1
export GANSU_DLPNO_NATIVE_GPU_PH2=1
export GANSU_DLPNO_NATIVE_GPU_PH3=1
export GANSU_DLPNO_NATIVE_GPU_TMP=1
export GANSU_DLPNO_NATIVE_GPU_TLVV=1
export GANSU_DLPNO_NATIVE_GPU_TR1=1
export GANSU_DLPNO_NATIVE_GPU_S1LVV=1
export GANSU_DLPNO_NATIVE_GPU_S1WVOVV=1
# shared by IP+EA
export GANSU_DLPNO_NATIVE_GPU_S1FOV=1
export GANSU_DLPNO_NATIVE_GPU_RESIDENT=1

echo "=== num_gpus probe | mol=$NAME | list=[$NGPU_LIST] | timeout=${TIMEOUT}s | $(date) ===" | tee "$SUMMARY"
printf "%-7s %-12s %-9s %-10s %s\n" NGPU STATUS WALL PEAK LAST_PHASE | tee -a "$SUMMARY"

for ng in $NGPU_LIST; do
  log="$OUTDIR/ngpu_${ng}.log"
  t0=$(date +%s)
  timeout "${TIMEOUT}" "$GANSU" -x "$XYZ" -g "$BASIS" --eri_method ri -ag "$AUX" \
      --post_hf_method dlpno_steom_ccsd --n_excited_states "$NSTATES" \
      --num_gpus "$ng" --dlpno_localizer pm > "$log" 2>&1
  rc=$?
  t1=$(date +%s); wall=$(( t1 - t0 ))
  peak=$(grep -h "peak=" "$log" | tail -1 | sed 's/.*peak=//')
  [ -z "$peak" ] && peak=$(grep -h "Peak usage:" "$log" | tail -1 | sed 's/.*Peak usage: //')
  # last timestamped phase line, e.g. "[01:23.456] START: compute_ea_eom_ccsd_impl"
  last=$(grep -hE "START:|----|solve|Operator build time|Peak usage" "$log" | tail -1 | cut -c1-70)
  if   [ "$rc" -eq 124 ]; then status="TIMEOUT(>${TIMEOUT}s)"
  elif grep -q "tracked_cudaMalloc failed" "$log"; then status="OOM"
  elif [ "$rc" -ne 0 ]; then status="ERR(rc=$rc)"
  else status="OK"; fi
  printf "%-7s %-12s %-9s %-10s %s\n" "$ng" "$status" "${wall}s" "${peak:-?}" "${last:-?}" | tee -a "$SUMMARY"
done
echo "=== done. per-num_gpus logs in $OUTDIR ; pick the fastest OK row for the ladder ===" | tee -a "$SUMMARY"
