#!/usr/bin/env bash
# steom_scaling_ladder.sh — measure how far full-native DLPNO-bt-STEOM-CCSD
# scales on a given GPU configuration before OOM.
#
# WHY: the production STEOM excited-state stage builds the full canonical MO ERI
# (nao^4) and canonical IP/EA/STEOM bar-H intermediates (nvir^4, etc.) on a
# SINGLE GPU (RI CIS-NTO is single-GPU, eri_stored_cis.cu:307). So the ceiling is
# set by PER-GPU memory, not GPU count: H200 (141 GB) reaches a larger molecule
# than A100 (80 GB). This script finds that ceiling empirically.
#
# The peak per-device memory is printed by report_memory_statistics() at end of
# every completed run ("Peak usage: ..."). An OOM throws "tracked_cudaMalloc
# failed: ... requested N bytes", which names the dominant allocation = the wall.
#
# USAGE (run from the build directory containing ./gansu):
#   bash ../script/steom_scaling_ladder.sh <NGPU> <LABEL> [TIMEOUT_SEC]
#   e.g.  bash ../script/steom_scaling_ladder.sh 8 A100x8 3600
#         bash ../script/steom_scaling_ladder.sh 4 H200x4 3600
#
# It first probes benzene at NGPU. If the CIS-NTO guard throws (multi-GPU CIS
# not implemented), it auto-falls back to --num_gpus 1 for the whole ladder and
# tells you (stage-1 ground state then also runs single-GPU).

set -u
NGPU="${1:-1}"
LABEL="${2:-run}"
TIMEOUT="${3:-3600}"
MODE="${4:-hybrid}"   # hybrid = plain canonical IP/EA/STEOM (PM-validated, completes under PM);
                      # native = full-native operators (crashes under PM via the MO-collision guard).
                      # Memory ceiling is identical either way (both build the canonical operators).

GANSU=./gansu
BASIS=cc-pvdz
AUX=../auxiliary_basis/cc-pvdz-rifit.gbs
NSTATES=5
OUTDIR="scaling_${LABEL}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"
SUMMARY="$OUTDIR/SUMMARY.txt"

# FAST config: dressed (dense-borrow) + full GPU-resident matvec. We deliberately
# do NOT set NATIVE_RING / NATIVE_BARE: the dense-free path is NOT GPU-ported (its
# per-matvec ph-ladder runs on host = >>10x slower), AND it does not reduce peak
# memory here anyway (the driver still builds the full MO ERI nao^4 and the
# canonical IP/EA/STEOM operators still build their nvir^4 intermediates — that is
# what sets the OOM wall, and it happens with or without RING/BARE). So this fast
# config measures the same memory ceiling as the dense-free one, much faster.
if [ "$MODE" = native ]; then
  export GANSU_DLPNO_NATIVE_EOM=1
  export GANSU_DLPNO_NATIVE_DRESSED=1
  # FULL device-only resident matvec. Strictly nested AND-gated; RESIDENT requires
  # EVERY term flag. Omitting any one => un-accelerated host path (4899s-class).
  export GANSU_DLPNO_NATIVE_GPU=1 GANSU_DLPNO_NATIVE_GPU_PROJ=1 GANSU_DLPNO_NATIVE_GPU_LIFT=1 GANSU_DLPNO_NATIVE_GPU_XPAIR=1
  export GANSU_DLPNO_NATIVE_GPU_T1=1 GANSU_DLPNO_NATIVE_GPU_T8=1 GANSU_DLPNO_NATIVE_GPU_PHL=1 GANSU_DLPNO_NATIVE_GPU_S1LOO=1 GANSU_DLPNO_NATIVE_GPU_S1WOOOV=1
  export GANSU_DLPNO_NATIVE_GPU_PH1=1 GANSU_DLPNO_NATIVE_GPU_PH2=1 GANSU_DLPNO_NATIVE_GPU_PH3=1 GANSU_DLPNO_NATIVE_GPU_TMP=1 GANSU_DLPNO_NATIVE_GPU_TLVV=1 GANSU_DLPNO_NATIVE_GPU_TR1=1 GANSU_DLPNO_NATIVE_GPU_S1LVV=1 GANSU_DLPNO_NATIVE_GPU_S1WVOVV=1
  export GANSU_DLPNO_NATIVE_GPU_S1FOV=1 GANSU_DLPNO_NATIVE_GPU_RESIDENT=1
fi
# hybrid mode (default): no native env → plain canonical IP/EA/STEOM operators.

# aromatic / conjugated scaling ladder (atom count in comment), small -> large.
LADDER=(
  "../xyz/Benzene.xyz"                              # 12
  "../xyz/Naphthalene.xyz"                          # 18
  "../xyz/Anthracene.xyz"                           # 24
  "../xyz/Tetracene.xyz"                            # 30
  "../xyz/Pentacene.xyz"                            # 36
  "../xyz/large_molecular/Phthalocyanine.xyz"       # 58
  "../xyz/large_molecular/Decacene.xyz"             # 66
  "../xyz/large_molecular/Tetraphenylporphyrin.xyz" # 78
  "../xyz/large_molecular/C90.xyz"                  # 90
  "../xyz/large_molecular/Beta-Carotene.xyz"        # 96
)

run_one() {  # $1 = xyz path, $2 = ngpu ; echoes "STATUS|peak|wall|note"
  local xyz="$1" ng="$2" name log rc t0 t1
  name=$(basename "$xyz" .xyz)
  log="$OUTDIR/${name}.log"
  t0=$(date +%s)
  timeout "${TIMEOUT}" "$GANSU" -x "$xyz" -g "$BASIS" --eri_method ri -ag "$AUX" \
      --post_hf_method dlpno_steom_ccsd --n_excited_states "$NSTATES" \
      --num_gpus "$ng" --dlpno_localizer pm > "$log" 2>&1
  rc=$?
  t1=$(date +%s)
  local wall=$(( t1 - t0 ))
  local peak oomsz status note
  # max per-device peak (the bottleneck device — usually GPU 0); fall back to the
  # global "Peak usage:" line. The per-device "peak=X.XX GB/MB" values are
  # normalized to MB then max-reduced so GB and MB compare correctly.
  peak=$(grep -hoE "peak=[0-9.]+ [GM]B" "$log" \
         | sed -E 's/peak=//; s/ GB/*1024/; s/ MB/*1/' \
         | awk -F'*' '{m=$1*$2; if(m>mx)mx=m} END{if(mx)printf "%.0f MB(max-dev)", mx}')
  [ -z "$peak" ] && peak=$(grep -h "Peak usage:" "$log" | tail -1 | sed 's/.*Peak usage: //')
  if [ "$rc" -eq 124 ]; then
      status="TIMEOUT"; note=">${TIMEOUT}s"
  elif grep -q "tracked_cudaMalloc failed" "$log"; then
      oomsz=$(grep "tracked_cudaMalloc failed" "$log" | tail -1)
      status="OOM"; note="$oomsz"
  elif [ "$rc" -ne 0 ]; then
      status="ERR(rc=$rc)"; note="$(grep -iE 'error|exception|throw' "$log" | tail -1)"
  else
      status="OK"; note="$(grep -iE 'state ?0|excited.*state' "$log" | head -1)"
  fi
  echo "${status}|${peak:-?}|${wall}s|${note}"
}

echo "=== STEOM scaling ladder | label=$LABEL | requested NGPU=$NGPU | $(date) ===" | tee "$SUMMARY"

# --- probe: is NGPU>1 actually usable for the full STEOM run? ---
# The production STEOM excited-state stage (IP/EA/STEOM solve) is single-GPU
# tuned; with NGPU>1 the per-matvec peer-gather overhead makes it pathologically
# slow (benzene >1h instead of ~4min). And RI CIS-NTO can outright reject NGPU>1.
# So we only keep NGPU>1 if benzene COMPLETES (status OK) at NGPU; otherwise fall
# back to --num_gpus 1, where the memory ceiling (the thing we are measuring) is
# identical and benzene runs in ~4min.
EFFNG="$NGPU"
if [ "$NGPU" -gt 1 ]; then
  echo "-- probing benzene at --num_gpus $NGPU (must COMPLETE, not just survive) --" | tee -a "$SUMMARY"
  res=$(run_one "../xyz/Benzene.xyz" "$NGPU")
  st="${res%%|*}"
  echo "   benzene @ $NGPU -> $res" | tee -a "$SUMMARY"
  if [ "$st" != "OK" ]; then
    echo "   --num_gpus $NGPU is not viable for production STEOM ($st). Falling back to --num_gpus 1 for the ladder (identical memory ceiling, much faster)." | tee -a "$SUMMARY"
    EFFNG=1
  fi
fi

printf "%-26s %-12s %-10s %-9s %s\n" MOLECULE STATUS PEAK WALL NOTE | tee -a "$SUMMARY"
for xyz in "${LADDER[@]}"; do
  name=$(basename "$xyz" .xyz)
  res=$(run_one "$xyz" "$EFFNG")
  st="${res%%|*}"; rest="${res#*|}"; pk="${rest%%|*}"; rest="${rest#*|}"; wl="${rest%%|*}"; nt="${rest#*|}"
  printf "%-26s %-12s %-10s %-9s %s\n" "$name" "$st" "$pk" "$wl" "$nt" | tee -a "$SUMMARY"
  # stop walking up once we hit the memory wall (larger molecules only OOM harder)
  if [ "$st" = "OOM" ]; then
    echo ">>> OOM wall reached at $name (effective NGPU=$EFFNG). Larger systems will also OOM. Stopping." | tee -a "$SUMMARY"
    break
  fi
done
echo "=== done. logs + SUMMARY.txt in $OUTDIR ===" | tee -a "$SUMMARY"
