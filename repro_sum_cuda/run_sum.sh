#!/usr/bin/env bash
# Build and run the repro_sum tests and benchmark on every GPU (or GPUS="0 1").
# Output: results/<host>/sum.log
set -euo pipefail
cd "$(dirname "$0")"
command -v nvcc >/dev/null || { echo "nvcc not found (on CIMS: module load cuda-12.4)"; exit 1; }
OUT=results/$(hostname -s)${RESULT_TAG:+_$RESULT_TAG}; mkdir -p "$OUT"; LOG=$OUT/sum.log; : > "$LOG"
log() { echo "$@" | tee -a "$LOG"; }

log "=== build ==="
mkdir -p build
for t in test_repro_sum bench_repro_sum; do
  nvcc -O3 -std=c++17 -arch=native -lineinfo -I. repro_sum.cu "$t.cu" -o "build/$t" >>"$LOG" 2>&1 \
    || { echo "build failed, see $LOG"; tail -n 30 "$LOG"; exit 1; }
done

export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [ -n "${GPUS:-}" ]; then read -r -a GPU_LIST <<<"$GPUS"
else read -r -a GPU_LIST <<<"$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')"; fi
FAIL=0
for g in "${GPU_LIST[@]}"; do
  export CUDA_VISIBLE_DEVICES=$g
  log "=== GPU $g: $(nvidia-smi -i "$g" --query-gpu=name --format=csv,noheader) ==="
  ./build/test_repro_sum 2>&1 | tee -a "$LOG" || FAIL=1
  ./build/bench_repro_sum 30 2>&1 | tee -a "$LOG" || FAIL=1
done
if ((FAIL)); then log "OVERALL: FAIL (see $LOG)"; exit 2; fi
log "OVERALL: PASS"
