#!/usr/bin/env bash
# Build and run the exp/log/softmax tests and benchmark on every GPU (or GPUS="0 1"),
# then hash repro_expf and repro_logf over all 2^32 inputs on each GPU.
# Output: results/<host>/softmax.log and results/<host>/repro_gpu<N>_<model>.txt
# Compare against the CPU with compare_math.sh.
set -euo pipefail
cd "$(dirname "$0")"
command -v nvcc >/dev/null || { echo "nvcc not found (on CIMS: module load cuda-12.4)"; exit 1; }
OUT=results/$(hostname -s)${RESULT_TAG:+_$RESULT_TAG}; mkdir -p "$OUT" build; LOG=$OUT/softmax.log; : > "$LOG"
log() { echo "$@" | tee -a "$LOG"; }

log "=== build ==="
nvcc --version | tail -n 2 | tee -a "$LOG"
SUM=../repro_sum_cuda
for t in test_repro_softmax bench_repro_math; do
  nvcc -O3 -std=c++17 -arch=native -lineinfo -I. -I$SUM repro_softmax.cu $SUM/repro_sum.cu "$t.cu" -o "build/$t" >>"$LOG" 2>&1 \
    || { echo "build failed, see $LOG"; tail -n 30 "$LOG"; exit 1; }
done
nvcc -O3 -std=c++17 -arch=native exhaustive_math.cu -o build/exhaustive_math >>"$LOG" 2>&1 \
  || { echo "build failed, see $LOG"; tail -n 30 "$LOG"; exit 1; }

export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [ -n "${GPUS:-}" ]; then read -r -a GPU_LIST <<<"$GPUS"
else read -r -a GPU_LIST <<<"$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')"; fi
FAIL=0
for g in "${GPU_LIST[@]}"; do
  export CUDA_VISIBLE_DEVICES=$g
  name=$(nvidia-smi -i "$g" --query-gpu=name --format=csv,noheader | tr ' ' '_')
  log "=== GPU $g: $name ==="
  ./build/test_repro_softmax 2>&1 | tee -a "$LOG" || FAIL=1
  ./build/bench_repro_math 20 2>&1 | tee -a "$LOG" || FAIL=1
  ./build/exhaustive_math "$OUT/repro_gpu${g}_${name}.txt" repro_expf repro_logf 2>&1 | tee -a "$LOG" || FAIL=1
done
if ((FAIL)); then log "OVERALL: FAIL (see $LOG)"; exit 2; fi
log "OVERALL: PASS"
