#!/usr/bin/env bash
# Build and run the repro_conv checks, tests, and benchmark on every GPU (or GPUS="0 1").
# Output: results/<host>/conv.log
set -euo pipefail
cd "$(dirname "$0")"
command -v nvcc >/dev/null || { echo "nvcc not found (on CIMS: module load cuda-12.4)"; exit 1; }
OUT=results/$(hostname -s)${RESULT_TAG:+_$RESULT_TAG}; mkdir -p "$OUT" build; LOG=$OUT/conv.log; : > "$LOG"
log() { echo "$@" | tee -a "$LOG"; }
GEMM=../repro_gemm_cuda
log "=== build ==="
g++ -O2 -std=c++17 -pthread -ffp-contract=off -I. -I$GEMM check_repro_conv.cpp -o build/check_repro_conv >>"$LOG" 2>&1 \
  || { echo "build failed, see $LOG"; tail -n 30 "$LOG"; exit 1; }
for t in test_repro_conv bench_repro_conv; do
  nvcc -O3 -std=c++17 -arch=native -lineinfo -I. -I$GEMM repro_conv.cu $GEMM/repro_gemm.cu "$t.cu" -o "build/$t" >>"$LOG" 2>&1 \
    || { echo "build failed, see $LOG"; tail -n 30 "$LOG"; exit 1; }
done
FAIL=0
./build/check_repro_conv 2>&1 | tee -a "$LOG" || FAIL=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [ -n "${GPUS:-}" ]; then read -r -a GPU_LIST <<<"$GPUS"
else read -r -a GPU_LIST <<<"$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')"; fi
for g in "${GPU_LIST[@]}"; do
  export CUDA_VISIBLE_DEVICES=$g
  log "=== GPU $g: $(nvidia-smi -i "$g" --query-gpu=name --format=csv,noheader) ==="
  ./build/test_repro_conv 2>&1 | tee -a "$LOG" || FAIL=1
  ./build/bench_repro_conv 30 2>&1 | tee -a "$LOG" || FAIL=1
done
if ((FAIL)); then log "OVERALL: FAIL (see $LOG)"; exit 2; fi
log "OVERALL: PASS"
