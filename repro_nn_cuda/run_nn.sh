#!/usr/bin/env bash
# Build and run the MaxPool and Adam checks and GPU tests on every GPU (or GPUS="0 1").
# Output: results/<host>/nn.log. The torch comparison runs only where torch is installed.
set -euo pipefail
cd "$(dirname "$0")"
command -v nvcc >/dev/null || { echo "nvcc not found (on CIMS: module load cuda-12.4)"; exit 1; }
OUT=results/$(hostname -s)${RESULT_TAG:+_$RESULT_TAG}; mkdir -p "$OUT" build; LOG=$OUT/nn.log; : > "$LOG"
log() { echo "$@" | tee -a "$LOG"; }
INC=(-I. -I../repro_math_cuda -I../repro_sum_cuda)
log "=== build ==="
g++ -O2 -std=c++17 -ffp-contract=off "${INC[@]}" check_repro_nn.cpp -o build/check_repro_nn >>"$LOG" 2>&1 \
  || { echo "build failed, see $LOG"; tail -n 30 "$LOG"; exit 1; }
nvcc -O3 -std=c++17 -arch=native -lineinfo "${INC[@]}" repro_nn.cu test_repro_nn.cu -o build/test_repro_nn >>"$LOG" 2>&1 \
  || { echo "build failed, see $LOG"; tail -n 30 "$LOG"; exit 1; }
FAIL=0
./build/check_repro_nn build/nn_cases.txt 2>&1 | tee -a "$LOG" || FAIL=1
if python3 -c "import torch" 2>/dev/null; then python3 check_repro_nn.py build/nn_cases.txt 2>&1 | tee -a "$LOG" || FAIL=1; fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [ -n "${GPUS:-}" ]; then read -r -a GPU_LIST <<<"$GPUS"
else read -r -a GPU_LIST <<<"$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')"; fi
for g in "${GPU_LIST[@]}"; do
  export CUDA_VISIBLE_DEVICES=$g
  log "=== GPU $g: $(nvidia-smi -i "$g" --query-gpu=name --format=csv,noheader) ==="
  ./build/test_repro_nn 2>&1 | tee -a "$LOG" || FAIL=1
done
if ((FAIL)); then log "OVERALL: FAIL (see $LOG)"; exit 2; fi
log "OVERALL: PASS"
