#!/usr/bin/env bash
# Build exhaustive_math and run it on every GPU of this machine (or GPUS="0").
# Results: results/<host>/gpu<N>_<model>.txt  (per-block hashes + summary)
#          results/<host>/math.log            (printed tables)
# Compare two GPUs or machines afterwards with compare_math.sh.
set -euo pipefail
cd "$(dirname "$0")"
command -v nvcc >/dev/null || { echo "nvcc not found (on CIMS: module load cuda-12.4)"; exit 1; }
HOST=$(hostname -s); OUT=results/$HOST; mkdir -p "$OUT" build
LOG=$OUT/math.log; : > "$LOG"
log() { echo "$@" | tee -a "$LOG"; }

log "=== build (default flags: no fast-math, like PyTorch) ==="
nvcc --version | tail -n 2 | tee -a "$LOG"
nvcc -O3 -std=c++17 -arch=native exhaustive_math.cu -o build/exhaustive_math >>"$LOG" 2>&1 \
  || { echo "build failed, see $LOG"; tail -n 30 "$LOG"; exit 1; }

export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [ -n "${GPUS:-}" ]; then read -r -a GPU_LIST <<<"$GPUS"
else read -r -a GPU_LIST <<<"$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')"; fi
for g in "${GPU_LIST[@]}"; do
  export CUDA_VISIBLE_DEVICES=$g
  name=$(nvidia-smi -i "$g" --query-gpu=name --format=csv,noheader | tr ' ' '_')
  log "=== GPU $g: $name ==="
  ./build/exhaustive_math "$OUT/gpu${g}_${name}.txt" 2>&1 | tee -a "$LOG"
done
log "Done. Compare GPUs/machines with: ./compare_math.sh <file1> <file2>"
