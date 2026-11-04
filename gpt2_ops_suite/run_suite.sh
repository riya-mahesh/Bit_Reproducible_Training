#!/usr/bin/env bash
# Run the GPT-2 operation suite on this machine's CPU and every GPU, then compare
# output hashes across all result files (including other machines' results/*).
#   ./run_suite.sh                 # CPU + all GPUs, 1024 tokens
#   GPUS="0" TOKENS=1024 ./run_suite.sh
# Results: results/<host>/{cpu,gpu<N>_<model>}.txt and results/<host>/suite.log
set -euo pipefail
cd "$(dirname "$0")"
HOST=$(hostname -s)${RESULT_TAG:+_$RESULT_TAG}; OUT=results/$HOST; mkdir -p "$OUT" build
LOG=$OUT/suite.log; : > "$LOG"
TOKENS=${TOKENS:-1024}
log() { echo "$@" | tee -a "$LOG"; }
SRC=(-I../repro_gemm_cuda -I../repro_sum_cuda -I../repro_math_cuda -I../repro_conv_cuda -I../repro_nn_cuda)

log "=== build ==="
# CPU-only build with the system C++ compiler: no CUDA toolkit involved, like a CPU verifier.
g++ -O2 -std=c++17 -pthread -fno-fast-math -ffp-contract=off -DSUITE_CPU_ONLY -x c++ "${SRC[@]}" \
    gpt2_ops_suite.cu -o build/suite_cpu >>"$LOG" 2>&1 || { echo "CPU build failed, see $LOG"; tail -n 30 "$LOG"; exit 1; }
if command -v nvcc >/dev/null; then
  nvcc -O3 -std=c++17 -arch=native -lineinfo "${SRC[@]}" ../repro_gemm_cuda/repro_gemm.cu ../repro_sum_cuda/repro_sum.cu ../repro_math_cuda/repro_softmax.cu ../repro_conv_cuda/repro_conv.cu ../repro_nn_cuda/repro_nn.cu \
       gpt2_ops_suite.cu -o build/suite_gpu -lcublas >>"$LOG" 2>&1 || { echo "GPU build failed, see $LOG"; tail -n 30 "$LOG"; exit 1; }
else
  log "nvcc not found (module load cuda-12.4): running CPU only"
fi

log "=== CPU: $(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2) ==="
./build/suite_cpu --device cpu --tokens "$TOKENS" --out "$OUT/cpu.txt" 2>&1 | tee -a "$LOG"

if [ -x build/suite_gpu ]; then
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  if [ -n "${GPUS:-}" ]; then read -r -a GPU_LIST <<<"$GPUS"
  else read -r -a GPU_LIST <<<"$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')"; fi
  for g in "${GPU_LIST[@]}"; do
    export CUDA_VISIBLE_DEVICES=$g
    name=$(nvidia-smi -i "$g" --query-gpu=name --format=csv,noheader | tr ' ' '_')
    log "=== GPU $g: $name ==="
    ./build/suite_gpu --device gpu --tokens "$TOKENS" --out "$OUT/gpu${g}_${name}.txt" 2>&1 | tee -a "$LOG"
  done
fi

log ""
log "=== hash comparison across all result files ==="
./compare_suite.sh results/*/*.txt 2>&1 | tee -a "$LOG"
