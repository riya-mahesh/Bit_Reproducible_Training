#!/usr/bin/env bash
# A few steps of ours, RepDL, and stock on one GPU, to catch build and run errors and to read
# ms per step before the long runs. Writes outside results/, so the comparison never counts it.
#   GPUS="0" experiments/smoke.sh cuda124
set -uo pipefail
source "$(dirname "$0")/common.sh"
load_config "$1"
cd "$REPO"
FAIL=0
for g in $(gpu_list); do
  export CUDA_VISIBLE_DEVICES=$g
  build_repro "$g"
  name=$(gpu_name "$g")
  echo "=== $name: load on this GPU ==="
  nvidia-smi -i "$g" --query-gpu=utilization.gpu,memory.used --format=csv,noheader
  for mode in repro repdl stock; do
    echo "=== $name, $TAG, $mode ==="
    python3 repro_torch/mnist_repro.py --device cuda --mode "$mode" --seed 0 --max-steps 100 \
        --out "$REPRO_WORK/smoke/$(hostname -s)_${TAG}_gpu${g}_${mode}.txt" \
        | grep -E "^mode|median ms|^time|accuracy" || { FAIL=1; echo "##### FAILED: $mode"; }
  done
done
[ "$FAIL" = 0 ] && echo "SMOKE TEST PASSED" || echo "SMOKE TEST FAILED"
exit $FAIL
