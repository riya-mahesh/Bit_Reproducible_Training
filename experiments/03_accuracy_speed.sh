#!/usr/bin/env bash
# Goals 2 and 3: training and test accuracy, and speed, for ours, RepDL, and stock PyTorch, with the
# same data, order, and initial weights on the same GPU. Several seeds for accuracy, plus one
# profiled run per method for the breakdown, plus RepDL's own example as a check of its install.
#   GPUS="0" SEEDS="0 1 2 3 4" experiments/03_accuracy_speed.sh cuda124
set -uo pipefail
source "$(dirname "$0")/common.sh"
load_config "$1"
cd "$REPO"
SEEDS=${SEEDS:-0 1 2 3 4}
MODES=${MODES:-repro repdl stock}
FAIL=0
for g in $(gpu_list); do
  export CUDA_VISIBLE_DEVICES=$g
  build_repro "$g"
  name=$(gpu_name "$g")
  OUT="$REPO/results/mnist/$(hostname -s)_${TAG}"
  for mode in $MODES; do
    if [ "$mode" = repdl ] && ! (cd /tmp && python3 -c "import repdl" 2>/dev/null); then
      echo "RepDL not installed in $TAG: skipping"; continue
    fi
    for seed in $SEEDS; do
      echo "=== $name, $mode, seed $seed ==="
      python3 repro_torch/mnist_repro.py --device cuda --mode "$mode" --seed "$seed" \
          --out "$OUT/gpu${g}_${name}_${mode}_seed${seed}.txt" | grep -v "^  epoch" || FAIL=1
    done
    echo "=== $name, $mode, profiled ==="
    python3 repro_torch/mnist_repro.py --device cuda --mode "$mode" --seed 0 --profile \
        --out "$OUT/gpu${g}_${name}_${mode}_profile.txt" | grep -E "median ms|^time" || FAIL=1
  done
  # RepDL's own script, with its own data loader and seed. Its README publishes the expected hashes.
  if [ -f "$ENV_DIR/RepDL/examples/mnist_trainig.py" ] && python3 -c "import torchvision" 2>/dev/null; then
    echo "=== $name, RepDL's own example (expected trained-model hash 31ee86a7...) ==="
    mkdir -p "$OUT"
    (cd "$OUT" && python3 "$ENV_DIR/RepDL/examples/mnist_trainig.py") 2>&1 | grep -E "Hash|accuracy" \
        | tee "$OUT/gpu${g}_${name}_repdl_official.log" || FAIL=1
  fi
  rm -rf "$OUT/data"
done
exit $FAIL
