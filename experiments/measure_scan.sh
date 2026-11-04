#!/usr/bin/env bash
# Measures what the NaN and infinity scan costs: our 1200 steps with and without it, alternating.
# Writes outside results/, so the comparison never counts these runs. Hashes must be identical.
#   GPUS="0" experiments/measure_scan.sh cuda124
set -uo pipefail
source "$(dirname "$0")/common.sh"
load_config "$1"
cd "$REPO"
g=$(gpu_list); g=${g%% *}
export CUDA_VISIBLE_DEVICES=$g
build_repro "$g"
OUT="$REPRO_WORK/scan_cost/$(hostname -s)_${TAG}"
for rep in 1 2 3; do
  for kind in scan noscan; do
    flag=""; [ "$kind" = noscan ] && flag="--no-scan"
    python3 repro_torch/mnist_repro.py --device cuda --mode repro --seed 0 $flag --out "$OUT/${kind}_${rep}.txt" >/dev/null || exit 1
    echo "$kind $rep: $(grep '^time' "$OUT/${kind}_${rep}.txt")"
  done
done
echo "final weights (must be one value):"
grep -h "hash weights after epoch 2" "$OUT"/*.txt | awk '{print $NF}' | sort -u
