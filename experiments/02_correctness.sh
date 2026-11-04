#!/usr/bin/env bash
# Goal 1: bit-reproducibility. Runs every kernel test and hash, and MNIST with our layers, on each
# GPU of this machine in one configuration. The results are compared across machines and
# configurations by compare_results.py.
#   GPUS="0" experiments/02_correctness.sh cuda124
# SKIP_CPU=1 skips the CPU proof, which only needs to run once per machine.
set -uo pipefail
source "$(dirname "$0")/common.sh"
load_config "$1"
cd "$REPO"
FAIL=0
run() { echo; echo "##### $1"; shift; "$@" || { FAIL=1; echo "##### FAILED: $*"; }; }
run "conv kernels" repro_conv_cuda/run_conv.sh
run "pool and Adam kernels" repro_nn_cuda/run_nn.sh
run "exp, log, softmax, cross-entropy (includes all 2^32 hashes)" repro_math_cuda/run_softmax.sh
run "exact sums" repro_sum_cuda/run_sum.sh
run "all operations, GEMM at S=3 included" gpt2_ops_suite/run_suite.sh
[ -n "${SKIP_CPU:-}" ] || run "exp and log on this CPU, all 2^32 inputs" repro_math_cuda/run_check_math.sh
for g in $(gpu_list); do
  export CUDA_VISIBLE_DEVICES=$g
  build_repro "$g"
  name=$(gpu_name "$g")
  OUT="$REPO/results/mnist/$(hostname -s)_${TAG}"
  for run_id in 1 2; do
    run "MNIST, ours, run $run_id on $name" python3 repro_torch/mnist_repro.py --device cuda --mode repro --seed 0 \
        --out "$OUT/gpu${g}_${name}_repro_seed0_run${run_id}.txt"
  done
done
echo
[ "$FAIL" = 0 ] && echo "CORRECTNESS RUN FINISHED: all steps completed" || echo "CORRECTNESS RUN FINISHED WITH FAILURES: search this output for FAILED"
exit $FAIL
