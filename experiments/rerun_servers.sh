#!/usr/bin/env bash
# Each invocation gets an isolated source/build tree and a unique host/result path.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SUPPORT="$ROOT/experiments/rerun_support.py"
HOST=$(hostname -s)
SUITE=mnist
CONFIGS=auto
SOURCE=
DRY=0
NCU=0
usage() {
  cat <<'EOF'
Usage: bash experiments/rerun_servers.sh --source PATH|working-tree [options]
  --suite mnist|kernels|full|a100   Default: mnist
  --configs "cuda124 cuda126 ..."  Default: historical matrix for this host
  --gpus "0 1"                    Default: all visible nvidia-smi indices
  --source PATH                   Atomic .tar/.tar.gz, or a source directory
  --source working-tree           Explicitly use current (possibly modified) files
  --ncu                           With a100: also collect targeted Nsight reports
  --dry-run                       Print plan only; no GPU/env/install needed

Environment:
  REPRO_WORK       Required scratch root for isolated builds; use outside home quota.
  REPRO_ENV_ROOT   Existing environments with TAG/config and TAG/bin/python3.
                   Default: $REPRO_WORK/envs (created by 01_make_env.sh).
  RESULTS_ROOT    Shared persistent results root; default repo/results/reruns.
  RUN_ID          Optional unique label; default UTC timestamp + process ID.
  PYTHON          Snapshot helper interpreter; default python3 (3.9+).
  GEMM_PROCESSES  Full suite GEMM subprocess repeats; default 3.
  SLICES          Full suite GEMM slice values; default "1 3".

mnist: two seed-0 repro runs per selected GPU; all three methods on first GPU,
       seeds 0..4 at cuda124, seed 0 otherwise; one stage profile per method;
       RepDL's official example. torchvision and pinned RepDL must be installed.
kernels: conv, pool/Adam, exact sums, custom exp/log/softmax, operation suite.
full: mnist + kernels; at cuda124 also standalone GEMM and 14 native math
      functions. GPU execution only; no CPU training/exhaustive CPU experiments.
      Tests may use CPU reference calculations to check GPU results. Can take hours.
a100: comparable MNIST additions on first GPU: three timing repeats, seeds 1..4,
      profiles, batches 50/200/500/1000 (repro/stock, two repeats).
EOF
}
while (($#)); do
  case "$1" in
    --source|--suite|--configs|--gpus)
      (($# >= 2)) || { usage; exit 2; }
      case "$1" in
        --source) SOURCE=$2 ;; --suite) SUITE=$2 ;;
        --configs) CONFIGS=$2 ;; --gpus) GPUS=$2 ;;
      esac
      shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --ncu) NCU=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done
[[ -n "$SOURCE" ]] || { echo 'Choose --source explicitly; see --help.' >&2; exit 2; }
[[ "$SOURCE" != working-tree ]] || SOURCE=$ROOT
case "$SUITE" in mnist|kernels|full|a100) ;; *) usage; exit 2 ;; esac
if [[ "$CONFIGS" == auto ]]; then
  case "$HOST" in
    cuda2|cuda5) CONFIGS='cuda124 cuda126 cuda130' ;;
    cuda3) CONFIGS='cuda124 cuda126' ;;
    *) echo 'Unknown host: supply --configs explicitly.' >&2; exit 2 ;;
  esac
fi
read -r -a TAGS <<< "$CONFIGS"
((${#TAGS[@]})) || { echo 'Empty configuration list'; exit 2; }
for tag in "${TAGS[@]}"; do
  case "$tag" in cuda124|cuda126|cuda130) ;; *) echo "Unsupported config: $tag"; exit 2 ;; esac
done
if ((NCU)) && [[ "$SUITE" != a100 ]]; then echo '--ncu requires --suite a100'; exit 2; fi
if ((DRY)); then
  printf 'Host: %s\nSuite: %s\nSource: %s\nGPUs: %s\n' "$HOST" "$SUITE" "$SOURCE" "${GPUS:-all}"
  for tag in "${TAGS[@]}"; do
    seeds=0; [[ "$tag" != cuda124 ]] || seeds='0 1 2 3 4'
    printf 'Config %s: historical MNIST seeds [%s]; isolated build and results\n' "$tag" "$seeds"
  done
  printf 'Output: %s/%s/<run-id>/<config>/\n' "${RESULTS_ROOT:-$ROOT/results/reruns}" "$HOST"
  exit 0
fi
: "${REPRO_WORK:?Set REPRO_WORK to a scratch directory outside your home quota}"
PY=${PYTHON:-python3}
SOURCE=$("$PY" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$SOURCE")
ENV_ROOT=${REPRO_ENV_ROOT:-$REPRO_WORK/envs}
ENV_ROOT=$("$PY" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$ENV_ROOT")
RESULTS_ROOT=${RESULTS_ROOT:-$ROOT/results/reruns}
mkdir -p "$RESULTS_ROOT" "$REPRO_WORK"
RESULTS_ROOT=$(cd "$RESULTS_ROOT" && pwd)
REPRO_WORK=$(cd "$REPRO_WORK" && pwd)
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_$$}
[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid RUN_ID'; exit 2; }
OUT="$RESULTS_ROOT/$HOST/$RUN_ID"
WORK="$REPRO_WORK/reruns/$HOST/$RUN_ID"
mkdir -p "$RESULTS_ROOT/$HOST" "$REPRO_WORK/reruns/$HOST"
mkdir "$OUT" "$WORK"  # Refuse existing run IDs rather than overwrite results.
exec 3>&1
exec > "$OUT/runner.log" 2>&1
progress() { echo "$*"; echo "$*" >&3; }
progress "Results: $OUT"
progress "Build workspace: $WORK"
ALL_FINISHED=0
trap 'rc=$?; if ((ALL_FINISHED == 0)); then rc=1; fi; if ((rc)); then echo "FAILED ($rc); inspect runner.log" > "$OUT/status.txt"; fi; trap - EXIT; exit "$rc"' EXIT
unset REPRO_BUILD REPRO_CPU_ONLY
export PYTHONDONTWRITEBYTECODE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [[ -z "${GPUS:-}" ]]; then
  [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] || {
    echo 'CUDA_VISIBLE_DEVICES is already set. Supply --gpus with allocated physical indices.'; exit 2;
  }
  GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')
fi
read -r -a GPU_LIST <<< "$GPUS"
((${#GPU_LIST[@]})) || { echo 'No GPUs selected'; exit 2; }
for gpu in "${GPU_LIST[@]}"; do [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "Invalid GPU index: $gpu"; exit 2; }; done
if [[ "$SUITE" == a100 ]]; then GPU_LIST=("${GPU_LIST[0]}"); GPUS=${GPU_LIST[0]}; fi
export GPUS
"$PY" "$SUPPORT" snapshot "$SOURCE" "$WORK/snapshot" "$OUT/source_manifest.json"
tar -czf "$OUT/source.tar.gz" -C "$WORK/snapshot" .
cp "$0" "$SUPPORT" "$OUT/"
printf 'suite=%s\nconfigs=%s\ngpus=%s\nncu=%s\nenv_root=%s\n' "$SUITE" "$CONFIGS" "$GPUS" "$NCU" "$ENV_ROOT" > "$OUT/plan.txt"

run_config() {
  tag=$1
  DEST="$OUT/$tag"
  CFGWORK="$WORK/$tag"
  mkdir -p "$DEST/environment" "$DEST/logs" "$CFGWORK/tmp"
  echo RUNNING > "$DEST/status.txt"
  FAIL=0
  FINISHED=0
  trap 'rc=$?; if ((FINISHED == 0)); then rc=1; fi; "$PY" "$SUPPORT" collect "$CFGWORK/source" "$DEST/kernel_results" || rc=1; if ((rc)); then echo FAILED > "$DEST/status.txt"; else echo COMPLETE > "$DEST/status.txt"; fi; trap - EXIT; exit "$rc"' EXIT
  cp -R "$WORK/snapshot" "$CFGWORK/source"
  SRC="$CFGWORK/source"
  VENV="$ENV_ROOT/$tag"
  [[ -f "$VENV/config" && -x "$VENV/bin/python3" ]] || {
    echo "Missing environment $VENV; prepare with experiments/01_make_env.sh (see guide)."; exit 1;
  }
  CUDA_MODULE=
  source "$VENV/config"
  if [[ -n "$CUDA_MODULE" ]]; then
    type module >/dev/null 2>&1 || { echo 'Initialize environment modules in this shell first.'; exit 1; }
    module load "$CUDA_MODULE"
  fi
  source "$VENV/bin/activate"
  PY="$VENV/bin/python3"
  export TMPDIR="$CFGWORK/tmp" RESULT_TAG="$tag" CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" REPRO_ARCH=native
  command -v nvcc >/dev/null
  nvcc --version > "$DEST/environment/nvcc.txt"
  nvidia-smi -q > "$DEST/environment/gpu.txt"
  lscpu > "$DEST/environment/cpu.txt"
  "$PY" -m pip freeze > "$DEST/environment/packages.txt"
  cp "$VENV/config" "$DEST/environment/config.txt"
  "$PY" - "$tag" "$DEST/environment/nvcc.txt" <<'PY'
import sys, re, torch
expected = {'cuda124': ('2.6.0', '12.4'), 'cuda126': ('2.7.1', '12.6'), 'cuda130': ('2.9.0', '13.0')}
version, cuda = expected[sys.argv[1]]
assert torch.__version__.split('+')[0] == version, torch.__version__
assert torch.version.cuda == cuda, torch.version.cuda
assert re.search(r'release ' + re.escape(cuda) + r'[,\s]', open(sys.argv[2]).read()), 'Wrong nvcc toolkit'
assert torch.cuda.is_available(), 'CUDA unavailable'
print(sys.version, torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
PY
  : > "$DEST/commands.log"
  : > "$DEST/status.tsv"
  : > "$DEST/expected_runs.tsv"
  run() {
    local label=$1 rc=0; shift
    printf '%s: ' "$label" >> "$DEST/commands.log"
    printf '%q ' "$@" >> "$DEST/commands.log"
    printf '\n' >> "$DEST/commands.log"
    progress "[$HOST/$tag] $label"
    "$@" > "$DEST/logs/$label.log" 2>&1 || rc=$?
    printf '%s\t%s\n' "$label" "$rc" >> "$DEST/status.tsv"
    if ((rc)); then FAIL=1; progress "FAILED: $label (exit $rc); see $DEST/logs/$label.log"; tail -n 12 "$DEST/logs/$label.log"; fi
  }
  train() {
    local label=$1 mode=$2 seed=$3 batch=$4 prof=$5
    local flags=()
    ((prof == 0)) || flags=(--profile)
    local output="$DEST/mnist/$label.txt"
    printf '%s\t%s\t%s\t%s\t%s\n' "$output" "$mode" "$seed" "$batch" "$prof" >> "$DEST/expected_runs.tsv"
    run "$label" "$PY" "$SRC/repro_torch/mnist_repro.py" --mode "$mode" --device cuda \
      --seed "$seed" --batch "$batch" --epochs 2 --lr 0.01 --slices 3 --tag "$tag" \
      --data "$SRC/repro_torch/data" --out "$output" ${flags[@]+"${flags[@]}"}
  }
  if [[ "$SUITE" != kernels ]]; then
    "$PY" -c 'import torch, repdl; print(repdl.__file__)' > "$DEST/environment/repdl.txt"
    for gpu in "${GPU_LIST[@]}"; do
      export CUDA_VISIBLE_DEVICES=$gpu
      export REPRO_BUILD="$CFGWORK/package_gpu$gpu"
      mkdir -p "$REPRO_BUILD/repro_torch"
      cp "$SRC/repro_torch/repro_torch/"*.py "$REPRO_BUILD/repro_torch/"
      (cd "$SRC/repro_torch"; "$PY" setup.py build_ext --force \
        --build-lib "$REPRO_BUILD" --build-temp "$CFGWORK/objects_gpu$gpu") > "$DEST/logs/build_gpu$gpu.log" 2>&1
      PYTHONPATH="$REPRO_BUILD" "$PY" -c 'import repro_torch; print(repro_torch._backend.__file__)' > "$DEST/environment/loaded_gpu$gpu.txt"
      sha256sum "$REPRO_BUILD/repro_torch/"*.so > "$DEST/environment/binary_gpu$gpu.sha256"
      if [[ "$SUITE" != a100 ]]; then
        for repeat in 1 2; do train "gpu${gpu}_repro_correctness_$repeat" repro 0 100 0; done
      fi
      [[ "$gpu" == "${GPU_LIST[0]}" ]] || continue
      seeds=(0); [[ "$tag" != cuda124 ]] || seeds=(0 1 2 3 4)
      if [[ "$SUITE" == a100 ]]; then
        seeds=(1 2 3 4)
        for repeat in 1 2 3; do
          for mode in repro repdl stock; do train "gpu${gpu}_${mode}_baseline_$repeat" "$mode" 0 100 0; done
        done
        for batch in 50 200 500 1000; do
          for repeat in 1 2; do
            for mode in repro stock; do train "gpu${gpu}_${mode}_batch${batch}_repeat$repeat" "$mode" 0 "$batch" 0; done
          done
        done
      fi
      for mode in repro repdl stock; do
        for seed in "${seeds[@]}"; do train "gpu${gpu}_${mode}_seed$seed" "$mode" "$seed" 100 0; done
        train "gpu${gpu}_${mode}_profile" "$mode" 0 100 1
      done
      if [[ "$SUITE" != a100 ]]; then
        if [[ -f "$VENV/RepDL/examples/mnist_trainig.py" ]] && "$PY" -c 'import torchvision' >/dev/null 2>&1; then
          mkdir -p "$DEST/repdl_official"
          (cd "$DEST/repdl_official"; "$PY" "$VENV/RepDL/examples/mnist_trainig.py") > "$DEST/logs/repdl_official.log" 2>&1 \
            && printf 'repdl_official\t0\n' >> "$DEST/status.tsv" \
            || { FAIL=1; printf 'repdl_official\t1\n' >> "$DEST/status.tsv"; }
        else
          echo 'RepDL official example unavailable: install torchvision and retain VENV/RepDL checkout.'
          printf 'repdl_official\tMISSING\n' >> "$DEST/status.tsv"; FAIL=1
        fi
      fi
      if ((NCU)); then
        mkdir -p "$DEST/profiling"
        for batch in 100 500; do
          run "ncu_wgrad_batch$batch" ncu --set full --kernel-name-base function \
            --kernel-name 'regex:weight_grad_kernel' --launch-skip 4 --launch-count 2 \
            --export "$DEST/profiling/wgrad_batch$batch" "$PY" "$SRC/repro_torch/mnist_repro.py" \
            --mode repro --device cuda --batch "$batch" --seed 0 --slices 3 --max-steps 10
          run "ncu_export_batch$batch" ncu --import "$DEST/profiling/wgrad_batch$batch.ncu-rep" --page details
        done
      fi
    done
    run check_training "$PY" "$SUPPORT" check "$DEST" "$DEST/expected_runs.tsv"
  fi
  if [[ "$SUITE" == kernels || "$SUITE" == full ]]; then
    export CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}"
    mkdir -p "$CFGWORK/kernel_bin" "$DEST/kernel_results"
    cd "$SRC"
    nvflags=(-O3 -std=c++17 -arch=native -lineinfo --fmad=false
      -Irepro_gemm_cuda -Irepro_sum_cuda -Irepro_math_cuda -Irepro_conv_cuda -Irepro_nn_cuda)
    for target in test_repro_conv bench_repro_conv test_repro_nn test_repro_sum bench_repro_sum \
                  test_repro_softmax bench_repro_math exhaustive_math suite_gpu; do
      case "$target" in
        *conv) sources=(repro_gemm_cuda/repro_gemm.cu repro_conv_cuda/repro_conv.cu "repro_conv_cuda/$target.cu") ;;
        test_repro_nn) sources=(repro_nn_cuda/repro_nn.cu repro_nn_cuda/test_repro_nn.cu) ;;
        *sum) sources=(repro_sum_cuda/repro_sum.cu "repro_sum_cuda/$target.cu") ;;
        test_repro_softmax|bench_repro_math) sources=(repro_sum_cuda/repro_sum.cu repro_math_cuda/repro_softmax.cu "repro_math_cuda/$target.cu") ;;
        exhaustive_math) sources=(repro_math_cuda/exhaustive_math.cu) ;;
        suite_gpu) sources=(repro_gemm_cuda/repro_gemm.cu repro_sum_cuda/repro_sum.cu repro_math_cuda/repro_softmax.cu
          repro_conv_cuda/repro_conv.cu repro_nn_cuda/repro_nn.cu gpt2_ops_suite/gpt2_ops_suite.cu -lcublas) ;;
      esac
      run "build_$target" nvcc "${nvflags[@]}" "${sources[@]}" -o "$CFGWORK/kernel_bin/$target"
    done
    for gpu in "${GPU_LIST[@]}"; do
      export CUDA_VISIBLE_DEVICES=$gpu
      for target in test_repro_conv test_repro_nn test_repro_sum test_repro_softmax; do
        run "gpu${gpu}_$target" "$CFGWORK/kernel_bin/$target"
      done
      for target in bench_repro_conv bench_repro_sum bench_repro_math; do
        run "gpu${gpu}_$target" "$CFGWORK/kernel_bin/$target" 30
      done
      run "gpu${gpu}_repro_math" "$CFGWORK/kernel_bin/exhaustive_math" \
        "$DEST/kernel_results/repro_math_gpu$gpu.txt" repro_expf repro_logf
      run "gpu${gpu}_operations" "$CFGWORK/kernel_bin/suite_gpu" --device gpu --tokens "${TOKENS:-1024}" \
        --out "$DEST/kernel_results/operations_gpu$gpu.txt"
      if [[ "$SUITE" == full && "$tag" == cuda124 ]]; then
        run "gpu${gpu}_native_math" "$CFGWORK/kernel_bin/exhaustive_math" "$DEST/kernel_results/native_math_gpu$gpu.txt" \
          expf __expf exp_via_double logf log1pf expm1f tanhf erff sigmoid_torch sqrtf rsqrtf \
          'recip_1/x' 'div_x/10' 'mul_x*0.1f'
      fi
    done
    if [[ "$SUITE" == full && "$tag" == cuda124 ]]; then
      export CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}"
      run gemm bash "$SRC/repro_gemm_cuda/run_repro.sh" "${GEMM_PROCESSES:-3}"
    fi
  fi
  FINISHED=1
  exit "$FAIL"
}
OVERALL=0
for tag in "${TAGS[@]}"; do
  # Do not put the subshell in an if/|| expression: that would disable errexit
  # inside run_config and allow setup/build failures to masquerade as success.
  set +e
  (set -e; run_config "$tag")
  rc=$?
  set -e
  ((rc == 0)) || OVERALL=1
done
if ((OVERALL)); then progress "FAILED: inspect $OUT/runner.log and per-configuration logs/status"; exit 1; fi
if [[ "$SUITE" != kernels ]]; then
  cat "$OUT"/*/expected_runs.tsv > "$OUT/expected_runs.tsv"
  "$PY" "$SUPPORT" check "$OUT" "$OUT/expected_runs.tsv"
fi
echo COMPLETE > "$OUT/status.txt"
ALL_FINISHED=1
progress "Completed. Preserve the entire directory: $OUT"
