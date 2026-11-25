#!/usr/bin/env bash
# Run INSIDE the A100 CUDA development container, from any directory.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SUPPORT="$ROOT/experiments/rerun_support.py"
PY=${REPRO_PYTHON:-}
NSIGHT=1
DRY=0
CONFIG=legacy
BASELINE_REPEATS=${BASELINE_REPEATS:-3}
[[ "$BASELINE_REPEATS" =~ ^[1-9][0-9]*$ ]] || { echo 'BASELINE_REPEATS must be a positive integer'; exit 2; }
usage() {
  cat <<'EOF'
Usage: bash experiments/run_a100_no_atomics.sh [--python /path/to/venv/bin/python3]
                                             [--config legacy|cuda124|cuda126|cuda130]
                                             [--skip-ncu] [--dry-run]
Run inside the CUDA development Singularity container on an allocated A100.
Default Python: REPRO_PYTHON, /venv/bin/python3, then repository/.venv/bin/python3.
Default: smoke check, 43 full MNIST runs, four Nsight captures, checks, archive.
  --skip-ncu   Explicitly omit Nsight; recorded in plan.json.
  --dry-run    Print the plan without checking/using an NVIDIA GPU.
  --config    legacy: torch 2.6/cu124 + nvcc 12.6 (default); other tags require
              matching toolkit/wheel and the same PyTorch versions as CIMS.
Environment:
  A100_RESULTS_ROOT  Persistent output root (default repo/results/a100_no_atomics)
  A100_BUILD_ROOT    Scratch build root (default repo/build/a100_no_atomics)
  RUN_ID            Unique label, default UTC timestamp + PID; never overwritten
  CUDA_VISIBLE_DEVICES  Preserve your allocation; defaults to 0 only if unset
  MAX_JOBS          Build parallelism, default 4
  BASELINE_REPEATS  Timing repeats per method, default 3 (43 full runs)
This script does not install dependencies or modify the atomic baseline checkout.
EOF
}
while (($#)); do
  case "$1" in
    --python) (($# >= 2)) || { usage; exit 2; }; PY=$2; shift 2 ;;
    --config) (($# >= 2)) || { usage; exit 2; }; CONFIG=$2; shift 2 ;;
    --skip-ncu) NSIGHT=0; shift ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done
case "$CONFIG" in
  legacy) EXPECT_TORCH=2.6.0; EXPECT_CUDA=12.4; EXPECT_NVCC=12.6;;
  cuda124) EXPECT_TORCH=2.6.0; EXPECT_CUDA=12.4; EXPECT_NVCC=12.4;;
  cuda126) EXPECT_TORCH=2.7.1; EXPECT_CUDA=12.6; EXPECT_NVCC=12.6;;
  cuda130) EXPECT_TORCH=2.9.0; EXPECT_CUDA=13.0; EXPECT_NVCC=13.0;;
  *) echo "Unknown A100 configuration: $CONFIG"; exit 2;;
esac
export A100_CONFIG="$CONFIG" EXPECT_TORCH EXPECT_CUDA EXPECT_NVCC
if ((DRY)); then
  cat <<EOF
Configuration: $CONFIG; PyTorch $EXPECT_TORCH; wheel CUDA $EXPECT_CUDA; nvcc $EXPECT_NVCC
Source: $ROOT (explicit CUDA atomic calls cause an error)
Build: isolated fresh sm_80 extension; verify imported extension path
Smoke: repro, batch 100, seed 0, 10 steps
Baseline: repro/repdl/stock, batch 100, seed 0, $BASELINE_REPEATS repeats ($((3 * BASELINE_REPEATS)) runs)
Accuracy: repro/repdl/stock, seeds 1..4, batch 100 (12 runs)
Scaling: repro/stock, batches 50/200/500/1000, 2 repeats (16 runs)
Stages: repro/repdl/stock, batch 100, seed 0, --profile (3 runs)
Repeatability: 3 additional repro stage-profile runs, batch 100, seed 0
Nsight: $NSIGHT (basic launch sample, conv/GEMM sample, weight gradients at 100/500)
Checks: complete runs, matching repro hashes, recorded source/software/build identity
Output: ${A100_RESULTS_ROOT:-$ROOT/results/a100_no_atomics}/<host>/<run-id>/
Archive: sibling <run-id>.tar.gz plus SHA256
EOF
  exit 0
fi
if [[ -z "$PY" ]]; then
  if [[ -x /venv/bin/python3 ]]; then PY=/venv/bin/python3
  else PY="$ROOT/.venv/bin/python3"; fi
fi
[[ -x "$PY" && -f "$SUPPORT" ]] || { echo 'Missing venv Python or experiments/rerun_support.py'; exit 2; }
PY=$("$PY" -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$PY")
command -v nvcc >/dev/null || { echo 'nvcc missing: enter the CUDA development container first.'; exit 2; }
[[ -x /usr/bin/g++ && -x /usr/bin/gcc ]] || { echo 'Expected container compilers /usr/bin/g++ and gcc'; exit 2; }
export CXX=/usr/bin/g++ CC=/usr/bin/gcc
export LDSHARED='/usr/bin/g++ -shared -pthread -B/usr/bin/'
CPP_RUNTIME=$(/usr/bin/g++ -print-file-name=libstdc++.so.6)
export LD_PRELOAD="$CPP_RUNTIME${LD_PRELOAD:+:$LD_PRELOAD}"
export REPRO_ARCH=sm_80 MAX_JOBS=${MAX_JOBS:-4} PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-0}
unset REPRO_BUILD REPRO_CPU_ONLY
HOST=$(hostname -s)
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_$$}
[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid RUN_ID'; exit 2; }
RESULTS=${A100_RESULTS_ROOT:-$ROOT/results/a100_no_atomics}
BUILDS=${A100_BUILD_ROOT:-$ROOT/build/a100_no_atomics}
mkdir -p "$RESULTS/$HOST" "$BUILDS/$HOST"
RESULTS=$(cd "$RESULTS/$HOST" && pwd)
BUILDS=$(cd "$BUILDS/$HOST" && pwd)
OUT="$RESULTS/$RUN_ID"
WORK="$BUILDS/$RUN_ID"
mkdir "$OUT" "$WORK"
exec 3>&1
exec > "$OUT/runner.log" 2>&1
progress() { echo "$*"; echo "$*" >&3; }
FINISHED=0
FAIL=0
finish() {
  local rc=$?
  trap - EXIT
  if ((FINISHED == 0 || FAIL)); then rc=1; fi
  if ((rc)); then echo FAILED > "$OUT/status.txt"; else echo COMPLETE > "$OUT/status.txt"; fi
  # Preserve partial results as well as successful runs.
  if tar -czf "$RESULTS/$RUN_ID.tar.gz" -C "$RESULTS" "$RUN_ID"; then
    "$PY" - "$RESULTS/$RUN_ID.tar.gz" <<'PY'
import hashlib, pathlib, sys
p = pathlib.Path(sys.argv[1])
h = hashlib.sha256()
with p.open('rb') as f:
    for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
p.with_name(p.name + '.sha256').write_text(h.hexdigest() + '  ' + p.name + '\n')
PY
    progress "Download: $RESULTS/$RUN_ID.tar.gz"
  else
    rc=1
    echo 'FAILED: result archive creation' > "$OUT/status.txt"
  fi
  progress "Status: $(cat "$OUT/status.txt"); detailed logs: $OUT"
  exit "$rc"
}
trap finish EXIT
progress "A100 experiments: $OUT"
mkdir -p "$OUT/environment" "$OUT/logs" "$OUT/profiling"
cp "$0" "$SUPPORT" "$OUT/"
"$PY" "$SUPPORT" snapshot "$ROOT" "$WORK/source" "$OUT/source_manifest.json"
"$PY" - "$OUT/source_manifest.json" <<'PY'
import json, sys
bad = json.load(open(sys.argv[1]))['explicit_atomic_call_lines']
if bad: raise SystemExit('Refusing old GPU atomic source:\n' + '\n'.join(bad))
print('PASS: snapshotted CUDA source has no explicit atomic calls')
PY
SRC="$WORK/source"
tar -czf "$OUT/source.tar.gz" -C "$SRC" .
"$PY" - "$OUT/plan.json" "$NSIGHT" <<'PY'
import json, os, sys
repeats = int(os.environ.get('BASELINE_REPEATS', '3'))
json.dump(dict(protocol='A100 no-atomics', full_training_runs=34 + 3 * repeats, baseline_repeats=repeats, epochs=2, slices=3,
               configuration=os.environ['A100_CONFIG'], expected_torch=os.environ['EXPECT_TORCH'],
               expected_cuda=os.environ['EXPECT_CUDA'], expected_nvcc=os.environ['EXPECT_NVCC'],
               container_image=os.environ.get('A100_CONTAINER_IMAGE', os.environ.get('SINGULARITY_CONTAINER')),
               cuda_home=os.environ.get('CUDA_HOME'),
               baseline_batch=100, lr=0.01, nsight_requested=bool(int(sys.argv[2])),
               cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')),
          open(sys.argv[1], 'w'), indent=2)
PY
nvcc --version > "$OUT/environment/nvcc.txt"
nvidia-smi -q > "$OUT/environment/gpu.txt"
lscpu > "$OUT/environment/cpu.txt"
"$PY" -m pip freeze > "$OUT/environment/packages.txt"
"$PY" - <<'PY'
import os, re, subprocess, sys, torch, repdl
assert torch.cuda.is_available(), 'CUDA unavailable'
assert 'A100' in torch.cuda.get_device_name(0), 'This runner is for the A100 comparison'
assert torch.cuda.get_device_capability(0) == (8, 0), 'sm_80 required'
assert torch.__version__.split('+')[0] == os.environ['EXPECT_TORCH'], ('Wrong torch', torch.__version__)
assert torch.version.cuda == os.environ['EXPECT_CUDA'], ('Wrong wheel CUDA', torch.version.cuda)
match = re.search(r'release (\d+\.\d+)', subprocess.check_output(['nvcc', '--version'], text=True))
assert match and match.group(1) == os.environ['EXPECT_NVCC'], 'nvcc does not match requested configuration'
print(sys.version, torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), repdl.__file__)
PY
if ((NSIGHT)); then
  command -v ncu >/dev/null || { echo 'ncu missing: use --skip-ncu explicitly to omit it'; exit 1; }
  command -v ncu > "$OUT/environment/ncu_path.txt"
  ncu --version > "$OUT/environment/ncu.txt"
fi
export REPRO_BUILD="$WORK/package"
mkdir -p "$REPRO_BUILD/repro_torch"
cp "$SRC/repro_torch/repro_torch/"*.py "$REPRO_BUILD/repro_torch/"
progress 'Building fresh CUDA extension (sm_80)'
(cd "$SRC/repro_torch"; "$PY" setup.py build_ext --force \
  --build-lib "$REPRO_BUILD" --build-temp "$WORK/objects") > "$OUT/logs/build.log" 2>&1
PYTHONPATH="$REPRO_BUILD" "$PY" - "$OUT/environment/loaded_extension.json" <<'PY'
import hashlib, json, os, pathlib, sys
sys.path.insert(0, os.environ['REPRO_BUILD'])
import repro_torch
p = pathlib.Path(repro_torch._backend.__file__).resolve()
assert pathlib.Path(os.environ['REPRO_BUILD']).resolve() in p.parents, str(p)
assert repro_torch.has_cuda, 'Loaded CPU-only extension'
json.dump(dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()), open(sys.argv[1], 'w'), indent=2)
print('Loaded fresh extension:', p)
PY
: > "$OUT/commands.log"
: > "$OUT/status.tsv"
: > "$OUT/expected_runs.tsv"
run() {
  local label=$1 rc=0; shift
  progress "$label"
  printf '%s: ' "$label" >> "$OUT/commands.log"
  printf '%q ' "$@" >> "$OUT/commands.log"
  printf '\n' >> "$OUT/commands.log"
  "$@" > "$OUT/logs/$label.log" 2>&1 || rc=$?
  printf '%s\t%s\n' "$label" "$rc" >> "$OUT/status.tsv"
  if ((rc)); then FAIL=1; progress "FAILED: $label; see $OUT/logs/$label.log"; fi
}
train() {
  local label=$1 mode=$2 seed=$3 batch=$4 profile=$5
  local flags=()
  ((profile == 0)) || flags=(--profile)
  local path="$OUT/mnist/$label.txt"
  printf '%s\t%s\t%s\t%s\t%s\n' "$path" "$mode" "$seed" "$batch" "$profile" >> "$OUT/expected_runs.tsv"
  run "$label" "$PY" "$SRC/repro_torch/mnist_repro.py" --mode "$mode" --device cuda \
    --epochs 2 --batch "$batch" --seed "$seed" --lr 0.01 --slices 3 \
    --tag a100_no_atomics --out "$path" ${flags[@]+"${flags[@]}"}
}
run smoke "$PY" "$SRC/repro_torch/mnist_repro.py" --mode repro --device cuda \
  --batch 100 --seed 0 --slices 3 --max-steps 10 --out "$OUT/smoke.txt"
((FAIL == 0)) || exit 1
"$PY" - "$OUT/smoke.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
expected = '286b280c7e33bcb70aa2bf329f1e01a8e7764765e90f2305f478afdbe895f64d'
assert d['steps'] == 10 and d['hashes']['weights_epoch_1'] == expected, 'Smoke hash differs from atomic baseline; investigate before benchmarking'
print('PASS: 10-step weights match the recorded atomic baseline')
PY
for ((repeat=1; repeat<=BASELINE_REPEATS; repeat++)); do
  for mode in repro repdl stock; do train "baseline_${mode}_repeat$repeat" "$mode" 0 100 0; done
done
for batch in 50 200 500 1000; do
  for repeat in 1 2; do
    for mode in repro stock; do train "scaling_${mode}_batch${batch}_repeat$repeat" "$mode" 0 "$batch" 0; done
  done
done
for seed in 1 2 3 4; do
  for mode in repro repdl stock; do train "accuracy_${mode}_seed$seed" "$mode" "$seed" 100 0; done
done
for mode in repro repdl stock; do train "stages_$mode" "$mode" 0 100 1; done
for repeat in 1 2 3; do train "repeatability_repro_$repeat" repro 0 100 1; done
run check_training "$PY" "$SUPPORT" check "$OUT" "$OUT/expected_runs.tsv"
run compare_atomic_baseline "$PY" - "$OUT" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
expected = dict(weights_epoch_1='7fed5293c5306f9c9ec9879b7a4c5ad35cfacedbc5090a6b18801e62a3360d9d',
                weights_epoch_2='6b8ce5843c58dcd10a95e3339e634b66f95989e2972cc475427c8fb7b83ab2f2',
                test_logits='5f22405beb10931b28774aa7b2e560085ca15598d91d96e5f79b8781da1b357a',
                train_logits='ada4cbfcdb1f4cdb9cdf9de2d1c82389a7153e18feb3f87348755325be957a47')
failures = []
for p in (out / 'mnist').glob('*.json'):
    d = json.loads(p.read_text())
    if d['mode'] == 'repro' and d['seed'] == 0 and d['batch'] == 100:
        failures.extend(f'{p.name}: {k}' for k,v in expected.items() if d['hashes'].get(k) != v)
json.dump(dict(passed=not failures, failures=failures), open(out / 'atomic_baseline_check.json', 'w'), indent=2)
if failures: raise SystemExit('\n'.join(failures))
print('PASS: seed-0 batch-100 weights/logits match the recorded atomic baseline')
PY
if ((NSIGHT)); then
  run ncu_sample ncu --set basic --launch-skip 100 --launch-count 10 \
    --export "$OUT/profiling/repro_sample" "$PY" "$SRC/repro_torch/mnist_repro.py" \
    --mode repro --device cuda --batch 100 --seed 0 --slices 3 --max-steps 20
  run ncu_conv_gemm ncu --set full --kernel-name-base function \
    --kernel-name 'regex:forward_kernel|weight_grad_kernel|input_grad_kernel|gemm_wmma_i8|gemm_tiled_dp4a|gemm_reference' \
    --launch-skip 20 --launch-count 12 --export "$OUT/profiling/repro_conv_gemm" \
    "$PY" "$SRC/repro_torch/mnist_repro.py" --mode repro --device cuda --batch 100 --seed 0 --slices 3 --max-steps 20
  for batch in 100 500; do
    run "ncu_wgrad_batch$batch" ncu --set full --kernel-name-base function \
      --kernel-name 'regex:weight_grad_kernel' --launch-skip 4 --launch-count 2 \
      --export "$OUT/profiling/wgrad_batch$batch" "$PY" "$SRC/repro_torch/mnist_repro.py" \
      --mode repro --device cuda --batch "$batch" --seed 0 --slices 3 --max-steps 10
  done
  for report in repro_sample repro_conv_gemm wgrad_batch100 wgrad_batch500; do
    run "export_$report" ncu --import "$OUT/profiling/$report.ncu-rep" --page details
    cp "$OUT/logs/export_$report.log" "$OUT/profiling/$report.txt"
  done
fi
FINISHED=1
exit "$FAIL"
