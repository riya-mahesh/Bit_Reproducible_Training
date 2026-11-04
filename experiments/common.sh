# Shared by the experiment scripts: where things live, and activating one configuration.
#
# REPRO_WORK must be outside the CIMS home, whose quota is 4 GB; one environment takes several.
: "${REPRO_WORK:?set REPRO_WORK to a scratch folder, for example: export REPRO_WORK=/scratch/\$USER/repro}"
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PIP_CACHE_DIR="$REPRO_WORK/pip-cache"
export TMPDIR="$REPRO_WORK/tmp"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

# load_config TAG: loads the CUDA module and the Python environment made by 01_make_env.sh.
load_config() {
  TAG=$1
  ENV_DIR="$REPRO_WORK/envs/$TAG"
  [ -f "$ENV_DIR/config" ] || { echo "no environment '$TAG': run experiments/01_make_env.sh first"; exit 1; }
  # shellcheck disable=SC1091
  source "$ENV_DIR/config"
  if [ -n "$CUDA_MODULE" ] && command -v module >/dev/null 2>&1; then module load "$CUDA_MODULE"; fi
  # shellcheck disable=SC1091
  source "$ENV_DIR/bin/activate"
  command -v nvcc >/dev/null || { echo "nvcc not found after loading '$CUDA_MODULE'"; exit 1; }
  export RESULT_TAG=$TAG
}

gpu_list() {
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  if [ -n "${GPUS:-}" ]; then echo "$GPUS"
  else nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' '; fi
}

gpu_name() { nvidia-smi -i "$1" --query-gpu=name --format=csv,noheader | tr ' ' '_'; }

# build_repro GPU_INDEX: builds repro_torch for this GPU and CUDA version once, and sets REPRO_BUILD.
# -arch=native is specific to the GPU, so each GPU model gets its own build.
build_repro() {
  local key
  key=$(gpu_name "$1")
  export REPRO_BUILD="$ENV_DIR/repro_build_$key"
  # Rebuilds when any kernel or binding source is newer than the build.
  local so
  so=$(ls "$REPRO_BUILD"/repro_torch/_backend*.so 2>/dev/null | head -n 1)
  if [ -z "$so" ] || [ -n "$(find "$REPO"/repro_*_cuda "$REPO"/repro_torch/src "$REPO"/repro_torch/setup.py -type f \( -name '*.cu' -o -name '*.cuh' -o -name '*.h' -o -name '*.cpp' -o -name 'setup.py' \) -newer "$so" -print -quit)" ]; then
    echo "=== building repro_torch for $key with $(nvcc --version | sed -n 2p) ==="
    rm -f "$REPRO_BUILD"/repro_torch/_backend*.so
    rm -rf "$ENV_DIR/tmp_$key"
    mkdir -p "$REPRO_BUILD/repro_torch"
    (cd "$REPO/repro_torch" && CUDA_VISIBLE_DEVICES=$1 python3 setup.py build_ext \
        --build-lib "$REPRO_BUILD" --build-temp "$ENV_DIR/tmp_$key" 2>&1 | tail -n 3)
    rm -f "$REPO"/repro_torch/repro_torch/_backend*.so
  fi
  cp "$REPO"/repro_torch/repro_torch/*.py "$REPRO_BUILD/repro_torch/"
}
