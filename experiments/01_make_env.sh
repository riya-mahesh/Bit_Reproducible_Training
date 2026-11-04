#!/usr/bin/env bash
# Makes one configuration: a CUDA module, a matching PyTorch wheel, and RepDL built against them.
#   export REPRO_WORK=/scratch/$USER/repro
#   experiments/01_make_env.sh cuda124 cuda-12.4 2.6.0 cu124
#   arguments: TAG  CUDA_MODULE  TORCH_VERSION  WHEEL_CUDA
# Use the wheel whose CUDA matches the module, so one process doesn't load two CUDA runtimes.
# CUDA_MODULE may be "" on a machine where nvcc is already on PATH. PYTHON selects the interpreter.
set -euo pipefail
source "$(dirname "$0")/common.sh"
[ $# -eq 4 ] || { echo "usage: $0 TAG CUDA_MODULE TORCH_VERSION WHEEL_CUDA"; exit 1; }
TAG=$1; CUDA_MODULE=$2; TORCH=$3; WHEEL=$4
ENV_DIR="$REPRO_WORK/envs/$TAG"
if [ -n "$CUDA_MODULE" ] && command -v module >/dev/null 2>&1; then module load "$CUDA_MODULE"; fi
command -v nvcc >/dev/null || { echo "nvcc not found: check the module name with 00_show_env.sh"; exit 1; }
mkdir -p "$ENV_DIR"
"${PYTHON:-python3}" -m venv "$ENV_DIR"
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"
pip install -q --upgrade pip wheel "setuptools<70"
pip install -q "torch==$TORCH" --index-url "https://download.pytorch.org/whl/$WHEEL"
pip install -q numpy mpmath
python3 -c "import torch; print('torch', torch.__version__, 'wheel CUDA', torch.version.cuda, 'GPU visible:', torch.cuda.is_available())"
echo "nvcc: $(nvcc --version | sed -n 2p)"

# RepDL: its build compiles CUDA kernels with the same toolkit. A failure here doesn't stop ours.
if [ ! -d "$ENV_DIR/RepDL" ]; then
  git clone -q https://github.com/microsoft/RepDL.git "$ENV_DIR/RepDL"
  git -C "$ENV_DIR/RepDL" checkout -q acd95bb   # the same RepDL everywhere
fi
# --no-build-isolation: RepDL's setup.py imports torch, so its build must see this environment.
if (cd "$ENV_DIR/RepDL" && pip install -q --no-build-isolation . ) ; then
  (cd /tmp && python3 -c "import repdl; print('RepDL installed')")
  # Pinned: an unpinned torchvision can replace torch with a newer version. Torch 2.Y.Z pairs with 0.(Y+15).Z.
  IFS=. read -r _ TORCH_MINOR TORCH_PATCH <<<"$TORCH"
  pip install -q "torchvision==0.$((TORCH_MINOR + 15)).$TORCH_PATCH" --index-url "https://download.pytorch.org/whl/$WHEEL" || echo "torchvision unavailable: RepDL's own example will be skipped"
else
  echo "WARNING: RepDL did not build in this configuration; --mode repdl will not run"
fi
python3 -c "
import sys, torch
if torch.__version__.split('+')[0] != '$TORCH' or (torch.version.cuda or '').replace('.', '') != '$WHEEL'[2:]:
    sys.exit('wrong torch in this environment: %s, wheel CUDA %s' % (torch.__version__, torch.version.cuda))
"
{ echo "CUDA_MODULE='$CUDA_MODULE'"; echo "TORCH='$TORCH'"; echo "WHEEL='$WHEEL'"; } > "$ENV_DIR/config"
echo "ready: $TAG"
