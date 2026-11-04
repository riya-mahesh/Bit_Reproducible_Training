#!/usr/bin/env bash
# Builds repro_torch and runs MNIST on every GPU (or GPUS="0 1"): ours twice, then stock PyTorch.
# Then compares every run with the others and with any results copied from other machines.
#   module load cuda-12.4; ./run_mnist.sh
# Needs a CUDA build of PyTorch that can find nvcc. RepDL: also pass REPDL=1 if repdl is installed.
set -euo pipefail
cd "$(dirname "$0")"
command -v nvcc >/dev/null || { echo "nvcc not found (on CIMS: module load cuda-12.4)"; exit 1; }
python3 -c "import torch; assert torch.cuda.is_available(), 'this PyTorch has no CUDA support'; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
HOST=$(hostname -s)${RESULT_TAG:+_$RESULT_TAG}; OUT=results/$HOST; mkdir -p "$OUT"
python3 setup.py build_ext --inplace 2>&1 | tail -n 3
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [ -n "${GPUS:-}" ]; then read -r -a GPU_LIST <<<"$GPUS"
else read -r -a GPU_LIST <<<"$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')"; fi
for g in "${GPU_LIST[@]}"; do
  export CUDA_VISIBLE_DEVICES=$g
  name=$(nvidia-smi -i "$g" --query-gpu=name --format=csv,noheader | tr ' ' '_')
  python3 mnist_repro.py --device cuda --out "$OUT/gpu${g}_${name}_repro1.txt" | grep -v "^  epoch"
  python3 mnist_repro.py --device cuda --out "$OUT/gpu${g}_${name}_repro2.txt" | grep -v "^  epoch"
  python3 mnist_repro.py --device cuda --mode stock --out "$OUT/gpu${g}_${name}_stock.txt" | grep -v "^  epoch"
  [ -z "${REPDL:-}" ] || python3 mnist_repro.py --device cuda --mode repdl --out "$OUT/gpu${g}_${name}_repdl.txt" | grep -v "^  epoch"
done
./compare_mnist.sh results/*/*repro*.txt
