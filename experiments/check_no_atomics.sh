#!/usr/bin/env bash
# Run inside a CUDA development environment on an allocated GPU.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BUILD_DIR=${BUILD_DIR:-"$ROOT/build/no_atomics"}
NVCC=${NVCC:-nvcc}
REPRO_ARCH=${REPRO_ARCH:-native}
mkdir -p "$BUILD_DIR"
cd "$ROOT"

python3 - <<'PY'
from pathlib import Path
import re
files = [p for d in ('repro_gemm_cuda', 'repro_sum_cuda', 'repro_math_cuda',
                     'repro_conv_cuda', 'repro_nn_cuda', 'repro_torch/src', 'gpt2_ops_suite')
         for p in Path(d).glob('*') if p.suffix in ('.cu', '.cuh')]
bad = [str(p) for p in files if re.search(r'\batomic\w*\s*\(', p.read_text())]
if bad:
    raise SystemExit('GPU atomic calls found: ' + ', '.join(bad))
print('PASS: no atomic calls in project CUDA sources/headers')
PY

command -v "$NVCC" >/dev/null || { echo "nvcc is required for GPU validation" >&2; exit 1; }
flags=(-O3 -std=c++17 "-arch=$REPRO_ARCH" --fmad=false -Xcompiler=-ffp-contract=off
       -Irepro_gemm_cuda -Irepro_sum_cuda -Irepro_math_cuda -Irepro_conv_cuda -Irepro_nn_cuda -Irepro_torch/src)

"${CXX:-c++}" -O2 -std=c++17 -fno-fast-math -ffp-contract=off \
  repro_sum_cuda/test_private_acc.cpp -o "$BUILD_DIR/test_private_acc"
"$BUILD_DIR/test_private_acc"

"$NVCC" "${flags[@]}" repro_sum_cuda/repro_sum.cu repro_sum_cuda/test_repro_sum.cu -o "$BUILD_DIR/test_repro_sum"
"$BUILD_DIR/test_repro_sum"
"$NVCC" "${flags[@]}" repro_sum_cuda/repro_sum.cu repro_math_cuda/repro_softmax.cu \
  repro_math_cuda/test_repro_softmax.cu -o "$BUILD_DIR/test_repro_softmax"
"$BUILD_DIR/test_repro_softmax"
"$NVCC" "${flags[@]}" repro_gemm_cuda/repro_gemm.cu repro_conv_cuda/repro_conv.cu \
  repro_conv_cuda/test_repro_conv.cu -o "$BUILD_DIR/test_repro_conv"
"$BUILD_DIR/test_repro_conv"
"$NVCC" "${flags[@]}" repro_torch/src/kernels.cu repro_torch/src/test_repro_scan.cu -o "$BUILD_DIR/test_repro_scan"
"$BUILD_DIR/test_repro_scan"

# Compile the changed benchmark/diagnostic paths without running long benchmarks
# or the full 2^32-input exhaustive scan.
"$NVCC" "${flags[@]}" repro_sum_cuda/repro_sum.cu repro_sum_cuda/bench_repro_sum.cu -o "$BUILD_DIR/bench_repro_sum"
"$NVCC" "${flags[@]}" repro_math_cuda/exhaustive_math.cu -o "$BUILD_DIR/exhaustive_math"
"$NVCC" "${flags[@]}" repro_torch/src/kernels.cu gpt2_ops_suite/gpt2_ops_suite.cu \
  -lcublas -o "$BUILD_DIR/gpt2_ops_suite"
echo "PASS: GPU regression tests and compilation checks completed"
