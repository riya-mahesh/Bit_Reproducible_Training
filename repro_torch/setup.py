#!/usr/bin/env python3
# Builds repro_torch/_backend in place:
#   python3 setup.py build_ext --inplace           GPU build if nvcc is on PATH, else CPU only
#   REPRO_CPU_ONLY=1 python3 setup.py build_ext --inplace
#   REPRO_ARCH=sm_75 python3 setup.py build_ext --inplace     instead of this machine's GPU
import os
import shutil

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

here = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(here)
includes = [os.path.join(root, d) for d in ("repro_gemm_cuda", "repro_sum_cuda", "repro_math_cuda", "repro_conv_cuda", "repro_nn_cuda")]
includes.append(os.path.join(here, "src"))
cxx = ["-O2", "-std=c++17", "-fno-fast-math", "-ffp-contract=off"]
link = ["-Wl,-rpath," + os.path.join(os.path.dirname(torch.__file__), "lib")]

if os.environ.get("REPRO_CPU_ONLY") or shutil.which("nvcc") is None:
    ext = CppExtension("repro_torch._backend", ["src/bindings.cpp"], include_dirs=includes,
                       extra_compile_args=cxx + ["-DREPRO_CPU_ONLY"], extra_link_args=link)
else:
    nvcc = ["-O3", "-std=c++17", "-arch=" + os.environ.get("REPRO_ARCH", "native"), "--fmad=false", "-lineinfo"]
    ext = CUDAExtension("repro_torch._backend", ["src/bindings_gpu.cu", "src/kernels.cu"], include_dirs=includes,
                        extra_compile_args={"cxx": cxx, "nvcc": nvcc}, extra_link_args=link)

setup(name="repro_torch", packages=["repro_torch"], ext_modules=[ext], cmdclass={"build_ext": BuildExtension})
