# Bit-Reproducible Training

GPU building blocks for neural-network training whose results are **bit-identical**: the same inputs give the same output bits on every run, process, kernel configuration, and GPU model tested. The goal is verifiable learning: a verifier can re-execute a training step and check the result exactly.

Ordinary GPU kernels are not bit-reproducible because they add floating-point numbers in an order that depends on tiling, parallel splits, library versions, and hardware, and each addition rounds. The components here avoid order-dependent floating-point arithmetic, so the result is defined by the math alone.

**MNIST results:** on LeNet-5, final weights are bit-identical across RTX 2080 Ti, TITAN V and RTX 4070 on CUDA 12.4, 12.6 and 13.0, and on an Apple ARM CPU, with accuracy matching RepDL and stock PyTorch. Ours is 4.2–4.6× faster than RepDL on two of these GPUs and 1.3× on the RTX 4070. See [`experiments/MNIST_RESULTS.md`](experiments/MNIST_RESULTS.md) for the full results and limits.

**To run the experiments** (reproducibility, accuracy, speed), see [`experiments/README.md`](experiments/README.md).

## Components

| Folder | What it is | Status |
|---|---|---|
| [`repro_gemm_cuda/`](repro_gemm_cuda/) | Bit-reproducible matrix multiplication (GEMM) for FP32, using exact integer arithmetic (an integer-exact Ozaki-style scheme with S = 1–3 int8 slices). Covers the forward and backward GEMMs of linear layers. | S = 3 suite operations match the CPU on RTX 2080 Ti, TITAN V and RTX 4070 |
| [`repro_math_cuda/`](repro_math_cuda/) | Correctly rounded float32 exp and log, softmax, and an exhaustive test (all 2³² inputs) of whether CUDA's math functions give bit-identical results across GPUs. | exp and log identical over all 2³² inputs on RTX 2080 Ti, TITAN V, RTX 4070 and CPUs |
| [`repro_conv_cuda/`](repro_conv_cuda/) | Reproducible Conv2d and both gradients, using GEMM's logic directly on the image. | GPU tests pass on RTX 2080 Ti, TITAN V and RTX 4070 |
| [`repro_nn_cuda/`](repro_nn_cuda/) | Reproducible MaxPool2d with a fixed tie rule, and Adam or AdamW with one defined rounding per operation. | GPU tests pass on RTX 2080 Ti, TITAN V and RTX 4070 |
| [`repro_torch/`](repro_torch/) | PyTorch layers and optimizer on these kernels, and LeNet-5 MNIST training with RepDL's settings. | MNIST training bit-identical across 3 GPU models, CUDA 12.4–13.0 and a CPU |
| [`repro_sum_cuda/`](repro_sum_cuda/) | Order-independent, correctly rounded sum and mean (exact superaccumulator). For bias gradients, loss means, softmax denominators, normalization statistics. | GPU tests pass on RTX 2080 Ti, TITAN V and RTX 4070 |

Each folder has its own README with the method, guarantees, build steps, and tests.

## Still needed for larger models

MNIST training is complete. For GPT-2 and larger models:

- LayerNorm/RMSNorm built on the sum primitive, GELU, and embeddings
- Batched/strided GEMM for attention
- Deterministic multi-GPU gradient all-reduce
- Replacements for atomics-based ops (embedding-gradient scatter-add, `index_add`)
- Seeded, recorded randomness (dropout, data order)

## Requirements

CUDA Toolkit 11.6+ (tested with 12.4, 12.6 and 13.0) and an NVIDIA GPU. See each component's README.
