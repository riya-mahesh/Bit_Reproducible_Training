# Bit-Reproducible Training

GPU building blocks for neural-network training whose results are **bit-identical**: the same inputs give the same output bits on every run, process, kernel configuration, and (by design) GPU model. The goal is verifiable learning: a verifier can re-execute a training step and check the result exactly.

Ordinary GPU kernels are not bit-reproducible because they add floating-point numbers in an order that depends on tiling, parallel splits, library versions, and hardware, and each addition rounds. The components here avoid order-dependent floating-point arithmetic, so the result is defined by the math alone.

**To run the experiments** (reproducibility, accuracy, speed), see [`experiments/README.md`](experiments/README.md).

## Components

| Folder | What it is | Status |
|---|---|---|
| [`repro_gemm_cuda/`](repro_gemm_cuda/) | Bit-reproducible matrix multiplication (GEMM) for FP32, using exact integer arithmetic (an integer-exact Ozaki-style scheme with S = 1–3 int8 slices). Covers the forward and backward GEMMs of linear layers. | S = 1 verified bit-identical on 2× RTX 2080 Ti; S = 1–3 verified against a CPU reference on the host, GPU run pending |
| [`repro_math_cuda/`](repro_math_cuda/) | Correctly rounded float32 exp and log, softmax, and an exhaustive test (all 2³² inputs) of whether CUDA's math functions give bit-identical results across GPUs. | exp and log verified on all 2³² inputs on a CPU; GPU run pending |
| [`repro_conv_cuda/`](repro_conv_cuda/) | Reproducible Conv2d and both gradients, using GEMM's logic directly on the image. | Verified against the GEMM definition on a CPU; GPU run pending |
| [`repro_nn_cuda/`](repro_nn_cuda/) | Reproducible MaxPool2d with a fixed tie rule, and Adam or AdamW with one defined rounding per operation. | Verified exactly and against PyTorch on a CPU; GPU run pending |
| [`repro_torch/`](repro_torch/) | PyTorch layers and optimizer on these kernels, and LeNet-5 MNIST training with RepDL's settings. | Full training run bit-identical across repeated runs on a CPU; GPU build pending |
| [`repro_sum_cuda/`](repro_sum_cuda/) | Order-independent, correctly rounded sum and mean (exact superaccumulator). For bias gradients, loss means, softmax denominators, normalization statistics. | Core verified against exact rational arithmetic on the host; GPU run pending |

Each folder has its own README with the method, guarantees, build steps, and tests.

## Still needed for a fully reproducible training step

- Conv2d (im2col + reproducible GEMM), softmax / cross-entropy, LayerNorm/RMSNorm built on the sum primitive
- Batched/strided GEMM for attention
- Deterministic multi-GPU gradient all-reduce
- Replacements for atomics-based ops (embedding-gradient scatter-add, `index_add`)
- Seeded, recorded randomness (dropout, data order)

## Requirements

CUDA Toolkit 11.6+ (tested with 12.4) and an NVIDIA GPU. See each component's README.
