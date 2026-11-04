# GPT-2 operation conformance suite

Every reproducible operation a GPT-2 (small) training step uses, run at GPT-2 shapes on the **CPU** (reference implementation of the same definition) and on each **GPU** (our kernels). For each operation it records an input hash, an output hash, and timing; `compare_suite.sh` checks that the output hashes are identical across every CPU and GPU.

- **Inputs** are generated with SplitMix64 and exact arithmetic only (no math library), so every machine produces identical input bits; the input hash confirms it.
- **CPU build** uses a plain C++ compiler with no CUDA toolkit (`-DSUITE_CPU_ONLY`) — the situation of a CPU-only verifier.
- **GPU timing** is the median of CUDA-event timings after warm-up; `base_ms` is the usual non-reproducible way (cuBLAS FP32 with TF32 off for GEMMs, a plain float reduction for sums), and `x base` is the cost of reproducibility.

## Operations (GPT-2 small: d = 768, 4d = 3072, vocab 50257, 12 heads × 64, T = 1024)

| Group | Operations |
|---|---|
| Linear layers (`attn_qkv`, `attn_proj`, `mlp_fc`, `mlp_proj`, `lm_head`) | forward `X·Wᵀ`, input grad `dY·W`, weight grad `dYᵀ·X` (S = 3 GEMM) |
| One attention head | `Q·Kᵀ`, `P·V`, `dO·Vᵀ`, `Pᵀ·dO`, `dS·K`, `dSᵀ·Q` |
| Reductions | bias gradients (4 widths), LayerNorm γ/β gradients, LayerNorm row mean, causal attention-softmax row sums, cross-entropy row sums over the vocabulary, loss mean |
| Softmax | MNIST logits, causal attention rows, vocabulary rows |
| Cross-entropy | loss, and loss plus gradient, at MNIST and vocabulary shapes |
| LeNet-5 pooling and Adam | MaxPool forward and backward, one Adam step over 61,706 parameters |
| LeNet-5 convolutions | forward, weight gradient, and input gradient at MNIST shapes |
| Pending (added as they are built) | exact dot product / sum of squares, LayerNorm, softmax backward, a fused softmax kernel, GELU, embedding forward/backward, batched attention GEMM, AdamW + gradient clipping |

## Run

```bash
module load cuda-12.4          # on CIMS
./run_suite.sh                 # CPU + every GPU on this machine; then compares all results/*/*.txt
GPUS="0" ./run_suite.sh        # one GPU
TOKENS=8192 GPUS="0" ./run_suite.sh   # larger batch (CPU run takes longer)
./compare_suite.sh results/*/*.txt    # compare any set of result files, e.g. after copying in another machine's
```

`results/mac/cpu.txt` holds the reference hashes from an Apple M4 Pro CPU (Apple clang, no CUDA).
