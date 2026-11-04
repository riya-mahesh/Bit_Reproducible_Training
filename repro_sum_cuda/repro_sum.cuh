#pragma once
#include <cuda_runtime.h>
#include <cstddef>

namespace repro_sum {

// Order-independent, correctly rounded reduction of float32 data.
//
// The input x is viewed as a contiguous row-major tensor [A, B, C]. For every b,
//   out[b] = round_to_float32( exact sum over a, c of x[a, b, c] )        (mean = false)
//   out[b] = round_to_float32( exact sum / (A * C) )                      (mean = true)
// with one round-to-nearest-even at the end. The exact sum is formed in a wide
// integer accumulator (see repro_sum_math.cuh), so the result does not depend on
// the order of the data, the number of chunks, the launch configuration, or the GPU.
//
// Common uses:
//   Linear bias gradient, sum over rows of [M, N]:      A = M, B = N, C = 1
//   Conv bias gradient, sum over N, H, W of NCHW:       A = N, B = C, C = H * W
//   Loss mean over a vector of n values:                A = 1, B = 1, C = n, mean = true
//   Per-row sums (softmax denominators, LayerNorm):     A = 1, B = rows, C = cols
//
// Exactly zero results are +0.0. NaN/Inf inputs give NaN for that output.
// Requires A * C <= 2^31 - 1.

struct SumConfig {
    int chunks = 0;     // work split along A*C per output; 0 = choose automatically
    bool mean = false;
};

struct SumPlan {
    int chunks;
    size_t workspace_bytes;
};

cudaError_t make_sum_plan(int A, int B, int C, SumConfig cfg, SumPlan* plan);

// Asynchronous, allocation-free. workspace must hold plan.workspace_bytes.
cudaError_t repro_sum_fp32_ws(const float* x, float* out, int A, int B, int C, SumConfig cfg,
                              void* workspace, size_t workspace_bytes, cudaStream_t stream = nullptr);

// Convenience version: allocates the workspace, runs, synchronizes, frees.
cudaError_t repro_sum_fp32(const float* x, float* out, int A, int B, int C,
                           SumConfig cfg = SumConfig(), cudaStream_t stream = nullptr);

} // namespace repro_sum
