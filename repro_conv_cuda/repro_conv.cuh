#pragma once
#include "repro_conv_math.cuh"
#include <cuda_runtime.h>
#include <cstddef>

namespace repro_conv {

// Reproducible 2D convolution and its two gradients on FP32, in PyTorch layouts, computed
// directly on the image with no im2col buffer (definition in repro_conv_math.cuh).
//
//   repro_conv2d:             out[N, F, OH, OW] from x[N, C, H, W], weight[F, C, KH, KW]
//   repro_conv2d_grad_weight: dW[F, C, KH, KW]  from dY[N, F, OH, OW], x
//   repro_conv2d_grad_input:  dX[N, C, H, W]    from dY[N, F, OH, OW], weight
//
// Bias is not included: add it with a plain addition, and get its gradient from repro_sum.
// slices S matches repro_gemm: 1..3, default 3. Inputs must be finite. Output is bit-identical
// across runs and GPUs, and to cpu_conv2d* in repro_conv_cpu.h. Limits: K <= 1024, F <= 256.
// Do not build with --use_fast_math / -ftz=true / -prec-div=false.

// Workspace sizes (the same plan for the three functions would be wasteful: W, dY, or both
// are quantized depending on the call). cudaErrorInvalidValue for unsupported shapes.
cudaError_t conv2d_workspace(const ConvShape& g, int S, size_t* bytes);
cudaError_t conv2d_grad_weight_workspace(const ConvShape& g, int S, size_t* bytes);
cudaError_t conv2d_grad_input_workspace(const ConvShape& g, int S, size_t* bytes);

// Asynchronous and allocation-free: use these in training loops.
cudaError_t repro_conv2d_ws(const float* x, const float* weight, float* out, const ConvShape& g, int S,
                            void* workspace, size_t workspace_bytes, cudaStream_t stream = nullptr);
cudaError_t repro_conv2d_grad_weight_ws(const float* dy, const float* x, float* dw, const ConvShape& g, int S,
                                        void* workspace, size_t workspace_bytes, cudaStream_t stream = nullptr);
cudaError_t repro_conv2d_grad_input_ws(const float* dy, const float* weight, float* dx, const ConvShape& g, int S,
                                       void* workspace, size_t workspace_bytes, cudaStream_t stream = nullptr);

} // namespace repro_conv
