#pragma once
#include <cuda_runtime.h>
#include <cstddef>

namespace repro_math {

// Reproducible elementwise exp and log over a float32 array (see repro_math.cuh).
cudaError_t repro_exp_fp32(const float* x, float* out, size_t n, cudaStream_t stream = nullptr);
cudaError_t repro_log_fp32(const float* x, float* out, size_t n, cudaStream_t stream = nullptr);

// Reproducible softmax over the last dimension of a row-major [rows, cols] tensor (see
// repro_softmax_math.cuh). Requires rows * cols < 2^32. out may equal x. Bit-identical
// across GPUs, CUDA versions, and the CPU function repro_math::softmax_row.
//
// Needs repro_sum.cu from ../repro_sum_cuda at link time.
struct SoftmaxPlan {
    size_t workspace_bytes;
};

cudaError_t make_softmax_plan(int rows, int cols, SoftmaxPlan* plan);

// Asynchronous, allocation-free: use this in training loops. workspace must hold
// plan.workspace_bytes (cudaMalloc alignment).
cudaError_t repro_softmax_fp32_ws(const float* x, float* out, int rows, int cols,
                                  void* workspace, size_t workspace_bytes, cudaStream_t stream = nullptr);

// Convenience version: allocates the workspace, runs, synchronizes, frees.
cudaError_t repro_softmax_fp32(const float* x, float* out, int rows, int cols, cudaStream_t stream = nullptr);

// Reproducible cross-entropy, mean over rows that are not ignored (definition in
// repro_softmax_math.cuh). x[rows, cols], target[rows] on the device, loss a device float.
//
// The forward call leaves exp values and row sums in its workspace, and the gradient call
// reuses them instead of recomputing softmax, so pass it the same workspace, unmodified,
// with the same target and ignore_index. seed is the gradient flowing into the loss:
// 1.0 for loss.backward().
struct CrossEntropyPlan {
    size_t workspace_bytes;
};

cudaError_t make_cross_entropy_plan(int rows, int cols, CrossEntropyPlan* plan);

cudaError_t repro_cross_entropy_fp32_ws(const float* x, const int* target, int ignore_index, float* loss,
                                        int rows, int cols, void* workspace, size_t workspace_bytes,
                                        cudaStream_t stream = nullptr);

cudaError_t repro_cross_entropy_grad_fp32_ws(const int* target, int ignore_index, float seed, float* grad,
                                             int rows, int cols, void* workspace, size_t workspace_bytes,
                                             cudaStream_t stream = nullptr);

} // namespace repro_math
