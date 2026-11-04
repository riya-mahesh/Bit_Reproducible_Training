#pragma once
#include "repro_nn_math.cuh"
#include <cuda_runtime.h>
#include <cstddef>

namespace repro_nn {

// Reproducible MaxPool2d and Adam (definitions in repro_nn_math.cuh). All are asynchronous and
// need no workspace. Bit-identical across runs and GPUs, and to repro_nn_cpu.h.

// out[N, C, OH, OW] and index[N, C, OH, OW] from x. Requires N * C <= 65535 and a window that fits.
cudaError_t repro_max_pool2d(const float* x, float* out, int* index, const PoolShape& g, cudaStream_t stream = nullptr);

// dx[N, C, H, W] from dy[N, C, OH, OW] and the index from the forward call.
cudaError_t repro_max_pool2d_backward(const float* dy, const int* index, float* dx, const PoolShape& g,
                                      cudaStream_t stream = nullptr);

// One Adam update of a parameter array of n values, in place on p, m, v. m and v start as zeros.
// Call adam_next_step once per optimizer step, then this once per parameter with its scalars.
cudaError_t repro_adam(float* p, const float* g, float* m, float* v, size_t n, const AdamScalars& s,
                       cudaStream_t stream = nullptr);

} // namespace repro_nn
