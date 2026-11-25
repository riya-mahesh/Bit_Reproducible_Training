#pragma once
#include <cuda_runtime.h>
#include <cstddef>

// One int per 256-element block. The workspace must remain alive until the
// stream completes; callers may reuse it for successive calls on the same stream.
inline size_t repro_scan_workspace_bytes(size_t n) {
    return (n / 256 + (n % 256 != 0)) * sizeof(int);
}

cudaError_t repro_scan_finite(const float* x, int* flag, size_t n, int op_id,
                             int* partial, cudaStream_t stream = nullptr);
