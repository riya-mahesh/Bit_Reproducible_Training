// All GPU kernels of the project, compiled as one unit for the PyTorch extension.
#include "../../repro_gemm_cuda/repro_gemm.cu"
#include "../../repro_sum_cuda/repro_sum.cu"
#include "../../repro_math_cuda/repro_softmax.cu"
#include "../../repro_conv_cuda/repro_conv.cu"
#include "../../repro_nn_cuda/repro_nn.cu"
#include "repro_scan.cuh"
#include "../../repro_sum_cuda/repro_reduce.cuh"

// Each block owns one partial flag; a second kernel merges them and updates the
// persistent flag. Successive scans must be ordered on the same stream.
__global__ void __launch_bounds__(256)
repro_scan_finite_kernel(const float* __restrict__ x, int* __restrict__ partial, size_t n, int op_id) {
    __shared__ int minima[8];
    size_t i = (size_t)blockIdx.x * 256 + threadIdx.x;
    bool bad = i < n && (__float_as_uint(x[i]) & 0x7F800000u) == 0x7F800000u;
    int v = repro_reduce::block_min(bad ? op_id : 0x7FFFFFFF, minima);
    if (threadIdx.x == 0) partial[blockIdx.x] = v;
}

__global__ void __launch_bounds__(256)
repro_scan_merge_kernel(const int* __restrict__ partial, int* __restrict__ flag, size_t blocks) {
    __shared__ int minima[8];
    int v = 0x7FFFFFFF;
    for (size_t i = threadIdx.x; i < blocks; i += 256) v = min(v, partial[i]);
    v = repro_reduce::block_min(v, minima);
    if (threadIdx.x == 0) *flag = min(*flag, v);
}

cudaError_t repro_scan_finite(const float* x, int* flag, size_t n, int op_id,
                             int* partial, cudaStream_t stream) {
    if (n == 0) return cudaSuccess;
    if (!x || !flag || !partial) return cudaErrorInvalidValue;
    size_t blocks = n / 256 + (n % 256 != 0);
    if (blocks > 0x7FFFFFFFu) return cudaErrorInvalidValue;
    repro_scan_finite_kernel<<<(unsigned)blocks, 256, 0, stream>>>(x, partial, n, op_id);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return e;
    repro_scan_merge_kernel<<<1, 256, 0, stream>>>(partial, flag, blocks);
    return cudaGetLastError();
}
