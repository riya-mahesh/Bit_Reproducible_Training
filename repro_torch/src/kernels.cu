// All GPU kernels of the project, compiled as one unit for the PyTorch extension.
#include "../../repro_gemm_cuda/repro_gemm.cu"
#include "../../repro_sum_cuda/repro_sum.cu"
#include "../../repro_math_cuda/repro_softmax.cu"
#include "../../repro_conv_cuda/repro_conv.cu"
#include "../../repro_nn_cuda/repro_nn.cu"

// Sets *flag to the smallest op id that saw a NaN or infinity. Runs alongside the operation, so
// checking costs no synchronization; the training loop reads the flag once per step.
__global__ void __launch_bounds__(256)
repro_scan_finite_kernel(const float* __restrict__ x, int* __restrict__ flag, size_t n, int op_id) {
    size_t i = (size_t)blockIdx.x * 256 + threadIdx.x;
    if (i < n && (__float_as_uint(x[i]) & 0x7F800000u) == 0x7F800000u) atomicMin(flag, op_id);
}

cudaError_t repro_scan_finite(const float* x, int* flag, size_t n, int op_id) {
    if (n == 0) return cudaSuccess;
    repro_scan_finite_kernel<<<(unsigned)((n + 255) / 256), 256>>>(x, flag, n, op_id);
    return cudaGetLastError();
}
