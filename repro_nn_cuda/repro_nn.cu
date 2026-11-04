#include "repro_nn.cuh"
#include <cuda_runtime.h>
#include <cstdint>

namespace repro_nn {

namespace {
constexpr int THREADS = 256;

bool valid(const PoolShape& g) {
    if (g.N <= 0 || g.C <= 0 || g.H <= 0 || g.W <= 0 || g.KH <= 0 || g.KW <= 0 || g.SY <= 0 || g.SX <= 0) return false;
    if (g.KH > g.H || g.KW > g.W) return false;
    return (int64_t)g.N * g.C <= 65535 && (int64_t)g.H * g.W < INT32_MAX;
}
}

// One block per image and channel, so a thread finds its output with no 64-bit division.
__global__ void __launch_bounds__(THREADS)
max_pool_kernel(const float* __restrict__ x, float* __restrict__ out, int* __restrict__ index, PoolShape g) {
    int ow = g.OW(), count = g.OH() * ow;
    size_t pl = blockIdx.x;
    const float* plane = x + pl * g.H * g.W;
    float* o = out + pl * count;
    int* id = index + pl * count;
    for (int i = threadIdx.x; i < count; i += THREADS) pool_window(plane, g, i / ow, i % ow, o[i], id[i]);
}

__global__ void __launch_bounds__(THREADS)
max_pool_backward_kernel(const float* __restrict__ dy, const int* __restrict__ index, float* __restrict__ dx, PoolShape g) {
    int count = g.OH() * g.OW(), pixels = g.H * g.W;
    size_t pl = blockIdx.x;
    const float* d = dy + pl * count;
    const int* id = index + pl * count;
    float* o = dx + pl * pixels;
    for (int i = threadIdx.x; i < pixels; i += THREADS) o[i] = pool_backward_pixel(d, id, g, i / g.W, i % g.W);
}

__global__ void __launch_bounds__(THREADS)
adam_kernel(float* __restrict__ p, const float* __restrict__ g, float* __restrict__ m, float* __restrict__ v, size_t n, AdamScalars s) {
    size_t i = (size_t)blockIdx.x * THREADS + threadIdx.x;
    if (i < n) adam_update(p[i], g[i], m[i], v[i], s);
}

cudaError_t repro_max_pool2d(const float* x, float* out, int* index, const PoolShape& g, cudaStream_t stream) {
    if (!valid(g)) return cudaErrorInvalidValue;
    max_pool_kernel<<<g.N * g.C, THREADS, 0, stream>>>(x, out, index, g);
    return cudaGetLastError();
}

cudaError_t repro_max_pool2d_backward(const float* dy, const int* index, float* dx, const PoolShape& g, cudaStream_t stream) {
    if (!valid(g)) return cudaErrorInvalidValue;
    max_pool_backward_kernel<<<g.N * g.C, THREADS, 0, stream>>>(dy, index, dx, g);
    return cudaGetLastError();
}

cudaError_t repro_adam(float* p, const float* g, float* m, float* v, size_t n, const AdamScalars& s, cudaStream_t stream) {
    if (n == 0) return cudaSuccess;
    adam_kernel<<<(unsigned)((n + THREADS - 1) / THREADS), THREADS, 0, stream>>>(p, g, m, v, n, s);
    return cudaGetLastError();
}

} // namespace repro_nn
