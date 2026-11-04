#include "repro_conv.cuh"
#include <cuda_runtime.h>
#include <cstdint>

namespace repro_conv {

namespace {
constexpr int THREADS = 256;
constexpr size_t ALIGN = 256;
size_t align_up(size_t x) { return (x + ALIGN - 1) / ALIGN * ALIGN; }

// float32 magnitudes compare like their bits, so atomicMax on bits gives the largest |x|.
__device__ __forceinline__ unsigned magnitude_bits(float x) { return __float_as_uint(x) & 0x7FFFFFFFu; }
}

// ---------------- weights ----------------

// One thread per filter: its scale, then its integers.
__global__ void __launch_bounds__(THREADS)
quantize_filters_kernel(const float* __restrict__ w, int* __restrict__ Wq, int* __restrict__ eF, int F, int K, int S) {
    int f = blockIdx.x * THREADS + threadIdx.x;
    if (f >= F) return;
    const float* p = w + (size_t)f * K;
    float m = 0.0f;
    for (int k = 0; k < K; ++k) { float a = abs_value(p[k]); if (a > m) m = a; }
    int e = scale_exponent(m);
    eF[f] = e;
    for (int k = 0; k < K; ++k) Wq[(size_t)f * K + k] = fixed_point(p[k], e, S);
}

// All weights with one scale, for the input gradient.
__global__ void __launch_bounds__(THREADS)
quantize_all_weights_kernel(const float* __restrict__ w, int* __restrict__ Wq, int* __restrict__ eW, int count, int S) {
    __shared__ unsigned m;
    __shared__ int e;
    int t = threadIdx.x;
    if (t == 0) m = 0u;
    __syncthreads();
    unsigned local = 0u;
    for (int i = t; i < count; i += THREADS) local = max(local, magnitude_bits(w[i]));
    atomicMax(&m, local);
    __syncthreads();
    if (t == 0) { e = scale_exponent(__uint_as_float(m)); *eW = e; }
    __syncthreads();
    for (int i = t; i < count; i += THREADS) Wq[i] = fixed_point(w[i], e, S);
}

// ---------------- forward ----------------

// One thread per window, reading the image directly. MAX_K sets the private storage: a
// kernel's stack is reserved for every possible thread, so it stays as small as the shape allows.
template<int MAX_K>
__global__ void __launch_bounds__(THREADS)
forward_kernel(const float* __restrict__ x, const int* __restrict__ Wq, const int* __restrict__ eF,
               float* __restrict__ out, ConvShape g, int S) {
    int i = blockIdx.x * THREADS + threadIdx.x;
    if (i >= g.windows()) return;
    int q[MAX_K];
    int n, oy, ox;
    window_of(g, i, n, oy, ox);
    forward_window(x, Wq, eF, out, g, n, oy, ox, S, q);
}

// ---------------- weight gradient ----------------

// Block per filter: the scale of dY[:, f, :, :], read as whole images in memory order.
__global__ void __launch_bounds__(THREADS)
filter_scales_kernel(const float* __restrict__ dy, int* __restrict__ eF, ConvShape g) {
    __shared__ unsigned m;
    int t = threadIdx.x, f = blockIdx.x;
    size_t per_filter = (size_t)g.OH() * g.OW();
    if (t == 0) m = 0u;
    __syncthreads();
    unsigned local = 0u;
    for (int n = 0; n < g.N; ++n) {
        const float* p = dy + ((size_t)n * g.F + f) * per_filter;
        for (size_t i = t; i < per_filter; i += THREADS) local = max(local, magnitude_bits(p[i]));
    }
    atomicMax(&m, local);
    __syncthreads();
    if (t == 0) eF[f] = scale_exponent(__uint_as_float(m));
}

// Rounds dY with its filter's scale. One block per (image, filter) saves a division per value.
__global__ void __launch_bounds__(THREADS)
quantize_dy_per_filter_kernel(const float* __restrict__ dy, const int* __restrict__ eF, int* __restrict__ D, ConvShape g, int S) {
    size_t per_filter = (size_t)g.OH() * g.OW();
    size_t base = ((size_t)blockIdx.y * g.F + blockIdx.x) * per_filter;
    int e = eF[blockIdx.x];
    for (size_t i = threadIdx.x; i < per_filter; i += THREADS) D[base + i] = fixed_point(dy[base + i], e, S);
}

// Block per position k. Each thread owns some window places (oy, ox) and handles them for all
// images: one local maximum and one atomic per thread for the scale, then the exact sums for
// every filter, merged in shared memory and rounded once.
template<int MAX_F>
__global__ void __launch_bounds__(THREADS)
weight_grad_kernel(const float* __restrict__ x, const int* __restrict__ D, const int* __restrict__ eF,
                   float* __restrict__ dw, ConvShape g, int S) {
    __shared__ unsigned m;
    __shared__ int64_t acc[MAX_F];
    int t = threadIdx.x, k = blockIdx.x, ow = g.OW(), per = g.OH() * ow;
    int c, ky, kx;
    position_of(g, k, c, ky, kx);
    if (t == 0) m = 0u;
    if (t < MAX_F) acc[t] = 0;
    __syncthreads();
    unsigned local = 0u;
    for (int i = t; i < per; i += THREADS) local = max(local, position_max_bits(x, g, i / ow, i % ow, c, ky, kx));
    atomicMax(&m, local);
    __syncthreads();
    int eX = scale_exponent(__uint_as_float(m));
    int64_t s[MAX_F];
    for (int f = 0; f < g.F; ++f) s[f] = 0;
    for (int i = t; i < per; i += THREADS) weight_grad_windows(x, D, g, i / ow, i % ow, c, ky, kx, eX, S, s);
    for (int f = 0; f < g.F; ++f)
        atomicAdd(reinterpret_cast<unsigned long long*>(&acc[f]), (unsigned long long)s[f]);
    __syncthreads();
    if (t < g.F) dw[(size_t)t * g.K() + k] = round_to_float(acc[t], eF[t] + eX - 14 * (S - 1));
}

// ---------------- input gradient ----------------

// Block per image: one scale for its dY, then its integers.
__global__ void __launch_bounds__(THREADS)
quantize_dy_per_image_kernel(const float* __restrict__ dy, int* __restrict__ D, int* __restrict__ eI, ConvShape g, int S) {
    __shared__ unsigned m;
    __shared__ int e;
    int t = threadIdx.x, n = blockIdx.x;
    size_t per = (size_t)g.F * g.OH() * g.OW();
    const float* p = dy + (size_t)n * per;
    if (t == 0) m = 0u;
    __syncthreads();
    unsigned local = 0u;
    for (size_t i = t; i < per; i += THREADS) local = max(local, magnitude_bits(p[i]));
    atomicMax(&m, local);
    __syncthreads();
    if (t == 0) { e = scale_exponent(__uint_as_float(m)); eI[n] = e; }
    __syncthreads();
    for (size_t i = t; i < per; i += THREADS) D[(size_t)n * per + i] = fixed_point(p[i], e, S);
}

__global__ void __launch_bounds__(THREADS)
input_grad_kernel(const int* __restrict__ D, const int* __restrict__ Wq, const int* __restrict__ eI, const int* __restrict__ eW,
                  float* __restrict__ dx, ConvShape g, int S) {
    int i = blockIdx.x * THREADS + threadIdx.x, per = g.C * g.H * g.W;
    if (i >= g.N * per) return;
    int n = i / per, r = i % per, hw = g.H * g.W;
    int c = r / hw, s = r % hw;
    dx[i] = input_grad_pixel(D, Wq, g, n, c, s / g.W, s % g.W, eI[n] + *eW - 14 * (S - 1));
}

// ---------------- plans and launches ----------------

namespace {

bool valid(const ConvShape& g, int S) {
    if (S < 1 || S > repro_gemm::MAX_SLICES) return false;
    if (g.N <= 0 || g.C <= 0 || g.H <= 0 || g.W <= 0 || g.F <= 0 || g.KH <= 0 || g.KW <= 0) return false;
    if (g.SY <= 0 || g.SX <= 0 || g.PY < 0 || g.PX < 0) return false;
    if (g.H + 2 * g.PY < g.KH || g.W + 2 * g.PX < g.KW) return false;
    if ((int64_t)g.C * g.KH * g.KW > 1024 || g.F > 256) return false;
    if (g.N > 65535) return false;                   // block grid limit in the y dimension
    if ((double)g.output_size() >= 2147483648.0 || (double)g.input_size() >= 2147483648.0) return false;
    return (double)g.windows() <= max_terms(S) && (double)g.KH * g.KW * g.F <= max_terms(S)
        && (double)g.K() <= max_terms(S);
}

} // namespace

cudaError_t conv2d_workspace(const ConvShape& g, int S, size_t* bytes) {
    if (!valid(g, S)) return cudaErrorInvalidValue;
    *bytes = align_up(g.weight_size() * 4) + align_up((size_t)g.F * 4);
    return cudaSuccess;
}

cudaError_t conv2d_grad_weight_workspace(const ConvShape& g, int S, size_t* bytes) {
    if (!valid(g, S)) return cudaErrorInvalidValue;
    *bytes = align_up(g.output_size() * 4) + align_up((size_t)g.F * 4);
    return cudaSuccess;
}

cudaError_t conv2d_grad_input_workspace(const ConvShape& g, int S, size_t* bytes) {
    if (!valid(g, S)) return cudaErrorInvalidValue;
    *bytes = align_up(g.output_size() * 4) + align_up(g.weight_size() * 4) + align_up((size_t)g.N * 4) + ALIGN;
    return cudaSuccess;
}

cudaError_t repro_conv2d_ws(const float* x, const float* weight, float* out, const ConvShape& g, int S,
                            void* workspace, size_t workspace_bytes, cudaStream_t stream) {
    size_t need;
    cudaError_t e = conv2d_workspace(g, S, &need);
    if (e != cudaSuccess) return e;
    if (workspace_bytes < need) return cudaErrorInvalidValue;
    int* Wq = static_cast<int*>(workspace);
    int* eF = reinterpret_cast<int*>(static_cast<char*>(workspace) + align_up(g.weight_size() * 4));
    quantize_filters_kernel<<<(g.F + THREADS - 1) / THREADS, THREADS, 0, stream>>>(weight, Wq, eF, g.F, g.K(), S);
    unsigned blocks = (unsigned)((g.windows() + THREADS - 1) / THREADS);
    int K = g.K();
    if (K <= 32) forward_kernel<32><<<blocks, THREADS, 0, stream>>>(x, Wq, eF, out, g, S);
    else if (K <= 256) forward_kernel<256><<<blocks, THREADS, 0, stream>>>(x, Wq, eF, out, g, S);
    else forward_kernel<1024><<<blocks, THREADS, 0, stream>>>(x, Wq, eF, out, g, S);
    return cudaGetLastError();
}

cudaError_t repro_conv2d_grad_weight_ws(const float* dy, const float* x, float* dw, const ConvShape& g, int S,
                                        void* workspace, size_t workspace_bytes, cudaStream_t stream) {
    size_t need;
    cudaError_t e = conv2d_grad_weight_workspace(g, S, &need);
    if (e != cudaSuccess) return e;
    if (workspace_bytes < need) return cudaErrorInvalidValue;
    int* D = static_cast<int*>(workspace);
    int* eF = reinterpret_cast<int*>(static_cast<char*>(workspace) + align_up(g.output_size() * 4));
    filter_scales_kernel<<<g.F, THREADS, 0, stream>>>(dy, eF, g);
    quantize_dy_per_filter_kernel<<<dim3(g.F, g.N), THREADS, 0, stream>>>(dy, eF, D, g, S);
    int K = g.K();
    if (g.F <= 16) weight_grad_kernel<16><<<K, THREADS, 0, stream>>>(x, D, eF, dw, g, S);
    else if (g.F <= 64) weight_grad_kernel<64><<<K, THREADS, 0, stream>>>(x, D, eF, dw, g, S);
    else weight_grad_kernel<256><<<K, THREADS, 0, stream>>>(x, D, eF, dw, g, S);
    return cudaGetLastError();
}

cudaError_t repro_conv2d_grad_input_ws(const float* dy, const float* weight, float* dx, const ConvShape& g, int S,
                                       void* workspace, size_t workspace_bytes, cudaStream_t stream) {
    size_t need;
    cudaError_t e = conv2d_grad_input_workspace(g, S, &need);
    if (e != cudaSuccess) return e;
    if (workspace_bytes < need) return cudaErrorInvalidValue;
    char* base = static_cast<char*>(workspace);
    int* D = reinterpret_cast<int*>(base);
    int* Wq = reinterpret_cast<int*>(base + align_up(g.output_size() * 4));
    int* eI = reinterpret_cast<int*>(base + align_up(g.output_size() * 4) + align_up(g.weight_size() * 4));
    int* eW = reinterpret_cast<int*>(base + align_up(g.output_size() * 4) + align_up(g.weight_size() * 4) + align_up((size_t)g.N * 4));
    quantize_dy_per_image_kernel<<<g.N, THREADS, 0, stream>>>(dy, D, eI, g, S);
    quantize_all_weights_kernel<<<1, THREADS, 0, stream>>>(weight, Wq, eW, (int)g.weight_size(), S);
    unsigned blocks = (unsigned)(((size_t)g.N * g.C * g.H * g.W + THREADS - 1) / THREADS);
    input_grad_kernel<<<blocks, THREADS, 0, stream>>>(D, Wq, eI, eW, dx, g, S);
    return cudaGetLastError();
}

} // namespace repro_conv
