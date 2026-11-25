#include "repro_conv.cuh"
#include "../repro_sum_cuda/repro_reduce.cuh"
#include <cuda_runtime.h>
#include <cstdint>

namespace repro_conv {

namespace {
constexpr int THREADS = 256;
// Images per weight-gradient block. Any value gives the same bits; smaller means more blocks.
constexpr int IMAGES_PER_CHUNK = 8;
constexpr size_t ALIGN = 256;
size_t align_up(size_t x) { return (x + ALIGN - 1) / ALIGN * ALIGN; }
int image_chunks(int N) { return (N + IMAGES_PER_CHUNK - 1) / IMAGES_PER_CHUNK; }

// Finite float32 magnitudes compare like their bits, enabling an integer max reduction.
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
    __shared__ unsigned maxima[THREADS / 32];
    __shared__ int e;
    int t = threadIdx.x;
    unsigned local = 0u;
    for (int i = t; i < count; i += THREADS) local = max(local, magnitude_bits(w[i]));
    unsigned m = repro_reduce::block_max(local, maxima);
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
//
// Every block handles one chunk of IMAGES_PER_CHUNK images, so the grid grows with the batch
// instead of staying at one block per filter or per weight position. Maxima and exact INT64
// sums do not depend on how the images are split, so the result is bit-identical to computing
// each scale and sum over the whole batch at once (and to cpu_conv2d_grad_weight).
//
//   1. partial_max_kernel       grid (F + K, chunks): largest |dY| per filter and largest |x| per
//                               position, over the chunk's images
//   2. merge_scales_kernel      one thread per filter/position: scale from its chunk maxima
//   3. quantize_dy_per_filter_kernel: dY to integers with its filter's scale
//   4. weight_grad_kernel       grid (K, chunks): exact INT64 sums over the chunk's images
//   5. weight_grad_finish_kernel one thread per weight: adds the chunks' sums in a fixed order,
//                               rounds once
//
// Each partial result has exactly one writer, so no atomics are needed anywhere.

// Rows 0..F-1: dY[:, f, :, :] over the chunk's images. Rows F..F+K-1: x at position k.
__global__ void __launch_bounds__(THREADS)
partial_max_kernel(const float* __restrict__ dy, const float* __restrict__ x, unsigned* __restrict__ pmax,
                   ConvShape g, int chunk) {
    __shared__ unsigned maxima[THREADS / 32];
    int t = threadIdx.x, row = blockIdx.x, part = blockIdx.y;
    int n0 = part * chunk, n1 = min(g.N, n0 + chunk);
    int ow = g.OW(), per = g.OH() * ow;
    unsigned local = 0u;
    if (row < g.F) {
        for (int n = n0; n < n1; ++n) {
            const float* p = dy + ((size_t)n * g.F + row) * per;
            for (int i = t; i < per; i += THREADS) local = max(local, magnitude_bits(p[i]));
        }
    } else {
        int c, ky, kx;
        position_of(g, row - g.F, c, ky, kx);
        for (int i = t; i < per; i += THREADS) local = max(local, position_max_bits(x, g, i / ow, i % ow, c, ky, kx, n0, n1));
    }
    unsigned m = repro_reduce::block_max(local, maxima);
    if (t == 0) pmax[(size_t)row * gridDim.y + part] = m;
}

// e[0..F-1] are the filter scales of dY, e[F..F+K-1] the position scales of x.
__global__ void __launch_bounds__(THREADS)
merge_scales_kernel(const unsigned* __restrict__ pmax, int* __restrict__ e, int rows, int chunks) {
    int row = blockIdx.x * THREADS + threadIdx.x;
    if (row >= rows) return;
    unsigned m = 0u;
    for (int part = 0; part < chunks; ++part) m = max(m, pmax[(size_t)row * chunks + part]);
    e[row] = scale_exponent(__uint_as_float(m));
}

// Rounds dY with its filter's scale. One block per (image, filter) saves a division per value.
__global__ void __launch_bounds__(THREADS)
quantize_dy_per_filter_kernel(const float* __restrict__ dy, const int* __restrict__ eF, int* __restrict__ D, ConvShape g, int S) {
    size_t per_filter = (size_t)g.OH() * g.OW();
    size_t base = ((size_t)blockIdx.y * g.F + blockIdx.x) * per_filter;
    int e = eF[blockIdx.x];
    for (size_t i = threadIdx.x; i < per_filter; i += THREADS) D[base + i] = fixed_point(dy[base + i], e, S);
}

// Block per (position k, chunk of images). Each thread owns some window places (oy, ox) and
// handles them for the chunk's images, then the per-filter INT64 sums are merged through one
// shared slot per warp and written as the chunk's partial sums. No location has two writers.
template<int MAX_F>
__global__ void __launch_bounds__(THREADS)
weight_grad_kernel(const float* __restrict__ x, const int* __restrict__ D, const int* __restrict__ eX,
                   int64_t* __restrict__ sums, ConvShape g, int S, int chunk) {
    __shared__ int64_t partial[THREADS / 32][MAX_F];
    int t = threadIdx.x, k = blockIdx.x, part = blockIdx.y, ow = g.OW(), per = g.OH() * ow;
    int lane = t & 31, warp = t >> 5;
    int n0 = part * chunk, n1 = min(g.N, n0 + chunk);
    int c, ky, kx;
    position_of(g, k, c, ky, kx);
    int e = eX[k];
    int64_t s[MAX_F];
    for (int f = 0; f < g.F; ++f) s[f] = 0;
    for (int i = t; i < per; i += THREADS) weight_grad_windows(x, D, g, i / ow, i % ow, c, ky, kx, e, S, s, n0, n1);
    for (int f = 0; f < g.F; ++f) {
        int64_t v = repro_reduce::warp_sum(s[f]);
        if (lane == 0) partial[warp][f] = v;
    }
    __syncthreads();
    if (t < g.F) {
        int64_t v = 0;
        for (int w = 0; w < THREADS / 32; ++w) v += partial[w][t];
        sums[((size_t)k * gridDim.y + part) * g.F + t] = v;
    }
}

// One thread per weight: adds its chunks' exact sums in chunk order and rounds once.
__global__ void __launch_bounds__(THREADS)
weight_grad_finish_kernel(const int64_t* __restrict__ sums, const int* __restrict__ eF, const int* __restrict__ eX,
                          float* __restrict__ dw, ConvShape g, int S, int chunks) {
    int K = g.K(), i = blockIdx.x * THREADS + threadIdx.x;
    if (i >= g.F * K) return;
    int f = i / K, k = i % K;
    int64_t v = 0;
    for (int part = 0; part < chunks; ++part) v += sums[((size_t)k * chunks + part) * g.F + f];
    dw[i] = round_to_float(v, eF[f] + eX[k] - 14 * (S - 1));
}

// ---------------- input gradient ----------------

// Block per image: one scale for its dY, then its integers.
__global__ void __launch_bounds__(THREADS)
quantize_dy_per_image_kernel(const float* __restrict__ dy, int* __restrict__ D, int* __restrict__ eI, ConvShape g, int S) {
    __shared__ unsigned maxima[THREADS / 32];
    __shared__ int e;
    int t = threadIdx.x, n = blockIdx.x;
    size_t per = (size_t)g.F * g.OH() * g.OW();
    const float* p = dy + (size_t)n * per;
    unsigned local = 0u;
    for (size_t i = t; i < per; i += THREADS) local = max(local, magnitude_bits(p[i]));
    unsigned m = repro_reduce::block_max(local, maxima);
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

// Layout: D (dY as integers) | scales e[F + K] | chunk maxima [F + K][chunks] | chunk sums [K][chunks][F].
cudaError_t conv2d_grad_weight_workspace(const ConvShape& g, int S, size_t* bytes) {
    if (!valid(g, S)) return cudaErrorInvalidValue;
    size_t rows = (size_t)g.F + g.K(), chunks = image_chunks(g.N);
    *bytes = align_up(g.output_size() * 4) + align_up(rows * 4) + align_up(rows * chunks * 4)
           + align_up((size_t)g.K() * chunks * g.F * 8);
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
    int K = g.K(), rows = g.F + K, chunks = image_chunks(g.N);
    char* base = static_cast<char*>(workspace);
    int* D = reinterpret_cast<int*>(base);
    base += align_up(g.output_size() * 4);
    int* e = reinterpret_cast<int*>(base);                   // eF = e, eX = e + F
    base += align_up((size_t)rows * 4);
    unsigned* pmax = reinterpret_cast<unsigned*>(base);
    base += align_up((size_t)rows * chunks * 4);
    int64_t* sums = reinterpret_cast<int64_t*>(base);
    partial_max_kernel<<<dim3(rows, chunks), THREADS, 0, stream>>>(dy, x, pmax, g, IMAGES_PER_CHUNK);
    merge_scales_kernel<<<(rows + THREADS - 1) / THREADS, THREADS, 0, stream>>>(pmax, e, rows, chunks);
    quantize_dy_per_filter_kernel<<<dim3(g.F, g.N), THREADS, 0, stream>>>(dy, e, D, g, S);
    dim3 grid(K, chunks);
    if (g.F <= 16) weight_grad_kernel<16><<<grid, THREADS, 0, stream>>>(x, D, e + g.F, sums, g, S, IMAGES_PER_CHUNK);
    else if (g.F <= 64) weight_grad_kernel<64><<<grid, THREADS, 0, stream>>>(x, D, e + g.F, sums, g, S, IMAGES_PER_CHUNK);
    else weight_grad_kernel<256><<<grid, THREADS, 0, stream>>>(x, D, e + g.F, sums, g, S, IMAGES_PER_CHUNK);
    weight_grad_finish_kernel<<<(g.F * K + THREADS - 1) / THREADS, THREADS, 0, stream>>>(sums, e, e + g.F, dw, g, S, chunks);
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
