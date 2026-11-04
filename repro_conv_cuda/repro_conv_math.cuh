#pragma once
// The arithmetic that defines repro_conv's output. It follows repro_gemm's logic: every
// input gets a power-of-two scale, is rounded once to a fixed-point integer, the products
// are summed exactly in INT64, and the exact sum is rounded once to float32.
//
// Layouts are PyTorch's: x[n, c, y, x], weight[f, c, ky, kx], output and its gradient
// [n, f, oy, ox]. A "window" is one (n, oy, ox); position k = (c, ky, kx) numbers the
// K = C * KH * KW values of a window. Pixels outside the image (padding) are zero.
//
// Scales, with e0 = -14 (S - 1):
//   forward     Y[w, f]  = round(sum_k X[w, k] * W[f, k],   e_window(w) + e_filter(f) + e0)
//   weight grad dW[f, k] = round(sum_w D[w, f] * X[w, k],   e_filter(dY, f) + e_position(k) + e0)
//   input grad  dX[pixel] = round(sum D[w, f] * W[f, k],    e_image(dY, n) + e_all_weights + e0)
//
// The forward and weight gradient are exactly repro_gemm applied to the im2col matrix, so
// they match it bit for bit. A pixel of the input gradient sums over several windows and
// all filters, so its terms need one shared scale: per image for dY, over all weights for W.
//
// These functions are __host__ __device__: the GPU kernels and the CPU version use the same code.
#include "repro_gemm_math.cuh"

namespace repro_conv {

using repro_gemm::fixed_point;
using repro_gemm::round_to_float;
using repro_gemm::scale_exponent;

struct ConvShape {
    int N, C, H, W;       // input [N, C, H, W]
    int F, KH, KW;        // weight [F, C, KH, KW]
    int SY = 1, SX = 1;   // stride
    int PY = 0, PX = 0;   // zero padding

    __host__ __device__ int K() const { return C * KH * KW; }
    __host__ __device__ int OH() const { return (H + 2 * PY - KH) / SY + 1; }
    __host__ __device__ int OW() const { return (W + 2 * PX - KW) / SX + 1; }
    __host__ __device__ int windows() const { return N * OH() * OW(); }
    __host__ __device__ size_t input_size() const { return (size_t)N * C * H * W; }
    __host__ __device__ size_t weight_size() const { return (size_t)F * K(); }
    __host__ __device__ size_t output_size() const { return (size_t)N * F * OH() * OW(); }
};

__host__ __device__ inline float abs_value(float x) { return x < 0.0f ? -x : x; }

// Input value at position (c, ky, kx) of window (n, oy, ox); zero in the padding.
__host__ __device__ inline float window_value(const float* x, const ConvShape& g, int n, int oy, int ox,
                                              int c, int ky, int kx) {
    int y = oy * g.SY - g.PY + ky, xx = ox * g.SX - g.PX + kx;
    if (y < 0 || y >= g.H || xx < 0 || xx >= g.W) return 0.0f;
    return x[(((size_t)n * g.C + c) * g.H + y) * g.W + xx];
}

__host__ __device__ inline void position_of(const ConvShape& g, int k, int& c, int& ky, int& kx) {
    int kk = g.KH * g.KW, r = k % kk;
    c = k / kk;
    ky = r / g.KW;
    kx = r % g.KW;
}

__host__ __device__ inline void window_of(const ConvShape& g, int w, int& n, int& oy, int& ox) {
    int oh = g.OH(), ow = g.OW(), per = oh * ow, r = w % per;
    n = w / per;
    oy = r / ow;
    ox = r % ow;
}

// Forward for one window: writes Y[n, f, oy, ox] for every filter. q holds K integers.
// Whole rows outside the image are skipped and in-image rows read through one pointer. The
// magnitude bits of finite floats order like the floats, so the scale needs no float compares.
__host__ __device__ inline void forward_window(const float* x, const int* Wq, const int* eF, float* out,
                                               const ConvShape& g, int n, int oy, int ox, int S, int* q) {
    uint32_t mb = 0;
    int K = g.K(), k = 0, y0 = oy * g.SY - g.PY, x0 = ox * g.SX - g.PX;
    for (int c = 0; c < g.C; ++c) {
        const float* image = x + ((size_t)n * g.C + c) * g.H * g.W;
        for (int ky = 0; ky < g.KH; ++ky) {
            int y = y0 + ky;
            if (y < 0 || y >= g.H) {
                for (int kx = 0; kx < g.KW; ++kx) q[k++] = 0;
                continue;
            }
            const float* row = image + (size_t)y * g.W;
            for (int kx = 0; kx < g.KW; ++kx) {
                int xx = x0 + kx;
                uint32_t b = (xx < 0 || xx >= g.W) ? 0u : repro_gemm::float_to_bits(row[xx]);
                uint32_t a = b & 0x7FFFFFFFu;
                if (a > mb) mb = a;
                q[k++] = (int)b;
            }
        }
    }
    int e = scale_exponent(repro_gemm::bits_to_float(mb));
    for (k = 0; k < K; ++k) q[k] = fixed_point(repro_gemm::bits_to_float((uint32_t)q[k]), e, S);
    int oh = g.OH(), ow = g.OW(), e0 = -14 * (S - 1);
    for (int f = 0; f < g.F; ++f) {
        const int* wf = Wq + (size_t)f * K;
        int64_t t = 0;
        for (k = 0; k < K; ++k) t += (int64_t)q[k] * wf[k];
        out[(((size_t)n * g.F + f) * oh + oy) * ow + ox] = round_to_float(t, e + eF[f] + e0);
    }
}

// Adds the weight-gradient terms at position (c, ky, kx) for the window at (oy, ox) of every
// image in [0, N). D is dY quantized per filter, in dY's layout; t holds one INT64 per filter.
// The window's place in the image is the same for all images, so it is worked out once.
__host__ __device__ inline void weight_grad_windows(const float* x, const int* D, const ConvShape& g, int oy, int ox,
                                                    int c, int ky, int kx, int eX, int S, int64_t* t) {
    int y = oy * g.SY - g.PY + ky, xx = ox * g.SX - g.PX + kx;
    if (y < 0 || y >= g.H || xx < 0 || xx >= g.W) return;    // padding: X = 0
    size_t per_image = (size_t)g.C * g.H * g.W, at = ((size_t)c * g.H + y) * g.W + xx;
    size_t per_filter = (size_t)g.OH() * g.OW(), offset = (size_t)oy * g.OW() + ox;
    for (int n = 0; n < g.N; ++n) {
        int X = fixed_point(x[(size_t)n * per_image + at], eX, S);
        if (X == 0) continue;
        const int* d = D + (size_t)n * g.F * per_filter + offset;
        for (int f = 0; f < g.F; ++f) t[f] += (int64_t)X * d[(size_t)f * per_filter];
    }
}

// Largest input magnitude bits at position (c, ky, kx) for the window at (oy, ox), all images.
__host__ __device__ inline uint32_t position_max_bits(const float* x, const ConvShape& g, int oy, int ox, int c, int ky, int kx) {
    int y = oy * g.SY - g.PY + ky, xx = ox * g.SX - g.PX + kx;
    if (y < 0 || y >= g.H || xx < 0 || xx >= g.W) return 0u;
    size_t per_image = (size_t)g.C * g.H * g.W, at = ((size_t)c * g.H + y) * g.W + xx;
    uint32_t m = 0;
    for (int n = 0; n < g.N; ++n) {
        uint32_t a = repro_gemm::float_to_bits(x[(size_t)n * per_image + at]) & 0x7FFFFFFFu;
        if (a > m) m = a;
    }
    return m;
}

// Input gradient at pixel (n, c, y, x): exact sum over the windows covering it and all filters.
// D is dY quantized per image, Wq the weights quantized with one scale.
__host__ __device__ inline float input_grad_pixel(const int* D, const int* Wq, const ConvShape& g,
                                                  int n, int c, int y, int xx, int E) {
    int oh = g.OH(), ow = g.OW();
    int64_t t = 0;
    for (int ky = 0; ky < g.KH; ++ky) {
        int ry = y + g.PY - ky;
        if (ry < 0) continue;
        int oy = ry;
        if (g.SY != 1) { if (ry % g.SY != 0) continue; oy = ry / g.SY; }
        if (oy >= oh) continue;
        for (int kx = 0; kx < g.KW; ++kx) {
            int rx = xx + g.PX - kx;
            if (rx < 0) continue;
            int ox = rx;
            if (g.SX != 1) { if (rx % g.SX != 0) continue; ox = rx / g.SX; }
            if (ox >= ow) continue;
            for (int f = 0; f < g.F; ++f)
                t += (int64_t)D[(((size_t)n * g.F + f) * oh + oy) * ow + ox] *
                     Wq[(((size_t)f * g.C + c) * g.KH + ky) * g.KW + kx];
        }
    }
    return round_to_float(t, E);
}

// Largest number of terms in one exact sum with S slices, keeping |sum| < 2^63.
__host__ __device__ inline double max_terms(int S) {
    double lim = 127.0 * (double)(1 << (7 * (S - 1)));
    return 9.0e18 / (lim * lim);
}

} // namespace repro_conv
