#pragma once
// Definitions of reproducible MaxPool2d and Adam. These functions are __host__ __device__, so
// the GPU kernels and the CPU version use the same code.
//
// MaxPool2d on x[N, C, H, W], kernel (KH, KW), stride (SY, SX), no padding:
//   Each output is the first largest entry of its window, scanning rows then columns: a later
//   entry replaces it only if strictly larger, so a tie (including +0 and -0) keeps the first.
//   index[n, c, oy, ox] = y * W + x of that entry within its image and channel.
//   Backward: dX at a pixel is the exact sum, rounded once, of dY over the outputs whose index is
//   that pixel; +0 if there are none. With stride >= kernel there is at most one.
//
// Adam on one parameter array, one rounding per operation, in this fixed order:
//   g'  = g + wd * p                                   (only when weight_decay != 0 and not AdamW)
//   p   = p - (lr * wd) * p                            (only when weight_decay != 0 and AdamW)
//   m   = m * b1 + g' * (1 - b1)
//   v   = v * b2 + (g' * g') * (1 - b2)
//   p   = p + (m / (sqrt(v) / sqrt(1 - b2^t) + eps)) * -(lr / (1 - b1^t))
// b1^t and b2^t come from repeated float32 multiplication, not a library pow. Nothing here is
// contracted into an FMA, and a NaN result always has the bits 0x7FC00000.
#include "repro_math.cuh"
#include "repro_sum_math.cuh"

namespace repro_nn {

using repro_math::fadd;
using repro_math::fdiv;
using repro_math::fmul;
using repro_math::fsub;

struct PoolShape {
    int N, C, H, W;
    int KH, KW;
    int SY, SX;
    __host__ __device__ int OH() const { return (H - KH) / SY + 1; }
    __host__ __device__ int OW() const { return (W - KW) / SX + 1; }
    __host__ __device__ size_t input_size() const { return (size_t)N * C * H * W; }
    __host__ __device__ size_t output_size() const { return (size_t)N * C * OH() * OW(); }
};

// Outputs (oy, ox) of one image and channel. plane points at x[n, c, 0, 0].
__host__ __device__ inline void pool_window(const float* plane, const PoolShape& g, int oy, int ox, float& value, int& index) {
    int y0 = oy * g.SY, x0 = ox * g.SX;
    float best = plane[(size_t)y0 * g.W + x0];
    int at = y0 * g.W + x0;
    for (int ky = 0; ky < g.KH; ++ky) {
        const float* row = plane + (size_t)(y0 + ky) * g.W + x0;
        for (int kx = 0; kx < g.KW; ++kx) {
            if (row[kx] > best) {
                best = row[kx];
                at = (y0 + ky) * g.W + x0 + kx;
            }
        }
    }
    value = best;
    index = at;
}

// Backward at pixel (y, x) of one image and channel. dy and idx point at that plane's outputs.
__host__ __device__ inline float pool_backward_pixel(const float* dy, const int* idx, const PoolShape& g, int y, int x) {
    int oh = g.OH(), ow = g.OW(), at = y * g.W + x, matches = 0;
    float first = 0.0f;
    int64_t acc[repro_sum::LIMBS];
    int nf = 0;
    for (int ky = 0; ky < g.KH; ++ky) {
        int ry = y - ky;
        if (ry < 0 || ry % g.SY != 0 || ry / g.SY >= oh) continue;
        int oy = ry / g.SY;
        for (int kx = 0; kx < g.KW; ++kx) {
            int rx = x - kx;
            if (rx < 0 || rx % g.SX != 0 || rx / g.SX >= ow) continue;
            int o = oy * ow + rx / g.SX;
            if (idx[o] != at) continue;
            if (matches == 0) {
                first = dy[o];
            } else {
                if (matches == 1) {
                    for (int l = 0; l < repro_sum::LIMBS; ++l) acc[l] = 0;
                    repro_sum::acc_add(acc, 1, first, nf);
                }
                repro_sum::acc_add(acc, 1, dy[o], nf);
            }
            ++matches;
        }
    }
    if (matches == 0) return 0.0f;
    if (matches == 1) {                              // +0 + v: one defined sign for a zero
        float v = fadd(0.0f, first);
        return v != v ? repro_math::canonical_nan() : v;
    }
    return repro_sum::finalize(acc, 1, 1u, nf);
}

// ---------------- Adam ----------------

struct AdamConfig {
    float lr = 0.001f, beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f, weight_decay = 0.0f;
    bool adamw = false;
};

// One optimizer's state: step count and the two running powers.
struct AdamClock {
    float b1_pow = 1.0f, b2_pow = 1.0f;
    int step = 0;
};

// The scalars of one step, the same for every parameter.
struct AdamScalars {
    float beta1, beta2, one_minus_b1, one_minus_b2, sqrt_bc2, neg_step, eps, decay;
    bool coupled_decay;
    float wd;
};

__host__ __device__ inline float fsqrt_rn(float x) {
#ifdef __CUDA_ARCH__
    return __fsqrt_rn(x);
#else
    return std::sqrt(x);
#endif
}

// Advances the clock by one step and returns that step's scalars. Plain IEEE operations on the
// host, so any machine gets the same bits.
__host__ __device__ inline AdamScalars adam_next_step(const AdamConfig& c, AdamClock& k) {
    k.b1_pow = fmul(k.b1_pow, c.beta1);
    k.b2_pow = fmul(k.b2_pow, c.beta2);
    ++k.step;
    AdamScalars s;
    s.beta1 = c.beta1;
    s.beta2 = c.beta2;
    s.one_minus_b1 = fsub(1.0f, c.beta1);
    s.one_minus_b2 = fsub(1.0f, c.beta2);
    s.sqrt_bc2 = fsqrt_rn(fsub(1.0f, k.b2_pow));
    s.neg_step = -fdiv(c.lr, fsub(1.0f, k.b1_pow));
    s.eps = c.eps;
    s.wd = c.weight_decay;
    s.coupled_decay = c.weight_decay != 0.0f && !c.adamw;
    s.decay = c.weight_decay != 0.0f && c.adamw ? fmul(c.lr, c.weight_decay) : 0.0f;
    return s;
}

__host__ __device__ inline void adam_update(float& p, float g, float& m, float& v, const AdamScalars& s) {
    if (s.coupled_decay) g = fadd(g, fmul(s.wd, p));
    if (s.decay != 0.0f) p = fsub(p, fmul(s.decay, p));
    m = fadd(fmul(m, s.beta1), fmul(g, s.one_minus_b1));
    v = fadd(fmul(v, s.beta2), fmul(fmul(g, g), s.one_minus_b2));
    float denom = fadd(fdiv(fsqrt_rn(v), s.sqrt_bc2), s.eps);
    p = fadd(p, fmul(fdiv(m, denom), s.neg_step));
    if (p != p) p = repro_math::canonical_nan();
    if (m != m) m = repro_math::canonical_nan();
    if (v != v) v = repro_math::canonical_nan();
}

} // namespace repro_nn
