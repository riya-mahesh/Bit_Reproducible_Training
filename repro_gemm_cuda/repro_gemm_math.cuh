#pragma once
// The scalar arithmetic that defines repro_gemm's output. These functions are
// __host__ __device__ so the GPU kernels and the CPU reference in the tests use
// literally the same code. Every operation here is either exact or a single
// IEEE rounding with a fixed mode, so results do not depend on the GPU.
#if defined(__CUDACC__)
#include <cuda_runtime.h>
#else  // plain C++ (CPU-only builds, e.g. a verifier without the CUDA toolkit)
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#endif
#include <cmath>
#include <cstdint>
#include <cstring>

namespace repro_gemm {

constexpr int MAX_SLICES = 3;

__host__ __device__ inline uint32_t float_to_bits(float x) {
#ifdef __CUDA_ARCH__
    return __float_as_uint(x);
#else
    uint32_t b;
    std::memcpy(&b, &x, 4);
    return b;
#endif
}

__host__ __device__ inline float bits_to_float(uint32_t b) {
#ifdef __CUDA_ARCH__
    return __uint_as_float(b);
#else
    float x;
    std::memcpy(&x, &b, 4);
    return x;
#endif
}

// x * 2^n, the same as ldexpf(x, n) for every use below but without a library call. Each
// factor is a normal float (|n| <= 252), and the product is exact unless the result is a
// subnormal, where the first factor cannot round: it only moves x toward the final value.
__host__ __device__ inline float scale_by_pow2(float x, int n) {
    int n1 = n >> 1, n2 = n - n1;
    float p1 = bits_to_float((uint32_t)(n1 + 127) << 23), p2 = bits_to_float((uint32_t)(n2 + 127) << 23);
#ifdef __CUDA_ARCH__
    return __fmul_rn(__fmul_rn(x, p1), p2);
#else
    return x * p1 * p2;
#endif
}

// Exponent e of the per-row/column scale 2^e: the smallest power of two
// >= m/127 (m = max |x|). Returns 0 (scale 1) when m/127 is zero.
__host__ __device__ inline int scale_exponent(float m) {
    if (!(m > 0.0f)) return 0;
    float v = m / 127.0f;
    uint32_t b = float_to_bits(v);
    int E = (int)((b >> 23) & 0xFFu);
    if (E != 0) return (b & 0x7FFFFFu) == 0 ? E - 127 : E - 126;  // v = 1.f * 2^(E-127)
    int e;                                // subnormal v
    float frac = frexpf(v, &e);           // v = frac * 2^e, frac in [0.5,1)
    if (frac == 0.0f) return 0;           // m/127 underflowed to zero
    return frac == 0.5f ? e - 1 : e;
}

// Fixed-point integer for x with row scale 2^se and S slices of 7 bits:
// X = round_to_nearest_even(x / 2^(se - 7(S-1))), clamped to |X| <= 127 * 2^(7(S-1)).
// Multiplying by a power of two is exact here, so this is a single rounding.
__host__ __device__ inline int fixed_point(float x, int se, int S) {
    float v = scale_by_pow2(x, 7 * (S - 1) - se);
#ifdef __CUDA_ARCH__
    int X = __float2int_rn(v);
#else
    int X = (int)nearbyintf(v);  // default rounding mode: nearest-even
#endif
    int lim = 127 << (7 * (S - 1));
    return X > lim ? lim : (X < -lim ? -lim : X);
}

__host__ __device__ inline int clz64(uint64_t x) {
#ifdef __CUDA_ARCH__
    return __clzll((long long)x);
#else
    return __builtin_clzll(x);
#endif
}

// Correctly rounded (nearest-even) FP32 value of the exact number T * 2^E,
// including subnormal results, signed zero, and overflow to infinity.
// Only integer operations plus one exact ldexpf, so it is the same everywhere.
__host__ __device__ inline float round_to_float(int64_t T, int E) {
    if (T == 0) return 0.0f;
    bool neg = T < 0;
    uint64_t m = neg ? (uint64_t)(-(T + 1)) + 1u : (uint64_t)T;
    int p = 63 - clz64(m);            // T * 2^E has its leading bit at 2^(p+E)
    int shift = p - 23;               // keep 24 significant bits...
    if (-149 - E > shift) shift = -149 - E;  // ...but nothing below 2^-149
    uint64_t q;
    int ex;
    if (shift <= 0) {                 // exactly representable
        q = m; ex = E;
    } else if (shift >= 64) {         // below half of the smallest subnormal
        q = 0; ex = 0;
    } else {
        q = m >> shift;
        uint64_t rem = m & ((1ull << shift) - 1), half = 1ull << (shift - 1);
        if (rem > half || (rem == half && (q & 1))) ++q;
        ex = E + shift;
    }
    // q <= 2^24 and q * 2^ex is representable (or overflows to inf), so this is exact.
    // ex > 127 means q >= 1 times at least 2^128: infinity, with no factor to build.
    float f = (q != 0 && ex > 127) ? bits_to_float(0x7F800000u) : scale_by_pow2((float)q, ex);
    return neg ? -f : f;
}

} // namespace repro_gemm
