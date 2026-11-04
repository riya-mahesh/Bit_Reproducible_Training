#pragma once
// Exact, order-independent summation of float32 values ("superaccumulator").
//
// Every finite float32 is an integer multiple of 2^-149 below 2^128, i.e. an
// integer U < 2^277 in units of 2^-149. A sum of up to 2^31 - 1 such values is
// therefore an integer below 2^308, held here as LIMBS = 10 signed 64-bit limbs
// of radix 2^32 (value = sum_l limb[l] * 2^(32 l) * 2^-149). Each input adds a
// value of magnitude < 2^32 to at most two limbs, so limbs cannot overflow for
// up to 2^31 - 1 inputs and carries are only resolved once, at the end.
//
// Integer addition is exact and associative, so accumulators can be built and
// merged in any order or grouping and always hold the same exact sum. The result
// is that exact sum (or exact sum / n) rounded once to float32, nearest-even.
//
// These functions are __host__ __device__: the GPU kernels and the CPU reference
// in the tests use the same code.
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
#include <cstdint>
#include <cstring>
#include <cmath>

namespace repro_sum {

constexpr int LIMBS = 10;

__host__ __device__ inline uint32_t float_bits(float x) {
#ifdef __CUDA_ARCH__
    return __float_as_uint(x);
#else
    uint32_t b;
    std::memcpy(&b, &x, 4);
    return b;
#endif
}

__host__ __device__ inline int clz32(uint32_t x) {
#ifdef __CUDA_ARCH__
    return __clz((int)x);
#else
    return __builtin_clz(x);
#endif
}

__host__ __device__ inline float quiet_nan() {
#ifdef __CUDA_ARCH__
    return __int_as_float(0x7FC00000);
#else
    return std::nanf("");
#endif
}

__host__ __device__ inline uint32_t get_bit(const uint32_t* q, int pos) {
    return (q[pos >> 5] >> (pos & 31)) & 1u;
}

// Splits x into (limb index, low addend, high addend) with
// x = (lo * 2^(32 idx) + hi * 2^(32 (idx+1))) * 2^-149 exactly.
// Returns false for NaN/Inf (addends are then zero).
__host__ __device__ inline bool split_float(float x, int& idx, int64_t& lo, int64_t& hi) {
    uint32_t b = float_bits(x);
    uint32_t e = (b >> 23) & 0xFFu, f = b & 0x7FFFFFu;
    if (e == 0xFFu) { idx = 0; lo = hi = 0; return false; }
    uint64_t m = e ? (uint64_t)(f | 0x800000u) : (uint64_t)f;  // 24-bit significand
    int sh = e ? (int)e - 1 : 0;                                // x = m * 2^(sh - 149)
    idx = sh >> 5;                                              // 0..7
    uint64_t v = m << (sh & 31);                                // < 2^55
    lo = (int64_t)(v & 0xFFFFFFFFull);
    hi = (int64_t)(v >> 32);
    if (b >> 31) { lo = -lo; hi = -hi; }
    return true;
}

// Adds x to an accumulator whose limb l is at acc[l * stride].
__host__ __device__ inline void acc_add(int64_t* acc, int stride, float x, int& nonfinite) {
    int idx;
    int64_t lo, hi;
    if (!split_float(x, idx, lo, hi)) { nonfinite = 1; return; }
    acc[idx * stride] += lo;
    acc[(idx + 1) * stride] += hi;
}

// Resolves carries so limbs 0..LIMBS-2 are in [0, 2^32) and the top limb holds the sign.
__host__ __device__ inline void normalize(int64_t* t) {
    for (int l = 0; l < LIMBS - 1; ++l) {
        int64_t c = t[l] >> 32;          // floor division by 2^32 (arithmetic shift)
        t[l] -= c * 4294967296LL;
        t[l + 1] += c;
    }
}

// Exact value (sum of acc limbs) / divisor, rounded to nearest-even float32.
// divisor = 1 gives the sum; divisor = n gives the mean. 1 <= divisor < 2^31.
// An exactly zero result is +0.0f; a nonzero result too small to represent
// rounds to a signed zero; overflow gives +/-inf; nonfinite input gives NaN.
__host__ __device__ inline float finalize(const int64_t* acc, int stride, uint32_t divisor, int nonfinite) {
    if (nonfinite) return quiet_nan();
    int64_t t[LIMBS];
    for (int l = 0; l < LIMBS; ++l) t[l] = acc[l * stride];
    normalize(t);
    bool neg = t[LIMBS - 1] < 0;
    if (neg) {
        for (int l = 0; l < LIMBS; ++l) t[l] = -t[l];
        normalize(t);
    }
    // Now t is the magnitude: limbs in [0, 2^32), value < 2^308 (top limb < 2^20).
    // Exact long division by the divisor, most significant limb first.
    uint32_t q[LIMBS];
    uint64_t rem = 0;
    if (divisor == 1) {
        for (int l = 0; l < LIMBS; ++l) q[l] = (uint32_t)t[l];
    } else {
        for (int l = LIMBS - 1; l >= 0; --l) {
            uint64_t cur = (rem << 32) | (uint64_t)t[l];  // < divisor * 2^32 <= 2^63
            q[l] = (uint32_t)(cur / divisor);
            rem = cur % divisor;
        }
    }
    int top = LIMBS - 1;
    while (top >= 0 && q[top] == 0) --top;
    float r;
    if (top < 0) {
        // |value| = rem/divisor * 2^-149 < 2^-149: rounds to 2^-149 only if above half
        // (a tie rounds to the even value, 0).
        if (rem == 0) return 0.0f;
        r = (2 * rem > divisor) ? ldexpf(1.0f, -149) : 0.0f;
        return neg ? -r : r;
    }
    int p = 32 * top + 31 - clz32(q[top]);           // index of the leading bit
    int shift = p - 23 > 0 ? p - 23 : 0;               // keep 24 bits, never below 2^-149
    // Kept bits [shift, shift + 24): they span at most two limbs.
    int kl = shift >> 5;
    uint64_t two = (uint64_t)q[kl] | (kl + 1 < LIMBS ? (uint64_t)q[kl + 1] << 32 : 0);
    uint32_t kept = (uint32_t)(two >> (shift & 31)) & 0xFFFFFFu;
    bool up;
    if (shift > 0) {
        int g = shift - 1;                             // guard bit position
        bool guard = get_bit(q, g);
        bool lower = rem != 0;                         // sticky: anything below the guard bit
        for (int l = 0; l < (g >> 5); ++l) lower = lower || q[l] != 0;
        lower = lower || (q[g >> 5] & ((1u << (g & 31)) - 1u)) != 0;
        up = guard && (lower || (kept & 1u));
    } else {
        up = 2 * rem > divisor || (2 * rem == divisor && (kept & 1u));
    }
    kept += up;                                        // <= 2^24, exactly representable
    r = ldexpf((float)kept, shift - 149);              // exact, or overflow to inf
    return neg ? -r : r;
}

} // namespace repro_sum
