#pragma once
// Correctly rounded exp and log for float32.
//
//   repro_expf(x) = exp(x) rounded to the nearest float32, ties to even
//   repro_logf(x) = log(x) rounded to the nearest float32, ties to even
//
// The result is defined by mathematics, not by this code, so it is the same on every
// GPU, CUDA version, and CPU. Each function uses only IEEE + - * / fma (explicit
// round-to-nearest intrinsics on the GPU) and tables of exact constants: no math
// library and no approximate hardware instructions.
//
// Fast path (FP32 only): float-float arithmetic, hi + lo, gives about 45 good bits.
// If the result is too close to a rounding midpoint for those bits to decide, the slow
// path recomputes it in double-double (about 100 bits). Over all float32 inputs the
// true value never comes closer to a midpoint than 2^-51.7 (exp) or 2^-57 (log), so the
// slow path always decides. check_repro_math.cpp verifies every one of the 2^32 inputs.
//
// NaN results always have the bits 0x7FC00000.
//
// Do not build with --use_fast_math, -ftz=true, or -ffast-math. A CPU build also needs
// -ffp-contract=off: the plain C++ operators below must not be fused into FMAs.
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
#include "repro_math_tables.h"

#if defined(__FAST_MATH__)
#error "repro_math.cuh must not be built with -ffast-math"
#endif

namespace repro_math {

struct ExpEntry { float hi, lo; };
struct LogEntry { float c, hi, lo; };

#if defined(__CUDACC__)
static __constant__ const ExpEntry kExpTableDevice[64] = REPRO_EXP_TABLE_DATA;
static __constant__ const LogEntry kLogTableDevice[1 << REPRO_LOG_N] = REPRO_LOG_TABLE_DATA;
#endif
static const ExpEntry kExpTableHost[64] = REPRO_EXP_TABLE_DATA;
static const LogEntry kLogTableHost[1 << REPRO_LOG_N] = REPRO_LOG_TABLE_DATA;

// ---------------- IEEE operations with explicit rounding ----------------
#ifdef __CUDA_ARCH__
__device__ __forceinline__ float fadd(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float fsub(float a, float b) { return __fsub_rn(a, b); }
__device__ __forceinline__ float fmul(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float ffma(float a, float b, float c) { return __fmaf_rn(a, b, c); }
__device__ __forceinline__ float fdiv(float a, float b) { return __fdiv_rn(a, b); }
__device__ __forceinline__ double dadd(double a, double b) { return __dadd_rn(a, b); }
__device__ __forceinline__ double dsub(double a, double b) { return __dsub_rn(a, b); }
__device__ __forceinline__ double dmul(double a, double b) { return __dmul_rn(a, b); }
__device__ __forceinline__ double ddiv(double a, double b) { return __ddiv_rn(a, b); }
__device__ __forceinline__ double dfma(double a, double b, double c) { return __fma_rn(a, b, c); }
__device__ __forceinline__ uint32_t bits_of(float x) { return __float_as_uint(x); }
__device__ __forceinline__ float float_of(uint32_t b) { return __uint_as_float(b); }
__device__ __forceinline__ uint64_t bits_of(double x) { return (uint64_t)__double_as_longlong(x); }
__device__ __forceinline__ double double_of(uint64_t b) { return __longlong_as_double((long long)b); }
__device__ __forceinline__ ExpEntry exp_entry(int j) { return kExpTableDevice[j]; }
__device__ __forceinline__ LogEntry log_entry(int i) { return kLogTableDevice[i]; }
#else
__host__ __device__ inline float fadd(float a, float b) { return a + b; }
__host__ __device__ inline float fsub(float a, float b) { return a - b; }
__host__ __device__ inline float fmul(float a, float b) { return a * b; }
__host__ __device__ inline float ffma(float a, float b, float c) { return std::fmaf(a, b, c); }
__host__ __device__ inline float fdiv(float a, float b) { return a / b; }
__host__ __device__ inline double dadd(double a, double b) { return a + b; }
__host__ __device__ inline double dsub(double a, double b) { return a - b; }
__host__ __device__ inline double dmul(double a, double b) { return a * b; }
__host__ __device__ inline double ddiv(double a, double b) { return a / b; }
__host__ __device__ inline double dfma(double a, double b, double c) { return std::fma(a, b, c); }
__host__ __device__ inline uint32_t bits_of(float x) { uint32_t b; std::memcpy(&b, &x, 4); return b; }
__host__ __device__ inline float float_of(uint32_t b) { float x; std::memcpy(&x, &b, 4); return x; }
__host__ __device__ inline uint64_t bits_of(double x) { uint64_t b; std::memcpy(&b, &x, 8); return b; }
__host__ __device__ inline double double_of(uint64_t b) { double x; std::memcpy(&x, &b, 8); return x; }
__host__ __device__ inline ExpEntry exp_entry(int j) { return kExpTableHost[j]; }
__host__ __device__ inline LogEntry log_entry(int i) { return kLogTableHost[i]; }
#endif

__host__ __device__ inline float canonical_nan() { return float_of(0x7FC00000u); }
__host__ __device__ inline float positive_inf() { return float_of(0x7F800000u); }
__host__ __device__ inline float abs_of(float x) { return float_of(bits_of(x) & 0x7FFFFFFFu); }
// 2^e for -126 <= e <= 127.
__host__ __device__ inline float pow2f(int e) { return float_of((uint32_t)(e + 127) << 23); }

// ---------------- float-float: error-free transformations ----------------
struct ff { float hi, lo; };

// a + b == hi + lo exactly, for any a, b.
__host__ __device__ inline ff two_sum(float a, float b) {
    float s = fadd(a, b), bb = fsub(s, a);
    return {s, fadd(fsub(a, fsub(s, bb)), fsub(b, bb))};
}
// a + b == hi + lo exactly, when |a| >= |b| or a == 0.
__host__ __device__ inline ff fast_two_sum(float a, float b) {
    float s = fadd(a, b);
    return {s, fsub(b, fsub(s, a))};
}
// a * b == hi + lo exactly (barring underflow of lo).
__host__ __device__ inline ff two_prod(float a, float b) {
    float p = fmul(a, b);
    return {p, ffma(a, b, -p)};
}

// ---------------- double-double: the slow path ----------------
struct dd { double hi, lo; };

__host__ __device__ inline dd dd_two_sum(double a, double b) {
    double s = dadd(a, b), bb = dsub(s, a);
    return {s, dadd(dsub(a, dsub(s, bb)), dsub(b, bb))};
}
__host__ __device__ inline dd dd_fast_two_sum(double a, double b) {
    double s = dadd(a, b);
    return {s, dsub(b, dsub(s, a))};
}
__host__ __device__ inline dd dd_two_prod(double a, double b) {
    double p = dmul(a, b);
    return {p, dfma(a, b, -p)};
}
__host__ __device__ inline dd dd_add(dd a, dd b) {
    dd s = dd_two_sum(a.hi, b.hi);
    return dd_fast_two_sum(s.hi, dadd(s.lo, dadd(a.lo, b.lo)));
}
__host__ __device__ inline dd dd_add(dd a, double b) {
    dd s = dd_two_sum(a.hi, b);
    return dd_fast_two_sum(s.hi, dadd(s.lo, a.lo));
}
__host__ __device__ inline dd dd_mul(dd a, dd b) {
    dd p = dd_two_prod(a.hi, b.hi);
    return dd_fast_two_sum(p.hi, dadd(p.lo, dadd(dmul(a.hi, b.lo), dmul(a.lo, b.hi))));
}
__host__ __device__ inline dd dd_mul(dd a, double b) {
    dd p = dd_two_prod(a.hi, b);
    return dd_fast_two_sum(p.hi, dadd(p.lo, dmul(a.lo, b)));
}

// exp(x) in double-double, relative error about 2^-96:
// exp(x) = 2^k * exp(r)^(2^8), r = (x - k ln2) / 2^8, Taylor series.
__host__ __device__ inline dd dd_exp(double x) {
    const dd LN2 = {0x1.62e42fefa39efp-1, 0x1.abc9e3b39803fp-56};
    double fk = dadd(ddiv(x, LN2.hi), x >= 0 ? 0.5 : -0.5);
    int k = (int)fk;                                 // truncation, so k is the nearest integer
    dd pk = dd_mul(LN2, (double)k);
    dd r = dd_add(dd{x, 0.0}, dd{-pk.hi, -pk.lo});
    r.hi = dmul(r.hi, 0x1p-8);
    r.lo = dmul(r.lo, 0x1p-8);
    dd term = {1.0, 0.0}, sum = {1.0, 0.0};
    for (int n = 1; n <= 14; ++n) {
        term = dd_mul(term, r);
        double q = ddiv(term.hi, (double)n);         // term / n in double-double
        dd qn = dd_two_prod(q, (double)n);
        double rem = ddiv(dadd(dsub(dsub(term.hi, qn.hi), qn.lo), term.lo), (double)n);
        term = dd_fast_two_sum(q, rem);
        sum = dd_add(sum, term);
    }
    for (int i = 0; i < 8; ++i) sum = dd_mul(sum, sum);
    double scale = double_of((uint64_t)(k + 1023) << 52);  // exact: every result is a normal double
    return {dmul(sum.hi, scale), dmul(sum.lo, scale)};
}

// Rounds hi + lo once to float32. hi alone is first moved to the neighbour with an odd
// last bit on the side of the true value ("round to odd"); with 53 >= 24 + 2 bits that
// makes the cast below equal to rounding the exact value, including for float32
// subnormals and overflow, so hi landing exactly on a float midpoint cannot mislead it.
__host__ __device__ inline float dd_to_float(dd v) {
    uint64_t b = bits_of(v.hi);
    if (v.lo != 0.0 && (b & 1u) == 0) b += ((v.lo > 0.0) == (v.hi > 0.0)) ? 1 : (uint64_t)-1;
    return (float)double_of(b);
}

// y0 is any approximation of log(x) good to about 2^-40. One Newton step in double-double.
__host__ __device__ inline dd dd_log_from(double x, double y0) {
    dd t = dd_mul(dd_exp(-y0), x);                   // x * exp(-y0), close to 1
    dd corr = dd_add(t, -1.0);
    return dd_add(dd{y0, 0.0}, corr);
}

// ---------------- exp ----------------

// Relative error bounds of the fast paths. Verified by check_repro_math.cpp, which
// measures the largest actual error over all inputs.
constexpr float EXP_DELTA = 0x1p-44f;   // absolute, for y in [0.7, 1.42]
constexpr float LOG_EPS = 0x1p-44f;       // relative to |y|

// exp(x) = (hi + lo) * 2^m, with hi + lo in about [0.7, 1.42]. |x| below the overflow
// and underflow thresholds.
__host__ __device__ inline ff exp_core(float x, int& m) {
    float t = fmul(x, REPRO_EXP_INV_C);
    int k = (int)fadd(t, x >= 0.0f ? 0.5f : -0.5f);  // truncation, so k is the nearest integer
    int j = k & 63;
    m = k >> 6;                                      // arithmetic shift: k = 64 m + j
    float fk = (float)k;

    // r = x - k * ln2/64 in float-float. k * C1 is exact and so is x - k * C1.
    float d = fsub(x, fmul(fk, REPRO_EXP_C1));
    ff p = two_prod(fk, REPRO_EXP_C2);
    ff r = two_sum(d, -p.hi);
    r = fast_two_sum(r.hi, fsub(fsub(r.lo, p.lo), fmul(fk, REPRO_EXP_C3)));

    // exp(r) = 1 + r + r^2/2 + tail; the leading part in float-float, the tail in float.
    ff s = two_prod(r.hi, r.hi);
    float s2_hi = fmul(s.hi, 0.5f);
    float s2_lo = fadd(fmul(s.lo, 0.5f), fmul(r.hi, r.lo));
    float tail = fmul(fmul(s.hi, r.hi),
                      fadd(REPRO_EXP_P3, fmul(r.hi, fadd(REPRO_EXP_P4, fmul(r.hi, REPRO_EXP_P5)))));
    ff a = fast_two_sum(1.0f, r.hi);
    ff c = fast_two_sum(a.hi, s2_hi);
    ff e = fast_two_sum(c.hi, fadd(fadd(fadd(a.lo, c.lo), fadd(r.lo, s2_lo)), tail));

    // Multiply by the table entry T[j] = 2^(j/64).
    ExpEntry T = exp_entry(j);
    ff y = two_prod(T.hi, e.hi);
    return fast_two_sum(y.hi, fadd(y.lo, fadd(fmul(T.hi, e.lo), fmul(T.lo, e.hi))));
}

__host__ __device__ inline float expf_slow(float x) { return dd_to_float(dd_exp((double)x)); }

// A subnormal result: y * 2^m rounds on the fixed grid 2^-149. Adding c = 2^(-126-m) to y
// puts the grid at the last place of the float near c, so an ordinary addition rounds at
// the right position. Requires y < c. Returns false if the slow path must decide.
__host__ __device__ inline bool exp_subnormal(ff y, int m, float& out) {
    float c = pow2f(-126 - m);
    float half = fmul(c, 0x1p-24f);                  // half the grid spacing
    ff s = fast_two_sum(c, y.hi);
    ff l = two_sum(s.lo, y.lo);
    // Too close to a tie, within the fast path's error or the bits of l.lo not looked at?
    if (abs_of(fsub(half, abs_of(l.hi))) <= fadd(EXP_DELTA, fmul(half, 0x1p-22f))) return false;
    float q = fsub(fadd(s.hi, l.hi), c);             // exact, on the grid
    out = fmul(fmul(q, pow2f(m + 64)), 0x1p-64f);    // both exact
    return true;
}

__host__ __device__ inline float repro_expf(float x) {
    if ((bits_of(x) & 0x7FFFFFFFu) > 0x7F800000u) return canonical_nan();
    if (x >= REPRO_EXP_OVERFLOW) return positive_inf();
    if (x <= REPRO_EXP_UNDERFLOW) return 0.0f;
    int m;
    ff y = exp_core(x, m);
    if (m <= -126 && (m < -126 || y.hi < 1.0f || (y.hi == 1.0f && y.lo < 0.0f))) {
        float out;
        return exp_subnormal(y, m, out) ? out : expf_slow(x);
    }
    // Rounding is monotonic: if both ends of the error interval round to the same float,
    // so does the true value. The scale is exact (or overflows exactly when it should).
    float up = fadd(y.hi, fadd(y.lo, EXP_DELTA));
    float dn = fadd(y.hi, fsub(y.lo, EXP_DELTA));
    if (up != dn) return expf_slow(x);
    int m1 = m >> 1;
    return fmul(fmul(up, pow2f(m1)), pow2f(m - m1));
}

// ---------------- log ----------------

// log(x) = hi + lo, for finite x > 0, x != 1.
__host__ __device__ inline ff log_core(float x) {
    uint32_t ix = bits_of(x);
    int e = 0;
    if (ix < 0x00800000u) {                          // subnormal: scale up exactly
        ix = bits_of(fmul(x, 0x1p23f));
        e = -23;
    }
    // x = 2^e * z with z in [0.699, 1.398), so z never sits just below 2: no cancellation
    // between e * ln2 and log(z) when x is close to 1.
    uint32_t tmp = ix - REPRO_LOG_OFF;
    int i = (int)((tmp >> (23 - REPRO_LOG_N)) & ((1u << REPRO_LOG_N) - 1));
    e += (int)((int32_t)tmp >> 23);
    float z = float_of(ix - (tmp & 0xFF800000u));
    LogEntry T = log_entry(i);

    // r = z * c - 1, exactly, as hi + lo.
    ff p = two_prod(z, T.c);
    ff r = fast_two_sum(fsub(p.hi, 1.0f), p.lo);

    // log1p(r) = r - r^2/2 + r^3/3 - tail; everything but the tail in float-float, since
    // when x is close to 1 the result is about r and needs its full relative accuracy.
    ff s = two_prod(r.hi, r.hi);
    float s_lo = fadd(s.lo, fmul(fadd(r.hi, r.hi), r.lo));
    ff cube = two_prod(s.hi, r.hi);
    float cube_lo = fadd(cube.lo, fadd(fmul(s_lo, r.hi), fmul(s.hi, r.lo)));
    ff th = two_prod(cube.hi, REPRO_LOG_THIRD);
    float th_lo = fadd(th.lo, fadd(fmul(cube.hi, REPRO_LOG_THIRDL), fmul(cube_lo, REPRO_LOG_THIRD)));
    float tail = fmul(fmul(cube.hi, r.hi),
                      fadd(-REPRO_LOG_P4, fmul(r.hi, fadd(REPRO_LOG_P5, fmul(r.hi,
                          fadd(-REPRO_LOG_P6, fmul(r.hi, REPRO_LOG_P7)))))));

    // Sum: e * ln2 + L + log1p(r), collecting every rounding error in lo.
    float fe = (float)e;
    ff a = fast_two_sum(fmul(fe, REPRO_LOG_LN2H), T.hi);
    ff pm = two_prod(fe, REPRO_LOG_LN2M);
    ff b = two_sum(a.hi, pm.hi);
    ff c = two_sum(b.hi, r.hi);
    ff d = two_sum(c.hi, -fmul(s.hi, 0.5f));
    ff f = two_sum(d.hi, th.hi);
    float lo = fadd(fadd(fadd(a.lo, b.lo), fadd(c.lo, d.lo)), f.lo);
    lo = fadd(lo, fadd(fadd(T.lo, pm.lo), fmul(fe, REPRO_LOG_LN2L)));
    lo = fadd(lo, fadd(fsub(r.lo, fmul(s_lo, 0.5f)), fadd(th_lo, tail)));
    return fast_two_sum(f.hi, lo);
}

__host__ __device__ inline float repro_logf(float x) {
    uint32_t b = bits_of(x);
    if (b >= 0x7F800000u || b == 0u) {               // +inf, NaN, negatives, or +0
        if ((b & 0x7FFFFFFFu) == 0u) return -positive_inf();
        return b == 0x7F800000u ? x : canonical_nan();
    }
    if (b == 0x3F800000u) return 0.0f;               // log(1) = +0, the only exact case
    ff y = log_core(x);
    float eps = fmul(LOG_EPS, abs_of(y.hi));
    float up = fadd(y.hi, fadd(y.lo, eps));
    float dn = fadd(y.hi, fsub(y.lo, eps));
    if (up != dn) return dd_to_float(dd_log_from((double)x, dadd((double)y.hi, (double)y.lo)));
    return up;
}

} // namespace repro_math
