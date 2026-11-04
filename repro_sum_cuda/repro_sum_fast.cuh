#pragma once
// Device-side fast exact accumulation, used by repro_sum.cu and repro_softmax.cu.
#include "repro_sum_math.cuh"
#include <cuda_runtime.h>
#include <cstdint>

namespace repro_sum {

// ---------------------------------------------------------------------------
// Fast exact accumulation.
//
// A finite float x = +/- m * 2^(sh - 149) with m < 2^24 is the integer
// v = +/- m << (sh mod 32) (|v| < 2^55) placed at limb idx = sh / 32 of the
// 10-limb accumulator (see repro_sum_math.cuh).
//
// Each thread fixes a window base `low` and keeps two 64-bit "wide" accumulators
// a0, a1 that sum v for values with idx == low and idx == low + 1. That costs one
// shift and one 64-bit add per value, with no per-limb work. After at most 255
// adds, |a| < 255 * 2^55 < 2^63, so the wide accumulators are folded into three
// radix-2^32 limbs r[0..2] (limbs low..low+2) by splitting each into its low and
// high 32 bits, which is exact. Values with idx outside {low, low+1} (rare) are
// split the same way and added straight into a full 10-limb accumulator.
//
// All of it is integer addition, so the result is the same exact sum regardless
// of window placement, thread count, or order; the window only affects speed.
// Limb bounds: r and the full accumulator grow by < 2^33 per fold, far below 2^63.
// ---------------------------------------------------------------------------

constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int FOLD_EVERY = 255;
constexpr int DEFAULT_LOW = 3;   // window around |x| ~ 2^-53 .. 2^10 when no sample is available

struct FastAcc {
    int64_t a0, a1;   // wide accumulators for idx == low, low + 1
    int64_t r[3];     // radix-2^32 limbs low, low + 1, low + 2
    int n;            // adds since the last fold
};

__device__ __forceinline__ void fast_init(FastAcc& s) {
    s.a0 = s.a1 = 0;
    s.r[0] = s.r[1] = s.r[2] = 0;
    s.n = 0;
}

__device__ __forceinline__ void fast_fold(FastAcc& s) {
    s.r[0] += (int64_t)(uint32_t)s.a0;
    s.r[1] += s.a0 >> 32;            // arithmetic shift: a0 = low32 + (a0 >> 32) * 2^32 exactly
    s.r[1] += (int64_t)(uint32_t)s.a1;
    s.r[2] += s.a1 >> 32;
    s.a0 = s.a1 = 0;
    s.n = 0;
}

// Window base for values whose shift is around sh: keeps about 16 bits of headroom
// above sh and 16-48 bits below it in the fast path.
__device__ __forceinline__ int window_low_for(int sh) {
    int low = ((sh + 16) >> 5) - 1;
    return low < 0 ? 0 : (low > LIMBS - 3 ? LIMBS - 3 : low);
}

// Shift of a nonzero finite value (x = m * 2^(sh-149)), or -1 for zero/NaN/Inf.
__device__ __forceinline__ int value_shift(float x) {
    uint32_t b = __float_as_uint(x);
    uint32_t e = (b >> 23) & 0xFFu;
    if (e == 0xFFu || (b & 0x7FFFFFFFu) == 0) return -1;
    return e ? (int)e - 1 : 0;
}

// Adds x. full: 10-limb overflow accumulator (shared memory if SHARED, else private).
template<bool SHARED>
__device__ __forceinline__ void fast_add(FastAcc& s, int low, float x, int64_t* full, int& nf) {
    uint32_t b = __float_as_uint(x);
    uint32_t e = (b >> 23) & 0xFFu;
    if (e == 0xFFu) { nf = 1; return; }
    uint32_t f = b & 0x7FFFFFu;
    uint64_t m = e ? (uint64_t)(f | 0x800000u) : (uint64_t)f;
    int sh = e ? (int)e - 1 : 0;
    int64_t v = (int64_t)(m << (sh & 31));
    if (b >> 31) v = -v;
    int d = (sh >> 5) - low;
    if (d == 0) {
        s.a0 += v;
    } else if (d == 1) {
        s.a1 += v;
    } else if (v) {
        int idx = sh >> 5;
        int64_t lo = (int64_t)(uint32_t)v, hi = v >> 32;
        if (SHARED) {
            atomicAdd(reinterpret_cast<unsigned long long*>(&full[idx]), (unsigned long long)lo);
            atomicAdd(reinterpret_cast<unsigned long long*>(&full[idx + 1]), (unsigned long long)hi);
        } else {
            full[idx] += lo;
            full[idx + 1] += hi;
        }
    }
    if (++s.n == FOLD_EVERY) fast_fold(s);
}

__device__ __forceinline__ int64_t warp_sum(int64_t v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xFFFFFFFFu, v, o);
    return v;
}

__device__ __forceinline__ int warp_max(int v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = max(v, __shfl_xor_sync(0xFFFFFFFFu, v, o));
    return v;
}

// Adds a contiguous segment p[0, len) using `lanes` cooperating threads with id `t`:
// scalar head up to 16-byte alignment, float4 body, scalar tail.
template<bool SHARED>
__device__ __forceinline__ void fast_segment(FastAcc& s, int low, const float* p, int len, int t, int lanes,
                                            int64_t* full, int& nf) {
    int head = (int)((4 - ((reinterpret_cast<uintptr_t>(p) >> 2) & 3)) & 3);
    if (head > len) head = len;
    if (t < head) fast_add<SHARED>(s, low, p[t], full, nf);
    int nvec = (len - head) >> 2;
    const float4* pv = reinterpret_cast<const float4*>(p + head);
    for (int j = t; j < nvec; j += lanes) {
        float4 v = pv[j];
        fast_add<SHARED>(s, low, v.x, full, nf);
        fast_add<SHARED>(s, low, v.y, full, nf);
        fast_add<SHARED>(s, low, v.z, full, nf);
        fast_add<SHARED>(s, low, v.w, full, nf);
    }
    int done = head + (nvec << 2);
    if (t < len - done) fast_add<SHARED>(s, low, p[done + t], full, nf);
}

} // namespace repro_sum
