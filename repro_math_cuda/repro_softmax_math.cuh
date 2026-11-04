#pragma once
// Definition of reproducible softmax over one row of n float32 values:
//
//   m    = largest non-NaN x[i]
//   e[i] = repro_expf(x[i] - m)            correctly rounded; the largest entry is exactly 1
//   S    = exact sum of all e[i], rounded once to float32   (independent of order)
//   p[i] = e[i] / S                        IEEE division
//
// Each step has one defined answer, so the output does not depend on order, batching, or
// hardware. -inf entries (masked attention positions) give exactly 0. If the row has NaN
// or +inf, or every entry is -inf, the whole row is NaN (0x7FC00000).
#include "repro_math.cuh"
#include "repro_sum_math.cuh"

namespace repro_math {

// x may equal p. CPU reference; the GPU kernels in repro_softmax.cu apply the same steps.
inline void softmax_row(const float* x, float* p, int n) {
    float m = -positive_inf();
    for (int i = 0; i < n; ++i) if (x[i] > m) m = x[i];
    int64_t acc[repro_sum::LIMBS] = {};
    int nf = 0;
    for (int i = 0; i < n; ++i) {
        p[i] = repro_expf(fsub(x[i], m));
        repro_sum::acc_add(acc, 1, p[i], nf);
    }
    float S = repro_sum::finalize(acc, 1, 1u, nf);
    for (int i = 0; i < n; ++i) p[i] = S != S ? canonical_nan() : fdiv(p[i], S);
}

// ---------------- cross-entropy ----------------
//
// Definition for logits x[rows, cols] and class indices target[rows]:
//
//   m, e[i], S    as in softmax, for each row
//   loss[r]       = repro_logf(S) - (x[t] - m)           IEEE subtraction, 0 for ignored rows
//   L             = exact sum of loss[r] / count, rounded once     count = rows not ignored
//   dL/dx[r][i]   = seed * (e[i] / S - (i == t ? 1 : 0)) / count   each operation IEEE
//
// A row whose target equals ignore_index contributes nothing and has an all +0 gradient.
// A target that is neither ignored nor in [0, cols) gives NaN, as do NaN or +inf logits or a
// -inf logit at the target class. If every row is ignored, L is NaN. NaN is always 0x7FC00000.

// Whether a row counts: 0 ignored, 1 valid, -1 invalid target.
__host__ __device__ inline int ce_row_kind(int t, int ignore_index, int cols) {
    return t == ignore_index ? 0 : (t >= 0 && t < cols ? 1 : -1);
}

__host__ __device__ inline float ce_row_loss(float S, float x_t, float m, int kind) {
    if (kind == 0) return 0.0f;
    if (kind < 0) return canonical_nan();
    float l = fsub(repro_logf(S), fsub(x_t, m));
    return l != l ? canonical_nan() : l;
}

__host__ __device__ inline float ce_grad(float e, float S, bool is_target, int kind, int count, float seed) {
    if (kind == 0) return 0.0f;
    if (kind < 0 || S != S) return canonical_nan();
    float p = fdiv(e, S);
    if (is_target) p = fsub(p, 1.0f);
    float g = fmul(fdiv(p, (float)count), seed);
    return g != g ? canonical_nan() : g;
}

// CPU version. Returns L; grad[rows * cols] may be null. par(n, f) runs f(0..n-1), in any
// order and on any threads, since rows do not depend on each other.
template<class Par>
inline float cross_entropy_cpu(const float* x, const int* target, int ignore_index, int rows, int cols,
                               float* grad, float seed, Par par) {
    float* loss = new float[rows];
    int count = 0;
    for (int r = 0; r < rows; ++r) count += ce_row_kind(target[r], ignore_index, cols) != 0;
    par(rows, [&](int r) {
        const float* p = x + (size_t)r * cols;
        float* e = new float[cols];
        float m = -positive_inf();
        for (int i = 0; i < cols; ++i) if (p[i] > m) m = p[i];
        int64_t acc[repro_sum::LIMBS] = {};
        int nf = 0;
        for (int i = 0; i < cols; ++i) {
            e[i] = repro_expf(fsub(p[i], m));
            repro_sum::acc_add(acc, 1, e[i], nf);
        }
        float S = repro_sum::finalize(acc, 1, 1u, nf);
        int t = target[r], kind = ce_row_kind(t, ignore_index, cols);
        loss[r] = ce_row_loss(S, kind > 0 ? p[t] : 0.0f, m, kind);
        if (grad)
            for (int i = 0; i < cols; ++i) grad[(size_t)r * cols + i] = ce_grad(e[i], S, i == t, kind, count, seed);
        delete[] e;
    });
    int64_t acc[repro_sum::LIMBS] = {};
    int nf = 0;
    for (int r = 0; r < rows; ++r) repro_sum::acc_add(acc, 1, loss[r], nf);
    delete[] loss;
    return count == 0 ? canonical_nan() : repro_sum::finalize(acc, 1, (uint32_t)count, nf);
}

} // namespace repro_math
