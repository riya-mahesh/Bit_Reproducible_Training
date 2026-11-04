#pragma once
// CPU version of repro_conv (no CUDA needed), multi-threaded, using the same scalar code as
// the GPU kernels. The output does not depend on the number of threads.
#include "repro_conv_math.cuh"
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <thread>
#include <vector>

namespace repro_conv {

template<class F>
inline void cpu_parallel_for(int n, F f) {
    const char* env = std::getenv("REPRO_CPU_THREADS");   // the result doesn't depend on this
    unsigned nt = env ? (unsigned)std::atoi(env) : std::max(1u, std::thread::hardware_concurrency());
    std::atomic<int> next{0};
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < nt; ++t)
        pool.emplace_back([&] { for (int i; (i = next.fetch_add(256)) < n;) for (int j = i; j < std::min(i + 256, n); ++j) f(j); });
    for (auto& th : pool) th.join();
}

inline void cpu_conv2d(const float* x, const float* w, float* out, const ConvShape& g, int S) {
    int K = g.K();
    std::vector<int> Wq((size_t)g.F * K), eF(g.F);
    for (int f = 0; f < g.F; ++f) {
        float m = 0.0f;
        for (int k = 0; k < K; ++k) m = std::max(m, abs_value(w[(size_t)f * K + k]));
        eF[f] = scale_exponent(m);
        for (int k = 0; k < K; ++k) Wq[(size_t)f * K + k] = fixed_point(w[(size_t)f * K + k], eF[f], S);
    }
    cpu_parallel_for(g.windows(), [&](int i) {
        std::vector<int> q(K);
        int n, oy, ox;
        window_of(g, i, n, oy, ox);
        forward_window(x, Wq.data(), eF.data(), out, g, n, oy, ox, S, q.data());
    });
}

// dW[f, k] for dY in [n, f, oy, ox].
inline void cpu_conv2d_grad_weight(const float* dy, const float* x, float* dw, const ConvShape& g, int S) {
    int K = g.K(), oh = g.OH(), ow = g.OW(), e0 = -14 * (S - 1);
    std::vector<float> mF(g.F, 0.0f);
    size_t per_filter = (size_t)oh * ow;
    for (size_t i = 0; i < g.output_size(); ++i) {
        int f = (int)((i / per_filter) % g.F);
        mF[f] = std::max(mF[f], abs_value(dy[i]));
    }
    std::vector<int> eF(g.F);
    for (int f = 0; f < g.F; ++f) eF[f] = scale_exponent(mF[f]);
    std::vector<int> D(g.output_size());
    for (size_t i = 0; i < D.size(); ++i) {
        int f = (int)((i / ((size_t)oh * ow)) % g.F);
        D[i] = fixed_point(dy[i], eF[f], S);
    }
    int per = oh * ow;
    cpu_parallel_for(K, [&](int k) {
        int c, ky, kx;
        position_of(g, k, c, ky, kx);
        uint32_t m = 0;
        for (int i = 0; i < per; ++i) m = std::max(m, position_max_bits(x, g, i / ow, i % ow, c, ky, kx));
        int eX = scale_exponent(repro_gemm::bits_to_float(m));
        std::vector<int64_t> t(g.F, 0);
        for (int i = 0; i < per; ++i) weight_grad_windows(x, D.data(), g, i / ow, i % ow, c, ky, kx, eX, S, t.data());
        for (int f = 0; f < g.F; ++f) dw[(size_t)f * K + k] = round_to_float(t[f], eF[f] + eX + e0);
    });
}

// dX[n, c, y, x] for dY in [n, f, oy, ox].
inline void cpu_conv2d_grad_input(const float* dy, const float* w, float* dx, const ConvShape& g, int S) {
    size_t per = (size_t)g.F * g.OH() * g.OW();
    int e0 = -14 * (S - 1);
    std::vector<int> eI(g.N), D(g.output_size()), Wq(g.weight_size());
    for (int n = 0; n < g.N; ++n) {
        float m = 0.0f;
        for (size_t i = 0; i < per; ++i) m = std::max(m, abs_value(dy[(size_t)n * per + i]));
        eI[n] = scale_exponent(m);
        for (size_t i = 0; i < per; ++i) D[(size_t)n * per + i] = fixed_point(dy[(size_t)n * per + i], eI[n], S);
    }
    float mw = 0.0f;
    for (size_t i = 0; i < Wq.size(); ++i) mw = std::max(mw, abs_value(w[i]));
    int eW = scale_exponent(mw);
    for (size_t i = 0; i < Wq.size(); ++i) Wq[i] = fixed_point(w[i], eW, S);
    int pixels_per_image = g.C * g.H * g.W;
    cpu_parallel_for(g.N * pixels_per_image, [&](int i) {
        int n = i / pixels_per_image, r = i % pixels_per_image;
        int c = r / (g.H * g.W), s = r % (g.H * g.W);
        dx[i] = input_grad_pixel(D.data(), Wq.data(), g, n, c, s / g.W, s % g.W, eI[n] + eW + e0);
    });
}

} // namespace repro_conv
