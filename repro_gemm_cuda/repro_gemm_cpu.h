#pragma once
// CPU version of repro_gemm (no CUDA needed), multi-threaded, using the same scalar code as the
// GPU kernels: C[M, N] = op(A)[M, K] * op(B)[K, N]. C must hold M * N values.
// ta: A is K x M row-major (use A^T). tb: B is N x K row-major (use B^T).
#include "repro_gemm_math.cuh"
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <thread>
#include <vector>

namespace repro_gemm {

template<class F>
inline void cpu_for(int n, F f) {
    const char* env = std::getenv("REPRO_CPU_THREADS");   // the result doesn't depend on this
    unsigned nt = env ? (unsigned)std::atoi(env) : std::max(1u, std::thread::hardware_concurrency());
    std::atomic<int> next{0};
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < nt; ++t)
        pool.emplace_back([&] { for (int i; (i = next.fetch_add(1)) < n;) f(i); });
    for (auto& th : pool) th.join();
}

inline void cpu_gemm(const float* A, const float* B, float* C, int M, int N, int K, bool ta, bool tb, int S) {
    std::vector<int> eA(M), eB(N);
    std::vector<int32_t> X((size_t)M * K), Y((size_t)N * K);
    auto a_at = [&](int r, int k) { return ta ? A[(size_t)k * M + r] : A[(size_t)r * K + k]; };
    auto b_at = [&](int k, int c) { return tb ? B[(size_t)c * K + k] : B[(size_t)k * N + c]; };
    cpu_for(M, [&](int r) {
        float m = 0.0f;
        for (int k = 0; k < K; ++k) m = std::fmax(m, std::fabs(a_at(r, k)));
        eA[r] = scale_exponent(m);
        for (int k = 0; k < K; ++k) X[(size_t)r * K + k] = fixed_point(a_at(r, k), eA[r], S);
    });
    cpu_for(N, [&](int c) {
        float m = 0.0f;
        for (int k = 0; k < K; ++k) m = std::fmax(m, std::fabs(b_at(k, c)));
        eB[c] = scale_exponent(m);
        for (int k = 0; k < K; ++k) Y[(size_t)c * K + k] = fixed_point(b_at(k, c), eB[c], S);
    });
    cpu_for(M, [&](int r) {
        const int32_t* x = &X[(size_t)r * K];
        for (int c = 0; c < N; ++c) {
            const int32_t* y = &Y[(size_t)c * K];
            int64_t t = 0;
            for (int k = 0; k < K; ++k) t += (int64_t)x[k] * y[k];
            C[(size_t)r * N + c] = round_to_float(t, eA[r] + eB[c] - 14 * (S - 1));
        }
    });
}

} // namespace repro_gemm
