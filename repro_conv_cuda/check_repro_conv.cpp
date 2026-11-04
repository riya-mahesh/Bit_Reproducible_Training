// CPU checks of the conv definition (no GPU needed):
//   1. forward and weight gradient equal repro_gemm's definition on the im2col matrix, bit for bit;
//   2. forward is close to the true convolution;
//   3. both gradients are correct: for random x, W, dY the adjoint identities hold,
//      <dY, conv(x, W)> = <dX, x> = <dW, W>, to within quantization error.
// Build: c++ -O2 -std=c++17 -pthread -ffp-contract=off -I. -I../repro_gemm_cuda check_repro_conv.cpp
#include "repro_conv_cpu.h"
#include <cstdio>
#include <cstring>
#include <random>

using namespace repro_conv;

static int failures = 0;
static void expect(bool ok, const char* what, const char* shape) {
    std::printf("  %-4s %-44s %s\n", ok ? "ok" : "FAIL", what, shape);
    if (!ok) ++failures;
}

static std::vector<float> normal(size_t n, float scale, std::mt19937_64& g) {
    std::normal_distribution<float> d(0.0f, scale);
    std::vector<float> v(n);
    for (auto& x : v) x = d(g);
    return v;
}

// The GEMM definition: C[M, N] = A[M, K] * B[K, N], row scales of A, column scales of B.
static std::vector<float> gemm_definition(const std::vector<float>& A, int M, int K, const std::vector<float>& B, int N, int S) {
    std::vector<int> eA(M), eB(N);
    std::vector<int> X((size_t)M * K), Y((size_t)N * K);
    for (int r = 0; r < M; ++r) {
        float m = 0.0f;
        for (int k = 0; k < K; ++k) m = std::max(m, abs_value(A[(size_t)r * K + k]));
        eA[r] = scale_exponent(m);
        for (int k = 0; k < K; ++k) X[(size_t)r * K + k] = fixed_point(A[(size_t)r * K + k], eA[r], S);
    }
    for (int c = 0; c < N; ++c) {
        float m = 0.0f;
        for (int k = 0; k < K; ++k) m = std::max(m, abs_value(B[(size_t)k * N + c]));
        eB[c] = scale_exponent(m);
        for (int k = 0; k < K; ++k) Y[(size_t)c * K + k] = fixed_point(B[(size_t)k * N + c], eB[c], S);
    }
    std::vector<float> C((size_t)M * N);
    cpu_parallel_for(M, [&](int r) {
        for (int c = 0; c < N; ++c) {
            int64_t t = 0;
            for (int k = 0; k < K; ++k) t += (int64_t)X[(size_t)r * K + k] * Y[(size_t)c * K + k];
            C[(size_t)r * N + c] = round_to_float(t, eA[r] + eB[c] - 14 * (S - 1));
        }
    });
    return C;
}

static bool same(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
}

static double dot(const std::vector<float>& a, const std::vector<float>& b) {
    double s = 0;
    for (size_t i = 0; i < a.size(); ++i) s += (double)a[i] * b[i];
    return s;
}

static void check(const char* name, ConvShape g, int S, std::mt19937_64& rng) {
    int K = g.K(), oh = g.OH(), ow = g.OW(), W = g.windows();
    auto x = normal(g.input_size(), 1.0f, rng);
    auto w = normal(g.weight_size(), 0.1f, rng);
    auto dy = normal(g.output_size(), 1e-3f, rng);

    // im2col: X[window, k]; dY reshaped to [window, f].
    std::vector<float> X((size_t)W * K), Wt((size_t)K * g.F), DY((size_t)g.F * W);
    for (int i = 0; i < W; ++i) {
        int n, oy, ox;
        window_of(g, i, n, oy, ox);
        for (int k = 0; k < K; ++k) {
            int c, ky, kx;
            position_of(g, k, c, ky, kx);
            X[(size_t)i * K + k] = window_value(x.data(), g, n, oy, ox, c, ky, kx);
        }
        for (int f = 0; f < g.F; ++f) DY[(size_t)f * W + i] = dy[(((size_t)n * g.F + f) * oh + oy) * ow + ox];
    }
    for (int f = 0; f < g.F; ++f)
        for (int k = 0; k < K; ++k) Wt[(size_t)k * g.F + f] = w[(size_t)f * K + k];

    std::vector<float> out(g.output_size()), dw(g.weight_size()), dx(g.input_size());
    cpu_conv2d(x.data(), w.data(), out.data(), g, S);
    cpu_conv2d_grad_weight(dy.data(), x.data(), dw.data(), g, S);
    cpu_conv2d_grad_input(dy.data(), w.data(), dx.data(), g, S);

    auto C = gemm_definition(X, W, K, Wt, g.F, S);
    std::vector<float> out_rows((size_t)W * g.F);
    for (int i = 0; i < W; ++i) {
        int n, oy, ox;
        window_of(g, i, n, oy, ox);
        for (int f = 0; f < g.F; ++f) out_rows[(size_t)i * g.F + f] = out[(((size_t)n * g.F + f) * oh + oy) * ow + ox];
    }
    expect(same(out_rows, C), "forward == GEMM on im2col, bit for bit", name);

    // dW = dY^T X: rows are filters (K' = windows), columns are positions.
    auto G = gemm_definition(DY, g.F, W, X, K, S);
    expect(same(dw, G), "weight grad == GEMM dY^T X, bit for bit", name);

    // Forward against the true convolution in double.
    double err = 0, norm = 0;
    for (int i = 0; i < W; ++i)
        for (int f = 0; f < g.F; ++f) {
            double t = 0;
            for (int k = 0; k < K; ++k) t += (double)X[(size_t)i * K + k] * Wt[(size_t)k * g.F + f];
            double d = t - out_rows[(size_t)i * g.F + f];
            err += d * d;
            norm += t * t;
        }
    char msg[96];
    double rel = std::sqrt(err / norm);
    std::snprintf(msg, sizeof msg, "forward relative error %.2e", rel);
    expect(rel < (S == 3 ? 1e-4 : 1.0), msg, name);

    // Adjoint identities: both gradients really are the gradients of the forward map.
    double a = dot(dy, out), b = dot(dx, x), c = dot(dw, w);
    double scale = std::sqrt(dot(dy, dy) * dot(out, out));
    std::snprintf(msg, sizeof msg, "adjoint: |<dY,Y>-<dX,x>| / |dY||Y| = %.2e", std::fabs(a - b) / scale);
    expect(std::fabs(a - b) / scale < (S == 3 ? 1e-4 : 1.0), msg, name);
    std::snprintf(msg, sizeof msg, "adjoint: |<dY,Y>-<dW,W>| / |dY||Y| = %.2e", std::fabs(a - c) / scale);
    expect(std::fabs(a - c) / scale < (S == 3 ? 1e-4 : 1.0), msg, name);
}

static ConvShape shape(int N, int C, int H, int W, int F, int KH, int KW, int SY, int SX, int PY, int PX) {
    ConvShape g;
    g.N = N; g.C = C; g.H = H; g.W = W; g.F = F; g.KH = KH; g.KW = KW; g.SY = SY; g.SX = SX; g.PY = PY; g.PX = PX;
    return g;
}

int main() {
    std::mt19937_64 rng(2024);
    check("LeNet conv1, S=3", shape(100, 1, 28, 28, 6, 5, 5, 1, 1, 2, 2), 3, rng);
    check("LeNet conv2, S=3", shape(100, 6, 14, 14, 16, 5, 5, 1, 1, 0, 0), 3, rng);
    check("LeNet conv1, S=1", shape(10, 1, 28, 28, 6, 5, 5, 1, 1, 2, 2), 1, rng);
    check("LeNet conv1, S=2", shape(10, 1, 28, 28, 6, 5, 5, 1, 1, 2, 2), 2, rng);
    check("stride 2, padding 1, non-square", shape(4, 3, 17, 23, 8, 3, 5, 2, 2, 1, 1), 3, rng);
    check("stride (3,1)", shape(2, 2, 19, 11, 5, 4, 3, 3, 1, 0, 2), 3, rng);
    check("1x1 filter", shape(3, 4, 7, 7, 2, 1, 1, 1, 1, 0, 0), 3, rng);
    std::printf("%s\n", failures ? "OVERALL: FAIL" : "OVERALL: PASS");
    return failures ? 1 : 0;
}
