// GPU tests of repro_conv: forward and both gradients equal the CPU version bit for bit,
// repeat runs are identical, and forward and weight gradient equal repro_gemm on im2col.
#include "repro_conv.cuh"
#include "repro_conv_cpu.h"
#include "repro_gemm.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>

using namespace repro_conv;

static void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(1); }
}

static int failures = 0;
static void expect(bool ok, const std::string& what) {
    std::printf("  %-4s %s\n", ok ? "ok" : "FAIL", what.c_str());
    if (!ok) ++failures;
}

static float* upload(const std::vector<float>& v) {
    float* d;
    ck(cudaMalloc(&d, v.size() * 4), "malloc");
    ck(cudaMemcpy(d, v.data(), v.size() * 4, cudaMemcpyHostToDevice), "upload");
    return d;
}

static std::vector<float> download(const float* d, size_t n) {
    std::vector<float> v(n);
    ck(cudaMemcpy(v.data(), d, n * 4, cudaMemcpyDeviceToHost), "download");
    return v;
}

static std::vector<float> normal(size_t n, float scale, std::mt19937_64& g) {
    std::normal_distribution<float> d(0.0f, scale);
    std::vector<float> v(n);
    for (auto& x : v) x = d(g);
    return v;
}

static bool same(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
}

template<class P, class R>
static std::vector<float> run(const ConvShape& g, int S, size_t out_size, P plan, R launch) {
    size_t bytes;
    ck(plan(g, S, &bytes), "plan");
    void* ws;
    float* o;
    ck(cudaMalloc(&ws, bytes), "ws");
    ck(cudaMalloc(&o, out_size * 4), "out");
    ck(launch(o, ws, bytes), "conv");
    ck(cudaDeviceSynchronize(), "sync");
    auto v = download(o, out_size);
    cudaFree(ws); cudaFree(o);
    return v;
}

static void check(const std::string& name, ConvShape g, int S, std::mt19937_64& rng) {
    int K = g.K(), oh = g.OH(), ow = g.OW(), W = g.windows();
    auto x = normal(g.input_size(), 1.0f, rng);
    auto w = normal(g.weight_size(), 0.1f, rng);
    auto dy = normal(g.output_size(), 1e-3f, rng);
    float *dx_ = upload(x), *dw_ = upload(w), *ddy = upload(dy);

    auto fwd = [&] { return run(g, S, g.output_size(), conv2d_workspace, [&](float* o, void* ws, size_t b) { return repro_conv2d_ws(dx_, dw_, o, g, S, ws, b); }); };
    auto gw = [&] { return run(g, S, g.weight_size(), conv2d_grad_weight_workspace, [&](float* o, void* ws, size_t b) { return repro_conv2d_grad_weight_ws(ddy, dx_, o, g, S, ws, b); }); };
    auto gi = [&] { return run(g, S, g.input_size(), conv2d_grad_input_workspace, [&](float* o, void* ws, size_t b) { return repro_conv2d_grad_input_ws(ddy, dw_, o, g, S, ws, b); }); };
    auto f1 = fwd(), w1 = gw(), i1 = gi();

    std::vector<float> cf(g.output_size()), cw(g.weight_size()), ci(g.input_size());
    cpu_conv2d(x.data(), w.data(), cf.data(), g, S);
    cpu_conv2d_grad_weight(dy.data(), x.data(), cw.data(), g, S);
    cpu_conv2d_grad_input(dy.data(), w.data(), ci.data(), g, S);
    expect(same(f1, cf), "forward GPU == CPU: " + name);
    expect(same(w1, cw), "weight grad GPU == CPU: " + name);
    expect(same(i1, ci), "input grad GPU == CPU: " + name);
    expect(same(f1, fwd()) && same(w1, gw()) && same(i1, gi()), "repeat runs identical: " + name);

    // repro_gemm on the im2col matrix must give the same bits.
    std::vector<float> X((size_t)W * K), Yrows((size_t)W * g.F), DY((size_t)W * g.F);
    for (int i = 0; i < W; ++i) {
        int n, oy, ox;
        window_of(g, i, n, oy, ox);
        for (int k = 0; k < K; ++k) {
            int c, ky, kx;
            position_of(g, k, c, ky, kx);
            X[(size_t)i * K + k] = window_value(x.data(), g, n, oy, ox, c, ky, kx);
        }
        for (int f = 0; f < g.F; ++f) {
            size_t at = (((size_t)n * g.F + f) * oh + oy) * ow + ox;
            Yrows[(size_t)i * g.F + f] = f1[at];
            DY[(size_t)i * g.F + f] = dy[at];
        }
    }
    float *dX = upload(X), *dDY = upload(DY), *o1, *o2;
    ck(cudaMalloc(&o1, (size_t)W * g.F * 4), "out");
    ck(cudaMalloc(&o2, g.weight_size() * 4), "out");
    repro_gemm::Config c1;
    c1.slices = S;
    c1.trans_b = true;     // Y = X W^T
    ck(repro_gemm::repro_gemm_fp32(dX, dw_, o1, W, g.F, K, c1), "gemm");
    repro_gemm::Config c2;
    c2.slices = S;
    c2.trans_a = true;     // dW = dY^T X
    ck(repro_gemm::repro_gemm_fp32(dDY, dX, o2, g.F, K, W, c2), "gemm");
    expect(same(download(o1, (size_t)W * g.F), Yrows), "forward == repro_gemm on im2col: " + name);
    expect(same(download(o2, g.weight_size()), w1), "weight grad == repro_gemm dY^T X: " + name);
    cudaFree(dX); cudaFree(dDY); cudaFree(o1); cudaFree(o2);
    cudaFree(dx_); cudaFree(dw_); cudaFree(ddy);
}

static ConvShape shape(int N, int C, int H, int W, int F, int KH, int KW, int SY, int SX, int PY, int PX) {
    ConvShape g;
    g.N = N; g.C = C; g.H = H; g.W = W; g.F = F; g.KH = KH; g.KW = KW; g.SY = SY; g.SX = SX; g.PY = PY; g.PX = PX;
    return g;
}

int main() {
    cudaDeviceProp prop;
    ck(cudaGetDeviceProperties(&prop, 0), "device");
    std::printf("device: %s\n", prop.name);
    std::mt19937_64 rng(2024);
    check("LeNet conv1, S=3", shape(100, 1, 28, 28, 6, 5, 5, 1, 1, 2, 2), 3, rng);
    check("LeNet conv2, S=3", shape(100, 6, 14, 14, 16, 5, 5, 1, 1, 0, 0), 3, rng);
    check("LeNet conv1, S=1", shape(10, 1, 28, 28, 6, 5, 5, 1, 1, 2, 2), 1, rng);
    check("stride 2, padding 1, non-square", shape(4, 3, 17, 23, 8, 3, 5, 2, 2, 1, 1), 3, rng);
    check("stride (3,1)", shape(2, 2, 19, 11, 5, 4, 3, 3, 1, 0, 2), 3, rng);
    check("many filters", shape(2, 4, 12, 12, 100, 3, 3, 1, 1, 1, 1), 3, rng);
    std::printf("%s\n", failures ? "OVERALL: FAIL" : "OVERALL: PASS");
    return failures ? 1 : 0;
}
