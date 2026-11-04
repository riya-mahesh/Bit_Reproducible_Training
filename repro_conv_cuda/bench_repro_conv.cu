// Time per batch at LeNet-5 shapes: repro_conv, repro_gemm on an im2col matrix (kernels only,
// without counting the copy), and an ordinary float direct convolution (not reproducible).
// Usage: bench_repro_conv [iters]
#include "repro_conv.cuh"
#include "repro_gemm.cuh"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

using namespace repro_conv;

static void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(1); }
}

__global__ void float_conv(const float* x, const float* w, float* out, ConvShape g) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= g.windows()) return;
    int n, oy, ox, K = g.K(), oh = g.OH(), ow = g.OW();
    window_of(g, i, n, oy, ox);
    for (int f = 0; f < g.F; ++f) {
        float t = 0.0f;
        int k = 0;
        for (int c = 0; c < g.C; ++c)
            for (int ky = 0; ky < g.KH; ++ky)
                for (int kx = 0; kx < g.KW; ++kx, ++k) t += window_value(x, g, n, oy, ox, c, ky, kx) * w[(size_t)f * K + k];
        out[(((size_t)n * g.F + f) * oh + oy) * ow + ox] = t;
    }
}

__global__ void im2col_kernel(const float* x, float* X, ConvShape g) {
    size_t i = (size_t)blockIdx.x * 256 + threadIdx.x;
    int K = g.K();
    if (i >= (size_t)g.windows() * K) return;
    int w = (int)(i / K), k = (int)(i % K), n, oy, ox, c, ky, kx;
    window_of(g, w, n, oy, ox);
    position_of(g, k, c, ky, kx);
    X[i] = window_value(x, g, n, oy, ox, c, ky, kx);
}

static double median_ms(int iters, const std::function<void()>& f) {
    std::vector<double> t;
    f();
    ck(cudaDeviceSynchronize(), "sync");
    for (int i = 0; i < iters; ++i) {
        auto a = std::chrono::steady_clock::now();
        f();
        ck(cudaDeviceSynchronize(), "sync");
        t.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count());
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

int main(int argc, char** argv) {
    int iters = argc > 1 ? std::atoi(argv[1]) : 30;
    cudaDeviceProp prop;
    ck(cudaGetDeviceProperties(&prop, 0), "device");
    std::printf("device: %s (sm_%d%d), S=3, milliseconds per batch of 100\n", prop.name, prop.major, prop.minor);
    ConvShape shapes[2];
    shapes[0].N = 100; shapes[0].C = 1; shapes[0].H = 28; shapes[0].W = 28; shapes[0].F = 6; shapes[0].KH = 5; shapes[0].KW = 5; shapes[0].PY = 2; shapes[0].PX = 2;
    shapes[1].N = 100; shapes[1].C = 6; shapes[1].H = 14; shapes[1].W = 14; shapes[1].F = 16; shapes[1].KH = 5; shapes[1].KW = 5;
    const char* names[2] = {"conv1", "conv2"};
    std::mt19937_64 rng(5);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (int s = 0; s < 2; ++s) {
        ConvShape g = shapes[s];
        const int S = 3;
        std::vector<float> x(g.input_size()), w(g.weight_size()), dy(g.output_size());
        for (auto& v : x) v = nd(rng);
        for (auto& v : w) v = 0.1f * nd(rng);
        for (auto& v : dy) v = 1e-3f * nd(rng);
        float *dx, *dw, *ddy, *out, *gw, *gi, *X, *Yg;
        ck(cudaMalloc(&dx, x.size() * 4), "m"); ck(cudaMalloc(&dw, w.size() * 4), "m"); ck(cudaMalloc(&ddy, dy.size() * 4), "m");
        ck(cudaMalloc(&out, dy.size() * 4), "m"); ck(cudaMalloc(&gw, w.size() * 4), "m"); ck(cudaMalloc(&gi, x.size() * 4), "m");
        size_t ncol = (size_t)g.windows() * g.K();
        ck(cudaMalloc(&X, ncol * 4), "m"); ck(cudaMalloc(&Yg, dy.size() * 4), "m");
        ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "c");
        ck(cudaMemcpy(dw, w.data(), w.size() * 4, cudaMemcpyHostToDevice), "c");
        ck(cudaMemcpy(ddy, dy.data(), dy.size() * 4, cudaMemcpyHostToDevice), "c");
        size_t b0, b1, b2;
        ck(conv2d_workspace(g, S, &b0), "p"); ck(conv2d_grad_weight_workspace(g, S, &b1), "p"); ck(conv2d_grad_input_workspace(g, S, &b2), "p");
        void *w0, *w1, *w2;
        ck(cudaMalloc(&w0, b0), "m"); ck(cudaMalloc(&w1, b1), "m"); ck(cudaMalloc(&w2, b2), "m");
        repro_gemm::Config cfg;
        cfg.trans_b = true;
        repro_gemm::Plan plan;
        ck(repro_gemm::make_plan(g.windows(), g.F, g.K(), cfg, &plan), "gemm plan");
        void* wg;
        ck(cudaMalloc(&wg, plan.workspace_bytes), "m");
        unsigned bl_col = (unsigned)((ncol + 255) / 256), bl_f = (unsigned)((g.windows() + 255) / 256);

        std::printf("%s [%d,%d,%d,%d] -> %d filters, %d x %d:\n", names[s], g.N, g.C, g.H, g.W, g.F, g.KH, g.KW);
        std::printf("  %-34s %9.3f\n", "float direct, forward", median_ms(iters, [&] { float_conv<<<bl_f, 256>>>(dx, dw, out, g); }));
        std::printf("  %-34s %9.3f\n", "repro_conv forward", median_ms(iters, [&] { ck(repro_conv2d_ws(dx, dw, out, g, S, w0, b0), "fwd"); }));
        std::printf("  %-34s %9.3f\n", "im2col copy only", median_ms(iters, [&] { im2col_kernel<<<bl_col, 256>>>(dx, X, g); }));
        std::printf("  %-34s %9.3f\n", "repro_gemm on im2col (no copy)", median_ms(iters, [&] {
            ck(repro_gemm::repro_gemm_fp32_ws(X, dw, Yg, g.windows(), g.F, g.K(), cfg, wg, plan.workspace_bytes), "gemm"); }));
        std::printf("  %-34s %9.3f\n", "repro_conv weight grad", median_ms(iters, [&] { ck(repro_conv2d_grad_weight_ws(ddy, dx, gw, g, S, w1, b1), "gw"); }));
        std::printf("  %-34s %9.3f\n", "repro_conv input grad", median_ms(iters, [&] { ck(repro_conv2d_grad_input_ws(ddy, dw, gi, g, S, w2, b2), "gi"); }));
        std::fflush(stdout);
    }
    return 0;
}
