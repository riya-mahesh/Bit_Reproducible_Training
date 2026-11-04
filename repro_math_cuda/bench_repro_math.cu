// Throughput of exp, log, and softmax: CUDA's FP32 functions (fast, not reproducible),
// RepDL's (float)f((double)x), and ours. Prints median time per element.
// Usage: bench_repro_math [iters]
#include "repro_softmax.cuh"
#include "repro_math.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

static void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(1); }
}

enum { CUDA_EXP, REPDL_EXP, CUDA_LOG, REPDL_LOG };

template<int F>
__global__ void __launch_bounds__(256) elementwise(const float* x, float* out, size_t n) {
    size_t i = (size_t)blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    out[i] = F == CUDA_EXP ? expf(v) : F == REPDL_EXP ? (float)exp((double)v)
           : F == CUDA_LOG ? logf(v) : (float)log((double)v);
}

// Softmax with ordinary float arithmetic, one warp per row, in the same stages as ours.
// DOUBLE selects RepDL's exp; the sum is plain float either way, so this measures the
// exp and arithmetic cost and is not a RepDL sum.
template<bool DOUBLE>
__global__ void __launch_bounds__(256) float_softmax(const float* x, float* out, int rows, int cols) {
    int lane = threadIdx.x & 31, row = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (row >= rows) return;
    const float* p = x + (size_t)row * cols;
    float* o = out + (size_t)row * cols;
    float m = -INFINITY;
    for (int c = lane; c < cols; c += 32) m = fmaxf(m, p[c]);
    for (int k = 16; k > 0; k >>= 1) m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, k));
    float s = 0.0f;
    for (int c = lane; c < cols; c += 32) {
        float v = p[c] - m;
        float e = DOUBLE ? (float)exp((double)v) : expf(v);
        o[c] = e;
        s += e;
    }
    for (int k = 16; k > 0; k >>= 1) s += __shfl_xor_sync(0xFFFFFFFFu, s, k);
    for (int c = lane; c < cols; c += 32) o[c] /= s;
}

static double median_ms(int iters, const std::function<void()>& run) {
    std::vector<double> t;
    run();  // warm-up
    ck(cudaDeviceSynchronize(), "sync");
    for (int i = 0; i < iters; ++i) {
        auto a = std::chrono::steady_clock::now();
        run();
        ck(cudaDeviceSynchronize(), "sync");
        t.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count());
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

static void row(const char* name, double ms, size_t n, double base_ms) {
    std::printf("  %-34s %10.3f ms  %8.3f ns/element  x%.2f\n", name, ms, ms * 1e6 / (double)n, ms / base_ms);
    std::fflush(stdout);
}

int main(int argc, char** argv) {
    int iters = argc > 1 ? std::atoi(argv[1]) : 20;
    cudaDeviceProp prop;
    ck(cudaGetDeviceProperties(&prop, 0), "device");
    std::printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    const size_t n = (size_t)1 << 24;
    std::mt19937_64 g(7);
    std::vector<float> xe(n), xl(n);
    std::normal_distribution<float> ne(-5.0f, 5.0f);
    std::uniform_real_distribution<float> ul(1.0f, 10.0f);
    for (size_t i = 0; i < n; ++i) { xe[i] = ne(g); xl[i] = ul(g); }

    float *de, *dl, *dout;
    ck(cudaMalloc(&de, n * 4), "malloc");
    ck(cudaMalloc(&dl, n * 4), "malloc");
    ck(cudaMalloc(&dout, n * 4), "malloc");
    ck(cudaMemcpy(de, xe.data(), n * 4, cudaMemcpyHostToDevice), "copy");
    ck(cudaMemcpy(dl, xl.data(), n * 4, cudaMemcpyHostToDevice), "copy");
    unsigned blocks = (unsigned)((n + 255) / 256);

    std::printf("exp, 2^24 elements, normal(-5, 5):\n");
    double b = median_ms(iters, [&] { elementwise<CUDA_EXP><<<blocks, 256>>>(de, dout, n); });
    row("CUDA expf (not reproducible)", b, n, b);
    row("RepDL (float)exp((double)x)", median_ms(iters, [&] { elementwise<REPDL_EXP><<<blocks, 256>>>(de, dout, n); }), n, b);
    row("repro_expf", median_ms(iters, [&] { ck(repro_math::repro_exp_fp32(de, dout, n), "exp"); }), n, b);

    std::printf("log, 2^24 elements, uniform(1, 10):\n");
    b = median_ms(iters, [&] { elementwise<CUDA_LOG><<<blocks, 256>>>(dl, dout, n); });
    row("CUDA logf (not reproducible)", b, n, b);
    row("RepDL (float)log((double)x)", median_ms(iters, [&] { elementwise<REPDL_LOG><<<blocks, 256>>>(dl, dout, n); }), n, b);
    row("repro_logf", median_ms(iters, [&] { ck(repro_math::repro_log_fp32(dl, dout, n), "log"); }), n, b);

    const struct { const char* name; int rows, cols; } shapes[] = {
        {"softmax MNIST 16384x10 (x164 batches)", 16384, 10},
        {"softmax attention 12288x1024", 12288, 1024},
        {"softmax vocab 300x50257", 300, 50257},
    };
    for (auto& s : shapes) {
        size_t total = (size_t)s.rows * s.cols;
        std::printf("%s:\n", s.name);
        unsigned bl = (unsigned)((s.rows + 7) / 8);
        double base = median_ms(iters, [&] { float_softmax<false><<<bl, 256>>>(de, dout, s.rows, s.cols); });
        row("float: CUDA expf + float sum", base, total, base);
        row("float: double exp + float sum", median_ms(iters, [&] { float_softmax<true><<<bl, 256>>>(de, dout, s.rows, s.cols); }), total, base);
        repro_math::SoftmaxPlan plan;
        ck(repro_math::make_softmax_plan(s.rows, s.cols, &plan), "plan");
        void* ws;
        ck(cudaMalloc(&ws, plan.workspace_bytes), "ws");
        row("repro_softmax", median_ms(iters, [&] {
            ck(repro_math::repro_softmax_fp32_ws(de, dout, s.rows, s.cols, ws, plan.workspace_bytes), "softmax");
        }), total, base);
        cudaFree(ws);

        std::vector<int> tg(s.rows);
        for (auto& t : tg) t = (int)(g() % (uint64_t)s.cols);
        int* dt;
        float* dloss;
        ck(cudaMalloc(&dt, tg.size() * 4), "malloc");
        ck(cudaMalloc(&dloss, 4), "malloc");
        ck(cudaMemcpy(dt, tg.data(), tg.size() * 4, cudaMemcpyHostToDevice), "copy");
        repro_math::CrossEntropyPlan cep;
        ck(repro_math::make_cross_entropy_plan(s.rows, s.cols, &cep), "plan");
        void* cws;
        ck(cudaMalloc(&cws, cep.workspace_bytes), "ws");
        row("repro_cross_entropy, forward", median_ms(iters, [&] {
            ck(repro_math::repro_cross_entropy_fp32_ws(de, dt, -100, dloss, s.rows, s.cols, cws, cep.workspace_bytes), "ce");
        }), total, base);
        row("repro_cross_entropy, forward + grad", median_ms(iters, [&] {
            ck(repro_math::repro_cross_entropy_fp32_ws(de, dt, -100, dloss, s.rows, s.cols, cws, cep.workspace_bytes), "ce");
            ck(repro_math::repro_cross_entropy_grad_fp32_ws(dt, -100, 1.0f, dout, s.rows, s.cols, cws, cep.workspace_bytes), "grad");
        }), total, base);
        cudaFree(cws); cudaFree(dt); cudaFree(dloss);
    }
    return 0;
}
