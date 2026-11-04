// Benchmark: repro_sum vs an ordinary float reduction using the same layout
// strategy: warp per row for row sums, thread per column for column sums, and
// per-block tree + atomicAdd (the usual GPU pattern) otherwise. Reports median time, effective bandwidth, whether each
// method gives the same bits on every run, and how far the float result is from
// the exactly rounded one.
// Usage: bench_repro_sum [iters]
#include "repro_sum.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <vector>

static void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(1); }
}

// Baseline: each block reduces a slice of output b's elements in float, then atomicAdd.
__global__ void float_sum_kernel(const float* x, float* out, int A, int B, int C, int chunks) {
    __shared__ float sh[256];
    int t = threadIdx.x, b = blockIdx.x, chunk = blockIdx.y;
    int count = A * C;
    int i0 = (int)(((long long)count * chunk) / chunks), i1 = (int)(((long long)count * (chunk + 1)) / chunks);
    float s = 0.0f;
    for (int i = i0 + t; i < i1; i += 256) { int a = i / C, c = i - a * C; s += x[((size_t)a * B + b) * C + c]; }
    sh[t] = s;
    __syncthreads();
    for (int k = 128; k > 0; k >>= 1) { if (t < k) sh[t] += sh[t + k]; __syncthreads(); }
    if (t == 0) atomicAdd(&out[b], sh[0]);
}

// Baseline for rows: one warp per row, shuffle tree (deterministic on one GPU).
__global__ void float_rows_kernel(const float* x, float* out, int rows, int C) {
    int lane = threadIdx.x & 31, row = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (row >= rows) return;
    const float* p = x + (size_t)row * C;
    float s = 0.0f;
    for (int c = lane; c < C; c += 32) s += p[c];
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xFFFFFFFFu, s, o);
    if (lane == 0) out[row] = s;
}

// Baseline for columns: one thread per column, sequential over rows.
__global__ void float_columns_kernel(const float* x, float* out, int A, int B) {
    int b = blockIdx.x * 256 + threadIdx.x;
    if (b >= B) return;
    float s = 0.0f;
    for (int a = 0; a < A; ++a) s += x[(size_t)a * B + b];
    out[b] = s;
}

int main(int argc, char** argv) {
    int iters = argc > 1 ? std::atoi(argv[1]) : 50;
    struct Shape { const char* name; int A, B, C; };
    const Shape shapes[] = {
        {"vector 16M", 1, 1, 1 << 24},
        {"conv bias grad 256x64x56x56", 256, 64, 3136},
        {"linear bias grad 8192x4096", 8192, 4096, 1},
        {"row sums 8192x4096", 1, 8192, 4096},
        {"LayerNorm-like rows 16384x1024", 1, 16384, 1024},
        {"softmax-like rows 65536x128", 1, 65536, 128},
        {"LeNet conv1 bias 100x6x28x28", 100, 6, 784},
        {"LeNet softmax rows 100x10", 1, 100, 10},
    };
    cudaDeviceProp prop;
    ck(cudaGetDeviceProperties(&prop, 0), "props");
    std::printf("device: %s\n", prop.name);
    std::printf("%-30s %-8s %10s %9s %14s %s\n", "shape", "method", "median_ms", "GB/s", "distinct_bits", "max |float-exact| (ulps)");
    std::mt19937_64 g(1);
    std::normal_distribution<float> nd(0.f, 1.f);
    cudaEvent_t e0, e1;
    ck(cudaEventCreate(&e0), "ev"); ck(cudaEventCreate(&e1), "ev");
    for (const Shape& s : shapes) {
        size_t n = (size_t)s.A * s.B * s.C;
        std::vector<float> hx(n);
        for (auto& v : hx) v = nd(g);
        float *dx, *dout;
        ck(cudaMalloc(&dx, n * 4), "malloc"); ck(cudaMalloc(&dout, (size_t)s.B * 4), "malloc");
        ck(cudaMemcpy(dx, hx.data(), n * 4, cudaMemcpyHostToDevice), "copy");
        repro_sum::SumConfig cfg;
        repro_sum::SumPlan plan;
        ck(repro_sum::make_sum_plan(s.A, s.B, s.C, cfg, &plan), "plan");
        void* ws;
        ck(cudaMalloc(&ws, plan.workspace_bytes), "ws");
        int fchunks = std::max(1, std::min(4 * prop.multiProcessorCount / s.B + 1, s.A * s.C / 4096 + 1));
        std::vector<float> exact(s.B), got(s.B);

        for (int method = 0; method < 2; ++method) {
            std::vector<float> times;
            std::set<std::vector<uint32_t>> distinct;
            double max_ulps = 0;
            for (int it = 0; it < iters + 3; ++it) {
                ck(cudaEventRecord(e0), "rec");
                if (method == 0) {
                    ck(repro_sum::repro_sum_fp32_ws(dx, dout, s.A, s.B, s.C, cfg, ws, plan.workspace_bytes), "sum");
                } else {
                    if (s.C == 1) {
                        float_columns_kernel<<<(s.B + 255) / 256, 256>>>(dx, dout, s.A, s.B);
                    } else if (s.A == 1 && s.C <= 8192 && s.B >= 256) {
                        float_rows_kernel<<<(s.B + 7) / 8, 256>>>(dx, dout, s.B, s.C);
                    } else {
                        ck(cudaMemsetAsync(dout, 0, (size_t)s.B * 4), "zero");
                        float_sum_kernel<<<dim3(s.B, fchunks), 256>>>(dx, dout, s.A, s.B, s.C, fchunks);
                    }
                }
                ck(cudaEventRecord(e1), "rec");
                ck(cudaEventSynchronize(e1), "sync");
                float ms;
                ck(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
                if (it >= 3) times.push_back(ms);
                ck(cudaMemcpy(got.data(), dout, (size_t)s.B * 4, cudaMemcpyDeviceToHost), "copy back");
                if (method == 0 && it == 0) exact = got;
                std::vector<uint32_t> bits(s.B);
                std::memcpy(bits.data(), got.data(), (size_t)s.B * 4);
                distinct.insert(bits);
                for (int b = 0; b < s.B; ++b) {
                    float ulp = std::nextafter(std::fabs(exact[b]), INFINITY) - std::fabs(exact[b]);
                    max_ulps = std::max(max_ulps, (double)std::fabs(got[b] - exact[b]) / ulp);
                }
            }
            std::sort(times.begin(), times.end());
            float med = times[times.size() / 2];
            std::printf("%-30s %-8s %10.3f %9.1f %14zu %.1f\n", s.name, method == 0 ? "repro" : "float",
                        med, n * 4 / (med * 1e6), distinct.size(), max_ulps);
        }
        cudaFree(dx); cudaFree(dout); cudaFree(ws);
    }
    return 0;
}
