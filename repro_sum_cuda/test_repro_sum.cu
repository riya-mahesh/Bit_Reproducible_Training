// Correctness suite for repro_sum. For each shape and input distribution:
//   - GPU output equals the CPU implementation bit for bit (sum and mean);
//   - output is identical for many chunk counts (different work splits);
//   - output is identical after shuffling each output's elements (different order);
// plus special cases (cancellation, overflow, NaN, subnormals, zeros).
// Shapes include every reduction in LeNet-5 MNIST training (batch 100).
#include "repro_sum.cuh"
#include "repro_sum_math.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace repro_sum;

static void ck(cudaError_t e, const char* where) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", where, cudaGetErrorString(e)); std::exit(1); }
}

// CPU reference: same exact accumulator, natural order.
static std::vector<float> cpu_sum(const std::vector<float>& x, int A, int B, int C, bool mean) {
    std::vector<float> out(B);
    for (int b = 0; b < B; ++b) {
        int64_t acc[LIMBS] = {};
        int nf = 0;
        for (int a = 0; a < A; ++a)
            for (int c = 0; c < C; ++c) acc_add(acc, 1, x[((size_t)a * B + b) * C + c], nf);
        out[b] = finalize(acc, 1, mean ? (uint32_t)(A * C) : 1u, nf);
    }
    return out;
}

static std::vector<float> gpu_sum(const std::vector<float>& x, int A, int B, int C, SumConfig cfg) {
    float *dx, *dout;
    ck(cudaMalloc(&dx, x.size() * 4), "malloc");
    ck(cudaMalloc(&dout, (size_t)B * 4), "malloc");
    ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "copy");
    ck(cudaMemset(dout, 0xFF, (size_t)B * 4), "poison");
    ck(repro_sum_fp32(dx, dout, A, B, C, cfg), "repro_sum");
    std::vector<float> out(B);
    ck(cudaMemcpy(out.data(), dout, (size_t)B * 4, cudaMemcpyDeviceToHost), "copy back");
    cudaFree(dx); cudaFree(dout);
    return out;
}

static bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
}

static int failures = 0;
static void expect(bool ok, const std::string& what) {
    if (!ok) { std::printf("  FAIL %s\n", what.c_str()); ++failures; }
}

static std::vector<float> make(const std::string& dist, size_t n, std::mt19937_64& g) {
    std::normal_distribution<double> nd(0.0, 1.0);
    std::uniform_int_distribution<int> ex(-150, 120);
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) {
        if (dist == "normal") v[i] = (float)nd(g);
        else if (dist == "wide") v[i] = (float)std::ldexp(nd(g), ex(g));        // all exponents, incl. subnormals
        // Huge values that cancel in pairs plus small ones: float summation loses the small ones.
        else if (dist == "cancel") v[i] = i % 3 == 2 ? (float)nd(g) : (float)((i % 3 ? -1e30 : 1e30) * (1 + (i / 3) % 7));
        else v[i] = (float)(nd(g) * 1e-3);
    }
    return v;
}

// Shuffle the A*C elements of every output b among themselves.
static std::vector<float> shuffle_per_output(const std::vector<float>& x, int A, int B, int C, std::mt19937_64& g) {
    std::vector<float> y = x;
    std::vector<float> tmp((size_t)A * C);
    for (int b = 0; b < B; ++b) {
        for (int a = 0; a < A; ++a) for (int c = 0; c < C; ++c) tmp[(size_t)a * C + c] = x[((size_t)a * B + b) * C + c];
        std::shuffle(tmp.begin(), tmp.end(), g);
        for (int a = 0; a < A; ++a) for (int c = 0; c < C; ++c) y[((size_t)a * B + b) * C + c] = tmp[(size_t)a * C + c];
    }
    return y;
}

int main() {
    struct Shape { const char* name; int A, B, C; };
    const Shape shapes[] = {
        {"Linear bias grad 100x120", 100, 120, 1},
        {"Linear bias grad 100x84", 100, 84, 1},
        {"Linear bias grad 100x10", 100, 10, 1},
        {"Conv1 bias grad 100x6x28x28", 100, 6, 784},
        {"Conv2 bias grad 100x16x10x10", 100, 16, 100},
        {"Loss over batch of 100", 1, 1, 100},
        {"Softmax row sums 100x10", 1, 100, 10},
        {"Columns 4096x1000", 4096, 1000, 1},
        {"Rows 300x5000", 1, 300, 5000},
        {"LayerNorm-like rows 1024x1024", 1, 1024, 1024},
        {"Short odd rows 2000x13", 1, 2000, 13},
        {"Rows 512x8191", 1, 512, 8191},
        {"Vector of 3,000,001", 1, 1, 3000001},
        {"Odd 7x3x13", 7, 3, 13},
    };
    std::mt19937_64 g(42);
    for (const Shape& s : shapes) {
        for (const char* dist : {"normal", "wide", "cancel"}) {
            int before = failures;
            std::vector<float> x = make(dist, (size_t)s.A * s.B * s.C, g);
            for (bool mean : {false, true}) {
                std::vector<float> want = cpu_sum(x, s.A, s.B, s.C, mean);
                SumConfig cfg;
                cfg.mean = mean;
                for (int chunks : {0, 1, 2, 3, 7, 64, 1000}) {
                    cfg.chunks = chunks;
                    expect(same_bits(gpu_sum(x, s.A, s.B, s.C, cfg), want),
                           std::string("matches CPU, chunks=") + std::to_string(chunks) + (mean ? " mean" : " sum"));
                }
                cfg.chunks = 0;
                std::vector<float> xs = shuffle_per_output(x, s.A, s.B, s.C, g);
                expect(same_bits(gpu_sum(xs, s.A, s.B, s.C, cfg), want), std::string("order invariance") + (mean ? " mean" : " sum"));
            }
            std::printf("%-30s %-7s %s\n", s.name, dist, failures == before ? "PASS" : "FAIL");
        }
    }

    // Special cases with known exact answers.
    auto one = [&](std::vector<float> v, bool mean) {
        SumConfig cfg;
        cfg.mean = mean;
        return gpu_sum(v, 1, 1, (int)v.size(), cfg)[0];
    };
    auto bits = [](float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; };
    int before = failures;
    expect(one({1e30f, 1.0f, -1e30f}, false) == 1.0f, "1e30 + 1 - 1e30 == 1 (float summation gives 0)");
    expect(std::isinf(one({3e38f, 3e38f}, false)), "overflow gives inf");
    expect(std::isnan(one({1.0f, NAN, 2.0f}, false)), "NaN input gives NaN");
    expect(std::isnan(one({INFINITY, 1.0f}, false)), "Inf input gives NaN");
    expect(bits(one({0.0f, -0.0f}, false)) == 0u, "exact zero is +0");
    expect(one({1e-45f, 1e-45f, 1e-45f}, false) == 3 * 1e-45f, "subnormals add exactly");
    expect(one({1.0f, 0x1p-24f}, false) == 1.0f, "exact tie rounds to even");
    expect(one({1.0f, 0x1p-24f, 0x1p-149f}, false) == 1.0f + 0x1p-23f, "just above tie rounds up");
    expect(one({1.0f, 2.0f, 4.0f}, true) == 7.0f / 3.0f, "mean 7/3 correctly rounded");
    std::printf("%-30s %-7s %s\n", "Special cases", "", failures == before ? "PASS" : "FAIL");

    if (failures) { std::printf("FAILED: %d checks\n", failures); return 2; }
    std::puts("PASS: all shapes, distributions, chunk counts, orders, and special cases");
    return 0;
}
