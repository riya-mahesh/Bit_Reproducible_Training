// GPU correctness suite for repro_expf, repro_logf, and repro_softmax:
//   - GPU output equals the CPU implementation bit for bit;
//   - softmax is unchanged by reordering a row (the output is reordered the same way);
//   - rows sum to about 1, masked (-inf) entries are exactly 0, in-place works;
//   - hard cases, special values, and subnormal exp results.
#include "repro_softmax.cuh"
#include "repro_softmax_math.cuh"
#include "hard_cases_table.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <vector>

using namespace repro_math;

static void ck(cudaError_t e, const char* where) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", where, cudaGetErrorString(e)); std::exit(1); }
}

static int failures = 0;
static void expect(bool ok, const std::string& what) {
    std::printf("  %-4s %s\n", ok ? "ok" : "FAIL", what.c_str());
    if (!ok) ++failures;
}

static size_t diff_bits(const std::vector<float>& a, const std::vector<float>& b) {
    size_t d = 0;
    for (size_t i = 0; i < a.size(); ++i) d += bits_of(a[i]) != bits_of(b[i]);
    return d;
}

template<class F>
static std::vector<float> on_gpu(const std::vector<float>& x, F run) {
    float *dx, *dout;
    ck(cudaMalloc(&dx, x.size() * 4), "malloc");
    ck(cudaMalloc(&dout, x.size() * 4), "malloc");
    ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "copy");
    run(dx, dout);
    ck(cudaDeviceSynchronize(), "sync");
    std::vector<float> out(x.size());
    ck(cudaMemcpy(out.data(), dout, x.size() * 4, cudaMemcpyDeviceToHost), "copy back");
    cudaFree(dx); cudaFree(dout);
    return out;
}

static std::vector<float> gpu_softmax(const std::vector<float>& x, int rows, int cols) {
    return on_gpu(x, [&](float* d, float* o) { ck(repro_softmax_fp32(d, o, rows, cols), "softmax"); });
}

static std::vector<float> cpu_softmax(const std::vector<float>& x, int rows, int cols) {
    std::vector<float> out(x.size());
    for (int r = 0; r < rows; ++r) softmax_row(&x[(size_t)r * cols], &out[(size_t)r * cols], cols);
    return out;
}

static void check_elementwise(const char* name, const std::vector<float>& x) {
    auto ge = on_gpu(x, [&](float* d, float* o) { ck(repro_exp_fp32(d, o, x.size()), "exp"); });
    auto gl = on_gpu(x, [&](float* d, float* o) { ck(repro_log_fp32(d, o, x.size()), "log"); });
    std::vector<float> ce(x.size()), cl(x.size());
    for (size_t i = 0; i < x.size(); ++i) { ce[i] = repro_expf(x[i]); cl[i] = repro_logf(x[i]); }
    expect(diff_bits(ge, ce) == 0, std::string("exp GPU == CPU: ") + name);
    expect(diff_bits(gl, cl) == 0, std::string("log GPU == CPU: ") + name);
}

static void check_softmax(const std::string& name, const std::vector<float>& x, int rows, int cols, bool finite,
                          std::mt19937_64& g) {
    auto gpu = gpu_softmax(x, rows, cols);
    size_t d = diff_bits(gpu, cpu_softmax(x, rows, cols));
    expect(d == 0, "softmax GPU == CPU: " + name + (d ? " (" + std::to_string(d) + " differ)" : ""));
    expect(diff_bits(gpu, gpu_softmax(x, rows, cols)) == 0, "softmax repeat run: " + name);

    // In place: out == x.
    {
        float* d2;
        ck(cudaMalloc(&d2, x.size() * 4), "malloc");
        ck(cudaMemcpy(d2, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "copy");
        ck(repro_softmax_fp32(d2, d2, rows, cols), "softmax");
        std::vector<float> ip(x.size());
        ck(cudaMemcpy(ip.data(), d2, x.size() * 4, cudaMemcpyDeviceToHost), "copy back");
        cudaFree(d2);
        expect(diff_bits(ip, gpu) == 0, "softmax in place: " + name);
    }

    // Reorder each row: the output must be reordered the same way, bit for bit.
    std::vector<int> perm(cols);
    for (int i = 0; i < cols; ++i) perm[i] = i;
    std::shuffle(perm.begin(), perm.end(), g);
    std::vector<float> xp(x.size());
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) xp[(size_t)r * cols + c] = x[(size_t)r * cols + perm[c]];
    auto gp = gpu_softmax(xp, rows, cols);
    size_t pd = 0;
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            pd += bits_of(gp[(size_t)r * cols + c]) != bits_of(gpu[(size_t)r * cols + perm[c]]);
    expect(pd == 0, "softmax independent of order: " + name);

    if (finite) {
        double worst = 0;
        for (int r = 0; r < rows; ++r) {
            double s = 0;
            for (int c = 0; c < cols; ++c) s += gpu[(size_t)r * cols + c];
            worst = std::max(worst, std::fabs(s - 1.0));
        }
        expect(worst < 1e-3, "rows sum to 1 (worst |sum-1| = " + std::to_string(worst) + "): " + name);
    }
}

static std::vector<float> normal(size_t n, float scale, std::mt19937_64& g) {
    std::normal_distribution<float> d(0.0f, scale);
    std::vector<float> v(n);
    for (auto& x : v) x = d(g);
    return v;
}

int main() {
    cudaDeviceProp prop;
    ck(cudaGetDeviceProperties(&prop, 0), "device");
    std::printf("device: %s\n", prop.name);
    std::mt19937_64 g(12345);

    // Elementwise: random bit patterns, the training range, near-1 log inputs, hard cases.
    std::vector<float> xs(1 << 22);
    for (auto& x : xs) x = float_of((uint32_t)g());
    check_elementwise("random bit patterns", xs);
    check_elementwise("normal, scale 30", normal(1 << 22, 30.0f, g));
    std::vector<float> near1(1 << 20);
    for (size_t i = 0; i < near1.size(); ++i) near1[i] = float_of(0x3F7F0000u + (uint32_t)(i & 0x1FFFFu));
    check_elementwise("near 1", near1);
    std::vector<float> sub(1 << 20);
    for (auto& x : sub) x = -87.0f - 17.5f * (float)(g() >> 40) / 16777216.0f;
    check_elementwise("subnormal exp results", sub);
    std::vector<float> spec = {0.0f, -0.0f, 1.0f, -1.0f, INFINITY, -INFINITY, NAN, float_of(0xFFC12345u),
                               float_of(1u), float_of(0x00800000u), float_of(0x7F7FFFFFu), REPRO_EXP_OVERFLOW,
                               REPRO_EXP_UNDERFLOW, -103.0f, 88.0f};
    check_elementwise("special values", spec);

    std::vector<float> hx;
    for (const HardCase& h : kHardCases) hx.push_back(float_of(h.bits));
    auto he = on_gpu(hx, [&](float* d, float* o) { ck(repro_exp_fp32(d, o, hx.size()), "exp"); });
    auto hl = on_gpu(hx, [&](float* d, float* o) { ck(repro_log_fp32(d, o, hx.size()), "log"); });
    int hard_bad = 0;
    for (size_t i = 0; i < hx.size(); ++i)
        hard_bad += bits_of(kHardCases[i].is_log ? hl[i] : he[i]) != bits_of(kHardCases[i].correct);
    expect(hard_bad == 0, "all " + std::to_string(hx.size()) + " hard cases correctly rounded");

    // Softmax at the shapes training uses.
    check_softmax("MNIST 100x10", normal(100 * 10, 5.0f, g), 100, 10, true, g);
    check_softmax("short rows 2000x3", normal(2000 * 3, 2.0f, g), 2000, 3, true, g);
    check_softmax("one row of 50257", normal(50257, 8.0f, g), 1, 50257, true, g);
    check_softmax("vocab 8x50257 (staged)", normal((size_t)8 * 50257, 8.0f, g), 8, 50257, true, g);
    check_softmax("vocab 256x50257 (fused)", normal((size_t)256 * 50257, 8.0f, g), 256, 50257, true, g);
    check_softmax("large gaps (subnormal exps) 300x257", normal(300 * 257, 60.0f, g), 300, 257, true, g);
    check_softmax("single element 5x1", normal(5, 3.0f, g), 5, 1, true, g);

    // Causal attention: row r may see columns 0..r; the rest are -inf and must give exactly 0.
    {
        const int T = 1024;
        auto x = normal((size_t)T * T, 4.0f, g);
        for (int r = 0; r < T; ++r)
            for (int c = r + 1; c < T; ++c) x[(size_t)r * T + c] = -INFINITY;
        check_softmax("causal attention 1024x1024", x, T, T, true, g);
        auto p = gpu_softmax(x, T, T);
        bool zero = true;
        for (int r = 0; r < T; ++r)
            for (int c = r + 1; c < T; ++c) zero = zero && bits_of(p[(size_t)r * T + c]) == 0u;
        expect(zero, "masked entries are exactly +0");
    }

    // Rows with NaN, +inf, all -inf become NaN rows; a mix of +0 and -0 is an ordinary row.
    {
        const int C = 4;
        std::vector<float> x = {1.0f, NAN, 2.0f, 3.0f,
                                1.0f, INFINITY, 2.0f, 3.0f,
                                -INFINITY, -INFINITY, -INFINITY, -INFINITY,
                                0.0f, -0.0f, 0.0f, -0.0f,
                                1.0f, -INFINITY, 1.0f, -INFINITY};
        int rows = (int)x.size() / C;
        auto p = gpu_softmax(x, rows, C);
        expect(diff_bits(p, cpu_softmax(x, rows, C)) == 0, "softmax special rows GPU == CPU");
        bool nan_rows = true;
        for (int i = 0; i < 12; ++i) nan_rows = nan_rows && bits_of(p[i]) == 0x7FC00000u;
        expect(nan_rows, "NaN, +inf, all-(-inf) rows give canonical NaN");
        expect(p[12] == 0.25f && p[15] == 0.25f && p[16] == 0.5f && p[17] == 0.0f, "ties and masks give exact values");
    }

    // Cross-entropy at the shapes training uses, including ignored rows and row order.
    const struct { const char* name; int rows, cols; float scale; int ignored; } ce[] = {
        {"MNIST 100x10", 100, 10, 5.0f, 0},
        {"GPT-2 8x50257, 3 rows ignored (staged)", 8, 50257, 8.0f, 3},
        {"GPT-2 256x50257, 3 rows ignored (fused)", 256, 50257, 8.0f, 3},
        {"short rows 2000x3, 500 rows ignored", 2000, 3, 2.0f, 500},
        {"large gaps 300x257", 300, 257, 60.0f, 0},
    };
    for (auto& c : ce) {
        auto x = normal((size_t)c.rows * c.cols, c.scale, g);
        std::vector<int> tg(c.rows);
        for (auto& t : tg) t = (int)(g() % (uint64_t)c.cols);
        for (int i = 0; i < c.ignored; ++i) tg[i] = -100;
        auto serial = [](int n, const std::function<void(int)>& f) { for (int i = 0; i < n; ++i) f(i); };
        std::vector<float> cg((size_t)c.rows * c.cols);
        float cl = cross_entropy_cpu(x.data(), tg.data(), -100, c.rows, c.cols, cg.data(), 1.0f, serial);

        auto gpu_ce = [&](const std::vector<float>& xx, const std::vector<int>& tt, float& loss, std::vector<float>& grad) {
            float *dx, *dl, *dg;
            int* dt;
            CrossEntropyPlan plan;
            void* ws;
            ck(make_cross_entropy_plan(c.rows, c.cols, &plan), "plan");
            ck(cudaMalloc(&ws, plan.workspace_bytes), "ws");
            ck(cudaMalloc(&dx, xx.size() * 4), "m");
            ck(cudaMalloc(&dt, tt.size() * 4), "m");
            ck(cudaMalloc(&dl, 4), "m");
            ck(cudaMalloc(&dg, xx.size() * 4), "m");
            ck(cudaMemcpy(dx, xx.data(), xx.size() * 4, cudaMemcpyHostToDevice), "c");
            ck(cudaMemcpy(dt, tt.data(), tt.size() * 4, cudaMemcpyHostToDevice), "c");
            ck(repro_cross_entropy_fp32_ws(dx, dt, -100, dl, c.rows, c.cols, ws, plan.workspace_bytes), "ce");
            ck(repro_cross_entropy_grad_fp32_ws(dt, -100, 1.0f, dg, c.rows, c.cols, ws, plan.workspace_bytes), "ce grad");
            ck(cudaDeviceSynchronize(), "sync");
            ck(cudaMemcpy(&loss, dl, 4, cudaMemcpyDeviceToHost), "c");
            grad.resize(xx.size());
            ck(cudaMemcpy(grad.data(), dg, xx.size() * 4, cudaMemcpyDeviceToHost), "c");
            cudaFree(dx); cudaFree(dt); cudaFree(dl); cudaFree(dg); cudaFree(ws);
        };
        float gl;
        std::vector<float> gg;
        gpu_ce(x, tg, gl, gg);
        expect(bits_of(gl) == bits_of(cl), std::string("cross-entropy loss GPU == CPU: ") + c.name);
        expect(diff_bits(gg, cg) == 0, std::string("cross-entropy grad GPU == CPU: ") + c.name);

        // Reversing the rows must not change the loss, and must reverse the gradient rows.
        std::vector<float> xr(x.size()), gr_want(x.size());
        std::vector<int> tr(c.rows);
        for (int r = 0; r < c.rows; ++r) {
            tr[r] = tg[c.rows - 1 - r];
            for (int k = 0; k < c.cols; ++k) {
                xr[(size_t)r * c.cols + k] = x[(size_t)(c.rows - 1 - r) * c.cols + k];
                gr_want[(size_t)r * c.cols + k] = gg[(size_t)(c.rows - 1 - r) * c.cols + k];
            }
        }
        float rl;
        std::vector<float> rg;
        gpu_ce(xr, tr, rl, rg);
        expect(bits_of(rl) == bits_of(gl) && diff_bits(rg, gr_want) == 0, std::string("cross-entropy independent of row order: ") + c.name);
    }

    std::printf("%s\n", failures ? "OVERALL: FAIL" : "OVERALL: PASS");
    return failures ? 1 : 0;
}
