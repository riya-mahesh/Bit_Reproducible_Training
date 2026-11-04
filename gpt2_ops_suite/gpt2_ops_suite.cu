// GPT-2 operation conformance suite.
//
// Runs every reproducible operation a GPT-2 (small) training step uses, at GPT-2
// shapes, on either the CPU (reference implementation of the same definition,
// multi-threaded) or the GPU (our kernels), and reports per operation:
//   - a hash of the inputs   (must match across machines: same inputs everywhere)
//   - a hash of the output   (must match across CPU and every GPU)
//   - median time            (GPU: CUDA events; CPU: wall clock)
//   - GPU only: median time of the usual non-reproducible baseline
//     (cuBLAS FP32 for GEMMs, a plain float reduction for sums)
// Results are also written to a file; compare_suite.sh compares files.
//
// Usage: gpt2_ops_suite --device cpu|gpu [--tokens 1024] [--reps 10] [--ops substr]
//                       [--slices 3] [--out file]
// Operations not implemented yet are listed as PENDING.
// Build CPU-only (no CUDA toolkit or GPU needed) with -DSUITE_CPU_ONLY.
#include "repro_gemm_cpu.h"
#include "repro_sum_math.cuh"
#include "repro_softmax_math.cuh"
#include "repro_conv_cpu.h"
#include "repro_nn_cpu.h"
#ifndef SUITE_CPU_ONLY
#include "repro_gemm.cuh"
#include "repro_sum.cuh"
#include "repro_softmax.cuh"
#include "repro_conv.cuh"
#include "repro_nn.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#ifndef SUITE_CPU_ONLY
static void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(1); }
}
static void ckb(cublasStatus_t s, const char* w) {
    if (s != CUBLAS_STATUS_SUCCESS) { std::fprintf(stderr, "%s: cuBLAS error %d\n", w, (int)s); std::exit(1); }
}
#else
typedef void* cudaStream_t;
#endif

// ---------------------------------------------------------------------------
// Portable inputs: SplitMix64 + exact arithmetic only (no libm), so every machine
// generates identical input bits.
// ---------------------------------------------------------------------------
struct Rng {
    uint64_t s;
    uint64_t next() {
        uint64_t z = (s += 0x9e3779b97f4a7c15ull);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    }
    double u01() { return (double)(next() >> 40) * (1.0 / 16777216.0); }
    double normal() { double t = 0; for (int i = 0; i < 12; ++i) t += u01(); return t - 6.0; }
};

static std::vector<float> gen_normal(size_t n, double scale, uint64_t seed) {
    Rng g{seed};
    std::vector<float> v(n);
    for (auto& x : v) x = (float)(scale * g.normal());
    return v;
}
static std::vector<float> gen_uniform(size_t n, double scale, uint64_t seed) {
    Rng g{seed};
    std::vector<float> v(n);
    for (auto& x : v) x = (float)(scale * g.u01());
    return v;
}

static uint64_t fnv1a(const std::vector<float>& v, uint64_t h = 0xcbf29ce484222325ull) {
    const unsigned char* p = reinterpret_cast<const unsigned char*>(v.data());
    for (size_t i = 0; i < v.size() * 4; ++i) { h ^= p[i]; h *= 0x100000001b3ull; }
    return h;
}

template<class F>
static void parallel_for(int n, F f) {
    unsigned nt = std::max(1u, std::thread::hardware_concurrency());
    std::atomic<int> next{0};
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < nt; ++t)
        pool.emplace_back([&] { for (int i; (i = next.fetch_add(1)) < n;) f(i); });
    for (auto& th : pool) th.join();
}

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------------------
// CPU reference implementations (same definitions as the GPU kernels)
// ---------------------------------------------------------------------------

// C[M,N] = op(A) * op(B): the repro_gemm definition, multi-threaded.
static void cpu_gemm(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C,
                     int M, int N, int K, bool ta, bool tb, int S) {
    C.assign((size_t)M * N, 0.0f);
    repro_gemm::cpu_gemm(A.data(), B.data(), C.data(), M, N, K, ta, tb, S);
}

// out[b] = exact sum (or mean) over a, c of x[a, b, c], rounded once.
static void cpu_sum(const std::vector<float>& x, std::vector<float>& out, int A, int B, int C, bool mean) {
    out.assign(B, 0.0f);
    parallel_for(B, [&](int b) {
        int64_t acc[repro_sum::LIMBS] = {};
        int nf = 0;
        for (int a = 0; a < A; ++a)
            for (int c = 0; c < C; ++c) repro_sum::acc_add(acc, 1, x[((size_t)a * B + b) * C + c], nf);
        out[b] = repro_sum::finalize(acc, 1, mean ? (uint32_t)((int64_t)A * C) : 1u, nf);
    });
}

#ifndef SUITE_CPU_ONLY
// ---------------------------------------------------------------------------
// GPU float baselines for sums (non-reproducible, for timing comparison only)
// ---------------------------------------------------------------------------
__global__ void base_columns(const float* x, float* out, int A, int B) {
    int b = blockIdx.x * 256 + threadIdx.x;
    if (b >= B) return;
    float s = 0.0f;
    for (int a = 0; a < A; ++a) s += x[(size_t)a * B + b];
    out[b] = s;
}
__global__ void base_rows(const float* x, float* out, int rows, int C) {
    int lane = threadIdx.x & 31, row = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (row >= rows) return;
    float s = 0.0f;
    for (int c = lane; c < C; c += 32) s += x[(size_t)row * C + c];
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xFFFFFFFFu, s, o);
    if (lane == 0) out[row] = s;
}
__global__ void base_softmax(const float* x, float* out, int rows, int cols) {
    int lane = threadIdx.x & 31, row = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (row >= rows) return;
    const float* p = x + (size_t)row * cols;
    float* o = out + (size_t)row * cols;
    float m = -INFINITY, s = 0.0f;
    for (int c = lane; c < cols; c += 32) m = fmaxf(m, p[c]);
    for (int k = 16; k > 0; k >>= 1) m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, k));
    for (int c = lane; c < cols; c += 32) { o[c] = expf(p[c] - m); s += o[c]; }
    for (int k = 16; k > 0; k >>= 1) s += __shfl_xor_sync(0xFFFFFFFFu, s, k);
    for (int c = lane; c < cols; c += 32) o[c] /= s;
}
__global__ void base_general(const float* x, float* out, int A, int B, int C, int chunks) {
    __shared__ float sh[256];
    int t = threadIdx.x, b = blockIdx.x, chunk = blockIdx.y;
    int count = A * C;
    int i0 = (int)((long long)count * chunk / chunks), i1 = (int)((long long)count * (chunk + 1) / chunks);
    float s = 0.0f;
    for (int i = i0 + t; i < i1; i += 256) { int a = i / C, c = i - a * C; s += x[((size_t)a * B + b) * C + c]; }
    sh[t] = s;
    __syncthreads();
    for (int k = 128; k > 0; k >>= 1) { if (t < k) sh[t] += sh[t + k]; __syncthreads(); }
    if (t == 0) atomicAdd(&out[b], sh[0]);
}
#endif

// ---------------------------------------------------------------------------
// Operation registry
// ---------------------------------------------------------------------------
struct Op {
    std::string name, shape;
    std::vector<std::vector<float>> inputs;
    std::function<void(const std::vector<std::vector<float>>&, std::vector<float>&)> cpu;  // CPU reference
    std::function<void(const std::vector<float*>&, float*, cudaStream_t)> gpu;   // reproducible GPU kernel(s)
    std::function<void(const std::vector<float*>&, float*, cudaStream_t)> base;  // non-reproducible baseline
    size_t out_size;
};

struct Ctx {
    int tokens, S;
#ifndef SUITE_CPU_ONLY
    cublasHandle_t cublas = nullptr;
#endif
    std::vector<void*> to_free;
};

// GEMM op: C[M,N] = op(A) * op(B) with PyTorch-style layouts.
static Op make_gemm(Ctx& ctx, const std::string& name, int M, int N, int K, bool ta, bool tb,
                    double sa, double sb, uint64_t seed, bool gpu_mode) {
    Op op;
    op.name = name;
    char buf[96];
    std::snprintf(buf, sizeof buf, "M=%d N=%d K=%d%s%s", M, N, K, ta ? " A^T" : "", tb ? " B^T" : "");
    op.shape = buf;
    op.inputs.push_back(gen_normal((size_t)M * K, sa, seed));
    op.inputs.push_back(gen_normal((size_t)K * N, sb, seed + 1));
    op.out_size = (size_t)M * N;
    int S = ctx.S;
    op.cpu = [M, N, K, ta, tb, S](const std::vector<std::vector<float>>& in, std::vector<float>& C) {
        cpu_gemm(in[0], in[1], C, M, N, K, ta, tb, S);
    };
#ifndef SUITE_CPU_ONLY
    if (gpu_mode) {
        repro_gemm::Config cfg;
        cfg.slices = S;
        cfg.trans_a = ta;
        cfg.trans_b = tb;
        repro_gemm::Plan plan;
        ck(repro_gemm::make_plan(M, N, K, cfg, &plan), "gemm plan");
        void* ws;
        ck(cudaMalloc(&ws, plan.workspace_bytes), "gemm workspace");
        ctx.to_free.push_back(ws);
        size_t bytes = plan.workspace_bytes;
        op.gpu = [=](const std::vector<float*>& in, float* out, cudaStream_t st) {
            ck(repro_gemm::repro_gemm_fp32_ws(in[0], in[1], out, M, N, K, cfg, ws, bytes, st), "repro gemm");
        };
        cublasHandle_t h = ctx.cublas;
        op.base = [=](const std::vector<float*>& in, float* out, cudaStream_t st) {
            // Row-major C = op(A) op(B)  <=>  column-major C^T = op(B)^T op(A)^T.
            const float one = 1.0f, zero = 0.0f;
            ckb(cublasSetStream(h, st), "stream");
            ckb(cublasSgemm(h, tb ? CUBLAS_OP_T : CUBLAS_OP_N, ta ? CUBLAS_OP_T : CUBLAS_OP_N, N, M, K, &one,
                            in[1], tb ? K : N, in[0], ta ? M : K, &zero, out, N), "sgemm");
        };
    }
#endif
    return op;
}

// Sum op: out[b] = sum over a, c of x[a, b, c] (or mean); x given by a generator.
static Op make_sum(Ctx& ctx, const std::string& name, int A, int B, int C, bool mean,
                   std::vector<float> x, bool gpu_mode) {
    Op op;
    op.name = name;
    char buf[96];
    std::snprintf(buf, sizeof buf, "[%d,%d,%d] -> %d%s", A, B, C, B, mean ? " mean" : "");
    op.shape = buf;
    op.inputs.push_back(std::move(x));
    op.out_size = B;
    op.cpu = [A, B, C, mean](const std::vector<std::vector<float>>& in, std::vector<float>& out) {
        cpu_sum(in[0], out, A, B, C, mean);
    };
#ifndef SUITE_CPU_ONLY
    if (gpu_mode) {
        repro_sum::SumConfig cfg;
        cfg.mean = mean;
        repro_sum::SumPlan plan;
        ck(repro_sum::make_sum_plan(A, B, C, cfg, &plan), "sum plan");
        void* ws = nullptr;
        if (plan.workspace_bytes) { ck(cudaMalloc(&ws, plan.workspace_bytes), "sum workspace"); ctx.to_free.push_back(ws); }
        size_t bytes = plan.workspace_bytes;
        op.gpu = [=](const std::vector<float*>& in, float* out, cudaStream_t st) {
            ck(repro_sum::repro_sum_fp32_ws(in[0], out, A, B, C, cfg, ws, bytes, st), "repro sum");
        };
        int sms = 0;
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
        int chunks = std::max(1, std::min(4 * sms / B + 1, A * C / 4096 + 1));
        op.base = [=](const std::vector<float*>& in, float* out, cudaStream_t st) {
            if (C == 1) base_columns<<<(B + 255) / 256, 256, 0, st>>>(in[0], out, A, B);
            else if (A == 1 && C <= 8192 && B >= 256) base_rows<<<(B + 7) / 8, 256, 0, st>>>(in[0], out, B, C);
            else {
                cudaMemsetAsync(out, 0, (size_t)B * 4, st);
                base_general<<<dim3(B, chunks), 256, 0, st>>>(in[0], out, A, B, C, chunks);
            }
            // (mean: the baseline would also divide; omitted, it only affects timing negligibly)
        };
    }
#endif
    return op;
}

// Softmax over the last dimension of [rows, cols]; x comes from a generator.
static Op make_softmax(Ctx& ctx, const std::string& name, int rows, int cols, std::vector<float> x, bool gpu_mode) {
    Op op;
    op.name = name;
    char buf[96];
    std::snprintf(buf, sizeof buf, "[%d,%d]", rows, cols);
    op.shape = buf;
    op.inputs.push_back(std::move(x));
    op.out_size = (size_t)rows * cols;
    op.cpu = [rows, cols](const std::vector<std::vector<float>>& in, std::vector<float>& out) {
        out.assign((size_t)rows * cols, 0.0f);
        parallel_for(rows, [&](int r) { repro_math::softmax_row(&in[0][(size_t)r * cols], &out[(size_t)r * cols], cols); });
    };
#ifndef SUITE_CPU_ONLY
    if (gpu_mode) {
        repro_math::SoftmaxPlan plan;
        ck(repro_math::make_softmax_plan(rows, cols, &plan), "softmax plan");
        void* ws;
        ck(cudaMalloc(&ws, plan.workspace_bytes), "softmax workspace");
        ctx.to_free.push_back(ws);
        size_t bytes = plan.workspace_bytes;
        op.gpu = [=](const std::vector<float*>& in, float* out, cudaStream_t st) {
            ck(repro_math::repro_softmax_fp32_ws(in[0], out, rows, cols, ws, bytes, st), "repro softmax");
        };
        op.base = [=](const std::vector<float*>& in, float* out, cudaStream_t st) {
            base_softmax<<<(rows + 7) / 8, 256, 0, st>>>(in[0], out, rows, cols);
        };
    }
#endif
    return op;
}

// Cross-entropy: output is the loss alone, or the gradient after a forward call.
static Op make_cross_entropy(Ctx& ctx, const std::string& name, int rows, int cols, bool want_grad, std::vector<float> x, uint64_t seed, bool gpu_mode) {
    Op op;
    op.name = name;
    char buf[96];
    std::snprintf(buf, sizeof buf, "[%d,%d]", rows, cols);
    op.shape = buf;
    op.inputs.push_back(std::move(x));
    op.out_size = want_grad ? (size_t)rows * cols : 1;
    Rng g{seed};
    std::vector<int> target(rows);
    for (auto& t : target) t = (int)(g.next() % (uint64_t)cols);
    op.cpu = [=](const std::vector<std::vector<float>>& in, std::vector<float>& out) {
        auto par = [](int n, const std::function<void(int)>& f) { parallel_for(n, f); };
        if (want_grad) {
            out.assign((size_t)rows * cols, 0.0f);
            repro_math::cross_entropy_cpu(in[0].data(), target.data(), -100, rows, cols, out.data(), 1.0f, par);
        } else {
            out.assign(1, repro_math::cross_entropy_cpu(in[0].data(), target.data(), -100, rows, cols, (float*)nullptr, 1.0f, par));
        }
    };
#ifndef SUITE_CPU_ONLY
    if (gpu_mode) {
        repro_math::CrossEntropyPlan plan;
        ck(repro_math::make_cross_entropy_plan(rows, cols, &plan), "ce plan");
        void* ws;
        int* dt;
        float* dl;
        ck(cudaMalloc(&ws, plan.workspace_bytes), "ce workspace");
        ck(cudaMalloc(&dt, target.size() * 4), "ce target");
        ck(cudaMalloc(&dl, 4), "ce loss");
        ck(cudaMemcpy(dt, target.data(), target.size() * 4, cudaMemcpyHostToDevice), "ce target");
        ctx.to_free.push_back(ws); ctx.to_free.push_back(dt); ctx.to_free.push_back(dl);
        size_t bytes = plan.workspace_bytes;
        op.gpu = [=](const std::vector<float*>& in, float* out, cudaStream_t st) {
            ck(repro_math::repro_cross_entropy_fp32_ws(in[0], dt, -100, want_grad ? dl : out, rows, cols, ws, bytes, st), "ce");
            if (want_grad) ck(repro_math::repro_cross_entropy_grad_fp32_ws(dt, -100, 1.0f, out, rows, cols, ws, bytes, st), "ce grad");
        };
    }
#endif
    return op;
}

// MaxPool: the output, or the backward result given dY (a forward call supplies the indices).
static Op make_pool(Ctx& ctx, const std::string& name, repro_nn::PoolShape g, bool backward, uint64_t seed, bool gpu_mode) {
    Op op;
    op.name = name;
    char buf[96];
    std::snprintf(buf, sizeof buf, "[%d,%d,%d,%d] %dx%d", g.N, g.C, g.H, g.W, g.KH, g.KW);
    op.shape = buf;
    op.inputs.push_back(gen_normal(g.input_size(), 1.0, seed));
    if (backward) op.inputs.push_back(gen_normal(g.output_size(), 1e-3, seed + 1));
    op.out_size = backward ? g.input_size() : g.output_size();
    op.cpu = [g, backward](const std::vector<std::vector<float>>& in, std::vector<float>& out) {
        std::vector<float> o(g.output_size());
        std::vector<int> id(g.output_size());
        repro_nn::cpu_max_pool2d(in[0].data(), o.data(), id.data(), g);
        if (!backward) { out = o; return; }
        out.assign(g.input_size(), 0.0f);
        repro_nn::cpu_max_pool2d_backward(in[1].data(), id.data(), out.data(), g);
    };
#ifndef SUITE_CPU_ONLY
    if (gpu_mode) {
        float* po;
        int* pi;
        ck(cudaMalloc(&po, g.output_size() * 4), "pool out");
        ck(cudaMalloc(&pi, g.output_size() * 4), "pool index");
        ctx.to_free.push_back(po); ctx.to_free.push_back(pi);
        op.gpu = [=](const std::vector<float*>& in, float* out, cudaStream_t st) {
            ck(repro_nn::repro_max_pool2d(in[0], backward ? po : out, pi, g, st), "pool");
            if (backward) ck(repro_nn::repro_max_pool2d_backward(in[1], pi, out, g, st), "pool backward");
        };
    }
#endif
    return op;
}

// Adam: one step from zero state; the output is the updated parameter array.
static Op make_adam(Ctx& ctx, const std::string& name, size_t n, uint64_t seed, bool gpu_mode) {
    Op op;
    op.name = name;
    char buf[96];
    std::snprintf(buf, sizeof buf, "n=%zu", n);
    op.shape = buf;
    op.inputs.push_back(gen_normal(n, 0.1, seed));
    op.inputs.push_back(gen_normal(n, 1e-3, seed + 1));
    op.out_size = n;
    repro_nn::AdamConfig c;
    c.lr = 0.01f;
    op.cpu = [n, c](const std::vector<std::vector<float>>& in, std::vector<float>& out) {
        out = in[0];
        std::vector<float> m(n, 0.0f), v(n, 0.0f);
        repro_nn::AdamClock k;
        repro_nn::cpu_adam(out.data(), in[1].data(), m.data(), v.data(), n, repro_nn::adam_next_step(c, k));
    };
#ifndef SUITE_CPU_ONLY
    if (gpu_mode) {
        float *dm, *dv;
        ck(cudaMalloc(&dm, n * 4), "adam m");
        ck(cudaMalloc(&dv, n * 4), "adam v");
        ctx.to_free.push_back(dm); ctx.to_free.push_back(dv);
        op.gpu = [=](const std::vector<float*>& in, float* out, cudaStream_t st) {
            repro_nn::AdamClock k;
            ck(cudaMemcpyAsync(out, in[0], n * 4, cudaMemcpyDeviceToDevice, st), "adam copy");
            ck(cudaMemsetAsync(dm, 0, n * 4, st), "adam m");
            ck(cudaMemsetAsync(dv, 0, n * 4, st), "adam v");
            ck(repro_nn::repro_adam(out, in[1], dm, dv, n, repro_nn::adam_next_step(c, k), st), "adam");
        };
    }
#endif
    return op;
}

// Conv2d at LeNet-5 shapes: forward, weight gradient, input gradient.
enum class ConvOp { Forward, GradWeight, GradInput };
static Op make_conv(Ctx& ctx, const std::string& name, repro_conv::ConvShape g, ConvOp kind, uint64_t seed, bool gpu_mode) {
    using namespace repro_conv;
    Op op;
    op.name = name;
    char buf[96];
    std::snprintf(buf, sizeof buf, "[%d,%d,%d,%d] F=%d %dx%d", g.N, g.C, g.H, g.W, g.F, g.KH, g.KW);
    op.shape = buf;
    int S = ctx.S;
    std::vector<float> x = gen_normal(g.input_size(), 1.0, seed), w = gen_normal(g.weight_size(), 0.1, seed + 1),
                       dy = gen_normal(g.output_size(), 1e-3, seed + 2);
    if (kind == ConvOp::Forward) { op.inputs = {x, w}; op.out_size = g.output_size(); }
    else if (kind == ConvOp::GradWeight) { op.inputs = {dy, x}; op.out_size = g.weight_size(); }
    else { op.inputs = {dy, w}; op.out_size = g.input_size(); }
    op.cpu = [g, S, kind](const std::vector<std::vector<float>>& in, std::vector<float>& out) {
        out.assign(kind == ConvOp::Forward ? g.output_size() : kind == ConvOp::GradWeight ? g.weight_size() : g.input_size(), 0.0f);
        if (kind == ConvOp::Forward) cpu_conv2d(in[0].data(), in[1].data(), out.data(), g, S);
        else if (kind == ConvOp::GradWeight) cpu_conv2d_grad_weight(in[0].data(), in[1].data(), out.data(), g, S);
        else cpu_conv2d_grad_input(in[0].data(), in[1].data(), out.data(), g, S);
    };
#ifndef SUITE_CPU_ONLY
    if (gpu_mode) {
        size_t bytes;
        if (kind == ConvOp::Forward) ck(conv2d_workspace(g, S, &bytes), "conv plan");
        else if (kind == ConvOp::GradWeight) ck(conv2d_grad_weight_workspace(g, S, &bytes), "conv plan");
        else ck(conv2d_grad_input_workspace(g, S, &bytes), "conv plan");
        void* ws;
        ck(cudaMalloc(&ws, bytes), "conv workspace");
        ctx.to_free.push_back(ws);
        op.gpu = [=](const std::vector<float*>& in, float* out, cudaStream_t st) {
            if (kind == ConvOp::Forward) ck(repro_conv2d_ws(in[0], in[1], out, g, S, ws, bytes, st), "conv");
            else if (kind == ConvOp::GradWeight) ck(repro_conv2d_grad_weight_ws(in[0], in[1], out, g, S, ws, bytes, st), "conv");
            else ck(repro_conv2d_grad_input_ws(in[0], in[1], out, g, S, ws, bytes, st), "conv");
        };
    }
#endif
    return op;
}

// GPT-2 small: d=768, 4d=3072, vocab=50257, 12 heads of 64, context T=1024.
static std::vector<Op> build_ops(Ctx& ctx, bool gpu_mode) {
    const int D = 768, FF = 3072, V = 50257, H = 12, HD = 64, T = 1024;
    const int N = ctx.tokens;
    std::vector<Op> ops;
    uint64_t seed = 1000;
    // Linear layers: forward Y = X W^T, input grad dX = dY W, weight grad dW = dY^T X.
    struct Lin { const char* name; int in, out; };
    const Lin lins[] = {{"attn_qkv", D, 3 * D}, {"attn_proj", D, D}, {"mlp_fc", D, FF}, {"mlp_proj", FF, D},
                        {"lm_head", D, V}};
    for (const Lin& l : lins) {
        ops.push_back(make_gemm(ctx, std::string(l.name) + ".fwd", N, l.out, l.in, false, true, 1.0, 0.02, seed += 10, gpu_mode));
        ops.push_back(make_gemm(ctx, std::string(l.name) + ".dX", N, l.in, l.out, false, false, 1e-3, 0.02, seed += 10, gpu_mode));
        ops.push_back(make_gemm(ctx, std::string(l.name) + ".dW", l.out, l.in, N, true, false, 1e-3, 1.0, seed += 10, gpu_mode));
    }
    // One attention head (batched mode is pending; per head these are plain GEMMs).
    ops.push_back(make_gemm(ctx, "attn_head.scores QK^T", T, T, HD, false, true, 1.0, 1.0, seed += 10, gpu_mode));
    ops.push_back(make_gemm(ctx, "attn_head.PV", T, HD, T, false, false, 2.0 / T, 1.0, seed += 10, gpu_mode));
    ops.push_back(make_gemm(ctx, "attn_head.dP = dO V^T", T, T, HD, false, true, 1e-3, 1.0, seed += 10, gpu_mode));
    ops.push_back(make_gemm(ctx, "attn_head.dV = P^T dO", T, HD, T, true, false, 2.0 / T, 1e-3, seed += 10, gpu_mode));
    ops.push_back(make_gemm(ctx, "attn_head.dQ = dS K", T, HD, T, false, false, 1e-4, 1.0, seed += 10, gpu_mode));
    ops.push_back(make_gemm(ctx, "attn_head.dK = dS^T Q", T, HD, T, true, false, 1e-4, 1.0, seed += 10, gpu_mode));

    // Reductions.
    const struct { const char* name; int width; } biases[] = {
        {"bias_grad attn_qkv", 3 * D}, {"bias_grad attn_proj", D}, {"bias_grad mlp_fc", FF}, {"bias_grad mlp_proj", D},
        {"layernorm gamma/beta grad", D}};
    for (auto& b : biases)
        ops.push_back(make_sum(ctx, b.name, N, b.width, 1, false, gen_normal((size_t)N * b.width, 1e-3, seed += 10), gpu_mode));
    ops.push_back(make_sum(ctx, "layernorm row sum (mean)", 1, N, D, true, gen_normal((size_t)N * D, 1.0, seed += 10), gpu_mode));
    {
        // Attention softmax denominators for one sequence: H*T rows of length T, causal
        // (entries after the diagonal are zero), positive exp-like values elsewhere.
        std::vector<float> e = gen_uniform((size_t)H * T * T, 1.0, seed += 10);
        for (int r = 0; r < H * T; ++r)
            for (int j = r % T + 1; j < T; ++j) e[(size_t)r * T + j] = 0.0f;
        ops.push_back(make_sum(ctx, "attn softmax row sum (causal)", 1, H * T, T, false, std::move(e), gpu_mode));
    }
    ops.push_back(make_sum(ctx, "cross_entropy row sum (vocab)", 1, N, V, false,
                           gen_uniform((size_t)N * V, 1e-3, seed += 10), gpu_mode));
    ops.push_back(make_sum(ctx, "loss mean over tokens", 1, 1, N, true, gen_uniform((size_t)N, 11.0, seed += 10), gpu_mode));
    // Softmax: MNIST logits, causal attention over one sequence, cross-entropy over the vocabulary.
    ops.push_back(make_softmax(ctx, "softmax MNIST logits", 100, 10, gen_normal(100 * 10, 5.0, seed += 10), gpu_mode));
    {
        std::vector<float> x = gen_normal((size_t)H * T * T, 4.0, seed += 10);
        for (int r = 0; r < H * T; ++r)
            for (int j = r % T + 1; j < T; ++j) x[(size_t)r * T + j] = -INFINITY;
        ops.push_back(make_softmax(ctx, "softmax attention (causal)", H * T, T, std::move(x), gpu_mode));
    }
    ops.push_back(make_softmax(ctx, "softmax vocab", N, V, gen_normal((size_t)N * V, 8.0, seed += 10), gpu_mode));

    // LeNet-5 convolutions.
    repro_conv::ConvShape c1, c2;
    c1.N = 100; c1.C = 1; c1.H = 28; c1.W = 28; c1.F = 6; c1.KH = 5; c1.KW = 5; c1.PY = 2; c1.PX = 2;
    c2.N = 100; c2.C = 6; c2.H = 14; c2.W = 14; c2.F = 16; c2.KH = 5; c2.KW = 5;
    ops.push_back(make_conv(ctx, "lenet conv1.fwd", c1, ConvOp::Forward, seed += 10, gpu_mode));
    ops.push_back(make_conv(ctx, "lenet conv1.dW", c1, ConvOp::GradWeight, seed += 10, gpu_mode));
    ops.push_back(make_conv(ctx, "lenet conv2.fwd", c2, ConvOp::Forward, seed += 10, gpu_mode));
    ops.push_back(make_conv(ctx, "lenet conv2.dW", c2, ConvOp::GradWeight, seed += 10, gpu_mode));
    ops.push_back(make_conv(ctx, "lenet conv2.dX", c2, ConvOp::GradInput, seed += 10, gpu_mode));

    // Cross-entropy: MNIST logits and GPT-2 vocabulary logits.
    {
        std::vector<float> mn = gen_normal(100 * 10, 5.0, seed += 10);
        ops.push_back(make_cross_entropy(ctx, "cross_entropy MNIST loss", 100, 10, false, mn, seed + 1, gpu_mode));
        ops.push_back(make_cross_entropy(ctx, "cross_entropy MNIST loss+grad", 100, 10, true, mn, seed + 1, gpu_mode));
        std::vector<float> vo = gen_normal((size_t)N * V, 8.0, seed += 10);
        ops.push_back(make_cross_entropy(ctx, "cross_entropy vocab loss", N, V, false, vo, seed + 1, gpu_mode));
        ops.push_back(make_cross_entropy(ctx, "cross_entropy vocab loss+grad", N, V, true, std::move(vo), seed + 1, gpu_mode));
    }

    // LeNet-5 pooling and the optimizer (61,706 parameters).
    repro_nn::PoolShape p1{100, 6, 28, 28, 2, 2, 2, 2}, p2{100, 16, 10, 10, 2, 2, 2, 2};
    ops.push_back(make_pool(ctx, "lenet pool1.fwd", p1, false, seed += 10, gpu_mode));
    ops.push_back(make_pool(ctx, "lenet pool1.back", p1, true, seed += 10, gpu_mode));
    ops.push_back(make_pool(ctx, "lenet pool2.fwd", p2, false, seed += 10, gpu_mode));
    ops.push_back(make_pool(ctx, "lenet pool2.back", p2, true, seed += 10, gpu_mode));
    ops.push_back(make_adam(ctx, "lenet adam step", 61706, seed += 10, gpu_mode));
    return ops;
}

static const char* kPending[] = {
    "exact dot product / sum of squares (LayerNorm variance, softmax backward, grad-norm)",
    "LayerNorm forward/backward", "softmax backward and a fused softmax kernel",
    "GELU (tanh) forward/backward",
    "embedding forward/backward (exact scatter-add by token id)", "batched attention GEMM (all heads at once)",
    "AdamW + global-norm gradient clipping"};

int main(int argc, char** argv) {
    std::string device = "gpu", filter, out_file;
    int tokens = 1024, reps = 10, S = 3;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto val = [&] { if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(1); } return std::string(argv[++i]); };
        if (a == "--device") device = val();
        else if (a == "--tokens") tokens = std::atoi(val().c_str());
        else if (a == "--reps") reps = std::atoi(val().c_str());
        else if (a == "--ops") filter = val();
        else if (a == "--slices") S = std::atoi(val().c_str());
        else if (a == "--out") out_file = val();
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 1; }
    }
    bool gpu_mode = device == "gpu";
    if (!gpu_mode && device != "cpu") { std::fprintf(stderr, "--device must be cpu or gpu\n"); return 1; }

    Ctx ctx{tokens, S};
    std::string where;
#ifdef SUITE_CPU_ONLY
    if (gpu_mode) { std::fprintf(stderr, "built with SUITE_CPU_ONLY: only --device cpu is available\n"); return 1; }
#else
    if (gpu_mode) {
        cudaDeviceProp prop;
        ck(cudaGetDeviceProperties(&prop, 0), "props");
        int rt = 0;
        cudaRuntimeGetVersion(&rt);
        char buf[160];
        std::snprintf(buf, sizeof buf, "GPU %s sm_%d%d CUDA %d", prop.name, prop.major, prop.minor, rt);
        where = buf;
        ckb(cublasCreate(&ctx.cublas), "cublasCreate");
        ckb(cublasSetMathMode(ctx.cublas, CUBLAS_DEFAULT_MATH), "math mode");   // no TF32
    }
#endif
    if (!gpu_mode) {
        where = "CPU " + std::to_string(std::thread::hardware_concurrency()) + " threads";
    }
    std::vector<Op> ops = build_ops(ctx, gpu_mode);

    FILE* fo = out_file.empty() ? nullptr : std::fopen(out_file.c_str(), "w");
    std::printf("device: %s | GPT-2 small, %d tokens, S=%d, %d reps\n", where.c_str(), tokens, S, reps);
    if (fo) std::fprintf(fo, "# %s tokens=%d S=%d\n", where.c_str(), tokens, S);
    std::printf("%-34s %-26s %-16s %-16s %10s %10s %7s\n", "operation", "shape", "input hash", "output hash",
                "ms", gpu_mode ? "base_ms" : "", gpu_mode ? "x base" : "");

    cudaStream_t st = nullptr;
#ifndef SUITE_CPU_ONLY
    cudaEvent_t e0, e1;
    if (gpu_mode) { ck(cudaEventCreate(&e0), "ev"); ck(cudaEventCreate(&e1), "ev"); }
#endif
    (void)st;
    for (Op& op : ops) {
        if (!filter.empty() && op.name.find(filter) == std::string::npos) continue;
        uint64_t hin = 0xcbf29ce484222325ull;
        for (auto& v : op.inputs) hin = fnv1a(v, hin);
        std::vector<float> out;
        double ms = 0, base_ms = -1;
        if (!gpu_mode) {
            double t0 = now_ms();
            op.cpu(op.inputs, out);
            ms = now_ms() - t0;
        } else {
#ifndef SUITE_CPU_ONLY
            std::vector<float*> din;
            for (auto& v : op.inputs) {
                float* d;
                ck(cudaMalloc(&d, v.size() * 4), "malloc in");
                ck(cudaMemcpy(d, v.data(), v.size() * 4, cudaMemcpyHostToDevice), "copy in");
                din.push_back(d);
            }
            float* dout;
            ck(cudaMalloc(&dout, op.out_size * 4), "malloc out");
            auto time_fn = [&](auto& fn) {
                for (int i = 0; i < 2; ++i) fn(din, dout, st);
                ck(cudaDeviceSynchronize(), "warmup");
                std::vector<float> t;
                for (int i = 0; i < reps; ++i) {
                    ck(cudaEventRecord(e0, st), "rec");
                    fn(din, dout, st);
                    ck(cudaEventRecord(e1, st), "rec");
                    ck(cudaEventSynchronize(e1), "sync");
                    float m;
                    ck(cudaEventElapsedTime(&m, e0, e1), "elapsed");
                    t.push_back(m);
                }
                std::sort(t.begin(), t.end());
                return (double)t[t.size() / 2];
            };
            if (op.base) base_ms = time_fn(op.base);   // only some operations have a baseline
            ck(cudaMemset(dout, 0xFF, op.out_size * 4), "poison");
            ms = time_fn(op.gpu);
            out.resize(op.out_size);
            ck(cudaMemcpy(out.data(), dout, op.out_size * 4, cudaMemcpyDeviceToHost), "copy out");
            for (float* d : din) cudaFree(d);
            cudaFree(dout);
#endif
        }
        uint64_t hout = fnv1a(out);
        char ratio[16] = "";
        if (base_ms > 0) std::snprintf(ratio, sizeof ratio, "%.2f", ms / base_ms);
        std::printf("%-34s %-26s %016llx %016llx %10.3f %10s %7s\n", op.name.c_str(), op.shape.c_str(),
                    (unsigned long long)hin, (unsigned long long)hout, ms,
                    base_ms >= 0 ? std::to_string(base_ms).substr(0, 8).c_str() : "", ratio);
        std::fflush(stdout);
        if (fo) std::fprintf(fo, "OP\t%s\t%s\tin=%016llx\tout=%016llx\tms=%.4f\tbase_ms=%.4f\n", op.name.c_str(),
                             op.shape.c_str(), (unsigned long long)hin, (unsigned long long)hout, ms, base_ms);
        if (fo) std::fflush(fo);   // keep finished results if a later operation fails
    }
    std::printf("\nPENDING (not implemented yet, will be added to this suite):\n");
    for (const char* p : kPending) std::printf("  - %s\n", p);
    if (fo) std::fclose(fo);
#ifndef SUITE_CPU_ONLY
    for (void* p : ctx.to_free) cudaFree(p);
    if (ctx.cublas) cublasDestroy(ctx.cublas);
#endif
    return 0;
}
