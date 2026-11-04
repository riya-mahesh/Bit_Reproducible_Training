#include "repro_softmax.cuh"
#include "repro_softmax_math.cuh"
#include "repro_sum.cuh"
#include "repro_sum_fast.cuh"
#include <cuda_runtime.h>
#include <cstdint>

namespace repro_math {

namespace {
constexpr int SM_THREADS = 256;
constexpr int SM_WARPS = SM_THREADS / 32;
// One warp per row runs everything in one launch, but a warp only works on one row, so very
// long rows with few of them keep the staged kernels, which spread each row over many blocks.
// Both give the same bits.
inline bool use_fused(int rows, int cols) { return rows >= 256 || cols <= 8192; }
}

__global__ void __launch_bounds__(SM_THREADS)
exp_kernel(const float* x, float* out, size_t n) {
    size_t i = (size_t)blockIdx.x * SM_THREADS + threadIdx.x;
    if (i < n) out[i] = repro_expf(x[i]);
}

__global__ void __launch_bounds__(SM_THREADS)
log_kernel(const float* x, float* out, size_t n) {
    size_t i = (size_t)blockIdx.x * SM_THREADS + threadIdx.x;
    if (i < n) out[i] = repro_logf(x[i]);
}

// Largest non-NaN entry of each row: one warp per row.
__global__ void __launch_bounds__(SM_THREADS)
row_max_kernel(const float* __restrict__ x, float* __restrict__ maxes, int rows, int cols) {
    int lane = threadIdx.x & 31, row = blockIdx.x * SM_WARPS + (threadIdx.x >> 5);
    if (row >= rows) return;
    const float* p = x + (size_t)row * cols;
    float m = -positive_inf();
    for (int c = lane; c < cols; c += 32) {
        float v = p[c];
        if (v > m) m = v;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        float other = __shfl_xor_sync(0xFFFFFFFFu, m, o);
        if (other > m) m = other;
    }
    if (lane == 0) maxes[row] = m;
}

// out[r][c] = repro_expf(x[r][c] - maxes[r]). x may equal out.
__global__ void __launch_bounds__(SM_THREADS)
softmax_exp_kernel(const float* x, const float* __restrict__ maxes, float* out, uint32_t total, int cols) {
    uint32_t i = (uint32_t)blockIdx.x * SM_THREADS + threadIdx.x;
    if (i >= total) return;
    out[i] = repro_expf(fsub(x[i], maxes[i / (uint32_t)cols]));
}

// out[r][c] /= sums[r]. A NaN sum means the row had NaN or +inf, so the row is NaN.
__global__ void __launch_bounds__(SM_THREADS)
softmax_div_kernel(const float* __restrict__ sums, float* __restrict__ out, uint32_t total, int cols) {
    uint32_t i = (uint32_t)blockIdx.x * SM_THREADS + threadIdx.x;
    if (i >= total) return;
    float S = sums[i / (uint32_t)cols];
    out[i] = S != S ? canonical_nan() : fdiv(out[i], S);
}

// loss[r] for each row.
__global__ void __launch_bounds__(SM_THREADS)
ce_loss_kernel(const float* __restrict__ x, const int* __restrict__ target, const float* __restrict__ maxes,
               const float* __restrict__ sums, float* __restrict__ loss, int ignore_index, int rows, int cols) {
    int r = blockIdx.x * SM_THREADS + threadIdx.x;
    if (r >= rows) return;
    int t = target[r], kind = ce_row_kind(t, ignore_index, cols);
    float x_t = kind > 0 ? x[(size_t)r * cols + t] : 0.0f;
    loss[r] = ce_row_loss(sums[r], x_t, maxes[r], kind);
}

// One block: the exact sum of the row losses over the number of rows that count, rounded once.
__global__ void __launch_bounds__(SM_THREADS)
ce_mean_kernel(const float* __restrict__ loss, const int* __restrict__ target, float* __restrict__ out,
               int* __restrict__ count_out, int ignore_index, int rows, int cols) {
    __shared__ int64_t acc[repro_sum::LIMBS];
    __shared__ int count, nf;
    int t = threadIdx.x;
    if (t < repro_sum::LIMBS) acc[t] = 0;
    if (t == 0) count = nf = 0;
    __syncthreads();
    for (int r = t; r < rows; r += SM_THREADS) {
        if (ce_row_kind(target[r], ignore_index, cols) != 0) atomicAdd(&count, 1);
        int idx;
        int64_t lo, hi;
        if (!repro_sum::split_float(loss[r], idx, lo, hi)) { atomicOr(&nf, 1); continue; }
        atomicAdd(reinterpret_cast<unsigned long long*>(&acc[idx]), (unsigned long long)lo);
        atomicAdd(reinterpret_cast<unsigned long long*>(&acc[idx + 1]), (unsigned long long)hi);
    }
    __syncthreads();
    if (t == 0) {
        *count_out = count;
        *out = count == 0 ? canonical_nan() : repro_sum::finalize(acc, 1, (uint32_t)count, nf);
    }
}

__global__ void __launch_bounds__(SM_THREADS)
ce_grad_kernel(const float* __restrict__ e, const float* __restrict__ sums, const int* __restrict__ target,
               const int* __restrict__ count, float* __restrict__ grad, float seed, int ignore_index, int rows, int cols) {
    int lane = threadIdx.x & 31, row = blockIdx.x * SM_WARPS + (threadIdx.x >> 5);
    if (row >= rows) return;
    int tg = target[row], kind = ce_row_kind(tg, ignore_index, cols), n = *count;
    float S = sums[row];
    size_t base = (size_t)row * cols;
    for (int c = lane; c < cols; c += 32) grad[base + c] = ce_grad(e[base + c], S, c == tg, kind, n, seed);
}

// ---------------- fused kernels: one warp per row, one launch ----------------

// exp, exact sum, and largest entry m of one row, inside its warp. Writes e, returns S.
__device__ __forceinline__ float row_exp_and_sum(const float* p, float* o, int cols, int lane, int64_t* full, float& m) {
    m = -positive_inf();
    for (int c = lane; c < cols; c += 32) {
        float v = p[c];
        if (v > m) m = v;
    }
#pragma unroll
    for (int k = 16; k > 0; k >>= 1) {
        float other = __shfl_xor_sync(0xFFFFFFFFu, m, k);
        if (other > m) m = other;
    }
    if (lane < repro_sum::LIMBS) full[lane] = 0;
    __syncwarp();
    // The largest entry gives exp = 1, so nearly every value lands in the window around it.
    const int low = repro_sum::window_low_for(126);
    repro_sum::FastAcc s;
    repro_sum::fast_init(s);
    int nf = 0;
    for (int c = lane; c < cols; c += 32) {
        float e = repro_expf(fsub(p[c], m));
        o[c] = e;
        repro_sum::fast_add<true>(s, low, e, full, nf);
    }
    repro_sum::fast_fold(s);
    int64_t r0 = repro_sum::warp_sum(s.r[0]), r1 = repro_sum::warp_sum(s.r[1]), r2 = repro_sum::warp_sum(s.r[2]);
    nf = __any_sync(0xFFFFFFFFu, nf);
    __syncwarp();                                    // spills into full are complete
    float S = 0.0f;
    if (lane == 0) {
        full[low] += r0;
        full[low + 1] += r1;
        full[low + 2] += r2;
        S = repro_sum::finalize(full, 1, 1u, nf);
    }
    return __shfl_sync(0xFFFFFFFFu, S, 0);
}

__global__ void __launch_bounds__(SM_THREADS)
softmax_fused_kernel(const float* x, float* out, int rows, int cols) {
    __shared__ int64_t full[SM_WARPS][repro_sum::LIMBS];
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5, row = blockIdx.x * SM_WARPS + w;
    if (row >= rows) return;
    float* o = out + (size_t)row * cols;
    float m;
    float S = row_exp_and_sum(x + (size_t)row * cols, o, cols, lane, full[w], m);
    for (int c = lane; c < cols; c += 32) o[c] = S != S ? canonical_nan() : fdiv(o[c], S);
}

__global__ void __launch_bounds__(SM_THREADS)
ce_fused_kernel(const float* __restrict__ x, const int* __restrict__ target, float* __restrict__ e,
                float* __restrict__ sums, float* __restrict__ loss, int ignore_index, int rows, int cols) {
    __shared__ int64_t full[SM_WARPS][repro_sum::LIMBS];
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5, row = blockIdx.x * SM_WARPS + w;
    if (row >= rows) return;
    const float* p = x + (size_t)row * cols;
    float m;
    float S = row_exp_and_sum(p, e + (size_t)row * cols, cols, lane, full[w], m);
    if (lane == 0) {
        sums[row] = S;
        int t = target[row], kind = ce_row_kind(t, ignore_index, cols);
        loss[row] = ce_row_loss(S, kind > 0 ? p[t] : 0.0f, m, kind);
    }
}

namespace {

constexpr size_t ALIGN = 256;
size_t align_up(size_t x) { return (x + ALIGN - 1) / ALIGN * ALIGN; }

// Workspace: maxes[rows], sums[rows], then the workspace of repro_sum.
cudaError_t layout(int rows, int cols, SoftmaxPlan& plan, size_t& sums_at, size_t& sum_ws_at, size_t& sum_ws) {
    if (rows <= 0 || cols <= 0 || (uint64_t)rows * (uint64_t)cols >= (1ull << 32)) return cudaErrorInvalidValue;
    repro_sum::SumPlan sp;
    cudaError_t e = repro_sum::make_sum_plan(1, rows, cols, repro_sum::SumConfig(), &sp);
    if (e != cudaSuccess) return e;
    sums_at = align_up((size_t)rows * sizeof(float));
    sum_ws_at = sums_at + align_up((size_t)rows * sizeof(float));
    sum_ws = sp.workspace_bytes;
    plan.workspace_bytes = sum_ws_at + sum_ws;
    return cudaSuccess;
}

} // namespace

namespace {

// Workspace: e[rows * cols], maxes[rows], sums[rows], loss[rows], count, repro_sum's workspace.
struct CeLayout { size_t maxes, sums, loss, count, sum_ws, sum_bytes, total; };

cudaError_t ce_layout(int rows, int cols, CeLayout& L) {
    if (rows <= 0 || cols <= 0 || (uint64_t)rows * (uint64_t)cols >= (1ull << 32)) return cudaErrorInvalidValue;
    repro_sum::SumPlan sp;
    cudaError_t e = repro_sum::make_sum_plan(1, rows, cols, repro_sum::SumConfig(), &sp);
    if (e != cudaSuccess) return e;
    size_t row_bytes = align_up((size_t)rows * sizeof(float));
    L.maxes = align_up((size_t)rows * cols * sizeof(float));
    L.sums = L.maxes + row_bytes;
    L.loss = L.sums + row_bytes;
    L.count = L.loss + row_bytes;
    L.sum_ws = L.count + ALIGN;
    L.sum_bytes = sp.workspace_bytes;
    L.total = L.sum_ws + L.sum_bytes;
    return cudaSuccess;
}

} // namespace

cudaError_t make_cross_entropy_plan(int rows, int cols, CrossEntropyPlan* plan) {
    CeLayout L;
    cudaError_t e = ce_layout(rows, cols, L);
    if (e == cudaSuccess) plan->workspace_bytes = L.total;
    return e;
}

cudaError_t repro_cross_entropy_fp32_ws(const float* x, const int* target, int ignore_index, float* loss,
                                        int rows, int cols, void* workspace, size_t workspace_bytes, cudaStream_t stream) {
    CeLayout L;
    cudaError_t e = ce_layout(rows, cols, L);
    if (e != cudaSuccess) return e;
    if (workspace_bytes < L.total) return cudaErrorInvalidValue;
    char* base = static_cast<char*>(workspace);
    float* ev = reinterpret_cast<float*>(base);
    float* maxes = reinterpret_cast<float*>(base + L.maxes);
    float* sums = reinterpret_cast<float*>(base + L.sums);
    float* row_loss = reinterpret_cast<float*>(base + L.loss);
    int* count = reinterpret_cast<int*>(base + L.count);
    uint32_t total = (uint32_t)((uint64_t)rows * (uint64_t)cols);
    unsigned blocks = (unsigned)(((uint64_t)total + SM_THREADS - 1) / SM_THREADS);

    if (use_fused(rows, cols)) {
        ce_fused_kernel<<<(rows + SM_WARPS - 1) / SM_WARPS, SM_THREADS, 0, stream>>>(x, target, ev, sums, row_loss, ignore_index, rows, cols);
    } else {
        row_max_kernel<<<(rows + SM_WARPS - 1) / SM_WARPS, SM_THREADS, 0, stream>>>(x, maxes, rows, cols);
        if ((e = cudaGetLastError()) != cudaSuccess) return e;
        softmax_exp_kernel<<<blocks, SM_THREADS, 0, stream>>>(x, maxes, ev, total, cols);
        if ((e = cudaGetLastError()) != cudaSuccess) return e;
        e = repro_sum::repro_sum_fp32_ws(ev, sums, 1, rows, cols, repro_sum::SumConfig(), base + L.sum_ws, L.sum_bytes, stream);
        if (e != cudaSuccess) return e;
        ce_loss_kernel<<<(rows + SM_THREADS - 1) / SM_THREADS, SM_THREADS, 0, stream>>>(x, target, maxes, sums, row_loss, ignore_index, rows, cols);
        if ((e = cudaGetLastError()) != cudaSuccess) return e;
    }
    if ((e = cudaGetLastError()) != cudaSuccess) return e;
    ce_mean_kernel<<<1, SM_THREADS, 0, stream>>>(row_loss, target, loss, count, ignore_index, rows, cols);
    return cudaGetLastError();
}

cudaError_t repro_cross_entropy_grad_fp32_ws(const int* target, int ignore_index, float seed, float* grad,
                                             int rows, int cols, void* workspace, size_t workspace_bytes, cudaStream_t stream) {
    CeLayout L;
    cudaError_t e = ce_layout(rows, cols, L);
    if (e != cudaSuccess) return e;
    if (workspace_bytes < L.total) return cudaErrorInvalidValue;
    char* base = static_cast<char*>(workspace);
    ce_grad_kernel<<<(rows + SM_WARPS - 1) / SM_WARPS, SM_THREADS, 0, stream>>>(reinterpret_cast<const float*>(base),
                                                   reinterpret_cast<const float*>(base + L.sums), target,
                                                   reinterpret_cast<const int*>(base + L.count), grad, seed, ignore_index, rows, cols);
    return cudaGetLastError();
}

cudaError_t repro_exp_fp32(const float* x, float* out, size_t n, cudaStream_t stream) {
    if (n == 0) return cudaSuccess;
    exp_kernel<<<(unsigned)((n + SM_THREADS - 1) / SM_THREADS), SM_THREADS, 0, stream>>>(x, out, n);
    return cudaGetLastError();
}

cudaError_t repro_log_fp32(const float* x, float* out, size_t n, cudaStream_t stream) {
    if (n == 0) return cudaSuccess;
    log_kernel<<<(unsigned)((n + SM_THREADS - 1) / SM_THREADS), SM_THREADS, 0, stream>>>(x, out, n);
    return cudaGetLastError();
}

cudaError_t make_softmax_plan(int rows, int cols, SoftmaxPlan* plan) {
    size_t a, b, c;
    return layout(rows, cols, *plan, a, b, c);
}

cudaError_t repro_softmax_fp32_ws(const float* x, float* out, int rows, int cols,
                                  void* workspace, size_t workspace_bytes, cudaStream_t stream) {
    SoftmaxPlan plan;
    size_t sums_at, sum_ws_at, sum_ws;
    cudaError_t e = layout(rows, cols, plan, sums_at, sum_ws_at, sum_ws);
    if (e != cudaSuccess) return e;
    if (workspace_bytes < plan.workspace_bytes) return cudaErrorInvalidValue;
    char* base = static_cast<char*>(workspace);
    float* maxes = reinterpret_cast<float*>(base);
    float* sums = reinterpret_cast<float*>(base + sums_at);
    uint32_t total = (uint32_t)((uint64_t)rows * (uint64_t)cols);
    unsigned blocks = (unsigned)(((uint64_t)total + SM_THREADS - 1) / SM_THREADS);

    if (use_fused(rows, cols)) {
        softmax_fused_kernel<<<(rows + SM_WARPS - 1) / SM_WARPS, SM_THREADS, 0, stream>>>(x, out, rows, cols);
        return cudaGetLastError();
    }
    row_max_kernel<<<(rows + SM_WARPS - 1) / SM_WARPS, SM_THREADS, 0, stream>>>(x, maxes, rows, cols);
    if ((e = cudaGetLastError()) != cudaSuccess) return e;
    softmax_exp_kernel<<<blocks, SM_THREADS, 0, stream>>>(x, maxes, out, total, cols);
    if ((e = cudaGetLastError()) != cudaSuccess) return e;
    e = repro_sum::repro_sum_fp32_ws(out, sums, 1, rows, cols, repro_sum::SumConfig(),
                                     base + sum_ws_at, sum_ws, stream);
    if (e != cudaSuccess) return e;
    softmax_div_kernel<<<blocks, SM_THREADS, 0, stream>>>(sums, out, total, cols);
    return cudaGetLastError();
}

cudaError_t repro_softmax_fp32(const float* x, float* out, int rows, int cols, cudaStream_t stream) {
    SoftmaxPlan plan;
    cudaError_t e = make_softmax_plan(rows, cols, &plan);
    if (e != cudaSuccess) return e;
    void* ws = nullptr;
    if ((e = cudaMalloc(&ws, plan.workspace_bytes)) != cudaSuccess) return e;
    e = repro_softmax_fp32_ws(x, out, rows, cols, ws, plan.workspace_bytes, stream);
    if (e == cudaSuccess) e = cudaStreamSynchronize(stream);
    cudaError_t f = cudaFree(ws);
    return e != cudaSuccess ? e : f;
}

} // namespace repro_math
