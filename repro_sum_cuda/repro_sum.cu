#include "repro_sum.cuh"
#include "repro_sum_math.cuh"
#include "repro_sum_fast.cuh"
#include <cuda_runtime.h>
#include <cstdint>

namespace repro_sum {

// ---------------------------------------------------------------------------
// Rows (A == 1, moderate C): one warp per row, e.g. softmax denominators and
// LayerNorm statistics. Anchor, merge, and rounding all happen inside the warp.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(THREADS)
sum_rows_kernel(const float* __restrict__ x, float* __restrict__ out, int rows, int C, uint32_t divisor) {
    __shared__ int64_t full[WARPS][LIMBS];
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    int row = blockIdx.x * WARPS + w;
    if (row >= rows) return;                         // whole warp exits together
    if (lane < LIMBS) full[w][lane] = 0;
    const float* p = x + (size_t)row * C;
    int low = warp_max(lane < C ? value_shift(p[lane]) : -1);
    low = low < 0 ? DEFAULT_LOW : window_low_for(low);
    __syncwarp();
    FastAcc s;
    fast_init(s);
    int nf = 0;
    fast_segment<true>(s, low, p, C, lane, 32, full[w], nf);
    fast_fold(s);
    int64_t r0 = warp_sum(s.r[0]), r1 = warp_sum(s.r[1]), r2 = warp_sum(s.r[2]);
    nf = __any_sync(0xFFFFFFFFu, nf);
    __syncwarp();                                    // spills into full[w] are complete
    if (lane == 0) {
        full[w][low] += r0;
        full[w][low + 1] += r1;
        full[w][low + 2] += r2;
        out[row] = finalize(full[w], 1, divisor, nf);
    }
}

// ---------------------------------------------------------------------------
// General case (C > 1): block (b, chunk) sums elements [i0, i1) of output b's A*C
// values. All threads share one window base, from the largest of the first 32
// values, so warps merge with shuffles. With chunks == 1 the block rounds and
// writes out[b] itself.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(THREADS)
sum_general_kernel(const float* __restrict__ x, int64_t* __restrict__ part, int* __restrict__ part_nf,
                   float* __restrict__ out, int A, int B, int C, int chunks, uint32_t divisor) {
    __shared__ int64_t full[LIMBS];
    __shared__ int64_t wsum[WARPS][3];
    __shared__ int low_sh, nf_sh;
    int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    int b = blockIdx.x, chunk = blockIdx.y;
    int count = A * C;
    int i0 = (int)(((int64_t)count * chunk) / chunks);
    int i1 = (int)(((int64_t)count * (chunk + 1)) / chunks);
    if (t < LIMBS) full[t] = 0;
    if (t == 0) nf_sh = 0;
    if (warp == 0) {
        int sh = -1;
        if (i0 + lane < i1) {
            int i = i0 + lane, a = i / C, c = i - a * C;
            sh = value_shift(x[((size_t)a * B + b) * C + c]);
        }
        sh = warp_max(sh);
        if (lane == 0) low_sh = sh < 0 ? DEFAULT_LOW : window_low_for(sh);
    }
    __syncthreads();
    int low = low_sh;

    FastAcc s;
    fast_init(s);
    int nf = 0;
    if (C >= 128) {
        int a = i0 / C, c = i0 - a * C, left = i1 - i0;
        while (left > 0) {
            int len = min(C - c, left);
            fast_segment<true>(s, low, x + ((size_t)a * B + b) * C + c, len, t, THREADS, full, nf);
            left -= len;
            ++a;
            c = 0;
        }
    } else {
        for (int i = i0 + t; i < i1; i += THREADS) {
            int a = i / C, c = i - a * C;
            fast_add<true>(s, low, x[((size_t)a * B + b) * C + c], full, nf);
        }
    }
    fast_fold(s);
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        int64_t v = warp_sum(s.r[k]);
        if (lane == 0) wsum[warp][k] = v;
    }
    if (nf) nf_sh = 1;
    __syncthreads();
    if (t < 3) {
        int64_t v = 0;
        for (int k = 0; k < WARPS; ++k) v += wsum[k][t];
        full[low + t] += v;   // spills are complete (barrier above); limb low+t is ours alone
    }
    __syncthreads();
    if (chunks == 1) {
        if (t == 0) out[b] = finalize(full, 1, divisor, nf_sh);
    } else {
        size_t slot = (size_t)b * chunks + chunk;
        if (t < LIMBS) part[slot * LIMBS + t] = full[t];
        if (t == 0) part_nf[slot] = nf_sh;
    }
}

// ---------------------------------------------------------------------------
// Columns (C == 1, e.g. bias gradient: sum over rows of [A, B]): each thread owns
// output b, so consecutive threads read consecutive addresses. Four rows per
// iteration keep several loads in flight. Each thread anchors its window at its
// first values and spills to a private 10-limb array.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(THREADS)
sum_columns_kernel(const float* __restrict__ x, int64_t* __restrict__ part, int* __restrict__ part_nf,
                   float* __restrict__ out, int A, int B, int chunks, uint32_t divisor) {
    int b = blockIdx.x * THREADS + threadIdx.x, chunk = blockIdx.y;
    if (b >= B) return;
    int a0 = (int)(((int64_t)A * chunk) / chunks);
    int a1 = (int)(((int64_t)A * (chunk + 1)) / chunks);
    int64_t full[LIMBS];
#pragma unroll
    for (int l = 0; l < LIMBS; ++l) full[l] = 0;
    int sh = -1;
    for (int a = a0; a < a1 && a < a0 + 4; ++a) sh = max(sh, value_shift(x[(size_t)a * B + b]));
    int low = sh < 0 ? DEFAULT_LOW : window_low_for(sh);
    FastAcc s;
    fast_init(s);
    int nf = 0;
    int a = a0;
    for (; a + 4 <= a1; a += 4) {
        float v0 = x[(size_t)a * B + b], v1 = x[(size_t)(a + 1) * B + b];
        float v2 = x[(size_t)(a + 2) * B + b], v3 = x[(size_t)(a + 3) * B + b];
        fast_add<false>(s, low, v0, full, nf);
        fast_add<false>(s, low, v1, full, nf);
        fast_add<false>(s, low, v2, full, nf);
        fast_add<false>(s, low, v3, full, nf);
    }
    for (; a < a1; ++a) fast_add<false>(s, low, x[(size_t)a * B + b], full, nf);
    fast_fold(s);
    full[low] += s.r[0];
    full[low + 1] += s.r[1];
    full[low + 2] += s.r[2];
    if (chunks == 1) {
        out[b] = finalize(full, 1, divisor, nf);
    } else {
        size_t slot = (size_t)b * chunks + chunk;
#pragma unroll
        for (int l = 0; l < LIMBS; ++l) part[slot * LIMBS + l] = full[l];
        part_nf[slot] = nf;
    }
}

// One thread per output: merge the chunk accumulators exactly, then round once.
__global__ void finalize_kernel(const int64_t* __restrict__ part, const int* __restrict__ part_nf,
                                float* __restrict__ out, int B, int chunks, uint32_t divisor) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    int64_t tot[LIMBS] = {};
    int nf = 0;
    for (int k = 0; k < chunks; ++k) {
        size_t slot = (size_t)b * chunks + k;
        for (int l = 0; l < LIMBS; ++l) tot[l] += part[slot * LIMBS + l];
        nf |= part_nf[slot];
    }
    out[b] = finalize(tot, 1, divisor, nf);
}

namespace {

constexpr size_t ALIGN = 256;
constexpr int ROWS_MAX_C = 8192;     // rows kernel handles rows up to this length
size_t align_up(size_t x) { return (x + ALIGN - 1) / ALIGN * ALIGN; }

enum class Kind { Rows, Columns, General };

Kind choose_kind(int A, int B, int C) {
    if (C == 1) return Kind::Columns;
    if (A == 1 && C <= ROWS_MAX_C && B >= 256) return Kind::Rows;
    return Kind::General;
}

cudaError_t layout(int A, int B, int C, SumConfig cfg, SumPlan& plan, size_t& nf_offset) {
    if (A <= 0 || B <= 0 || C <= 0 || cfg.chunks < 0) return cudaErrorInvalidValue;
    if ((int64_t)A * C > INT32_MAX) return cudaErrorInvalidValue;
    int count = A * C;
    Kind kind = choose_kind(A, B, C);
    int chunks = 1;
    if (kind != Kind::Rows) {
        int max_chunks = kind == Kind::Columns ? A : count;
        if (max_chunks > 65535) max_chunks = 65535;  // grid y limit
        chunks = cfg.chunks;
        if (chunks == 0) {
            int dev, sms;
            cudaError_t e;
            if ((e = cudaGetDevice(&dev)) != cudaSuccess) return e;
            if ((e = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev)) != cudaSuccess) return e;
            int target_blocks = 4 * sms;
            if (kind == Kind::Columns) {
                int bx = (B + THREADS - 1) / THREADS;
                chunks = (target_blocks + bx - 1) / bx;
                int per = A / 64;                // at least ~64 rows per chunk
                if (chunks > per) chunks = per;
            } else {
                chunks = (target_blocks + B - 1) / B;
                int per = count / 8192;          // at least ~8192 elements per chunk
                if (chunks > per) chunks = per;
            }
            if (chunks < 1) chunks = 1;
        }
        if (chunks > max_chunks) chunks = max_chunks;
    }
    plan.chunks = chunks;
    size_t slots = chunks > 1 ? (size_t)B * chunks : 0;  // chunks == 1 needs no workspace
    nf_offset = align_up(slots * LIMBS * sizeof(int64_t));
    plan.workspace_bytes = nf_offset + align_up(slots * sizeof(int));
    return cudaSuccess;
}

} // namespace

cudaError_t make_sum_plan(int A, int B, int C, SumConfig cfg, SumPlan* plan) {
    if (!plan) return cudaErrorInvalidValue;
    size_t nf_offset;
    return layout(A, B, C, cfg, *plan, nf_offset);
}

cudaError_t repro_sum_fp32_ws(const float* x, float* out, int A, int B, int C, SumConfig cfg,
                              void* workspace, size_t workspace_bytes, cudaStream_t stream) {
    if (!x || !out) return cudaErrorInvalidValue;
    SumPlan plan;
    size_t nf_offset;
    cudaError_t e = layout(A, B, C, cfg, plan, nf_offset);
    if (e != cudaSuccess) return e;
    if (plan.workspace_bytes && (!workspace || workspace_bytes < plan.workspace_bytes)) return cudaErrorInvalidValue;
    int64_t* part = static_cast<int64_t*>(workspace);
    int* part_nf = plan.workspace_bytes ? reinterpret_cast<int*>(static_cast<char*>(workspace) + nf_offset) : nullptr;
    int chunks = plan.chunks;
    uint32_t divisor = cfg.mean ? (uint32_t)(A * C) : 1u;
    switch (choose_kind(A, B, C)) {
        case Kind::Rows:
            sum_rows_kernel<<<(B + WARPS - 1) / WARPS, THREADS, 0, stream>>>(x, out, B, C, divisor);
            break;
        case Kind::Columns:
            sum_columns_kernel<<<dim3((B + THREADS - 1) / THREADS, chunks), THREADS, 0, stream>>>(
                x, part, part_nf, out, A, B, chunks, divisor);
            break;
        case Kind::General:
            sum_general_kernel<<<dim3(B, chunks), THREADS, 0, stream>>>(x, part, part_nf, out, A, B, C, chunks, divisor);
            break;
    }
    if (chunks > 1)
        finalize_kernel<<<(B + 127) / 128, 128, 0, stream>>>(part, part_nf, out, B, chunks, divisor);
    return cudaGetLastError();
}

cudaError_t repro_sum_fp32(const float* x, float* out, int A, int B, int C, SumConfig cfg, cudaStream_t stream) {
    SumPlan plan;
    cudaError_t e = make_sum_plan(A, B, C, cfg, &plan);
    if (e != cudaSuccess) return e;
    void* ws = nullptr;
    if (plan.workspace_bytes && (e = cudaMalloc(&ws, plan.workspace_bytes)) != cudaSuccess) return e;
    e = repro_sum_fp32_ws(x, out, A, B, C, cfg, ws, plan.workspace_bytes, stream);
    if (e == cudaSuccess) e = cudaStreamSynchronize(stream);
    if (ws) cudaFree(ws);
    return e;
}

} // namespace repro_sum
