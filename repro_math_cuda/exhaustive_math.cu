// Exhaustive cross-GPU consistency test for float32 math functions.
//
// Evaluates each function on all 2^32 float32 bit patterns and prints:
//   - a hash of all outputs (equal hashes on two GPUs => identical results on
//     every input, up to a negligible collision probability), and a second hash
//     "<name>~nan" with every NaN output replaced by one canonical pattern, which
//     separates real numeric differences from implementation-defined NaN bits;
//   - how many outputs differ from a double-precision reference, and the
//     largest difference in ulps (accuracy, not reproducibility);
// and writes one hash per block of 2^20 inputs, so two GPUs that disagree can be
// compared block by block (compare_math.sh) and then input by input (dump mode).
//
// Usage:
//   exhaustive_math [out_file] [func ...]      run (default: all functions)
//   exhaustive_math --dump <func> <block>      print every input/output of one block
//   exhaustive_math --list                     list function names
//
// Build with default nvcc flags (no --use_fast_math), like PyTorch.
#include <cuda_runtime.h>
#include "repro_math.cuh"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); std::exit(1); } } while (0)

enum Func {
    F_EXPF, F_EXPF_FAST, F_EXP_VIA_DOUBLE, F_LOGF, F_LOG1PF, F_EXPM1F, F_TANHF, F_ERFF,
    F_SIGMOID_TORCH, F_SQRTF, F_RSQRTF, F_RECIP, F_DIV10, F_MUL_POINT1, F_REPRO_EXP, F_REPRO_LOG, F_COUNT
};

static const char* kNames[F_COUNT] = {
    "expf", "__expf", "exp_via_double", "logf", "log1pf", "expm1f", "tanhf", "erff",
    "sigmoid_torch", "sqrtf", "rsqrtf", "recip_1/x", "div_x/10", "mul_x*0.1f",
    "repro_expf", "repro_logf",
};

static const char* kNotes[F_COUNT] = {
    "CUDA expf (PyTorch torch.exp)",
    "fast-math intrinsic (approximate)",
    "RepDL style: (float)exp((double)x)",
    "CUDA logf (torch.log)",
    "CUDA log1pf",
    "CUDA expm1f",
    "CUDA tanhf (torch.tanh)",
    "CUDA erff (exact GELU)",
    "PyTorch formula: 1/(1+expf(-x))",
    "IEEE sqrt (should be exact)",
    "approximate reciprocal sqrt",
    "IEEE division 1/x (should be exact)",
    "IEEE division x/10 (should be exact)",
    "x * 0.1f: what 'x/10' becomes if division is turned into multiplication",
    "ours: correctly rounded exp",
    "ours: correctly rounded log",
};

template<int F>
__device__ __forceinline__ float eval(float x) {
    switch (F) {
        case F_EXPF: return expf(x);
        case F_EXPF_FAST: return __expf(x);
        case F_EXP_VIA_DOUBLE: return (float)exp((double)x);
        case F_LOGF: return logf(x);
        case F_LOG1PF: return log1pf(x);
        case F_EXPM1F: return expm1f(x);
        case F_TANHF: return tanhf(x);
        case F_ERFF: return erff(x);
        case F_SIGMOID_TORCH: return 1.0f / (1.0f + expf(-x));
        case F_SQRTF: return sqrtf(x);
        case F_RSQRTF: return rsqrtf(x);
        case F_RECIP: return 1.0f / x;
        case F_DIV10: return x / 10.0f;
        case F_MUL_POINT1: return x * 0.1f;
        case F_REPRO_EXP: return repro_math::repro_expf(x);
        case F_REPRO_LOG: return repro_math::repro_logf(x);
    }
    return 0.0f;
}

// High-precision reference: computed in double, rounded once to float.
template<int F>
__device__ __forceinline__ float reference(float xf) {
    double x = xf;
    switch (F) {
        case F_EXPF: case F_EXPF_FAST: case F_EXP_VIA_DOUBLE: case F_REPRO_EXP: return (float)exp(x);
        case F_LOGF: case F_REPRO_LOG: return (float)log(x);
        case F_LOG1PF: return (float)log1p(x);
        case F_EXPM1F: return (float)expm1(x);
        case F_TANHF: return (float)tanh(x);
        case F_ERFF: return (float)erf(x);
        case F_SIGMOID_TORCH: return (float)(1.0 / (1.0 + exp(-x)));
        case F_SQRTF: return (float)sqrt(x);
        case F_RSQRTF: return (float)(1.0 / sqrt(x));
        case F_RECIP: return (float)(1.0 / x);
        case F_DIV10: case F_MUL_POINT1: return (float)(x / 10.0);
    }
    return 0.0f;
}

__device__ __forceinline__ uint64_t mix64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

// Float bits -> integers ordered like the floats, so differences count ulps.
__device__ __forceinline__ int64_t ordered(uint32_t u) {
    return (u & 0x80000000u) ? -(int64_t)(u & 0x7FFFFFFFu) : (int64_t)u;
}

constexpr int BLOCK_BITS = 20;                 // 2^20 inputs per hash block
constexpr int NBLOCKS = 1 << (32 - BLOCK_BITS);
constexpr int THREADS = 256;

struct Stats {
    unsigned long long mismatches;  // output != reference (both non-NaN, or exactly one NaN)
    unsigned long long max_ulp;
};

// CUDA block k covers inputs [k * 2^20, (k+1) * 2^20). Hash = sum over inputs of
// mix64(input << 32 | output): order-independent integer sum, so deterministic.
constexpr uint32_t CANON_NAN = 0x7FFFFFFFu;

template<int F>
__global__ void __launch_bounds__(THREADS) run_kernel(unsigned long long* block_hash, unsigned long long* block_hash_nan,
                                                      Stats* stats) {
    __shared__ unsigned long long sh[THREADS / 32], sn[THREADS / 32], sm[THREADS / 32], su[THREADS / 32];
    uint32_t base = (uint32_t)blockIdx.x << BLOCK_BITS;
    unsigned long long h = 0, hn = 0, mism = 0, maxu = 0;
    for (uint32_t j = threadIdx.x; j < (1u << BLOCK_BITS); j += THREADS) {
        uint32_t in = base + j;
        float x = __uint_as_float(in);
        uint32_t out = __float_as_uint(eval<F>(x));
        h += mix64(((uint64_t)in << 32) | out);
        uint32_t ref = __float_as_uint(reference<F>(x));
        bool nan_o = (out & 0x7FFFFFFFu) > 0x7F800000u, nan_r = (ref & 0x7FFFFFFFu) > 0x7F800000u;
        hn += mix64(((uint64_t)in << 32) | (nan_o ? CANON_NAN : out));
        if (nan_o != nan_r) {
            ++mism;
        } else if (!nan_o && out != ref) {
            ++mism;
            int64_t d = ordered(out) - ordered(ref);
            unsigned long long u = (unsigned long long)(d < 0 ? -d : d);
            if (u > maxu) maxu = u;
        }
    }
    for (int o = 16; o > 0; o >>= 1) {
        h += __shfl_down_sync(0xFFFFFFFFu, h, o);
        hn += __shfl_down_sync(0xFFFFFFFFu, hn, o);
        mism += __shfl_down_sync(0xFFFFFFFFu, mism, o);
        maxu = max(maxu, __shfl_down_sync(0xFFFFFFFFu, maxu, o));
    }
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    if (lane == 0) { sh[w] = h; sn[w] = hn; sm[w] = mism; su[w] = maxu; }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int k = 1; k < THREADS / 32; ++k) { sh[0] += sh[k]; sn[0] += sn[k]; sm[0] += sm[k]; su[0] = max(su[0], su[k]); }
        block_hash[blockIdx.x] = sh[0];
        block_hash_nan[blockIdx.x] = sn[0];
        atomicAdd(&stats->mismatches, sm[0]);
        atomicMax(&stats->max_ulp, su[0]);
    }
}

template<int F>
__global__ void dump_kernel(uint32_t base, uint32_t* out) {
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < (1u << BLOCK_BITS)) out[j] = __float_as_uint(eval<F>(__uint_as_float(base + j)));
}

template<int F>
static void launch(unsigned long long* d_hash, unsigned long long* d_hash_nan, Stats* d_stats) {
    run_kernel<F><<<NBLOCKS, THREADS>>>(d_hash, d_hash_nan, d_stats);
}

template<int F>
static void launch_dump(uint32_t base, uint32_t* d_out) {
    dump_kernel<F><<<(1u << BLOCK_BITS) / 256, 256>>>(base, d_out);
}

static void dispatch(int f, unsigned long long* h, unsigned long long* hn, Stats* s) {
    switch (f) {
        case F_EXPF: launch<F_EXPF>(h, hn, s); break;
        case F_EXPF_FAST: launch<F_EXPF_FAST>(h, hn, s); break;
        case F_EXP_VIA_DOUBLE: launch<F_EXP_VIA_DOUBLE>(h, hn, s); break;
        case F_LOGF: launch<F_LOGF>(h, hn, s); break;
        case F_LOG1PF: launch<F_LOG1PF>(h, hn, s); break;
        case F_EXPM1F: launch<F_EXPM1F>(h, hn, s); break;
        case F_TANHF: launch<F_TANHF>(h, hn, s); break;
        case F_ERFF: launch<F_ERFF>(h, hn, s); break;
        case F_SIGMOID_TORCH: launch<F_SIGMOID_TORCH>(h, hn, s); break;
        case F_SQRTF: launch<F_SQRTF>(h, hn, s); break;
        case F_RSQRTF: launch<F_RSQRTF>(h, hn, s); break;
        case F_RECIP: launch<F_RECIP>(h, hn, s); break;
        case F_DIV10: launch<F_DIV10>(h, hn, s); break;
        case F_MUL_POINT1: launch<F_MUL_POINT1>(h, hn, s); break;
        case F_REPRO_EXP: launch<F_REPRO_EXP>(h, hn, s); break;
        case F_REPRO_LOG: launch<F_REPRO_LOG>(h, hn, s); break;
    }
}

static void dispatch_dump(int f, uint32_t base, uint32_t* o) {
    switch (f) {
        case F_EXPF: launch_dump<F_EXPF>(base, o); break;
        case F_EXPF_FAST: launch_dump<F_EXPF_FAST>(base, o); break;
        case F_EXP_VIA_DOUBLE: launch_dump<F_EXP_VIA_DOUBLE>(base, o); break;
        case F_LOGF: launch_dump<F_LOGF>(base, o); break;
        case F_LOG1PF: launch_dump<F_LOG1PF>(base, o); break;
        case F_EXPM1F: launch_dump<F_EXPM1F>(base, o); break;
        case F_TANHF: launch_dump<F_TANHF>(base, o); break;
        case F_ERFF: launch_dump<F_ERFF>(base, o); break;
        case F_SIGMOID_TORCH: launch_dump<F_SIGMOID_TORCH>(base, o); break;
        case F_SQRTF: launch_dump<F_SQRTF>(base, o); break;
        case F_RSQRTF: launch_dump<F_RSQRTF>(base, o); break;
        case F_RECIP: launch_dump<F_RECIP>(base, o); break;
        case F_DIV10: launch_dump<F_DIV10>(base, o); break;
        case F_MUL_POINT1: launch_dump<F_MUL_POINT1>(base, o); break;
        case F_REPRO_EXP: launch_dump<F_REPRO_EXP>(base, o); break;
        case F_REPRO_LOG: launch_dump<F_REPRO_LOG>(base, o); break;
    }
}

static int find_func(const char* name) {
    for (int f = 0; f < F_COUNT; ++f) if (std::strcmp(name, kNames[f]) == 0) return f;
    std::fprintf(stderr, "unknown function '%s' (see --list)\n", name);
    std::exit(1);
}

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--list") == 0) {
        for (int f = 0; f < F_COUNT; ++f) std::printf("%-16s %s\n", kNames[f], kNotes[f]);
        return 0;
    }
    if (argc > 1 && std::strcmp(argv[1], "--dump") == 0) {
        if (argc < 4) { std::fprintf(stderr, "usage: --dump <func> <block 0..%d>\n", NBLOCKS - 1); return 1; }
        int f = find_func(argv[2]);
        uint32_t block = (uint32_t)std::strtoul(argv[3], nullptr, 10);
        uint32_t* d_out;
        CK(cudaMalloc(&d_out, sizeof(uint32_t) << BLOCK_BITS));
        dispatch_dump(f, block << BLOCK_BITS, d_out);
        CK(cudaGetLastError());
        std::vector<uint32_t> out(1u << BLOCK_BITS);
        CK(cudaMemcpy(out.data(), d_out, sizeof(uint32_t) << BLOCK_BITS, cudaMemcpyDeviceToHost));
        for (uint32_t j = 0; j < (1u << BLOCK_BITS); ++j) {
            uint32_t in = (block << BLOCK_BITS) + j;
            float x, y;
            std::memcpy(&x, &in, 4); std::memcpy(&y, &out[j], 4);
            std::printf("%08x %08x  %.9g -> %.9g\n", in, out[j], x, y);
        }
        return 0;
    }

    const char* out_file = argc > 1 ? argv[1] : "math_hashes.txt";
    std::vector<int> funcs;
    for (int i = 2; i < argc; ++i) funcs.push_back(find_func(argv[i]));
    if (funcs.empty()) for (int f = 0; f < F_COUNT; ++f) funcs.push_back(f);

    cudaDeviceProp prop;
    int dev = 0, rt = 0, drv = 0;
    CK(cudaGetDevice(&dev));
    CK(cudaGetDeviceProperties(&prop, dev));
    cudaRuntimeGetVersion(&rt);
    cudaDriverGetVersion(&drv);
    FILE* fo = std::fopen(out_file, "w");
    if (!fo) { std::perror(out_file); return 1; }
    std::printf("device: %s (sm_%d%d)  CUDA runtime %d  driver %d\n", prop.name, prop.major, prop.minor, rt, drv);
    std::fprintf(fo, "# device %s sm_%d%d runtime %d driver %d\n", prop.name, prop.major, prop.minor, rt, drv);
    std::printf("%-16s %-18s %14s %9s %8s   %s\n", "function", "hash(all 2^32)", "!=reference", "(%)", "max_ulp", "note");

    unsigned long long *d_hash, *d_hash_nan;
    Stats* d_stats;
    CK(cudaMalloc(&d_hash, sizeof(unsigned long long) * NBLOCKS));
    CK(cudaMalloc(&d_hash_nan, sizeof(unsigned long long) * NBLOCKS));
    CK(cudaMalloc(&d_stats, sizeof(Stats)));
    std::vector<unsigned long long> hash(NBLOCKS), hash_nan(NBLOCKS);
    for (int f : funcs) {
        CK(cudaMemset(d_stats, 0, sizeof(Stats)));
        dispatch(f, d_hash, d_hash_nan, d_stats);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        Stats st;
        CK(cudaMemcpy(hash.data(), d_hash, sizeof(unsigned long long) * NBLOCKS, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hash_nan.data(), d_hash_nan, sizeof(unsigned long long) * NBLOCKS, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(&st, d_stats, sizeof(Stats), cudaMemcpyDeviceToHost));
        unsigned long long total = 0, total_nan = 0;
        for (int k = 0; k < NBLOCKS; ++k) {
            total += hash[k];
            total_nan += hash_nan[k];
            std::fprintf(fo, "B %s %d %016llx\n", kNames[f], k, hash[k]);
            std::fprintf(fo, "B %s~nan %d %016llx\n", kNames[f], k, hash_nan[k]);
        }
        std::fprintf(fo, "SUMMARY %s %016llx %llu %llu\n", kNames[f], total, st.mismatches, st.max_ulp);
        std::fprintf(fo, "SUMMARY %s~nan %016llx %llu %llu\n", kNames[f], total_nan, st.mismatches, st.max_ulp);
        std::printf("%-16s %016llx   %14llu %8.4f%% %8llu   %s\n", kNames[f], total, st.mismatches,
                    100.0 * (double)st.mismatches / 4294967296.0, st.max_ulp, kNotes[f]);
        std::fflush(stdout);
    }
    std::fclose(fo);
    std::printf("block hashes written to %s\n", out_file);
    return 0;
}
