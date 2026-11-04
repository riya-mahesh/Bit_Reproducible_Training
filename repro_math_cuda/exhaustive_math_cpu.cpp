// CPU counterpart of exhaustive_math.cu: evaluates the same functions with the
// C math library (glibc on Linux) on all 2^32 float32 inputs, multi-threaded, and
// writes results in the same format, so compare_math.sh can compare CPU and GPU.
//
// Usage:
//   exhaustive_math_cpu [out_file] [func ...]     (default: all functions)
//   exhaustive_math_cpu --dump <func> <block>
//
// Notes:
//  - __expf (a CUDA intrinsic) has no CPU equivalent and is skipped.
//  - rsqrtf has no standard CPU function; it is computed as 1.0f / sqrtf(x),
//    which is what PyTorch's CPU rsqrt does. It is expected to differ from CUDA's
//    approximate rsqrtf.
//  - This is the C library. PyTorch's CPU kernels mostly use their own vectorized
//    math (SLEEF), which is a different implementation again.
//  - Build without fast-math or FP contraction: see run_math_cpu.sh.
#include "repro_math.cuh"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#ifdef __GLIBC__
#include <gnu/libc-version.h>
#endif

enum Func {
    F_EXPF, F_EXP_VIA_DOUBLE, F_LOGF, F_LOG1PF, F_EXPM1F, F_TANHF, F_ERFF,
    F_SIGMOID_TORCH, F_SQRTF, F_RSQRTF, F_RECIP, F_DIV10, F_MUL_POINT1, F_REPRO_EXP, F_REPRO_LOG, F_COUNT
};

// Names match exhaustive_math.cu so compare_math.sh lines them up.
static const char* kNames[F_COUNT] = {
    "expf", "exp_via_double", "logf", "log1pf", "expm1f", "tanhf", "erff",
    "sigmoid_torch", "sqrtf", "rsqrtf", "recip_1/x", "div_x/10", "mul_x*0.1f",
    "repro_expf", "repro_logf",
};

static const char* kNotes[F_COUNT] = {
    "C library expf",
    "RepDL style: (float)exp((double)x)",
    "C library logf",
    "C library log1pf",
    "C library expm1f",
    "C library tanhf",
    "C library erff",
    "PyTorch formula: 1/(1+expf(-x))",
    "IEEE sqrt (should be exact)",
    "1.0f/sqrtf(x) (PyTorch CPU rsqrt; CUDA's rsqrtf is approximate)",
    "IEEE division 1/x (should be exact)",
    "IEEE division x/10 (should be exact)",
    "x * 0.1f",
    "ours: correctly rounded exp",
    "ours: correctly rounded log",
};

// volatile-free, contraction-free evaluation (built with -ffp-contract=off).
static float eval(int f, float x) {
    switch (f) {
        case F_EXPF: return expf(x);
        case F_EXP_VIA_DOUBLE: return (float)exp((double)x);
        case F_LOGF: return logf(x);
        case F_LOG1PF: return log1pf(x);
        case F_EXPM1F: return expm1f(x);
        case F_TANHF: return tanhf(x);
        case F_ERFF: return erff(x);
        case F_SIGMOID_TORCH: return 1.0f / (1.0f + expf(-x));
        case F_SQRTF: return sqrtf(x);
        case F_RSQRTF: return 1.0f / sqrtf(x);
        case F_RECIP: return 1.0f / x;
        case F_DIV10: return x / 10.0f;
        case F_MUL_POINT1: return x * 0.1f;
        case F_REPRO_EXP: return repro_math::repro_expf(x);
        case F_REPRO_LOG: return repro_math::repro_logf(x);
    }
    return 0.0f;
}

// Same double-precision reference as the GPU program, using this machine's libm.
static float reference(int f, float xf) {
    double x = xf;
    switch (f) {
        case F_EXPF: case F_EXP_VIA_DOUBLE: case F_REPRO_EXP: return (float)exp(x);
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

static inline uint32_t bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static inline float from_bits(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

// Identical to the GPU program's hash.
static inline uint64_t mix64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

static inline int64_t ordered(uint32_t u) {
    return (u & 0x80000000u) ? -(int64_t)(u & 0x7FFFFFFFu) : (int64_t)u;
}

constexpr int BLOCK_BITS = 20;
constexpr int NBLOCKS = 1 << (32 - BLOCK_BITS);

struct BlockResult { uint64_t hash, hash_nan, mismatches, max_ulp; };
constexpr uint32_t CANON_NAN = 0x7FFFFFFFu;   // same canonical NaN as the GPU program

static BlockResult run_block(int f, uint32_t block) {
    BlockResult r{0, 0, 0, 0};
    uint32_t base = block << BLOCK_BITS;
    for (uint32_t j = 0; j < (1u << BLOCK_BITS); ++j) {
        uint32_t in = base + j;
        float x = from_bits(in);
        uint32_t out = bits(eval(f, x));
        r.hash += mix64(((uint64_t)in << 32) | out);
        uint32_t ref = bits(reference(f, x));
        bool nan_o = (out & 0x7FFFFFFFu) > 0x7F800000u, nan_r = (ref & 0x7FFFFFFFu) > 0x7F800000u;
        r.hash_nan += mix64(((uint64_t)in << 32) | (nan_o ? CANON_NAN : out));
        if (nan_o != nan_r) {
            ++r.mismatches;
        } else if (!nan_o && out != ref) {
            ++r.mismatches;
            int64_t d = ordered(out) - ordered(ref);
            uint64_t u = (uint64_t)(d < 0 ? -d : d);
            if (u > r.max_ulp) r.max_ulp = u;
        }
    }
    return r;
}

static int find_func(const char* name) {
    for (int f = 0; f < F_COUNT; ++f) if (std::strcmp(name, kNames[f]) == 0) return f;
    std::fprintf(stderr, "unknown or GPU-only function '%s'\n", name);
    std::exit(1);
}

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--dump") == 0) {
        if (argc < 4) { std::fprintf(stderr, "usage: --dump <func> <block>\n"); return 1; }
        int f = find_func(argv[2]);
        uint32_t block = (uint32_t)std::strtoul(argv[3], nullptr, 10);
        for (uint32_t j = 0; j < (1u << BLOCK_BITS); ++j) {
            uint32_t in = (block << BLOCK_BITS) + j;
            float y = eval(f, from_bits(in));
            std::printf("%08x %08x  %.9g -> %.9g\n", in, bits(y), from_bits(in), y);
        }
        return 0;
    }

    const char* out_file = argc > 1 ? argv[1] : "math_hashes_cpu.txt";
    std::vector<int> funcs;
    for (int i = 2; i < argc; ++i) funcs.push_back(find_func(argv[i]));
    if (funcs.empty()) for (int f = 0; f < F_COUNT; ++f) funcs.push_back(f);

    std::string libc = "unknown libm";
#ifdef __GLIBC__
    libc = std::string("glibc ") + gnu_get_libc_version();
#elif defined(__APPLE__)
    libc = "Apple libm";
#endif
    unsigned nthreads = std::max(1u, std::thread::hardware_concurrency());
    FILE* fo = std::fopen(out_file, "w");
    if (!fo) { std::perror(out_file); return 1; }
    std::printf("device: CPU (%s), %u threads\n", libc.c_str(), nthreads);
    std::fprintf(fo, "# device CPU %s threads %u\n", libc.c_str(), nthreads);
    std::printf("%-16s %-18s %14s %9s %8s   %s\n", "function", "hash(all 2^32)", "!=reference", "(%)", "max_ulp", "note");

    std::vector<BlockResult> res(NBLOCKS);
    for (int f : funcs) {
        std::atomic<int> next{0};
        std::vector<std::thread> pool;
        for (unsigned t = 0; t < nthreads; ++t)
            pool.emplace_back([&] {
                for (int k; (k = next.fetch_add(1)) < NBLOCKS;) res[k] = run_block(f, (uint32_t)k);
            });
        for (auto& th : pool) th.join();
        uint64_t total = 0, total_nan = 0, mism = 0, maxu = 0;
        for (int k = 0; k < NBLOCKS; ++k) {
            total += res[k].hash;
            total_nan += res[k].hash_nan;
            mism += res[k].mismatches;
            maxu = std::max(maxu, res[k].max_ulp);
            std::fprintf(fo, "B %s %d %016llx\n", kNames[f], k, (unsigned long long)res[k].hash);
            std::fprintf(fo, "B %s~nan %d %016llx\n", kNames[f], k, (unsigned long long)res[k].hash_nan);
        }
        std::fprintf(fo, "SUMMARY %s %016llx %llu %llu\n", kNames[f], (unsigned long long)total,
                     (unsigned long long)mism, (unsigned long long)maxu);
        std::fprintf(fo, "SUMMARY %s~nan %016llx %llu %llu\n", kNames[f], (unsigned long long)total_nan,
                     (unsigned long long)mism, (unsigned long long)maxu);
        std::printf("%-16s %016llx   %14llu %8.4f%% %8llu   %s\n", kNames[f], (unsigned long long)total,
                    (unsigned long long)mism, 100.0 * (double)mism / 4294967296.0, (unsigned long long)maxu, kNotes[f]);
        std::fflush(stdout);
    }
    std::fclose(fo);
    std::printf("block hashes written to %s\n", out_file);
    return 0;
}
