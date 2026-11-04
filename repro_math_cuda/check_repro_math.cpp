// Proof of repro_expf and repro_logf on the CPU: runs all 2^32 float32 inputs and checks
//   1. the result equals the double-double path rounded to float32 (our own reference);
//   2. the result equals CORE-MATH (an independent library), when built with -DWITH_CORE_MATH;
//   3. the hard cases and special values;
// With a file argument, also writes every slow-path input and every 4096th input to that
// file, for check_with_mpmath.py to recompute independently in 300-bit arithmetic.
// and reports the largest actual error of the fast path, which must stay below the bound
// in the rounding test, and how often the slow path runs.
//
// Build: see run_check_math.sh. Takes several minutes on a laptop.
#include "repro_math.cuh"
#include "hard_cases_table.h"
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <thread>
#include <vector>

#ifdef WITH_CORE_MATH
extern "C" float cr_expf(float);
extern "C" float cr_logf(float);
#endif

using namespace repro_math;

struct Tally {
    uint64_t inputs = 0, wrong_dd = 0, wrong_cm = 0, slow = 0;
    double max_err = 0;     // largest fast-path error, absolute on y (exp) or relative (log)
    uint32_t argmax = 0;
    uint32_t first_wrong = 0;
    std::vector<std::pair<uint32_t, uint32_t>> dump;   // input bits, output bits
    void merge(const Tally& o) {
        inputs += o.inputs; wrong_dd += o.wrong_dd; wrong_cm += o.wrong_cm; slow += o.slow;
        if (o.max_err > max_err) { max_err = o.max_err; argmax = o.argmax; }
        if (!first_wrong) first_wrong = o.first_wrong;
        dump.insert(dump.end(), o.dump.begin(), o.dump.end());
    }
};

static bool same(float a, float b) {
    if (std::isnan(a) && std::isnan(b)) return true;   // NaN bits are checked separately
    return bits_of(a) == bits_of(b);
}

static double dd_minus(dd v, ff y, double scale) {     // |v / scale - y|
    double a = (v.hi / scale - (double)y.hi) + (v.lo / scale - (double)y.lo);
    return std::fabs(a);
}

template<class F>
static Tally for_all(F f) {
    unsigned nt = std::max(1u, std::thread::hardware_concurrency());
    std::vector<Tally> part(nt);
    std::atomic<uint32_t> next{0};
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < nt; ++t)
        pool.emplace_back([&, t] {
            for (uint32_t blk; (blk = next.fetch_add(1)) < 4096;)
                for (uint32_t j = 0; j < (1u << 20); ++j) f((blk << 20) | j, part[t]);
        });
    for (auto& th : pool) th.join();
    Tally out;
    for (auto& p : part) out.merge(p);
    return out;
}

static void check_exp(uint32_t b, Tally& T) {
    float x = float_of(b);
    float out = repro_expf(x);
    ++T.inputs;
    if (std::isnan(x)) { if (bits_of(out) != 0x7FC00000u) { ++T.wrong_dd; T.first_wrong = b; } return; }
#ifdef WITH_CORE_MATH
    if (!same(out, cr_expf(x))) { ++T.wrong_cm; if (!T.first_wrong) T.first_wrong = b; }
#endif
    if (x >= REPRO_EXP_OVERFLOW || x <= REPRO_EXP_UNDERFLOW) return;
    int m;
    ff y = exp_core(x, m);
    dd v = dd_exp((double)x);
    if (!same(out, dd_to_float(v))) { ++T.wrong_dd; if (!T.first_wrong) T.first_wrong = b; }
    double err = dd_minus(v, y, std::ldexp(1.0, m));
    if (err > T.max_err) { T.max_err = err; T.argmax = b; }
    bool slow;
    if (m <= -126 && (m < -126 || y.hi < 1.0f || (y.hi == 1.0f && y.lo < 0.0f))) {
        float o;
        slow = !exp_subnormal(y, m, o);
    } else {
        slow = fadd(y.hi, fadd(y.lo, EXP_DELTA)) != fadd(y.hi, fsub(y.lo, EXP_DELTA));
    }
    T.slow += slow;
    if (slow || (b & 0xFFFu) == 0x5A5u) T.dump.push_back({b, bits_of(out)});
}

static void check_log(uint32_t b, Tally& T) {
    float x = float_of(b);
    float out = repro_logf(x);
    ++T.inputs;
    if (std::isnan(out)) {
        bool ok = bits_of(out) == 0x7FC00000u && (std::isnan(x) || x < 0.0f);
        if (!ok) { ++T.wrong_dd; T.first_wrong = b; }
        return;
    }
#ifdef WITH_CORE_MATH
    if (!same(out, cr_logf(x))) { ++T.wrong_cm; if (!T.first_wrong) T.first_wrong = b; }
#endif
    if (!(x > 0.0f) || std::isinf(x) || x == 1.0f) return;
    ff y = log_core(x);
    dd v = dd_log_from((double)x, std::log((double)x));   // independent starting point
    if (!same(out, dd_to_float(v))) { ++T.wrong_dd; if (!T.first_wrong) T.first_wrong = b; }
    double err = dd_minus(v, y, 1.0) / std::fabs(v.hi);
    if (err > T.max_err) { T.max_err = err; T.argmax = b; }
    float eps = fmul(LOG_EPS, abs_of(y.hi));
    bool slow = fadd(y.hi, fadd(y.lo, eps)) != fadd(y.hi, fsub(y.lo, eps));
    T.slow += slow;
    if (slow || (b & 0xFFFu) == 0x5A5u) T.dump.push_back({b, bits_of(out)});
}

static void report(const char* name, const Tally& T, double bound) {
    std::printf("\n%s: %llu inputs\n", name, (unsigned long long)T.inputs);
    std::printf("  wrong vs double-double reference: %llu\n", (unsigned long long)T.wrong_dd);
#ifdef WITH_CORE_MATH
    std::printf("  wrong vs CORE-MATH:               %llu\n", (unsigned long long)T.wrong_cm);
#else
    std::printf("  CORE-MATH comparison:             not built\n");
#endif
    if (T.first_wrong) std::printf("  first wrong input bits: %08x\n", T.first_wrong);
    std::printf("  fast path max error: 2^%.2f at bits %08x; bound in the rounding test: 2^%.2f  (%s)\n",
                std::log2(T.max_err), T.argmax, std::log2(bound), T.max_err < bound ? "ok" : "BOUND TOO SMALL");
    std::printf("  slow path: %llu inputs, 1 in %.0f\n", (unsigned long long)T.slow,
                T.slow ? (double)T.inputs / (double)T.slow : 0.0);
}

int main(int argc, char** argv) {
    int bad = 0;
    for (const HardCase& h : kHardCases) {
        float x = float_of(h.bits);
        float out = h.is_log ? repro_logf(x) : repro_expf(x);
        if (bits_of(out) != bits_of(h.correct)) {
            std::printf("hard case %08x: got %a, correct %a\n", h.bits, out, h.correct);
            ++bad;
        }
    }
    std::printf("hard cases: %d of %d wrong\n", bad, (int)(sizeof(kHardCases) / sizeof(kHardCases[0])));
    struct { float x, e, l; } sp[] = {
        {0.0f, 1.0f, -INFINITY}, {-0.0f, 1.0f, -INFINITY}, {1.0f, 0x1.5bf0a8p+1f, 0.0f},
        {INFINITY, INFINITY, INFINITY}, {-INFINITY, 0.0f, NAN},
    };
    for (auto& s : sp) {
        if (!same(repro_expf(s.x), s.e)) { std::printf("exp(%g) wrong\n", s.x); ++bad; }
        if (!same(repro_logf(s.x), s.l)) { std::printf("log(%g) wrong\n", s.x); ++bad; }
    }
    if (bits_of(repro_logf(-1.0f)) != 0x7FC00000u || bits_of(repro_expf(float_of(0xFFC12345u))) != 0x7FC00000u) {
        std::printf("NaN bits wrong\n"); ++bad;
    }

    Tally te = for_all(check_exp);
    report("repro_expf", te, EXP_DELTA);
    Tally tl = for_all(check_log);
    report("repro_logf", tl, LOG_EPS);
    if (argc > 1) {
        FILE* fo = std::fopen(argv[1], "w");
        if (!fo) { std::perror(argv[1]); return 1; }
        for (auto& d : te.dump) std::fprintf(fo, "exp %08x %08x\n", d.first, d.second);
        for (auto& d : tl.dump) std::fprintf(fo, "log %08x %08x\n", d.first, d.second);
        std::fclose(fo);
    }
    bool pass = !bad && !te.wrong_dd && !te.wrong_cm && !tl.wrong_dd && !tl.wrong_cm &&
                te.max_err < EXP_DELTA && tl.max_err < LOG_EPS;
    std::printf("\nOVERALL: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
