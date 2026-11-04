// Why does (float)exp((double)x) give the same bits on every platform?
//
// Rounding a double approximation d of f(x) to float gives the correctly rounded
// float RN32(f(x)) whenever no float rounding midpoint lies between f(x) and d.
// A double-precision library has error |d - f(x)| <= e ulp_64(f(x)), with e about 1.
// So if, over all float inputs, f(x) never comes within e double-ulps of a float
// midpoint, then EVERY double library with error <= e returns the correctly rounded
// float on every input -- and therefore all of them agree.
//
// This program evaluates exp(x) and log(x) for every float32 input in double-double
// arithmetic (~100 bits) and reports the closest approach of f(x) to a float
// rounding midpoint, in units of ulp_64(f(x)).
//
// Build: c++ -O2 -std=c++17 -ffp-contract=off -pthread why_via_double.cpp
// (FMA is used only explicitly, for exact products.)
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

// ---------------- double-double arithmetic (error-free transforms) ----------------
struct dd { double hi, lo; };
static inline dd two_sum(double a, double b) { double s = a + b, bb = s - a; return {s, (a - (s - bb)) + (b - bb)}; }
static inline dd quick_two_sum(double a, double b) { double s = a + b; return {s, b - (s - a)}; }
static inline dd two_prod(double a, double b) { double p = a * b; return {p, std::fma(a, b, -p)}; }
static inline dd add(dd a, dd b) { dd s = two_sum(a.hi, b.hi); s.lo += a.lo + b.lo; return quick_two_sum(s.hi, s.lo); }
static inline dd add(dd a, double b) { dd s = two_sum(a.hi, b); s.lo += a.lo; return quick_two_sum(s.hi, s.lo); }
static inline dd mul(dd a, dd b) { dd p = two_prod(a.hi, b.hi); p.lo += a.hi * b.lo + a.lo * b.hi; return quick_two_sum(p.hi, p.lo); }
static inline dd mul(dd a, double b) { dd p = two_prod(a.hi, b); p.lo += a.lo * b; return quick_two_sum(p.hi, p.lo); }
static inline dd neg(dd a) { return {-a.hi, -a.lo}; }

// ln 2 to ~107 bits.
static const dd LN2 = {0x1.62e42fefa39efp-1, 0x1.abc9e3b39803fp-56};

// exp in double-double: exp(x) = 2^k * exp(r)^(2^8), r = (x - k ln2) / 2^8, Taylor series.
static dd dd_exp(double x) {
    double k = std::nearbyint(x / LN2.hi);
    dd r = add(dd{x, 0.0}, neg(mul(LN2, k)));       // x - k ln2, |r| <= ~0.35
    r.hi = std::ldexp(r.hi, -8);
    r.lo = std::ldexp(r.lo, -8);                     // exact scaling
    // Taylor: sum_{n=0}^{14} r^n / n!   (|r| < 0.0014, so the tail is < 1e-45)
    dd term = {1.0, 0.0}, sum = {1.0, 0.0};
    for (int n = 1; n <= 14; ++n) {
        term = mul(term, r);
        // divide by n: term / n via multiplication by an exact reciprocal in dd
        double q = term.hi / n;
        dd qn = two_prod(q, (double)n);
        double rem = ((term.hi - qn.hi) - qn.lo + term.lo) / n;
        term = quick_two_sum(q, rem);
        sum = add(sum, term);
    }
    for (int i = 0; i < 8; ++i) sum = mul(sum, sum);  // undo the 2^8 reduction
    return {std::ldexp(sum.hi, (int)k), std::ldexp(sum.lo, (int)k)};
}

// log in double-double: one Newton step from the double log, y1 = y0 + x*exp(-y0) - 1.
static dd dd_log(double x) {
    double y0 = std::log(x);
    dd e = dd_exp(-y0);
    dd t = mul(e, x);                                // x * exp(-y0), close to 1
    dd corr = add(t, -1.0);
    return add(dd{y0, 0.0}, corr);
}

// ------------- distance from v (> 0) to the nearest float32 rounding midpoint -------------
// Returned in units of ulp_64(v) = 2^(floor(log2 v) - 52). Midpoints between adjacent
// floats (including the subnormal range and the overflow threshold FLT_MAX + ulp/2)
// have at most 25 significant bits, so they are exact doubles.
static double midpoint_distance(dd v) {
    float f = (float)v.hi;                                // a float next to v
    float below = std::nextafterf(f, 0.0f), above = std::nextafterf(f, INFINITY);
    double best = INFINITY;
    for (float a : {below, f}) {
        float b = std::nextafterf(a, INFINITY);
        double m = std::isinf(b) ? 0x1.ffffffp+127 /* FLT_MAX + half ulp */ : ((double)a + (double)b) / 2;
        if (a == 0.0f && !(b > 0)) continue;
        double d = std::fabs((v.hi - m) + v.lo);         // v.hi - m is exact (Sterbenz) near m
        best = std::min(best, d);
    }
    (void)above;
    int ex;
    std::frexp(v.hi, &ex);                               // v.hi in [2^(ex-1), 2^ex)
    double ulp64 = std::ldexp(1.0, ex - 1 - 52);
    return best / ulp64;
}

// hi + lo rounded once to float32. (float)hi alone, or (float)(hi + lo), rounds twice and
// is wrong when the true value is within half a double ulp of a float midpoint. Moving hi
// to the neighbour with an odd last bit on the side of the true value makes the cast exact.
static float round_once(dd v) {
    uint64_t b;
    std::memcpy(&b, &v.hi, 8);
    if (v.lo != 0.0 && (b & 1u) == 0) b += ((v.lo > 0.0) == (v.hi > 0.0)) ? 1 : (uint64_t)-1;
    double d;
    std::memcpy(&d, &b, 8);
    return (float)d;
}

struct Result {
    double min_dist = INFINITY;
    uint32_t argmin = 0;
    uint64_t below[6] = {};      // counts with distance < 1, 2, 4, 16, 256, 65536 double-ulps
    uint64_t evaluated = 0;
    std::vector<std::pair<uint32_t, double>> hard;   // inputs within 2 double-ulps of a midpoint
};
static const double THRESH[6] = {1, 2, 4, 16, 256, 65536};

template<class F>
static Result scan(F f) {
    unsigned nt = std::max(1u, std::thread::hardware_concurrency());
    std::vector<Result> part(nt);
    std::atomic<uint32_t> next{0};
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < nt; ++t)
        pool.emplace_back([&, t] {
            Result& R = part[t];
            for (uint32_t blk; (blk = next.fetch_add(1)) < 4096;) {
                for (uint32_t j = 0; j < (1u << 20); ++j) {
                    uint32_t bits = (blk << 20) | j;
                    float x;
                    std::memcpy(&x, &bits, 4);
                    double dist;
                    if (!f(x, dist)) continue;
                    ++R.evaluated;
                    if (dist < R.min_dist) { R.min_dist = dist; R.argmin = bits; }
                    for (int i = 0; i < 6; ++i) if (dist < THRESH[i]) ++R.below[i];
                    if (dist < 2.0) R.hard.push_back({bits, dist});
                }
            }
        });
    for (auto& th : pool) th.join();
    Result out;
    for (auto& R : part) {
        out.evaluated += R.evaluated;
        if (R.min_dist < out.min_dist) { out.min_dist = R.min_dist; out.argmin = R.argmin; }
        for (int i = 0; i < 6; ++i) out.below[i] += R.below[i];
        out.hard.insert(out.hard.end(), R.hard.begin(), R.hard.end());
    }
    std::sort(out.hard.begin(), out.hard.end());
    return out;
}

static void report(const char* name, const Result& R, double (*libf)(double), dd (*ddf)(double)) {
    float x;
    std::memcpy(&x, &R.argmin, 4);
    dd v = ddf((double)x);
    std::printf("\n%s: %llu inputs evaluated\n", name, (unsigned long long)R.evaluated);
    std::printf("  closest approach to a float rounding midpoint: %.3f double-ulps  (= 2^%.1f relative)\n",
                R.min_dist, std::log2(R.min_dist) - 52);
    std::printf("  at x = %a (%.9g): f(x) = %.20e\n", x, x, v.hi);
    std::printf("  this machine's double %s there rounds to %a; correctly rounded float is %a\n", name,
                (float)libf((double)x), round_once(v));
    std::printf("  inputs within N double-ulps of a midpoint:");
    for (int i = 0; i < 6; ++i) std::printf("  <%g: %llu", THRESH[i], (unsigned long long)R.below[i]);
    std::printf("\n");
    std::printf("  inputs within 2 double-ulps (where libraries with ~1 ulp error could disagree):\n");
    for (auto& h : R.hard) {
        float hx;
        std::memcpy(&hx, &h.first, 4);
        dd hv = ddf((double)hx);
        float cr = round_once(hv), mine = (float)libf((double)hx);
        std::printf("    x = %-16.9g (bits %08x)  f(x) = %-16.9g  distance %.3f ulp64  correct float %a  this libm %a %s\n",
                    hx, h.first, hv.hi, h.second, cr, mine, cr == mine ? "" : "<-- WRONG");
    }
}

int main() {
    // Self-check of the double-double functions against the machine's double libm on samples:
    // they must agree to within ~1 double ulp (the libm's own error).
    double worst_e = 0, worst_l = 0;
    for (int i = 0; i < 200000; ++i) {
        double x = -100.0 + 188.0 * i / 200000.0;
        dd v = dd_exp(x);
        worst_e = std::max(worst_e, std::fabs(v.hi - std::exp(x)) / std::ldexp(1.0, std::ilogb(v.hi) - 52));
        double y = std::ldexp(1.0 + i / 200000.0, i % 200 - 100);
        dd w = dd_log(y);
        if (w.hi != 0) worst_l = std::max(worst_l, std::fabs(w.hi - std::log(y)) / std::ldexp(1.0, std::ilogb(w.hi) - 52));
    }
    std::printf("self-check: double-double vs this machine's libm, max difference: exp %.2f ulp, log %.2f ulp "
                "(should be <= ~1)\n", worst_e, worst_l);

    // exp: finite results that are not trivially 0 (x > -104) or inf (x < 89).
    Result re = scan([](float x, double& dist) {
        if (!(x > -104.0f && x < 89.0f)) return false;
        dist = midpoint_distance(dd_exp((double)x));
        return true;
    });
    report("exp", re, [](double x) { return std::exp(x); }, dd_exp);

    // log: positive finite inputs except x == 1 (log 1 = 0 exactly).
    Result rl = scan([](float x, double& dist) {
        if (!(x > 0.0f) || std::isinf(x) || x == 1.0f) return false;
        dd v = dd_log((double)x);
        dd a = v.hi < 0 ? neg(v) : v;                     // distance is symmetric in sign
        dist = midpoint_distance(a);
        return true;
    });
    report("log", rl, [](double x) { return std::log(x); }, dd_log);
    return 0;
}
