// CPU checks of cross_entropy_cpu; writes random cases for check_cross_entropy.py, which
// verifies every rounding in 300-bit arithmetic and compares with PyTorch in double.
// Usage: check_cross_entropy <file>
// Build: c++ -O2 -std=c++17 -ffp-contract=off -I. -I../repro_sum_cuda check_cross_entropy.cpp
#include "repro_softmax_math.cuh"
#include <cstdio>
#include <functional>
#include <random>
#include <vector>

using namespace repro_math;

static void serial(int n, const std::function<void(int)>& f) { for (int i = 0; i < n; ++i) f(i); }

static int failures = 0;
static void expect(bool ok, const char* what) {
    std::printf("  %-4s %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <file>\n", argv[0]); return 1; }
    std::mt19937_64 g(99);
    FILE* fo = std::fopen(argv[1], "w");
    if (!fo) { std::perror(argv[1]); return 1; }
    const struct { int rows, cols; float scale; int ignored; } cases[] = {
        {100, 10, 5.0f, 0}, {20, 10, 0.5f, 0}, {30, 10, 40.0f, 0}, {50, 3, 2.0f, 10}, {4, 2000, 8.0f, 1}, {10, 1, 3.0f, 0},
        {25, 10, 80.0f, 5},
    };
    for (auto& c : cases) {
        std::normal_distribution<float> d(0.0f, c.scale);
        size_t n = (size_t)c.rows * c.cols;
        std::vector<float> x(n), grad(n);
        std::vector<int> t(c.rows);
        for (auto& v : x) v = d(g);
        for (auto& v : t) v = (int)(g() % (uint64_t)c.cols);
        for (int i = 0; i < c.ignored; ++i) t[i] = -100;
        float L = cross_entropy_cpu(x.data(), t.data(), -100, c.rows, c.cols, grad.data(), 1.0f, serial);
        std::fprintf(fo, "case %d %d %08x\n", c.rows, c.cols, bits_of(L));
        for (int r = 0; r < c.rows; ++r) std::fprintf(fo, "%d ", t[r]);
        std::fprintf(fo, "\n");
        for (size_t i = 0; i < n; ++i) std::fprintf(fo, "%08x %08x\n", bits_of(x[i]), bits_of(grad[i]));
    }
    std::fclose(fo);

    // Special cases.
    {
        std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f}, grad(4);
        std::vector<int> all_ignored = {-100, -100}, bad = {0, 7}, ok = {1, 1};
        float L = cross_entropy_cpu(x.data(), all_ignored.data(), -100, 2, 2, grad.data(), 1.0f, serial);
        expect(bits_of(L) == 0x7FC00000u && grad[0] == 0.0f, "all rows ignored: NaN loss, zero gradient");
        L = cross_entropy_cpu(x.data(), bad.data(), -100, 2, 2, grad.data(), 1.0f, serial);
        expect(bits_of(L) == 0x7FC00000u && bits_of(grad[2]) == 0x7FC00000u, "out-of-range target: NaN");
        std::vector<float> g1(4), g2(4);
        cross_entropy_cpu(x.data(), ok.data(), -100, 2, 2, g1.data(), 1.0f, serial);
        cross_entropy_cpu(x.data(), ok.data(), -100, 2, 2, g2.data(), 0.5f, serial);
        expect(g2[0] == fmul(g1[0], 0.5f), "seed scales the gradient");
        std::vector<float> xn = {1.0f, NAN, 3.0f, 4.0f};
        L = cross_entropy_cpu(xn.data(), ok.data(), -100, 2, 2, grad.data(), 1.0f, serial);
        expect(bits_of(L) == 0x7FC00000u && bits_of(grad[1]) == 0x7FC00000u && !std::isnan(grad[2]), "NaN logit: NaN loss, that row NaN");
        std::vector<float> xm = {-INFINITY, 2.0f, 3.0f, 4.0f};
        std::vector<int> tz = {1, 0};
        L = cross_entropy_cpu(xm.data(), tz.data(), -100, 2, 2, grad.data(), 1.0f, serial);
        expect(!std::isnan(L) && grad[0] == 0.0f, "masked -inf logit away from the target is exact 0");
    }
    std::printf("%s\n", failures ? "OVERALL: FAIL" : "OVERALL: PASS");
    return failures ? 1 : 0;
}
