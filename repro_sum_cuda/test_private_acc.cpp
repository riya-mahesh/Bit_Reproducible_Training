// Host regression of the actual private-spill arithmetic used by GPU kernels.
// Compares normalized exact limbs, not just their rounded FP32 outputs.
#include "repro_sum_fast.cuh"
#include <array>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using Acc = std::array<int64_t, repro_sum::LIMBS>;

static float from_bits(uint32_t b) {
    float x;
    std::memcpy(&x, &b, sizeof(x));
    return x;
}

static bool check(const std::vector<float>& x, int lanes, int low) {
    Acc want{}, got{};
    int want_nf = 0, got_nf = 0;
    for (float v : x) repro_sum::acc_add(want.data(), 1, v, want_nf);
    for (int t = 0; t < lanes; ++t) {
        Acc local{};
        repro_sum::FastAcc fast;
        repro_sum::fast_init(fast);
        int nf = 0;
        // Test both a common window and different windows for each thread.
        int window = low < 0 ? t % (repro_sum::LIMBS - 2) : low;
        for (size_t i = t; i < x.size(); i += lanes)
            repro_sum::fast_add(fast, window, x[i], local.data(), nf);
        repro_sum::fast_finish(fast, window, local.data());
        for (int l = 0; l < repro_sum::LIMBS; ++l) got[l] += local[l];
        got_nf |= nf;
    }
    // Check sum and mean rounding before normalization mutates the arrays.
    for (uint32_t divisor : {1u, 3u, 100u, 2147483647u}) {
        float a = repro_sum::finalize(want.data(), 1, divisor, want_nf);
        float b = repro_sum::finalize(got.data(), 1, divisor, got_nf);
        if (repro_sum::float_bits(a) != repro_sum::float_bits(b)) return false;
    }
    repro_sum::normalize(want.data());
    repro_sum::normalize(got.data());
    return want_nf == got_nf && want == got;
}

int main() {
    std::mt19937 rng(20261122);
    std::vector<std::vector<float>> cases = {
        {}, {0.0f, -0.0f}, {1e30f, 1.0f, -1e30f},
        {1.0f, 0x1p-24f, 0x1p-149f},
        {INFINITY, -INFINITY, NAN, 1.0f, -2.0f},
        std::vector<float>(65537, -0x1.fffffep127f),
        std::vector<float>(65537, 0x1p-149f),
        std::vector<float>(65537, -1.0f)
    };
    std::vector<float> wide(100003);
    for (auto& x : wide) {
        uint32_t b = rng();
        if ((b & 0x7F800000u) == 0x7F800000u) b ^= 0x00800000u;
        x = from_bits(b);
    }
    cases.push_back(wide);
    std::vector<float> cancellation;
    for (float x : wide) { cancellation.push_back(x); cancellation.push_back(-x); }
    cancellation.push_back(0x1p-149f);
    cases.push_back(cancellation);
    int checked = 0;
    for (const auto& x : cases) for (int lanes : {1, 2, 31, 32, 256}) {
        for (int low = -1; low <= repro_sum::LIMBS - 3; ++low) {
            if (!check(x, lanes, low)) {
                std::fprintf(stderr, "FAIL n=%zu lanes=%d low=%d\n", x.size(), lanes, low);
                return 1;
            }
            ++checked;
        }
    }
    std::printf("PASS: %d private-accumulator cases; exact limbs, flags, sum and mean bits\n", checked);
}
