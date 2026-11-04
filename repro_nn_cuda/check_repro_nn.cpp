// CPU checks of MaxPool and Adam, and a file of cases for check_repro_nn.py, which recomputes
// every Adam rounding exactly and compares MaxPool and Adam with PyTorch.
// Usage: check_repro_nn <file>
#include "repro_nn_cpu.h"
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace repro_nn;
using repro_math::bits_of;

static int failures = 0;
static void expect(bool ok, const char* what) {
    std::printf("  %-4s %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <file>\n", argv[0]); return 1; }
    FILE* fo = std::fopen(argv[1], "w");
    if (!fo) { std::perror(argv[1]); return 1; }
    std::mt19937_64 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    // Ties: the first largest entry in row-major order wins, including +0 and -0.
    {
        PoolShape g{1, 1, 2, 2, 2, 2, 2, 2};
        float x1[4] = {1.0f, 3.0f, 3.0f, 2.0f}, o;
        int id;
        cpu_max_pool2d(x1, &o, &id, g);
        expect(o == 3.0f && id == 1, "tie keeps the first in row-major order");
        float x2[4] = {-0.0f, 0.0f, 0.0f, -1.0f};
        cpu_max_pool2d(x2, &o, &id, g);
        expect(id == 0 && bits_of(o) == 0x80000000u, "+0 and -0 tie keeps the first");
        float dy = 2.0f, dx[4];
        id = 3;
        cpu_max_pool2d_backward(&dy, &id, dx, g);
        expect(dx[3] == 2.0f && dx[0] == 0.0f && dx[1] == 0.0f && dx[2] == 0.0f, "backward sends the gradient to the index");
        dy = -0.0f;
        cpu_max_pool2d_backward(&dy, &id, dx, g);
        expect(bits_of(dx[3]) == 0u, "backward gives +0 for a -0 gradient");
    }
    // Overlapping windows: outputs sharing a pixel sum exactly.
    {
        PoolShape g{1, 1, 1, 4, 1, 2, 1, 1};           // windows (0,1) (1,2) (2,3)
        float x[4] = {0.0f, 5.0f, 1.0f, 0.0f}, o[3], dx[4];
        int id[3];
        cpu_max_pool2d(x, o, id, g);
        expect(id[0] == 1 && id[1] == 1 && id[2] == 2, "overlapping windows pick their largest");
        float dy[3] = {1e8f, -1e8f, 7.0f};
        cpu_max_pool2d_backward(dy, id, dx, g);
        expect(dx[1] == 0.0f && dx[2] == 7.0f, "shared pixel gets the exact sum");
        float dz[3] = {1.0f, 0x1p-30f, 0.0f};
        cpu_max_pool2d_backward(dz, id, dx, g);
        expect(dx[1] == 1.0f, "shared pixel is rounded once");
    }

    // MaxPool cases for PyTorch: LeNet's two pools and overlapping windows. Values are distinct,
    // so PyTorch's own tie rule plays no part.
    const PoolShape shapes[] = {{20, 6, 28, 28, 2, 2, 2, 2}, {20, 16, 10, 10, 2, 2, 2, 2}, {3, 2, 9, 13, 3, 2, 1, 2}, {2, 2, 8, 8, 3, 3, 2, 2}};
    for (const PoolShape& g : shapes) {
        std::vector<float> x(g.input_size()), o(g.output_size()), dy(g.output_size()), dx(g.input_size());
        std::vector<int> id(g.output_size());
        for (auto& v : x) v = nd(rng);
        for (auto& v : dy) v = nd(rng);
        cpu_max_pool2d(x.data(), o.data(), id.data(), g);
        cpu_max_pool2d_backward(dy.data(), id.data(), dx.data(), g);
        std::fprintf(fo, "pool %d %d %d %d %d %d %d %d\n", g.N, g.C, g.H, g.W, g.KH, g.KW, g.SY, g.SX);
        for (auto v : x) std::fprintf(fo, "%08x ", bits_of(v));
        std::fprintf(fo, "\n");
        for (auto v : o) std::fprintf(fo, "%08x ", bits_of(v));
        std::fprintf(fo, "\n");
        for (auto v : id) std::fprintf(fo, "%d ", v);
        std::fprintf(fo, "\n");
        for (auto v : dy) std::fprintf(fo, "%08x ", bits_of(v));
        std::fprintf(fo, "\n");
        for (auto v : dx) std::fprintf(fo, "%08x ", bits_of(v));
        std::fprintf(fo, "\n");
    }

    // Adam: a few steps from the same data, recording the state after every step.
    const struct { float lr, wd; int adamw; } opts[] = {{0.01f, 0.0f, 0}, {0.001f, 0.01f, 0}, {0.001f, 0.01f, 1}};
    for (auto& op : opts) {
        AdamConfig c;
        c.lr = op.lr;
        c.weight_decay = op.wd;
        c.adamw = op.adamw != 0;
        AdamClock k;
        const int n = 200, steps = 25;
        std::vector<float> p(n), m(n, 0.0f), v(n, 0.0f), g(n);
        for (auto& x : p) x = nd(rng);
        std::fprintf(fo, "adam %d %d %08x %08x %d\n", n, steps, bits_of(op.lr), bits_of(op.wd), op.adamw);
        for (auto x : p) std::fprintf(fo, "%08x ", bits_of(x));
        std::fprintf(fo, "\n");
        for (int s = 0; s < steps; ++s) {
            for (auto& x : g) x = 0.1f * nd(rng);
            for (auto x : g) std::fprintf(fo, "%08x ", bits_of(x));
            std::fprintf(fo, "\n");
            cpu_adam(p.data(), g.data(), m.data(), v.data(), n, adam_next_step(c, k));
            for (auto x : p) std::fprintf(fo, "%08x ", bits_of(x));
            std::fprintf(fo, "\n");
        }
    }
    std::fclose(fo);
    std::printf("%s\n", failures ? "OVERALL: FAIL" : "OVERALL: PASS");
    return failures ? 1 : 0;
}
