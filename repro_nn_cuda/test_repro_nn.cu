// GPU tests of MaxPool and Adam: output equals the CPU version bit for bit, including heavy ties
// (ReLU output has many equal zeros), overlapping windows, and many Adam steps.
#include "repro_nn.cuh"
#include "repro_nn_cpu.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace repro_nn;

static void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(1); }
}

static int failures = 0;
static void expect(bool ok, const std::string& what) {
    std::printf("  %-4s %s\n", ok ? "ok" : "FAIL", what.c_str());
    if (!ok) ++failures;
}

template<class T>
static T* upload(const std::vector<T>& v) {
    T* d;
    ck(cudaMalloc(&d, v.size() * sizeof(T)), "malloc");
    ck(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}

template<class T>
static std::vector<T> download(const T* d, size_t n) {
    std::vector<T> v(n);
    ck(cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return v;
}

template<class T>
static bool same(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}

static void check_pool(const std::string& name, PoolShape g, bool ties, std::mt19937_64& rng) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x(g.input_size()), dy(g.output_size());
    for (auto& v : x) { v = nd(rng); if (ties) v = v < 0.5f ? 0.0f : (v < 1.0f ? 1.0f : v); }
    for (auto& v : dy) v = nd(rng);
    std::vector<float> co(g.output_size()), cdx(g.input_size());
    std::vector<int> cid(g.output_size());
    cpu_max_pool2d(x.data(), co.data(), cid.data(), g);
    cpu_max_pool2d_backward(dy.data(), cid.data(), cdx.data(), g);

    float *dxi = upload(x), *ddy = upload(dy), *o, *b;
    int* id;
    ck(cudaMalloc(&o, g.output_size() * 4), "m");
    ck(cudaMalloc(&id, g.output_size() * 4), "m");
    ck(cudaMalloc(&b, g.input_size() * 4), "m");
    ck(repro_max_pool2d(dxi, o, id, g), "pool");
    ck(repro_max_pool2d_backward(ddy, id, b, g), "pool back");
    ck(cudaDeviceSynchronize(), "sync");
    expect(same(download(o, g.output_size()), co), "pool output GPU == CPU: " + name);
    expect(same(download(id, g.output_size()), cid), "pool index GPU == CPU: " + name);
    expect(same(download(b, g.input_size()), cdx), "pool backward GPU == CPU: " + name);
    cudaFree(dxi); cudaFree(ddy); cudaFree(o); cudaFree(id); cudaFree(b);
}

static void check_adam(const std::string& name, AdamConfig c, std::mt19937_64& rng) {
    const size_t n = 61706;
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> p(n), m(n, 0.0f), v(n, 0.0f), g(n);
    for (auto& x : p) x = 0.1f * nd(rng);
    float *dp = upload(p), *dm = upload(m), *dv = upload(v), *dg;
    ck(cudaMalloc(&dg, n * 4), "m");
    AdamClock kc, kg;
    bool ok = true;
    for (int s = 0; s < 200; ++s) {
        for (auto& x : g) x = 1e-2f * nd(rng);
        ck(cudaMemcpy(dg, g.data(), n * 4, cudaMemcpyHostToDevice), "copy");
        cpu_adam(p.data(), g.data(), m.data(), v.data(), n, adam_next_step(c, kc));
        ck(repro_adam(dp, dg, dm, dv, n, adam_next_step(c, kg)), "adam");
        ck(cudaDeviceSynchronize(), "sync");
        ok = ok && same(download(dp, n), p) && same(download(dm, n), m) && same(download(dv, n), v);
    }
    expect(ok, "Adam GPU == CPU over 200 steps (p, m, v): " + name);
    cudaFree(dp); cudaFree(dm); cudaFree(dv); cudaFree(dg);
}

int main() {
    cudaDeviceProp prop;
    ck(cudaGetDeviceProperties(&prop, 0), "device");
    std::printf("device: %s\n", prop.name);
    std::mt19937_64 rng(11);
    check_pool("LeNet pool1", PoolShape{100, 6, 28, 28, 2, 2, 2, 2}, false, rng);
    check_pool("LeNet pool1, many ties", PoolShape{100, 6, 28, 28, 2, 2, 2, 2}, true, rng);
    check_pool("LeNet pool2, many ties", PoolShape{100, 16, 10, 10, 2, 2, 2, 2}, true, rng);
    check_pool("overlapping 3x2 stride (1,2), many ties", PoolShape{3, 2, 9, 13, 3, 2, 1, 2}, true, rng);
    check_pool("large plane 300x300", PoolShape{2, 1, 300, 300, 3, 3, 2, 2}, false, rng);
    AdamConfig a;
    a.lr = 0.01f;
    check_adam("lr 0.01", a, rng);
    AdamConfig w;
    w.weight_decay = 0.01f;
    check_adam("Adam with weight decay", w, rng);
    w.adamw = true;
    check_adam("AdamW", w, rng);
    std::printf("%s\n", failures ? "OVERALL: FAIL" : "OVERALL: PASS");
    return failures ? 1 : 0;
}
