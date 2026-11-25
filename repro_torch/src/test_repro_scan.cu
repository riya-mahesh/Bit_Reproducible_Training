// Checks the two-stage finite scan, including partial blocks, persistent flags,
// repeated workspace reuse and a non-default stream. No PyTorch dependency.
#include "repro_scan.cuh"
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void ck(cudaError_t e) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(e)); std::exit(1); }
}

int main() {
    cudaStream_t stream;
    ck(cudaStreamCreate(&stream));
    int* flag;
    ck(cudaMalloc(&flag, sizeof(int)));
    int checked = 0;
    for (size_t n : {size_t(0), size_t(1), size_t(31), size_t(32), size_t(255),
                     size_t(256), size_t(257), size_t(65537)}) {
        std::vector<float> x(n, 1.0f);
        float* dx = nullptr;
        int* partial = nullptr;
        if (n) {
            ck(cudaMalloc(&dx, n * sizeof(float)));
            ck(cudaMalloc(&partial, repro_scan_workspace_bytes(n)));
        }
        int expected = INT_MAX;
        ck(cudaMemcpyAsync(flag, &expected, sizeof(int), cudaMemcpyHostToDevice, stream));
        auto run = [&](int id) {
            if (n) {
                ck(cudaMemcpyAsync(dx, x.data(), n * sizeof(float), cudaMemcpyHostToDevice, stream));
                ck(cudaMemsetAsync(partial, 0xFF, repro_scan_workspace_bytes(n), stream));
            }
            ck(repro_scan_finite(dx, flag, n, id, partial, stream));
            int got;
            ck(cudaMemcpyAsync(&got, flag, sizeof(int), cudaMemcpyDeviceToHost, stream));
            ck(cudaStreamSynchronize(stream));
            if (got != expected) {
                std::fprintf(stderr, "FAIL scan n=%zu op=%d got=%d expected=%d\n", n, id, got, expected);
                std::exit(2);
            }
            ++checked;
        };
        run(9); // finite, including no input
        if (n) {
            x[n - 1] = INFINITY; expected = 9; run(9); // final partial block
            x[0] = -INFINITY; expected = 4; run(4);
            x[n / 2] = NAN; run(12); // preserve the earlier, smaller ID
            for (auto& v : x) v = -0.0f;
            run(1); // finite scan cannot lower the flag
            x[n / 2] = NAN; expected = 2; run(2);
        }
        if (partial) ck(cudaFree(partial));
        if (dx) ck(cudaFree(dx));
    }
    ck(cudaFree(flag));
    ck(cudaStreamDestroy(stream));
    std::printf("PASS: %d finite-scan checks\n", checked);
}
