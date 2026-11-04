// RepDL's (float)exp((double)x) and (float)log((double)x) on the inputs where it is
// most likely to fail: those whose exp/log lies within 2 double-ulps of a float rounding
// midpoint (found by why_via_double.cpp over all 2^32 inputs; see hard_cases_table.h).
// Everywhere else, any double library with <= 1 ulp error gives the correctly rounded
// float, so these 19 inputs decide whether RepDL's approach is reproducible on a platform.
//
// Checks the GPU (CUDA's double exp/log) and this machine's C library, against the
// correctly rounded results.
// Build: nvcc -O3 -arch=native hard_cases.cu -o build/hard_cases
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include "hard_cases_table.h"

constexpr int N = sizeof(kHardCases) / sizeof(kHardCases[0]);

__global__ void gpu_eval(const HardCase* cases, float* out) {
    int i = threadIdx.x;
    if (i >= N) return;
    float x = __uint_as_float(cases[i].bits);
    out[i] = cases[i].is_log ? (float)log((double)x) : (float)exp((double)x);
}

int main() {
    HardCase* d_cases;
    float* d_out;
    float gpu[N];
    cudaMalloc(&d_cases, sizeof(kHardCases));
    cudaMalloc(&d_out, sizeof(gpu));
    cudaMemcpy(d_cases, kHardCases, sizeof(kHardCases), cudaMemcpyHostToDevice);
    gpu_eval<<<1, 32>>>(d_cases, d_out);
    if (cudaMemcpy(gpu, d_out, sizeof(gpu), cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));
        return 1;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int rt;
    cudaRuntimeGetVersion(&rt);
    std::printf("GPU: %s (CUDA runtime %d)    CPU: host C library\n", prop.name, rt);
    std::printf("%-4s %-16s %-10s %-16s %-16s %-16s\n", "fn", "x", "dist", "correct", "GPU", "CPU");
    int gpu_wrong = 0, cpu_wrong = 0, disagree = 0;
    for (int i = 0; i < N; ++i) {
        float x;
        std::memcpy(&x, &kHardCases[i].bits, 4);
        float cpu = kHardCases[i].is_log ? (float)std::log((double)x) : (float)std::exp((double)x);
        float c = kHardCases[i].correct;
        gpu_wrong += gpu[i] != c;
        cpu_wrong += cpu != c;
        disagree += gpu[i] != cpu;
        std::printf("%-4s %-16.9g %-10.3f %-16a %-16a%s %-16a%s\n", kHardCases[i].is_log ? "log" : "exp", x,
                    kHardCases[i].dist_ulp64, c, gpu[i], gpu[i] == c ? " " : "!", cpu, cpu == c ? "" : " !");
    }
    std::printf("\nwrong vs correctly rounded:  GPU %d of %d,  CPU %d of %d;   GPU and CPU disagree on %d\n",
                gpu_wrong, N, cpu_wrong, N, disagree);
    std::printf("(dist = distance of the true value from a float rounding midpoint, in double ulps)\n");
    return 0;
}
