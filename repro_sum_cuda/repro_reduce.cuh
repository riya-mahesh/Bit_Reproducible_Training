#pragma once
#include <cuda_runtime.h>

// Full-warp reductions. Call from all 32 lanes; sums/minima are returned in lane 0.
// Block helpers require a whole number of warps, all threads participating, and
// one shared scratch entry per warp. Every scratch location has one writer.
namespace repro_reduce {

template<class T>
__device__ __forceinline__ T warp_sum(T v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xFFFFFFFFu, v, o);
    return v;
}

template<class T>
__device__ __forceinline__ T warp_min(T v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = min(v, __shfl_down_sync(0xFFFFFFFFu, v, o));
    return v;
}

__device__ __forceinline__ unsigned block_max(unsigned v, unsigned* scratch) {
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = max(v, __shfl_down_sync(0xFFFFFFFFu, v, o));
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = lane < blockDim.x / 32 ? scratch[lane] : 0u;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v = max(v, __shfl_down_sync(0xFFFFFFFFu, v, o));
        if (lane == 0) scratch[0] = v;
    }
    __syncthreads();
    return scratch[0];
}

__device__ __forceinline__ int block_min(int v, int* scratch) {
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_min(v);
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = lane < blockDim.x / 32 ? scratch[lane] : 0x7FFFFFFF;
        v = warp_min(v);
        if (lane == 0) scratch[0] = v;
    }
    __syncthreads();
    return scratch[0];
}

} // namespace repro_reduce
