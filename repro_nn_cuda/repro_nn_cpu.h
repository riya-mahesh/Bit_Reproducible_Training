#pragma once
// CPU version of repro_nn, using the same scalar code as the GPU kernels.
#include "repro_nn_math.cuh"

namespace repro_nn {

inline void cpu_max_pool2d(const float* x, float* out, int* index, const PoolShape& g) {
    int oh = g.OH(), ow = g.OW();
    size_t plane = (size_t)g.H * g.W, oplane = (size_t)oh * ow;
    for (size_t pl = 0; pl < (size_t)g.N * g.C; ++pl)
        for (int oy = 0; oy < oh; ++oy)
            for (int ox = 0; ox < ow; ++ox) {
                size_t o = pl * oplane + (size_t)oy * ow + ox;
                pool_window(x + pl * plane, g, oy, ox, out[o], index[o]);
            }
}

inline void cpu_max_pool2d_backward(const float* dy, const int* index, float* dx, const PoolShape& g) {
    size_t oplane = (size_t)g.OH() * g.OW(), plane = (size_t)g.H * g.W;
    for (size_t pl = 0; pl < (size_t)g.N * g.C; ++pl)
        for (int y = 0; y < g.H; ++y)
            for (int x = 0; x < g.W; ++x)
                dx[pl * plane + (size_t)y * g.W + x] = pool_backward_pixel(dy + pl * oplane, index + pl * oplane, g, y, x);
}

inline void cpu_adam(float* p, const float* g, float* m, float* v, size_t n, const AdamScalars& s) {
    for (size_t i = 0; i < n; ++i) adam_update(p[i], g[i], m[i], v[i], s);
}

} // namespace repro_nn
