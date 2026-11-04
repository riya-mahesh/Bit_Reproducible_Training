// PyTorch bindings: each function checks shapes, then runs the GPU kernel for a CUDA tensor or
// the CPU version of the same definition for a CPU tensor. CUDA calls queue on the default
// stream, which PyTorch also uses, so they stay in order with PyTorch's own operations.
#include <torch/extension.h>

#include "repro_conv_cpu.h"
#include "repro_gemm_cpu.h"
#include "repro_nn_cpu.h"
#include "repro_softmax_math.cuh"
#ifndef REPRO_CPU_ONLY
#include "repro_conv.cuh"
#include "repro_gemm.cuh"
#include "repro_nn.cuh"
#include "repro_softmax.cuh"
#include "repro_sum.cuh"
#include <cuda_runtime.h>
cudaError_t repro_scan_finite(const float* x, int* flag, size_t n, int op_id);
#endif
#include <functional>
#include <stdexcept>
#include <string>

namespace {

using torch::Tensor;

#ifndef REPRO_CPU_ONLY
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
Tensor workspace(size_t bytes, const Tensor& like) {
    return torch::empty({(int64_t)std::max<size_t>(bytes, 1)}, torch::TensorOptions().dtype(torch::kUInt8).device(like.device()));
}
#endif

void require_float(const Tensor& t, const char* name) {
    if (t.dtype() != torch::kFloat32) throw std::invalid_argument(std::string(name) + " must be float32");
}

auto par = [](int n, const std::function<void(int)>& f) { repro_gemm::cpu_for(n, f); };

// C[M, N] = op(A) * op(B).
Tensor gemm(Tensor A, Tensor B, int M, int N, int K, bool ta, bool tb, int S) {
    require_float(A, "A");
    require_float(B, "B");
    A = A.contiguous();
    B = B.contiguous();
    if (A.numel() != (int64_t)M * K || B.numel() != (int64_t)N * K) throw std::invalid_argument("gemm: shape mismatch");
    Tensor C = torch::empty({M, N}, A.options());
    if (A.is_cuda()) {
#ifndef REPRO_CPU_ONLY
        repro_gemm::Config cfg;
        cfg.slices = S;
        cfg.trans_a = ta;
        cfg.trans_b = tb;
        repro_gemm::Plan plan;
        ck(repro_gemm::make_plan(M, N, K, cfg, &plan), "gemm plan");
        Tensor ws = workspace(plan.workspace_bytes, A);
        ck(repro_gemm::repro_gemm_fp32_ws(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K, cfg,
                                          ws.data_ptr<uint8_t>(), plan.workspace_bytes), "gemm");
        return C;
#endif
    }
    repro_gemm::cpu_gemm(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K, ta, tb, S);
    return C;
}

// out[b] = exact sum over a, c of x[a, b, c], rounded once; the same divided by A * C if mean.
Tensor exact_sum(Tensor x, int A, int B, int C, bool mean) {
    require_float(x, "x");
    x = x.contiguous();
    if (x.numel() != (int64_t)A * B * C) throw std::invalid_argument("sum: shape mismatch");
    Tensor out = torch::empty({B}, x.options());
    if (x.is_cuda()) {
#ifndef REPRO_CPU_ONLY
        repro_sum::SumConfig cfg;
        cfg.mean = mean;
        repro_sum::SumPlan plan;
        ck(repro_sum::make_sum_plan(A, B, C, cfg, &plan), "sum plan");
        Tensor ws = workspace(plan.workspace_bytes, x);
        ck(repro_sum::repro_sum_fp32_ws(x.data_ptr<float>(), out.data_ptr<float>(), A, B, C, cfg,
                                        ws.data_ptr<uint8_t>(), plan.workspace_bytes), "sum");
        return out;
#endif
    }
    const float* p = x.data_ptr<float>();
    float* o = out.data_ptr<float>();
    par(B, [&](int b) {
        int64_t acc[repro_sum::LIMBS] = {};
        int nf = 0;
        for (int a = 0; a < A; ++a)
            for (int c = 0; c < C; ++c) repro_sum::acc_add(acc, 1, p[((size_t)a * B + b) * C + c], nf);
        o[b] = repro_sum::finalize(acc, 1, mean ? (uint32_t)((int64_t)A * C) : 1u, nf);
    });
    return out;
}

repro_conv::ConvShape conv_shape(const Tensor& x_shape_ref, int F, int KH, int KW, int SY, int SX, int PY, int PX) {
    repro_conv::ConvShape g;
    g.N = (int)x_shape_ref.size(0);
    g.C = (int)x_shape_ref.size(1);
    g.H = (int)x_shape_ref.size(2);
    g.W = (int)x_shape_ref.size(3);
    g.F = F;
    g.KH = KH;
    g.KW = KW;
    g.SY = SY;
    g.SX = SX;
    g.PY = PY;
    g.PX = PX;
    return g;
}

Tensor conv2d(Tensor x, Tensor w, int SY, int SX, int PY, int PX, int S) {
    require_float(x, "x");
    require_float(w, "weight");
    x = x.contiguous();
    w = w.contiguous();
    auto g = conv_shape(x, (int)w.size(0), (int)w.size(2), (int)w.size(3), SY, SX, PY, PX);
    Tensor out = torch::empty({g.N, g.F, g.OH(), g.OW()}, x.options());
    if (x.is_cuda()) {
#ifndef REPRO_CPU_ONLY
        size_t bytes;
        ck(repro_conv::conv2d_workspace(g, S, &bytes), "conv plan");
        Tensor ws = workspace(bytes, x);
        ck(repro_conv::repro_conv2d_ws(x.data_ptr<float>(), w.data_ptr<float>(), out.data_ptr<float>(), g, S,
                                       ws.data_ptr<uint8_t>(), bytes), "conv");
        return out;
#endif
    }
    repro_conv::cpu_conv2d(x.data_ptr<float>(), w.data_ptr<float>(), out.data_ptr<float>(), g, S);
    return out;
}

Tensor conv2d_grad_weight(Tensor dy, Tensor x, int KH, int KW, int SY, int SX, int PY, int PX, int S) {
    require_float(dy, "dy");
    require_float(x, "x");
    dy = dy.contiguous();
    x = x.contiguous();
    auto g = conv_shape(x, (int)dy.size(1), KH, KW, SY, SX, PY, PX);
    Tensor dw = torch::empty({g.F, g.C, KH, KW}, x.options());
    if (x.is_cuda()) {
#ifndef REPRO_CPU_ONLY
        size_t bytes;
        ck(repro_conv::conv2d_grad_weight_workspace(g, S, &bytes), "conv plan");
        Tensor ws = workspace(bytes, x);
        ck(repro_conv::repro_conv2d_grad_weight_ws(dy.data_ptr<float>(), x.data_ptr<float>(), dw.data_ptr<float>(), g, S,
                                                   ws.data_ptr<uint8_t>(), bytes), "conv weight grad");
        return dw;
#endif
    }
    repro_conv::cpu_conv2d_grad_weight(dy.data_ptr<float>(), x.data_ptr<float>(), dw.data_ptr<float>(), g, S);
    return dw;
}

Tensor conv2d_grad_input(Tensor dy, Tensor w, std::vector<int64_t> x_shape, int SY, int SX, int PY, int PX, int S) {
    require_float(dy, "dy");
    require_float(w, "weight");
    dy = dy.contiguous();
    w = w.contiguous();
    repro_conv::ConvShape g;
    g.N = (int)x_shape[0]; g.C = (int)x_shape[1]; g.H = (int)x_shape[2]; g.W = (int)x_shape[3];
    g.F = (int)w.size(0); g.KH = (int)w.size(2); g.KW = (int)w.size(3);
    g.SY = SY; g.SX = SX; g.PY = PY; g.PX = PX;
    Tensor dx = torch::empty(x_shape, dy.options());
    if (dy.is_cuda()) {
#ifndef REPRO_CPU_ONLY
        size_t bytes;
        ck(repro_conv::conv2d_grad_input_workspace(g, S, &bytes), "conv plan");
        Tensor ws = workspace(bytes, dy);
        ck(repro_conv::repro_conv2d_grad_input_ws(dy.data_ptr<float>(), w.data_ptr<float>(), dx.data_ptr<float>(), g, S,
                                                  ws.data_ptr<uint8_t>(), bytes), "conv input grad");
        return dx;
#endif
    }
    repro_conv::cpu_conv2d_grad_input(dy.data_ptr<float>(), w.data_ptr<float>(), dx.data_ptr<float>(), g, S);
    return dx;
}

py::tuple max_pool2d(Tensor x, int KH, int KW, int SY, int SX) {
    require_float(x, "x");
    x = x.contiguous();
    repro_nn::PoolShape g{(int)x.size(0), (int)x.size(1), (int)x.size(2), (int)x.size(3), KH, KW, SY, SX};
    Tensor out = torch::empty({g.N, g.C, g.OH(), g.OW()}, x.options());
    Tensor index = torch::empty({g.N, g.C, g.OH(), g.OW()}, torch::TensorOptions().dtype(torch::kInt32).device(x.device()));
    if (x.is_cuda()) {
#ifndef REPRO_CPU_ONLY
        ck(repro_nn::repro_max_pool2d(x.data_ptr<float>(), out.data_ptr<float>(), index.data_ptr<int>(), g), "pool");
        return py::make_tuple(out, index);
#endif
    }
    repro_nn::cpu_max_pool2d(x.data_ptr<float>(), out.data_ptr<float>(), index.data_ptr<int>(), g);
    return py::make_tuple(out, index);
}

Tensor max_pool2d_backward(Tensor dy, Tensor index, std::vector<int64_t> x_shape, int KH, int KW, int SY, int SX) {
    require_float(dy, "dy");
    dy = dy.contiguous();
    repro_nn::PoolShape g{(int)x_shape[0], (int)x_shape[1], (int)x_shape[2], (int)x_shape[3], KH, KW, SY, SX};
    Tensor dx = torch::empty(x_shape, dy.options());
    if (dy.is_cuda()) {
#ifndef REPRO_CPU_ONLY
        ck(repro_nn::repro_max_pool2d_backward(dy.data_ptr<float>(), index.data_ptr<int>(), dx.data_ptr<float>(), g), "pool backward");
        return dx;
#endif
    }
    repro_nn::cpu_max_pool2d_backward(dy.data_ptr<float>(), index.data_ptr<int>(), dx.data_ptr<float>(), g);
    return dx;
}

// Returns (loss, gradient state). Backward passes the state back: on the GPU it is the kernel's
// workspace, which holds the exp values and row sums; on the CPU backward recomputes them.
py::tuple cross_entropy(Tensor x, Tensor target, int ignore_index) {
    require_float(x, "logits");
    x = x.contiguous();
    target = target.to(torch::kInt32).contiguous();
    int rows = (int)x.size(0), cols = (int)x.size(1);
    if (x.is_cuda()) {
#ifndef REPRO_CPU_ONLY
        repro_math::CrossEntropyPlan plan;
        ck(repro_math::make_cross_entropy_plan(rows, cols, &plan), "cross-entropy plan");
        Tensor ws = workspace(plan.workspace_bytes, x);
        Tensor loss = torch::empty({}, x.options());
        ck(repro_math::repro_cross_entropy_fp32_ws(x.data_ptr<float>(), target.data_ptr<int>(), ignore_index, loss.data_ptr<float>(),
                                                   rows, cols, ws.data_ptr<uint8_t>(), plan.workspace_bytes), "cross-entropy");
        return py::make_tuple(loss, ws);
#endif
    }
    Tensor loss = torch::empty({}, x.options());
    *loss.data_ptr<float>() = repro_math::cross_entropy_cpu(x.data_ptr<float>(), target.data_ptr<int>(), ignore_index, rows, cols,
                                                            (float*)nullptr, 1.0f, par);
    return py::make_tuple(loss, x);
}

Tensor cross_entropy_grad(Tensor state, Tensor target, int ignore_index, int rows, int cols, float seed) {
    target = target.to(torch::kInt32).contiguous();
    Tensor grad = torch::empty({rows, cols}, torch::TensorOptions().dtype(torch::kFloat32).device(state.device()));
    if (state.is_cuda()) {
#ifndef REPRO_CPU_ONLY
        ck(repro_math::repro_cross_entropy_grad_fp32_ws(target.data_ptr<int>(), ignore_index, seed, grad.data_ptr<float>(), rows, cols,
                                                        state.data_ptr<uint8_t>(), (size_t)state.numel()), "cross-entropy grad");
        return grad;
#endif
    }
    repro_math::cross_entropy_cpu(state.data_ptr<float>(), target.data_ptr<int>(), ignore_index, rows, cols, grad.data_ptr<float>(), seed, par);
    return grad;
}

Tensor softmax(Tensor x) {
    require_float(x, "x");
    x = x.contiguous();
    int rows = (int)x.size(0), cols = (int)x.size(1);
    Tensor out = torch::empty({rows, cols}, x.options());
    if (x.is_cuda()) {
#ifndef REPRO_CPU_ONLY
        repro_math::SoftmaxPlan plan;
        ck(repro_math::make_softmax_plan(rows, cols, &plan), "softmax plan");
        Tensor ws = workspace(plan.workspace_bytes, x);
        ck(repro_math::repro_softmax_fp32_ws(x.data_ptr<float>(), out.data_ptr<float>(), rows, cols,
                                             ws.data_ptr<uint8_t>(), plan.workspace_bytes), "softmax");
        return out;
#endif
    }
    const float* p = x.data_ptr<float>();
    float* o = out.data_ptr<float>();
    par(rows, [&](int r) { repro_math::softmax_row(p + (size_t)r * cols, o + (size_t)r * cols, cols); });
    return out;
}

void adam(Tensor p, Tensor g, Tensor m, Tensor v, const repro_nn::AdamScalars& s) {
    require_float(p, "parameter");
    require_float(g, "gradient");
    if (!p.is_contiguous() || !m.is_contiguous() || !v.is_contiguous()) throw std::invalid_argument("adam: state must be contiguous");
    g = g.contiguous();
    size_t n = (size_t)p.numel();
    if (p.is_cuda()) {
#ifndef REPRO_CPU_ONLY
        ck(repro_nn::repro_adam(p.data_ptr<float>(), g.data_ptr<float>(), m.data_ptr<float>(), v.data_ptr<float>(), n, s), "adam");
        return;
#endif
    }
    repro_nn::cpu_adam(p.data_ptr<float>(), g.data_ptr<float>(), m.data_ptr<float>(), v.data_ptr<float>(), n, s);
}

// Lowers flag[0] to op_id if x holds a NaN or infinity, without synchronizing.
void scan_finite(Tensor x, Tensor flag, int op_id) {
    require_float(x, "x");
    x = x.contiguous();
    if (x.is_cuda()) {
#ifndef REPRO_CPU_ONLY
        ck(repro_scan_finite(x.data_ptr<float>(), flag.data_ptr<int>(), (size_t)x.numel(), op_id), "finite scan");
        return;
#endif
    }
    const float* p = x.data_ptr<float>();
    int* f = flag.data_ptr<int>();
    int64_t n = x.numel();
    for (int64_t i = 0; i < n; ++i)
        if ((repro_math::bits_of(p[i]) & 0x7F800000u) == 0x7F800000u) { if (op_id < *f) *f = op_id; break; }
}

// Python's float is converted to float32 once here, which any machine does identically.
struct AdamOptimizerClock {
    repro_nn::AdamConfig config;
    repro_nn::AdamClock clock;
    repro_nn::AdamScalars scalars;
    AdamOptimizerClock(float lr, float b1, float b2, float eps, float wd, bool adamw) {
        config.lr = lr; config.beta1 = b1; config.beta2 = b2; config.eps = eps; config.weight_decay = wd; config.adamw = adamw;
    }
    void next() { scalars = repro_nn::adam_next_step(config, clock); }
};

} // namespace

PYBIND11_MODULE(_backend, m) {
    m.def("gemm", &gemm);
    m.def("exact_sum", &exact_sum);
    m.def("conv2d", &conv2d);
    m.def("conv2d_grad_weight", &conv2d_grad_weight);
    m.def("conv2d_grad_input", &conv2d_grad_input);
    m.def("max_pool2d", &max_pool2d);
    m.def("max_pool2d_backward", &max_pool2d_backward);
    m.def("cross_entropy", &cross_entropy);
    m.def("cross_entropy_grad", &cross_entropy_grad);
    m.def("softmax", &softmax);
    m.def("scan_finite", &scan_finite);
    py::class_<AdamOptimizerClock>(m, "AdamClock")
        .def(py::init<float, float, float, float, float, bool>())
        .def("next", &AdamOptimizerClock::next)
        .def_property_readonly("step", [](const AdamOptimizerClock& c) { return c.clock.step; });
    m.def("adam", [](Tensor p, Tensor g, Tensor mm, Tensor v, const AdamOptimizerClock& c) { adam(p, g, mm, v, c.scalars); });
#ifdef REPRO_CPU_ONLY
    m.attr("has_cuda") = false;
#else
    m.attr("has_cuda") = true;
#endif
}
