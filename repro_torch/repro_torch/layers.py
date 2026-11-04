"""Layers and optimizer built on the reproducible kernels, as custom autograd functions."""

import torch

from . import _backend as B

_slices = 3


def set_slices(s):
    """Sets S for every GEMM and convolution: 21-bit inputs at the default S = 3."""
    global _slices
    _slices = s


def get_slices():
    return _slices


class FiniteGuard:
    """Stops training with a clear error once a NaN or infinity reaches a GEMM or convolution.

    Their integer conversion is undefined for such values, so results past that point could
    differ between machines. Scans are queued with the operation and need no synchronization;
    check() reads the result once per step, so the error arrives at the same step everywhere.
    """

    _NONE = 2 ** 31 - 1

    def __init__(self):
        self._names = []
        self._ids = {}
        self._flag = None

    def _id(self, name):
        if name not in self._ids:
            self._ids[name] = len(self._names)
            self._names.append(name)
        return self._ids[name]

    def scan(self, x, name):
        if self._flag is None or self._flag.device != x.device:
            self._flag = torch.full((1,), self._NONE, dtype=torch.int32, device=x.device)
        B.scan_finite(x, self._flag, self._id(name))

    def check(self, step=None):
        if self._flag is None:
            return
        found = int(self._flag.item())
        if found != self._NONE:
            where = "" if step is None else " at step %d" % step
            raise FloatingPointError("non-finite value reached %s%s" % (self._names[found], where))


_guard = FiniteGuard()


def guard():
    return _guard


def name_layers(model, prefix=""):
    """Names layers after their place in the model, for error messages."""
    for name, m in model.named_modules():
        if isinstance(m, (Linear, Conv2d)):
            m.repro_name = prefix + name
    return model


class _LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, name):
        ctx.save_for_backward(x, weight)
        ctx.name = name
        _guard.scan(x, name + " input")
        _guard.scan(weight, name + " weight")
        m, k, n = x.shape[0], weight.shape[1], weight.shape[0]
        y = B.gemm(x, weight, m, n, k, False, True, _slices)
        return y + bias if bias is not None else y

    @staticmethod
    def backward(ctx, dy):
        x, weight = ctx.saved_tensors
        _guard.scan(dy, ctx.name + " gradient")
        m, k, n = x.shape[0], weight.shape[1], weight.shape[0]
        dx = dw = db = None
        if ctx.needs_input_grad[0]:
            dx = B.gemm(dy, weight, m, k, n, False, False, _slices)
        if ctx.needs_input_grad[1]:
            dw = B.gemm(dy, x, n, k, m, True, False, _slices)
        if ctx.needs_input_grad[2]:
            db = B.exact_sum(dy, m, n, 1, False)
        return dx, dw, db, None


class Linear(torch.nn.Linear):
    repro_name = "linear"

    def forward(self, x):
        return _LinearFn.apply(x, self.weight, self.bias, self.repro_name)


class _ConvFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding, name):
        ctx.save_for_backward(x, weight)
        ctx.stride, ctx.padding, ctx.name = stride, padding, name
        _guard.scan(x, name + " input")
        _guard.scan(weight, name + " weight")
        y = B.conv2d(x, weight, stride[0], stride[1], padding[0], padding[1], _slices)
        return y + bias[None, :, None, None] if bias is not None else y

    @staticmethod
    def backward(ctx, dy):
        x, weight = ctx.saved_tensors
        sy, sx = ctx.stride
        py, px = ctx.padding
        _guard.scan(dy, ctx.name + " gradient")
        dx = dw = db = None
        if ctx.needs_input_grad[0]:
            dx = B.conv2d_grad_input(dy, weight, list(x.shape), sy, sx, py, px, _slices)
        if ctx.needs_input_grad[1]:
            dw = B.conv2d_grad_weight(dy, x, weight.shape[2], weight.shape[3], sy, sx, py, px, _slices)
        if ctx.needs_input_grad[2]:
            db = B.exact_sum(dy, dy.shape[0], dy.shape[1], dy.shape[2] * dy.shape[3], False)
        return dx, dw, db, None, None, None


class Conv2d(torch.nn.Conv2d):
    repro_name = "conv2d"

    def forward(self, x):
        if self.groups != 1 or self.dilation != (1, 1) or isinstance(self.padding, str) or self.padding_mode != "zeros":
            raise NotImplementedError("repro_torch.Conv2d supports stride and zero padding only")
        return _ConvFn.apply(x, self.weight, self.bias, self.stride, self.padding, self.repro_name)


class _ReluFn(torch.autograd.Function):
    # Defined by a comparison, so the output and gradient are exact, and a zero is always +0
    # rather than whatever sign an implementation's max(x, 0) leaves.
    @staticmethod
    def forward(ctx, x):
        keep = x > 0
        ctx.keep = keep
        return torch.where(keep, x, torch.zeros_like(x))

    @staticmethod
    def backward(ctx, dy):
        return torch.where(ctx.keep, dy, torch.zeros_like(dy))


class ReLU(torch.nn.Module):
    def forward(self, x):
        return _ReluFn.apply(x)


class _MaxPoolFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, kernel, stride):
        y, index = B.max_pool2d(x, kernel[0], kernel[1], stride[0], stride[1])
        ctx.index, ctx.kernel, ctx.stride, ctx.x_shape = index, kernel, stride, list(x.shape)
        return y

    @staticmethod
    def backward(ctx, dy):
        k, s = ctx.kernel, ctx.stride
        return B.max_pool2d_backward(dy, ctx.index, ctx.x_shape, k[0], k[1], s[0], s[1]), None, None


def _pair(v):
    return (v, v) if isinstance(v, int) else tuple(v)


class MaxPool2d(torch.nn.Module):
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride if stride is not None else kernel_size)

    def forward(self, x):
        return _MaxPoolFn.apply(x, self.kernel_size, self.stride)


class _CrossEntropyFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, target, ignore_index):
        loss, state = B.cross_entropy(logits, target, ignore_index)
        ctx.state, ctx.target, ctx.ignore_index, ctx.shape = state, target, ignore_index, logits.shape
        return loss

    @staticmethod
    def backward(ctx, seed):
        rows, cols = ctx.shape
        return B.cross_entropy_grad(ctx.state, ctx.target, ctx.ignore_index, rows, cols, float(seed.item())), None, None


def cross_entropy(logits, target, ignore_index=-100):
    return _CrossEntropyFn.apply(logits, target, ignore_index)


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self, ignore_index=-100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        return cross_entropy(logits, target, self.ignore_index)


def softmax(x):
    """Reproducible softmax over the last dimension of a [rows, cols] tensor. Forward only for now."""
    return B.softmax(x)


class Adam(torch.optim.Optimizer):
    """Adam, or AdamW with adamw=True, with one defined rounding per operation."""

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, adamw=False):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, adamw=adamw))
        self._clocks = {}

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise NotImplementedError("repro_torch.Adam does not take a closure")
        for i, group in enumerate(self.param_groups):
            clock = self._clocks.get(i)
            if clock is None:
                b1, b2 = group["betas"]
                clock = B.AdamClock(group["lr"], b1, b2, group["eps"], group["weight_decay"], group["adamw"])
                self._clocks[i] = clock
            clock.next()
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                B.adam(p, p.grad, state["m"], state["v"], clock)
