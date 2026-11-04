"""Initial weights, shuffling, and hashes that every machine computes identically: only integer
arithmetic and single float32 IEEE operations, with no library random-number generator."""

import hashlib

import numpy as np
import torch


class SplitMix64:
    def __init__(self, seed):
        self.state = seed & (2 ** 64 - 1)

    def next(self):
        mask = 2 ** 64 - 1
        self.state = (self.state + 0x9E3779B97F4A7C15) & mask
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
        return z ^ (z >> 31)


def _uniform(shape, bound, rng):
    """float32 values uniform in [-bound, bound): u = (24 random bits) / 2^24, then 2 b u - b."""
    n = int(np.prod(shape)) if len(shape) else 1
    bits = np.array([rng.next() >> 40 for _ in range(n)], dtype=np.float32)   # below 2^24: exact
    u = bits * np.float32(2.0 ** -24)                                          # exact
    v = (u * (np.float32(2.0) * bound)) - bound
    return torch.from_numpy(v.astype(np.float32).reshape(shape).copy())


@torch.no_grad()
def reproducible_init(model, seed=0):
    """Same distribution as PyTorch's defaults for Linear and Conv2d (weights and biases uniform
    in +/- 1 / sqrt(fan_in)), but identical bits on every machine."""
    rng = SplitMix64(seed)
    for module in model.modules():
        weight = getattr(module, "weight", None)
        if weight is None:
            continue
        fan_in = int(np.prod(weight.shape[1:]))
        bound = np.float32(1.0) / np.sqrt(np.float32(fan_in))
        weight.copy_(_uniform(tuple(weight.shape), bound, rng))
        bias = getattr(module, "bias", None)
        if bias is not None:
            bias.copy_(_uniform(tuple(bias.shape), bound, rng))
    return model


def shuffled(n, rng):
    """A permutation of 0..n-1 by Fisher-Yates, from the integer generator."""
    order = list(range(n))
    for i in range(n - 1, 0, -1):
        j = rng.next() % (i + 1)
        order[i], order[j] = order[j], order[i]
    return order


def tensor_hash(x):
    return hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest()


def model_hash(model):
    h = hashlib.sha256()
    for p in model.parameters():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()
