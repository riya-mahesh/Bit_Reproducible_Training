"""Bit-reproducible layers for PyTorch: these replace the layers whose results depend on hardware
or summation order, and loss.backward() works as usual."""

import torch  # loads the libraries the extension links against

from . import _backend
from .layers import (Adam, Conv2d, CrossEntropyLoss, FiniteGuard, Linear, MaxPool2d, ReLU, cross_entropy,
                     get_slices, name_layers, set_slices, softmax)
from .utils import SplitMix64, model_hash, reproducible_init, shuffled, tensor_hash

has_cuda = _backend.has_cuda

__all__ = ["Adam", "Conv2d", "CrossEntropyLoss", "FiniteGuard", "Linear", "MaxPool2d", "ReLU", "SplitMix64",
           "cross_entropy", "get_slices", "has_cuda", "model_hash", "name_layers", "reproducible_init", "set_slices",
           "shuffled", "softmax", "tensor_hash"]
