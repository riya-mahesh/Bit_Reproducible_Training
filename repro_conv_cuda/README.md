# Reproducible Conv2d

`repro_conv` computes a 2D convolution and its two gradients on FP32 using `repro_gemm`'s logic: power-of-two scales, one rounding to fixed-point integers, exact INT64 sums, and one rounding to FP32. It reads the image directly, with no im2col buffer. The exact definition is in `repro_conv_math.cuh`.

| Function | Computes | Scales |
|---|---|---|
| `repro_conv2d_ws` | Output from input and weights | Per window, per filter |
| `repro_conv2d_grad_weight_ws` | Weight gradient | Per filter, per window position |
| `repro_conv2d_grad_input_ws` | Input gradient | Per image, all weights together |

The forward pass and weight gradient are bit-identical to `repro_gemm` on the im2col matrix. A pixel of the input gradient sums over several windows and all filters, so its terms need one shared scale, which GEMM's per-row and per-column scales don't provide.

Bias is separate: add it with one IEEE addition, and get its gradient from `repro_sum`. Supports stride and zero padding, but not groups or dilation. Limits: K ≤ 1024 values per window, and at most 256 filters.

## Results

**CPU** (`check_repro_conv.cpp`, on an Apple M4 Pro): passes. Forward and weight gradient match the GEMM definition bit for bit at S = 1, 2, 3 and several strides and paddings. The adjoint identities `<dY, Y> = <dX, x> = <dW, W>` hold (about 1e-8 relative at S=3), which shows both gradients are correct. Forward error against the true convolution is about 1e-6 at S=3.

**GPU:** not run yet. `./run_conv.sh`.
