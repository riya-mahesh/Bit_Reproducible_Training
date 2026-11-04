# Reproducible MaxPool2d and Adam

The definitions are in `repro_nn_math.cuh`. The same scalar code runs on the GPU (`repro_nn.cu`) and the CPU (`repro_nn_cpu.h`).

**MaxPool2d:** each output is the first largest entry in row-major order, and a tie keeps the first. ReLU output has many equal zeros, so without a rule the gradient could go to a different pixel on different hardware. Backward sends each gradient to its recorded index. With overlapping windows, gradients that land on one pixel are summed exactly and rounded once. Padding, dilation, and `ceil_mode` aren't supported; LeNet-5 doesn't use them.

**Adam and AdamW:** each operation rounds once, in a fixed order, with no fused multiply-adds. `b1^t` and `b2^t` come from repeated float32 multiplication rather than a library `pow`, which is what RepDL's Python `**` depends on.

## Results

**CPU** (Apple M4 Pro): every Adam rounding over 25 steps and 3 configurations matches an exact-rational recomputation. MaxPool outputs and indices match `torch.nn.functional.max_pool2d` on inputs without ties. In double, backward differs from PyTorch's by at most 1.8e-7 and Adam's weights by at most 1.7e-6 after 25 steps.

**GPU:** not run yet. `./run_nn.sh`.
