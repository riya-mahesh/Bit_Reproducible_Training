# repro_torch: reproducible layers in PyTorch

Layers and an optimizer that replace the ones whose results depend on hardware or summation order. `loss.backward()` and the rest of the training loop work as usual.

```python
import repro_torch as rt
model = torch.nn.Sequential(rt.Conv2d(1, 6, 5, padding=2), rt.ReLU(), rt.MaxPool2d(2, 2), ...)
rt.name_layers(model)
rt.reproducible_init(model, seed=0)
optimizer = rt.Adam(model.parameters(), lr=0.01)
loss = rt.cross_entropy(model(x), y)
loss.backward()
optimizer.step()
rt.layers.guard().check(step)      # stops with a named error on NaN or infinity
```

| Layer | Kernels |
|---|---|
| `rt.Linear` | `repro_gemm`; bias gradient from `repro_sum` |
| `rt.Conv2d` | `repro_conv`; bias gradient from `repro_sum` |
| `rt.MaxPool2d`, `rt.Adam` | `repro_nn` |
| `rt.cross_entropy`, `rt.softmax` | `repro_math_cuda` (softmax is forward only for now) |
| `rt.ReLU` | A comparison, so a zero is always `+0` |

Stock `Flatten`, bias addition, and tensor indexing are kept: they are exact or a single IEEE operation.

**Non-finite values:** GEMM and convolution convert to integers, which is undefined for NaN and infinity. Each call scans its inputs without synchronizing, and `guard().check(step)` raises `FloatingPointError` naming the layer. Training always stops; there is no option to continue.

**Initial weights and data order:** `reproducible_init` and `shuffled` use integer arithmetic and single IEEE operations rather than a library random-number generator, so they don't depend on the machine.

## MNIST

`mnist_repro.py` trains LeNet-5 with RepDL's settings. `--mode stock` and `--mode repdl` run the same data and initial weights through other layers.

```bash
module load cuda-12.4
./run_mnist.sh                       # build, then ours twice and stock on each GPU
REPRO_CPU_ONLY=1 python3 setup.py build_ext --inplace   # CPU-only build
python3 mnist_repro.py --device cpu
```

`nvcc` must be available. The GPU build compiles all kernels in `src/kernels.cu`, with `-arch=native` unless `REPRO_ARCH` is set, so rebuild when moving to a different GPU model.

## Results

**Apple M4 Pro CPU, 2 epochs, S = 3, seed 0:**

| Run | Test accuracy | Median step | Weights hash |
|---|---|---|---|
| Ours, run 1 | 0.9826 | 35.3 ms | `6b8ce584…` |
| Ours, run 2 | 0.9826 | 35.1 ms | `6b8ce584…` |
| Stock PyTorch | 0.9734 | 19.3 ms | `9a4b3abb…` |

Accuracy is from a single seed, so the gap between the two says little. RepDL reports 0.9804 with its own seed and initialization. The two runs of ours match in every recorded hash and in the bits of every printed loss.

**GPU:** not built or run yet.
