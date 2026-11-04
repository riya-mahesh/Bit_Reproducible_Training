# 1. Bit-reproducibility

## MNIST training

| Method | Seed | S | Configurations | All hashes identical | Final weights |
|---|---|---|---|---|---|
| repro | 0 | 3 | 1 | yes | 6b8ce5843c58 |
| repro | 1 | 3 | 1 | yes | c955a8b61852 |
| stock | 0 | 3 | 1 | yes | 9a4b3abb463c |
| stock | 1 | 3 | 1 | yes | 8f66da8a3aa7 |

Configurations compared:

- arm / untagged / torch 2.8.0 / wheel CUDA None / driver none

## exp and log over all 2^32 inputs

| Function | Result files | Identical |
|---|---|---|
| repro_expf | 1 | yes |
| repro_logf | 1 | yes |

## Kernels: every suite operation

47 operations across 1 result files: all identical.

## Test programs

| Log | Result |
|---|---|
| repro_math_cuda/results/mac/check_math.log | pass |

**Our results so far: no differences found**

# 2. Accuracy after 2 epochs (mean ± standard deviation over seeds)

| Method | Seeds | Train accuracy | Test accuracy | Last batch loss |
|---|---|---|---|---|
| repro | 2 | 0.9831 ± 0.0007 | 0.9817 ± 0.0013 | 0.172 ± 0.093 |
| stock | 2 | 0.9800 ± 0.0049 | 0.9773 ± 0.0055 | 0.237 ± 0.039 |

# 3. Speed: median ms per training step

Same GPU, configuration, data, order, and initial weights. The ratio is against RepDL (above 1 means slower than RepDL).

| Configuration | Ours | RepDL | Stock | Ours / RepDL | Ours / stock | Our peak MiB |
|---|---|---|---|---|---|---|
| arm / untagged / torch 2.8.0 / wheel CUDA None / driver none | 34.85 (42.4 s) | - | 19.17 (23.0 s) | - | 1.82 | 0 |

Where the time goes (median ms; each part ends with a synchronization):

| Configuration | Method | Forward | Backward | Optimizer |
|---|---|---|---|---|
| arm / untagged / torch 2.8.0 / wheel CUDA None / driver none | repro | 5.87 | 28.51 | 0.21 |

