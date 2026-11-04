# 1. Bit-reproducibility

## MNIST training

| Method | Seed | S | Configurations | All hashes identical | Final weights |
|---|---|---|---|---|---|
| repdl | 0 | 3 | 9 | yes | f3b851eed7b7 |
| repdl | 1 | 3 | 4 | yes | 4b35cbf8a5a8 |
| repdl | 2 | 3 | 3 | yes | 471bd14dcb33 |
| repdl | 3 | 3 | 3 | yes | bb29e0a2b685 |
| repdl | 4 | 3 | 3 | yes | 5698511868c3 |
| repro | 0 | 3 | 10 | yes | 6b8ce5843c58 |
| repro | 1 | 3 | 5 | yes | c955a8b61852 |
| repro | 2 | 3 | 3 | yes | 20eea7769625 |
| repro | 3 | 3 | 3 | yes | 7ad32ff9c8b3 |
| repro | 4 | 3 | 3 | yes | f5347082020d |
| stock | 0 | 3 | 10 | **NO: 17 different** | 0bbe25240374, 517156bfbb33, 52af3eecd803, 5743ea8c7074, 6171eef46940, 72e34f8f258f, 894adff7fb31, 92c3915f4645, 982b8f9b27c9, 9a4b3abb463c, b2c269afd02f, bf86d3aec2c9, e8aca18fc999, ed488893ae30, ee437b94a425, f0dd4e8e945d, f597bffc4499 |
| stock | 1 | 3 | 5 | **NO: 4 different** | 0ccea9bd3a49, 287b86ac5323, 8f66da8a3aa7, ebf23ceeee1b |
| stock | 2 | 3 | 3 | **NO: 3 different** | 652e6e5989b3, 7b18dfe70bdd, c8e0cce80b16 |
| stock | 3 | 3 | 3 | **NO: 3 different** | 226206ea41b5, adb290579353, c3bcdef5d843 |
| stock | 4 | 3 | 3 | **NO: 3 different** | 43240bd628be, 7d73bb9b8cb0, f5bd7439f7b0 |

Configurations compared:

- NVIDIA GeForce RTX 2080 Ti / cuda124 / torch 2.6.0+cu124 / wheel CUDA 12.4 / driver 595.104.02
- NVIDIA GeForce RTX 2080 Ti / cuda126 / torch 2.7.1+cu126 / wheel CUDA 12.6 / driver 595.104.02
- NVIDIA GeForce RTX 2080 Ti / cuda130 / torch 2.9.0+cu130 / wheel CUDA 13.0 / driver 595.104.02
- NVIDIA GeForce RTX 4070 / cuda124 / torch 2.6.0+cu124 / wheel CUDA 12.4 / driver 595.104.02
- NVIDIA GeForce RTX 4070 / cuda126 / torch 2.7.1+cu126 / wheel CUDA 12.6 / driver 595.104.02
- NVIDIA GeForce RTX 4070 / cuda130 / torch 2.9.0+cu130 / wheel CUDA 13.0 / driver 595.104.02
- NVIDIA TITAN V / cuda124 / torch 2.6.0+cu124 / wheel CUDA 12.4 / driver 580.196.07
- NVIDIA TITAN V / cuda126 / torch 2.7.1+cu126 / wheel CUDA 12.6 / driver 580.196.07
- arm / one_thread / torch 2.8.0 / wheel CUDA None / driver none
- arm / untagged / torch 2.8.0 / wheel CUDA None / driver none

## exp and log over all 2^32 inputs

| Function | Result files | Identical |
|---|---|---|
| repro_expf | 14 | yes |
| repro_logf | 14 | yes |

## Kernels: every suite operation

Of the result files, 6 are from GPUs and 7 from CPUs.

33 of 47 operations have a result from at least one GPU.

47 operations across 13 result files: all identical.

**Incomplete: 14 operations are missing from some result files, so they were not compared there.**

| Operation | Files with a result | Result files |
|---|---|---|
| cross_entropy MNIST loss | 7 | 13 |
| cross_entropy MNIST loss+grad | 7 | 13 |
| cross_entropy vocab loss | 7 | 13 |
| cross_entropy vocab loss+grad | 7 | 13 |
| lenet adam step | 7 | 13 |
| lenet conv1.dW | 7 | 13 |
| lenet conv1.fwd | 7 | 13 |
| lenet conv2.dW | 7 | 13 |
| lenet conv2.dX | 7 | 13 |
| lenet conv2.fwd | 7 | 13 |
| lenet pool1.back | 7 | 13 |
| lenet pool1.fwd | 7 | 13 |
| lenet pool2.back | 7 | 13 |
| lenet pool2.fwd | 7 | 13 |

## Test programs

| Log | Result |
|---|---|
| repro_conv_cuda/results/cuda2_cuda124/conv.log | pass |
| repro_conv_cuda/results/cuda2_cuda126/conv.log | pass |
| repro_conv_cuda/results/cuda2_cuda130/conv.log | pass |
| repro_conv_cuda/results/cuda3_cuda124/conv.log | pass |
| repro_conv_cuda/results/cuda3_cuda126/conv.log | pass |
| repro_conv_cuda/results/cuda5_cuda124/conv.log | pass |
| repro_math_cuda/results/cuda2_cuda124/check_math.log | pass |
| repro_math_cuda/results/cuda2_cuda124/softmax.log | pass |
| repro_math_cuda/results/cuda2_cuda126/softmax.log | pass |
| repro_math_cuda/results/cuda2_cuda130/softmax.log | pass |
| repro_math_cuda/results/cuda3_cuda124/check_math.log | pass |
| repro_math_cuda/results/cuda3_cuda124/softmax.log | pass |
| repro_math_cuda/results/cuda3_cuda126/softmax.log | pass |
| repro_math_cuda/results/cuda5_cuda124/softmax.log | pass |
| repro_math_cuda/results/mac/check_math.log | pass |
| repro_nn_cuda/results/cuda2_cuda124/nn.log | pass |
| repro_nn_cuda/results/cuda2_cuda126/nn.log | pass |
| repro_nn_cuda/results/cuda2_cuda130/nn.log | pass |
| repro_nn_cuda/results/cuda3_cuda124/nn.log | pass |
| repro_nn_cuda/results/cuda3_cuda126/nn.log | pass |
| repro_nn_cuda/results/cuda5_cuda124/nn.log | pass |
| repro_sum_cuda/results/cuda2_cuda124/sum.log | pass |
| repro_sum_cuda/results/cuda2_cuda126/sum.log | pass |
| repro_sum_cuda/results/cuda2_cuda130/sum.log | pass |
| repro_sum_cuda/results/cuda3_cuda124/sum.log | pass |
| repro_sum_cuda/results/cuda3_cuda126/sum.log | pass |
| repro_sum_cuda/results/cuda5_cuda124/sum.log | pass |

**Our results so far: no differences found; the suite comparison is INCOMPLETE, see above**

# 2. Accuracy after 2 epochs (mean ± standard deviation over seeds)

A method with identical results everywhere counts once per seed. Stock differs between runs, so every GPU run in a configuration with all the seeds counts.

| Method | Seeds | Runs counted | Train accuracy | Test accuracy | Last batch loss |
|---|---|---|---|---|---|
| repro | 5 | 5 | 0.9814 ± 0.0029 | 0.9800 ± 0.0028 | 0.107 ± 0.080 |
| repdl | 5 | 5 | 0.9808 ± 0.0007 | 0.9796 ± 0.0013 | 0.150 ± 0.123 |
| stock | 5 | 15 | 0.9829 ± 0.0024 | 0.9807 ± 0.0029 | 0.112 ± 0.071 |

# 3. Speed: median ms per training step

Same GPU, configuration, data, order, and initial weights. The ratio is against RepDL (above 1 means slower than RepDL).

| Configuration | Ours | RepDL | Stock | Ours / RepDL | Ours / stock | Ours peak MiB | RepDL peak MiB | Stock peak MiB |
|---|---|---|---|---|---|---|---|---|
| NVIDIA GeForce RTX 2080 Ti / cuda124 / torch 2.6.0+cu124 / wheel CUDA 12.4 / driver 595.104.02 | 3.50 (4.3 s) | 14.82 (17.9 s) | 2.80 (3.6 s) | 0.24 | 1.25 | 222 | 219 | 236 |
| NVIDIA GeForce RTX 2080 Ti / cuda126 / torch 2.7.1+cu126 / wheel CUDA 12.6 / driver 595.104.02 | 3.45 (4.2 s) | 14.50 (17.5 s) | 2.46 (3.4 s) | 0.24 | 1.40 | 222 | 219 | 239 |
| NVIDIA GeForce RTX 2080 Ti / cuda130 / torch 2.9.0+cu130 / wheel CUDA 13.0 / driver 595.104.02 | 3.48 (4.3 s) | 14.52 (17.6 s) | 2.78 (3.7 s) | 0.24 | 1.25 | 222 | 219 | 236 |
| NVIDIA GeForce RTX 4070 / cuda124 / torch 2.6.0+cu124 / wheel CUDA 12.4 / driver 595.104.02 | 10.89 (11.5 s) | 14.10 (17.1 s) | 8.89 (10.3 s) | 0.77 | 1.22 | 222 | 219 | 235 |
| NVIDIA GeForce RTX 4070 / cuda126 / torch 2.7.1+cu126 / wheel CUDA 12.6 / driver 595.104.02 | 10.90 (12.3 s) | 13.95 (16.9 s) | 8.80 (10.0 s) | 0.78 | 1.24 | 222 | 219 | 235 |
| NVIDIA GeForce RTX 4070 / cuda130 / torch 2.9.0+cu130 / wheel CUDA 13.0 / driver 595.104.02 | 10.60 (12.0 s) | 14.01 (17.0 s) | 9.10 (10.8 s) | 0.76 | 1.16 | 222 | 219 | 240 |
| NVIDIA TITAN V / cuda124 / torch 2.6.0+cu124 / wheel CUDA 12.4 / driver 580.196.07 | 3.19 (3.9 s) | 14.90 (17.5 s) | 1.84 (2.5 s) | 0.21 | 1.73 | 222 | 219 | 235 |
| NVIDIA TITAN V / cuda126 / torch 2.7.1+cu126 / wheel CUDA 12.6 / driver 580.196.07 | 3.13 (3.8 s) | 13.70 (16.6 s) | 2.11 (2.9 s) | 0.23 | 1.48 | 222 | 219 | 240 |
| arm / one_thread / torch 2.8.0 / wheel CUDA None / driver none | 60.79 (73.5 s) | 175.88 (211.3 s) | 12.95 (15.6 s) | 0.35 | 4.69 | 0 | 0 | 0 |
| arm / untagged / torch 2.8.0 / wheel CUDA None / driver none | 34.85 (42.4 s) | - | 19.17 (23.0 s) | - | 1.82 | 0 | - | 0 |

Where the time goes (median ms; each part ends with a synchronization):

| Configuration | Method | Forward | Backward | Optimizer |
|---|---|---|---|---|
| NVIDIA GeForce RTX 2080 Ti / cuda124 / torch 2.6.0+cu124 / wheel CUDA 12.4 / driver 595.104.02 | repdl | 1.24 | 10.12 | 3.73 |
| NVIDIA GeForce RTX 2080 Ti / cuda124 / torch 2.6.0+cu124 / wheel CUDA 12.4 / driver 595.104.02 | repro | 1.83 | 1.88 | 0.28 |
| NVIDIA GeForce RTX 2080 Ti / cuda124 / torch 2.6.0+cu124 / wheel CUDA 12.4 / driver 595.104.02 | repro | 1.54 | 2.06 | 0.23 |
| NVIDIA GeForce RTX 2080 Ti / cuda124 / torch 2.6.0+cu124 / wheel CUDA 12.4 / driver 595.104.02 | stock | 0.89 | 1.53 | 0.52 |
| NVIDIA GeForce RTX 2080 Ti / cuda126 / torch 2.7.1+cu126 / wheel CUDA 12.6 / driver 595.104.02 | repdl | 1.20 | 10.05 | 3.34 |
| NVIDIA GeForce RTX 2080 Ti / cuda126 / torch 2.7.1+cu126 / wheel CUDA 12.6 / driver 595.104.02 | repro | 1.44 | 1.98 | 0.24 |
| NVIDIA GeForce RTX 2080 Ti / cuda126 / torch 2.7.1+cu126 / wheel CUDA 12.6 / driver 595.104.02 | stock | 0.85 | 1.47 | 0.51 |
| NVIDIA GeForce RTX 2080 Ti / cuda130 / torch 2.9.0+cu130 / wheel CUDA 13.0 / driver 595.104.02 | repdl | 1.23 | 10.38 | 3.54 |
| NVIDIA GeForce RTX 2080 Ti / cuda130 / torch 2.9.0+cu130 / wheel CUDA 13.0 / driver 595.104.02 | repro | 1.46 | 1.69 | 0.22 |
| NVIDIA GeForce RTX 2080 Ti / cuda130 / torch 2.9.0+cu130 / wheel CUDA 13.0 / driver 595.104.02 | stock | 0.87 | 1.47 | 0.53 |
| NVIDIA GeForce RTX 4070 / cuda124 / torch 2.6.0+cu124 / wheel CUDA 12.4 / driver 595.104.02 | repdl | 1.45 | 8.00 | 5.17 |
| NVIDIA GeForce RTX 4070 / cuda124 / torch 2.6.0+cu124 / wheel CUDA 12.4 / driver 595.104.02 | repro | 5.12 | 5.17 | 0.88 |
| NVIDIA GeForce RTX 4070 / cuda124 / torch 2.6.0+cu124 / wheel CUDA 12.4 / driver 595.104.02 | stock | 2.82 | 4.48 | 1.82 |
| NVIDIA GeForce RTX 4070 / cuda126 / torch 2.7.1+cu126 / wheel CUDA 12.6 / driver 595.104.02 | repdl | 1.49 | 8.09 | 5.25 |
| NVIDIA GeForce RTX 4070 / cuda126 / torch 2.7.1+cu126 / wheel CUDA 12.6 / driver 595.104.02 | repro | 5.04 | 4.96 | 0.95 |
| NVIDIA GeForce RTX 4070 / cuda126 / torch 2.7.1+cu126 / wheel CUDA 12.6 / driver 595.104.02 | stock | 2.82 | 4.40 | 1.84 |
| NVIDIA GeForce RTX 4070 / cuda130 / torch 2.9.0+cu130 / wheel CUDA 13.0 / driver 595.104.02 | repdl | 1.43 | 8.02 | 4.98 |
| NVIDIA GeForce RTX 4070 / cuda130 / torch 2.9.0+cu130 / wheel CUDA 13.0 / driver 595.104.02 | repro | 4.90 | 4.99 | 0.86 |
| NVIDIA GeForce RTX 4070 / cuda130 / torch 2.9.0+cu130 / wheel CUDA 13.0 / driver 595.104.02 | stock | 2.81 | 4.61 | 1.75 |
| NVIDIA TITAN V / cuda124 / torch 2.6.0+cu124 / wheel CUDA 12.4 / driver 580.196.07 | repdl | 0.90 | 10.83 | 2.36 |
| NVIDIA TITAN V / cuda124 / torch 2.6.0+cu124 / wheel CUDA 12.4 / driver 580.196.07 | repro | 1.07 | 1.91 | 0.20 |
| NVIDIA TITAN V / cuda124 / torch 2.6.0+cu124 / wheel CUDA 12.4 / driver 580.196.07 | repro | 1.07 | 2.12 | 0.20 |
| NVIDIA TITAN V / cuda124 / torch 2.6.0+cu124 / wheel CUDA 12.4 / driver 580.196.07 | stock | 0.61 | 0.85 | 0.46 |
| NVIDIA TITAN V / cuda126 / torch 2.7.1+cu126 / wheel CUDA 12.6 / driver 580.196.07 | repdl | 0.88 | 10.81 | 2.31 |
| NVIDIA TITAN V / cuda126 / torch 2.7.1+cu126 / wheel CUDA 12.6 / driver 580.196.07 | repro | 1.05 | 2.09 | 0.21 |
| NVIDIA TITAN V / cuda126 / torch 2.7.1+cu126 / wheel CUDA 12.6 / driver 580.196.07 | stock | 0.62 | 1.10 | 0.46 |
| arm / untagged / torch 2.8.0 / wheel CUDA None / driver none | repro | 5.87 | 28.51 | 0.21 |

RepDL's own example (its README publishes trained-model hash 31ee86a7f75d...):

| Log | Starting weights | Published trained hash |
|---|---|---|
| results/mnist/cuda2_cuda124/gpu0_NVIDIA_GeForce_RTX_2080_Ti_repdl_official.log | 2a2d133895b1 | matches |
| results/mnist/cuda2_cuda124/gpu1_NVIDIA_GeForce_RTX_2080_Ti_repdl_official.log | 2a2d133895b1 | matches |
| results/mnist/cuda2_cuda126/gpu0_NVIDIA_GeForce_RTX_2080_Ti_repdl_official.log | 2a2d133895b1 | matches |
| results/mnist/cuda2_cuda130/gpu0_NVIDIA_GeForce_RTX_2080_Ti_repdl_official.log | 2a2d133895b1 | matches |
| results/mnist/cuda3_cuda124/gpu0_NVIDIA_TITAN_V_repdl_official.log | 2a2d133895b1 | matches |
| results/mnist/cuda3_cuda124/gpu1_NVIDIA_TITAN_V_repdl_official.log | 2a2d133895b1 | matches |
| results/mnist/cuda3_cuda126/gpu0_NVIDIA_TITAN_V_repdl_official.log | 2a2d133895b1 | matches |
| results/mnist/cuda5_cuda124/gpu0_NVIDIA_GeForce_RTX_4070_repdl_official.log | 3ac26bdaa251 | **differs** |
| results/mnist/cuda5_cuda126/gpu0_NVIDIA_GeForce_RTX_4070_repdl_official.log | 3ac26bdaa251 | **differs** |
| results/mnist/cuda5_cuda130/gpu0_NVIDIA_GeForce_RTX_4070_repdl_official.log | 3ac26bdaa251 | **differs** |

