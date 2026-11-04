# MNIST results: bit-reproducible training

LeNet-5 on MNIST, comparing our method with RepDL and stock PyTorch on 3 GPU models and 3 CUDA versions at NYU CIMS.

## Main conclusions

1. **Our training is bit-identical on everything tested.** Final weights and logits match on all 8 GPU and CUDA combinations and on an Apple ARM CPU.
2. **RepDL's training is also bit-identical when it starts from fixed weights.** This comparison doesn't show ours to be more reproducible than RepDL's. Stock PyTorch isn't reproducible.
3. **Accuracy shows no meaningful difference.** Test accuracy is 98.00% for ours, 97.96% for RepDL and 98.07% for stock, with gaps well inside the spread between seeds.
4. **Ours is faster than RepDL on all 3 GPUs:** 4.2–4.3× on the RTX 2080 Ti, 4.3–4.6× on the TITAN V, and 1.3× on the RTX 4070. The RTX 4070 figure comes with a caveat (see [Speed](#speed)).
5. **Ours is slower than stock PyTorch** by 1.2–1.7×, depending on the GPU.

## Setup

### Hardware and software

| Server | GPU | Generation | Memory | INT8 Tensor Cores | FP64 speed | Driver | CPU |
|---|---|---|---|---|---|---|---|
| `cuda2` | 2× RTX 2080 Ti | Turing | 11 GB | Yes | 1/32 | 595.104.02 | Xeon E5-2660 v3, 40 threads |
| `cuda3` | 2× TITAN V | Volta | 12 GB | No | 1/2 | 580.196.07 | 48 threads (model not recorded) |
| `cuda5` | 1× RTX 4070 | Ada | 12 GB | Yes | 1/64 | 595.104.02 | Xeon E5-2650 v2, 16 threads, no AVX2 |
| Mac | None (CPU only) | Apple ARM | n/a | n/a | n/a | n/a | 14 threads |

| Setup | CUDA | PyTorch | Ran on |
|---|---|---|---|
| `cuda124` | 12.4 | 2.6.0 | RTX 2080 Ti, TITAN V, RTX 4070 |
| `cuda126` | 12.6 | 2.7.1 | RTX 2080 Ti, TITAN V, RTX 4070 |
| `cuda130` | 13.0 | 2.9.0 | RTX 2080 Ti, RTX 4070 |

The Mac used PyTorch 2.8.0. FP64 speeds are nominal figures, not measured here. RepDL is commit `acd95bb`.

### Training settings, the same for all methods

| Setting | Value |
|---|---|
| Model | LeNet-5, 61,706 parameters |
| Data | MNIST: 60,000 training and 10,000 test images, same files checked by hash |
| Batch and length | Batch 100, 2 epochs, 1,200 steps |
| Optimizer | Adam: learning rate 0.01, betas (0.9, 0.999), eps 1e-8, no weight decay |
| Loss | Mean cross-entropy |
| Starting weights and shuffling | Same seeded SplitMix64 generator for all methods, seeds 0–4 |
| Our precision setting | S = 3 slices, FP32 |

Each method uses its own layers, loss and Adam. Ours also checks for NaN and infinity on every step, and that time is counted. RepDL runs with `use_deterministic_algorithms(True)`, as in its own example. Stock is ordinary PyTorch without deterministic settings.

### Runs

| | Ours | RepDL | Stock |
|---|---|---|---|
| GPU runs | 52 | 28 | 28 |
| Mac CPU runs | 5 | 2 | 4 |

Seed 0 ran on all 8 GPU and CUDA combinations and the Mac. Seed 1 ran on all 3 GPUs at CUDA 12.4 and the Mac. Seeds 2–4 ran on all 3 GPUs at CUDA 12.4. Ours also ran seed 0 on the second card of `cuda2` and `cuda3`.

## Bit-reproducibility

The hash is SHA-256 over the raw bytes of all weights after training. The train and test logits are hashed the same way and also match wherever the weights match.

| Seed | Starting weights, all methods | Ours: final weights | RepDL: final weights | Stock: distinct GPU results |
|---|---|---|---|---|
| 0 | `aaed3ac81bea` | `6b8ce5843c58` | `f3b851eed7b7` | 16 of 16 runs |
| 1 | `7da4f3954ddb` | `c955a8b61852` | `4b35cbf8a5a8` | 3 of 3 |
| 2 | `e618c78846ab` | `20eea7769625` | `471bd14dcb33` | 3 of 3 |
| 3 | `1c01656ccece` | `7ad32ff9c8b3` | `bb29e0a2b685` | 3 of 3 |
| 4 | `cd386c006d5d` | `f5347082020d` | `5698511868c3` | 3 of 3 |

Within each seed, every run of ours gave the one hash shown, and so did every run of RepDL. MNIST data hash for all runs: `55c41ac0f5a8`.

Our seed 0 hash also didn't change with the NaN scan on or off, or with 1 or 14 CPU threads on the Mac.

Stock's seed 0 test accuracy ranged from 97.45% to 98.39% from the same starting weights. Each stock setup ran seed 0 twice, one regular run and one profiled run, and each pair differed.

### Supporting checks

| Check | Result |
|---|---|
| `repro_expf` and `repro_logf`, all 2³² inputs | Identical on 11 GPU runs and 3 CPUs: `4ccc7cddaea06b26` and `117fa2a6bbdea567` |
| Exp and log against a higher-precision reference on the CPU | 0 wrong of 2³² inputs each; 0 of 19 known hard cases wrong |
| Conv, pool, Adam, softmax, exact-sum and CPU exp/log test programs | All 27 logs pass |
| Operation suite, GPU vs. Mac hashes | 33 of 47 operations match, on GPU 0 in 6 setups. The other 14 weren't reached (see [Limits](#limits)) |

The exp, log, test-program and suite checks were stopped before they ran on the RTX 4070 at CUDA 12.6 and 13.0.

### RepDL's published example

RepDL's README publishes trained-model hash `31ee86a7f75d…`. Running its own example gave:

| Where | Starting weights | Trained model | Test accuracy | Published hash |
|---|---|---|---|---|
| Mac CPU | `2a2d133895b1` | `31ee86a7f75d` | 98.04% | Matches |
| RTX 2080 Ti (12.4, 12.6, 13.0) | `2a2d133895b1` | `31ee86a7f75d` | 98.04% | Matches |
| TITAN V (12.4, 12.6) | `2a2d133895b1` | `31ee86a7f75d` | 98.04% | Matches |
| RTX 4070 (12.4, 12.6, 13.0) | `3ac26bdaa251` | `021a79d5cff9` | 98.15% | **Differs** |

The cause is PyTorch's default weight initialization, not RepDL's arithmetic. The starting weights already differ before training, and the example creates them on the CPU. The check below used the first convolution layer with the GPU hidden:

| Run | PyTorch CPU code | Weight hash |
|---|---|---|
| `cuda2`, normal | AVX2 | `9514b5db8089` |
| `cuda2`, forced with `ATEN_CPU_CAPABILITY=default` | Non-AVX2 | `1a085059b25e` |
| `cuda5`, normal (CPU has no AVX2) | Non-AVX2 | `1a085059b25e` |

So the same seed doesn't guarantee the same starting weights across CPUs, and bit-reproducible training has to fix the initialization too.

## Accuracy

Mean ± standard deviation after 2 epochs.

| Method | Seeds | Runs counted | Train accuracy | Test accuracy | Last batch loss |
|---|---|---|---|---|---|
| Ours | 5 | 5 | 98.14% ± 0.29 | 98.00% ± 0.28 | 0.107 ± 0.080 |
| RepDL | 5 | 5 | 98.08% ± 0.07 | 97.96% ± 0.13 | 0.150 ± 0.123 |
| Stock | 5 | 15 | 98.29% ± 0.24 | 98.07% ± 0.29 | 0.112 ± 0.071 |

Ours and RepDL give the same result on every machine, so each seed counts once. Stock differs between GPUs, so its row counts all 3 GPUs at CUDA 12.4.

| Seed | Ours train / test | RepDL train / test |
|---|---|---|
| 0 | 98.36 / 98.26 | 98.03 / 97.76 |
| 1 | 98.26 / 98.07 | 98.15 / 98.06 |
| 2 | 98.11 / 97.94 | 98.01 / 98.06 |
| 3 | 98.30 / 98.19 | 98.15 / 97.88 |
| 4 | 97.64 / 97.55 | 98.08 / 98.04 |

Neither method wins consistently: ours has the higher test accuracy on seeds 0, 1 and 3, and RepDL on seeds 2 and 4. Five seeds can't resolve a gap this small.

## Speed

Median ms per training step on GPU 0, with the median total training time in parentheses. Accuracy checks, hashing and data loading aren't timed. All numbers come from the same card for each method.

| GPU | CUDA | Ours | RepDL | Stock | RepDL vs. ours | Ours vs. stock |
|---|---|---|---|---|---|---|
| RTX 2080 Ti | 12.4 | 3.50 (4.3 s) | 14.82 (17.9 s) | 2.80 (3.6 s) | **4.2× slower** | 1.25× slower |
| RTX 2080 Ti | 12.6 | 3.36 (4.0 s) | 14.50 (17.5 s) | 2.46 (3.4 s) | **4.3× slower** | 1.37× slower |
| RTX 2080 Ti | 13.0 | 3.43 (4.3 s) | 14.52 (17.6 s) | 2.78 (3.7 s) | **4.2× slower** | 1.23× slower |
| TITAN V | 12.4 | 3.21 (3.9 s) | 14.90 (17.5 s) | 1.84 (2.5 s) | **4.6× slower** | 1.74× slower |
| TITAN V | 12.6 | 3.17 (3.9 s) | 13.70 (16.6 s) | 2.11 (2.9 s) | **4.3× slower** | 1.50× slower |
| RTX 4070 | 12.4 | 10.89 (11.5 s) | 14.10 (17.1 s) | 8.89 (10.3 s) | 1.3× slower | 1.22× slower |
| RTX 4070 | 12.6 | 10.90 (12.3 s) | 13.95 (16.9 s) | 8.80 (10.0 s) | 1.3× slower | 1.24× slower |
| RTX 4070 | 13.0 | 10.60 (12.0 s) | 14.01 (17.0 s) | 9.10 (10.8 s) | 1.3× slower | 1.16× slower |

The CUDA 12.4 rows use 5 seeds for RepDL and stock. Other rows use one regular run for RepDL and stock, and three for ours. Identical repeats varied by up to 13% in a controlled test, so small gaps shouldn't be over-read.

### Where the time goes

Median ms from profiled runs at CUDA 12.4. Profiling adds synchronization, so these parts don't add up exactly to the step times above.

| GPU | Method | Forward | Backward | Adam |
|---|---|---|---|---|
| RTX 2080 Ti | Ours | 1.83 | 1.88 | 0.28 |
| RTX 2080 Ti | RepDL | 1.24 | 10.12 | 3.73 |
| RTX 2080 Ti | Stock | 0.89 | 1.53 | 0.52 |
| TITAN V | Ours | 1.07 | 1.91 | 0.20 |
| TITAN V | RepDL | 0.90 | 10.83 | 2.36 |
| TITAN V | Stock | 0.61 | 0.85 | 0.46 |
| RTX 4070 | Ours | 5.12 | 5.17 | 0.88 |
| RTX 4070 | RepDL | 1.45 | 8.00 | 5.17 |
| RTX 4070 | Stock | 2.82 | 4.48 | 1.82 |

Ours gains its lead in the backward pass and Adam. RepDL's forward pass is faster than ours on all three GPUs.

### What the speed results don't show

- **FP64 speed doesn't track RepDL's total time.** It stays near 14 ms although nominal FP64 speed ranges from 1/2 to 1/64.
- **Missing INT8 Tensor Cores don't slow ours on this model.** Ours is fastest on the TITAN V, which has none.

The cause of either pattern hasn't been tested.

### The RTX 4070 slows down during a run

Ours starts at about 4 ms per step and settles near 10.7 ms somewhere between steps 100 and 400:

| Run length | Median per step | Mean per step | Total |
|---|---|---|---|
| 100 steps | 3.99 ms | 4.11 ms | 0.47 s |
| 200 steps | 4.03 ms | 6.46 ms | 1.34 s |
| 400 steps | 10.88 ms | 9.44 ms | 3.81 s |
| 800 steps | 10.75 ms | 9.71 ms | 7.80 s |
| 1,200 steps | 10.73 ms | 9.32 ms | 11.22 s |

- **Ruled out:** other load on the server (load average 0.45), GPU clock (steady at 2,505 MHz) and throttling.
- **Observed:** the GPU moves from power state P0 to P2 and is only about 11% busy, so the limit may be outside GPU computation.
- **Not established:** the cause. PCIe power management is one untested possibility.

The table above reports the sustained 1,200-step numbers, since that's what full training gets. Over the first 100 steps, ours led RepDL by about 3.4×, but that lead doesn't last.

### NaN and infinity scan

Removing the scan entirely saved about 2.7% of training time on the RTX 2080 Ti (3 runs each way, close to run-to-run noise). The scan was kept as is, and its cost is included in all our numbers.

### Memory

Peak GPU memory allocated through PyTorch was 222 MiB for ours, 219 MiB for RepDL and 235–240 MiB for stock. MNIST held on the GPU accounts for about 220 MiB, so this doesn't separate the methods. Stock's figure may also leave out memory used inside CUDA's own libraries.

### Mac CPU, 1 thread

| Method | Median per step | Total | Compared with RepDL |
|---|---|---|---|
| Ours | 60.79 ms | 73.5 s | 2.9× faster |
| RepDL | 175.88 ms | 211.3 s | n/a |
| Stock | 12.95 ms | 15.6 s | 13.6× faster |

These are 2-seed runs. RepDL's Mac build had no OpenMP and no GPU backend, so all three ran at 1 thread for a fair comparison.

## Limits

- **Burst cluster not run:** the L4 and A100 aren't included.
- **Limited GPU and version coverage:** this covers 3 GPU models and 3 CUDA versions. The TITAN V didn't run CUDA 13.0, because CUDA 13 is believed to have dropped it (not tested).
- **Operation suite incomplete on GPU:** a bug in the test harness stopped it after 33 of 47 operations. The bug is now fixed, but the suite hasn't been re-run. The 14 missing operations are conv, pool, Adam and cross-entropy; MNIST training exercises them, except the two vocabulary-size cross-entropy tests.
- **Stock comparison is against ordinary PyTorch:** stock wasn't tested with deterministic settings.
- **Only one model:** speed gaps could differ on larger models such as GPT-2.
- **Seeds 2–4 ran only at CUDA 12.4:** seeds 0 and 1 had wider coverage, as listed under [Runs](#runs).

## Reproducing this

```bash
# On each server, once per setup
export REPRO_WORK=/scratch/$USER/repro PYTHON=python3.12
experiments/01_make_env.sh cuda124 cuda-12.4 2.6.0 cu124

# Accuracy and speed: ours, RepDL, stock
GPUS=0 experiments/03_accuracy_speed.sh cuda124

# Correctness checks
experiments/02_correctness.sh cuda124

# On the machine holding all the results
python3 experiments/compare_results.py --out results/report.md
```

Per-run data is under `results/mnist/` and the full generated comparison is in `results/report.md`.
