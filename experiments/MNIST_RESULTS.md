# MNIST results

LeNet-5 trained on MNIST for 2 epochs (1,200 steps), comparing our method with RepDL and stock PyTorch. All methods used the same data, batches, hyperparameters and starting weights.

## Conclusions

- **Reproducibility:** our final weights are bit-identical on every GPU and CUDA version tested, and on a CPU. RepDL's are too when it starts from the same weights. Stock PyTorch's are not.
- **Accuracy:** there is no meaningful difference between the three methods.
- **Speed:** ours is 4.2–4.6× faster than RepDL on two GPUs and 1.3× faster on the third, and 1.2–1.7× slower than stock PyTorch.

## What was tested

| GPU | CUDA versions |
|---|---|
| RTX 2080 Ti | 12.4, 12.6, 13.0 |
| TITAN V | 12.4, 12.6 |
| RTX 4070 | 12.4, 12.6, 13.0 |

That is 8 GPU and CUDA combinations, plus an Apple ARM CPU. Each method ran 5 seeds at CUDA 12.4 and seed 0 on the other versions.

## Reproducibility

| Method | Same result on all GPUs and CUDA versions? |
|---|---|
| **Ours** | **Yes** |
| **RepDL**, same starting weights | **Yes** |
| **Stock PyTorch** | **No:** results differed between runs, even on the same GPU |

Our correctly rounded exp and log also give identical results over all 2³² float32 inputs on all 3 GPUs and on CPUs.

## Accuracy

| Method | Train accuracy | Test accuracy |
|---|---|---|
| Ours | 98.14% ± 0.29 | 98.00% ± 0.28 |
| RepDL | 98.08% ± 0.07 | 97.96% ± 0.13 |
| Stock PyTorch | 98.29% ± 0.24 | 98.07% ± 0.29 |

Mean ± standard deviation over 5 seeds. Stock's row covers all 3 GPUs at CUDA 12.4 (15 runs). The gaps are within the spread between seeds.

## Speed

Median ms per training step on one GPU, averaged here across CUDA versions, which made little difference.

| GPU | Ours | RepDL | Stock | Ours vs. RepDL | Ours vs. stock |
|---|---|---|---|---|---|
| RTX 2080 Ti | 3.4 | 14.6 | 2.7 | **4.2–4.3× faster** | 1.2–1.4× slower |
| TITAN V | 3.2 | 14.3 | 2.0 | **4.3–4.6× faster** | 1.5–1.7× slower |
| RTX 4070 | 10.8 | 14.0 | 8.9 | **1.3× faster** | 1.2× slower |

A full run takes about 4 s of training for ours on the RTX 2080 Ti and TITAN V, against about 17 s for RepDL.

### Where the time goes

Median ms per step on the RTX 2080 Ti at CUDA 12.4:

| Method | Forward | Backward | Adam |
|---|---|---|---|
| Ours | 1.8 | 1.9 | 0.3 |
| RepDL | 1.2 | 10.1 | 3.7 |
| Stock | 0.9 | 1.5 | 0.5 |

Ours gains its lead in the backward pass and Adam. RepDL's forward pass is faster than ours, and this holds on all 3 GPUs.

## Other observations

- **RepDL's time barely changes across GPUs**, staying near 14 ms even though their FP64 speeds differ widely. The cause wasn't tested.
- **Missing INT8 Tensor Cores don't slow ours on this model:** it is fastest on the TITAN V, which has none.
- **The RTX 4070 slows down during a run:** ours starts near 4 ms per step, then settles near 10.7 ms within the first 400 steps. Stock is also slow on this server. The GPU's clock stays high and it is only about 11% busy, so the limit may be outside GPU computation, but the cause is unconfirmed. The table uses the sustained numbers.
- **Seeding alone doesn't fix starting weights across CPUs:** RepDL's published example misses its published hash on the RTX 4070 server because PyTorch's default initialization depends on the CPU's instruction set (AVX2 or not). This happens before any RepDL arithmetic, and fixing the initialization removes the difference.
- **Checking for NaN and infinity costs little:** removing the check saved about 2.7% of training time on the RTX 2080 Ti, so it stays on and is included in our times.
- **Memory doesn't separate the methods:** peak GPU memory is about 220–240 MiB for all three, most of which is the dataset.
- **On a CPU at 1 thread (Apple ARM, 2 seeds):** ours is 2.9× faster than RepDL and 4.7× slower than stock.

## Limits

- This covers one small model, and the speed gaps could differ on larger ones such as GPT-2.
- The burst cluster's L4 and A100 GPUs aren't included.
- Stock PyTorch ran without deterministic settings, so this doesn't show what it could do with them.
- Each stock setup's same-GPU difference comes from one regular run and one profiled run.
- Seeds 1–4 ran only at CUDA 12.4.
- Separate per-operation GPU tests are incomplete: 14 of 47 didn't run, because of a test bug that's now fixed. MNIST training exercises most of those operations, except the two vocabulary-size cross-entropy tests.
