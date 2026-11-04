# Experiments

Three questions, one set of scripts:

1. **Correctness:** are results bit-identical across GPUs and CUDA versions, on CIMS and the burst cluster?
2. **Accuracy:** training and test accuracy of ours, RepDL, and stock PyTorch.
3. **Speed:** ours against RepDL, on the same GPU with the same data, order, and initial weights.

| Script | Purpose | Where |
|---|---|---|
| `00_show_env.sh` | Shows GPUs, CUDA modules, disk, and Slurm details; changes nothing | Each machine |
| `01_make_env.sh TAG MODULE TORCH WHEEL` | Makes one configuration: CUDA module, PyTorch wheel, RepDL | Once per configuration, per filesystem |
| `02_correctness.sh TAG` | All kernel tests and hashes, the CPU proof, and MNIST twice with our layers | Each GPU and configuration |
| `03_accuracy_speed.sh TAG` | Ours, RepDL, and stock over 5 seeds, profiled runs, and RepDL's own example | At least one configuration per GPU |
| `slurm/burst_job.sbatch` | Runs steps 02 and 03 as a burst job | Burst cluster |
| `compare_results.py` | Turns everything under `results/` into the three answers | Your Mac, after copying results back |

## What counts as a pass

- **Correctness:** `compare_results.py` shows our hashes identical for every configuration, including your Mac's CPU: final weights `6b8ce584…` at seed 0, `repro_expf` `4ccc7cddaea06b26`, `repro_logf` `117fa2a6bbdea567`.
- **RepDL's install:** its own example gives trained-model hash `31ee86a7…`, published in its README.
- **Accuracy:** ours is within the spread of stock PyTorch and RepDL over the seeds.
- **Speed:** measured on each GPU. RepDL uses FP64 for exp and log, so the gap should depend on the GPU's FP64 speed.

## Choosing configurations

Pair each CUDA module with the PyTorch wheel of the same CUDA version, so a process doesn't load two CUDA runtimes:

| Tag | CUDA module | PyTorch | Wheel |
|---|---|---|---|
| `cuda118` | `cuda-11.8` | 2.6.0 | `cu118` |
| `cuda124` | `cuda-12.4` | 2.6.0 | `cu124` |
| `cuda126` | `cuda-12.6` | 2.7.1 | `cu126` |
| `cuda128` | `cuda-12.8` | 2.8.0 | `cu128` |

Use the versions `00_show_env.sh` lists. The `CUDA Version` from `nvidia-smi` is the newest the driver can run. Results don't depend on the toolkit's math library by design; finding out is the point of the test.

`REPRO_WORK` must be outside the CIMS home, whose quota is 4 GB. Each environment takes several GB.
