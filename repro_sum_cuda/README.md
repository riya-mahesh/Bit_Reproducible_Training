# Order-independent, correctly rounded summation — CUDA

`out[b] = round_to_float32( exact Σ x )` (or exact Σ / n for a mean), for the reductions in neural-network training: bias gradients, loss means, softmax denominators, and later BatchNorm/LayerNorm statistics.

The result is the **exact real sum of the inputs, rounded once** (nearest-even). It does not depend on the order of the data, how the work is split across threads/blocks/chunks, or the GPU, and it is as accurate as a float32 result can be: `1e30 + 1 − 1e30 = 1` exactly, where any floating-point summation gives `0`.

## How it works

- Every finite float32 is an integer multiple of 2⁻¹⁴⁹ below 2¹²⁸, so it is exactly an integer of at most 277 bits in units of 2⁻¹⁴⁹.
- The exact sum lives in a **superaccumulator**: 10 signed 64-bit limbs of radix 2³² (320 bits). Adding a float = one shift and two integer adds into the limbs it overlaps. Each add is below 2³² in magnitude, so up to 2³¹ − 1 inputs fit without resolving carries.
- **Fast path:** a value is the integer `v = m << (sh mod 32)` (< 2⁵⁵) at limb `sh / 32`. Each thread fixes a window base and keeps two 64-bit *wide* accumulators for the two limb positions where nearly all values land: one shift and one 64-bit add per value. Every 255 adds (so they cannot overflow) the wide accumulators are folded into three 32-bit limbs by splitting off their low and high halves, which is exact. Values outside the window (rare) go straight into a full 10-limb accumulator with exact integer atomics. The window changes speed only, never the result.
- **Kernels by layout:** one warp per row for row sums (softmax, LayerNorm), one thread per column for column sums (Linear bias gradient), and cooperative blocks plus chunks otherwise (conv bias gradient, long vectors).
- Threads, warps, blocks, and chunks merge accumulators with integer adds — exact and associative, so **any grouping gives the same bits**.
- One final step resolves carries, optionally divides exactly by n (integer long division for the mean), and rounds once to nearest-even, handling subnormals, overflow to ±inf, and NaN/Inf inputs (→ NaN).

The arithmetic is in [`repro_sum_math.cuh`](repro_sum_math.cuh) (`__host__ __device__`), shared by the GPU kernels and the CPU reference.

## API

```cpp
#include "repro_sum.cuh"
// x viewed as row-major [A, B, C]; out[b] reduces over a and c.
repro_sum::SumConfig cfg;          // cfg.mean = true for the mean; cfg.chunks = 0 (auto)
repro_sum::SumPlan plan;
repro_sum::make_sum_plan(A, B, C, cfg, &plan);          // once per shape
repro_sum::repro_sum_fp32_ws(x, out, A, B, C, cfg, ws, plan.workspace_bytes, stream);
```

| Use | A | B | C |
|---|---|---|---|
| Linear bias gradient (sum over rows of `[M, N]`) | M | N | 1 |
| Conv bias gradient (NCHW, sum over N, H, W) | N | C | H·W |
| Loss mean over n values | 1 | 1 | n (`mean = true`) |
| Per-row sums (softmax, LayerNorm) | 1 | rows | cols |

Exactly zero results are `+0.0`. Requires A·C ≤ 2³¹ − 1.

## Verification

- **Exact reference (on CPU):** the core was checked against exact rational arithmetic (Python `Fraction`) on 430 vectors — full exponent range including subnormals, catastrophic cancellation, overflow, exact ties and just-above-ties, random means — for both sum and mean: 0 mismatches. Each vector was also summed in 20 random orders and accumulator groupings: all identical.
- **Fast-path logic (on CPU):** a host emulation of the GPU rules (wide accumulators, folding, spills) on 8 million values, about a quarter forced outside the window and with fold intervals as small as 1, matched the plain exact sum in all 4,000 trials.
- **GPU (`test_repro_sum`):** GPU vs CPU bit for bit for every LeNet-5 reduction shape plus large and odd shapes, three input distributions, sum and mean, 7 chunk counts, and shuffled element order; plus special cases with known exact answers.
- **Benchmark (`bench_repro_sum`):** time and bandwidth vs an ordinary float reduction with `atomicAdd`, the number of distinct result bit patterns across runs for each, and the float result's error in ulps.

## Build and run

```bash
module load cuda-12.4      # on CIMS
./run_sum.sh               # all GPUs; GPUS="0" for one. Log: results/<host>/sum.log
```

## Performance notes

- Wide register accumulators (above), 16-byte vector loads, no per-element integer division on row segments, four loads in flight per thread for columns.
- When one block covers an output (`chunks == 1`), that block rounds and writes the result itself: one kernel launch, no workspace.
- History on an RTX 2080 Ti — v1 (shared-memory accumulators, two launches): 270–417 GB/s on large shapes, 1.2–1.9× the time of a plain float `atomicAdd` reduction, which gave up to 33 different results across 33 identical runs and errors up to ~42,000 ulps. v2 (5-limb register window updated with selects) became integer-compute-bound (~70–85 instructions per value) and was slower on some shapes; v3 is the current design.
- Possible next step: fuse bias-gradient sums into the kernels that already read dY.
