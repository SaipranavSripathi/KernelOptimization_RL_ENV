# GPU Kernel Autotuning: Hackathon Summary

## What We Built

We built a measured-oracle GPU autotuning benchmark for Triton kernels on an H200-class machine.

The system has two layers:

- Search benchmark
  - compares `surrogate` vs `random`
  - measures how quickly each method finds good kernel configs
- Runtime benchmark
  - compares the selected Triton kernels against eager PyTorch and `torch.compile`
  - measures actual execution latency in milliseconds

The benchmark is discrete-action, uses real measured kernel runtimes, and supports held-out-shape and held-out-family evaluation.

## Kernel Families

Current benchmark families:

- Family A:
  - `softmax`
  - `layernorm`
- Family B:
  - `grouped_gemm`

Current shape set:

- `M = 4096`
- `N ∈ {256, 512, 1024, 1536, 2048, 3072, 4096, 6144, 8192}`

Total tasks:

- `27` tasks

## Why This Matters

There are two distinct questions in GPU autotuning:

1. Can an optimizer find good kernels quickly under a short tuning budget?
2. Are the kernels it finds actually fast in real runtime?

This project answers both.

## Search Results

### Held-out Shapes

Against the `random` baseline, the `surrogate` optimizer achieved:

- `98.4%` lower best-so-far regret AUC
- `96.4%` lower `regret@6`
- `+15.8` percentage points higher final oracle-hit rate

Interpretation:

- On held-out shapes, the surrogate is dramatically more sample-efficient than random.
- It finds near-optimal kernels almost immediately.

### Held-out Family (`grouped_gemm`)

Against the `random` baseline, the `surrogate` optimizer achieved:

- `66.9%` lower best-so-far regret AUC
- `15.3%` lower `regret@6`
- `+16.7` percentage points higher final oracle-hit rate

Interpretation:

- The surrogate also transfers across kernel families.
- The cross-family problem is harder, but the optimizer still wins clearly.

## Runtime Results

We then reran the selected Triton kernels live and compared them to:

- eager PyTorch
- `torch.compile`
- `triton.autotune` on selected large softmax tasks

### Softmax

Average surrogate-best Triton runtime:

- about `0.0336 ms`

Average `torch.compile` runtime:

- about `0.1115 ms`

Result:

- surrogate-selected Triton is about `3.32x` faster than `torch.compile`

### LayerNorm

Average surrogate-best Triton runtime:

- about `0.0323 ms`

Average `torch.compile` runtime:

- about `0.0950 ms`

Result:

- surrogate-selected Triton is about `2.86x` faster than `torch.compile`

### Grouped GEMM

Average surrogate-best Triton runtime:

- about `0.0442 ms`

Average `torch.compile` runtime:

- about `0.1018 ms`

Result:

- surrogate-selected Triton is about `2.42x` faster than `torch.compile`

### Triton Autotune on Large Softmax

We also compared the surrogate directly against `triton.autotune` on:

- `softmax_m4096_n4096`
- `softmax_m4096_n6144`
- `softmax_m4096_n8192`

Result:

- mean surrogate compile plus first call: `98.6 ms`
- mean `triton.autotune` first call: `4316.7 ms`
- mean steady-state latency was effectively the same:
  - surrogate: `0.03009 ms`
  - `triton.autotune`: `0.03012 ms`

Interpretation:

- The surrogate reaches a strong tuned kernel far faster than `triton.autotune` on these tested softmax shapes.
- Final steady-state kernel quality is essentially matched on average.

## What The Results Mean

The current system shows two real advantages:

- The surrogate is much better than random at short-budget tuning.
- The resulting Triton kernels are materially faster than `torch.compile`.
- On selected large softmax tasks, the surrogate also reaches competitive tuned kernels far faster than `triton.autotune`.

The subtle point is that the surrogate's biggest gain is in search efficiency, not necessarily in a huge final-endpoint gap over a lucky random search. That is why regret/AUC is the right optimization metric here.

## What Is Strong

- Real GPU measurements, not synthetic rewards
- Hidden-oracle benchmark protocol
- Discrete action space
- Held-out-shape evaluation
- Held-out-family evaluation
- Runtime comparison against eager PyTorch and `torch.compile`

## What Is Not Final Yet

- This is not publication-grade yet
- Only one `M` dimension is used
- Grouped GEMM was added recently and should be stress-validated further
- `torch.compile` cold-start accounting could be measured even more rigorously

## Bottom Line

This project demonstrates a credible autotuning workflow:

- surrogate-guided search beats random on search quality
- tuned Triton kernels beat `torch.compile` on runtime
- the benchmark already shows cross-family transfer

That is a strong hackathon result with professional-grade measurement discipline.
