# GPU Autotuning Benchmark Report

## Project Summary

This project implements a discrete-action, measured-oracle autotuning benchmark for GPU kernels on an H200-class machine. The optimizer is a surrogate-guided search policy that selects kernel configurations under a short evaluation budget. The benchmark is intentionally structured to answer two different questions:

1. How much more search-efficient is a surrogate-guided policy than a random baseline?
2. Do the kernel configurations found by the search process translate into materially better runtime than strong PyTorch baselines such as eager execution and `torch.compile`?

The current benchmark answers both questions with real measurements.

## What Was Built

The current system includes:

- A shared multi-family measurement cache:
  - `softmax`
  - `layernorm`
  - `grouped_gemm`
- A discrete-action environment with:
  - hidden oracle table
  - short-budget search episodes
  - regret-based metrics
  - train-task priors for cross-task generalization
- Split-based evaluation:
  - held-out shapes
  - held-out family
- Runtime reference benchmarking against:
  - eager PyTorch
  - `torch.compile`
  - live reruns of the best Triton configs found by `random`
  - live reruns of the best Triton configs found by `surrogate`

The key design choice is that search evaluation and runtime evaluation are separated:

- Search benchmark: measures how quickly each method finds good configs
- Runtime benchmark: measures how fast the final chosen kernels actually run

## Benchmark Structure

### Families

- Family A:
  - `softmax`
  - `layernorm`
- Family B:
  - `grouped_gemm`

### Shapes

Current task shapes use:

- fixed `M = 4096`
- `N ∈ {256, 512, 1024, 1536, 2048, 3072, 4096, 6144, 8192}`

This yields:

- `9` softmax tasks
- `9` layernorm tasks
- `9` grouped GEMM tasks
- `27` tasks total

### Search Space

Current kernel config axes:

- `block_size`
- `num_warps`
- `num_stages`

Search is over discrete `config_id`s, not projected continuous actions. That makes the random baseline a true uniform baseline over legal configs.

### Evaluation Splits

The benchmark currently uses:

- `shape_generalization`
  - train on lower/intermediate shapes from each family
  - test on held-out larger shapes within the same families
- `family_holdout`
  - train on `softmax` + `layernorm`
  - test on all `grouped_gemm` tasks

At the time of the main run:

- shape-generalization test tasks: `6`
- family-holdout test tasks: `9`
- unique held-out runtime-reference tasks: `13`

## Metrics

### Search Metrics

The main optimization metrics are:

- `regret@k`
  - best-so-far latency relative to oracle best after `k` search steps
- best-so-far AUC
  - average regret across the whole search trajectory
  - lower is better
- final oracle hit rate
  - how often the optimizer reaches the oracle-best config by the end of the episode

These metrics intentionally emphasize search efficiency, not just the final endpoint.

### Runtime Metrics

The runtime-reference benchmark records:

- eager PyTorch steady-state latency
- `torch.compile` compile-plus-first-call time
- `torch.compile` steady-state latency
- live rerun latency of best Triton config found by `random`
- live rerun latency of best Triton config found by `surrogate`
- Triton speedups vs eager and vs compiled

## Main Search Results

The main search results came from the held-out-shape / held-out-family evaluation in `run13.log`.

### Held-out Shapes

Random:

- `mean_regret_at`:
  - `@1 = 0.31341859698295593`
  - `@3 = 0.13305269181728363`
  - `@5 = 0.1070360466837883`
  - `@6 = 0.06344400346279144`
- `mean_best_so_far_auc = 0.1483089178800583`
- `mean_oracle_hit_rate_final = 0.6749999523162842`

Surrogate:

- `mean_regret_at`:
  - `@1 = 0.002321675419807434`
  - `@3 = 0.002293013734742999`
  - `@5 = 0.002293013734742999`
  - `@6 = 0.002293013734742999`
- `mean_best_so_far_auc = 0.0023013732861727476`
- `mean_oracle_hit_rate_final = 0.8333333134651184`

Interpretation:

- Surrogate reduced best-so-far AUC by about `98.4%` versus random.
- Surrogate reduced final `regret@6` by about `96.4%`.
- Surrogate improved final oracle-hit rate by about `15.8` percentage points.

This is a very strong within-family / held-out-shape result.

### Held-out Family (`grouped_gemm`)

Random:

- `mean_regret_at`:
  - `@1 = 2.341181755065918`
  - `@3 = 0.8532703518867493`
  - `@5 = 0.3116174638271332`
  - `@6 = 0.21012252569198608`
- `mean_best_so_far_auc = 0.9102223515510559`
- `mean_oracle_hit_rate_final = 0.17777778208255768`

Surrogate:

- `mean_regret_at`:
  - `@1 = 0.4722703695297241`
  - `@3 = 0.29785311222076416`
  - `@5 = 0.20862582325935364`
  - `@6 = 0.17804712057113647`
- `mean_best_so_far_auc = 0.3014116585254669`
- `mean_oracle_hit_rate_final = 0.3444444239139557`

Interpretation:

- Surrogate reduced best-so-far AUC by about `66.9%` versus random.
- Surrogate reduced final `regret@6` by about `15.3%`.
- Surrogate improved final oracle-hit rate by about `16.7` percentage points.

This is a good cross-family transfer result. The gap is smaller than in the held-out-shape setting, which is expected.

## Main Runtime Results

The runtime-reference benchmark compares eager PyTorch, `torch.compile`, and the live reruns of the best Triton configs found by `random` and `surrogate`.

### Summary by Family

#### Softmax

Average eager latency:

- `0.1103919968008995 ms`

Average `torch.compile` steady-state latency:

- `0.11152799427509308 ms`

Average compile-plus-first-call time:

- `529.9687260048813 ms`

Average Triton speedup vs eager:

- random-best: `3.362561387683493x`
- surrogate-best: `3.286588301595338x`

Average Triton speedup vs compiled:

- random-best: `3.3985671575178635x`
- surrogate-best: `3.321742054891467x`

Interpretation:

- `torch.compile` is effectively flat vs eager on this softmax set.
- Tuned Triton is substantially faster than both.
- Surrogate-best and random-best final kernels are very close in absolute runtime.

#### LayerNorm

Average eager latency:

- `0.05939200147986412 ms`

Average `torch.compile` steady-state latency:

- `0.09503999352455139 ms`

Average compile-plus-first-call time:

- `440.1235789991915 ms`

Average Triton speedup vs eager:

- random-best: `1.8776593781360051x`
- surrogate-best: `1.8364378273209185x`

Average Triton speedup vs compiled:

- random-best: `2.927484944635789x`
- surrogate-best: `2.862647103483093x`

Interpretation:

- `torch.compile` is slower than eager on this LayerNorm set.
- Tuned Triton is materially faster than both eager and compiled.
- Again, surrogate-best and random-best final kernels are close in endpoint runtime.

#### Grouped GEMM

Average eager latency:

- `0.05589688859052128 ms`

Average `torch.compile` steady-state latency:

- `0.101806221736802 ms`

Average compile-plus-first-call time:

- `102.45987688863858 ms`

Average Triton speedup vs eager:

- random-best: `1.2771213149737215x`
- surrogate-best: `1.2644549628354071x`

Average Triton speedup vs compiled:

- random-best: `2.4414293463407355x`
- surrogate-best: `2.4156697207038382x`

Interpretation:

- `torch.compile` is materially slower than eager on this grouped-GEMM set.
- Tuned Triton is faster than both eager and compiled.
- Endpoint difference between surrogate-best and random-best remains small.

## Triton Autotune Comparison

We also compared the surrogate search workflow directly against `triton.autotune` on three large softmax tasks:

- `softmax_m4096_n4096`
- `softmax_m4096_n6144`
- `softmax_m4096_n8192`

This comparison measures two things:

- search plus compile cost to the first usable tuned kernel
- steady-state runtime of the resulting tuned kernel

### Per-task Results

#### `softmax_m4096_n4096`

- oracle best: `0.02127999998629093 ms`
- surrogate:
  - decision time: `33.06370500649791 ms`
  - compile plus first call: `294.734695009538 ms`
  - steady-state: `0.02127999998629093 ms`
  - regret vs oracle: `0.0`
- `triton.autotune`:
  - first call: `8970.702438004082 ms`
  - steady-state: `0.021856000646948814 ms`
  - regret vs oracle: `0.0270677002363231`

#### `softmax_m4096_n6144`

- oracle best: `0.030719999223947525 ms`
- surrogate:
  - decision time: `15.47088599181734 ms`
  - compile plus first call: `0.9627069957787171 ms`
  - steady-state: `0.031007999554276466 ms`
  - regret vs oracle: `0.009375010989727928`
- `triton.autotune`:
  - first call: `1990.3547260037158 ms`
  - steady-state: `0.031039999797940254 ms`
  - regret vs oracle: `0.010416685614473398`

#### `softmax_m4096_n8192`

- oracle best: `0.03747199848294258 ms`
- surrogate:
  - decision time: `15.47144899086561 ms`
  - compile plus first call: `0.09679699724074453 ms`
  - steady-state: `0.03798399865627289 ms`
  - regret vs oracle: `0.013663540618560122`
- `triton.autotune`:
  - first call: `1988.913670007605 ms`
  - steady-state: `0.03747199848294258 ms`
  - regret vs oracle: `0.0`

### Summary

- mean surrogate compile plus first call: `98.59806633418582 ms`
- mean surrogate steady-state latency: `0.03009066606561343 ms`
- mean `triton.autotune` first call: `4316.656944671801 ms`
- mean `triton.autotune` steady-state latency: `0.030122666309277218 ms`

Interpretation:

- The surrogate reaches a competitive tuned kernel far faster than `triton.autotune` on these tasks.
- Steady-state performance is effectively the same on average:
  - surrogate mean steady-state: `0.0300907 ms`
  - `triton.autotune` mean steady-state: `0.0301227 ms`
- On one task the surrogate exactly matched the oracle best.
- On the other two tasks the surrogate was slightly off the oracle, but still close.
- `triton.autotune` won one task in steady-state quality, but paid a much larger first-call search cost.

This is an important result because it shows the surrogate is not only better than a random search baseline. It is also competitive with Triton's built-in autotuning in final kernel quality while being dramatically cheaper in tuning-time-to-first-good-kernel on these tested shapes.

## What The Results Mean

The results support the following conclusions:

1. The surrogate optimizer is genuinely useful as a search policy.
   - It is substantially more sample-efficient than random.
   - It reaches good kernels much earlier in the budget.
   - This effect is very strong on held-out shapes and still meaningful on held-out family transfer.

2. The resulting Triton kernels are genuinely useful as runtime implementations.
   - They are faster than eager PyTorch.
   - They are faster than `torch.compile`.
   - The advantage is strongest on `softmax`, then `layernorm`, then `grouped_gemm`.

3. The surrogate is also competitive with `triton.autotune` on final steady-state runtime while being much cheaper in first-call tuning cost on the tested softmax shapes.

4. The main value of the surrogate is search efficiency, not necessarily a dramatically better final endpoint than a lucky random search.
   - By the end of the short search budget, random and surrogate can still land on very similar endpoint kernels.
   - This is visible in the runtime benchmark, where random-best and surrogate-best final kernels are often close in ms.
   - The surrogate still wins decisively on regret and AUC.

This is a coherent and valuable result. The optimizer is improving how quickly good kernels are found, and the kernels it finds are fast in absolute runtime.

## Caveats and Professional Notes

This benchmark is strong for a hackathon project, but it should not be oversold.

Important caveats:

- The benchmark is not publication-grade yet.
  - Only one `M` value is used.
  - Family/task distributions are still narrow.
  - Grouped GEMM was added recently and should be validated more deeply.
- `torch.compile` compile time measurement should be interpreted carefully.
  - The measured compile-plus-first-call times vary across tasks.
  - A more publication-grade measurement would reset compiler state more aggressively and isolate cold-start behavior more carefully.
- The runtime benchmark uses live reruns.
  - That means small run-to-run variation is expected.
- The endpoint runtime gap between surrogate-best and random-best is small.
  - This is not a failure of the benchmark.
  - It means the benchmark currently demonstrates search-efficiency gains more strongly than final-endpoint gains.

These caveats do not invalidate the results. They define the proper scope of the claims.

## Recommended Project Framing

The most honest and compelling project framing is:

> We built a measured-oracle GPU autotuning benchmark with held-out-shape and held-out-family evaluation. Our surrogate-guided optimizer substantially outperforms a random baseline on short-budget search efficiency, and the Triton kernels it finds are materially faster than both eager PyTorch and `torch.compile`.

That statement matches the data.

## Recommended Next Steps

Highest-value next steps:

1. Expand the runtime benchmark:
   - more held-out tasks
   - more end-to-end summaries

2. Improve search without changing the measurement cache:
   - stronger acquisition strategies
   - family-aware priors
   - feature engineering before simply increasing parameter count

3. Make the collector more production-friendly:
   - incremental writes
   - progress logging
   - resumable measurement collection

4. If needed, increase benchmark rigor:
   - broader shape sets
   - more families
   - more careful cold-start `torch.compile` accounting

## Deliverable Status

At the current stage, this project is:

- hackathon-ready
- technically credible
- professionally explainable
- useful as a platform for next-stage kernel autotuning work

It is not yet:

- a finished research benchmark
- a final systems paper artifact

That is the correct level of rigor for the current results.
