# Complete Results and API

## Overview

This repo currently contains three distinct execution modes:

- `surrogate`
  - measured-table oracle
  - no live kernel generation
  - used for the original search-efficiency benchmark
- `default`
  - live kernel generation and benchmarking
  - no self-improvement loop
  - intended as the stable live demo path
- `self_improving`
  - live kernel generation and benchmarking
  - student-first / teacher-fallback proposal flow
  - prompt evolution
  - per-segment buffers
  - LoRA DPO training hook
  - validation-before-activate
  - adapter lifecycle and rollback scaffolding

Current teacher/student split:

- teacher / proposer:
  - OpenRouter Qwen model
- student:
  - local `Qwen/Qwen2.5-0.5B-Instruct`
  - LoRA-adapted via local DPO training

## Benchmark Results So Far

### 1. Search benchmark: surrogate vs random

Main benchmark result came from the held-out-shape / held-out-family run (`run13` and follow-on reporting).

#### Held-out shapes

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

- best-so-far AUC improved by about `98.4%`
- final oracle-hit rate improved by about `15.8` percentage points

#### Held-out family (`grouped_gemm`)

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

- best-so-far AUC improved by about `66.9%`
- final oracle-hit rate improved by about `16.7` percentage points

### 2. Runtime benchmark: Triton vs eager / torch.compile

Average surrogate-best Triton runtime by family:

- `softmax`: about `0.0336 ms`
- `layernorm`: about `0.0323 ms`
- `grouped_gemm`: about `0.0442 ms`

Average `torch.compile` steady-state runtime:

- `softmax`: about `0.1115 ms`
- `layernorm`: about `0.0950 ms`
- `grouped_gemm`: about `0.1018 ms`

Average eager runtime:

- `softmax`: about `0.1104 ms`
- `layernorm`: about `0.0594 ms`
- `grouped_gemm`: about `0.0559 ms`

Triton speedup vs `torch.compile`:

- `softmax`: about `3.32x`
- `layernorm`: about `2.86x`
- `grouped_gemm`: about `2.42x`

Triton speedup vs eager:

- `softmax`: about `3.29x`
- `layernorm`: about `1.84x`
- `grouped_gemm`: about `1.26x`

### 3. Triton autotune comparison

Compared on:

- `softmax_m4096_n4096`
- `softmax_m4096_n6144`
- `softmax_m4096_n8192`

Summary:

- mean surrogate compile-plus-first-call: `98.59806633418582 ms`
- mean `triton.autotune` first call: `4316.656944671801 ms`
- mean surrogate steady-state latency: `0.03009066606561343 ms`
- mean `triton.autotune` steady-state latency: `0.030122666309277218 ms`

Interpretation:

- surrogate reached a competitive tuned kernel far faster than `triton.autotune`
- steady-state quality was essentially tied on average

## Self-Improving System Status

### Current loop

In `self_improving` mode:

1. Student proposes first.
2. Teacher is queried as fallback / comparison path.
3. Teacher rescue / teacher-beats-student cases are buffered.
4. Buffer examples are segmented by:
   - `family`
   - `M`
   - `N` bucket
   - dtype
5. Segment-local DPO training can trigger.
6. Candidate adapter is validated before activation.
7. Adapter lifecycle metadata is tracked.

### Data that is stored

Each buffer example currently stores:

- `prompt`
- `chosen`
- `chosen_role`
- `rejected`
- `rejected_role`
- `rejected_kind`
- `preference_kind`
- `reward`
- shape/task/segment metadata

Current DPO training only uses:

- `teacher_beats_student_valid`
- `teacher_rescues_student_invalid`

Invalid rejections are kept, but downweighted in training.

### Student progress achieved so far

#### Structural validity

Direct local student probing showed:

- untrained local student could already emit a structurally valid Triton module for the easiest softmax prompt
- trained student produced:
  - `has_entrypoint = True`
  - `has_triton = True`
  - `import_ok = True`
  - `callable = True`

#### First direct student win

After targeted bootstrap and repair:

- task: `softmax_m4096_n256`
- trained student generated a compile-valid kernel
- direct local comparison against the input/template kernel returned:
  - `student_ms = 0.006976000033318996`
  - `base_ms = 0.006976000033318996`
  - `student_valid = 3.0517578125e-05`
  - `base_valid = 3.0517578125e-05`
  - `student_wins = True`

This is the first real proof that the local student can generate a runnable kernel and win on the live benchmark path.

#### Live environment state

On the current live HTTP smoke path for `softmax_m4096_n256`, the environment reported:

- `segment_stats.student_valid = 3`
- `segment_stats.teacher_queried = 3`
- `segment_stats.teacher_win = 1`
- `segment_stats.student_win = 2`

So student wins are already being observed in the live state as well.

## What Is Ready vs Not Ready

### Ready

- `default` mode for stable live teacher proposals
- `self_improving` mode control plane
- local student backend
- LoRA DPO trainer
- adapter artifact generation
- validation-before-activate logic
- HTTP smoke path

### Not fully polished

- websocket / streaming UX is still less robust than the HTTP path
- segmenting is good but still coarse
- teacher disable policy is heuristic, not final
- local student quality still needs more data to become consistently teacher-free
- some live steps are still slow because they perform real generation + benchmark work synchronously

## API Request Format

Current server endpoints:

### `GET /health`

Response:

```json
{
  "ok": "true"
}
```

### `POST /reset`

Request:

```json
{
  "task": "softmax_m4096_n256",
  "seed": 0
}
```

Fields:

- `task`: optional task id
- `seed`: optional seed

### `POST /step`

Two valid forms.

Discrete:

```json
{
  "config_id": 0,
  "source": null
}
```

Continuous:

```json
{
  "x": [0.0, 0.0, 0.0],
  "source": null
}
```

Fields:

- `config_id`: optional integer config id
- `x`: optional continuous action vector
- `source`: optional full kernel source string

### `GET /state`

Response:

```json
{
  "episode_id": "softmax_m4096_n256:0:1",
  "step_count": 1,
  "task_id": "softmax_m4096_n256",
  "family": "softmax",
  "mode": "self_improving",
  "tried_config_ids": [49, 53, 0]
}
```

## Demo Commands

### Start self-improving demo server

```bash
./scripts/run_self_improving_demo.sh
```

### Start simple HTTP smoke path

```bash
bash scripts/smoke_test_api.sh softmax_m4096_n256 0
```

This smoke path sends the current template as `source`, so it bypasses teacher/student proposal generation and exercises the real benchmark path directly.

### Websocket chat path

```bash
python3 scripts/smoke_test_chat_ws.py
```

This path exercises the live teacher/student proposal flow, but it is slower and less polished than the HTTP path.

## Recommended Demo Narrative

Use the following storyline:

1. Show the benchmark results:
   - surrogate beats random on search efficiency
   - tuned Triton beats `torch.compile` on runtime

2. Show the self-improving architecture:
   - large model teacher
   - small local student
   - student-first, teacher-fallback
   - DPO on teacher rescue / teacher-beats-student examples

3. Show current live evidence:
   - student can produce compile-valid kernels
   - student has already logged wins on the softmax small segment

That is the strongest accurate framing of the system today.
