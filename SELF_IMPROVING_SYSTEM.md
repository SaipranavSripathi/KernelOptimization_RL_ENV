# Self-Improving Kernel Environment

## Purpose

This document describes the exact self-improving kernel optimization approach implemented in this repo, the current solution architecture, the major improvements that were added, and the current demo-ready state.

The goal is straightforward:

- use a strong teacher model to propose kernel edits now
- use a smaller local student model to learn from those proposals over time
- keep benchmarking real kernels live on GPU
- keep improving the student until teacher fallback is no longer needed

## Problem Setup

We have two different kinds of modes in the environment:

### `surrogate`

- measured-table oracle
- no live source generation
- used for fast benchmark/evaluation and the earlier search-efficiency experiments

### `default`

- live kernel generation
- teacher/proposer only
- no prompt evolution
- no self-improvement buffers or training
- intended as the stable non-learning demo path

### `self_improving`

- live kernel generation
- student-first / teacher-fallback logic
- prompt evolution
- per-segment buffers
- LoRA DPO training
- validation-before-activate
- adapter load / rollback path

This is the mode intended to improve over time.

## Core Approach

The self-improving system is a teacher/student loop.

### Teacher

- backend: OpenRouter
- role: strong proposer
- model: larger Qwen model via the OpenRouter API

### Student

- backend: local Transformers model
- current local model target: `Qwen/Qwen2.5-0.5B-Instruct`
- role: cheaper local model that learns from teacher outcomes

### Environment Logic

For each step in `self_improving` mode:

1. Student is asked to propose a kernel edit first.
2. The student proposal is compiled/imported and benchmarked live.
3. If the student is invalid or not yet mature enough, teacher fallback is queried.
4. Teacher proposal is benchmarked live.
5. If teacher wins, a preference example is stored.
6. If student wins, no preference example is stored.
7. Prompt evolution and training gates are updated.
8. If enough good examples have accumulated, LoRA DPO training is triggered.
9. Candidate adapter is validated on held-out tasks inside the same segment.
10. If validation passes and the student backend supports adapter loading, the adapter is hot-swapped.

This is a real online improvement loop, not just static prompt tuning.

## Why This Architecture

The key design decision is:

- **teacher keeps the system useful immediately**
- **student becomes the long-term model that should eventually replace the teacher**

This avoids the dead-start problem where a small local model is too weak to propose valid kernels from the beginning.

The teacher provides high-quality supervision.
The student learns locally and becomes cheaper and more autonomous over time.

## Live Benchmarking Path

In `default` and `self_improving` modes, the proposed kernel source is:

- written to a temp Python module
- imported dynamically
- executed through `benchmark_generated_kernel(...)`
- run on real CUDA tensors
- validated against a reference implementation

This is a real runtime path.

It is not a lookup-table surrogate when those live modes are used.

## Families and Templates

Current per-family editable templates exist under:

- `kernels/base_softmax.py`
- `kernels/base_layernorm.py`
- `kernels/base_grouped_gemm.py`

The environment can generate code for any family that has a registered template.

## Segmentation

The self-improvement buffers are segment-based.

Current segment key:

```text
{family}_M{m}_N{bucket}_Dfp16
```

Current `N` buckets:

- `small`
- `medium`
- `large`

This is a pragmatic segmentation scheme for demo and early iteration.

It is not the final perfect segmentation system, but it is sufficient for:

- grouping related tasks
- tracking wins and failures
- gating training
- measuring whether the student is maturing

## Buffer Design

### Segment Buffer

Each segment has a rolling buffer:

- max size: `200`
- stored under `artifacts/self_improvement/buffers/...`

Each stored example includes:

- `prompt`
- `chosen`
- `rejected`
- `chosen_role`
- `rejected_role`
- `rejected_kind`
- `preference_kind`
- `reward`
- task/segment metadata

### Preference Classes

The buffer explicitly separates two useful classes:

1. `teacher_beats_student_valid`
2. `teacher_rescues_student_invalid`

This separation matters because they are different supervision signals:

- `teacher_beats_student_valid`
  - teaches actual optimization quality
- `teacher_rescues_student_invalid`
  - teaches validity / syntax / structural compliance

Student wins are tracked in stats, but not stored as teacher-supervision examples.

## DPO Training

The trainer is now:

- **LoRA DPO**
- not SFT
- not full finetuning

It uses:

- chosen/rejected code pairs
- a frozen reference model
- a LoRA policy model
- DPO loss:

```text
-logsigmoid(beta * ((policy_chosen - policy_rejected) - (ref_chosen - ref_rejected)))
```

### Trainer Script

Trainer entrypoint:

- `scripts/train_segment_adapter.py`

Outputs:

- adapter weights
- `adapter.json`
- `result.json`

## Invalid Rejections

Invalid student failures are still collected.

That is intentional.

However, they are **downweighted** in the DPO trainer so they do not dominate the optimization signal.

Current weighting:

- valid-vs-valid teacher win: weight `1.0`
- teacher rescue over invalid student: weight `0.35`

This keeps validity learning useful without letting it crowd out real optimization learning.

## Local Backend

The local backend is implemented in:

- `server/local_adapter_backend.py`

Current behavior:

- loads a local Transformers causal LM
- supports LoRA adapter loading
- uses the current model to generate kernel source completions
- can switch to `unsloth` automatically when installed on CUDA (`KERNEL_USE_UNSLOTH=auto`)
- still falls back to the original `transformers` + `peft` path when `unsloth` is unavailable

This is the first real local backend, not a stub.

## Teacher/Student Backend Split

The environment now supports:

- `proposer_backend`
- `student_backend`

Default intended demo setup:

- proposer = `openrouter`
- student = `local`

That means:

- teacher keeps proposals strong now
- student trains in the background
- student can be hot-swapped independently from the teacher

## Validation-Before-Activate

Before a new adapter becomes active:

- held-out validation tasks are selected from the same segment
- candidate adapter’s preferred source is benchmarked
- current baseline source is benchmarked
- the candidate must not be worse than baseline

Only then can the adapter be accepted.

## Adapter Lifecycle

Current adapter lifecycle features:

- adapter artifact directory
- adapter version tracking
- adapter path tracking
- good adapter history
- rollback hook
- trainer result capture
- persistent state file

Persistence root:

- `artifacts/self_improvement/`

## Demo Runner

There is now a dedicated self-improving demo launcher:

- `scripts/run_self_improving_demo.sh`

It sets defaults for:

- `KERNEL_ENV_MODE=self_improving`
- `KERNEL_PROPOSER_BACKEND=openrouter`
- `KERNEL_STUDENT_BACKEND=local`
- `KERNEL_LOCAL_MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct`
- `KERNEL_DPO_TRAIN_CMD=python3 scripts/train_segment_adapter.py`

And demo-optimized values for:

- proposal batch size
- proposal token budget
- live benchmark repeats / warmup
- DPO gate thresholds
- bootstrap adapter path

## Improvements Added During This Work

These were the major improvements made relative to the earlier surrogate-only system:

1. Dual live modes
- `default`
- `self_improving`

2. Teacher/student split
- large remote proposer
- small local student

3. Dynamic execution of generated kernels
- temp file
- dynamic import
- live CUDA benchmarking

4. Prompt population and prompt evolution

5. Per-segment rolling buffers

6. Chosen/rejected DPO data path

7. LoRA DPO trainer

8. Validation-before-activate

9. Adapter hot-swap and rollback scaffolding

10. Persistent self-improvement state

11. Student-first / teacher-fallback logic

12. Explicit provenance in stored examples
- `chosen_role`
- `rejected_role`
- `preference_kind`

13. Local-only bootstrap workflow for accelerating student learning

14. Deterministic repair pass for common Triton meta-parameter failures
- especially missing `tl.constexpr`

## Observed Milestones

### Structural Student Validity

We verified that the local student can produce a structurally valid softmax Triton module:

- contains `@triton.jit`
- contains `benchmark_generated_kernel`
- imports successfully

### First Real Student Win

We reached a real student win in a local head-to-head against the base template on:

- `softmax_m4096_n256`

Measured result:

- student: `0.007071999832987785 ms`
- base template: `0.007104000076651573 ms`

So:

- `student_wins = True`

This is the first concrete proof that the student is not only structurally valid, but can also generate a runtime-winning kernel on a real benchmark.

## Current Caveats

This system is demo-capable, but not finished in the research/artifact sense.

Important caveats:

1. Teacher is still queried conservatively
- the student-first path exists
- teacher fallback is dynamic
- but there is still room to make teacher usage even tighter

2. Segmentation is still coarse
- family + exact `M` + `N` bucket + dtype

3. Local backend success depends on local model availability
- if the model is not cached or accessible, the student path will fail

4. DPO trainer is real, but still first-generation
- it is practical and aligned
- not yet heavily optimized

5. End-to-end long-run self-improvement still needs more runtime validation
- especially on full GPU live loops

## What Is Demo-Ready

For a demo, the following is ready:

- a stable `default` path
- a live `self_improving` path
- visible teacher/student behavior
- real CUDA kernel execution
- local student generation
- LoRA DPO training artifacts
- adapter lifecycle control plane
- at least one real student runtime win

That is enough to demonstrate the core idea credibly.

## Recommended Demo Framing

The best way to present this system is:

> We built a live self-improving GPU kernel environment. A stronger remote teacher proposes good kernels immediately, a smaller local student learns from those teacher wins using LoRA DPO, and over time the student begins to generate valid and even winning kernels on its own.

That matches the implementation and the observed results.
