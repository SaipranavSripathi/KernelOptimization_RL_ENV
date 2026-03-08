---
title: RL Surrogate ENV
emoji: rocket
colorFrom: slate
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
short_description: Multi-family GPU autotuning surrogate environment and benchmark API.
---

# KernelOptimization_RL_ENV

A reinforcement learning environment for GPU kernel autotuning. Optimizes Triton kernel configurations (block_size, num_warps, num_stages) for softmax, layernorm, and grouped_gemm operations using surrogate models.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/SaipranavSripathi/KernelOptimization_RL_ENV.git
cd KernelOptimization_RL_ENV

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Check CUDA availability
python3 scripts/check_torch_cuda.py

# Run the full pipeline
./scripts/run_full_pipeline.sh
```

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (for full benchmarking)
- Triton (for kernel autotuning)
- PyTorch with CUDA support

## Project Structure

- `server/` - OpenEnv server and RL environment
- `kernels/` - Base kernel implementations (softmax, layernorm, grouped_gemm)
- `scripts/` - Benchmark and training scripts
- `data/` - Measurement caches and benchmark splits
- `frontend/` - React frontend for visualization
- `backend/` - Terminal backend for interactive sessions
- `models.py` - Surrogate model definitions
- `client.py` - Client for interacting with the environment
- `space_app.py` - HuggingFace Space deployment

## Usage

## TL;DR

```bash
python3 scripts/check_torch_cuda.py
./scripts/run_full_pipeline.sh
```

The pipeline now:
- collects a shared measurement cache for `softmax`, `layernorm`, and `grouped_gemm`
- builds split manifests for shape holdout and family holdout
- smoke-tests the local OpenEnv-style environment
- evaluates `random` vs `surrogate` using `regret@k` and best-so-far AUC
- benchmarks eager PyTorch and `torch.compile` against best Triton configs

## Measurement cache

Main collector:

```bash
python3 scripts/collect_multifamily_measurements.py \
  --output data/autotune_measurements.csv \
  --families softmax layernorm grouped_gemm \
  --n-cols 256 512 1024 1536 2048 3072 4096 6144 8192 \
  --m 4096 \
  --repeats 200 \
  --warmup 25 \
  --seed 0 \
  --append
```

Current implemented families:
- `softmax`
- `layernorm`
- `grouped_gemm`

The shared CSV schema includes:
- `family_group`
- `family`
- `task_id`
- `m`, `n`
- `config_id`
- `block_size`, `num_warps`, `num_stages`
- `shape_json`, `config_json`
- `median_ms`, `effective_gbps`, `score`, `validation_error`

## Splits and eval

Build split manifests:

```bash
python3 scripts/build_benchmark_splits.py \
  --measurement-path data/autotune_measurements.csv \
  --output data/benchmark_splits.json \
  --heldout-family grouped_gemm
```

Run split-based evaluation:

```bash
python3 scripts/eval_generalization.py \
  --measurement-path data/autotune_measurements.csv \
  --splits data/benchmark_splits.json \
  --episodes 20 \
  --budget 6 \
  --seed 2 \
  --acquisition ucb \
  --beta 2.0
```

Benchmark absolute runtime references:

```bash
python3 scripts/benchmark_runtime_references.py \
  --generalization-results outputs/generalization_eval.json \
  --repeats 100 \
  --warmup 10 \
  --seed 123
```

Metrics:
- `mean_regret_at`
- `median_regret_at`
- `mean_best_so_far_auc`
- `mean_oracle_hit_rate_final`
- `eager_latency_ms`
- `compile_plus_first_call_ms`
- `compiled_latency_ms`
- Triton speedups vs eager / compiled

## Environment

OpenEnv metadata is in:
- `openenv.yaml`

The environment server still uses:
- `server/app.py`
- `server/softmax_surrogate_environment.py`

Despite the filename, the env is now multi-family and supports a train-task prior for held-out-shape / held-out-family evaluation.

## Hugging Face Space

The Hugging Face Space deployment uses:
- `space_app.py`
- `Dockerfile`
- `requirements-space.txt`

That deployment intentionally runs the surrogate environment in CPU-safe mode so the Space stays usable without the full local Triton benchmarking stack. The repository still includes the full GPU benchmark workflows, frontend prototype, and self-improving environment code for local or H100-backed runs.

## Local student stack

The local LoRA student now supports an optional `unsloth` path for both adapter training and local generation.

Defaults:
- `KERNEL_USE_UNSLOTH=auto`
- `KERNEL_UNSLOTH_LOAD_IN_4BIT=0`

If `unsloth` is installed and CUDA is available, `auto` switches the local backend and `scripts/train_segment_adapter.py` onto `unsloth`. If you want to force the old path, set:

```bash
export KERNEL_USE_UNSLOTH=0
```

If you want to force `unsloth`, set:

```bash
export KERNEL_USE_UNSLOTH=1
```

`unsloth` installation is intentionally not pinned in `requirements.txt` because its install matrix depends on your CUDA / Torch stack. Install it using the official `unsloth` instructions for your machine, then reuse the same repo commands.

## Qwen2.5-0.5B exact-kernel pipeline

This repo now also includes a model-specific benchmark pipeline for the exact inference kernel roles needed by `Qwen/Qwen2.5-0.5B`.

Kernel roles covered:
- `rmsnorm`
- attention `softmax`
- `q_proj`, `k_proj`, `v_proj`, `o_proj`
- `gate_proj`, `up_proj`, `down_proj`

Run it with:

```bash
./scripts/run_qwen_05b_pipeline.sh
```

Key files:
- `scripts/qwen_05b_spec.py`
- `scripts/collect_qwen_05b_measurements.py`
- `scripts/build_qwen_05b_splits.py`
- `scripts/benchmark_qwen_05b_runtime.py`

Outputs:
- `data/qwen_05b_measurements.csv`
- `data/qwen_05b_splits.json`
- `outputs/qwen_05b_generalization_eval.json`
- `outputs/qwen_05b_runtime_references.json`
