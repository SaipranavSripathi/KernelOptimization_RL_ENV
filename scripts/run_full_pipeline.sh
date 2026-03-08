#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MEASUREMENTS="${REPO_ROOT}/data/autotune_measurements.csv"
SPLITS="${REPO_ROOT}/data/benchmark_splits.json"
OUTPUT_DIR="${REPO_ROOT}/outputs"
mkdir -p "${OUTPUT_DIR}"
EPISODES=20
BUDGET=6
BETA=2.0

echo "[step] 0) checking workspace"
python3 - <<'PY'
import torch

print(f"python={__import__('sys').executable}")
print(f"torch={getattr(__import__('torch'), '__version__', 'missing')}")
print(f"torch.cuda.is_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available. Run this pipeline on a GPU machine.")
print("ready")
PY

echo "[step] 1) collect multi-family measurements"
python3 scripts/collect_multifamily_measurements.py \
  --output "${MEASUREMENTS}" \
  --families softmax layernorm grouped_gemm \
  --n-cols 256 512 1024 1536 2048 3072 4096 6144 8192 \
  --m 4096 \
  --repeats 200 \
  --warmup 25 \
  --seed 0 \
  --append

echo "[step] 2) build train/test splits"
python3 scripts/build_benchmark_splits.py \
  --measurement-path "${MEASUREMENTS}" \
  --output "${SPLITS}" \
  --heldout-family grouped_gemm

echo "[step] 3) local smoke test"
python3 scripts/smoke_test_client.py | tee "${OUTPUT_DIR}/smoke_test_client.json"

echo "[step] 4) evaluate held-out shapes and held-out family"
python3 scripts/eval_generalization.py \
  --measurement-path "${MEASUREMENTS}" \
  --splits "${SPLITS}" \
  --episodes "${EPISODES}" \
  --budget "${BUDGET}" \
  --seed 2 \
  --acquisition ucb \
  --beta "${BETA}" \
  --output "${OUTPUT_DIR}/generalization_eval.json"

echo "[step] 5) benchmark eager vs torch.compile vs best Triton configs"
python3 scripts/benchmark_runtime_references.py \
  --generalization-results "${OUTPUT_DIR}/generalization_eval.json" \
  --repeats 100 \
  --warmup 10 \
  --seed 123 \
  --output "${OUTPUT_DIR}/runtime_references.json"

python3 - <<'PY'
import json
from pathlib import Path

summary = json.loads(Path("outputs/generalization_eval.json").read_text(encoding="utf-8"))
for section_name, section in summary["results"].items():
    print(section_name)
    print("  random:", section["random_summary"])
    print("  surrogate:", section["surrogate_summary"])

runtime = json.loads(Path("outputs/runtime_references.json").read_text(encoding="utf-8"))
for task_id, task in runtime["results"].items():
    print(task_id)
    print("  torch:", task["torch"])
    print("  speedups:", task["speedups"])
PY

echo "[done] outputs in outputs/"
