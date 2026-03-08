#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MEASUREMENTS="${REPO_ROOT}/data/qwen_05b_measurements.csv"
SPLITS="${REPO_ROOT}/data/qwen_05b_splits.json"
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

echo "[step] 1) collect exact Qwen2.5-0.5B kernel measurements"
python3 scripts/collect_qwen_05b_measurements.py \
  --output "${MEASUREMENTS}" \
  --repeats 200 \
  --warmup 25 \
  --seed 0 \
  --append

echo "[step] 2) build Qwen splits"
python3 scripts/build_qwen_05b_splits.py --output "${SPLITS}"

echo "[step] 3) local smoke test against Qwen measurement cache"
python3 - <<'PY'
import json
from pathlib import Path
from client import SoftmaxSurrogateEnvClient

client = SoftmaxSurrogateEnvClient(measurement_path="data/qwen_05b_measurements.csv")
reset_out = client.reset()
first_config = reset_out["observation"]["tried_config_ids"][0]
step_out = client.step({"config_id": first_config})
summary = {"reset": reset_out, "step": step_out}
Path("outputs/qwen_05b_smoke_test.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

echo "[step] 4) evaluate held-out shapes and held-out family"
python3 scripts/eval_generalization.py \
  --measurement-path "${MEASUREMENTS}" \
  --splits "${SPLITS}" \
  --episodes "${EPISODES}" \
  --budget "${BUDGET}" \
  --seed 2 \
  --acquisition ucb \
  --beta "${BETA}" \
  --output "${OUTPUT_DIR}/qwen_05b_generalization_eval.json"

echo "[step] 5) benchmark eager vs torch.compile vs best Triton configs"
python3 scripts/benchmark_qwen_05b_runtime.py \
  --generalization-results "${OUTPUT_DIR}/qwen_05b_generalization_eval.json" \
  --repeats 100 \
  --warmup 10 \
  --seed 123 \
  --output "${OUTPUT_DIR}/qwen_05b_runtime_references.json"

python3 - <<'PY'
import json
from pathlib import Path

eval_summary = json.loads(Path("outputs/qwen_05b_generalization_eval.json").read_text(encoding="utf-8"))
for section_name, section in eval_summary["results"].items():
    print(section_name)
    print("  random:", section["random_summary"])
    print("  surrogate:", section["surrogate_summary"])

runtime = json.loads(Path("outputs/qwen_05b_runtime_references.json").read_text(encoding="utf-8"))
for task_id, task in runtime["results"].items():
    print(task_id)
    print("  torch:", task["torch"])
    print("  speedups:", task["speedups"])
PY

echo "[done] Qwen outputs in outputs/"
