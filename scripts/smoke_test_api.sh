#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
TASK="${1:-softmax_m4096_n256}"
CONFIG_ID="${2:-0}"
SEED="${SEED:-0}"
USE_TEMPLATE_SOURCE="${USE_TEMPLATE_SOURCE:-1}"
export TASK CONFIG_ID SEED

tmpdir="$(mktemp -d)"
cleanup() {
  rm -rf "${tmpdir}"
}
trap cleanup EXIT

health_json="${tmpdir}/health.json"
reset_json="${tmpdir}/reset.json"
step_json="${tmpdir}/step.json"
state_json="${tmpdir}/state.json"

echo "[api] GET ${BASE_URL}/health"
curl -sS "${BASE_URL}/health" > "${health_json}"

echo "[api] POST ${BASE_URL}/reset task=${TASK} seed=${SEED}"
python3 - <<'PY' > "${tmpdir}/reset_payload.json"
import json, os
print(json.dumps({"task": os.environ["TASK"], "seed": int(os.environ["SEED"])}))
PY
curl -sS -X POST "${BASE_URL}/reset" \
  -H "Content-Type: application/json" \
  --data @"${tmpdir}/reset_payload.json" > "${reset_json}"

echo "[api] POST ${BASE_URL}/step config_id=${CONFIG_ID}"
export USE_TEMPLATE_SOURCE
python3 - <<'PY' > "${tmpdir}/step_payload.json"
import json, os
from pathlib import Path

task = os.environ["TASK"]
payload = {"config_id": int(os.environ["CONFIG_ID"])}
if os.environ.get("USE_TEMPLATE_SOURCE", "1") == "1":
    if task.startswith("softmax_"):
        path = Path("kernels/base_softmax.py")
    elif task.startswith("layernorm_"):
        path = Path("kernels/base_layernorm.py")
    elif task.startswith("grouped_gemm_"):
        path = Path("kernels/base_grouped_gemm.py")
    else:
        path = None
    if path is not None and path.exists():
        payload["source"] = path.read_text(encoding="utf-8")
print(json.dumps(payload))
PY
curl -sS -X POST "${BASE_URL}/step" \
  -H "Content-Type: application/json" \
  --data @"${tmpdir}/step_payload.json" > "${step_json}"

echo "[api] GET ${BASE_URL}/state"
curl -sS "${BASE_URL}/state" > "${state_json}"

export TMPDIR_PATH="${tmpdir}"
python3 - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["TMPDIR_PATH"])
health = json.loads((root / "health.json").read_text(encoding="utf-8"))
reset = json.loads((root / "reset.json").read_text(encoding="utf-8"))
step = json.loads((root / "step.json").read_text(encoding="utf-8"))
state = json.loads((root / "state.json").read_text(encoding="utf-8"))

print("[result] health", health)
print("[result] reset")
print(json.dumps({
    "task_id": reset["observation"]["task_id"],
    "family": reset["observation"]["family"],
    "mode": reset["observation"]["mode"],
    "best_so_far_ms": reset["info"]["best_so_far_ms"],
    "oracle_best_ms": reset["info"]["oracle_best_ms"],
    "segment_key": reset["info"]["segment_key"],
    "training_status": reset["info"]["training_status"],
    "adapter_version": reset["info"]["adapter_version"],
}, indent=2))
print("[result] step")
print(json.dumps({
    "last_trial": step["observation"]["last_trial"],
    "reward": step["reward"],
    "best_so_far_ms": step["info"]["best_so_far_ms"],
    "oracle_best_ms": step["info"]["oracle_best_ms"],
    "current_regret": step["info"]["current_regret"],
    "segment_stats": step["info"].get("segment_stats", {}),
    "training_status": step["info"]["training_status"],
    "adapter_version": step["info"]["adapter_version"],
}, indent=2))
print("[result] state")
print(json.dumps(state, indent=2))
PY
