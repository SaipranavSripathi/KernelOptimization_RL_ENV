#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.qwen_05b_spec import DECODE_CTX_LENS, PREFILL_SEQ_LENS, qwen_05b_tasks


def build_splits() -> dict:
    tasks = qwen_05b_tasks()
    long_prefill = max(PREFILL_SEQ_LENS)
    long_decode = max(DECODE_CTX_LENS)

    shape_train = []
    shape_test = []
    for task in tasks:
        if task.mode == "prefill" and task.seq_len == long_prefill:
            shape_test.append(task.task_id)
        elif task.mode == "decode" and task.ctx_len == long_decode:
            shape_test.append(task.task_id)
        else:
            shape_train.append(task.task_id)

    family_holdout_train = [task.task_id for task in tasks if task.family != "gemm"]
    family_holdout_test = [task.task_id for task in tasks if task.family == "gemm"]

    return {
        "model_id": "Qwen/Qwen2.5-0.5B",
        "shape_generalization": {
            "train_tasks": sorted(shape_train),
            "test_tasks": sorted(shape_test),
        },
        "family_holdout": {
            "heldout_family": "gemm",
            "train_tasks": sorted(family_holdout_train),
            "test_tasks": sorted(family_holdout_test),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train/test split manifests for Qwen2.5-0.5B kernel tuning.")
    parser.add_argument("--output", type=Path, default=Path("data/qwen_05b_splits.json"))
    args = parser.parse_args()

    splits = build_splits()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(splits, handle, indent=2)
    print(json.dumps(splits, indent=2))


if __name__ == "__main__":
    main()
