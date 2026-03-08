#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def _load_tasks(path: Path) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[row["task_id"]].append(row)
    return grouped


def _task_n(task_id: str) -> int:
    return int(task_id.split("_n")[-1])


def build_splits(grouped: Dict[str, List[dict]], heldout_family: str | None) -> Dict[str, object]:
    tasks_by_family: Dict[str, List[str]] = defaultdict(list)
    for task_id, rows in grouped.items():
        tasks_by_family[rows[0]["family"]].append(task_id)

    for family in tasks_by_family:
        tasks_by_family[family].sort(key=_task_n)

    families = sorted(tasks_by_family.keys())
    if not families:
        raise RuntimeError("No tasks found in measurement file.")

    shape_train: List[str] = []
    shape_test: List[str] = []
    for family, tasks in tasks_by_family.items():
        holdout_count = 2 if len(tasks) >= 4 else 1
        split_idx = max(1, len(tasks) - holdout_count)
        shape_train.extend(tasks[:split_idx])
        shape_test.extend(tasks[split_idx:])

    if heldout_family is None:
        heldout_family = families[-1]
    if heldout_family not in tasks_by_family:
        raise ValueError(f"Held-out family {heldout_family} is not present.")

    family_train = [task_id for family, tasks in tasks_by_family.items() if family != heldout_family for task_id in tasks]
    family_test = list(tasks_by_family[heldout_family])

    return {
        "families_present": families,
        "shape_generalization": {
            "train_tasks": sorted(shape_train),
            "test_tasks": sorted(shape_test),
        },
        "family_holdout": {
            "heldout_family": heldout_family,
            "train_tasks": sorted(family_train),
            "test_tasks": sorted(family_test),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train/test split manifests for the multi-family benchmark.")
    parser.add_argument("--measurement-path", type=Path, default=Path("data/autotune_measurements.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/benchmark_splits.json"))
    parser.add_argument("--heldout-family", type=str, default=None)
    args = parser.parse_args()

    splits = build_splits(_load_tasks(args.measurement_path), args.heldout_family)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(splits, handle, indent=2)
    print(json.dumps(splits, indent=2))


if __name__ == "__main__":
    main()
