#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def _load_rows(path: Path) -> Dict[str, List[float]]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            grouped[row["task_id"]].append(float(row["median_ms"]))
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description="Report task hardness from measured latency table.")
    parser.add_argument("--measurement-path", type=Path, default=Path("data/autotune_measurements.csv"))
    parser.add_argument("--budget", type=int, default=6)
    args = parser.parse_args()

    grouped = _load_rows(args.measurement_path)
    for task_id, vals in sorted(grouped.items()):
        vals = sorted(vals)
        best = vals[0]
        ncfg = len(vals)
        within1 = sum(v <= best * 1.01 for v in vals)
        within2 = sum(v <= best * 1.02 for v in vals)
        within5 = sum(v <= best * 1.05 for v in vals)
        hit_best = 1.0 - (1.0 - 1.0 / ncfg) ** args.budget
        print(
            f"{task_id} ncfg={ncfg} best_ms={best:.9f} "
            f"within1={within1} within2={within2} within5={within5} "
            f"random_hit_best@{args.budget}={hit_best:.4f}"
        )


if __name__ == "__main__":
    main()
