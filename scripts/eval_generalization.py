#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.run_random_baseline import run_random_baseline
from scripts.run_surrogate_baseline import run_surrogate_baseline


def _average_metric_dict(records: List[Dict[str, float]]) -> Dict[str, float]:
    if not records:
        return {}
    keys = sorted({key for record in records for key in record.keys()}, key=lambda value: int(value))
    return {
        key: float(np.mean(np.asarray([record[key] for record in records if key in record], dtype=np.float32)))
        for key in keys
    }


def _summarize_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    mean_regret_records = [run["aggregate_metrics"].get("mean_regret_at", {}) for run in runs]
    median_regret_records = [run["aggregate_metrics"].get("median_regret_at", {}) for run in runs]
    auc_values = [run["aggregate_metrics"].get("mean_auc_regret") for run in runs]
    oracle_hit_values = [run["aggregate_metrics"].get("oracle_hit_rate_final") for run in runs]
    return {
        "mean_regret_at": _average_metric_dict(mean_regret_records),
        "median_regret_at": _average_metric_dict(median_regret_records),
        "mean_best_so_far_auc": float(np.mean(np.asarray(auc_values, dtype=np.float32))) if auc_values else None,
        "mean_oracle_hit_rate_final": float(np.mean(np.asarray(oracle_hit_values, dtype=np.float32))) if oracle_hit_values else None,
    }


def _section_payload(section_name: str, split: Dict[str, Any], random_runs: List[Dict[str, Any]], surrogate_runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "section": section_name,
        "train_tasks": split["train_tasks"],
        "test_tasks": split["test_tasks"],
        "random_summary": _summarize_runs(random_runs),
        "surrogate_summary": _summarize_runs(surrogate_runs),
        "task_runs": {
            "random": random_runs,
            "surrogate": surrogate_runs,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate random vs surrogate on shape and family holdout splits.")
    parser.add_argument("--measurement-path", type=str, default="data/autotune_measurements.csv")
    parser.add_argument("--splits", type=Path, default=Path("data/benchmark_splits.json"))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--budget", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--acquisition", choices=("mean", "ucb", "ei"), default="ucb")
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--xi", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=Path("outputs/generalization_eval.json"))
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    splits = json.loads(args.splits.read_text(encoding="utf-8"))
    sections = {
        "shape_generalization": splits["shape_generalization"],
        "family_holdout": splits["family_holdout"],
    }
    existing_results = {}
    if not args.no_resume and args.output.exists():
        try:
            existing_results = json.loads(args.output.read_text(encoding="utf-8")).get("results", {})
        except Exception:
            existing_results = {}

    results: Dict[str, Any] = {}
    for section_name, split in sections.items():
        existing_section = existing_results.get(section_name, {})
        random_runs = list(existing_section.get("task_runs", {}).get("random", []))
        surrogate_runs = list(existing_section.get("task_runs", {}).get("surrogate", []))
        completed_random = {run["task"] for run in random_runs}
        completed_surrogate = {run["task"] for run in surrogate_runs}
        for idx, task in enumerate(split["test_tasks"]):
            task_seed = args.seed + idx * 1000
            print(f"[progress] section={section_name} task {idx + 1}/{len(split['test_tasks'])}: {task}")
            if task not in completed_random:
                random_runs.append(
                    run_random_baseline(
                        task=task,
                        episodes=args.episodes,
                        budget=args.budget,
                        seed=task_seed,
                        measurement_path=args.measurement_path,
                    )
                )
                completed_random.add(task)
                print(f"[progress] completed random for {task}")
            else:
                print(f"[progress] skipping completed random for {task}")
            if task not in completed_surrogate:
                surrogate_runs.append(
                    run_surrogate_baseline(
                        task=task,
                        episodes=args.episodes,
                        budget=args.budget,
                        seed=task_seed,
                        measurement_path=args.measurement_path,
                        train_task_ids=split["train_tasks"],
                        acquisition=args.acquisition,
                        beta=args.beta,
                        xi=args.xi,
                    )
                )
                completed_surrogate.add(task)
                print(f"[progress] completed surrogate for {task}")
            else:
                print(f"[progress] skipping completed surrogate for {task}")
            results[section_name] = _section_payload(section_name, split, random_runs, surrogate_runs)
            summary = {
                "measurement_path": args.measurement_path,
                "splits_path": str(args.splits),
                "episodes": args.episodes,
                "budget": args.budget,
                "acquisition": args.acquisition,
                "beta": args.beta,
                "xi": args.xi,
                "results": results,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        results[section_name] = _section_payload(section_name, split, random_runs, surrogate_runs)

    summary = {
        "measurement_path": args.measurement_path,
        "splits_path": str(args.splits),
        "episodes": args.episodes,
        "budget": args.budget,
        "acquisition": args.acquisition,
        "beta": args.beta,
        "xi": args.xi,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
