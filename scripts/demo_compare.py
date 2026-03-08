#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from server.softmax_surrogate_environment import SoftmaxSurrogateEnvironment
from scripts.collect_measurements import benchmark_single_config
from scripts.run_random_baseline import run_random_baseline
from scripts.run_surrogate_baseline import run_surrogate_baseline


@dataclass
class BaselineResult:
    method: str
    latency_ms: float
    config: Dict[str, int]
    regret: float


def _search_metric_key(summary: Dict[str, Any], oracle_best_ms: float) -> tuple[float, float, float]:
    metrics = summary.get("aggregate_metrics", {})
    mean_auc_regret = float(metrics.get("mean_auc_regret", float("inf")))
    oracle_hit_rate_final = float(metrics.get("oracle_hit_rate_final", 0.0))
    best_latency_ms = float(summary["best_overall"]["latency_ms"])
    latency_regret = best_latency_ms / oracle_best_ms - 1.0
    return (mean_auc_regret, -oracle_hit_rate_final, latency_regret)


def _heuristic_for_task(task_id: str, task_rows: List[Dict[str, Any]], env: SoftmaxSurrogateEnvironment) -> BaselineResult:
    n = int(task_id.split("_n")[-1])
    block = min(row["block_size"] for row in task_rows if row["block_size"] >= n)
    warp = 4 if 4 in {row["num_warps"] for row in task_rows} else 2
    stage = 2 if 2 in {row["num_stages"] for row in task_rows} else 1

    candidate = None
    for row in task_rows:
        if row["block_size"] == block and row["num_warps"] == warp and row["num_stages"] == stage:
            candidate = row
            break
    if candidate is None:
        candidate = min(
            task_rows,
            key=lambda row: abs(row["block_size"] - block) + 10 * abs(row["num_warps"] - warp),
        )

    latency_ms = env.measured_latency_ms(candidate["config_id"])
    oracle_best_ms = env.oracle_best()["median_ms"]
    return BaselineResult(
        method="heuristic",
        latency_ms=float(latency_ms),
        config=candidate,
        regret=float(latency_ms / oracle_best_ms - 1.0),
    )


def _pick_task(task_arg: str | None, measurement_path: str, budget: int) -> str:
    env = SoftmaxSurrogateEnvironment(measurement_path=measurement_path, budget=budget, seed=0)
    if task_arg:
        env.reset(task=task_arg)
    else:
        env.reset()
    return env.state()["task_id"]


def _run_all(
    task: str,
    budget: int,
    episodes: int,
    seed: int,
    measurement_path: str,
    acquisition: str,
    beta: float,
    xi: float,
) -> Dict[str, Any]:
    env = SoftmaxSurrogateEnvironment(measurement_path=measurement_path, budget=budget, seed=seed)
    env.reset(task=task)
    task_rows = env.available_configs()
    oracle_best = env.oracle_best()

    heuristic = _heuristic_for_task(task, task_rows, env)
    random_summary = run_random_baseline(task, episodes=episodes, budget=budget, seed=seed, measurement_path=measurement_path)
    surrogate_summary = run_surrogate_baseline(
        task,
        episodes=episodes,
        budget=budget,
        seed=seed,
        measurement_path=measurement_path,
        acquisition=acquisition,
        beta=beta,
        xi=xi,
    )

    search_summaries = {
        "random": random_summary,
        "surrogate": surrogate_summary,
    }
    winner_method, winner_summary = min(
        search_summaries.items(),
        key=lambda item: _search_metric_key(item[1], oracle_best["median_ms"]),
    )
    winner_cfg = winner_summary["best_overall"]["config"]
    winner_regret = float(winner_summary["best_overall"]["latency_ms"] / oracle_best["median_ms"] - 1.0)
    n = int(task.split("_n")[-1])

    live = benchmark_single_config(
        n=n,
        block_size=winner_cfg["block_size"],
        num_warps=winner_cfg["num_warps"],
        num_stages=winner_cfg["num_stages"],
        repeats=max(200, budget * 20),
        warmup=25,
        seed=seed + 999,
    )

    return {
        "task": task,
        "seed": seed,
        "budget": budget,
        "episodes": episodes,
        "acquisition": acquisition,
        "beta": beta,
        "xi": xi,
        "oracle_best": oracle_best,
        "heuristic": heuristic.__dict__,
        "random": random_summary["best_overall"],
        "random_aggregate_metrics": random_summary.get("aggregate_metrics", {}),
        "surrogate": surrogate_summary["best_overall"],
        "surrogate_aggregate_metrics": surrogate_summary.get("aggregate_metrics", {}),
        "winner": {
            "method": winner_method,
            "selection_metric": "min(mean_auc_regret), tie-break max(oracle_hit_rate_final), then best latency",
            "latency_ms": winner_summary["best_overall"]["latency_ms"],
            "config": winner_cfg,
            "regret": winner_regret,
            "live_rerun": live.__dict__,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare heuristic/random/surrogate baselines.")
    parser.add_argument(
        "--task",
        default="softmax_m4096_n2048",
        help="Task ID (e.g., softmax_m4096_n2048)",
    )
    parser.add_argument("--budget", type=int, default=6)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument(
        "--acquisition",
        type=str,
        choices=("mean", "ucb", "ei"),
        default="ucb",
    )
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--xi", type=float, default=0.0)
    parser.add_argument(
        "--measurement-path",
        type=str,
        default="data/autotune_measurements.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/demo_compare.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task = _pick_task(args.task, args.measurement_path, args.budget)
    summary = _run_all(
        task=task,
        budget=args.budget,
        episodes=args.episodes,
        seed=args.seed,
        measurement_path=args.measurement_path,
        acquisition=args.acquisition,
        beta=args.beta,
        xi=args.xi,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
