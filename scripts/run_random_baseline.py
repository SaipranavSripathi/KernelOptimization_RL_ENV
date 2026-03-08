#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from server.softmax_surrogate_environment import SoftmaxSurrogateEnvironment


@dataclass
class RunRecord:
    task_id: str
    episode: int
    best_latency_ms: float
    best_config: Dict[str, int]
    final_validation_mse: float
    final_state: Dict[str, Any]
    final_regret: float
    history: List[Dict[str, Any]]


def _aggregate_metrics(episode_records: List[Dict[str, Any]], budget: int) -> Dict[str, Any]:
    ks = sorted(set(k for k in (1, 3, 5, budget) if k <= budget))
    regrets_by_k: Dict[int, List[float]] = {k: [] for k in ks}
    auc_regrets: List[float] = []

    for episode in episode_records:
        regrets = [float(step["regret"]) for step in episode["history"]]
        if regrets:
            auc_regrets.append(float(sum(regrets) / len(regrets)))
        for k in ks:
            if len(regrets) >= k:
                regrets_by_k[k].append(regrets[k - 1])

    return {
        "mean_regret_at": {
            str(k): float(sum(vals) / len(vals)) for k, vals in regrets_by_k.items() if vals
        },
        "median_regret_at": {
            str(k): float(np.median(np.asarray(vals, dtype=np.float32))) for k, vals in regrets_by_k.items() if vals
        },
        "mean_auc_regret": float(sum(auc_regrets) / len(auc_regrets)) if auc_regrets else None,
        "oracle_hit_rate_final": float(
            sum(1 for episode in episode_records if float(episode["final_regret"]) == 0.0) / len(episode_records)
        ) if episode_records else None,
    }


def _pick_task_from_input(args: argparse.Namespace) -> str:
    if args.task:
        return args.task
    env = SoftmaxSurrogateEnvironment(
        measurement_path=args.measurement_path,
        budget=args.budget,
        seed=args.seed,
        mode=args.mode,
    )
    return env.reset()["observation"]["task_id"]


def run_random_baseline(
    task: str,
    episodes: int,
    budget: int,
    seed: int,
    measurement_path: str,
    mode: str = "surrogate",
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    best_overall: Dict[str, Any] = {"latency_ms": float("inf"), "config": None, "task_id": task}
    episode_records: List[Dict[str, Any]] = []

    env = SoftmaxSurrogateEnvironment(
        measurement_path=measurement_path,
        budget=budget,
        seed=seed,
        mode=mode,
    )

    for episode in range(episodes):
        env.reset(task=task, seed=seed + episode)
        done = False
        episode_best = float("inf")
        episode_best_cfg: Dict[str, int] | None = None
        history: List[Dict[str, Any]] = []

        while not done:
            if mode != "surrogate":
                action = {"x": rng.uniform(-1.0, 1.0, size=3).tolist()}
                step_out = env.step(action)
                config_id = int(step_out["observation"]["last_trial"]["config_id"])
            else:
                unseen = [config_id for config_id in env.available_config_ids() if config_id not in env.seen_config_ids()]
                choice_pool = unseen if unseen else env.available_config_ids()
                config_id = int(rng.choice(choice_pool))
                step_out = env.step({"config_id": config_id})
            obs = step_out["observation"]
            trial = obs["last_trial"]
            history.append(
                {
                    "config_id": config_id,
                    "latency_ms": trial["latency_ms"],
                    "config": trial["config"],
                    "reward": step_out["reward"],
                    "regret": step_out["info"]["current_regret"],
                    "validation_mse": step_out["info"]["validation_mse"],
                }
            )
            if obs["best_so_far_ms"] < episode_best:
                episode_best = obs["best_so_far_ms"]
                best_id = env.seen_config_ids()[int(np.argmin([env.measured_latency_ms(cid) for cid in env.seen_config_ids()]))]
                episode_best_cfg = env.config_info(best_id)
            done = bool(step_out["done"])

        if episode_best < best_overall["latency_ms"]:
            best_overall = {
                "latency_ms": float(episode_best),
                "config": episode_best_cfg,
                "task_id": task,
            }

        diagnostics = env.diagnostics()
        episode_records.append(
            RunRecord(
                task_id=task,
                episode=episode,
                best_latency_ms=float(episode_best),
                best_config=episode_best_cfg or {},
                final_validation_mse=float(diagnostics["validation_mse"]),
                final_state=env.state(),
                final_regret=float(diagnostics["current_regret"]),
                history=history,
            ).__dict__
        )

    return {
        "task": task,
        "method": "random",
        "episodes": episodes,
        "budget": budget,
        "seed": seed,
        "mode": mode,
        "oracle_best_ms": env.oracle_best()["median_ms"],
        "best_overall": best_overall,
        "aggregate_metrics": _aggregate_metrics(episode_records, budget),
        "episodes_summary": episode_records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random baseline for surrogate environment.")
    parser.add_argument("--task", default=None, help="Task ID (e.g., softmax_m4096_n2048)")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--budget", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", type=str, default="surrogate", choices=("surrogate", "default", "self_improving", "generative"))
    parser.add_argument(
        "--measurement-path",
        type=str,
        default="data/autotune_measurements.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/random_baseline.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task = _pick_task_from_input(args)
    summary = run_random_baseline(
        task=task,
        episodes=args.episodes,
        budget=args.budget,
        seed=args.seed,
        measurement_path=args.measurement_path,
        mode=args.mode,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
