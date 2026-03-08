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

from scripts.collect_measurements import BLOCK_SIZES, NUM_STAGES, NUM_WARPS
from server.softmax_surrogate_environment import SoftmaxSurrogateEnvironment, _normalize_discrete


def _choose_surrogate_action(
    env: SoftmaxSurrogateEnvironment,
    acquisition: str,
    beta: float,
    xi: float,
) -> int:
    seen = set(env.seen_config_ids())
    best_config_id = -1
    best_score = float("-inf")

    for config_id in env.available_config_ids():
        if config_id in seen and len(seen) < len(env.available_config_ids()):
            continue
        score = env.acquisition_score(config_id, strategy=acquisition, beta=beta, xi=xi)
        if score > best_score:
            best_score = score
            best_config_id = config_id

    if best_config_id < 0:
        raise RuntimeError("Failed to choose a surrogate action.")
    return best_config_id


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


def run_surrogate_baseline(
    task: str,
    episodes: int,
    budget: int,
    seed: int,
    measurement_path: str,
    train_task_ids: List[str] | None = None,
    acquisition: str = "ucb",
    beta: float = 1.5,
    xi: float = 0.0,
    mode: str = "surrogate",
) -> Dict[str, Any]:
    env = SoftmaxSurrogateEnvironment(
        measurement_path=measurement_path,
        budget=budget,
        seed=seed,
        train_task_ids=train_task_ids,
        mode=mode,
    )

    best_overall = {"latency_ms": float("inf"), "config": None, "task_id": task}
    episode_records: List[Dict[str, Any]] = []

    for episode in range(episodes):
        env.reset(task=task, seed=seed + episode)
        episode_rng = np.random.default_rng(seed + episode)
        done = False
        episode_best = float("inf")
        episode_best_cfg: Dict[str, int] | None = None
        history: List[Dict[str, Any]] = []

        while not done:
            if mode != "surrogate":
                if env.seen_config_ids():
                    best_seen = min(env.seen_config_ids(), key=env.measured_latency_ms)
                    best_cfg = env.config_info(best_seen)
                    action = [
                        _normalize_discrete(BLOCK_SIZES, best_cfg["block_size"]) + float(episode_rng.normal(0.0, 0.15)),
                        _normalize_discrete(NUM_WARPS, best_cfg["num_warps"]) + float(episode_rng.normal(0.0, 0.15)),
                        _normalize_discrete(NUM_STAGES, best_cfg["num_stages"]) + float(episode_rng.normal(0.0, 0.15)),
                    ]
                    action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0).tolist()
                else:
                    action = episode_rng.uniform(-1.0, 1.0, size=3).tolist()
                out = env.step({"x": action})
                config_id = int(out["observation"]["last_trial"]["config_id"])
            else:
                config_id = _choose_surrogate_action(env, acquisition=acquisition, beta=beta, xi=xi)
                out = env.step({"config_id": config_id})
            obs = out["observation"]
            trial = obs["last_trial"]
            history.append(
                {
                    "config_id": config_id,
                    "latency_ms": trial["latency_ms"],
                    "config": trial["config"],
                    "reward": out["reward"],
                    "regret": out["info"]["current_regret"],
                    "validation_mse": out["info"]["validation_mse"],
                }
            )
            if obs["best_so_far_ms"] < episode_best:
                episode_best = obs["best_so_far_ms"]
                best_seen = min(env.seen_config_ids(), key=env.measured_latency_ms)
                episode_best_cfg = env.config_info(best_seen)
            done = bool(out["done"])

        if episode_best < best_overall["latency_ms"]:
            best_overall = {
                "latency_ms": float(episode_best),
                "config": episode_best_cfg,
                "task_id": task,
            }

        diagnostics = env.diagnostics()
        episode_records.append(
            {
                "task_id": task,
                "episode": episode,
                "best_latency_ms": episode_best,
                "best_config": episode_best_cfg or {},
                "final_validation_mse": diagnostics["validation_mse"],
                "final_regret": diagnostics["current_regret"],
                "history": history,
            }
        )

    return {
        "task": task,
        "method": "surrogate",
        "episodes": episodes,
        "budget": budget,
        "seed": seed,
        "mode": mode,
        "train_task_ids": list(train_task_ids or []),
        "acquisition": acquisition,
        "beta": beta,
        "xi": xi,
        "oracle_best_ms": env.oracle_best()["median_ms"],
        "best_overall": best_overall,
        "aggregate_metrics": _aggregate_metrics(episode_records, budget),
        "episodes_summary": episode_records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Surrogate-guided baseline.")
    parser.add_argument("--task", default=None, help="Task ID (e.g., softmax_m4096_n2048)")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--budget", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", type=str, default="surrogate", choices=("surrogate", "default", "self_improving", "generative"))
    parser.add_argument(
        "--acquisition",
        type=str,
        choices=("mean", "ucb", "ei"),
        default="ucb",
        help="Candidate selection mode: mean, ucb, or ei.",
    )
    parser.add_argument("--beta", type=float, default=1.5, help="UCB exploration strength.")
    parser.add_argument("--xi", type=float, default=0.0, help="Expected-improvement margin.")
    parser.add_argument(
        "--train-tasks-file",
        type=Path,
        default=None,
        help="Optional JSON file containing a list of train task ids.",
    )
    parser.add_argument(
        "--measurement-path",
        type=str,
        default="data/autotune_measurements.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/surrogate_baseline.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.task:
        env = SoftmaxSurrogateEnvironment(
            measurement_path=args.measurement_path,
            budget=args.budget,
            seed=args.seed,
            mode=args.mode,
        )
        args.task = env.reset()["observation"]["task_id"]

    train_task_ids = None
    if args.train_tasks_file is not None:
        train_task_ids = json.loads(args.train_tasks_file.read_text(encoding="utf-8"))

    summary = run_surrogate_baseline(
        task=args.task,
        episodes=args.episodes,
        budget=args.budget,
        seed=args.seed,
        measurement_path=args.measurement_path,
        train_task_ids=train_task_ids,
        acquisition=args.acquisition,
        beta=args.beta,
        xi=args.xi,
        mode=args.mode,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
