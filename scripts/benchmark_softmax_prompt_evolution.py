#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - benchmark environments are expected to have torch.
    torch = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.collect_measurements import BLOCK_SIZES, NUM_STAGES, NUM_WARPS
from scripts.run_surrogate_baseline import _aggregate_metrics
from server.softmax_surrogate_environment import SoftmaxSurrogateEnvironment, _normalize_discrete


def _average_metric_dict(records: List[Dict[str, float]]) -> Dict[str, float]:
    if not records:
        return {}
    keys = sorted({key for record in records for key in record.keys()}, key=lambda value: int(value))
    return {
        key: float(np.mean(np.asarray([record[key] for record in records if key in record], dtype=np.float32)))
        for key in keys
    }


def _summarize_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    mean_regret_records = [run.get("aggregate_metrics", {}).get("mean_regret_at", {}) for run in runs]
    median_regret_records = [run.get("aggregate_metrics", {}).get("median_regret_at", {}) for run in runs]
    auc_values = [run.get("aggregate_metrics", {}).get("mean_auc_regret") for run in runs if run.get("aggregate_metrics", {}).get("mean_auc_regret") is not None]
    oracle_hit_values = [run.get("aggregate_metrics", {}).get("oracle_hit_rate_final") for run in runs if run.get("aggregate_metrics", {}).get("oracle_hit_rate_final") is not None]
    return {
        "mean_regret_at": _average_metric_dict(mean_regret_records),
        "median_regret_at": _average_metric_dict(median_regret_records),
        "mean_best_so_far_auc": float(np.mean(np.asarray(auc_values, dtype=np.float32))) if auc_values else None,
        "mean_oracle_hit_rate_final": float(np.mean(np.asarray(oracle_hit_values, dtype=np.float32))) if oracle_hit_values else None,
    }


def _extract_softmax_surrogate_baseline(path: Path, selected_tasks: List[str]) -> Dict[str, Any]:
    results = json.loads(path.read_text(encoding="utf-8"))
    section = results["results"]["shape_generalization"]
    selected_runs = [run for run in section["task_runs"]["surrogate"] if run["task"] in selected_tasks]
    return {
        "tasks": [run["task"] for run in selected_runs],
        "summary": _summarize_runs(selected_runs),
        "task_runs": selected_runs,
    }


def _load_existing_output(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _candidate_runs_by_task(existing: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    runs = existing.get("candidate", {}).get("task_runs", [])
    return {run["task"]: run for run in runs}


def _progress(message: str) -> None:
    print(message, flush=True)


def _default_parallel_episodes() -> int:
    override = os.environ.get("SOFTMAX_PROMPT_DEFAULT_PARALLEL_EPISODES")
    if override:
        return max(1, int(override))

    gpu_name = ""
    total_memory_gib = 0.0
    if torch is not None:
        try:
            if torch.cuda.is_available():
                gpu_name = str(torch.cuda.get_device_name(0)).lower()
                total_memory_gib = float(torch.cuda.get_device_properties(0).total_memory) / float(1024 ** 3)
        except Exception:
            gpu_name = ""
            total_memory_gib = 0.0

    if "h200" in gpu_name or total_memory_gib >= 120.0:
        return 16
    if "h100" in gpu_name or total_memory_gib >= 70.0:
        return 12
    if "a100" in gpu_name or total_memory_gib >= 40.0:
        return 8
    return 4


def _write_checkpoint(
    output_path: Path,
    measurement_path: str,
    tasks: List[str],
    episodes: int,
    budget: int,
    seed: int,
    beta: float,
    baseline: Dict[str, Any],
    candidate_runs: List[Dict[str, Any]],
) -> None:
    payload = {
        "measurement_path": measurement_path,
        "tasks": tasks,
        "episodes": episodes,
        "budget": budget,
        "seed": seed,
        "baseline": baseline,
        "candidate": {
            "name": "prompt_evolution_pipeline",
            "summary": _summarize_runs(candidate_runs),
            "task_runs": candidate_runs,
        },
        "beta": beta,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_task_run(
    task: str,
    episodes: int,
    budget: int,
    seed: int,
    episode_records: List[Dict[str, Any]],
    oracle_best_ms: Optional[float],
) -> Dict[str, Any]:
    ordered_records = sorted(episode_records, key=lambda record: int(record["episode"]))
    best_overall: Dict[str, Any] = {
        "latency_ms": float("inf"),
        "config": None,
        "task_id": task,
    }
    for record in ordered_records:
        latency_ms = float(record["best_latency_ms"])
        if latency_ms < float(best_overall["latency_ms"]):
            best_overall = {
                "latency_ms": latency_ms,
                "config": dict(record.get("best_config", {})) or None,
                "task_id": task,
            }
    return {
        "task": task,
        "method": "prompt_evolution",
        "episodes": episodes,
        "budget": budget,
        "seed": seed,
        "mode": "generative",
        "beta": None,
        "oracle_best_ms": oracle_best_ms,
        "best_overall": best_overall,
        "aggregate_metrics": _aggregate_metrics(ordered_records, budget),
        "episodes_summary": ordered_records,
    }


def _run_prompt_evolution_episode(
    task: str,
    task_index: int,
    task_count: int,
    episode: int,
    episodes: int,
    budget: int,
    task_seed: int,
    measurement_path: str,
    benchmark_workers: int,
    log_steps: bool,
) -> Dict[str, Any]:
    env = SoftmaxSurrogateEnvironment(
        measurement_path=measurement_path,
        budget=budget,
        seed=task_seed,
        mode="generative",
        benchmark_workers=benchmark_workers,
    )
    started_at = time.perf_counter()
    reset = env.reset(task=task, seed=task_seed + episode)
    reset_info = reset["info"]
    _progress(
        f"[progress] task {task_index + 1}/{task_count} task={task} "
        f"episode {episode + 1}/{episodes} started "
        f"seed={task_seed + episode} observed={reset_info['observed_count']} "
        f"best_ms={float(reset_info['best_so_far_ms']):.6f}"
    )

    episode_rng = np.random.default_rng(task_seed + episode)
    done = False
    episode_best = float("inf")
    episode_best_cfg: Dict[str, int] | None = None
    history: List[Dict[str, Any]] = []

    while not done:
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
        obs = out["observation"]
        trial = obs["last_trial"]
        config_id = int(trial["config_id"])
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
        if log_steps:
            _progress(
                f"[progress] task={task} episode {episode + 1}/{episodes} "
                f"step {len(history)}/{budget} config_id={config_id} "
                f"latency_ms={float(trial['latency_ms']):.6f} "
                f"best_ms={float(obs['best_so_far_ms']):.6f} "
                f"regret={float(out['info']['current_regret']):.6f} "
                f"duplicate={bool(trial['duplicate'])}"
            )
        done = bool(out["done"])

    diagnostics = env.diagnostics()
    elapsed_s = time.perf_counter() - started_at
    episode_record = {
        "task_id": task,
        "episode": episode,
        "best_latency_ms": episode_best,
        "best_config": episode_best_cfg or {},
        "final_validation_mse": diagnostics["validation_mse"],
        "final_regret": diagnostics["current_regret"],
        "history": history,
    }
    _progress(
        f"[progress] task={task} episode {episode + 1}/{episodes} "
        f"finished best_ms={episode_best:.6f} "
        f"final_regret={float(diagnostics['current_regret']):.6f} "
        f"elapsed_s={elapsed_s:.2f}"
    )
    return {
        "episode_record": episode_record,
        "oracle_best_ms": float(env.oracle_best()["median_ms"]),
        "elapsed_s": elapsed_s,
    }


def _run_prompt_evolution_task(
    task: str,
    task_index: int,
    task_count: int,
    episodes: int,
    budget: int,
    seed: int,
    measurement_path: str,
    existing_run: Optional[Dict[str, Any]],
    checkpoint_fn: Callable[[Dict[str, Any]], None],
    output_path: Path,
    parallel_episodes: int,
    benchmark_workers: int,
    log_steps: bool,
) -> Dict[str, Any]:
    episode_records_by_index: Dict[int, Dict[str, Any]] = {}
    oracle_best_ms = float(existing_run["oracle_best_ms"]) if existing_run and existing_run.get("oracle_best_ms") is not None else None
    if existing_run:
        for record in existing_run.get("episodes_summary", []):
            episode_records_by_index[int(record["episode"])] = record

    remaining_episodes = [episode for episode in range(episodes) if episode not in episode_records_by_index]
    if not remaining_episodes:
        _progress(f"[progress] task {task_index + 1}/{task_count} already complete: {task}")
        return _build_task_run(
            task=task,
            episodes=episodes,
            budget=budget,
            seed=seed,
            episode_records=list(episode_records_by_index.values()),
            oracle_best_ms=oracle_best_ms,
        )

    _progress(
        f"[progress] task {task_index + 1}/{task_count}: {task} "
        f"(resume from episode {len(episode_records_by_index)}/{episodes}, "
        f"remaining={len(remaining_episodes)}, parallel_episodes={parallel_episodes}, "
        f"benchmark_workers={benchmark_workers})"
    )

    def handle_completed_episode(result: Dict[str, Any]) -> None:
        nonlocal oracle_best_ms
        record = result["episode_record"]
        episode_records_by_index[int(record["episode"])] = record
        if oracle_best_ms is None:
            oracle_best_ms = float(result["oracle_best_ms"])
        partial_run = _build_task_run(
            task=task,
            episodes=episodes,
            budget=budget,
            seed=seed,
            episode_records=list(episode_records_by_index.values()),
            oracle_best_ms=oracle_best_ms,
        )
        checkpoint_fn(partial_run)
        _progress(
            f"[progress] checkpoint task={task} "
            f"completed_episodes={len(episode_records_by_index)}/{episodes} "
            f"output={output_path}"
        )

    if parallel_episodes <= 1:
        for episode in remaining_episodes:
            result = _run_prompt_evolution_episode(
                task=task,
                task_index=task_index,
                task_count=task_count,
                episode=episode,
                episodes=episodes,
                budget=budget,
                task_seed=seed,
                measurement_path=measurement_path,
                benchmark_workers=benchmark_workers,
                log_steps=log_steps,
            )
            handle_completed_episode(result)
    else:
        with ThreadPoolExecutor(max_workers=parallel_episodes) as executor:
            future_to_episode = {
                executor.submit(
                    _run_prompt_evolution_episode,
                    task,
                    task_index,
                    task_count,
                    episode,
                    episodes,
                    budget,
                    seed,
                    measurement_path,
                    benchmark_workers,
                    log_steps,
                ): episode
                for episode in remaining_episodes
            }
            for future in as_completed(future_to_episode):
                handle_completed_episode(future.result())

    final_run = _build_task_run(
        task=task,
        episodes=episodes,
        budget=budget,
        seed=seed,
        episode_records=list(episode_records_by_index.values()),
        oracle_best_ms=oracle_best_ms,
    )
    _progress(
        f"[progress] completed task {task_index + 1}/{task_count}: {task} "
        f"best_overall_ms={float(final_run['best_overall']['latency_ms']):.6f}"
    )
    return final_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark prompt evolution on held-out softmax tasks against existing surrogate results.")
    parser.add_argument("--measurement-path", type=str, default="data/autotune_measurements.csv")
    parser.add_argument("--existing-surrogate-results", type=Path, default=Path("outputs/generalization_eval.json"))
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=("softmax_m4096_n6144", "softmax_m4096_n8192"),
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--budget", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument(
        "--parallel-episodes",
        type=int,
        default=None,
        help="Episode worker count. Default is auto-tuned by detected GPU class.",
    )
    parser.add_argument("--benchmark-workers", type=int, default=1)
    parser.add_argument("--quiet-step-logs", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/softmax_prompt_evolution_benchmark.json"))
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    tasks = list(args.tasks)
    parallel_source = "cli" if args.parallel_episodes is not None else "auto"
    resolved_parallel_episodes = max(1, int(args.parallel_episodes or _default_parallel_episodes()))
    baseline = _extract_softmax_surrogate_baseline(args.existing_surrogate_results, tasks)
    existing = {} if args.no_resume else _load_existing_output(args.output)
    candidate_runs_by_task = _candidate_runs_by_task(existing)
    candidate_runs_ordered: List[Dict[str, Any]] = []
    log_steps = not args.quiet_step_logs

    _progress(
        f"[progress] benchmark config tasks={len(tasks)} episodes={args.episodes} "
        f"budget={args.budget} parallel_episodes={resolved_parallel_episodes} "
        f"parallel_source={parallel_source} benchmark_workers={args.benchmark_workers} "
        f"resume={not args.no_resume}"
    )

    def checkpoint_update(partial_run: Dict[str, Any]) -> None:
        candidate_runs_by_task[partial_run["task"]] = partial_run
        ordered = [candidate_runs_by_task[task] for task in tasks if task in candidate_runs_by_task]
        _write_checkpoint(
            output_path=args.output,
            measurement_path=args.measurement_path,
            tasks=tasks,
            episodes=args.episodes,
            budget=args.budget,
            seed=args.seed,
            beta=args.beta,
            baseline=baseline,
            candidate_runs=ordered,
        )

    for idx, task in enumerate(tasks):
        task_seed = args.seed + idx * 1000
        run = _run_prompt_evolution_task(
            task=task,
            task_index=idx,
            task_count=len(tasks),
            episodes=args.episodes,
            budget=args.budget,
            seed=task_seed,
            measurement_path=args.measurement_path,
            existing_run=candidate_runs_by_task.get(task),
            checkpoint_fn=checkpoint_update,
            output_path=args.output,
            parallel_episodes=resolved_parallel_episodes,
            benchmark_workers=max(1, int(args.benchmark_workers)),
            log_steps=log_steps,
        )
        candidate_runs_by_task[task] = run
        candidate_runs_ordered = [candidate_runs_by_task[t] for t in tasks if t in candidate_runs_by_task]
        _write_checkpoint(
            output_path=args.output,
            measurement_path=args.measurement_path,
            tasks=tasks,
            episodes=args.episodes,
            budget=args.budget,
            seed=args.seed,
            beta=args.beta,
            baseline=baseline,
            candidate_runs=candidate_runs_ordered,
        )

    final_runs = [candidate_runs_by_task[task] for task in tasks if task in candidate_runs_by_task]
    payload = {
        "measurement_path": args.measurement_path,
        "tasks": tasks,
        "episodes": args.episodes,
        "budget": args.budget,
        "seed": args.seed,
        "baseline": baseline,
        "candidate": {
            "name": "prompt_evolution_pipeline",
            "summary": _summarize_runs(final_runs),
            "task_runs": final_runs,
        },
        "beta": args.beta,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _progress(
        f"[progress] wrote final benchmark summary to {args.output} "
        f"for {len(final_runs)}/{len(tasks)} tasks"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
