#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import triton
import triton.language as tl
import triton.testing as ttesting

from scripts.collect_measurements import BLOCK_SIZES, NUM_STAGES, NUM_WARPS, fused_rowwise_softmax_kernel
from server.softmax_surrogate_environment import SoftmaxSurrogateEnvironment


DEFAULT_TASKS = [
    "softmax_m4096_n4096",
    "softmax_m4096_n6144",
    "softmax_m4096_n8192",
]


def _choose_surrogate_config_id(env: SoftmaxSurrogateEnvironment, acquisition: str, beta: float, xi: float) -> int:
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
        raise RuntimeError("Failed to select surrogate config.")
    return best_config_id


def _valid_autotune_configs(n_cols: int) -> List[triton.Config]:
    configs = []
    for block_size in BLOCK_SIZES:
        if block_size < n_cols:
            continue
        for num_warps in NUM_WARPS:
            for num_stages in NUM_STAGES:
                configs.append(
                    triton.Config(
                        {"BLOCK_SIZE": block_size},
                        num_warps=num_warps,
                        num_stages=num_stages,
                    )
                )
    return configs


def _compile_plus_first_call_fixed(sample: torch.Tensor, config: Dict[str, int]) -> float:
    output = torch.empty_like(sample)
    grid = (sample.shape[0],)
    torch.cuda.synchronize()
    start = time.perf_counter()
    fused_rowwise_softmax_kernel[grid](
        sample,
        output,
        sample.stride(0),
        sample.stride(1),
        output.stride(0),
        output.stride(1),
        sample.shape[1],
        BLOCK_SIZE=int(config["block_size"]),
        num_warps=int(config["num_warps"]),
        num_stages=int(config["num_stages"]),
    )
    torch.cuda.synchronize()
    return float((time.perf_counter() - start) * 1000.0)


def _steady_state_fixed(sample: torch.Tensor, config: Dict[str, int], repeats: int, warmup: int) -> float:
    output = torch.empty_like(sample)
    grid = (sample.shape[0],)

    def launch() -> None:
        fused_rowwise_softmax_kernel[grid](
            sample,
            output,
            sample.stride(0),
            sample.stride(1),
            output.stride(0),
            output.stride(1),
            sample.shape[1],
            BLOCK_SIZE=int(config["block_size"]),
            num_warps=int(config["num_warps"]),
            num_stages=int(config["num_stages"]),
        )

    return float(
        ttesting.do_bench(
            launch,
            warmup=warmup,
            rep=repeats,
            quantiles=[0.5],
            return_mode="median",
        )
    )


def _benchmark_triton_autotune(sample: torch.Tensor, repeats: int, warmup: int) -> Dict[str, float]:
    output = torch.empty_like(sample)
    n_cols = sample.shape[1]
    configs = _valid_autotune_configs(n_cols)

    @triton.autotune(configs=configs, key=["n_cols"])
    @triton.jit
    def autotuned_softmax_kernel(
        X_ptr,
        Y_ptr,
        stride_xm,
        stride_xn,
        stride_ym,
        stride_yn,
        n_cols,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        col_offsets = tl.arange(0, BLOCK_SIZE)
        x_ptr = X_ptr + row_idx * stride_xm + col_offsets
        y_ptr = Y_ptr + row_idx * stride_ym + col_offsets
        mask = col_offsets < n_cols

        x = tl.load(x_ptr, mask=mask, other=-float("inf"))
        x = x - tl.max(x, axis=0)
        numerator = tl.exp(x)
        denominator = tl.sum(numerator, axis=0)
        y = numerator / denominator
        tl.store(y_ptr, y, mask=mask)

    grid = (sample.shape[0],)
    torch.cuda.synchronize()
    start = time.perf_counter()
    autotuned_softmax_kernel[grid](
        sample,
        output,
        sample.stride(0),
        sample.stride(1),
        output.stride(0),
        output.stride(1),
        n_cols,
    )
    torch.cuda.synchronize()
    first_call_ms = float((time.perf_counter() - start) * 1000.0)

    def launch() -> None:
        autotuned_softmax_kernel[grid](
            sample,
            output,
            sample.stride(0),
            sample.stride(1),
            output.stride(0),
            output.stride(1),
            n_cols,
        )

    steady_ms = float(
        ttesting.do_bench(
            launch,
            warmup=warmup,
            rep=repeats,
            quantiles=[0.5],
            return_mode="median",
        )
    )
    return {
        "autotune_first_call_ms": first_call_ms,
        "autotune_steady_ms": steady_ms,
    }


def _build_summary(
    measurement_path: str,
    tasks: List[str],
    acquisition: str,
    beta: float,
    xi: float,
    results: Dict[str, Any],
) -> Dict[str, Any]:
    if results:
        surrogate_first = [task["surrogate"]["compile_plus_first_call_ms"] for task in results.values()]
        surrogate_steady = [task["surrogate"]["steady_ms"] for task in results.values()]
        autotune_first = [task["triton_autotune"]["autotune_first_call_ms"] for task in results.values()]
        autotune_steady = [task["triton_autotune"]["autotune_steady_ms"] for task in results.values()]
        surrogate_speedup = [
            task["triton_autotune"]["autotune_first_call_ms"] / max(task["surrogate"]["compile_plus_first_call_ms"], 1e-9)
            for task in results.values()
        ]
        summary = {
            "mean_surrogate_compile_plus_first_call_ms": float(np.mean(surrogate_first)),
            "mean_surrogate_steady_ms": float(np.mean(surrogate_steady)),
            "mean_autotune_first_call_ms": float(np.mean(autotune_first)),
            "mean_autotune_steady_ms": float(np.mean(autotune_steady)),
            "mean_search_time_speedup_surrogate_vs_autotune": float(np.mean(surrogate_speedup)),
            "completed_task_count": len(results),
        }
    else:
        summary = {
            "mean_surrogate_compile_plus_first_call_ms": None,
            "mean_surrogate_steady_ms": None,
            "mean_autotune_first_call_ms": None,
            "mean_autotune_steady_ms": None,
            "mean_search_time_speedup_surrogate_vs_autotune": None,
            "completed_task_count": 0,
        }
    return {
        "measurement_path": measurement_path,
        "tasks": tasks,
        "acquisition": acquisition,
        "beta": beta,
        "xi": xi,
        "results": results,
        "summary": summary,
    }


def run_benchmark(
    measurement_path: str,
    tasks: List[str],
    repeats: int,
    warmup: int,
    seed: int,
    acquisition: str,
    beta: float,
    xi: float,
    output_path: Path,
    resume: bool,
) -> Dict[str, Any]:
    env_probe = SoftmaxSurrogateEnvironment(measurement_path=measurement_path, budget=1, seed=seed)
    available_tasks = [task for task in env_probe.available_tasks() if task.startswith("softmax_m4096_n")]
    train_task_ids = [task for task in available_tasks if task not in tasks]

    results: Dict[str, Any] = {}
    if resume and output_path.exists():
        try:
            existing = json.loads(output_path.read_text(encoding="utf-8"))
            results = dict(existing.get("results", {}))
        except Exception:
            results = {}

    for index, task in enumerate(tasks):
        if task in results:
            print(f"[progress] skipping completed task {index + 1}/{len(tasks)}: {task}")
            continue
        task_seed = seed + index
        print(f"[progress] starting task {index + 1}/{len(tasks)}: {task}")
        env = SoftmaxSurrogateEnvironment(
            measurement_path=measurement_path,
            budget=6,
            seed=task_seed,
            train_task_ids=train_task_ids,
        )
        reset_out = env.reset(task=task, seed=task_seed)
        decision_start = time.perf_counter()
        surrogate_config_id = _choose_surrogate_config_id(env, acquisition=acquisition, beta=beta, xi=xi)
        decision_ms = float((time.perf_counter() - decision_start) * 1000.0)
        surrogate_config = env.config_info(surrogate_config_id)
        sample = torch.randn((env._task_rows[0].m, env._task_rows[0].n), device="cuda", dtype=torch.float16)

        surrogate_first_call_ms = _compile_plus_first_call_fixed(sample, surrogate_config)
        surrogate_steady_ms = _steady_state_fixed(sample, surrogate_config, repeats=repeats, warmup=warmup)
        autotune_metrics = _benchmark_triton_autotune(sample, repeats=repeats, warmup=warmup)
        oracle_best = env.oracle_best()

        results[task] = {
            "seeded_config_ids": reset_out["observation"]["tried_config_ids"],
            "train_task_count": len(train_task_ids),
            "oracle_best_ms": oracle_best["median_ms"],
            "surrogate": {
                "config": surrogate_config,
                "decision_ms": decision_ms,
                "compile_plus_first_call_ms": surrogate_first_call_ms,
                "steady_ms": surrogate_steady_ms,
                "regret_vs_oracle": float(surrogate_steady_ms / oracle_best["median_ms"] - 1.0),
            },
            "triton_autotune": {
                **autotune_metrics,
                "regret_vs_oracle": float(autotune_metrics["autotune_steady_ms"] / oracle_best["median_ms"] - 1.0),
            },
        }
        print(
            "[progress] finished"
            f" task={task}"
            f" surrogate_first_ms={results[task]['surrogate']['compile_plus_first_call_ms']:.3f}"
            f" surrogate_steady_ms={results[task]['surrogate']['steady_ms']:.3f}"
            f" autotune_first_ms={results[task]['triton_autotune']['autotune_first_call_ms']:.3f}"
            f" autotune_steady_ms={results[task]['triton_autotune']['autotune_steady_ms']:.3f}"
        )
        snapshot = _build_summary(
            measurement_path=measurement_path,
            tasks=tasks,
            acquisition=acquisition,
            beta=beta,
            xi=xi,
            results=results,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    return _build_summary(
        measurement_path=measurement_path,
        tasks=tasks,
        acquisition=acquisition,
        beta=beta,
        xi=xi,
        results=results,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick benchmark: surrogate-guided softmax config choice vs Triton autotune.")
    parser.add_argument("--measurement-path", default="data/autotune_measurements.csv")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--acquisition", choices=("mean", "ucb", "ei"), default="ucb")
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--xi", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=Path("outputs/surrogate_vs_triton_autotune.json"))
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_benchmark(
        measurement_path=args.measurement_path,
        tasks=args.tasks,
        repeats=args.repeats,
        warmup=args.warmup,
        seed=args.seed,
        acquisition=args.acquisition,
        beta=args.beta,
        xi=args.xi,
        output_path=args.output,
        resume=not args.no_resume,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
