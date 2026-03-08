#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.collect_qwen_05b_measurements import EPS, benchmark_qwen_task
from scripts.qwen_05b_spec import QwenKernelTask, qwen_05b_tasks


TASK_BY_ID = {task.task_id: task for task in qwen_05b_tasks()}


def _bench_callable(fn, args: Tuple[Any, ...], repeats: int, warmup: int) -> float:
    for _ in range(max(1, warmup)):
        fn(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    durations = []
    for _ in range(max(1, repeats)):
        torch.cuda.synchronize()
        start.record()
        fn(*args)
        end.record()
        end.synchronize()
        durations.append(start.elapsed_time(end))
    return float(np.median(np.asarray(durations, dtype=np.float32)))


def _build_qwen_callable(task: QwenKernelTask, seed: int):
    torch.manual_seed(seed)
    if task.family == "softmax":
        x = torch.randn((task.m, task.n), device="cuda", dtype=torch.float16)

        def fn(inp: torch.Tensor):
            return torch.softmax(inp, dim=-1)

        return fn, (x,)

    if task.family == "rmsnorm":
        x = torch.randn((task.m, task.n), device="cuda", dtype=torch.float16)

        def fn(inp: torch.Tensor):
            return inp.float() * torch.rsqrt(inp.float().pow(2).mean(dim=-1, keepdim=True) + EPS)

        return fn, (x,)

    if task.family == "gemm":
        a = torch.randn((task.m, task.k), device="cuda", dtype=torch.float16)
        b = torch.randn((task.k, task.n), device="cuda", dtype=torch.float16)

        def fn(lhs: torch.Tensor, rhs: torch.Tensor):
            return torch.matmul(lhs, rhs)

        return fn, (a, b)

    raise ValueError(f"Unsupported family: {task.family}")


def _benchmark_torch(task: QwenKernelTask, seed: int, repeats: int, warmup: int) -> Dict[str, float]:
    eager_fn, args = _build_qwen_callable(task, seed)
    eager_latency_ms = _bench_callable(eager_fn, args, repeats=repeats, warmup=warmup)

    compiled_fn = torch.compile(eager_fn)
    torch.cuda.synchronize()
    start = time.perf_counter()
    compiled_fn(*args)
    torch.cuda.synchronize()
    compile_plus_first_call_ms = float((time.perf_counter() - start) * 1000.0)
    compiled_latency_ms = _bench_callable(compiled_fn, args, repeats=repeats, warmup=warmup)
    return {
        "eager_latency_ms": eager_latency_ms,
        "compile_plus_first_call_ms": compile_plus_first_call_ms,
        "compiled_latency_ms": compiled_latency_ms,
    }


def _task_best_configs(eval_results: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    task_map: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for section in eval_results["results"].values():
        for method in ("random", "surrogate"):
            for run in section["task_runs"][method]:
                task_map.setdefault(run["task"], {})[method] = run["best_overall"]["config"]
    return task_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark eager/torch.compile and best Triton configs for Qwen2.5-0.5B exact kernels.")
    parser.add_argument("--generalization-results", type=Path, default=Path("outputs/qwen_05b_generalization_eval.json"))
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=Path, default=Path("outputs/qwen_05b_runtime_references.json"))
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    generalization_results = json.loads(args.generalization_results.read_text(encoding="utf-8"))
    task_configs = _task_best_configs(generalization_results)

    results = {}
    if not args.no_resume and args.output.exists():
        try:
            results = json.loads(args.output.read_text(encoding="utf-8")).get("results", {})
        except Exception:
            results = {}

    for idx, task_id in enumerate(sorted(task_configs.keys())):
        if task_id in results:
            print(f"[progress] skipping completed runtime task {idx + 1}/{len(task_configs)}: {task_id}")
            continue
        print(f"[progress] runtime task {idx + 1}/{len(task_configs)}: {task_id}")
        task = TASK_BY_ID[task_id]
        seed = args.seed + idx
        torch_metrics = _benchmark_torch(task, seed=seed, repeats=args.repeats, warmup=args.warmup)
        triton_results = {
            method: benchmark_qwen_task(
                task=task,
                block_size=int(config["block_size"]),
                num_warps=int(config["num_warps"]),
                num_stages=int(config["num_stages"]),
                repeats=args.repeats,
                warmup=args.warmup,
                seed=seed,
            ).__dict__
            for method, config in task_configs[task_id].items()
        }
        results[task_id] = {
            "family": task.family,
            "role": task.role,
            "mode": task.mode,
            "torch": torch_metrics,
            "triton": triton_results,
            "speedups": {
                method: {
                    "vs_eager": float(torch_metrics["eager_latency_ms"] / row["median_ms"]),
                    "vs_compiled": float(torch_metrics["compiled_latency_ms"] / row["median_ms"]),
                }
                for method, row in triton_results.items()
            },
        }
        summary = {
            "generalization_results": str(args.generalization_results),
            "repeats": args.repeats,
            "warmup": args.warmup,
            "seed": args.seed,
            "task_count": len(results),
            "results": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(
            f"[progress] completed {task_id} "
            f"eager_ms={torch_metrics['eager_latency_ms']:.6f} "
            f"compiled_ms={torch_metrics['compiled_latency_ms']:.6f}"
        )

    summary = {
        "generalization_results": str(args.generalization_results),
        "repeats": args.repeats,
        "warmup": args.warmup,
        "seed": args.seed,
        "task_count": len(results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
