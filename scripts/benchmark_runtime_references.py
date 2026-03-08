#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.collect_measurements import benchmark_single_config as benchmark_softmax_config
from scripts.collect_multifamily_measurements import (
    EPS,
    GROUPED_GEMM_GROUP_COUNT,
    GROUPED_GEMM_K,
    benchmark_grouped_gemm_config,
    benchmark_layernorm_config,
)


TASK_RE = re.compile(
    r"^(?P<family>[a-z_]+?)(?:_g(?P<g>\d+)_k(?P<k>\d+))?_m(?P<m>\d+)_n(?P<n>\d+)$"
)


def _parse_task(task_id: str) -> Dict[str, int | str | None]:
    match = TASK_RE.match(task_id)
    if not match:
        raise ValueError(f"Cannot parse task id: {task_id}")
    data = match.groupdict()
    return {
        "family": data["family"],
        "m": int(data["m"]),
        "n": int(data["n"]),
        "g": int(data["g"]) if data["g"] is not None else None,
        "k": int(data["k"]) if data["k"] is not None else None,
    }


def _bench_callable(fn, args: Tuple[Any, ...], repeats: int, warmup: int) -> float:
    for _ in range(max(1, warmup)):
        fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    latencies_ms: List[float] = []
    for _ in range(max(1, repeats)):
        torch.cuda.synchronize()
        start.record()
        fn(*args)
        end.record()
        end.synchronize()
        latencies_ms.append(start.elapsed_time(end))
    return float(np.median(np.asarray(latencies_ms, dtype=np.float32)))


def _build_family_callable(task_meta: Dict[str, Any], seed: int) -> Tuple[Any, Tuple[Any, ...]]:
    family = str(task_meta["family"])
    m = int(task_meta["m"])
    n = int(task_meta["n"])
    torch.manual_seed(seed)

    if family == "softmax":
        x = torch.randn((m, n), device="cuda", dtype=torch.float16)

        def fn(inp: torch.Tensor) -> torch.Tensor:
            return torch.softmax(inp, dim=-1)

        return fn, (x,)

    if family == "layernorm":
        x = torch.randn((m, n), device="cuda", dtype=torch.float16)

        def fn(inp: torch.Tensor) -> torch.Tensor:
            return F.layer_norm(inp, (inp.shape[-1],), eps=EPS)

        return fn, (x,)

    if family == "grouped_gemm":
        group_count = int(task_meta.get("g") or GROUPED_GEMM_GROUP_COUNT)
        k_dim = int(task_meta.get("k") or GROUPED_GEMM_K)
        group_m = max(64, m // group_count)
        a_groups = [torch.randn((group_m, k_dim), device="cuda", dtype=torch.float16) for _ in range(group_count)]
        b_groups = [torch.randn((k_dim, n), device="cuda", dtype=torch.float16) for _ in range(group_count)]

        def fn(*inputs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
            mid = len(inputs) // 2
            a_list = inputs[:mid]
            b_list = inputs[mid:]
            return tuple(torch.matmul(a, b) for a, b in zip(a_list, b_list))

        return fn, tuple(a_groups + b_groups)

    raise ValueError(f"Unsupported family: {family}")


def _benchmark_torch_compile(task_meta: Dict[str, Any], seed: int, repeats: int, warmup: int) -> Dict[str, float]:
    eager_fn, args = _build_family_callable(task_meta, seed=seed)
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


def _benchmark_triton_best(task_id: str, config: Dict[str, Any], repeats: int, warmup: int, seed: int) -> Dict[str, Any]:
    task = _parse_task(task_id)
    family = str(task["family"])
    common = {
        "n": int(task["n"]),
        "block_size": int(config["block_size"]),
        "num_warps": int(config["num_warps"]),
        "num_stages": int(config["num_stages"]),
        "m": int(task["m"]),
        "repeats": repeats,
        "warmup": warmup,
        "seed": seed,
    }
    if family == "softmax":
        row = benchmark_softmax_config(**common)
        return row.__dict__
    if family == "layernorm":
        row = benchmark_layernorm_config(**common)
        return row.__dict__
    if family == "grouped_gemm":
        row = benchmark_grouped_gemm_config(**common)
        return row.__dict__
    raise ValueError(f"Unsupported family: {family}")


def _collect_task_best_configs(generalization_results: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    task_map: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for section in generalization_results["results"].values():
        for method in ("random", "surrogate"):
            for run in section["task_runs"][method]:
                task_map.setdefault(run["task"], {})[method] = run["best_overall"]["config"]
    return task_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark eager/torch.compile and live Triton configs on held-out tasks.")
    parser.add_argument(
        "--generalization-results",
        type=Path,
        default=Path("outputs/generalization_eval.json"),
    )
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/runtime_references.json"),
    )
    args = parser.parse_args()

    generalization_results = json.loads(args.generalization_results.read_text(encoding="utf-8"))
    task_best_configs = _collect_task_best_configs(generalization_results)

    results: Dict[str, Any] = {}
    for idx, task_id in enumerate(sorted(task_best_configs.keys())):
        task_seed = args.seed + idx
        task_meta = _parse_task(task_id)
        torch_metrics = _benchmark_torch_compile(task_meta, seed=task_seed, repeats=args.repeats, warmup=args.warmup)
        method_results = {
            method: _benchmark_triton_best(
                task_id=task_id,
                config=config,
                repeats=args.repeats,
                warmup=args.warmup,
                seed=task_seed,
            )
            for method, config in task_best_configs[task_id].items()
        }
        results[task_id] = {
            "family": task_meta["family"],
            "torch": torch_metrics,
            "triton": method_results,
            "speedups": {
                method: {
                    "vs_eager": float(torch_metrics["eager_latency_ms"] / row["median_ms"]),
                    "vs_compiled": float(torch_metrics["compiled_latency_ms"] / row["median_ms"]),
                }
                for method, row in method_results.items()
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
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
