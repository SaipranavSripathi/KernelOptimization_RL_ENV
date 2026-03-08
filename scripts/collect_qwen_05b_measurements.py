#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

try:
    import triton
    import triton.language as tl
    import triton.testing as ttesting
except Exception as err:  # pragma: no cover
    raise RuntimeError("Triton is required for Qwen kernel measurement.") from err

from scripts.collect_measurements import BLOCK_SIZES, NUM_STAGES, NUM_WARPS, benchmark_single_config
from scripts.qwen_05b_spec import QwenKernelTask, qwen_05b_tasks


EPS = 1e-5


@dataclass(frozen=True)
class QwenMeasurementRow:
    family_group: str
    family: str
    task_id: str
    m: int
    n: int
    k: int
    config_id: int
    block_size: int
    num_warps: int
    num_stages: int
    shape_json: str
    config_json: str
    median_ms: float
    effective_gbps: float
    score: float
    validation_error: float


@triton.jit
def fused_rowwise_rmsnorm_kernel(
    X_ptr,
    Y_ptr,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    x_ptr = X_ptr + row_idx * stride_xm + col_offsets
    y_ptr = Y_ptr + row_idx * stride_ym + col_offsets

    x = tl.load(x_ptr, mask=mask, other=0.0).to(tl.float32)
    mean_sq = tl.sum(x * x, axis=0) / n_cols
    inv_rms = tl.rsqrt(mean_sq + eps)
    y = x * inv_rms
    tl.store(y_ptr, y.to(tl.float16), mask=mask)


@triton.jit
def matmul_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        offs_k += BLOCK_K

    c = acc.to(tl.float16)
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def _gemm_blocks(block_size: int) -> tuple[int, int, int]:
    block_m = max(16, min(block_size // 4, 128))
    block_n = max(16, min(block_size // 4, 128))
    block_k = 32
    return block_m, block_n, block_k


def _effective_gbps(bytes_processed: int, median_ms: float) -> float:
    if median_ms <= 0:
        return 0.0
    return float(bytes_processed) / (median_ms / 1000.0) / 1e9


def _score(ms: float) -> float:
    return float(-math.log(max(ms, np.finfo(float).tiny)))


def _config_json(block_size: int, num_warps: int, num_stages: int) -> str:
    return json.dumps(
        {"block_size": block_size, "num_warps": num_warps, "num_stages": num_stages},
        sort_keys=True,
    )


def _valid_configs(task: QwenKernelTask) -> List[tuple[int, int, int]]:
    configs: List[tuple[int, int, int]] = []
    for block_size in BLOCK_SIZES:
        if task.family in {"softmax", "rmsnorm"} and block_size < task.n:
            continue
        if task.family == "gemm" and block_size > 1024:
            continue
        for num_warps in NUM_WARPS:
            for num_stages in NUM_STAGES:
                configs.append((block_size, num_warps, num_stages))
    return configs


def _benchmark_rmsnorm(task: QwenKernelTask, block_size: int, num_warps: int, num_stages: int, repeats: int, warmup: int, seed: int) -> QwenMeasurementRow:
    torch.manual_seed(seed)
    sample = torch.randn((task.m, task.n), device="cuda", dtype=torch.float16)
    output = torch.empty_like(sample)
    grid = (sample.shape[0],)

    def launch() -> None:
        fused_rowwise_rmsnorm_kernel[grid](
            sample,
            output,
            sample.stride(0),
            sample.stride(1),
            output.stride(0),
            output.stride(1),
            sample.shape[1],
            EPS,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    for _ in range(max(1, warmup)):
        launch()
    torch.cuda.synchronize()
    if ttesting is not None:
        result = ttesting.do_bench(launch, warmup=0, rep=repeats, quantiles=[0.5], return_mode="median")
        median_ms = float(result.get("median", 0.0) if isinstance(result, dict) else result)
    else:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        durations: List[float] = []
        for _ in range(max(1, repeats)):
            torch.cuda.synchronize()
            start.record()
            launch()
            end.record()
            end.synchronize()
            durations.append(start.elapsed_time(end))
        median_ms = float(np.median(np.asarray(durations, dtype=np.float32)))

    ref = sample.float() * torch.rsqrt(sample.float().pow(2).mean(dim=-1, keepdim=True) + EPS)
    fused_rowwise_rmsnorm_kernel[grid](
        sample,
        output,
        sample.stride(0),
        sample.stride(1),
        output.stride(0),
        output.stride(1),
        sample.shape[1],
        EPS,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    torch.cuda.synchronize()
    max_err = float((output.float() - ref.float()).abs().max().item())
    config_id = _valid_configs(task).index((block_size, num_warps, num_stages))
    return QwenMeasurementRow(
        family_group=task.family_group,
        family=task.family,
        task_id=task.task_id,
        m=task.m,
        n=task.n,
        k=task.k,
        config_id=config_id,
        block_size=block_size,
        num_warps=num_warps,
        num_stages=num_stages,
        shape_json=json.dumps(task.shape_fields(), sort_keys=True),
        config_json=_config_json(block_size, num_warps, num_stages),
        median_ms=median_ms,
        effective_gbps=_effective_gbps(sample.numel() * sample.element_size() * 2, median_ms),
        score=_score(median_ms),
        validation_error=max_err,
    )


def _benchmark_gemm(task: QwenKernelTask, block_size: int, num_warps: int, num_stages: int, repeats: int, warmup: int, seed: int) -> QwenMeasurementRow:
    torch.manual_seed(seed)
    a = torch.randn((task.m, task.k), device="cuda", dtype=torch.float16)
    b = torch.randn((task.k, task.n), device="cuda", dtype=torch.float16)
    c = torch.empty((task.m, task.n), device="cuda", dtype=torch.float16)
    block_m, block_n, block_k = _gemm_blocks(block_size)

    def launch() -> None:
        grid = (triton.cdiv(task.m, block_m) * triton.cdiv(task.n, block_n),)
        matmul_kernel[grid](
            a,
            b,
            c,
            task.m,
            task.n,
            task.k,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    for _ in range(max(1, warmup)):
        launch()
    torch.cuda.synchronize()
    if ttesting is not None:
        result = ttesting.do_bench(launch, warmup=0, rep=repeats, quantiles=[0.5], return_mode="median")
        median_ms = float(result.get("median", 0.0) if isinstance(result, dict) else result)
    else:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        durations: List[float] = []
        for _ in range(max(1, repeats)):
            torch.cuda.synchronize()
            start.record()
            launch()
            end.record()
            end.synchronize()
            durations.append(start.elapsed_time(end))
        median_ms = float(np.median(np.asarray(durations, dtype=np.float32)))

    ref = torch.matmul(a.float(), b.float()).to(torch.float16)
    launch()
    torch.cuda.synchronize()
    max_err = float((c.float() - ref.float()).abs().max().item())
    config_id = _valid_configs(task).index((block_size, num_warps, num_stages))
    bytes_processed = a.numel() * a.element_size() + b.numel() * b.element_size() + c.numel() * c.element_size()
    return QwenMeasurementRow(
        family_group=task.family_group,
        family=task.family,
        task_id=task.task_id,
        m=task.m,
        n=task.n,
        k=task.k,
        config_id=config_id,
        block_size=block_size,
        num_warps=num_warps,
        num_stages=num_stages,
        shape_json=json.dumps(task.shape_fields(), sort_keys=True),
        config_json=_config_json(block_size, num_warps, num_stages),
        median_ms=median_ms,
        effective_gbps=_effective_gbps(bytes_processed, median_ms),
        score=_score(median_ms),
        validation_error=max_err,
    )


def benchmark_qwen_task(task: QwenKernelTask, block_size: int, num_warps: int, num_stages: int, repeats: int, warmup: int, seed: int) -> QwenMeasurementRow:
    if task.family == "softmax":
        row = benchmark_single_config(
            n=task.n,
            block_size=block_size,
            num_warps=num_warps,
            num_stages=num_stages,
            m=task.m,
            repeats=repeats,
            warmup=warmup,
            seed=seed,
        )
        config_id = _valid_configs(task).index((block_size, num_warps, num_stages))
        return QwenMeasurementRow(
            family_group=task.family_group,
            family=task.family,
            task_id=task.task_id,
            m=task.m,
            n=task.n,
            k=0,
            config_id=config_id,
            block_size=block_size,
            num_warps=num_warps,
            num_stages=num_stages,
            shape_json=json.dumps(task.shape_fields(), sort_keys=True),
            config_json=_config_json(block_size, num_warps, num_stages),
            median_ms=float(row.median_ms),
            effective_gbps=float(row.effective_gbps),
            score=float(row.score),
            validation_error=float(row.validation_error),
        )
    if task.family == "rmsnorm":
        return _benchmark_rmsnorm(task, block_size, num_warps, num_stages, repeats, warmup, seed)
    if task.family == "gemm":
        return _benchmark_gemm(task, block_size, num_warps, num_stages, repeats, warmup, seed)
    raise ValueError(f"Unsupported family: {task.family}")


def _write_header_if_needed(output_path: Path, append: bool) -> None:
    if append and output_path.exists() and output_path.stat().st_size > 0:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "family_group",
                "family",
                "task_id",
                "m",
                "n",
                "k",
                "config_id",
                "block_size",
                "num_warps",
                "num_stages",
                "shape_json",
                "config_json",
                "median_ms",
                "effective_gbps",
                "score",
                "validation_error",
            ]
        )


def _append_row(output_path: Path, row: QwenMeasurementRow) -> None:
    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                row.family_group,
                row.family,
                row.task_id,
                row.m,
                row.n,
                row.k,
                row.config_id,
                row.block_size,
                row.num_warps,
                row.num_stages,
                row.shape_json,
                row.config_json,
                row.median_ms,
                row.effective_gbps,
                row.score,
                row.validation_error,
            ]
        )


def collect_qwen_measurements(output_path: Path, repeats: int, warmup: int, seed: int, append: bool = True) -> List[QwenMeasurementRow]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = set()
    if output_path.exists():
        with output_path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                existing.add((row["task_id"], int(row["config_id"])))

    _write_header_if_needed(output_path, append=append)
    results: List[QwenMeasurementRow] = []
    tasks = qwen_05b_tasks()
    total_configs = sum(len(_valid_configs(task)) for task in tasks)
    completed = len(existing)
    print(f"[progress] qwen measurement existing={completed}/{total_configs}")
    for task_index, task in enumerate(tasks, start=1):
        task_configs = _valid_configs(task)
        pending = sum(1 for config_id in range(len(task_configs)) if (task.task_id, config_id) not in existing)
        print(
            f"[progress] task {task_index}/{len(tasks)} family={task.family} role={task.role} mode={task.mode} "
            f"shape=({task.m},{task.n},{task.k}) pending={pending}/{len(task_configs)}"
        )
        for config_id, (block_size, num_warps, num_stages) in enumerate(task_configs):
            key = (task.task_id, config_id)
            if append and key in existing:
                continue
            row = benchmark_qwen_task(
                task=task,
                block_size=block_size,
                num_warps=num_warps,
                num_stages=num_stages,
                repeats=repeats,
                warmup=warmup,
                seed=seed,
            )
            results.append(row)
            _append_row(output_path, row)
            completed += 1
            print(
                f"[progress] completed {completed}/{total_configs} "
                f"task={task.task_id} config_id={config_id} median_ms={row.median_ms:.6f}"
            )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect exact-kernel measurements for Qwen2.5-0.5B.")
    parser.add_argument("--output", type=Path, default=Path("data/qwen_05b_measurements.csv"))
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    append = args.append and not args.overwrite
    collect_qwen_measurements(
        output_path=args.output,
        repeats=args.repeats,
        warmup=args.warmup,
        seed=args.seed,
        append=append,
    )


if __name__ == "__main__":
    main()
