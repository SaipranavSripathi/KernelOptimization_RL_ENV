#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

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
    raise RuntimeError(
        "Triton is required for the multi-family measurement collector."
    ) from err

from scripts.collect_measurements import (
    BLOCK_SIZES,
    NUM_STAGES,
    NUM_WARPS,
    N_VALUES,
    benchmark_single_config as benchmark_softmax_config,
)


EPS = 1e-5
GROUPED_GEMM_GROUP_COUNT = 4
GROUPED_GEMM_K = 512


@dataclass(frozen=True)
class MultiFamilyMeasurementRow:
    family_group: str
    family: str
    task_id: str
    m: int
    n: int
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
def fused_rowwise_layernorm_kernel(
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
    x_ptr = X_ptr + row_idx * stride_xm + col_offsets
    y_ptr = Y_ptr + row_idx * stride_ym + col_offsets
    mask = col_offsets < n_cols

    x = tl.load(x_ptr, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / n_cols
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / n_cols
    inv_std = tl.rsqrt(var + eps)
    y = x_centered * inv_std
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


def _task_id(family: str, m: int, n: int) -> str:
    if family == "grouped_gemm":
        return f"{family}_g{GROUPED_GEMM_GROUP_COUNT}_k{GROUPED_GEMM_K}_m{m}_n{n}"
    return f"{family}_m{m}_n{n}"


def _valid_configs(family: str, n: int) -> List[Tuple[int, int, int]]:
    configs: List[Tuple[int, int, int]] = []
    if family == "grouped_gemm":
        candidate_blocks = tuple(block for block in BLOCK_SIZES if block <= 1024)
    else:
        candidate_blocks = BLOCK_SIZES
    for block_size in candidate_blocks:
        if family != "grouped_gemm" and block_size < n:
            continue
        for num_warps in NUM_WARPS:
            for num_stages in NUM_STAGES:
                configs.append((block_size, num_warps, num_stages))
    return configs


def _effective_gbps(sample: torch.Tensor, median_ms: float) -> float:
    bytes_processed = float(sample.numel() * sample.element_size() * 2)
    if median_ms <= 0:
        return 0.0
    return bytes_processed / (median_ms / 1000.0) / 1e9


def _benchmark_layernorm_config(
    sample: torch.Tensor,
    block_size: int,
    num_warps: int,
    num_stages: int,
    repeats: int,
    warmup: int,
) -> float:
    output = torch.empty_like(sample)
    grid = (sample.shape[0],)

    def launch() -> None:
        fused_rowwise_layernorm_kernel[grid](
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
        result = ttesting.do_bench(
            launch,
            warmup=0,
            rep=repeats,
            quantiles=[0.5],
            return_mode="median",
        )
        if isinstance(result, dict):
            return float(result.get("median", 0.0))
        return float(result)

    events: List[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(max(1, repeats)):
        torch.cuda.synchronize()
        start.record()
        launch()
        end.record()
        end.synchronize()
        events.append(start.elapsed_time(end))
    return float(np.median(np.asarray(events, dtype=np.float32)))


def _validate_layernorm(sample: torch.Tensor, block_size: int, num_warps: int, num_stages: int) -> float:
    ref = torch.nn.functional.layer_norm(sample.float(), (sample.shape[1],), eps=EPS).to(sample.dtype)
    out = torch.empty_like(sample)
    fused_rowwise_layernorm_kernel[(sample.shape[0],)](
        sample,
        out,
        sample.stride(0),
        sample.stride(1),
        out.stride(0),
        out.stride(1),
        sample.shape[1],
        EPS,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    torch.cuda.synchronize()
    return float((out.to(torch.float32) - ref).abs().max().item())


def benchmark_layernorm_config(
    n: int,
    block_size: int,
    num_warps: int,
    num_stages: int,
    m: int = 4096,
    repeats: int = 200,
    warmup: int = 25,
    seed: int = 0,
) -> MultiFamilyMeasurementRow:
    if block_size < n:
        raise ValueError(f"Invalid config: BLOCK_SIZE {block_size} < N {n}")

    torch.manual_seed(seed)
    sample = torch.randn((m, n), device="cuda", dtype=torch.float16)
    median_ms = _benchmark_layernorm_config(
        sample=sample,
        block_size=block_size,
        num_warps=num_warps,
        num_stages=num_stages,
        repeats=repeats,
        warmup=warmup,
    )
    val_err = _validate_layernorm(
        sample=sample,
        block_size=block_size,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    gbps = _effective_gbps(sample, median_ms)
    return _make_row(
        family="layernorm",
        m=m,
        n=n,
        block_size=block_size,
        num_warps=num_warps,
        num_stages=num_stages,
        median_ms=median_ms,
        effective_gbps=gbps,
        validation_error=val_err,
    )


def _grouped_gemm_shapes(m: int, n: int) -> Tuple[int, int, int]:
    group_m = max(64, m // GROUPED_GEMM_GROUP_COUNT)
    return GROUPED_GEMM_GROUP_COUNT, group_m, GROUPED_GEMM_K


def _matmul_meta_from_block(block_size: int) -> Tuple[int, int, int]:
    block_m = max(32, min(block_size // 4, 256))
    block_n = max(32, min(block_size // 4, 256))
    block_k = 32
    return block_m, block_n, block_k


def _benchmark_grouped_gemm_config(
    a_groups: Sequence[torch.Tensor],
    b_groups: Sequence[torch.Tensor],
    block_size: int,
    num_warps: int,
    num_stages: int,
    repeats: int,
    warmup: int,
) -> float:
    c_groups = [torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype) for a, b in zip(a_groups, b_groups)]
    block_m, block_n, block_k = _matmul_meta_from_block(block_size)

    def launch() -> None:
        for a, b, c in zip(a_groups, b_groups, c_groups):
            grid = (triton.cdiv(a.shape[0], block_m) * triton.cdiv(b.shape[1], block_n),)
            matmul_kernel[grid](
                a,
                b,
                c,
                a.shape[0],
                b.shape[1],
                a.shape[1],
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
        result = ttesting.do_bench(
            launch,
            warmup=0,
            rep=repeats,
            quantiles=[0.5],
            return_mode="median",
        )
        if isinstance(result, dict):
            return float(result.get("median", 0.0))
        return float(result)

    durations_ms: List[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(max(1, repeats)):
        torch.cuda.synchronize()
        start.record()
        launch()
        end.record()
        end.synchronize()
        durations_ms.append(start.elapsed_time(end))
    return float(np.median(np.asarray(durations_ms, dtype=np.float32)))


def _validate_grouped_gemm(
    a_groups: Sequence[torch.Tensor],
    b_groups: Sequence[torch.Tensor],
    block_size: int,
    num_warps: int,
    num_stages: int,
) -> float:
    c_groups = [torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype) for a, b in zip(a_groups, b_groups)]
    block_m, block_n, block_k = _matmul_meta_from_block(block_size)
    for a, b, c in zip(a_groups, b_groups, c_groups):
        grid = (triton.cdiv(a.shape[0], block_m) * triton.cdiv(b.shape[1], block_n),)
        matmul_kernel[grid](
            a,
            b,
            c,
            a.shape[0],
            b.shape[1],
            a.shape[1],
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
    torch.cuda.synchronize()

    max_err = 0.0
    for a, b, c in zip(a_groups, b_groups, c_groups):
        ref = torch.matmul(a.float(), b.float()).to(torch.float16)
        max_err = max(max_err, float((c.float() - ref.float()).abs().max().item()))
    return max_err


def benchmark_grouped_gemm_config(
    n: int,
    block_size: int,
    num_warps: int,
    num_stages: int,
    m: int = 4096,
    repeats: int = 200,
    warmup: int = 25,
    seed: int = 0,
) -> MultiFamilyMeasurementRow:
    group_count, group_m, k_dim = _grouped_gemm_shapes(m, n)
    torch.manual_seed(seed)
    a_groups = [torch.randn((group_m, k_dim), device="cuda", dtype=torch.float16) for _ in range(group_count)]
    b_groups = [torch.randn((k_dim, n), device="cuda", dtype=torch.float16) for _ in range(group_count)]

    median_ms = _benchmark_grouped_gemm_config(
        a_groups=a_groups,
        b_groups=b_groups,
        block_size=block_size,
        num_warps=num_warps,
        num_stages=num_stages,
        repeats=repeats,
        warmup=warmup,
    )
    val_err = _validate_grouped_gemm(
        a_groups=a_groups,
        b_groups=b_groups,
        block_size=block_size,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    bytes_processed = group_count * (
        a_groups[0].numel() * a_groups[0].element_size()
        + b_groups[0].numel() * b_groups[0].element_size()
        + group_m * n * a_groups[0].element_size()
    )
    effective_gbps = bytes_processed / (median_ms / 1000.0) / 1e9 if median_ms > 0 else 0.0
    return _make_row(
        family="grouped_gemm",
        m=m,
        n=n,
        block_size=block_size,
        num_warps=num_warps,
        num_stages=num_stages,
        median_ms=median_ms,
        effective_gbps=effective_gbps,
        validation_error=val_err,
    )


def _make_row(
    family: str,
    m: int,
    n: int,
    block_size: int,
    num_warps: int,
    num_stages: int,
    median_ms: float,
    effective_gbps: float,
    validation_error: float,
) -> MultiFamilyMeasurementRow:
    configs = _valid_configs(family, n)
    config_id = configs.index((block_size, num_warps, num_stages))
    return MultiFamilyMeasurementRow(
        family_group="A" if family in {"softmax", "layernorm"} else "B",
        family=family,
        task_id=_task_id(family, m, n),
        m=m,
        n=n,
        config_id=config_id,
        block_size=block_size,
        num_warps=num_warps,
        num_stages=num_stages,
        shape_json=json.dumps(
            {
                "family": family,
                "m": m,
                "n": n,
                "group_count": GROUPED_GEMM_GROUP_COUNT if family == "grouped_gemm" else None,
                "k": GROUPED_GEMM_K if family == "grouped_gemm" else None,
            },
            sort_keys=True,
        ),
        config_json=json.dumps(
            {
                "block_size": block_size,
                "num_warps": num_warps,
                "num_stages": num_stages,
            },
            sort_keys=True,
        ),
        median_ms=float(median_ms),
        effective_gbps=float(effective_gbps),
        score=float(-math.log(max(median_ms, np.finfo(float).tiny))),
        validation_error=float(validation_error),
    )


def _softmax_row_to_multi(row: object) -> MultiFamilyMeasurementRow:
    return _make_row(
        family="softmax",
        m=int(row.m),
        n=int(row.n),
        block_size=int(row.block_size),
        num_warps=int(row.num_warps),
        num_stages=int(row.num_stages),
        median_ms=float(row.median_ms),
        effective_gbps=float(row.effective_gbps),
        validation_error=float(row.validation_error),
    )


def collect_multifamily_measurements(
    output_path: Path,
    families: Sequence[str],
    n_values: Iterable[int],
    repeats: int,
    warmup: int,
    seed: int,
    m: int = 4096,
    append: bool = True,
) -> List[MultiFamilyMeasurementRow]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run measurements on GPU.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = set()
    if output_path.exists():
        with output_path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                existing.add(_row_key(row["family"], int(row["m"]), int(row["n"]), int(row["config_id"])))

    results: List[MultiFamilyMeasurementRow] = []
    for family in families:
        if family not in {"softmax", "layernorm", "grouped_gemm"}:
            raise ValueError(f"Unsupported family: {family}")
        for n in n_values:
            for config_id, (block_size, num_warps, num_stages) in enumerate(_valid_configs(family, n)):
                key = _row_key(family, m, n, config_id)
                if append and key in existing:
                    continue
                if family == "softmax":
                    row = _softmax_row_to_multi(
                        benchmark_softmax_config(
                            n=n,
                            block_size=block_size,
                            num_warps=num_warps,
                            num_stages=num_stages,
                            m=m,
                            repeats=repeats,
                            warmup=warmup,
                            seed=seed,
                        )
                    )
                elif family == "layernorm":
                    row = benchmark_layernorm_config(
                        n=n,
                        block_size=block_size,
                        num_warps=num_warps,
                        num_stages=num_stages,
                        m=m,
                        repeats=repeats,
                        warmup=warmup,
                        seed=seed,
                    )
                else:
                    row = benchmark_grouped_gemm_config(
                        n=n,
                        block_size=block_size,
                        num_warps=num_warps,
                        num_stages=num_stages,
                        m=m,
                        repeats=repeats,
                        warmup=warmup,
                        seed=seed,
                    )
                results.append(row)

    if results:
        write_mode = "a" if output_path.exists() and append else "w"
        with output_path.open(write_mode, newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if write_mode == "w" or output_path.stat().st_size == 0:
                writer.writerow(
                    [
                        "family_group",
                        "family",
                        "task_id",
                        "m",
                        "n",
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
            for row in results:
                writer.writerow(
                    [
                        row.family_group,
                        row.family,
                        row.task_id,
                        row.m,
                        row.n,
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
    return results


def _row_key(family: str, m: int, n: int, config_id: int) -> str:
    return f"{family}|{m}|{n}|{config_id}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect multi-family GPU measurements.")
    parser.add_argument("--output", type=Path, default=Path("data/autotune_measurements.csv"))
    parser.add_argument(
        "--families",
        nargs="+",
        default=("softmax", "layernorm", "grouped_gemm"),
        choices=("softmax", "layernorm", "grouped_gemm"),
    )
    parser.add_argument("--n-cols", type=int, nargs="+", default=N_VALUES)
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    append = args.append and not args.overwrite
    collect_multifamily_measurements(
        output_path=args.output,
        families=args.families,
        n_values=args.n_cols,
        repeats=args.repeats,
        warmup=args.warmup,
        seed=args.seed,
        m=args.m,
        append=append,
    )


if __name__ == "__main__":
    main()
