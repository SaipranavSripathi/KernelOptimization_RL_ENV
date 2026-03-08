#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Optional

import numpy as np
import torch

try:
    import triton
    import triton.language as tl
    import triton.testing as ttesting
except Exception as err:  # pragma: no cover
    raise RuntimeError(
        "Triton is required for this measurement collector. Install Triton and rerun."
    ) from err


BLOCK_SIZES = (256, 512, 1024, 2048, 4096, 8192)
NUM_WARPS = (1, 2, 4, 8)
NUM_STAGES = (1, 2, 3, 4)
N_VALUES = (256, 512, 1024, 1536, 2048, 3072, 4096, 6144, 8192)


@dataclass(frozen=True)
class MeasurementRow:
    task_id: str
    m: int
    n: int
    block_size: int
    num_warps: int
    num_stages: int
    median_ms: float
    effective_gbps: float
    score: float
    validation_error: float


@triton.jit
def fused_rowwise_softmax_kernel(
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


def _task_id(m: int, n: int) -> str:
    return f"softmax_m{m}_n{n}"


def _effective_gbps(sample: torch.Tensor, median_ms: float) -> float:
    bytes_processed = float(sample.numel() * sample.element_size() * 2)
    if median_ms <= 0:
        return 0.0
    return bytes_processed / (median_ms / 1000.0) / 1e9


def _benchmark_config(
    sample: torch.Tensor,
    block_size: int,
    num_warps: int,
    num_stages: int,
    repeats: int,
    warmup: int,
) -> float:
    output = torch.empty_like(sample)
    m, n = sample.shape
    grid = (m,)

    def launch() -> None:
        fused_rowwise_softmax_kernel[grid](
            sample,
            output,
            sample.stride(0),
            sample.stride(1),
            output.stride(0),
            output.stride(1),
            n,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    # warmup to compile and stabilize caches / clocks.
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

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    durations_ms: List[float] = []
    for _ in range(max(1, repeats)):
        torch.cuda.synchronize()
        start.record()
        launch()
        end.record()
        end.synchronize()
        durations_ms.append(start.elapsed_time(end))
    return float(statistics.median(durations_ms))


def _validate_correctness(sample: torch.Tensor, block_size: int, num_warps: int, num_stages: int) -> float:
    ref = torch.softmax(sample.float(), dim=-1).to(sample.dtype)
    out = torch.empty_like(sample)

    fused_rowwise_softmax_kernel[(sample.shape[0],)](
        sample,
        out,
        sample.stride(0),
        sample.stride(1),
        out.stride(0),
        out.stride(1),
        sample.shape[1],
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    torch.cuda.synchronize()

    err = (out.to(torch.float32) - ref).abs().max().item()
    return float(err)


def benchmark_single_config(
    n: int,
    block_size: int,
    num_warps: int,
    num_stages: int,
    m: int = 4096,
    repeats: int = 200,
    warmup: int = 25,
    seed: int = 0,
) -> MeasurementRow:
    if block_size < n:
        raise ValueError(f"Invalid config: BLOCK_SIZE {block_size} < N {n}")

    torch.manual_seed(seed)
    sample = torch.randn((m, n), device="cuda", dtype=torch.float16)

    if num_warps not in NUM_WARPS:
        raise ValueError(f"Unsupported num_warps={num_warps}")
    if num_stages not in NUM_STAGES:
        raise ValueError(f"Unsupported num_stages={num_stages}")
    if block_size not in BLOCK_SIZES:
        raise ValueError(f"Unsupported BLOCK_SIZE={block_size}")

    median_ms = _benchmark_config(
        sample=sample,
        block_size=block_size,
        num_warps=num_warps,
        num_stages=num_stages,
        repeats=repeats,
        warmup=warmup,
    )

    val_err = _validate_correctness(
        sample=sample,
        block_size=block_size,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    gbps = _effective_gbps(sample, median_ms)
    score = -math.log(max(median_ms, np.finfo(float).tiny))
    return MeasurementRow(
        task_id=_task_id(m, n),
        m=m,
        n=n,
        block_size=block_size,
        num_warps=num_warps,
        num_stages=num_stages,
        median_ms=float(median_ms),
        effective_gbps=float(gbps),
        score=float(score),
        validation_error=float(val_err),
    )


def collect_measurements(
    output_path: Path,
    n_values: Iterable[int],
    repeats: int,
    warmup: int,
    seed: int,
    m: int = 4096,
    append: bool = True,
) -> List[MeasurementRow]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run measurements on H100.")
    if not torch.cuda.get_device_name(0):
        raise RuntimeError("No CUDA device found.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing: Dict[str, MeasurementRow] = {}
    if output_path.exists():
        with output_path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = _row_key(
                    int(row["m"]),
                    int(row["n"]),
                    int(row["block_size"]),
                    int(row["num_warps"]),
                    int(row["num_stages"]),
                )
                existing[key] = MeasurementRow(
                    task_id=row["task_id"],
                    m=int(row["m"]),
                    n=int(row["n"]),
                    block_size=int(row["block_size"]),
                    num_warps=int(row["num_warps"]),
                    num_stages=int(row["num_stages"]),
                    median_ms=float(row["median_ms"]),
                    effective_gbps=float(row["effective_gbps"]),
                    score=float(row["score"]),
                    validation_error=float(row["validation_error"]),
                )

    results: List[MeasurementRow] = []
    for n in n_values:
        if n < 0:
            raise ValueError(f"Invalid n value: {n}")
        for block in BLOCK_SIZES:
            if block < n:
                continue
            for num_warps in NUM_WARPS:
                for num_stages in NUM_STAGES:
                    key = _row_key(m, n, block, num_warps, num_stages)
                    if append and key in existing:
                        continue

                    torch.cuda.synchronize()
                    row = benchmark_single_config(
                        n=n,
                        block_size=block,
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
        with output_path.open(write_mode, newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_mode == "w" or (output_path.stat().st_size == 0):
                writer.writerow(
                    [
                        "task_id",
                        "m",
                        "n",
                        "block_size",
                        "num_warps",
                        "num_stages",
                        "median_ms",
                        "effective_gbps",
                        "score",
                        "validation_error",
                    ]
                )
            for r in results:
                writer.writerow(
                    [
                        r.task_id,
                        r.m,
                        r.n,
                        r.block_size,
                        r.num_warps,
                        r.num_stages,
                        r.median_ms,
                        r.effective_gbps,
                        r.score,
                        r.validation_error,
                    ]
                )
    return results


def _row_key(m: int, n: int, block_size: int, num_warps: int, num_stages: int) -> str:
    return f"{m}|{n}|{block_size}|{num_warps}|{num_stages}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect real H100 measurements for Triton row-wise fused softmax."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/softmax_measurements.csv"),
        help="CSV output path.",
    )
    parser.add_argument(
        "--n-cols",
        type=int,
        nargs="+",
        default=N_VALUES,
        help="Softmax inner dimension N values to benchmark.",
    )
    parser.add_argument("--m", type=int, default=4096, help="Outer dimension M.")
    parser.add_argument("--repeats", type=int, default=200, help="Benchmark repeats.")
    parser.add_argument("--warmup", type=int, default=25, help="Benchmark warmup runs.")
    parser.add_argument("--seed", type=int, default=0, help="Torch/random seed.")
    parser.add_argument(
        "--single-run",
        action="store_true",
        help="Run one specific config and print JSON-like output.",
    )
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing CSV file (default). If false, overwrite.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing CSV data instead of appending.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    if not torch.cuda.get_device_name(0):
        raise RuntimeError("No CUDA device was found.")

    append = args.append and not args.overwrite
    if args.single_run:
        row = benchmark_single_config(
            n=args.n_cols[0],
            block_size=args.block_size,
            num_warps=args.num_warps,
            num_stages=args.num_stages,
            m=args.m,
            repeats=args.repeats,
            warmup=args.warmup,
            seed=args.seed,
        )
        print(row.__dict__)
        return

    collect_measurements(
        output_path=args.output,
        n_values=args.n_cols,
        repeats=args.repeats,
        warmup=args.warmup,
        seed=args.seed,
        m=args.m,
        append=append,
    )


if __name__ == "__main__":
    main()
