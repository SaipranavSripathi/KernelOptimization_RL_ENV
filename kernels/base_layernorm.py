from __future__ import annotations

import math
import os
import statistics
from typing import Dict, List

import numpy as np
import torch
import triton
import triton.language as tl

try:
    import triton.testing as ttesting
except Exception:
    ttesting = None


EPS = 1e-5


@triton.jit
def generated_layernorm_kernel(
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


def benchmark_generated_kernel(
    n: int,
    block_size: int,
    num_warps: int,
    num_stages: int,
    m: int = 4096,
    repeats: int = 200,
    warmup: int = 25,
    seed: int = 0,
) -> Dict[str, float]:
    repeats = int(os.environ.get("KERNEL_GENERATED_REPEATS", str(repeats)))
    warmup = int(os.environ.get("KERNEL_GENERATED_WARMUP", str(warmup)))
    torch.manual_seed(seed)
    sample = torch.randn((m, n), device="cuda", dtype=torch.float16)
    output = torch.empty_like(sample)
    grid = (m,)

    def launch() -> None:
        generated_layernorm_kernel[grid](
            sample,
            output,
            sample.stride(0),
            sample.stride(1),
            output.stride(0),
            output.stride(1),
            n,
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
        median_ms = float(result.get("median", 0.0) if isinstance(result, dict) else result)
    else:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times: List[float] = []
        for _ in range(max(1, repeats)):
            torch.cuda.synchronize()
            start.record()
            launch()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))
        median_ms = float(statistics.median(times))

    ref = torch.nn.functional.layer_norm(sample.float(), (sample.shape[1],), eps=EPS).to(sample.dtype)
    validation_error = float((output.float() - ref.float()).abs().max().item())
    effective_gbps = float(sample.numel() * sample.element_size() * 2) / (median_ms / 1000.0) / 1e9
    score = -math.log(max(median_ms, np.finfo(float).tiny))
    return {
        "median_ms": median_ms,
        "effective_gbps": effective_gbps,
        "score": float(score),
        "validation_error": validation_error,
    }
