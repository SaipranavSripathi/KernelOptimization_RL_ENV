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


GROUP_COUNT = 4
K_DIM = 512


@triton.jit
def generated_matmul_kernel(
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
    group_m = max(64, m // GROUP_COUNT)
    torch.manual_seed(seed)
    a_groups = [torch.randn((group_m, K_DIM), device="cuda", dtype=torch.float16) for _ in range(GROUP_COUNT)]
    b_groups = [torch.randn((K_DIM, n), device="cuda", dtype=torch.float16) for _ in range(GROUP_COUNT)]
    c_groups = [torch.empty((group_m, n), device="cuda", dtype=torch.float16) for _ in range(GROUP_COUNT)]

    block_m = max(32, min(block_size // 4, 256))
    block_n = max(32, min(block_size // 4, 256))
    block_k = 32

    def launch() -> None:
        for a, b, c in zip(a_groups, b_groups, c_groups):
            grid = (triton.cdiv(a.shape[0], block_m) * triton.cdiv(b.shape[1], block_n),)
            generated_matmul_kernel[grid](
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

    validation_error = 0.0
    for a, b, c in zip(a_groups, b_groups, c_groups):
        ref = torch.matmul(a.float(), b.float()).to(torch.float16)
        validation_error = max(validation_error, float((c.float() - ref.float()).abs().max().item()))

    bytes_processed = GROUP_COUNT * (
        a_groups[0].numel() * a_groups[0].element_size()
        + b_groups[0].numel() * b_groups[0].element_size()
        + group_m * n * a_groups[0].element_size()
    )
    effective_gbps = bytes_processed / (median_ms / 1000.0) / 1e9 if median_ms > 0 else 0.0
    score = -math.log(max(median_ms, np.finfo(float).tiny))
    return {
        "median_ms": median_ms,
        "effective_gbps": float(effective_gbps),
        "score": float(score),
        "validation_error": float(validation_error),
    }
