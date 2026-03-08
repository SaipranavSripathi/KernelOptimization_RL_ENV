#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


MODEL_ID = "Qwen/Qwen2.5-0.5B"
HIDDEN_SIZE = 896
INTERMEDIATE_SIZE = 4864
NUM_HIDDEN_LAYERS = 24
NUM_ATTENTION_HEADS = 14
NUM_KEY_VALUE_HEADS = 2
HEAD_DIM = HIDDEN_SIZE // NUM_ATTENTION_HEADS
KV_PROJ_SIZE = NUM_KEY_VALUE_HEADS * HEAD_DIM
DTYPE = "bfloat16"
BATCH_SIZE = 1

PREFILL_SEQ_LENS = (128, 512, 2048)
DECODE_CTX_LENS = (128, 512, 2048, 8192)

MODE_IDS = {
    "prefill": 1,
    "decode": 2,
}

ROLE_IDS = {
    "rmsnorm": 1,
    "attn_softmax": 2,
    "q_proj": 3,
    "k_proj": 4,
    "v_proj": 5,
    "o_proj": 6,
    "gate_proj": 7,
    "up_proj": 8,
    "down_proj": 9,
}


@dataclass(frozen=True)
class QwenKernelTask:
    family_group: str
    family: str
    task_id: str
    role: str
    mode: str
    m: int
    n: int
    k: int
    seq_len: int
    ctx_len: int

    def shape_fields(self) -> Dict[str, int | str]:
        return {
            "family_group": self.family_group,
            "family": self.family,
            "role": self.role,
            "mode": self.mode,
            "role_id": ROLE_IDS[self.role],
            "mode_id": MODE_IDS[self.mode],
            "m": self.m,
            "n": self.n,
            "k": self.k,
            "seq_len": self.seq_len,
            "ctx_len": self.ctx_len,
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": INTERMEDIATE_SIZE,
            "num_attention_heads": NUM_ATTENTION_HEADS,
            "num_key_value_heads": NUM_KEY_VALUE_HEADS,
            "head_dim": HEAD_DIM,
            "dtype": DTYPE,
            "model_id": MODEL_ID,
        }


def _task_id(role: str, mode: str, m: int, n: int, k: int, seq_len: int, ctx_len: int) -> str:
    extra = f"_k{k}" if k > 0 else ""
    ctx = f"_ctx{ctx_len}" if ctx_len > 0 else ""
    return f"qwen05b_{role}_{mode}_m{m}_n{n}{extra}_seq{seq_len}{ctx}"


def qwen_05b_tasks() -> List[QwenKernelTask]:
    tasks: List[QwenKernelTask] = []

    for seq_len in PREFILL_SEQ_LENS:
        tasks.extend(
            [
                QwenKernelTask("QWEN", "rmsnorm", _task_id("rmsnorm", "prefill", seq_len, HIDDEN_SIZE, 0, seq_len, seq_len), "rmsnorm", "prefill", seq_len, HIDDEN_SIZE, 0, seq_len, seq_len),
                QwenKernelTask("QWEN", "softmax", _task_id("attn_softmax", "prefill", NUM_ATTENTION_HEADS * seq_len, seq_len, 0, seq_len, seq_len), "attn_softmax", "prefill", NUM_ATTENTION_HEADS * seq_len, seq_len, 0, seq_len, seq_len),
                QwenKernelTask("QWEN", "gemm", _task_id("q_proj", "prefill", seq_len, HIDDEN_SIZE, HIDDEN_SIZE, seq_len, seq_len), "q_proj", "prefill", seq_len, HIDDEN_SIZE, HIDDEN_SIZE, seq_len, seq_len),
                QwenKernelTask("QWEN", "gemm", _task_id("k_proj", "prefill", seq_len, KV_PROJ_SIZE, HIDDEN_SIZE, seq_len, seq_len), "k_proj", "prefill", seq_len, KV_PROJ_SIZE, HIDDEN_SIZE, seq_len, seq_len),
                QwenKernelTask("QWEN", "gemm", _task_id("v_proj", "prefill", seq_len, KV_PROJ_SIZE, HIDDEN_SIZE, seq_len, seq_len), "v_proj", "prefill", seq_len, KV_PROJ_SIZE, HIDDEN_SIZE, seq_len, seq_len),
                QwenKernelTask("QWEN", "gemm", _task_id("o_proj", "prefill", seq_len, HIDDEN_SIZE, HIDDEN_SIZE, seq_len, seq_len), "o_proj", "prefill", seq_len, HIDDEN_SIZE, HIDDEN_SIZE, seq_len, seq_len),
                QwenKernelTask("QWEN", "gemm", _task_id("gate_proj", "prefill", seq_len, INTERMEDIATE_SIZE, HIDDEN_SIZE, seq_len, seq_len), "gate_proj", "prefill", seq_len, INTERMEDIATE_SIZE, HIDDEN_SIZE, seq_len, seq_len),
                QwenKernelTask("QWEN", "gemm", _task_id("up_proj", "prefill", seq_len, INTERMEDIATE_SIZE, HIDDEN_SIZE, seq_len, seq_len), "up_proj", "prefill", seq_len, INTERMEDIATE_SIZE, HIDDEN_SIZE, seq_len, seq_len),
                QwenKernelTask("QWEN", "gemm", _task_id("down_proj", "prefill", seq_len, HIDDEN_SIZE, INTERMEDIATE_SIZE, seq_len, seq_len), "down_proj", "prefill", seq_len, HIDDEN_SIZE, INTERMEDIATE_SIZE, seq_len, seq_len),
            ]
        )

    for ctx_len in DECODE_CTX_LENS:
        tasks.extend(
            [
                QwenKernelTask("QWEN", "rmsnorm", _task_id("rmsnorm", "decode", 1, HIDDEN_SIZE, 0, 1, ctx_len), "rmsnorm", "decode", 1, HIDDEN_SIZE, 0, 1, ctx_len),
                QwenKernelTask("QWEN", "softmax", _task_id("attn_softmax", "decode", NUM_ATTENTION_HEADS, ctx_len, 0, 1, ctx_len), "attn_softmax", "decode", NUM_ATTENTION_HEADS, ctx_len, 0, 1, ctx_len),
                QwenKernelTask("QWEN", "gemm", _task_id("q_proj", "decode", 1, HIDDEN_SIZE, HIDDEN_SIZE, 1, ctx_len), "q_proj", "decode", 1, HIDDEN_SIZE, HIDDEN_SIZE, 1, ctx_len),
                QwenKernelTask("QWEN", "gemm", _task_id("k_proj", "decode", 1, KV_PROJ_SIZE, HIDDEN_SIZE, 1, ctx_len), "k_proj", "decode", 1, KV_PROJ_SIZE, HIDDEN_SIZE, 1, ctx_len),
                QwenKernelTask("QWEN", "gemm", _task_id("v_proj", "decode", 1, KV_PROJ_SIZE, HIDDEN_SIZE, 1, ctx_len), "v_proj", "decode", 1, KV_PROJ_SIZE, HIDDEN_SIZE, 1, ctx_len),
                QwenKernelTask("QWEN", "gemm", _task_id("o_proj", "decode", 1, HIDDEN_SIZE, HIDDEN_SIZE, 1, ctx_len), "o_proj", "decode", 1, HIDDEN_SIZE, HIDDEN_SIZE, 1, ctx_len),
                QwenKernelTask("QWEN", "gemm", _task_id("gate_proj", "decode", 1, INTERMEDIATE_SIZE, HIDDEN_SIZE, 1, ctx_len), "gate_proj", "decode", 1, INTERMEDIATE_SIZE, HIDDEN_SIZE, 1, ctx_len),
                QwenKernelTask("QWEN", "gemm", _task_id("up_proj", "decode", 1, INTERMEDIATE_SIZE, HIDDEN_SIZE, 1, ctx_len), "up_proj", "decode", 1, INTERMEDIATE_SIZE, HIDDEN_SIZE, 1, ctx_len),
                QwenKernelTask("QWEN", "gemm", _task_id("down_proj", "decode", 1, HIDDEN_SIZE, INTERMEDIATE_SIZE, 1, ctx_len), "down_proj", "decode", 1, HIDDEN_SIZE, INTERMEDIATE_SIZE, 1, ctx_len),
            ]
        )

    return tasks
