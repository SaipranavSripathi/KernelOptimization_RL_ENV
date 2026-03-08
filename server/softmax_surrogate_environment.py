from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
import json
import math
import os
import functools
import queue
import random
import re
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

try:
    from triton.compiler.errors import CompilationError as TritonCompilationError
except Exception:  # pragma: no cover - Triton is expected in benchmark environments.
    TritonCompilationError = None


DEFAULT_MEASUREMENT_PATH = "data/autotune_measurements.csv"
DEFAULT_BUDGET = 6
INITIAL_DATASET_SIZE = 2
DUPLICATE_PENALTY = -1e-4
FAMILY_INDEX = {
    "softmax": 0,
    "layernorm": 1,
    "grouped_gemm": 2,
}
KERNEL_TEMPLATE_PATHS = {
    "softmax": ROOT / "kernels" / "base_softmax.py",
    "layernorm": ROOT / "kernels" / "base_layernorm.py",
    "grouped_gemm": ROOT / "kernels" / "base_grouped_gemm.py",
}
PROMPT_MUTATIONS = (
    "Prefer a small localized edit over a full rewrite.",
    "Bias toward reducing register pressure and avoiding spills.",
    "Bias toward more coalesced loads and fewer redundant reads.",
    "Keep numerics stable and do not change algorithm semantics.",
    "Favor changes that could reduce first-call compile burden.",
)
SEGMENT_BUFFER_MAX = 200
MIN_WINS_BEFORE_DPO = int(os.environ.get("KERNEL_MIN_WINS_BEFORE_DPO", "12"))
MIN_UNIQUE_RATIO = float(os.environ.get("KERNEL_MIN_UNIQUE_RATIO", "0.65"))
MAX_TRITON_COMPILE_RETRIES = 3
TRAIN_INTERVAL_EPISODES = int(os.environ.get("KERNEL_TRAIN_INTERVAL_EPISODES", "10"))
VALIDATION_TASK_COUNT = int(os.environ.get("KERNEL_VALIDATION_TASK_COUNT", "3"))
GOOD_ADAPTER_KEEP = int(os.environ.get("KERNEL_GOOD_ADAPTER_KEEP", "3"))
MIN_STUDENT_ATTEMPTS_BEFORE_TEACHER_DISABLE = int(os.environ.get("KERNEL_MIN_STUDENT_ATTEMPTS", "8"))
MIN_STUDENT_VALID_RATE = float(os.environ.get("KERNEL_MIN_STUDENT_VALID_RATE", "0.8"))
MIN_STUDENT_WIN_RATE = float(os.environ.get("KERNEL_MIN_STUDENT_WIN_RATE", "0.7"))
TRITON_CONSTEXPR_NAMES = ("BLOCK_SIZE", "BLOCK_M", "BLOCK_N", "BLOCK_K", "GROUP_SIZE_M")
BLOCK_SIZES = (256, 512, 1024, 2048, 4096, 8192)
NUM_WARPS = (1, 2, 4, 8)
NUM_STAGES = (1, 2, 3, 4)


@functools.lru_cache(maxsize=1)
def _live_benchmark_helpers() -> Dict[str, Callable[..., Any]]:
    try:
        from scripts.collect_measurements import benchmark_single_config as benchmark_softmax_config
        from scripts.collect_multifamily_measurements import (
            benchmark_grouped_gemm_config,
            benchmark_layernorm_config,
        )
    except Exception as exc:  # pragma: no cover - live benchmarking is optional in Spaces.
        raise RuntimeError(
            "Live benchmarking requires the Triton measurement stack and GPU-capable dependencies."
        ) from exc

    return {
        "softmax": benchmark_softmax_config,
        "layernorm": benchmark_layernorm_config,
        "grouped_gemm": benchmark_grouped_gemm_config,
    }


@dataclass(frozen=True)
class Measurement:
    family_group: str
    family: str
    task_id: str
    m: int
    n: int
    config_id: int
    block_size: int
    num_warps: int
    num_stages: int
    median_ms: float
    effective_gbps: float
    score: float
    validation_error: float


def _normalize_discrete(values: Sequence[int], value: int) -> float:
    idx = list(values).index(int(value))
    if len(values) == 1:
        return 0.0
    return 2.0 * (idx / (len(values) - 1)) - 1.0


def _de_norm(value: float, values: Sequence[int]) -> int:
    idx = int(round((value + 1.0) * 0.5 * (len(values) - 1)))
    idx = max(0, min(len(values) - 1, idx))
    return int(values[idx])


class SoftmaxSurrogateEnvironment:
    """
    Multi-family autotuning environment with three user-facing modes:

    - mode="surrogate": measured-table oracle + surrogate scoring
    - mode="default": live benchmarking + fixed prompting, no self-improvement
    - mode="self_improving": live benchmarking + prompt evolution + buffers + trainer hooks

    Legacy alias:
    - mode="generative" -> "self_improving"
    """

    def __init__(
        self,
        measurement_path: str = DEFAULT_MEASUREMENT_PATH,
        budget: int = DEFAULT_BUDGET,
        seed: int = 0,
        initial_samples: int = INITIAL_DATASET_SIZE,
        train_task_ids: Optional[Sequence[str]] = None,
        mode: str = "surrogate",
        live_bench: Optional[bool] = None,
        benchmark_workers: int = 1,
        model_backend: Optional[str] = None,
        proposer_backend: Optional[str] = None,
        student_backend: Optional[str] = None,
        proposal_batch_size: Optional[int] = None,
    ) -> None:
        self.measurement_path = Path(measurement_path)
        self.budget = int(budget)
        self.seed = int(seed)
        self.initial_samples = max(1, int(initial_samples))
        self.train_task_ids = set(train_task_ids or [])

        if mode not in {"surrogate", "default", "self_improving", "generative"}:
            raise ValueError(f"Unsupported mode: {mode}")
        normalized_mode = "self_improving" if mode == "generative" else mode
        self._mode = normalized_mode
        self._live_bench = (normalized_mode in {"default", "self_improving"}) if live_bench is None else bool(live_bench)
        self._prompt_evolution_enabled = normalized_mode == "self_improving"
        self._training_enabled = normalized_mode == "self_improving"
        self._benchmark_workers = max(1, int(benchmark_workers))
        backend_default = (model_backend or os.environ.get("KERNEL_LLM_BACKEND", "openrouter")).strip().lower()
        self._proposer_backend_name = (
            proposer_backend
            or os.environ.get("KERNEL_PROPOSER_BACKEND")
            or ("openrouter" if normalized_mode == "self_improving" else backend_default)
        ).strip().lower()
        self._student_backend_name = (
            student_backend
            or os.environ.get("KERNEL_STUDENT_BACKEND")
            or ("local" if normalized_mode == "self_improving" else backend_default)
        ).strip().lower()
        default_batch_size = 8 if normalized_mode == "self_improving" else 1
        env_batch = os.environ.get("KERNEL_PROPOSAL_BATCH_SIZE")
        batch_value = proposal_batch_size if proposal_batch_size is not None else (int(env_batch) if env_batch else default_batch_size)
        self._proposal_batch_size = max(1, int(batch_value))
        self._proposal_max_tokens = int(os.environ.get("KERNEL_PROPOSAL_MAX_TOKENS", "1536"))
        self._teacher_max_tokens = max(
            self._proposal_max_tokens,
            int(os.environ.get("KERNEL_TEACHER_MAX_TOKENS", "4096")),
        )
        preview_default = min(self._proposal_max_tokens, 256)
        self._chat_student_preview_max_tokens = max(
            64,
            min(
                self._proposal_max_tokens,
                int(os.environ.get("KERNEL_CHAT_STUDENT_PREVIEW_MAX_TOKENS", str(preview_default))),
            ),
        )
        teacher_reasoning_env = os.environ.get("KERNEL_TEACHER_REASONING_MAX_TOKENS")
        self._teacher_reasoning_max_tokens = (
            max(0, int(teacher_reasoning_env))
            if teacher_reasoning_env is not None
            else 128
        )
        self._teacher_reasoning_effort = os.environ.get("KERNEL_TEACHER_REASONING_EFFORT", "low").strip().lower()
        self._eval_cache: Dict[str, Measurement] = {}
        self._executor = (
            ThreadPoolExecutor(max_workers=self._benchmark_workers)
            if self._live_bench
            else None
        )

        self._measurements = self._load_measurements()
        self._task_ids = sorted(self._measurements.keys())
        if not self._task_ids:
            raise RuntimeError("No measurement data found. Run the measurement collectors first.")

        self._rng = random.Random(self.seed)
        self._episode_counter = 0

        self._task_id: Optional[str] = None
        self._family: Optional[str] = None
        self._episode_id: Optional[str] = None
        self._task_rows: List[Measurement] = []
        self._prior_rows: List[Measurement] = []
        self._config_by_id: Dict[int, Measurement] = {}
        self._observed_ids: List[int] = []
        self._observed_id_set = set()
        self._observed_rows: List[Measurement] = []
        self._observed_latencies: List[float] = []
        self._steps_taken = 0
        self._steps_remaining = 0
        self._best_latency_ms = float("inf")
        self._best_config_id: Optional[int] = None
        self._validation_mse = 0.0

        self._surrogate_version = 0
        self._surrogate_fitted_version = -1
        self._surrogate_x: Optional[np.ndarray] = None
        self._surrogate_y: Optional[np.ndarray] = None
        self._surrogate_alpha: Optional[np.ndarray] = None
        self._surrogate_k: Optional[np.ndarray] = None
        self._surrogate_length_scale = 0.5
        self.current_kernel_source = ""
        self._best_kernel_source = ""
        self.prompt_population: List[Dict[str, Any]] = []
        self._prompt_history: List[Dict[str, Any]] = []
        self._active_variant_id: Optional[str] = None
        self.llm_client: Optional[Any] = None
        self.student_client: Optional[Any] = None
        self.segment_buffer: Dict[str, List[Dict[str, Any]]] = {}
        self.segment_best: Dict[str, float] = {}
        self.segment_stats: Dict[str, Dict[str, int]] = {}
        self._adapter_version = "base"
        self._adapter_path: Optional[str] = None
        self._last_train_episode = 0
        self._training_status = "idle"
        self._last_training_segment: Optional[str] = None
        self._last_training_result: Optional[Dict[str, Any]] = None
        self._good_adapter_history: List[Dict[str, Any]] = []
        self._runtime_event_sink: Optional[Callable[[Dict[str, Any]], None]] = None
        self._self_improve_root = ROOT / "artifacts" / "self_improvement"
        self._segment_buffer_dir = self._self_improve_root / "buffers"
        self._train_job_dir = self._self_improve_root / "jobs"
        self._adapter_dir = self._self_improve_root / "adapters"
        self._state_path = self._self_improve_root / "state.json"
        for path in (self._self_improve_root, self._segment_buffer_dir, self._train_job_dir, self._adapter_dir):
            path.mkdir(parents=True, exist_ok=True)
        self._load_persisted_state()
        bootstrap_adapter = os.environ.get("KERNEL_BOOTSTRAP_ADAPTER", "").strip()
        if bootstrap_adapter and self._student_backend_name == "local":
            try:
                client = self._get_student_backend()
                if client.load_adapter(bootstrap_adapter):
                    self._adapter_path = bootstrap_adapter
                    self._adapter_version = f"bootstrap:{Path(bootstrap_adapter).name}"
                    self._good_adapter_history.append(
                        {
                            "version": self._adapter_version,
                            "path": self._adapter_path,
                            "segment_key": "bootstrap",
                            "timestamp": int(time.time()),
                        }
                    )
                    self._persist_global_state()
            except Exception:
                pass

    def reset(self, task: Optional[str] = None, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            self._rng = random.Random(int(seed))

        if task is None:
            task = self._rng.choice(self._task_ids)
        if task not in self._measurements:
            raise ValueError(f"Unknown task: {task}")

        rows = self._measurements[task]
        self._task_id = task
        self._family = rows[0].family
        self._task_rows = rows
        self._config_by_id = {row.config_id: row for row in rows}
        self._prior_rows = self._build_prior_rows(task) if self._mode == "surrogate" else []
        self._observed_ids = []
        self._observed_id_set = set()
        self._observed_rows = []
        self._observed_latencies = []
        self._steps_taken = 0
        self._steps_remaining = self.budget
        self._best_latency_ms = float("inf")
        self._best_config_id = None
        self._episode_counter += 1
        self._episode_id = f"{task}:{self.seed}:{self._episode_counter}"
        if self._mode in {"default", "self_improving"}:
            self.current_kernel_source = self._load_template_source(self._family)
            self._best_kernel_source = self.current_kernel_source
            self.prompt_population = self._default_prompt_population(self._family) if self._llm_edit_enabled() else []
            self._prompt_history = []
            self._active_variant_id = None

        sample_count = min(self.initial_samples, len(rows))
        for config_id in self._rng.sample(list(self._config_by_id.keys()), k=sample_count):
            self._observe_config(config_id)

        self._invalidate_surrogate()
        self._validation_mse = self._compute_validation_mse()

        return self._format_step_output(
            observation=self._observation_payload(kind="reset"),
            reward=0.0,
            done=False,
            info=self.diagnostics(),
        )

    def step(self, action: Any) -> Dict[str, Any]:
        if self._task_id is None:
            raise RuntimeError("Call reset() before step().")
        if self._steps_remaining <= 0:
            return self._format_step_output(
                observation=self._observation_payload(kind="done"),
                reward=0.0,
                done=True,
                info=self.diagnostics(),
            )

        config_id, source = self._extract_action(action)
        variant_record: Optional[Dict[str, Any]] = None
        precomputed_observed: Optional[Measurement] = None
        rejected_source: Optional[str] = None
        incumbent_source = self.current_kernel_source
        if self._mode == "self_improving" and source is None and self._llm_edit_enabled():
            variants = self._prepare_variants(self._proposal_batch_size, self._prompt_history)
            self._active_variant_id = ",".join(str(item["id"]) for item in variants)
            segment_key = self._current_segment_key()
            print(
                f"[self_improving] step={self._steps_taken + 1} task={self._task_id} "
                f"segment={segment_key} phase=student_propose batch={len(variants)}",
                flush=True,
            )
            student_candidates = self._generate_candidates_for_variants(variants, backend="student")
            valid_student, invalid_student = self._benchmark_candidate_batch(config_id, student_candidates)
            teacher_needed = self._teacher_needed_for_segment(segment_key, valid_student=bool(valid_student), invalid_student=bool(invalid_student))
            valid_teacher: List[Dict[str, Any]] = []
            if teacher_needed:
                print(
                    f"[self_improving] step={self._steps_taken + 1} task={self._task_id} "
                    f"segment={segment_key} phase=teacher_fallback",
                    flush=True,
                )
                teacher_candidates = self._generate_candidates_for_variants(variants, backend="teacher")
                valid_teacher, _ = self._benchmark_candidate_batch(config_id, teacher_candidates)

            if valid_student:
                self._increment_segment_stat(segment_key, "student_valid")
            if invalid_student:
                self._increment_segment_stat(segment_key, "student_invalid")
            if teacher_needed:
                self._increment_segment_stat(segment_key, "teacher_queried")

            if valid_student and (not teacher_needed or not valid_teacher):
                chosen = valid_student[0]
                variant_record = chosen["variant"]
                variant_record["_chosen_role"] = "student"
                source = chosen["source"]
                precomputed_observed = chosen["measurement"]
                self._increment_segment_stat(segment_key, "student_win")
                print(
                    f"[self_improving] step={self._steps_taken + 1} task={self._task_id} "
                    f"winner=student latency_ms={precomputed_observed.median_ms:.6f}",
                    flush=True,
                )
            elif valid_teacher and (not valid_student or float(valid_teacher[0]["measurement"].median_ms) < float(valid_student[0]["measurement"].median_ms)):
                chosen = valid_teacher[0]
                variant_record = chosen["variant"]
                variant_record["_chosen_role"] = "teacher"
                source = chosen["source"]
                precomputed_observed = chosen["measurement"]
                rejected_source = valid_student[0]["source"] if valid_student else self._invalid_candidate_marker("student", invalid_student)
                variant_record["_rejected_role"] = "student" if valid_student else "student_invalid"
                self._increment_segment_stat(segment_key, "teacher_win")
                print(
                    f"[self_improving] step={self._steps_taken + 1} task={self._task_id} "
                    f"winner=teacher latency_ms={precomputed_observed.median_ms:.6f}",
                    flush=True,
                )
            elif valid_student:
                chosen = valid_student[0]
                variant_record = chosen["variant"]
                variant_record["_chosen_role"] = "student"
                source = chosen["source"]
                precomputed_observed = chosen["measurement"]
                self._increment_segment_stat(segment_key, "student_win")
                print(
                    f"[self_improving] step={self._steps_taken + 1} task={self._task_id} "
                    f"winner=student latency_ms={precomputed_observed.median_ms:.6f}",
                    flush=True,
                )
            elif valid_teacher:
                chosen = valid_teacher[0]
                variant_record = chosen["variant"]
                variant_record["_chosen_role"] = "teacher"
                source = chosen["source"]
                precomputed_observed = chosen["measurement"]
                rejected_source = self._invalid_candidate_marker("student", invalid_student)
                variant_record["_rejected_role"] = "student_invalid"
                self._increment_segment_stat(segment_key, "student_invalid_teacher_fallback")
                print(
                    f"[self_improving] step={self._steps_taken + 1} task={self._task_id} "
                    f"winner=teacher_after_student_invalid latency_ms={precomputed_observed.median_ms:.6f}",
                    flush=True,
                )
            else:
                source = incumbent_source
                self._increment_segment_stat(segment_key, "both_invalid_fallback")
                print(
                    f"[self_improving] step={self._steps_taken + 1} task={self._task_id} "
                    f"winner=incumbent both_invalid=true",
                    flush=True,
                )
        elif self._mode in {"default", "self_improving"} and source is None and self._llm_edit_enabled():
            proposals = self._propose_batch(self._proposal_batch_size, self._prompt_history)
            self._active_variant_id = ",".join(str(item["variant"]["id"]) for item in proposals)
            prev_best = self._best_latency_ms
            ranked = self._benchmark_proposal_batch(config_id, proposals)
            best_item = ranked[0]
            worst_item = ranked[-1]
            variant_record = best_item["variant"]
            source = best_item["source"]
            precomputed_observed = best_item["measurement"]
            rejected_source = worst_item["source"]
            if self._training_enabled:
                student_source, student_status = self._shadow_student_attempt(str(variant_record.get("_prompt_text", "")))
                variant_record["_student_status"] = student_status
                if student_source:
                    variant_record["_student_source"] = student_source
                    rejected_source = student_source
                else:
                    rejected_source = f"# STUDENT_FAILURE\n# {student_status}\n"
            for item in ranked:
                self._record_prompt_result(item["variant"], item["measurement"], item["source"], prev_best)
        elif self._mode in {"default", "self_improving"} and source is None:
            source = self.current_kernel_source
        prev_best = self._best_latency_ms
        duplicate = config_id in self._observed_id_set
        row = self._row_for_id(config_id)

        if not duplicate:
            observed = self._observe_config(config_id, source=source, precomputed=precomputed_observed)
            row = observed
            if self._mode == "surrogate":
                self._surrogate_version += 1
            if self._mode in {"default", "self_improving"}:
                if self._training_enabled:
                    if rejected_source is not None:
                        self._maybe_store_segment_win(
                            observed=observed,
                            source=source,
                            variant=variant_record,
                            previous_source=rejected_source,
                            force_store=True,
                        )
                if source is not None and observed.median_ms <= prev_best:
                    self.current_kernel_source = source
                    self._best_kernel_source = source

        self._steps_taken += 1
        self._steps_remaining -= 1
        self._validation_mse = self._compute_validation_mse()
        if self._prompt_evolution_enabled and self._llm_edit_enabled() and self._steps_taken > 0 and self._steps_taken % 3 == 0:
            self._evolve_prompt_population()

        reward = DUPLICATE_PENALTY if duplicate else max(0.0, math.log(prev_best) - math.log(self._best_latency_ms))
        observation = self._observation_payload(
            kind="step",
            last_trial={
                "config_id": config_id,
                "config": self.config_info(config_id),
                "latency_ms": row.median_ms,
                "score": row.score,
                "duplicate": duplicate,
            },
        )
        return self._format_step_output(
            observation=observation,
            reward=reward,
            done=self._steps_remaining <= 0,
            info=self.diagnostics(),
        )

    def chat_optimize_events(
        self,
        task: Optional[str] = None,
        config_id: Optional[int] = None,
        source: Optional[str] = None,
        seed: Optional[int] = None,
        instruction: Optional[str] = None,
    ):
        runtime_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        previous_runtime_sink = self._runtime_event_sink
        self._runtime_event_sink = runtime_queue.put
        try:
            requested_task = task
            yield {
                "type": "request_received",
                "requested_task": requested_task,
                "requested_config_id": config_id,
                "mode": self._mode,
            }
            reset_out = self.reset(task=task, seed=seed)
            if source:
                self.current_kernel_source = source
                self._best_kernel_source = source
            if config_id is None:
                config_id = self._best_config_id if self._best_config_id is not None else 0

            row = self._row_for_id(int(config_id))
            segment_key = self._current_segment_key()
            variant = self._select_prompt_variant() if self._llm_edit_enabled() else {"id": "chat", "instruction": instruction or "Optimize conservatively."}
            if instruction:
                variant = dict(variant)
                variant["instruction"] = instruction
            variant["_prompt_text"] = self._build_full_prompt(variant, self._prompt_history)

            yield {
                "type": "reset_complete",
                "task_id": self._task_id,
                "family": self._family,
                "config_id": config_id,
                "mode": self._mode,
                "segment_key": segment_key,
                "best_so_far_ms": self._best_latency_ms,
                "oracle_best_ms": self.oracle_best()["median_ms"],
                "reset_info": reset_out.get("info", {}),
            }

            yield {
                "type": "proposal_started",
                "task_id": self._task_id,
                "segment_key": segment_key,
                "backends": ["student", "teacher"],
            }
            event_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
            student_candidate: Optional[Dict[str, Any]] = None
            teacher_candidate: Optional[Dict[str, Any]] = None
            student_preview_complete = False
            student_done = False
            teacher_done = False

            threading.Thread(
                target=self._emit_student_chat_events,
                args=(variant, event_queue),
                daemon=True,
            ).start()
            threading.Thread(
                target=self._emit_teacher_chat_events,
                args=(variant, event_queue),
                daemon=True,
            ).start()

            while not (student_done and teacher_done):
                item = event_queue.get()
                kind = str(item.get("kind", ""))
                if kind == "student_preview":
                    candidate = item["candidate"]
                    student_preview_complete = bool(item.get("is_complete"))
                    yield {
                        "type": "student_preview",
                        "task_id": self._task_id,
                        "segment_key": segment_key,
                        "variant_id": variant["id"],
                        "status": candidate["status"],
                        "source": candidate["source"],
                        "is_complete": student_preview_complete,
                        "error": candidate.get("error"),
                    }
                elif kind == "student_candidate":
                    student_candidate = item["candidate"]
                    yield {
                        "type": "student_ready",
                        "task_id": self._task_id,
                        "segment_key": segment_key,
                        "variant_id": variant["id"],
                        "status": student_candidate["status"],
                        "source": student_candidate["source"],
                        "error": student_candidate.get("error"),
                    }
                elif kind == "teacher_thinking_delta":
                    yield {
                        "type": "teacher_thinking_delta",
                        "task_id": self._task_id,
                        "segment_key": segment_key,
                        "text": item.get("text", ""),
                    }
                elif kind == "teacher_content_delta":
                    yield {
                        "type": "teacher_content_delta",
                        "task_id": self._task_id,
                        "segment_key": segment_key,
                        "text": item.get("text", ""),
                    }
                elif kind == "teacher_candidate":
                    teacher_candidate = item["candidate"]
                    yield {
                        "type": "teacher_ready",
                        "task_id": self._task_id,
                        "segment_key": segment_key,
                        "variant_id": variant["id"],
                        "status": teacher_candidate["status"],
                        "source": teacher_candidate["source"],
                        "error": teacher_candidate.get("error"),
                    }
                elif kind == "student_done":
                    student_done = True
                elif kind == "teacher_done":
                    teacher_done = True
                for runtime_event in self._drain_runtime_events(runtime_queue):
                    yield runtime_event

            if student_candidate is None:
                student_candidate = {
                    "backend": "student",
                    "variant": variant,
                    "prompt_text": str(variant.get("_prompt_text", "")),
                    "source": None,
                    "status": "proposal_failed:MissingStudentCandidate",
                    "error": "student candidate was not produced",
                }
            if teacher_candidate is None:
                teacher_candidate = {
                    "backend": "teacher",
                    "variant": variant,
                    "prompt_text": str(variant.get("_prompt_text", "")),
                    "source": None,
                    "status": "proposal_failed:MissingTeacherCandidate",
                    "error": "teacher candidate was not produced",
                }
            yield {
                "type": "kernels_ready",
                "task_id": self._task_id,
                "segment_key": segment_key,
                "variant_id": variant["id"],
                "student_preview_complete": student_preview_complete,
                "student_status": student_candidate["status"],
                "student_source": student_candidate["source"],
                "teacher_status": teacher_candidate["status"],
                "teacher_source": teacher_candidate["source"],
            }
            for runtime_event in self._drain_runtime_events(runtime_queue):
                yield runtime_event

            yield {
                "type": "benchmark_started",
                "task_id": self._task_id,
                "config_id": config_id,
            }
            valid, invalid = self._benchmark_candidate_batch(int(config_id), [student_candidate, teacher_candidate])
            valid_by_backend = {item["backend"]: item for item in valid}
            invalid_by_backend = {item["backend"]: item for item in invalid}

            baseline_source = self.current_kernel_source
            baseline_measurement = self._benchmark_or_lookup(int(config_id), source=baseline_source)
            student_valid = "student" in valid_by_backend
            teacher_valid = "teacher" in valid_by_backend
            student_ms = float(valid_by_backend["student"]["measurement"].median_ms) if student_valid else None
            teacher_ms = float(valid_by_backend["teacher"]["measurement"].median_ms) if teacher_valid else None
            baseline_ms = float(baseline_measurement.median_ms)
            result_class = self._classify_chat_result(
                teacher_valid=teacher_valid,
                student_valid=student_valid,
                teacher_ms=teacher_ms,
                student_ms=student_ms,
            )

            teacher_win = bool(teacher_valid and (not student_valid or float(teacher_ms) < float(student_ms)))
            student_win = bool(student_valid and (not teacher_valid or float(student_ms) <= float(teacher_ms)))
            already_best = baseline_ms <= min([value for value in [student_ms, teacher_ms] if value is not None] or [baseline_ms])

            if student_valid:
                self._increment_segment_stat(segment_key, "student_valid")
            else:
                self._increment_segment_stat(segment_key, "student_invalid")
            self._increment_segment_stat(segment_key, "teacher_queried")
            if teacher_win:
                self._increment_segment_stat(segment_key, "teacher_win")

                training_thread = threading.Thread(
                    target=self._maybe_store_segment_win,
                    kwargs={
                        "observed": valid_by_backend["teacher"]["measurement"],
                        "source": teacher_candidate.get("source"),
                        "variant": {**variant, "_chosen_role": "teacher", "_rejected_role": "student" if student_valid else "student_invalid"},
                        "previous_source": student_candidate.get("source") if student_valid else self._invalid_candidate_marker("student", [invalid_by_backend.get("student", {})]),
                        "force_store": True,
                    },
                    daemon=True,
                )
                training_thread.start()
                while training_thread.is_alive():
                    for runtime_event in self._drain_runtime_events(runtime_queue):
                        yield runtime_event
                    time.sleep(0.05)
                training_thread.join()
                for runtime_event in self._drain_runtime_events(runtime_queue):
                    yield runtime_event
                self.current_kernel_source = teacher_candidate.get("source") or self.current_kernel_source
            elif student_win:
                self._increment_segment_stat(segment_key, "student_win")
                if student_candidate.get("source"):
                    self.current_kernel_source = student_candidate["source"]

            yield {
                "type": "final_result",
                "task_id": self._task_id,
                "family": self._family,
                "config_id": config_id,
                "teacher_valid": teacher_valid,
                "student_valid": student_valid,
                "teacher_win": teacher_win,
                "student_win": student_win,
                "result_class": result_class,
                "already_best": already_best,
                "teacher_ms": teacher_ms,
                "student_ms": student_ms,
                "baseline_ms": baseline_ms,
                "teacher_error": invalid_by_backend.get("teacher", {}).get("error"),
                "student_error": invalid_by_backend.get("student", {}).get("error"),
                "segment_stats": self.segment_stats.get(segment_key, {}),
                "training_status": self._training_status,
                "adapter_version": self._adapter_version,
            }
        finally:
            self._runtime_event_sink = previous_runtime_sink

    def state(self) -> Dict[str, Any]:
        if self._task_id is None:
            return {"status": "uninitialized"}
        return {
            "episode_id": self._episode_id,
            "step_count": self._steps_taken,
            "task_id": self._task_id,
            "family": self._family,
            "mode": self._mode,
            "tried_config_ids": list(self._observed_ids),
        }

    def diagnostics(self) -> Dict[str, Any]:
        if self._task_id is None:
            return {"status": "uninitialized"}
        oracle_best_ms = self.oracle_best()["median_ms"]
        live_regret = self._best_latency_ms / oracle_best_ms - 1.0
        return {
            "validation_mse": self._validation_mse,
            "best_so_far_ms": self._best_latency_ms,
            "oracle_best_ms": oracle_best_ms,
            "current_regret": live_regret,
            "live_regret": live_regret,
            "observed_count": len(self._observed_ids),
            "prior_count": len(self._prior_rows),
            "mode": self._mode,
            "live_bench": self._live_bench,
            "benchmark_workers": self._benchmark_workers,
            "proposer_backend": self._proposer_backend_name,
            "student_backend": self._student_backend_name,
            "proposal_batch_size": self._proposal_batch_size,
            "active_variant_id": self._active_variant_id,
            "llm_edit_enabled": self._llm_edit_enabled(),
            "prompt_evolution_enabled": self._prompt_evolution_enabled,
            "training_enabled": self._training_enabled,
            "segment_key": self._current_segment_key(),
            "segment_buffer_size": self._segment_buffer_size(self._current_segment_key()),
            "segment_best_relative_latency": self.segment_best.get(self._current_segment_key()),
            "segment_unique_ratio": self._segment_unique_ratio(self._current_segment_key()),
            "segment_preference_counts": self._segment_preference_counts(self._current_segment_key()),
            "segment_train_ready": self._should_run_dpo(self._current_segment_key()),
            "segment_count": len(self.segment_buffer),
            "segment_stats": self.segment_stats.get(self._current_segment_key(), {}),
            "adapter_version": self._adapter_version,
            "adapter_path": self._adapter_path,
            "training_status": self._training_status,
            "last_training_segment": self._last_training_segment,
            "last_training_result": self._last_training_result,
            "good_adapter_versions": [entry["version"] for entry in self._good_adapter_history],
            "adapter_backend": self._get_student_backend().backend_name if self.student_client is not None else None,
            "supports_local_adapters": self._get_student_backend().supports_local_adapters if self.student_client is not None else False,
        }

    def available_tasks(self) -> List[str]:
        return list(self._task_ids)

    def available_config_ids(self) -> List[int]:
        if self._task_id is None:
            raise RuntimeError("Call reset() before accessing config ids.")
        return sorted(self._config_by_id.keys())

    def available_configs(self) -> List[Dict[str, Any]]:
        return [self.config_info(config_id) for config_id in self.available_config_ids()]

    def config_info(self, config_id: int) -> Dict[str, Any]:
        row = self._row_for_id(config_id)
        return {
            "config_id": int(config_id),
            "family": row.family,
            "task_id": row.task_id,
            "block_size": row.block_size,
            "num_warps": row.num_warps,
            "num_stages": row.num_stages,
        }

    def measured_latency_ms(self, config_id: int) -> float:
        return self._row_for_id(config_id).median_ms

    def oracle_best(self) -> Dict[str, Any]:
        if self._task_id is None:
            raise RuntimeError("Call reset() before querying oracle_best().")
        best = min(self._task_rows, key=lambda row: row.median_ms)
        return {
            "config_id": best.config_id,
            "family": best.family,
            "task_id": best.task_id,
            "block_size": best.block_size,
            "num_warps": best.num_warps,
            "num_stages": best.num_stages,
            "median_ms": best.median_ms,
            "score": best.score,
        }

    def predict_score(self, config_id: int) -> float:
        if self._mode != "surrogate":
            raise RuntimeError("predict_score is only available in surrogate mode.")
        return float(self._predict_with_uncertainty(config_id)[0])

    def acquisition_score(
        self,
        config_id: int,
        strategy: str = "ucb",
        beta: float = 1.0,
        xi: float = 0.0,
    ) -> float:
        if self._mode != "surrogate":
            raise RuntimeError("acquisition_score is only available in surrogate mode.")
        mean, sigma = self._predict_with_uncertainty(config_id)
        if strategy == "mean":
            return float(mean)
        if strategy == "ucb":
            return float(mean + float(beta) * sigma)
        if strategy == "ei":
            best_observed = max(row.score for row in self._observed_rows) if self._observed_rows else mean
            delta = mean - best_observed - float(xi)
            if sigma <= 0.0:
                return float(max(delta, 0.0))
            z = delta / sigma
            return float(max(delta * _normal_cdf(z) + sigma * _normal_pdf(z), 0.0))
        raise ValueError(f"Unknown acquisition strategy: {strategy}")

    def seen_config_ids(self) -> List[int]:
        return list(self._observed_ids)

    def _build_prior_rows(self, current_task: str) -> List[Measurement]:
        if not self.train_task_ids:
            return []
        prior_rows: List[Measurement] = []
        for task_id in sorted(self.train_task_ids):
            if task_id == current_task or task_id not in self._measurements:
                continue
            prior_rows.extend(self._measurements[task_id])
        return prior_rows

    def _predict_with_uncertainty(self, config_id: int) -> Tuple[float, float]:
        rows = self._prior_rows + self._observed_rows
        if not rows:
            raise RuntimeError("No surrogate data available.")
        self._fit_surrogate()
        if self._surrogate_x is None or self._surrogate_y is None:
            raise RuntimeError("Surrogate model unavailable.")
        if self._surrogate_x.shape[0] == 1:
            return float(self._surrogate_y[0]), 0.0

        cfg = _config_to_vector(self._row_for_id(config_id)).reshape(1, -1)
        if self._surrogate_k is None or self._surrogate_alpha is None:
            raise RuntimeError("Surrogate model unavailable.")
        k = _rbf_kernel(self._surrogate_x, cfg, self._surrogate_length_scale).reshape(-1)
        pred = float(k @ self._surrogate_alpha)
        solve = np.linalg.solve(self._surrogate_k, k)
        var = max(0.0, float(1.0 - k @ solve))
        return pred, float(math.sqrt(max(var, 1e-12)))

    def _fit_surrogate(self) -> None:
        if self._mode != "surrogate":
            raise RuntimeError("_fit_surrogate is only available in surrogate mode.")
        if self._surrogate_fitted_version == self._surrogate_version:
            return
        rows = self._prior_rows + self._observed_rows
        if not rows:
            self._surrogate_x = None
            self._surrogate_y = None
            self._surrogate_alpha = None
            self._surrogate_k = None
            self._surrogate_fitted_version = self._surrogate_version
            return

        self._surrogate_x = np.array([_config_to_vector(row) for row in rows], dtype=np.float32)
        self._surrogate_y = np.array([row.score for row in rows], dtype=np.float32)
        if self._surrogate_x.shape[0] == 1:
            self._surrogate_alpha = self._surrogate_y.copy()
            self._surrogate_k = None
            self._surrogate_fitted_version = self._surrogate_version
            return

        pairwise = _pairwise_sq_dists(self._surrogate_x)
        triu = pairwise[np.triu_indices(self._surrogate_x.shape[0], k=1)]
        med_dist = float(np.median(np.sqrt(triu))) if triu.size else 0.5
        self._surrogate_length_scale = max(0.15, med_dist)
        k = _rbf_kernel(self._surrogate_x, self._surrogate_x, self._surrogate_length_scale)
        k[np.diag_indices_from(k)] += 1e-3
        self._surrogate_k = k
        self._surrogate_alpha = np.linalg.solve(k, self._surrogate_y)
        self._surrogate_fitted_version = self._surrogate_version

    def _compute_validation_mse(self) -> float:
        if self._mode != "surrogate":
            return 0.0
        if not self._task_rows:
            return float("inf")
        preds = np.array(
            [self._predict_with_uncertainty(config_id)[0] for config_id in self.available_config_ids()],
            dtype=np.float32,
        )
        target = np.array([self._row_for_id(config_id).score for config_id in self.available_config_ids()], dtype=np.float32)
        return float(np.mean((preds - target) ** 2))

    def _observe_config(
        self,
        config_id: int,
        source: Optional[str] = None,
        precomputed: Optional[Measurement] = None,
    ) -> Measurement:
        observed = precomputed if precomputed is not None else self._benchmark_or_lookup(config_id, source=source)
        self._observed_ids.append(config_id)
        self._observed_id_set.add(config_id)
        self._observed_rows.append(observed)
        self._observed_latencies.append(observed.median_ms)
        if observed.median_ms < self._best_latency_ms:
            self._best_latency_ms = observed.median_ms
            self._best_config_id = config_id
        return observed

    def _benchmark_or_lookup(self, config_id: int, source: Optional[str] = None) -> Measurement:
        row = self._row_for_id(config_id)
        if not self._live_bench:
            return row
        if self._mode in {"default", "self_improving"} and source is None:
            source = self.current_kernel_source
        cache_key = self._make_cache_key(row, source=source)
        if cache_key in self._eval_cache:
            return self._eval_cache[cache_key]
        if self._executor is None:
            raise RuntimeError("Live benchmark executor is not available.")
        measured = self._submit_live_benchmark_with_retry(row, source)
        self._eval_cache[cache_key] = measured
        return measured

    def _make_cache_key(self, row: Measurement, source: Optional[str] = None) -> str:
        payload = json.dumps(
            {
                "task_id": row.task_id,
                "family": row.family,
                "config_id": row.config_id,
                "block_size": row.block_size,
                "num_warps": row.num_warps,
                "num_stages": row.num_stages,
                "source": source,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _submit_live_benchmark_with_retry(self, row: Measurement, source: Optional[str]) -> Measurement:
        if self._executor is None:
            raise RuntimeError("Live benchmark executor is not available.")
        max_attempts = MAX_TRITON_COMPILE_RETRIES + 1
        for attempt in range(1, max_attempts + 1):
            future = self._executor.submit(self._benchmark_live_config, row, source)
            try:
                return future.result()
            except Exception as exc:
                if not self._is_retryable_triton_compile_error(exc) or attempt >= max_attempts:
                    raise
                print(
                    "[warn] Triton compile failed for "
                    f"task={row.task_id} config_id={row.config_id}; "
                    f"retrying {attempt}/{MAX_TRITON_COMPILE_RETRIES}",
                    file=sys.stderr,
                    flush=True,
                )
        raise RuntimeError("Unreachable Triton compile retry state.")

    def _is_retryable_triton_compile_error(self, exc: BaseException) -> bool:
        current: Optional[BaseException] = exc
        while current is not None:
            if TritonCompilationError is not None and isinstance(current, TritonCompilationError):
                return True
            exc_type = type(current)
            if "triton" in exc_type.__module__.lower() and "compilationerror" in exc_type.__name__.lower():
                return True
            current = current.__cause__ or current.__context__
        return False

    def _benchmark_live_config(self, row: Measurement, source: Optional[str] = None) -> Measurement:
        if source:
            return self._benchmark_generated_source(row, source)
        helpers = _live_benchmark_helpers()
        kwargs = {
            "n": row.n,
            "block_size": row.block_size,
            "num_warps": row.num_warps,
            "num_stages": row.num_stages,
            "m": row.m,
        }
        benchmark_fn = helpers.get(row.family)
        if benchmark_fn is None:
            raise ValueError(f"Unsupported family: {row.family}")
        bench = benchmark_fn(**kwargs)
        return Measurement(
            family_group=getattr(bench, "family_group", row.family_group),
            family=getattr(bench, "family", row.family),
            task_id=getattr(bench, "task_id", row.task_id),
            m=bench.m,
            n=bench.n,
            config_id=row.config_id,
            block_size=bench.block_size,
            num_warps=bench.num_warps,
            num_stages=bench.num_stages,
            median_ms=bench.median_ms,
            effective_gbps=bench.effective_gbps,
            score=bench.score,
            validation_error=bench.validation_error,
        )

    def _observation_payload(
        self,
        kind: str,
        last_trial: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "type": kind,
            "task_id": self._task_id,
            "family": self._family,
            "M": self._task_rows[0].m if self._task_rows else None,
            "N": self._task_rows[0].n if self._task_rows else None,
            "dtype": "fp16",
            "mode": self._mode,
            "tried_config_ids": list(self._observed_ids),
            "tried_latencies_ms": list(self._observed_latencies),
            "best_so_far_ms": self._best_latency_ms,
            "steps_remaining": self._steps_remaining,
        }
        if last_trial is not None:
            payload["last_trial"] = last_trial
        return payload

    def _extract_action(self, action: Any) -> Tuple[int, Optional[str]]:
        source = None
        if isinstance(action, (str, bytes)):
            action = json.loads(action)
        if isinstance(action, dict):
            source = action.get("source")
            if "config_id" in action:
                return int(action["config_id"]), source
            if "x" in action:
                normalized = self._extract_legacy_action(action["x"])
                return self._map_legacy_action_to_config(normalized), source
        if isinstance(action, (int, np.integer)):
            return int(action), None
        raise TypeError("Action must be an int config_id or dict with config_id/x.")

    def _load_template_source(self, family: Optional[str]) -> str:
        if family is None or family not in KERNEL_TEMPLATE_PATHS:
            raise ValueError(f"No kernel template registered for family={family}")
        return KERNEL_TEMPLATE_PATHS[family].read_text(encoding="utf-8")

    def _llm_edit_enabled(self) -> bool:
        return self._mode in {"default", "self_improving"} and self._family in KERNEL_TEMPLATE_PATHS

    def _default_prompt_population(self, family: Optional[str]) -> List[Dict[str, Any]]:
        base = [
            "Make a minimal performance-oriented edit to the kernel while preserving exact semantics.",
            "Prioritize memory coalescing and avoiding redundant loads for this kernel family.",
            "Reduce register pressure and unnecessary temporaries while keeping the code simple.",
            "Favor better occupancy without changing the external Python API of the module.",
            "Look for opportunities to simplify control flow and tighten masks.",
            "Try a conservative tiling or indexing improvement that is easy to validate.",
            "Bias toward reducing first-call compile burden while keeping good steady-state runtime.",
            "Optimize for the current shape and config while preserving correctness and benchmark entrypoints.",
            "Prefer localized edits around the Triton kernel body instead of rewriting the whole file.",
            "Keep the output format identical and do not remove benchmark_generated_kernel.",
        ]
        return [
            {
                "id": f"{family or 'kernel'}-{idx}",
                "instruction": prompt,
                "uses": 0,
                "score": 0.0,
            }
            for idx, prompt in enumerate(base)
        ]

    def _build_backend(self, backend_name: str) -> Any:
        if backend_name == "openrouter":
            from server.openrouter_client import OpenRouterClient

            return OpenRouterClient()
        if backend_name == "local":
            from server.local_adapter_backend import LocalAdapterBackend

            return LocalAdapterBackend()
        raise ValueError(f"Unsupported backend: {backend_name}")

    def _llm_messages_from_prompt(self, prompt_text: str) -> List[Dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "You edit Triton kernel modules for GPU autotuning. "
                    "Return only valid Python source code for the full module. "
                    "Keep the module-level benchmark_generated_kernel entrypoint intact."
                ),
            },
            {"role": "user", "content": prompt_text},
        ]

    def _get_llm_client(self) -> Any:
        if self.llm_client is None:
            self.llm_client = self._build_backend(self._proposer_backend_name)
        return self.llm_client

    def _get_student_backend(self) -> Any:
        if self.student_client is None:
            self.student_client = self._build_backend(self._student_backend_name)
        return self.student_client

    def _shadow_student_attempt(self, prompt_text: str) -> Tuple[Optional[str], str]:
        try:
            client = self._get_student_backend()
        except Exception as exc:
            return None, f"student_backend_init_failed:{type(exc).__name__}"
        try:
            response = client.complete(
                self._llm_messages_from_prompt(prompt_text),
                max_tokens=self._proposal_max_tokens,
            )
            source = self._extract_python_source(response)
            if not source.strip():
                return None, "student_empty_response"
            return source, "student_proposed"
        except Exception as exc:
            return None, f"student_proposal_failed:{type(exc).__name__}"

    def _propose_batch(self, batch_size: int, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        proposals: List[Dict[str, Any]] = []
        for _ in range(max(1, batch_size)):
            variant = self._select_prompt_variant()
            variant["_prompt_text"] = self._build_full_prompt(variant, history)
            source = self._propose_kernel_edit_with_variant(variant, history)
            proposals.append({"variant": variant, "source": source})
        return proposals

    def _prepare_variants(self, batch_size: int, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        variants: List[Dict[str, Any]] = []
        for _ in range(max(1, batch_size)):
            variant = self._select_prompt_variant()
            variant["_prompt_text"] = self._build_full_prompt(variant, history)
            variants.append(variant)
        return variants

    def _generate_candidate(
        self,
        variant: Dict[str, Any],
        backend: str,
        *,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        if backend == "teacher":
            client = self._get_llm_client()
        elif backend == "student":
            client = self._get_student_backend()
        else:
            raise ValueError(f"Unsupported proposal backend role: {backend}")

        prompt_text = str(variant.get("_prompt_text", ""))
        try:
            response = client.complete(
                self._llm_messages_from_prompt(prompt_text),
                max_tokens=self._max_tokens_for_backend(backend) if max_tokens is None else int(max_tokens),
            )
            source = self._extract_python_source(response)
            status = "proposed" if source.strip() else "empty"
            source_value = source if source.strip() else None
            error = None
        except Exception as exc:
            status = f"proposal_failed:{type(exc).__name__}"
            source_value = None
            error = str(exc)
        return {
            "backend": backend,
            "variant": variant,
            "prompt_text": prompt_text,
            "source": source_value,
            "status": status,
            "error": error,
        }

    def _generate_candidates_for_variants(self, variants: List[Dict[str, Any]], backend: str) -> List[Dict[str, Any]]:
        return [self._generate_candidate(variant, backend) for variant in variants]

    def _max_tokens_for_backend(self, backend: str) -> int:
        if backend == "teacher":
            return self._teacher_max_tokens
        return self._proposal_max_tokens

    def _source_looks_complete(self, source: Optional[str]) -> bool:
        if not source or "def benchmark_generated_kernel" not in source:
            return False
        try:
            ast.parse(source)
        except SyntaxError:
            return False
        return True

    def _teacher_reasoning_config(self) -> Optional[Dict[str, Any]]:
        if self._teacher_reasoning_max_tokens > 0:
            return {"max_tokens": self._teacher_reasoning_max_tokens}
        if self._teacher_reasoning_effort:
            return {"effort": self._teacher_reasoning_effort}
        return None

    def _emit_student_chat_events(self, variant: Dict[str, Any], event_queue: "queue.Queue[Dict[str, Any]]") -> None:
        try:
            preview_tokens = min(self._proposal_max_tokens, self._chat_student_preview_max_tokens)
            preview_candidate = self._generate_candidate(variant, "student", max_tokens=preview_tokens)
            preview_complete = self._source_looks_complete(preview_candidate.get("source"))
            event_queue.put(
                {
                    "kind": "student_preview",
                    "candidate": preview_candidate,
                    "is_complete": preview_complete,
                }
            )
            final_candidate = preview_candidate
            if not preview_complete and preview_tokens < self._proposal_max_tokens:
                final_candidate = self._generate_candidate(variant, "student", max_tokens=self._proposal_max_tokens)
            event_queue.put({"kind": "student_candidate", "candidate": final_candidate})
        finally:
            event_queue.put({"kind": "student_done"})

    def _emit_teacher_chat_events(self, variant: Dict[str, Any], event_queue: "queue.Queue[Dict[str, Any]]") -> None:
        prompt_text = str(variant.get("_prompt_text", ""))
        try:
            client = self._get_llm_client()
            if not hasattr(client, "stream_complete"):
                event_queue.put({"kind": "teacher_candidate", "candidate": self._generate_candidate(variant, "teacher")})
                return

            content_parts: List[str] = []
            reasoning_config = self._teacher_reasoning_config()
            reasoning_extra = {"reasoning": reasoning_config} if reasoning_config else None
            for delta in client.stream_complete(
                self._llm_messages_from_prompt(prompt_text),
                max_tokens=self._teacher_max_tokens,
                extra_body=reasoning_extra,
            ):
                reasoning_text = str(delta.get("reasoning", ""))
                if reasoning_text:
                    event_queue.put({"kind": "teacher_thinking_delta", "text": reasoning_text})
                content_text = str(delta.get("content", ""))
                if content_text:
                    content_parts.append(content_text)
                    event_queue.put({"kind": "teacher_content_delta", "text": content_text})

            response = "".join(content_parts)
            source = self._extract_python_source(response)
            event_queue.put(
                {
                    "kind": "teacher_candidate",
                    "candidate": {
                        "backend": "teacher",
                        "variant": variant,
                        "prompt_text": prompt_text,
                        "source": source if source.strip() else None,
                        "status": "proposed" if source.strip() else "empty",
                        "error": None,
                    },
                }
            )
        except Exception as exc:
            event_queue.put(
                {
                    "kind": "teacher_candidate",
                    "candidate": {
                        "backend": "teacher",
                        "variant": variant,
                        "prompt_text": prompt_text,
                        "source": None,
                        "status": f"proposal_failed:{type(exc).__name__}",
                        "error": str(exc),
                    },
                }
            )
        finally:
            event_queue.put({"kind": "teacher_done"})

    def _classify_chat_result(
        self,
        *,
        teacher_valid: bool,
        student_valid: bool,
        teacher_ms: Optional[float],
        student_ms: Optional[float],
    ) -> str:
        if teacher_valid and student_valid:
            return "teacher_won" if float(teacher_ms) < float(student_ms) else "student_won"
        if teacher_valid:
            return "teacher_valid"
        if student_valid:
            return "student_valid"
        return "both_invalid"

    def _emit_observability_event(self, name: str, **fields: Any) -> None:
        sink = self._runtime_event_sink
        if sink is None:
            return
        payload = {
            "type": "observability",
            "name": name,
            "task_id": self._task_id,
            "segment_key": self._current_segment_key(),
            **fields,
        }
        sink(payload)

    def _drain_runtime_events(self, runtime_queue: "queue.Queue[Dict[str, Any]]"):
        while True:
            try:
                yield runtime_queue.get_nowait()
            except queue.Empty:
                break

    def _benchmark_candidate_batch(
        self,
        config_id: int,
        candidates: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        row = self._row_for_id(config_id)
        valid: List[Dict[str, Any]] = []
        invalid: List[Dict[str, Any]] = []
        pending: List[Tuple[Dict[str, Any], str]] = []

        for item in candidates:
            source = item.get("source")
            if not source:
                invalid.append(item)
                continue
            cache_key = self._make_cache_key(row, source=source)
            cached = self._eval_cache.get(cache_key)
            if cached is not None:
                valid.append({**item, "measurement": cached})
            else:
                pending.append((item, cache_key))

        if self._executor is None:
            raise RuntimeError("Live benchmark executor is not available.")

        futures: List[Tuple[Any, Dict[str, Any], str]] = []
        for item, cache_key in pending:
            futures.append((self._executor.submit(self._benchmark_live_config, row, item["source"]), item, cache_key))

        for future, item, cache_key in futures:
            try:
                measurement = future.result()
                self._eval_cache[cache_key] = measurement
                valid.append({**item, "measurement": measurement})
            except Exception as exc:
                invalid.append({**item, "status": f"benchmark_failed:{type(exc).__name__}", "error": str(exc)})

        valid.sort(key=lambda item: float(item["measurement"].median_ms))
        return valid, invalid

    def _invalid_candidate_marker(self, role: str, invalid: List[Dict[str, Any]]) -> str:
        statuses = [str(item.get("status", "invalid")) for item in invalid[:4]]
        payload = {"role": role, "statuses": statuses}
        return "# INVALID_CANDIDATE\n" + json.dumps(payload, sort_keys=True)

    def _increment_segment_stat(self, segment_key: str, field: str) -> None:
        stats = self.segment_stats.setdefault(segment_key, {})
        stats[field] = int(stats.get(field, 0)) + 1
        self._persist_global_state()

    def _teacher_needed_for_segment(self, segment_key: str, valid_student: bool, invalid_student: bool) -> bool:
        stats = self.segment_stats.get(segment_key, {})
        valid = int(stats.get("student_valid", 0))
        invalid = int(stats.get("student_invalid", 0))
        compared = int(stats.get("teacher_win", 0) + stats.get("student_win", 0))
        student_wins = int(stats.get("student_win", 0))
        attempts = valid + invalid
        if invalid_student:
            return True
        if attempts < MIN_STUDENT_ATTEMPTS_BEFORE_TEACHER_DISABLE:
            return True
        valid_rate = valid / max(1, attempts)
        if valid_rate < MIN_STUDENT_VALID_RATE:
            return True
        if compared < MIN_STUDENT_ATTEMPTS_BEFORE_TEACHER_DISABLE:
            return True
        student_win_rate = student_wins / max(1, compared)
        if student_win_rate < MIN_STUDENT_WIN_RATE:
            return True
        return not valid_student

    def _benchmark_proposal_batch(self, config_id: int, proposals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        row = self._row_for_id(config_id)
        ranked: List[Dict[str, Any]] = []
        pending: List[Tuple[Dict[str, Any], str]] = []
        for item in proposals:
            cache_key = self._make_cache_key(row, source=item["source"])
            cached = self._eval_cache.get(cache_key)
            if cached is not None:
                ranked.append({"variant": item["variant"], "source": item["source"], "measurement": cached})
            else:
                pending.append((item, cache_key))

        if self._executor is None:
            raise RuntimeError("Live benchmark executor is not available.")
        futures: List[Tuple[Any, Dict[str, Any], str]] = []
        for item, cache_key in pending:
            futures.append((self._executor.submit(self._benchmark_live_config, row, item["source"]), item, cache_key))
        for future, item, cache_key in futures:
            measurement = future.result()
            self._eval_cache[cache_key] = measurement
            ranked.append({"variant": item["variant"], "source": item["source"], "measurement": measurement})
        ranked.sort(key=lambda item: float(item["measurement"].median_ms))
        return ranked

    def _select_prompt_variant(self) -> Dict[str, Any]:
        if not self.prompt_population:
            raise RuntimeError("Prompt population is empty.")
        unused = [variant for variant in self.prompt_population if int(variant["uses"]) == 0]
        if unused:
            return unused[0]
        return max(
            self.prompt_population,
            key=lambda variant: float(variant["score"]) + 0.1 / max(1, int(variant["uses"])),
        )

    def _recent_history(self, limit: int = 6) -> List[Dict[str, Any]]:
        return self._prompt_history[-limit:]

    def _propose_kernel_edit_with_variant(
        self,
        variant: Dict[str, Any],
        history: List[Dict[str, Any]],
    ) -> str:
        client = self._get_llm_client()
        current_config = self.config_info(self._best_config_id) if self._best_config_id is not None else None
        prompt_text = json.dumps(
            {
                "family": self._family,
                "task_id": self._task_id,
                "config": current_config,
                "variant_id": variant["id"],
                "variant_instruction": variant["instruction"],
                "recent_history": history,
                "current_kernel_source": self.current_kernel_source,
            },
            indent=2,
            sort_keys=True,
        )
        messages = self._llm_messages_from_prompt(prompt_text)
        response = client.complete(messages, max_tokens=self._teacher_max_tokens)
        return self._extract_python_source(response)

    def _extract_python_source(self, response: str) -> str:
        text = response.strip()
        if "```" not in text:
            return self._repair_generated_source(text)
        parts = text.split("```")
        for chunk in parts:
            chunk = chunk.strip()
            if not chunk:
                continue
            if chunk.startswith("python"):
                return self._repair_generated_source(chunk[len("python") :].lstrip())
            if "def benchmark_generated_kernel" in chunk or "@triton.jit" in chunk:
                return self._repair_generated_source(chunk)
        return self._repair_generated_source(text.replace("```", "").strip())

    def _repair_generated_source(self, source: str) -> str:
        repaired_lines: List[str] = []
        for line in source.splitlines():
            updated = line
            if line.lstrip().startswith("def "):
                for name in TRITON_CONSTEXPR_NAMES:
                    updated = re.sub(
                        rf"\b{name}\b(?!\s*:)\s*([,)])",
                        rf"{name}: tl.constexpr\1",
                        updated,
                    )
            repaired_lines.append(updated)
        return "\n".join(repaired_lines)

    def _record_prompt_result(
        self,
        variant: Optional[Dict[str, Any]],
        observed: Measurement,
        source: Optional[str],
        prev_best: float,
    ) -> None:
        if variant is None or not self._llm_edit_enabled():
            return
        variant["uses"] = int(variant["uses"]) + 1
        reward = max(0.0, math.log(prev_best) - math.log(observed.median_ms)) if prev_best < float("inf") else 0.0
        variant["score"] = (float(variant["score"]) * (int(variant["uses"]) - 1) + reward) / max(1, int(variant["uses"]))
        self._prompt_history.append(
            {
                "step": self._steps_taken,
                "variant_id": variant["id"],
                "latency_ms": observed.median_ms,
                "score": observed.score,
                "reward": reward,
                "config_id": observed.config_id,
                "source_hash": hashlib.sha256((source or self.current_kernel_source).encode("utf-8")).hexdigest()[:12],
            }
        )

    def _segment_key_for_measurement(self, measurement: Measurement) -> str:
        return f"{measurement.family}_M{measurement.m}_N{self._n_bucket(measurement.n)}_Dfp16"

    def _current_segment_key(self) -> str:
        if not self._task_rows:
            return "uninitialized"
        return self._segment_key_for_measurement(self._task_rows[0])

    def _n_bucket(self, n: int) -> str:
        if n <= 1024:
            return "small"
        if n <= 4096:
            return "medium"
        return "large"

    def _segment_buffer_size(self, segment_key: str) -> int:
        return len(self.segment_buffer.get(segment_key, []))

    def _segment_unique_ratio(self, segment_key: str) -> float:
        buf = self.segment_buffer.get(segment_key, [])
        if not buf:
            return 0.0
        unique = len({str(ex["chosen"])[:300] for ex in buf})
        return float(unique / len(buf))

    def _segment_preference_counts(self, segment_key: str) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for example in self.segment_buffer.get(segment_key, []):
            key = str(example.get("preference_kind", "unknown"))
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _should_run_dpo(self, segment_key: str) -> bool:
        if segment_key == "uninitialized":
            return False
        buf = self.segment_buffer.get(segment_key, [])
        if len(buf) < MIN_WINS_BEFORE_DPO:
            return False
        if len({int(ex["n"]) for ex in buf}) < 2:
            return False
        if self._segment_unique_ratio(segment_key) < MIN_UNIQUE_RATIO:
            return False
        return True

    def _build_full_prompt(self, variant: Optional[Dict[str, Any]], history: List[Dict[str, Any]]) -> str:
        config = self.config_info(self._best_config_id) if self._best_config_id is not None else None
        history_lines = []
        for item in history[-6:]:
            history_lines.append(
                "- "
                + f"step={item.get('step')} "
                + f"variant={item.get('variant_id')} "
                + f"latency_ms={item.get('latency_ms')} "
                + f"reward={item.get('reward')} "
                + f"config_id={item.get('config_id')}"
            )
        if not history_lines:
            history_lines.append("- no prior attempts yet")

        config_text = "none"
        if config is not None:
            config_text = (
                f"config_id={config['config_id']}, "
                f"block_size={config['block_size']}, "
                f"num_warps={config['num_warps']}, "
                f"num_stages={config['num_stages']}"
            )

        instruction = "keep the kernel correct and make a conservative performance-oriented edit"
        if variant is not None:
            instruction = str(variant.get("instruction", instruction))

        return "\n".join(
            [
                "You are editing a Triton kernel module for GPU autotuning.",
                "Return a complete Python module only. Do not include prose.",
                "The module must preserve the benchmark_generated_kernel entrypoint and stay semantically correct.",
                "",
                f"Family: {self._family}",
                f"Task: {self._task_id}",
                f"Current best config: {config_text}",
                f"Variant id: {None if variant is None else variant.get('id')}",
                f"Instruction: {instruction}",
                "",
                "Recent attempt summary:",
                *history_lines,
                "",
                "Current kernel module:",
                "```python",
                self.current_kernel_source,
                "```",
            ]
        )

    def _classify_preference_kind(self, variant: Optional[Dict[str, Any]], previous_source: str) -> str:
        chosen_role = None if variant is None else variant.get("_chosen_role")
        rejected_role = None if variant is None else variant.get("_rejected_role")
        if chosen_role == "teacher" and rejected_role == "student":
            return "teacher_beats_student_valid"
        if chosen_role == "teacher" and rejected_role == "student_invalid":
            return "teacher_rescues_student_invalid"
        if chosen_role == "student" and rejected_role == "teacher":
            return "student_beats_teacher"
        if previous_source.startswith("# INVALID_CANDIDATE"):
            return "teacher_rescues_student_invalid"
        return "teacher_beats_student_valid"

    def _maybe_store_segment_win(
        self,
        observed: Measurement,
        source: Optional[str],
        variant: Optional[Dict[str, Any]],
        previous_source: str,
        force_store: bool = False,
    ) -> None:
        if self._mode != "self_improving":
            return
        segment_key = self._segment_key_for_measurement(observed)
        oracle_best_ms = self.oracle_best()["median_ms"]
        relative_latency = float(observed.median_ms / oracle_best_ms)
        best_so_far = self.segment_best.get(segment_key, float("inf"))
        if not force_store and not relative_latency < best_so_far:
            return

        full_prompt = (
            str(variant["_prompt_text"])
            if variant is not None and "_prompt_text" in variant
            else self._build_full_prompt(variant, self._recent_history())
        )
        example = {
            "prompt": full_prompt,
            "chosen": source or self.current_kernel_source,
            "chosen_role": None if variant is None else variant.get("_chosen_role"),
            "rejected": previous_source,
            "rejected_role": None if variant is None else variant.get("_rejected_role"),
            "rejected_kind": "invalid" if previous_source.startswith("# INVALID_CANDIDATE") else "valid",
            "preference_kind": self._classify_preference_kind(variant, previous_source),
            "reward": -float(observed.median_ms),
            "segment": segment_key,
            "task_id": observed.task_id,
            "family": observed.family,
            "m": observed.m,
            "n": observed.n,
            "n_bucket": self._n_bucket(observed.n),
            "config_id": observed.config_id,
            "latency_ms": float(observed.median_ms),
            "relative_latency": relative_latency,
            "variant_id": None if variant is None else variant["id"],
            "student_status": None if variant is None else variant.get("_student_status"),
            "student_source": None if variant is None else variant.get("_student_source"),
            "source_hash": hashlib.sha256((source or self.current_kernel_source).encode("utf-8")).hexdigest(),
        }
        self.segment_buffer.setdefault(segment_key, []).append(example)
        if len(self.segment_buffer[segment_key]) > SEGMENT_BUFFER_MAX:
            self.segment_buffer[segment_key].pop(0)
        self.segment_best[segment_key] = min(best_so_far, relative_latency)
        self._persist_segment_state(segment_key)
        self._maybe_run_segment_training(segment_key)

    def _segment_tasks(self, segment_key: str) -> List[str]:
        return [
            task_id
            for task_id, rows in self._measurements.items()
            if rows and self._segment_key_for_measurement(rows[0]) == segment_key
        ]

    def _validation_tasks_for_segment(self, segment_key: str, exclude_task_id: Optional[str] = None) -> List[str]:
        tasks = [task_id for task_id in self._segment_tasks(segment_key) if task_id != exclude_task_id]
        return tasks[:VALIDATION_TASK_COUNT]

    def _persist_segment_state(self, segment_key: str) -> None:
        buffer = self.segment_buffer.get(segment_key, [])
        segment_dir = self._segment_buffer_dir / segment_key
        segment_dir.mkdir(parents=True, exist_ok=True)
        (segment_dir / "buffer.json").write_text(json.dumps(buffer, indent=2), encoding="utf-8")
        stats = {
            "segment_key": segment_key,
            "buffer_size": len(buffer),
            "best_relative_latency": self.segment_best.get(segment_key),
            "unique_ratio": self._segment_unique_ratio(segment_key),
            "train_ready": self._should_run_dpo(segment_key),
            "tasks": self._segment_tasks(segment_key),
        }
        (segment_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        self._persist_global_state()

    def _persist_global_state(self) -> None:
        payload = {
            "segment_best": self.segment_best,
            "segment_stats": self.segment_stats,
            "adapter_version": self._adapter_version,
            "adapter_path": self._adapter_path,
            "last_train_episode": self._last_train_episode,
            "training_status": self._training_status,
            "last_training_segment": self._last_training_segment,
            "last_training_result": self._last_training_result,
            "good_adapter_history": self._good_adapter_history,
        }
        self._state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load_persisted_state(self) -> None:
        if self._state_path.exists():
            try:
                payload = json.loads(self._state_path.read_text(encoding="utf-8"))
                self.segment_best = {str(k): float(v) for k, v in payload.get("segment_best", {}).items()}
                self.segment_stats = {
                    str(k): {str(sk): int(sv) for sk, sv in v.items()}
                    for k, v in payload.get("segment_stats", {}).items()
                }
                self._adapter_version = str(payload.get("adapter_version", self._adapter_version))
                self._adapter_path = payload.get("adapter_path")
                self._last_train_episode = int(payload.get("last_train_episode", self._last_train_episode))
                self._training_status = str(payload.get("training_status", self._training_status))
                self._last_training_segment = payload.get("last_training_segment")
                self._last_training_result = payload.get("last_training_result")
                self._good_adapter_history = list(payload.get("good_adapter_history", []))
            except Exception:
                pass

        if self._segment_buffer_dir.exists():
            for segment_dir in self._segment_buffer_dir.iterdir():
                if not segment_dir.is_dir():
                    continue
                buffer_path = segment_dir / "buffer.json"
                if not buffer_path.exists():
                    continue
                try:
                    self.segment_buffer[segment_dir.name] = json.loads(buffer_path.read_text(encoding="utf-8"))
                except Exception:
                    continue

    def _maybe_run_segment_training(self, segment_key: str) -> None:
        if not self._training_enabled:
            self._emit_observability_event("dpo_skipped", reason="training_disabled")
            return
        if not self._should_run_dpo(segment_key):
            self._emit_observability_event(
                "dpo_skipped",
                reason="segment_not_ready",
                buffer_size=len(self.segment_buffer.get(segment_key, [])),
            )
            return
        episodes_since_last_train = self._episode_counter - self._last_train_episode
        if episodes_since_last_train < TRAIN_INTERVAL_EPISODES and self._last_training_segment == segment_key:
            self._emit_observability_event(
                "dpo_skipped",
                reason="train_interval",
                episodes_since_last_train=episodes_since_last_train,
            )
            return
        total_start = time.perf_counter()
        self._emit_observability_event(
            "dpo_started",
            buffer_size=len(self.segment_buffer.get(segment_key, [])),
            adapter_version=self._adapter_version,
        )
        self._training_status = "exporting_job"
        export_start = time.perf_counter()
        job_dir = self._export_segment_training_job(segment_key)
        self._emit_observability_event(
            "dpo_exported",
            latency_ms=round((time.perf_counter() - export_start) * 1000.0, 3),
            job_dir=str(job_dir),
        )
        self._training_status = "running_trainer"
        trainer_start = time.perf_counter()
        result = self._run_segment_training_job(segment_key, job_dir)
        trainer_latency_ms = round((time.perf_counter() - trainer_start) * 1000.0, 3)
        self._last_train_episode = self._episode_counter
        self._last_training_segment = segment_key
        self._last_training_result = result
        self._emit_observability_event(
            "dpo_trainer_finished",
            latency_ms=trainer_latency_ms,
            status=result.get("status"),
            adapter_version=result.get("adapter_version"),
            adapter_path=result.get("adapter_path"),
            returncode=result.get("returncode"),
        )
        if result.get("status") != "completed":
            self._training_status = result.get("status", "trainer_failed")
            self._emit_observability_event(
                "dpo_finished",
                latency_ms=round((time.perf_counter() - total_start) * 1000.0, 3),
                status=self._training_status,
                accepted=False,
            )
            return
        self._training_status = "validating"
        validation_start = time.perf_counter()
        validation = self._validate_candidate_adapter(segment_key, result)
        validation_latency_ms = round((time.perf_counter() - validation_start) * 1000.0, 3)
        result["validation"] = validation
        self._last_training_result = result
        self._emit_observability_event(
            "dpo_validated",
            latency_ms=validation_latency_ms,
            accepted=bool(validation.get("accepted")),
            reason=validation.get("reason"),
            baseline_mean_ms=validation.get("baseline_mean_ms"),
            candidate_mean_ms=validation.get("candidate_mean_ms"),
        )
        if validation.get("accepted"):
            swap_start = time.perf_counter()
            self._hot_swap_adapter(result)
            swap_latency_ms = round((time.perf_counter() - swap_start) * 1000.0, 3)
            self._training_status = "adapter_active"
            self._emit_observability_event(
                "dpo_adapter_swapped",
                latency_ms=swap_latency_ms,
                adapter_version=self._adapter_version,
                adapter_path=self._adapter_path,
            )
        else:
            self._training_status = validation.get("reason", "validation_failed")
        self._emit_observability_event(
            "dpo_finished",
            latency_ms=round((time.perf_counter() - total_start) * 1000.0, 3),
            status=self._training_status,
            accepted=bool(validation.get("accepted")),
        )

    def _export_segment_training_job(self, segment_key: str) -> Path:
        timestamp = int(time.time() * 1000)
        job_dir = self._train_job_dir / f"{segment_key}-{timestamp}"
        job_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "segment_key": segment_key,
            "adapter_version_base": self._adapter_version,
            "adapter_path_base": self._adapter_path,
            "family": self._family,
            "buffer_size": len(self.segment_buffer.get(segment_key, [])),
            "examples": self.segment_buffer.get(segment_key, []),
            "validation_tasks": self._validation_tasks_for_segment(segment_key, exclude_task_id=self._task_id),
            "task_id": self._task_id,
            "current_kernel_source": self.current_kernel_source,
            "best_kernel_source": self._best_kernel_source,
        }
        (job_dir / "payload.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return job_dir

    def _run_segment_training_job(self, segment_key: str, job_dir: Path) -> Dict[str, Any]:
        trainer_command = os.environ.get("KERNEL_DPO_TRAIN_CMD", "").strip()
        result = {
            "segment_key": segment_key,
            "job_dir": str(job_dir),
            "status": "skipped_no_trainer",
            "adapter_version": f"{segment_key}-{int(time.time())}",
            "adapter_path": None,
        }
        if not trainer_command:
            trainer_command = "python3 scripts/train_segment_adapter.py"

        adapter_output_dir = self._adapter_dir / result["adapter_version"]
        adapter_output_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["KERNEL_DPO_JOB_PATH"] = str(job_dir / "payload.json")
        env["KERNEL_DPO_OUTPUT_DIR"] = str(adapter_output_dir)
        env["KERNEL_DPO_SEGMENT_KEY"] = segment_key
        cmd = shlex.split(trainer_command)
        completed = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
        (job_dir / "trainer_stdout.log").write_text(completed.stdout, encoding="utf-8")
        (job_dir / "trainer_stderr.log").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            result["status"] = "trainer_failed"
            result["returncode"] = completed.returncode
            return result

        trainer_result_path = adapter_output_dir / "result.json"
        if trainer_result_path.exists():
            trainer_result = json.loads(trainer_result_path.read_text(encoding="utf-8"))
            result.update(trainer_result)
        else:
            result.update(
                {
                    "status": "completed",
                    "adapter_path": str(adapter_output_dir),
                    "validation_passed": False,
                    "validation_reason": "trainer_result_missing",
                }
            )
        return result

    def _validate_candidate_adapter(self, segment_key: str, training_result: Dict[str, Any]) -> Dict[str, Any]:
        client = self._get_student_backend()
        if not client.supports_local_adapters:
            return {
                "accepted": False,
                "reason": "backend_does_not_support_adapter_hotswap",
                "validation_tasks": self._validation_tasks_for_segment(segment_key, exclude_task_id=self._task_id),
            }
        adapter_path = training_result.get("adapter_path")
        if not adapter_path:
            return {
                "accepted": False,
                "reason": "adapter_path_missing",
                "validation_tasks": self._validation_tasks_for_segment(segment_key, exclude_task_id=self._task_id),
            }
        adapter_json = Path(adapter_path) / "adapter.json"
        if not adapter_json.exists():
            return {
                "accepted": False,
                "reason": "adapter_metadata_missing",
                "validation_tasks": self._validation_tasks_for_segment(segment_key, exclude_task_id=self._task_id),
            }
        adapter_payload = json.loads(adapter_json.read_text(encoding="utf-8"))
        candidate_source = adapter_payload.get("preferred_source")
        if not candidate_source:
            return {
                "accepted": False,
                "reason": "preferred_source_missing",
                "validation_tasks": self._validation_tasks_for_segment(segment_key, exclude_task_id=self._task_id),
            }
        validation_tasks = self._validation_tasks_for_segment(segment_key, exclude_task_id=self._task_id)
        if not validation_tasks:
            return {"accepted": True, "reason": "no_validation_tasks", "validation_tasks": []}

        baseline_source = self.current_kernel_source or self._best_kernel_source or candidate_source
        baseline_latencies: List[float] = []
        candidate_latencies: List[float] = []
        for task_id in validation_tasks:
            task_rows = self._measurements[task_id]
            best_row = min(task_rows, key=lambda row: row.median_ms)
            baseline_measurement = self._benchmark_generated_source(best_row, baseline_source)
            candidate_measurement = self._benchmark_generated_source(best_row, candidate_source)
            baseline_latencies.append(float(baseline_measurement.median_ms))
            candidate_latencies.append(float(candidate_measurement.median_ms))
        baseline_mean = float(np.mean(np.asarray(baseline_latencies, dtype=np.float32)))
        candidate_mean = float(np.mean(np.asarray(candidate_latencies, dtype=np.float32)))
        if candidate_mean > baseline_mean:
            return {
                "accepted": False,
                "reason": "candidate_worse_than_baseline",
                "validation_tasks": validation_tasks,
                "baseline_mean_ms": baseline_mean,
                "candidate_mean_ms": candidate_mean,
            }
        return {
            "accepted": True,
            "reason": "candidate_not_worse_than_baseline",
            "validation_tasks": validation_tasks,
            "baseline_mean_ms": baseline_mean,
            "candidate_mean_ms": candidate_mean,
        }

    def _hot_swap_adapter(self, training_result: Dict[str, Any]) -> None:
        client = self._get_student_backend()
        adapter_path = training_result.get("adapter_path")
        adapter_version = str(training_result.get("adapter_version", f"adapter-{int(time.time())}"))
        if not adapter_path or not client.load_adapter(str(adapter_path)):
            self._training_status = "adapter_load_failed"
            return
        self._adapter_path = str(adapter_path)
        self._adapter_version = adapter_version
        self._good_adapter_history.append(
            {
                "version": adapter_version,
                "path": self._adapter_path,
                "segment_key": training_result.get("segment_key"),
                "timestamp": int(time.time()),
            }
        )
        if len(self._good_adapter_history) > GOOD_ADAPTER_KEEP:
            self._good_adapter_history = self._good_adapter_history[-GOOD_ADAPTER_KEEP:]

    def rollback_adapter(self, version: Optional[str] = None) -> bool:
        client = self._get_student_backend()
        if not client.supports_local_adapters:
            return False
        history = list(self._good_adapter_history)
        if version is not None:
            history = [entry for entry in history if entry["version"] == version]
        if not history:
            return False
        target = history[-1]
        if not client.load_adapter(str(target["path"])):
            return False
        self._adapter_version = str(target["version"])
        self._adapter_path = str(target["path"])
        self._training_status = "rolled_back"
        return True

    def _evolve_prompt_population(self) -> None:
        if not self.prompt_population or not self._llm_edit_enabled():
            return
        ranked = sorted(self.prompt_population, key=lambda variant: float(variant["score"]), reverse=True)
        survivor_count = max(4, len(ranked) // 2)
        survivors = [
            {
                "id": variant["id"],
                "instruction": variant["instruction"],
                "uses": variant["uses"],
                "score": variant["score"],
            }
            for variant in ranked[:survivor_count]
        ]
        children: List[Dict[str, Any]] = []
        mutation_index = 0
        while len(survivors) + len(children) < len(self.prompt_population):
            parent = survivors[(len(children) + mutation_index) % len(survivors)]
            mutation = PROMPT_MUTATIONS[mutation_index % len(PROMPT_MUTATIONS)]
            mutation_index += 1
            children.append(
                {
                    "id": f"{parent['id']}-m{mutation_index}",
                    "instruction": f"{parent['instruction']} {mutation}",
                    "uses": 0,
                    "score": float(parent["score"]) * 0.5,
                }
            )
        self.prompt_population = survivors + children

    def _benchmark_generated_source(self, row: Measurement, source: str) -> Measurement:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as handle:
            handle.write(source)
            temp_path = Path(handle.name)
        try:
            spec = importlib.util.spec_from_file_location(f"generated_kernel_{temp_path.stem}", temp_path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Unable to import generated kernel module from {temp_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if not hasattr(module, "benchmark_generated_kernel"):
                raise RuntimeError("Generated module must define benchmark_generated_kernel.")
            result = module.benchmark_generated_kernel(
                n=row.n,
                block_size=row.block_size,
                num_warps=row.num_warps,
                num_stages=row.num_stages,
                m=row.m,
            )
        finally:
            temp_path.unlink(missing_ok=True)

        return Measurement(
            family_group=row.family_group,
            family=row.family,
            task_id=row.task_id,
            m=row.m,
            n=row.n,
            config_id=row.config_id,
            block_size=row.block_size,
            num_warps=row.num_warps,
            num_stages=row.num_stages,
            median_ms=float(result["median_ms"]),
            effective_gbps=float(result["effective_gbps"]),
            score=float(result.get("score", -math.log(max(float(result["median_ms"]), np.finfo(float).tiny)))),
            validation_error=float(result["validation_error"]),
        )

    def _extract_legacy_action(self, action: Any) -> List[float]:
        arr = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        if arr.shape != (3,):
            raise ValueError("Legacy action vector must have 3 values.")
        return arr.tolist()

    def _map_legacy_action_to_config(self, action: Sequence[float]) -> int:
        base = (
            _de_norm(float(action[0]), BLOCK_SIZES),
            _de_norm(float(action[1]), NUM_WARPS),
            _de_norm(float(action[2]), NUM_STAGES),
        )
        best_id = min(
            self.available_config_ids(),
            key=lambda config_id: (
                self._row_for_id(config_id).block_size - base[0]
            ) ** 2
            + (self._row_for_id(config_id).num_warps - base[1]) ** 2
            + (self._row_for_id(config_id).num_stages - base[2]) ** 2,
        )
        return int(best_id)

    def _row_for_id(self, config_id: int) -> Measurement:
        if config_id not in self._config_by_id:
            raise ValueError(f"Unknown config_id={config_id}")
        return self._config_by_id[int(config_id)]

    def _invalidate_surrogate(self) -> None:
        self._surrogate_version += 1
        self._surrogate_fitted_version = -1
        self._surrogate_x = None
        self._surrogate_y = None
        self._surrogate_alpha = None
        self._surrogate_k = None

    def _format_step_output(
        self,
        observation: Dict[str, Any],
        reward: float,
        done: bool,
        info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "observation": observation,
            "reward": float(reward),
            "done": bool(done),
            "state": self.state(),
            "info": info or {},
        }

    def _load_measurements(self) -> Dict[str, List[Measurement]]:
        if not self.measurement_path.exists():
            raise FileNotFoundError(
                f"Missing measurement file at {self.measurement_path}. "
                "Run the measurement collectors first."
            )

        grouped: Dict[str, List[Measurement]] = {}
        with self.measurement_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fallback_config_ids: Dict[str, int] = {}
            for row in reader:
                family = row.get("family", "softmax")
                family_group = row.get("family_group", "A" if family in {"softmax", "layernorm"} else "B")
                task_id = row["task_id"]
                block_size = int(row["block_size"])
                num_warps = int(row["num_warps"])
                num_stages = int(row["num_stages"])
                config_id_raw = row.get("config_id")
                if config_id_raw in (None, ""):
                    key = f"{task_id}|{block_size}|{num_warps}|{num_stages}"
                    if key not in fallback_config_ids:
                        fallback_config_ids[key] = len([k for k in fallback_config_ids if k.startswith(f"{task_id}|")])
                    config_id = fallback_config_ids[key]
                else:
                    config_id = int(config_id_raw)

                measurement = Measurement(
                    family_group=family_group,
                    family=family,
                    task_id=task_id,
                    m=int(row["m"]),
                    n=int(row["n"]),
                    config_id=config_id,
                    block_size=block_size,
                    num_warps=num_warps,
                    num_stages=num_stages,
                    median_ms=float(row["median_ms"]),
                    effective_gbps=float(row["effective_gbps"]),
                    score=float(row["score"]),
                    validation_error=float(row["validation_error"]),
                )
                grouped.setdefault(task_id, []).append(measurement)

        for task_id in grouped:
            grouped[task_id].sort(key=lambda row: row.config_id)
        return grouped


class GenerativeKernelEnvironment(SoftmaxSurrogateEnvironment):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("mode", "self_improving")
        kwargs.setdefault("live_bench", True)
        super().__init__(*args, **kwargs)


def _config_to_vector(row: Measurement) -> np.ndarray:
    family_vec = np.zeros(len(FAMILY_INDEX), dtype=np.float32)
    if row.family in FAMILY_INDEX:
        family_vec[FAMILY_INDEX[row.family]] = 1.0
    numeric = np.array(
        [
            math.log2(max(row.m, 1)) / 16.0,
            math.log2(max(row.n, 1)) / 16.0,
            _normalize_discrete(BLOCK_SIZES, row.block_size),
            _normalize_discrete(NUM_WARPS, row.num_warps),
            _normalize_discrete(NUM_STAGES, row.num_stages),
        ],
        dtype=np.float32,
    )
    return np.concatenate([family_vec, numeric], axis=0)


def _pairwise_sq_dists(X: np.ndarray) -> np.ndarray:
    diff = X[:, None, :] - X[None, :, :]
    return np.sum(diff * diff, axis=2)


def _rbf_kernel(X: np.ndarray, Y: np.ndarray, length_scale: float) -> np.ndarray:
    sigma2 = float(length_scale * length_scale)
    if sigma2 <= 0:
        sigma2 = 1e-6
    xy = X @ Y.T
    x2 = np.sum(X * X, axis=1)[:, None]
    y2 = np.sum(Y * Y, axis=1)[None, :]
    d2 = np.maximum(x2 - 2.0 * xy + y2, 0.0)
    return np.exp(-0.5 * d2 / sigma2).astype(np.float32)


def _normal_pdf(z: float) -> float:
    inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)
    return float(inv_sqrt_2pi * math.exp(-0.5 * z * z))


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
