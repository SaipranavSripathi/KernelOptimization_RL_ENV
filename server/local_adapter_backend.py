from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

try:
    from unsloth import FastLanguageModel
except Exception:
    FastLanguageModel = None

try:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception as err:  # pragma: no cover
    AutoModelForCausalLM = None
    AutoTokenizer = None
    PeftModel = None
    _IMPORT_ERROR = err
else:
    _IMPORT_ERROR = None


DEFAULT_LOCAL_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _use_unsloth() -> bool:
    mode = os.environ.get("KERNEL_USE_UNSLOTH", "auto").strip().lower()
    if mode in {"0", "false", "no", "off", "disable", "disabled"}:
        return False
    if mode in {"1", "true", "yes", "on", "force", "forced", "require", "required"}:
        if FastLanguageModel is None:
            raise RuntimeError("KERNEL_USE_UNSLOTH is enabled, but unsloth is not installed.")
        if not torch.cuda.is_available():
            raise RuntimeError("KERNEL_USE_UNSLOTH requires CUDA.")
        return True
    return FastLanguageModel is not None and torch.cuda.is_available()


def _unsloth_dtype(default_dtype: torch.dtype) -> torch.dtype | None:
    value = os.environ.get("KERNEL_UNSLOTH_DTYPE", "auto").strip().lower()
    if value in {"", "auto"}:
        return None
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16", "half"}:
        return torch.float16
    if value in {"fp32", "float32", "float"}:
        return torch.float32
    return default_dtype


class LocalAdapterBackend:
    def __init__(
        self,
        model_id: Optional[str] = None,
        local_files_only: Optional[bool] = None,
    ) -> None:
        if _IMPORT_ERROR is not None:
            raise RuntimeError(
                "Local adapter backend requires transformers and peft."
            ) from _IMPORT_ERROR

        self._use_unsloth = _use_unsloth()
        self.backend_name = "local_unsloth" if self._use_unsloth else "local_transformers"
        self.supports_local_adapters = True
        self.model_id = model_id or os.environ.get("KERNEL_LOCAL_MODEL_ID", DEFAULT_LOCAL_MODEL_ID)
        if local_files_only is None:
            local_files_only = os.environ.get("KERNEL_LOCAL_FILES_ONLY", "0").strip() == "1"
        self.local_files_only = bool(local_files_only)
        self.do_sample = os.environ.get("KERNEL_LOCAL_DO_SAMPLE", "0").strip() == "1"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.max_seq_length = int(os.environ.get("KERNEL_TRAIN_MAX_LEN", "4096"))

        if self._use_unsloth:
            self.base_model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_id,
                max_seq_length=self.max_seq_length,
                dtype=_unsloth_dtype(self.dtype),
                load_in_4bit=_env_flag("KERNEL_UNSLOTH_LOAD_IN_4BIT", default=False),
                local_files_only=self.local_files_only,
                trust_remote_code=True,
            )
            FastLanguageModel.for_inference(self.base_model)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                local_files_only=self.local_files_only,
                trust_remote_code=True,
            )
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
                device_map="auto" if self.device == "cuda" else None,
                local_files_only=self.local_files_only,
                trust_remote_code=True,
            )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = self.base_model
        self._adapter_path: Optional[str] = None

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        lines = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            lines.append(f"{role.upper()}:\n{content}\n")
        lines.append("ASSISTANT:\n")
        return "\n".join(lines)

    def _model_device(self) -> torch.device:
        if hasattr(self.model, "device"):
            return torch.device(self.model.device)
        return next(self.model.parameters()).device

    def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.6,
        max_tokens: int = 4096,
    ) -> str:
        prompt = self._format_messages(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        model_device = self._model_device()
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        generation = self.model.generate(
            **inputs,
            do_sample=self.do_sample and temperature > 0,
            temperature=max(temperature, 1e-5),
            max_new_tokens=max_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        generated = generation[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def load_adapter(self, adapter_path: str) -> bool:
        if PeftModel is None:
            return False
        root = Path(adapter_path)
        if not root.exists():
            return False
        try:
            self.model = PeftModel.from_pretrained(self.base_model, str(root), is_trainable=False)
            if self._use_unsloth and FastLanguageModel is not None:
                FastLanguageModel.for_inference(self.model)
            self.model.eval()
            self._adapter_path = str(root)
            return True
        except Exception:
            return False

    def get_loaded_adapter_path(self) -> Optional[str]:
        return self._adapter_path
