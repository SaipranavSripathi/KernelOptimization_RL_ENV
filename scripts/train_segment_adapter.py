#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from unsloth import FastLanguageModel
except Exception:
    FastLanguageModel = None

try:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception as err:  # pragma: no cover
    raise RuntimeError(
        "train_segment_adapter.py requires transformers and peft."
    ) from err


DEFAULT_LOCAL_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DPO_ALLOWED_PREFERENCE_KINDS = {
    "teacher_beats_student_valid",
    "teacher_rescues_student_invalid",
}


def _load_payload() -> dict:
    payload_path = Path(os.environ["KERNEL_DPO_JOB_PATH"])
    return json.loads(payload_path.read_text(encoding="utf-8"))


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _dtype() -> torch.dtype:
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


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


def _unsloth_gradient_checkpointing() -> bool | str:
    value = os.environ.get("KERNEL_UNSLOTH_GRADIENT_CHECKPOINTING", "unsloth").strip()
    lowered = value.lower()
    if lowered in {"0", "false", "no", "off", "disable", "disabled"}:
        return False
    if lowered in {"1", "true", "yes", "on"}:
        return True
    return value


def _load_model_with_backend(
    model_id: str,
    *,
    local_files_only: bool,
    dtype: torch.dtype,
    max_length: int,
    use_unsloth: bool,
) -> tuple[Any, Any]:
    if use_unsloth:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=max_length,
            dtype=_unsloth_dtype(dtype),
            load_in_4bit=_env_flag("KERNEL_UNSLOTH_LOAD_IN_4BIT", default=False),
            local_files_only=local_files_only,
            trust_remote_code=True,
        )
        return model, tokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        local_files_only=local_files_only,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=None,
        local_files_only=local_files_only,
        trust_remote_code=True,
    )
    return model, tokenizer


def _build_sequence(
    prompt_ids: List[int],
    completion_ids: List[int],
    eos_token_id: int,
    max_length: int,
) -> Dict[str, List[int]]:
    completion = completion_ids + [eos_token_id]
    prompt = list(prompt_ids)
    total = len(prompt) + len(completion)
    if total > max_length:
        trim = total - max_length
        if trim < len(prompt):
            prompt = prompt[trim:]
        else:
            prompt = []
            completion = completion[trim - len(prompt) :]
            completion = completion[-max_length:]
    input_ids = prompt + completion
    labels = [-100] * len(prompt) + completion
    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


class PreferenceDPODataset(Dataset):
    def __init__(self, tokenizer: Any, examples: List[Dict[str, Any]], max_length: int = 4096) -> None:
        self.rows: List[Dict[str, torch.Tensor]] = []
        for example in examples:
            prompt = str(example["prompt"])
            chosen = str(example["chosen"])
            rejected = str(example["rejected"])
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            chosen_ids = tokenizer(chosen, add_special_tokens=False)["input_ids"]
            rejected_ids = tokenizer(rejected, add_special_tokens=False)["input_ids"]

            chosen_seq = _build_sequence(prompt_ids, chosen_ids, tokenizer.eos_token_id, max_length)
            rejected_seq = _build_sequence(prompt_ids, rejected_ids, tokenizer.eos_token_id, max_length)
            self.rows.append(
                {
                    "chosen_input_ids": torch.tensor(chosen_seq["input_ids"], dtype=torch.long),
                    "chosen_attention_mask": torch.tensor(chosen_seq["attention_mask"], dtype=torch.long),
                    "chosen_labels": torch.tensor(chosen_seq["labels"], dtype=torch.long),
                    "rejected_input_ids": torch.tensor(rejected_seq["input_ids"], dtype=torch.long),
                    "rejected_attention_mask": torch.tensor(rejected_seq["attention_mask"], dtype=torch.long),
                    "rejected_labels": torch.tensor(rejected_seq["labels"], dtype=torch.long),
                    "weight": torch.tensor(
                        0.35 if str(example.get("preference_kind", "")) == "teacher_rescues_student_invalid" else 1.0,
                        dtype=torch.float32,
                    ),
                }
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.rows[idx]


@dataclass
class PreferenceCollator:
    tokenizer: Any

    def _pad_tensor_list(self, values: List[torch.Tensor], pad_value: int) -> torch.Tensor:
        max_len = max(value.shape[0] for value in values)
        padded = [F.pad(value, (0, max_len - value.shape[0]), value=pad_value) for value in values]
        return torch.stack(padded, dim=0)

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return {
            "chosen_input_ids": self._pad_tensor_list([item["chosen_input_ids"] for item in batch], self.tokenizer.pad_token_id),
            "chosen_attention_mask": self._pad_tensor_list([item["chosen_attention_mask"] for item in batch], 0),
            "chosen_labels": self._pad_tensor_list([item["chosen_labels"] for item in batch], -100),
            "rejected_input_ids": self._pad_tensor_list([item["rejected_input_ids"] for item in batch], self.tokenizer.pad_token_id),
            "rejected_attention_mask": self._pad_tensor_list([item["rejected_attention_mask"] for item in batch], 0),
            "rejected_labels": self._pad_tensor_list([item["rejected_labels"] for item in batch], -100),
            "weight": torch.stack([item["weight"] for item in batch], dim=0),
        }


def _target_modules(model: Any) -> List[str]:
    names = []
    for name, _ in model.named_modules():
        if name.endswith(("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")):
            names.append(name.split(".")[-1])
    deduped = sorted(set(names))
    return deduped or ["q_proj", "v_proj"]


def _sequence_logprob(model: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :].float()
    target = labels[:, 1:].clone()
    mask = target != -100
    safe_target = target.masked_fill(~mask, 0)
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, safe_target.unsqueeze(-1)).squeeze(-1)
    token_log_probs = token_log_probs * mask
    return token_log_probs.sum(dim=-1)


def main() -> None:
    payload = _load_payload()
    output_dir = Path(os.environ["KERNEL_DPO_OUTPUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = payload.get("examples", [])
    if not examples:
        result = {
            "status": "trainer_failed",
            "reason": "no_examples",
            "adapter_path": str(output_dir),
            "adapter_version": output_dir.name,
        }
        (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return

    preference_counts: Dict[str, int] = {}
    for example in examples:
        key = str(example.get("preference_kind", "unknown"))
        preference_counts[key] = preference_counts.get(key, 0) + 1
    dpo_examples = [
        example
        for example in examples
        if str(example.get("preference_kind", "")) in DPO_ALLOWED_PREFERENCE_KINDS
    ]
    if not dpo_examples:
        result = {
            "status": "trainer_failed",
            "reason": "no_dpo_examples",
            "adapter_path": str(output_dir),
            "adapter_version": output_dir.name,
            "preference_counts": preference_counts,
        }
        (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return

    model_id = os.environ.get("KERNEL_LOCAL_MODEL_ID", DEFAULT_LOCAL_MODEL_ID)
    local_files_only = os.environ.get("KERNEL_LOCAL_FILES_ONLY", "0").strip() == "1"
    beta = float(os.environ.get("KERNEL_DPO_BETA", "0.1"))
    learning_rate = float(os.environ.get("KERNEL_TRAIN_LR", "1e-5"))
    epochs = int(float(os.environ.get("KERNEL_TRAIN_EPOCHS", "1")))
    batch_size = int(os.environ.get("KERNEL_TRAIN_BATCH_SIZE", "1"))
    grad_accum = int(os.environ.get("KERNEL_TRAIN_GRAD_ACC", "4"))
    max_length = int(os.environ.get("KERNEL_TRAIN_MAX_LEN", "4096"))
    use_unsloth = _use_unsloth()
    rank = int(os.environ.get("KERNEL_LORA_RANK", "16"))
    lora_alpha = int(os.environ.get("KERNEL_LORA_ALPHA", "32"))
    lora_dropout = float(os.environ.get("KERNEL_LORA_DROPOUT", "0.05"))
    device = _device()
    dtype = _dtype()

    policy_base, tokenizer = _load_model_with_backend(
        model_id,
        local_files_only=local_files_only,
        dtype=dtype,
        max_length=max_length,
        use_unsloth=use_unsloth,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    reference_model, _ = _load_model_with_backend(
        model_id,
        local_files_only=local_files_only,
        dtype=dtype,
        max_length=max_length,
        use_unsloth=use_unsloth,
    )
    reference_model.to(device)
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False

    if use_unsloth:
        policy_model = FastLanguageModel.get_peft_model(
            policy_base,
            r=rank,
            target_modules=_target_modules(policy_base),
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            use_gradient_checkpointing=_unsloth_gradient_checkpointing(),
            use_rslora=_env_flag("KERNEL_UNSLOTH_USE_RSLORA", default=False),
        )
    else:
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=_target_modules(policy_base),
        )
        policy_model = get_peft_model(policy_base, lora_config)
    policy_model.to(device)
    policy_model.train()

    dataset = PreferenceDPODataset(tokenizer, dpo_examples, max_length=max_length)
    collator = PreferenceCollator(tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)

    optimizer = torch.optim.AdamW(
        [param for param in policy_model.parameters() if param.requires_grad],
        lr=learning_rate,
    )

    step_count = 0
    running_losses: List[float] = []
    for _ in range(max(1, epochs)):
        optimizer.zero_grad(set_to_none=True)
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}

            policy_chosen = _sequence_logprob(
                policy_model,
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"],
                batch["chosen_labels"],
            )
            policy_rejected = _sequence_logprob(
                policy_model,
                batch["rejected_input_ids"],
                batch["rejected_attention_mask"],
                batch["rejected_labels"],
            )

            with torch.no_grad():
                ref_chosen = _sequence_logprob(
                    reference_model,
                    batch["chosen_input_ids"],
                    batch["chosen_attention_mask"],
                    batch["chosen_labels"],
                )
                ref_rejected = _sequence_logprob(
                    reference_model,
                    batch["rejected_input_ids"],
                    batch["rejected_attention_mask"],
                    batch["rejected_labels"],
                )

            preference_margin = (policy_chosen - policy_rejected) - (ref_chosen - ref_rejected)
            dpo_loss = -F.logsigmoid(beta * preference_margin)
            loss = (dpo_loss * batch["weight"]).sum() / batch["weight"].sum().clamp_min(1e-6)
            (loss / grad_accum).backward()
            running_losses.append(float(loss.item()))
            step_count += 1
            if step_count % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        if step_count % grad_accum != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    policy_model.save_pretrained(str(output_dir))

    best_example = max(dpo_examples, key=lambda example: float(example["reward"]))
    adapter_payload = {
        "segment_key": payload["segment_key"],
        "adapter_version": output_dir.name,
        "preferred_source": best_example["chosen"],
        "preferred_prompt": best_example["prompt"],
        "reward": best_example["reward"],
        "source_hash": best_example["source_hash"],
        "model_id": model_id,
        "training_backend": "unsloth" if use_unsloth else "transformers_peft",
        "training_examples": len(dpo_examples),
        "buffer_examples": len(examples),
        "preference_counts": preference_counts,
        "objective": "lora_dpo",
        "beta": beta,
        "mean_training_loss": float(sum(running_losses) / len(running_losses)) if running_losses else None,
    }
    (output_dir / "adapter.json").write_text(json.dumps(adapter_payload, indent=2), encoding="utf-8")

    result = {
        "status": "completed",
        "adapter_path": str(output_dir),
        "adapter_version": output_dir.name,
        "training_examples": len(dpo_examples),
        "buffer_examples": len(examples),
        "preference_counts": preference_counts,
        "validation_passed": True,
        "validation_reason": "lora_dpo_completed",
        "training_backend": adapter_payload["training_backend"],
        "objective": "lora_dpo",
        "mean_training_loss": adapter_payload["mean_training_loss"],
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
