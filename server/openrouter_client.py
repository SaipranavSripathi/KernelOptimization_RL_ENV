from __future__ import annotations

import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "qwen/qwen3-vl-235b-a22b-thinking"


def _load_api_key() -> str:
    direct = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if direct:
        return direct
    bashrc = Path.home() / ".bashrc"
    if bashrc.exists():
        text = bashrc.read_text(encoding="utf-8", errors="ignore")
        match = re.search(r"OPENROUTER_API_KEY\s*=\s*['\"]?([^'\n\"]+)['\"]?", text)
        if match:
            return match.group(1).strip()
    raise RuntimeError("OPENROUTER_API_KEY is not set.")


class OpenRouterClient:
    def __init__(self, model: Optional[str] = None) -> None:
        self.model = (
            model
            or os.environ.get("KERNEL_OPENROUTER_MODEL")
            or os.environ.get("OPENROUTER_MODEL")
            or DEFAULT_MODEL
        )
        self.api_key = _load_api_key()
        self.backend_name = "openrouter"
        self.supports_local_adapters = False

    def _coerce_text(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts = [item.get("text", "") for item in value if isinstance(item, dict)]
            return "\n".join(part for part in parts if part)
        return ""

    def _coerce_reasoning(self, delta: Dict[str, Any]) -> str:
        parts: List[str] = []
        direct = self._coerce_text(delta.get("reasoning", ""))
        if direct:
            parts.append(direct)
        for item in delta.get("reasoning_details", []) or []:
            if not isinstance(item, dict):
                continue
            for key in ("text", "content"):
                text = self._coerce_text(item.get(key, ""))
                if text:
                    parts.append(text)
            summary = item.get("summary")
            if isinstance(summary, list):
                for summary_item in summary:
                    if not isinstance(summary_item, dict):
                        continue
                    text = self._coerce_text(summary_item.get("text", ""))
                    if text:
                        parts.append(text)
        return "\n".join(part for part in parts if part)

    def _request_payload(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        *,
        stream: bool = False,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> bytes:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if stream:
            payload["stream"] = True
        if extra_body:
            payload.update(extra_body)
        return json.dumps(payload).encode("utf-8")

    def _request(self, payload: bytes) -> urllib.request.Request:
        return urllib.request.Request(
            OPENROUTER_URL,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://openrouter.ai/",
                "X-Title": "cuda-generative-kernel-environment",
            },
            method="POST",
        )

    def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.6,
        max_tokens: int = 4096,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> str:
        payload = self._request_payload(
            messages,
            temperature,
            max_tokens,
            extra_body=extra_body,
        )
        request = self._request(payload)
        with urllib.request.urlopen(request, timeout=300) as response:
            body: Dict[str, Any] = json.loads(response.read().decode("utf-8"))
        choices = body.get("choices", [])
        if not choices:
            raise RuntimeError(f"OpenRouter returned no choices: {body}")
        message = choices[0].get("message", {})
        return self._coerce_text(message.get("content", ""))

    def stream_complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.6,
        max_tokens: int = 4096,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Dict[str, Any]]:
        payload = self._request_payload(
            messages,
            temperature,
            max_tokens,
            stream=True,
            extra_body=extra_body,
        )
        request = self._request(payload)
        with urllib.request.urlopen(request, timeout=300) as response:
            event_lines: List[str] = []
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                if not line:
                    if not event_lines:
                        continue
                    data_lines = [item[5:].lstrip() for item in event_lines if item.startswith("data:")]
                    event_lines = []
                    if not data_lines:
                        continue
                    payload_text = "\n".join(data_lines)
                    if payload_text == "[DONE]":
                        break
                    chunk: Dict[str, Any] = json.loads(payload_text)
                    delta = (((chunk.get("choices") or [{}])[0]).get("delta") or {})
                    yield {
                        "content": self._coerce_text(delta.get("content", "")),
                        "reasoning": self._coerce_reasoning(delta),
                        "raw": chunk,
                    }
                    continue
                if line.startswith(":"):
                    continue
                event_lines.append(line)

    def load_adapter(self, adapter_path: str) -> bool:
        return False
