#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import os
import time

import websockets


async def main() -> None:
    uri = os.environ.get("WS_URI", "ws://127.0.0.1:8000/ws/chat")
    start = time.perf_counter()
    teacher_thinking_total = 0
    teacher_content_total = 0
    async with websockets.connect(
        uri,
        max_size=2**24,
        ping_interval=None,
        ping_timeout=None,
        open_timeout=None,
        close_timeout=None,
    ) as ws:
        await ws.send(
            json.dumps(
                {
                    "task": "softmax_m4096_n256",
                    "config_id": 0,
                    "instruction": "Optimize the current Triton softmax kernel conservatively and preserve exact semantics.",
                }
            )
        )
        while True:
            event = json.loads(await ws.recv())
            event_type = str(event.get("type", "unknown"))
            summary = {
                "t_s": round(time.perf_counter() - start, 3),
                "type": event_type,
            }
            if event_type == "teacher_thinking_delta":
                delta_chars = len(event.get("text") or "")
                teacher_thinking_total += delta_chars
                summary["delta_chars"] = delta_chars
                summary["total_chars"] = teacher_thinking_total
            elif event_type == "teacher_content_delta":
                delta_chars = len(event.get("text") or "")
                teacher_content_total += delta_chars
                summary["delta_chars"] = delta_chars
                summary["total_chars"] = teacher_content_total
            else:
                summary.update({k: v for k, v in event.items() if k not in {"source", "student_source", "teacher_source", "text"}})
                for key in ("source", "student_source", "teacher_source"):
                    if key in event:
                        summary[f"{key}_len"] = len(event.get(key) or "")
            print(json.dumps(summary, indent=2))
            if event.get("type") == "final_result":
                break


if __name__ == "__main__":
    asyncio.run(main())
