from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import WebSocket
from fastapi import WebSocketDisconnect
from pydantic import BaseModel

from models import ResetResult, StepResult
from server.softmax_surrogate_environment import GenerativeKernelEnvironment, SoftmaxSurrogateEnvironment


app = FastAPI(title="Autotune Benchmark OpenEnv Server")
ENV_MODE = os.environ.get("KERNEL_ENV_MODE", "surrogate").strip().lower()
env = GenerativeKernelEnvironment(mode=ENV_MODE) if ENV_MODE in {"default", "self_improving", "generative"} else SoftmaxSurrogateEnvironment(mode=ENV_MODE)


class ResetRequest(BaseModel):
    task: Optional[str] = None
    seed: Optional[int] = None


class StepRequest(BaseModel):
    config_id: Optional[int] = None
    x: Optional[list[float]] = None
    source: Optional[str] = None


class ChatOptimizeRequest(BaseModel):
    task: Optional[str] = None
    config_id: Optional[int] = None
    source: Optional[str] = None
    seed: Optional[int] = None
    instruction: Optional[str] = None


@app.get("/health")
def health() -> Dict[str, str]:
    return {"ok": "true"}


@app.post("/reset")
def reset(payload: ResetRequest) -> Dict[str, Any]:
    result = env.reset(task=payload.task, seed=payload.seed)
    return result


@app.post("/step")
def step(payload: StepRequest) -> Dict[str, Any]:
    if payload.config_id is not None:
        result = env.step({"config_id": payload.config_id, "source": payload.source})
        return result
    if payload.x is not None:
        result = env.step({"x": payload.x, "source": payload.source})
        return result
    raise HTTPException(status_code=400, detail="Missing config_id.")
    return result


@app.get("/state")
def state() -> Dict[str, Any]:
    return env.state()


@app.websocket("/ws/chat")
async def chat_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            payload = await websocket.receive_json()
            request = ChatOptimizeRequest(**payload)
            for event in env.chat_optimize_events(
                task=request.task,
                config_id=request.config_id,
                source=request.source,
                seed=request.seed,
                instruction=request.instruction,
            ):
                await websocket.send_json(event)
    except WebSocketDisconnect:
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run softmax surrogate environment server.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    try:
        import uvicorn

        uvicorn.run("server.app:app", host=args.host, port=args.port, reload=False)
    except Exception as err:  # pragma: no cover
        raise RuntimeError("uvicorn not available") from err
