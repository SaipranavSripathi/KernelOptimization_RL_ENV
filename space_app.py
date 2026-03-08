from __future__ import annotations

import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from server.softmax_surrogate_environment import (
    GenerativeKernelEnvironment,
    SoftmaxSurrogateEnvironment,
)


ENV_MODE = os.environ.get("KERNEL_ENV_MODE", "surrogate").strip().lower()
app = FastAPI(
    title="RL Surrogate ENV",
    description=(
        "CPU-safe surrogate API for the multi-family GPU autotuning benchmark. "
        "The full repo remains available in the Space, while the deployed app exposes "
        "interactive API docs and stateful reset/step endpoints."
    ),
    version="0.1.0",
)
env: Optional[SoftmaxSurrogateEnvironment] = None


class ResetRequest(BaseModel):
    task: Optional[str] = None
    seed: Optional[int] = None


class StepRequest(BaseModel):
    config_id: Optional[int] = None
    x: Optional[list[float]] = None
    source: Optional[str] = None


def _build_env() -> SoftmaxSurrogateEnvironment:
    if ENV_MODE in {"default", "self_improving", "generative"}:
        return GenerativeKernelEnvironment(mode=ENV_MODE)
    return SoftmaxSurrogateEnvironment(mode="surrogate")


def _get_env() -> SoftmaxSurrogateEnvironment:
    global env
    if env is None:
        try:
            env = _build_env()
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Environment unavailable: {exc}",
            ) from exc
    return env


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/info")
def info() -> Dict[str, Any]:
    return {
        "mode": ENV_MODE,
        "entrypoint": "space_app:app",
        "notes": [
            "The deployed Space uses surrogate mode by default so it works without GPU benchmarking dependencies.",
            "The full benchmark, frontend, and self-improving workflows are still present in the repository.",
        ],
    }


@app.post("/reset")
def reset(payload: ResetRequest) -> Dict[str, Any]:
    return _get_env().reset(task=payload.task, seed=payload.seed)


@app.post("/step")
def step(payload: StepRequest) -> Dict[str, Any]:
    environment = _get_env()
    if payload.config_id is not None:
        return environment.step({"config_id": payload.config_id, "source": payload.source})
    if payload.x is not None:
        return environment.step({"x": payload.x, "source": payload.source})
    raise HTTPException(status_code=400, detail="Provide config_id or x.")


@app.get("/state")
def state() -> Dict[str, Any]:
    return _get_env().state()
