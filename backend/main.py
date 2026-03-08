from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.terminal_manager import ALLOWED_JOBS, TerminalManager

app = FastAPI(
    title="RL Autotuning Backend",
    description="Backend API for the multi-family GPU autotuning benchmark",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

env = None
terminal_manager = TerminalManager()


def _get_env():
    global env
    if env is None:
        try:
            from server.softmax_surrogate_environment import SoftmaxSurrogateEnvironment
            env = SoftmaxSurrogateEnvironment()
        except ImportError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Environment unavailable – missing dependency: {exc.name}",
            )
    return env


class ResetRequest(BaseModel):
    task: Optional[str] = None
    seed: Optional[int] = None


class StepRequest(BaseModel):
    config_id: Optional[int] = None
    x: Optional[List[float]] = None


class SessionRequest(BaseModel):
    job_id: str
    restart: bool = False


class SessionInputRequest(BaseModel):
    data: str
    append_newline: bool = True


class SessionResizeRequest(BaseModel):
    cols: int
    rows: int


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/reset")
def reset(payload: ResetRequest) -> Dict[str, Any]:
    return _get_env().reset(task=payload.task, seed=payload.seed)


@app.post("/step")
def step(payload: StepRequest) -> Dict[str, Any]:
    e = _get_env()
    if payload.config_id is not None:
        return e.step({"config_id": payload.config_id})
    if payload.x is not None:
        return e.step({"x": payload.x})
    raise HTTPException(status_code=400, detail="Provide config_id or x.")


@app.get("/state")
def state() -> Dict[str, Any]:
    return _get_env().state()


@app.get("/terminal/jobs")
def terminal_jobs() -> Dict[str, Any]:
    return {"jobs": terminal_manager.list_jobs()}


@app.post("/terminal/sessions")
async def create_terminal_session(payload: SessionRequest) -> Dict[str, Any]:
    if payload.job_id not in ALLOWED_JOBS:
        raise HTTPException(status_code=404, detail=f"Unknown job_id: {payload.job_id}")
    session = await terminal_manager.ensure_session(payload.job_id, restart=payload.restart)
    return session.snapshot()


@app.get("/terminal/sessions/{session_id}")
def terminal_session_snapshot(session_id: str) -> Dict[str, Any]:
    session = terminal_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.snapshot()


@app.post("/terminal/sessions/{session_id}/input")
def terminal_session_input(session_id: str, payload: SessionInputRequest) -> Dict[str, Any]:
    session = terminal_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if not session.is_active:
        raise HTTPException(status_code=409, detail="Session is not running")
    session.write(payload.data, append_newline=payload.append_newline)
    return {"ok": True}


@app.post("/terminal/sessions/{session_id}/resize")
def terminal_session_resize(session_id: str, payload: SessionResizeRequest) -> Dict[str, Any]:
    session = terminal_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    session.resize(payload.cols, payload.rows)
    return {"ok": True}


@app.post("/terminal/sessions/{session_id}/stop")
def terminal_session_stop(session_id: str) -> Dict[str, Any]:
    session = terminal_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    session.interrupt()
    return {"ok": True}


@app.websocket("/terminal/sessions/{session_id}/stream")
async def terminal_session_stream(websocket: WebSocket, session_id: str) -> None:
    session = terminal_manager.get_session(session_id)
    if session is None:
        await websocket.close(code=4404)
        return

    await websocket.accept()
    queue = await session.subscribe()
    try:
        await websocket.send_json(session.snapshot())
        while True:
            event = await queue.get()
            await websocket.send_json(event)
    except WebSocketDisconnect:
        pass
    finally:
        session.unsubscribe(queue)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
