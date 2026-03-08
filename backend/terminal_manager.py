from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import pty
import signal
import struct
import subprocess
import termios
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
BUFFER_LIMIT = 160_000
DEFAULT_COLS = 120
DEFAULT_ROWS = 36
PYTHON_CANDIDATES = (
    "/usr/local/bin/python3",
    "/opt/homebrew/bin/python3",
    "/Users/amannindra/miniconda3/bin/python3",
)


@dataclass(frozen=True)
class AllowedJob:
    job_id: str
    label: str
    description: str
    command: tuple[str, ...]
    cwd: Path

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "label": self.label,
            "description": self.description,
            "command": list(self.command),
            "cwd": str(self.cwd),
        }


ALLOWED_JOBS: dict[str, AllowedJob] = {
    "qwen": AllowedJob(
        job_id="qwen",
        label="Qwen Baseline",
        description="Runs the exact-kernel Qwen2.5-0.5B benchmark pipeline.",
        command=("bash", "scripts/run_qwen_05b_pipeline.sh"),
        cwd=REPO_ROOT,
    ),
    "rl-agent": AllowedJob(
        job_id="rl-agent",
        label="RL Agent",
        description="Runs the multi-family surrogate and runtime benchmark pipeline.",
        command=("bash", "scripts/run_full_pipeline.sh"),
        cwd=REPO_ROOT,
    ),
}


def _probe_python(path: str) -> dict[str, Any] | None:
    if not Path(path).exists():
        return None

    script = (
        "import importlib.util, json, sys; "
        "print(json.dumps({"
        "'executable': sys.executable, "
        "'torch': bool(importlib.util.find_spec('torch')), "
        "'triton': bool(importlib.util.find_spec('triton'))"
        "}))"
    )
    try:
        result = subprocess.run(
            [path, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    try:
        payload = json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        return None
    payload["path"] = path
    return payload


def _best_python_runtime() -> dict[str, Any] | None:
    explicit = os.environ.get("TERMINAL_PYTHON_BIN")
    if explicit:
        probe = _probe_python(explicit)
        if probe is not None:
            probe["score"] = int(probe["torch"]) + int(probe["triton"])
            probe["explicit"] = True
            return probe

    best: dict[str, Any] | None = None
    for candidate in PYTHON_CANDIDATES:
        probe = _probe_python(candidate)
        if probe is None:
            continue
        score = int(probe["torch"]) + int(probe["triton"])
        probe["score"] = score
        if best is None or score > best["score"]:
            best = probe
    return best


class TerminalSession:
    def __init__(self, job: AllowedJob, loop: asyncio.AbstractEventLoop) -> None:
        self.id = uuid.uuid4().hex
        self.job = job
        self.loop = loop
        self.created_at = time.time()
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.exit_code: int | None = None
        self.status = "starting"
        self.cols = DEFAULT_COLS
        self.rows = DEFAULT_ROWS
        self.python_runtime = _best_python_runtime()

        self._buffer = ""
        self._buffer_lock = threading.Lock()
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._subscriber_lock = threading.Lock()

        self._master_fd, slave_fd = pty.openpty()
        self._resize_fd(self.cols, self.rows)

        env = os.environ.copy()
        env.setdefault("TERM", "xterm-256color")
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("FORCE_COLOR", "1")
        if self.python_runtime is not None:
            python_dir = str(Path(self.python_runtime["path"]).parent)
            env["PATH"] = f"{python_dir}:{env.get('PATH', '')}"
            env["PYTHON_BIN"] = self.python_runtime["path"]

        self._append_buffer(self._launcher_banner())

        self.process = subprocess.Popen(
            self.job.command,
            cwd=str(self.job.cwd),
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            env=env,
            preexec_fn=os.setsid,
            close_fds=True,
        )
        os.close(slave_fd)

        self.started_at = time.time()
        self.status = "running"

        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._waiter_thread = threading.Thread(target=self._wait_loop, daemon=True)
        self._reader_thread.start()
        self._waiter_thread.start()

    @property
    def command_display(self) -> str:
        return " ".join(self.job.command)

    @property
    def is_active(self) -> bool:
        return self.process.poll() is None

    def snapshot(self) -> dict[str, Any]:
        with self._buffer_lock:
            buffer = self._buffer
        return {
            "type": "snapshot",
            "session": {
                "id": self.id,
                "job_id": self.job.job_id,
                "label": self.job.label,
                "description": self.job.description,
                "cwd": str(self.job.cwd),
                "command": self.command_display,
                "status": self.status,
                "created_at": self.created_at,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "exit_code": self.exit_code,
                "cols": self.cols,
                "rows": self.rows,
            },
            "buffer": buffer,
        }

    async def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        with self._subscriber_lock:
            self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        with self._subscriber_lock:
            self._subscribers.discard(queue)

    def write(self, data: str, append_newline: bool = True) -> None:
        if not data:
            return
        payload = data + ("\n" if append_newline else "")
        os.write(self._master_fd, payload.encode("utf-8", errors="replace"))

    def resize(self, cols: int, rows: int) -> None:
        self.cols = max(20, cols)
        self.rows = max(8, rows)
        self._resize_fd(self.cols, self.rows)

    def interrupt(self) -> None:
        if self.process.poll() is None:
            os.killpg(os.getpgid(self.process.pid), signal.SIGINT)

    def terminate(self) -> None:
        if self.process.poll() is None:
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)

    def _resize_fd(self, cols: int, rows: int) -> None:
        winsize = struct.pack("HHHH", rows, cols, 0, 0)
        try:
            termios.tcsetwinsize(self._master_fd, (rows, cols))
        except AttributeError:
            pass
        try:
            import fcntl

            fcntl.ioctl(self._master_fd, termios.TIOCSWINSZ, winsize)
        except OSError:
            pass

    def _append_buffer(self, chunk: str) -> None:
        with self._buffer_lock:
            self._buffer = (self._buffer + chunk)[-BUFFER_LIMIT:]

    def _launcher_banner(self) -> str:
        lines = [
            f"[launcher] job: {self.job.label}",
            f"[launcher] cwd: {self.job.cwd}",
            f"[launcher] command: {self.command_display}",
        ]
        if self.python_runtime is not None:
            modules = []
            modules.append(f"torch={'yes' if self.python_runtime['torch'] else 'no'}")
            modules.append(f"triton={'yes' if self.python_runtime['triton'] else 'no'}")
            lines.append(f"[launcher] python3: {self.python_runtime['path']} ({', '.join(modules)})")
            if self.python_runtime.get("explicit"):
                lines.append("[launcher] python3 source: TERMINAL_PYTHON_BIN")
            if not self.python_runtime["triton"]:
                lines.append("[launcher] warning: Triton is not installed in the selected Python runtime.")
        else:
            lines.append("[launcher] warning: no preferred Python runtime detected; falling back to PATH lookup.")
        return "\n".join(lines) + "\n\n"

    def _publish(self, event: dict[str, Any]) -> None:
        with self._subscriber_lock:
            subscribers = tuple(self._subscribers)
        for queue in subscribers:
            self.loop.call_soon_threadsafe(self._safe_put, queue, event)

    @staticmethod
    def _safe_put(queue: asyncio.Queue[dict[str, Any]], event: dict[str, Any]) -> None:
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            pass

    def _reader_loop(self) -> None:
        while True:
            try:
                data = os.read(self._master_fd, 4096)
            except OSError:
                break
            if not data:
                break
            text = data.decode("utf-8", errors="replace")
            self._append_buffer(text)
            self._publish({"type": "output", "data": text})

    def _wait_loop(self) -> None:
        exit_code = self.process.wait()
        self.exit_code = exit_code
        self.finished_at = time.time()
        self.status = "exited" if exit_code == 0 else "failed"
        self._publish(
            {
                "type": "exit",
                "exit_code": exit_code,
                "status": self.status,
                "finished_at": self.finished_at,
            }
        )
        try:
            os.close(self._master_fd)
        except OSError:
            pass


class TerminalManager:
    def __init__(self) -> None:
        self._sessions: dict[str, TerminalSession] = {}
        self._latest_by_job: dict[str, str] = {}
        self._lock = threading.Lock()

    def list_jobs(self) -> list[dict[str, Any]]:
        return [job.as_dict() for job in ALLOWED_JOBS.values()]

    def get_session(self, session_id: str) -> TerminalSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    async def ensure_session(self, job_id: str, restart: bool = False) -> TerminalSession:
        if job_id not in ALLOWED_JOBS:
            raise KeyError(job_id)

        with self._lock:
            existing_id = self._latest_by_job.get(job_id)
            existing = self._sessions.get(existing_id) if existing_id else None

        if existing and existing.is_active and not restart:
            return existing

        if existing and restart:
            existing.interrupt()

        session = TerminalSession(ALLOWED_JOBS[job_id], asyncio.get_running_loop())
        with self._lock:
            self._sessions[session.id] = session
            self._latest_by_job[job_id] = session.id
        return session
