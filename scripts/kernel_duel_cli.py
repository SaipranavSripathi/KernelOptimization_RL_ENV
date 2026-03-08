#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

import websockets
from rich import box
from rich.align import Align
from rich.console import Console, Group, RenderableType
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text


def _clip_tail(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[-limit:]


def _clip_lines(lines: List[str], limit: int) -> List[str]:
    return lines[-limit:]


@dataclass
class PaneState:
    label: str
    source: str = ""
    status: str = "idle"
    compile_status: str = "pending"
    latency_ms: Optional[float] = None
    preview: bool = False
    error: Optional[str] = None


@dataclass
class DuelState:
    uri: str
    task: str
    config_id: int
    history: List[str] = field(default_factory=list)
    timeline: List[str] = field(default_factory=list)
    student: PaneState = field(default_factory=lambda: PaneState(label="Student"))
    teacher: PaneState = field(default_factory=lambda: PaneState(label="Teacher"))
    teacher_thinking: str = ""
    teacher_streamed_code: str = ""
    phase: str = "idle"
    winner_text: str = "READY"
    winner_style: str = "bold white on dark_blue"
    result_class: Optional[str] = None
    training_status: Optional[str] = None
    adapter_version: Optional[str] = None
    last_error: Optional[str] = None

    def reset_for_prompt(self, prompt: str) -> None:
        self.history.append(f"You: {prompt}")
        self.history = _clip_lines(self.history, 18)
        self.timeline = _clip_lines(self.timeline + [f"Queued request for {self.task} / config {self.config_id}"], 24)
        self.student = PaneState(label="Student")
        self.teacher = PaneState(label="Teacher")
        self.teacher_thinking = ""
        self.teacher_streamed_code = ""
        self.phase = "connecting"
        self.winner_text = "AWAITING KERNELS"
        self.winner_style = "bold black on yellow"
        self.result_class = None
        self.training_status = None
        self.adapter_version = None
        self.last_error = None

    def note(self, elapsed_s: float, message: str) -> None:
        self.timeline.append(f"{elapsed_s:6.2f}s  {message}")
        self.timeline = _clip_lines(self.timeline, 24)

    def ingest(self, event: dict[str, Any], elapsed_s: float) -> None:
        event_type = str(event.get("type", "unknown"))
        if event_type == "request_received":
            self.phase = "request_received"
            self.note(elapsed_s, "Request accepted by websocket backend")
            return
        if event_type == "reset_complete":
            self.phase = "proposal_setup"
            self.note(elapsed_s, f"Reset complete for {event.get('task_id')}")
            return
        if event_type == "proposal_started":
            self.phase = "proposing"
            self.note(elapsed_s, "Teacher and student proposal phase started")
            return
        if event_type == "student_preview":
            self.student.source = event.get("source") or self.student.source
            self.student.status = "preview"
            self.student.preview = True
            self.student.error = event.get("error")
            self.note(elapsed_s, f"Student preview ready ({len(self.student.source)} chars)")
            self.winner_text = "STUDENT PREVIEW READY"
            self.winner_style = "bold black on bright_yellow"
            return
        if event_type == "student_ready":
            self.student.source = event.get("source") or self.student.source
            self.student.status = "proposed"
            self.student.preview = False
            self.student.error = event.get("error")
            self.note(elapsed_s, f"Student final kernel ready ({len(self.student.source)} chars)")
            return
        if event_type == "teacher_thinking_delta":
            self.teacher_thinking = _clip_tail(self.teacher_thinking + str(event.get("text") or ""), 5000)
            if self.phase == "proposing":
                self.winner_text = "TEACHER THINKING LIVE"
                self.winner_style = "bold white on blue"
            return
        if event_type == "teacher_content_delta":
            first_chunk = not self.teacher_streamed_code
            self.teacher_streamed_code = _clip_tail(self.teacher_streamed_code + str(event.get("text") or ""), 12000)
            self.teacher.source = self.teacher_streamed_code
            self.teacher.status = "streaming"
            if first_chunk:
                self.note(elapsed_s, "Teacher code started streaming")
                self.winner_text = "TEACHER CODE LIVE"
                self.winner_style = "bold white on bright_blue"
            return
        if event_type == "teacher_ready":
            self.teacher.source = event.get("source") or self.teacher_streamed_code or self.teacher.source
            self.teacher.status = "proposed"
            self.teacher.error = event.get("error")
            self.note(elapsed_s, f"Teacher final kernel ready ({len(self.teacher.source)} chars)")
            return
        if event_type == "kernels_ready":
            self.student.source = event.get("student_source") or self.student.source
            self.teacher.source = event.get("teacher_source") or self.teacher.source
            self.student.status = str(event.get("student_status", self.student.status))
            self.teacher.status = str(event.get("teacher_status", self.teacher.status))
            self.phase = "kernels_ready"
            self.note(elapsed_s, "Both kernels are available")
            self.winner_text = "COMPILING + BENCHMARKING"
            self.winner_style = "bold black on bright_magenta"
            return
        if event_type == "benchmark_started":
            self.phase = "benchmarking"
            self.note(elapsed_s, "Compilation and benchmark started")
            return
        if event_type == "observability":
            name = str(event.get("name", "observability"))
            latency_ms = event.get("latency_ms")
            if latency_ms is None:
                self.note(elapsed_s, f"OBS {name}")
            else:
                self.note(elapsed_s, f"OBS {name} ({latency_ms} ms)")
            return
        if event_type == "final_result":
            self.phase = "complete"
            self.result_class = event.get("result_class")
            self.student.compile_status = "compiled" if event.get("student_valid") else "failed"
            self.teacher.compile_status = "compiled" if event.get("teacher_valid") else "failed"
            self.student.latency_ms = event.get("student_ms")
            self.teacher.latency_ms = event.get("teacher_ms")
            self.training_status = event.get("training_status")
            self.adapter_version = event.get("adapter_version")
            self.note(elapsed_s, f"Final result: {self.result_class}")
            if self.result_class == "student_won":
                self.winner_text = "STUDENT WINS"
                self.winner_style = "bold white on green"
            elif self.result_class == "teacher_won":
                self.winner_text = "TEACHER WINS"
                self.winner_style = "bold white on blue"
            elif self.result_class == "student_valid":
                self.winner_text = "STUDENT COMPILED / TEACHER FAILED"
                self.winner_style = "bold white on green"
            elif self.result_class == "teacher_valid":
                self.winner_text = "TEACHER COMPILED / STUDENT FAILED"
                self.winner_style = "bold white on blue"
            else:
                self.winner_text = "NO VALID KERNEL"
                self.winner_style = "bold white on red"
            return
        self.note(elapsed_s, f"Event: {event_type}")


class KernelDuelCLI:
    def __init__(self, uri: str, task: str, config_id: int, console: Console) -> None:
        self.console = console
        self.state = DuelState(uri=uri, task=task, config_id=config_id)

    def render(self) -> RenderableType:
        layout = Layout(name="root")
        layout.split_column(
            Layout(self._render_header(), name="header", size=5),
            Layout(name="body", ratio=1),
            Layout(self._render_footer(), name="footer", size=3),
        )
        layout["body"].split_row(
            Layout(self._render_sidebar(), name="sidebar", ratio=28),
            Layout(self._render_duel(), name="duel", ratio=72),
        )
        return layout

    def _render_header(self) -> RenderableType:
        title = Text("KERNEL DUEL", justify="center", style="bold white")
        subtitle = Text(
            f"{self.state.task}  |  config {self.state.config_id}  |  {self.state.phase}",
            justify="center",
            style="bold cyan",
        )
        winner = Text(self.state.winner_text, justify="center", style=self.state.winner_style)
        return Panel(
            Group(Align.center(title), Align.center(subtitle), Align.center(winner)),
            box=box.DOUBLE,
            border_style="bright_white",
        )

    def _render_sidebar(self) -> RenderableType:
        info = Table.grid(padding=(0, 1))
        info.add_column(style="bold cyan", no_wrap=True)
        info.add_column(style="white")
        info.add_row("WS", self.state.uri)
        info.add_row("Task", self.state.task)
        info.add_row("Config", str(self.state.config_id))
        if self.state.training_status:
            info.add_row("Training", str(self.state.training_status))
        if self.state.adapter_version:
            info.add_row("Adapter", str(self.state.adapter_version))

        history_text = Text("\n".join(self.state.history[-10:]) or "No prompts yet.", style="white")
        timeline_text = Text("\n".join(self.state.timeline[-16:]) or "No events yet.", style="white")
        if self.state.last_error:
            timeline_text.append(f"\nERROR: {self.state.last_error}", style="bold red")

        return Panel(
            Group(
                Panel(info, title="Session", border_style="cyan"),
                Panel(history_text, title="Chat", border_style="bright_blue"),
                Panel(timeline_text, title="Timeline", border_style="bright_black"),
            ),
            title="Control",
            border_style="white",
            box=box.ROUNDED,
        )

    def _render_duel(self) -> RenderableType:
        inner = Layout()
        inner.split_row(
            Layout(self._render_pane(self.state.student, pane_style="green"), name="student"),
            Layout(self._render_pane(self.state.teacher, pane_style="blue", thinking=self.state.teacher_thinking), name="teacher"),
        )
        return inner

    def _render_pane(self, pane: PaneState, pane_style: str, thinking: str = "") -> RenderableType:
        status = Table.grid(expand=True)
        status.add_column(justify="left", ratio=1)
        status.add_column(justify="right", ratio=1)
        compile_style = "green" if pane.compile_status == "compiled" else ("red" if pane.compile_status == "failed" else "yellow")
        latency_text = f"{pane.latency_ms:.3f} ms" if pane.latency_ms is not None else "pending"
        status.add_row(
            f"[bold]{pane.label.upper()}[/bold]  [{pane_style}]{pane.status}[/{pane_style}]",
            f"[{compile_style}]compile: {pane.compile_status}[/{compile_style}]  [white]latency: {latency_text}[/white]",
        )

        body: RenderableType
        if pane.source:
            body = Syntax(pane.source, "python", line_numbers=True, word_wrap=False, theme="monokai")
        elif thinking:
            body = Text(_clip_tail(thinking, 2500), style="italic bright_black")
        else:
            body = Text("Waiting for kernel...", style="dim")

        details = [status, Rule(style=pane_style), body]
        if pane.error:
            details.extend([Rule(style="red"), Text(pane.error, style="bold red")])

        return Panel(
            Group(*details),
            border_style=pane_style,
            box=box.HEAVY,
        )

    def _render_footer(self) -> RenderableType:
        footer = Text()
        footer.append("Enter an optimization request. ", style="bold white")
        footer.append("Type ", style="white")
        footer.append("/quit", style="bold cyan")
        footer.append(" to exit.", style="white")
        return Panel(footer, border_style="bright_black")

    async def run_prompt(self, prompt: str, live: Live) -> None:
        self.state.reset_for_prompt(prompt)
        live.update(self.render())
        payload = {
            "task": self.state.task,
            "config_id": self.state.config_id,
            "instruction": prompt,
        }
        try:
            async with websockets.connect(
                self.state.uri,
                max_size=2**24,
                ping_interval=None,
                ping_timeout=None,
                open_timeout=20,
                close_timeout=20,
            ) as ws:
                start = time.perf_counter()
                await ws.send(json.dumps(payload))
                while True:
                    event = json.loads(await ws.recv())
                    self.state.ingest(event, time.perf_counter() - start)
                    live.update(self.render())
                    if event.get("type") == "final_result":
                        break
        except Exception as exc:
            self.state.last_error = str(exc)
            self.state.winner_text = "REQUEST FAILED"
            self.state.winner_style = "bold white on red"
            self.state.note(0.0, f"Request failed: {type(exc).__name__}")
            live.update(self.render())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Terminal UI for teacher vs student kernel optimization.")
    parser.add_argument("--uri", default="ws://127.0.0.1:8000/ws/chat")
    parser.add_argument("--task", default="softmax_m4096_n256")
    parser.add_argument("--config-id", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    console = Console()
    app = KernelDuelCLI(args.uri, args.task, args.config_id, console)

    while True:
        console.clear()
        console.print(app.render())
        try:
            prompt = console.input("\n[bold cyan]kernel>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return 0
        if not prompt:
            continue
        if prompt.lower() in {"/quit", "quit", "exit"}:
            return 0
        with Live(app.render(), console=console, refresh_per_second=12, screen=True) as live:
            asyncio.run(app.run_prompt(prompt, live))
        console.print("\n[dim]Press Enter for another round, or type /quit next time.[/dim]")
        try:
            input()
        except EOFError:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
