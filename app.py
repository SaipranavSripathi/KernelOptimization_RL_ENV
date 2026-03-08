#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import gradio as gr

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from server.softmax_surrogate_environment import SoftmaxSurrogateEnvironment


_env: Optional[SoftmaxSurrogateEnvironment] = None


def get_env() -> SoftmaxSurrogateEnvironment:
    global _env
    if _env is None:
        mode = os.environ.get("KERNEL_ENV_MODE", "surrogate").strip().lower()
        _env = SoftmaxSurrogateEnvironment(mode=mode)
    return _env


def format_json(data):
    return json.dumps(data, indent=2)


def list_tasks():
    env = get_env()
    tasks = env.available_tasks()
    return "\n".join(tasks) if tasks else "No tasks available"


def list_configs():
    env = get_env()
    try:
        configs = env.available_configs()
        return json.dumps(configs, indent=2)
    except RuntimeError as e:
        return f"Error: {str(e)}. Call reset first."


def reset_environment(task_name: Optional[str], seed: int = 0):
    env = get_env()
    try:
        result = env.reset(task=task_name if task_name else None, seed=int(seed))
        return format_json(result)
    except Exception as e:
        return f"Error: {str(e)}"


def step_environment(config_id: int, source: Optional[str]):
    env = get_env()
    try:
        payload = {"config_id": int(config_id)}
        if source and source.strip():
            payload["source"] = source
        result = env.step(payload)
        return format_json(result)
    except Exception as e:
        return f"Error: {str(e)}"


def get_state():
    env = get_env()
    try:
        return format_json(env.state())
    except Exception as e:
        return f"Error: {str(e)}"


def get_diagnostics():
    env = get_env()
    try:
        return format_json(env.diagnostics())
    except Exception as e:
        return f"Error: {str(e)}"


def oracle_best():
    env = get_env()
    try:
        return format_json(env.oracle_best())
    except Exception as e:
        return f"Error: {str(e)}"


def get_mode():
    env = get_env()
    return env._mode


with gr.Blocks(title="RL Surrogate Kernel Autotuning", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# RL Surrogate Kernel Autotuning Environment")
    gr.Markdown("""
    This is a reinforcement learning environment for GPU kernel autotuning.
    - **Mode**: Surrogate (uses pre-collected measurements)
    - **Families**: softmax, layernorm, grouped_gemm
    
    Use the tabs below to interact with the environment.
    """)

    mode_display = gr.Markdown()
    
    with gr.Tab("Available Tasks"):
        gr.Markdown("List of available kernel tasks:")
        task_list = gr.Textbox(label="Tasks", lines=15, interactive=False)
        with gr.Row():
            gr.Button("Refresh Tasks").click(list_tasks, outputs=task_list)

    with gr.Tab("Reset"):
        gr.Markdown("Initialize the environment with a specific task.")
        with gr.Row():
            task_input = gr.Textbox(
                label="Task Name", 
                placeholder="softmax_m4096_n256",
                info="Leave empty for random task"
            )
            seed_input = gr.Number(label="Seed", value=0, info="Random seed for reproducibility")
        with gr.Row():
            reset_btn = gr.Button("Reset Environment", variant="primary")
        reset_output = gr.Textbox(label="Result", lines=20, interactive=False)
        reset_btn.click(reset_environment, inputs=[task_input, seed_input], outputs=reset_output)

    with gr.Tab("Available Configs"):
        gr.Markdown("View available configurations for the current task (after reset).")
        gr.Button("Show Configs").click(list_configs, outputs=gr.Textbox(label="Configs", lines=20, interactive=False))

    with gr.Tab("Step"):
        gr.Markdown("""
        Take a step in the environment.
        - **Config ID**: The Triton configuration to benchmark
        - **Source**: Optional kernel source code (for optimization)
        """)
        with gr.Row():
            config_input = gr.Number(label="Config ID", value=0, info="Configuration to evaluate")
            source_input = gr.Textbox(
                label="Kernel Source (optional)", 
                lines=5,
                placeholder="Provide kernel source code to optimize...",
                info="Leave empty to use current kernel"
            )
        with gr.Row():
            step_btn = gr.Button("Take Step", variant="primary")
        step_output = gr.Textbox(label="Result", lines=20, interactive=False)
        step_btn.click(step_environment, inputs=[config_input, source_input], outputs=step_output)

    with gr.Tab("State"):
        gr.Markdown("Get current environment state.")
        state_btn = gr.Button("Get State", variant="primary")
        state_output = gr.Textbox(label="State", lines=20, interactive=False)
        state_btn.click(get_state, outputs=state_output)

    with gr.Tab("Diagnostics"):
        gr.Markdown("Get detailed diagnostics about the environment.")
        diag_btn = gr.Button("Get Diagnostics", variant="primary")
        diag_output = gr.Textbox(label="Diagnostics", lines=30, interactive=False)
        diag_btn.click(get_diagnostics, outputs=diag_output)

    with gr.Tab("Oracle Best"):
        gr.Markdown("Get the best configuration from the oracle (for comparison).")
        oracle_btn = gr.Button("Get Oracle Best", variant="primary")
        oracle_output = gr.Textbox(label="Oracle Best", lines=15, interactive=False)
        oracle_btn.click(oracle_best, outputs=oracle_output)

    demo.load(lambda: f"**Current Mode**: `{get_env()._mode}`", outputs=mode_display)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )
