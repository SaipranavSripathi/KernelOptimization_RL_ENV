from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Optional

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from server.softmax_surrogate_environment import DEFAULT_BUDGET, SoftmaxSurrogateEnvironment


class SoftmaxSurrogateEnvClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        measurement_path: str = "data/autotune_measurements.csv",
        budget: int = DEFAULT_BUDGET,
        seed: int = 0,
        mode: Optional[str] = None,
    ) -> None:
        self.base_url = base_url
        self._local_env = None
        self.mode = mode or os.environ.get("KERNEL_ENV_MODE", "surrogate")
        if base_url is None:
            self._local_env = SoftmaxSurrogateEnvironment(
                measurement_path=measurement_path,
                budget=budget,
                seed=seed,
                mode=self.mode,
            )

    def reset(self, task: Optional[str] = None, seed: Optional[int] = None) -> dict:
        if self._local_env is not None:
            return self._local_env.reset(task=task, seed=seed)
        payload = {}
        if task is not None:
            payload["task"] = task
        if seed is not None:
            payload["seed"] = seed
        resp = requests.post(f"{self.base_url}/reset", json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()

    def step(self, action: Any) -> dict:
        if self._local_env is not None:
            return self._local_env.step(action)
        payload = action if isinstance(action, dict) else {"x": action}
        resp = requests.post(f"{self.base_url}/step", json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()

    def state(self) -> dict:
        if self._local_env is not None:
            return self._local_env.state()
        resp = requests.get(f"{self.base_url}/state", timeout=60)
        resp.raise_for_status()
        return resp.json()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", default=None, help="Optional base URL (e.g. http://127.0.0.1:8000)")
    parser.add_argument("--task", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", default=None, choices=("surrogate", "default", "self_improving", "generative"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = SoftmaxSurrogateEnvClient(base_url=args.remote, seed=args.seed, mode=args.mode)
    print(client.reset(task=args.task))


if __name__ == "__main__":
    main()
