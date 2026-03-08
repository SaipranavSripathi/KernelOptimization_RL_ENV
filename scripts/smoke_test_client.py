#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from client import SoftmaxSurrogateEnvClient


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="surrogate", choices=("surrogate", "default", "self_improving", "generative"))
    parser.add_argument("--live-bench", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = SoftmaxSurrogateEnvClient(mode=args.mode)
    if args.mode != "surrogate" and client._local_env is not None and not args.live_bench:
        client._local_env._live_bench = False
    reset_out = client.reset()
    if args.mode != "surrogate" and client._local_env is not None:
        step_out = client.step(
            {
                "config_id": 0,
                "source": client._local_env.current_kernel_source,
            }
        )
    else:
        step_out = client.step({"config_id": 0})
    summary = {"reset": reset_out, "step": step_out}
    out = Path("outputs/smoke_test_client.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
