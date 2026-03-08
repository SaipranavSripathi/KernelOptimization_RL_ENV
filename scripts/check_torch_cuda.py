#!/usr/bin/env python3
from __future__ import annotations

import torch


def main() -> None:
    print(f"python: {__import__('sys').executable}")
    print(f"torch: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_device_name: {torch.cuda.get_device_name(0)}")
        print(f"cuda_capability: {torch.cuda.get_device_capability(0)}")


if __name__ == "__main__":
    main()

