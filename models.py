from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence


@dataclass
class Action:
    config_id: int


@dataclass
class StepResult:
    observation: Dict[str, Any]
    reward: float
    done: bool
    state: Dict[str, Any]
    info: Dict[str, Any]


@dataclass
class ResetResult:
    observation: Dict[str, Any]
    reward: float
    done: bool
    state: Dict[str, Any]
    info: Dict[str, Any]
