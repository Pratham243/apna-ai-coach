"""Abstract base classes and data models for exercise rep counting and form checking."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from src.vision.pose import PoseResult


class RepStage(StrEnum):
    """Lifecycle stage of a single exercise repetition."""

    START = "START"
    TRANSITION = "TRANSITION"
    INFLECTION = "INFLECTION"  # Peak tension / bottom / top lockout
    RETURN = "RETURN"
    COMPLETED = "COMPLETED"


@dataclass
class FormIssue:
    """Detected form flaw on a specific frame or during a rep."""

    code: str
    message: str
    severity: str = "warning"  # warning | error | info
    timestamp_ms: int | None = None
    metric_value: float | None = None


@dataclass
class ExerciseMetrics:
    """Per-frame metrics and state produced by an exercise detector."""

    rep_count: int = 0
    stage: RepStage = RepStage.START
    primary_angle: float = 0.0
    secondary_angle: float = 0.0
    active_errors: list[str] = field(default_factory=list)
    rep_errors: list[str] = field(default_factory=list)
    rep_completed_now: bool = False
    debug_info: dict[str, Any] = field(default_factory=dict)


class ExerciseDetector(ABC):
    """Abstract base class for exercise state machine and form checker."""

    name: str = "exercise"
    description: str = ""

    def __init__(self) -> None:
        self.rep_count: int = 0
        self.stage: RepStage = RepStage.START
        self.current_rep_errors: set[str] = set()
        self.history_errors: list[str] = []
        self.frame_count: int = 0

    def reset(self) -> None:
        """Reset internal rep count, stage, and error trackers."""
        self.rep_count = 0
        self.stage = RepStage.START
        self.current_rep_errors.clear()
        self.history_errors.clear()
        self.frame_count = 0

    @abstractmethod
    def process_frame(self, pose: PoseResult, timestamp_ms: int | None = None) -> ExerciseMetrics:
        """Process a single pose result and update rep counting and form checks."""
        raise NotImplementedError

    def get_summary(self) -> dict[str, Any]:
        """Return workout statistics collected so far."""
        return {
            "exercise": self.name,
            "rep_count": self.rep_count,
            "total_frames": self.frame_count,
            "total_errors": len(self.history_errors),
            "error_breakdown": {
                err: self.history_errors.count(err) for err in set(self.history_errors)
            },
        }
