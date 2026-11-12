"""Factory functions and registry for exercise detectors."""

from __future__ import annotations

from src.exercises.base import ExerciseDetector
from src.exercises.bicep_curl import BicepCurlDetector
from src.exercises.lunge import LungeDetector
from src.exercises.pushup import PushupDetector
from src.exercises.shoulder_press import ShoulderPressDetector
from src.exercises.squat import SquatDetector

EXERCISE_REGISTRY: dict[str, type[ExerciseDetector]] = {
    "squats": SquatDetector,
    "pushups": PushupDetector,
    "bicep_curls": BicepCurlDetector,
    "lunges": LungeDetector,
    "shoulder_press": ShoulderPressDetector,
}

EXERCISE_DISPLAY_NAMES: dict[str, str] = {
    "squats": "Squats",
    "pushups": "Push-ups",
    "bicep_curls": "Bicep Curls",
    "lunges": "Lunges",
    "shoulder_press": "Shoulder Press",
}


def get_exercise_detector(exercise_key: str) -> ExerciseDetector:
    """Instantiate an ExerciseDetector by exercise key."""
    normalized = exercise_key.lower().strip().replace(" ", "_").replace("-", "_")
    if normalized not in EXERCISE_REGISTRY:
        options = list(EXERCISE_REGISTRY.keys())
        raise ValueError(f"Unknown exercise '{exercise_key}'. Available exercises: {options}")
    return EXERCISE_REGISTRY[normalized]()


def list_supported_exercises() -> list[str]:
    """Return keys of all supported exercises."""
    return list(EXERCISE_REGISTRY.keys())
