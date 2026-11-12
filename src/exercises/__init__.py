"""Exercise detection package."""

from src.exercises.base import ExerciseDetector, ExerciseMetrics, RepStage
from src.exercises.bicep_curl import BicepCurlDetector
from src.exercises.factory import (
    EXERCISE_DISPLAY_NAMES,
    EXERCISE_REGISTRY,
    get_exercise_detector,
    list_supported_exercises,
)
from src.exercises.lunge import LungeDetector
from src.exercises.pushup import PushupDetector
from src.exercises.shoulder_press import ShoulderPressDetector
from src.exercises.squat import SquatDetector

__all__ = [
    "EXERCISE_DISPLAY_NAMES",
    "EXERCISE_REGISTRY",
    "BicepCurlDetector",
    "ExerciseDetector",
    "ExerciseMetrics",
    "LungeDetector",
    "PushupDetector",
    "RepStage",
    "ShoulderPressDetector",
    "SquatDetector",
    "get_exercise_detector",
    "list_supported_exercises",
]
