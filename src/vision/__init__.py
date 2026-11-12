"""Computer vision, pose landmark detection, joint geometry, and HUD overlays."""

from src.vision.geometry import (
    calculate_angle,
    calculate_vertical_angle,
    euclidean_distance,
    midpoint,
)
from src.vision.overlay import annotate_frame, draw_hud
from src.vision.pose import LandmarkPoint, PoseDetector, PoseLandmark, PoseResult

__all__ = [
    "LandmarkPoint",
    "PoseDetector",
    "PoseLandmark",
    "PoseResult",
    "annotate_frame",
    "calculate_angle",
    "calculate_vertical_angle",
    "draw_hud",
    "euclidean_distance",
    "midpoint",
]
