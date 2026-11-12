"""Visual overlay rendering module: drawing HUD, angles, and error banners onto video frames."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from src.vision.pose import PoseDetector, PoseResult

if TYPE_CHECKING:
    from src.exercises.base import ExerciseMetrics


def draw_hud(
    frame_bgr: np.ndarray,
    exercise_name: str,
    metrics: ExerciseMetrics,
    fps: float | None = None,
) -> np.ndarray:
    """Render heads-up display overlay with rep counter, stage, angles, and error alerts."""
    h, w = frame_bgr.shape[:2]
    canvas = frame_bgr.copy()

    # Semi-transparent top banner for exercise and rep counter
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, 75), (20, 24, 30), -1)

    # Semi-transparent bottom banner for active form warnings
    has_errors = bool(metrics.active_errors)
    bottom_height = 55 if has_errors else 35
    bottom_color = (15, 20, 110) if has_errors else (20, 24, 30)  # Red tint if errors
    cv2.rectangle(overlay, (0, h - bottom_height), (w, h), bottom_color, -1)

    # Blend top and bottom banners
    cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

    # Left: Exercise name and Stage
    clean_exercise_name = exercise_name.replace("_", " ").title()
    cv2.putText(
        canvas,
        f"{clean_exercise_name}",
        (18, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"Stage: {metrics.stage.value}",
        (18, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (160, 220, 255),
        1,
        cv2.LINE_AA,
    )

    # Center-Right: Primary & Secondary Angles
    angle_text = f"Angle 1: {metrics.primary_angle:.0f} deg"
    if metrics.secondary_angle > 0.0:
        angle_text += f" | Angle 2: {metrics.secondary_angle:.0f} deg"
    cv2.putText(
        canvas,
        angle_text,
        (w // 2 - 100, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 230, 200),
        1,
        cv2.LINE_AA,
    )

    # Far Right: Big Rep Counter
    rep_badge_text = f"{metrics.rep_count}"
    rep_label = "REPS"
    cv2.putText(
        canvas,
        rep_label,
        (w - 110, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        rep_badge_text,
        (w - 95, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (0, 230, 140),
        3,
        cv2.LINE_AA,
    )

    # Optional FPS indicator
    if fps is not None and fps > 0:
        cv2.putText(
            canvas,
            f"{fps:.1f} fps",
            (w - 65, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (140, 140, 140),
            1,
            cv2.LINE_AA,
        )

    # Bottom Error Banner
    if has_errors:
        err_msg = " ! " + " | ".join(metrics.active_errors)
        cv2.putText(
            canvas,
            err_msg[:95],
            (15, h - 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (80, 120, 255),
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            canvas,
            "Form Clean - Keep Going",
            (15, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (100, 220, 120),
            1,
            cv2.LINE_AA,
        )

    return canvas


def annotate_frame(
    frame_bgr: np.ndarray,
    pose: PoseResult,
    exercise_name: str,
    metrics: ExerciseMetrics,
    fps: float | None = None,
) -> np.ndarray:
    """Full frame annotation pipeline: draws skeleton overlay then renders HUD."""
    with_skeleton = PoseDetector.draw_skeleton(
        frame_bgr,
        pose,
        form_errors=metrics.active_errors,
    )
    return draw_hud(with_skeleton, exercise_name, metrics, fps=fps)
