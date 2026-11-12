"""MediaPipe Pose Landmarker wrapper and OpenCV skeleton overlay renderer."""

from __future__ import annotations

import os
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options

DEFAULT_POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "pose_landmarker_lite.task"


class PoseLandmark(IntEnum):
    """Indices for the 33 MediaPipe Pose Landmarker keypoints."""

    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


@dataclass(frozen=True)
class LandmarkPoint:
    """Single normalized 3D pose landmark with visibility score."""

    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0


@dataclass
class PoseResult:
    """Output of pose detection for a single video frame."""

    landmarks: list[LandmarkPoint] = field(default_factory=list)
    detected: bool = False

    def get(self, idx: int) -> LandmarkPoint | None:
        """Return landmark at `idx` if present."""
        if 0 <= idx < len(self.landmarks):
            return self.landmarks[idx]
        return None


def ensure_pose_model(model_path: Path | None = None) -> Path:
    """Ensure the MediaPipe Pose Landmarker .task file exists locally, downloading if missing."""
    env_path = os.getenv("POSE_MODEL_PATH")
    target = Path(env_path) if env_path else (model_path or DEFAULT_MODEL_PATH)
    if target.exists() and target.stat().st_size > 0:
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(".task.tmp")
    urllib.request.urlretrieve(DEFAULT_POSE_MODEL_URL, tmp_path)
    tmp_path.replace(target)
    return target


class PoseDetector:
    """Local MediaPipe Pose Landmarker detector for 33 body keypoints."""

    def __init__(
        self,
        model_path: Path | None = None,
        video_mode: bool = True,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        resolved_model = ensure_pose_model(model_path)
        self._video_mode = video_mode
        self._last_timestamp_ms = -1

        running_mode = vision.RunningMode.VIDEO if video_mode else vision.RunningMode.IMAGE
        options = vision.PoseLandmarkerOptions(
            base_options=base_options.BaseOptions(model_asset_path=str(resolved_model)),
            running_mode=running_mode,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int | None = None) -> PoseResult:
        """Detect 33 pose landmarks in a BGR OpenCV frame."""
        if frame_bgr is None or frame_bgr.size == 0:
            return PoseResult(landmarks=[], detected=False)

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if self._video_mode:
            if timestamp_ms is None or timestamp_ms <= self._last_timestamp_ms:
                timestamp_ms = self._last_timestamp_ms + 33
            self._last_timestamp_ms = timestamp_ms
            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        else:
            result = self._landmarker.detect(mp_image)

        if not result.pose_landmarks:
            return PoseResult(landmarks=[], detected=False)

        raw = result.pose_landmarks[0]
        points = [
            LandmarkPoint(
                x=float(lm.x),
                y=float(lm.y),
                z=float(lm.z),
                visibility=float(getattr(lm, "visibility", 1.0) or 0.0),
            )
            for lm in raw
        ]
        return PoseResult(landmarks=points, detected=len(points) == 33)

    @staticmethod
    def draw_skeleton(
        frame_bgr: np.ndarray,
        pose: PoseResult,
        form_errors: Iterable[str] | None = None,
        min_visibility: float = 0.35,
    ) -> np.ndarray:
        """Draw pose skeleton and keypoints onto a copy of `frame_bgr`."""
        annotated = frame_bgr.copy()
        if not pose.detected or len(pose.landmarks) < 33:
            return annotated

        h, w = annotated.shape[:2]
        has_errors = bool(list(form_errors)) if form_errors is not None else False
        line_color = (0, 140, 255) if has_errors else (0, 220, 120)
        joint_color = (60, 80, 255) if has_errors else (255, 255, 255)

        for conn in vision.PoseLandmarksConnections.POSE_LANDMARKS:
            p1 = pose.landmarks[conn.start]
            p2 = pose.landmarks[conn.end]
            if p1.visibility < min_visibility or p2.visibility < min_visibility:
                continue
            pt1 = (int(np.clip(p1.x, 0.0, 1.0) * w), int(np.clip(p1.y, 0.0, 1.0) * h))
            pt2 = (int(np.clip(p2.x, 0.0, 1.0) * w), int(np.clip(p2.y, 0.0, 1.0) * h))
            cv2.line(annotated, pt1, pt2, line_color, 2, cv2.LINE_AA)

        for lm in pose.landmarks:
            if lm.visibility < min_visibility:
                continue
            cx = int(np.clip(lm.x, 0.0, 1.0) * w)
            cy = int(np.clip(lm.y, 0.0, 1.0) * h)
            cv2.circle(annotated, (cx, cy), 4, joint_color, -1, cv2.LINE_AA)

        return annotated

    def close(self) -> None:
        """Release underlying MediaPipe resources."""
        if getattr(self, "_landmarker", None) is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self) -> PoseDetector:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
