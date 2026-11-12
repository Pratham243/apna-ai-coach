"""Utility script to generate an animated demo GIF showing skeleton overlay and HUD metrics."""

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
from PIL import Image

from src.exercises.factory import get_exercise_detector
from src.vision.overlay import annotate_frame
from src.vision.pose import PoseDetector
VIDEO_PATH = ROOT_DIR / "tests" / "evaluation" / "test_videos" / "squats.mp4"
ASSETS_DIR = ROOT_DIR / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_GIF_PATH = ASSETS_DIR / "demo.gif"


def generate_gif(max_gif_frames: int = 90, sample_stride: int = 2) -> None:
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Missing video at {VIDEO_PATH}")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    detector = PoseDetector(video_mode=True)
    ed = get_exercise_detector("squats")

    frames: list[Image.Image] = []
    idx = 0

    try:
        while cap.isOpened() and len(frames) < max_gif_frames:
            ret, frame = cap.read()
            if not ret:
                break
            idx += 1
            # Skip intro stationary segment
            if idx < 75:
                continue
            if idx % sample_stride != 0:
                continue

            pose = detector.detect(frame)
            metrics = ed.process_frame(pose)
            annotated = annotate_frame(frame, pose, "squats", metrics, fps=25.0)

            # Resize to 540x304 for crisp 16:9 format
            resized = cv2.resize(annotated, (540, 304), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
    finally:
        cap.release()
        detector.close()

    if frames:
        frames[0].save(
            str(OUTPUT_GIF_PATH),
            save_all=True,
            append_images=frames[1:],
            duration=80,
            loop=0,
            optimize=True,
        )
        size_kb = os.path.getsize(OUTPUT_GIF_PATH) / 1024.0
        print(f"Generated {OUTPUT_GIF_PATH} with {len(frames)} frames ({size_kb:.1f} KB)")


if __name__ == "__main__":
    generate_gif()
