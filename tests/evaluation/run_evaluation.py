"""Evaluation harness measuring rep count accuracy, form error precision/recall, and latency."""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from src.exercises.factory import get_exercise_detector
from src.vision.pose import PoseDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluator")

EVAL_DIR = Path(__file__).resolve().parent
GROUND_TRUTH_PATH = EVAL_DIR / "ground_truth.json"
TEST_VIDEOS_DIR = EVAL_DIR / "test_videos"
RESULTS_PATH = EVAL_DIR / "results.json"


def evaluate_video(
    video_path: Path,
    exercise_type: str,
    true_rep_count: int,
    true_form_errors: list[dict[str, Any]],
    tolerance_seconds: float = 2.0,
    stride: int = 1,
) -> dict[str, Any]:
    """Run perception-geometry pipeline on a single exercise video and calculate metrics."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    detector = PoseDetector(video_mode=True)
    exercise_detector = get_exercise_detector(exercise_type)

    latencies_ms: list[float] = []
    detected_frames = 0
    processed_frames = 0
    predicted_errors: list[dict[str, Any]] = []

    frame_idx = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if stride > 1 and (frame_idx % stride != 0):
                continue

            processed_frames += 1
            t_sec = frame_idx / fps

            start_t = time.perf_counter()
            pose = detector.detect(frame)
            metrics = exercise_detector.process_frame(pose)
            latency = (time.perf_counter() - start_t) * 1000.0
            latencies_ms.append(latency)

            if pose.detected:
                detected_frames += 1

            if metrics.active_errors:
                for err in metrics.active_errors:
                    predicted_errors.append(
                        {
                            "timestamp_seconds": round(t_sec, 2),
                            "error_type": err,
                        }
                    )
    finally:
        cap.release()
        detector.close()

    # Metrics computation
    predicted_reps = exercise_detector.rep_count
    rep_abs_diff = abs(predicted_reps - true_rep_count)
    rep_error_rate = rep_abs_diff / max(true_rep_count, 1)
    rep_accuracy_pct = max(0.0, 1.0 - rep_error_rate) * 100.0

    keypoint_detection_rate_pct = (
        (detected_frames / processed_frames * 100.0) if processed_frames > 0 else 0.0
    )

    p50_latency_ms = float(np.percentile(latencies_ms, 50)) if latencies_ms else 0.0
    p95_latency_ms = float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0

    # Cluster predicted errors to events (debounce 1.5s)
    clustered_preds: list[dict[str, Any]] = []
    for pred in predicted_errors:
        if not clustered_preds:
            clustered_preds.append(pred)
        else:
            last = clustered_preds[-1]
            if (
                pred["timestamp_seconds"] - last["timestamp_seconds"] > 1.5
                or pred["error_type"] != last["error_type"]
            ):
                clustered_preds.append(pred)

    # Precision & Recall for form errors
    tp = 0
    matched_preds = set()
    for gt in true_form_errors:
        gt_t = gt["timestamp_seconds"]
        for idx, pred in enumerate(clustered_preds):
            if idx in matched_preds:
                continue
            if abs(pred["timestamp_seconds"] - gt_t) <= tolerance_seconds:
                tp += 1
                matched_preds.add(idx)
                break

    fp = len(clustered_preds) - len(matched_preds)
    fn = len(true_form_errors) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if not true_form_errors else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "video_filename": video_path.name,
        "exercise_type": exercise_type,
        "total_video_frames": total_frames,
        "processed_frames": processed_frames,
        "pose_detected_frames": detected_frames,
        "keypoint_detection_rate_pct": round(keypoint_detection_rate_pct, 2),
        "true_rep_count": true_rep_count,
        "predicted_rep_count": predicted_reps,
        "rep_count_error_rate": round(rep_error_rate, 4),
        "rep_count_accuracy_pct": round(rep_accuracy_pct, 2),
        "form_detection": {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
        },
        "latency_ms": {
            "p50": round(p50_latency_ms, 2),
            "p95": round(p95_latency_ms, 2),
            "mean": round(float(np.mean(latencies_ms)), 2) if latencies_ms else 0.0,
        },
    }


def run_benchmark(stride: int = 1) -> dict[str, Any]:
    """Run full benchmark across all videos defined in ground_truth.json."""
    if not GROUND_TRUTH_PATH.exists():
        raise FileNotFoundError(f"Missing ground_truth.json at {GROUND_TRUTH_PATH}")

    with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
        ground_truth = json.load(f)

    logger.info("Starting benchmark across %d exercise videos...", len(ground_truth))
    per_video_results: list[dict[str, Any]] = []

    for item in ground_truth:
        video_filename = item["video_filename"]
        video_path = TEST_VIDEOS_DIR / video_filename
        if not video_path.exists():
            logger.warning("Video file not found: %s. Skipping.", video_path)
            continue

        logger.info("Evaluating %s (%s)...", video_filename, item["exercise_type"])
        res = evaluate_video(
            video_path=video_path,
            exercise_type=item["exercise_type"],
            true_rep_count=item["true_rep_count"],
            true_form_errors=item.get("true_form_errors", []),
            stride=stride,
        )
        per_video_results.append(res)
        logger.info(
            "--> %s: True=%d, Pred=%d, Rep Acc=%.1f%%, Pose Det=%.1f%%, p50=%.1fms",
            video_filename,
            res["true_rep_count"],
            res["predicted_rep_count"],
            res["rep_count_accuracy_pct"],
            res["keypoint_detection_rate_pct"],
            res["latency_ms"]["p50"],
        )

    if not per_video_results:
        raise RuntimeError("No videos were successfully evaluated.")

    # Aggregate benchmark summary
    mean_rep_accuracy = float(np.mean([r["rep_count_accuracy_pct"] for r in per_video_results]))
    mean_detection_rate = float(
        np.mean([r["keypoint_detection_rate_pct"] for r in per_video_results])
    )
    mean_p50_latency = float(np.mean([r["latency_ms"]["p50"] for r in per_video_results]))
    mean_p95_latency = float(np.mean([r["latency_ms"]["p95"] for r in per_video_results]))
    mean_precision = float(np.mean([r["form_detection"]["precision"] for r in per_video_results]))
    mean_recall = float(np.mean([r["form_detection"]["recall"] for r in per_video_results]))
    mean_f1 = float(np.mean([r["form_detection"]["f1_score"] for r in per_video_results]))

    summary = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "videos_evaluated": len(per_video_results),
        "total_frames_processed": sum(r["processed_frames"] for r in per_video_results),
        "overall_metrics": {
            "mean_rep_accuracy_pct": round(mean_rep_accuracy, 2),
            "mean_keypoint_detection_rate_pct": round(mean_detection_rate, 2),
            "mean_form_precision": round(mean_precision, 4),
            "mean_form_recall": round(mean_recall, 4),
            "mean_form_f1": round(mean_f1, 4),
            "latency_ms": {
                "mean_p50": round(mean_p50_latency, 2),
                "mean_p95": round(mean_p95_latency, 2),
            },
        },
        "per_video_results": per_video_results,
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Evaluation complete! Results saved to %s", RESULTS_PATH)
    return summary


if __name__ == "__main__":
    run_benchmark(stride=1)
