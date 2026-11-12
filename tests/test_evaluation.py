"""Tests validating the evaluation harness schema and results."""

import json
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent / "evaluation"
RESULTS_PATH = EVAL_DIR / "results.json"
GROUND_TRUTH_PATH = EVAL_DIR / "ground_truth.json"


def test_ground_truth_exists_and_valid():
    assert GROUND_TRUTH_PATH.exists()
    with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
        gt = json.load(f)
    assert len(gt) == 5
    for item in gt:
        assert "video_filename" in item
        assert "exercise_type" in item
        assert "true_rep_count" in item
        assert "true_form_errors" in item
        assert item["true_rep_count"] > 0


def test_results_json_schema():
    assert RESULTS_PATH.exists()
    with open(RESULTS_PATH, encoding="utf-8") as f:
        results = json.load(f)

    assert results["videos_evaluated"] == 5
    assert results["total_frames_processed"] > 1000
    metrics = results["overall_metrics"]
    assert "mean_rep_accuracy_pct" in metrics
    assert "mean_keypoint_detection_rate_pct" in metrics
    assert "latency_ms" in metrics
    assert metrics["mean_rep_accuracy_pct"] > 70.0
    assert metrics["mean_keypoint_detection_rate_pct"] > 75.0

    per_vid = results["per_video_results"]
    assert len(per_vid) == 5
    exercise_types = {v["exercise_type"] for v in per_vid}
    assert exercise_types == {"squats", "pushups", "bicep_curls", "lunges", "shoulder_press"}
