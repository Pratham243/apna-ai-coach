"""Unit tests for all 5 exercise detectors."""

import pytest

from src.exercises.base import RepStage
from src.exercises.bicep_curl import BicepCurlDetector
from src.exercises.factory import get_exercise_detector, list_supported_exercises
from src.exercises.lunge import LungeDetector
from src.exercises.pushup import PushupDetector
from src.exercises.shoulder_press import ShoulderPressDetector
from src.exercises.squat import SquatDetector
from src.vision.pose import LandmarkPoint, PoseLandmark, PoseResult


def make_dummy_landmarks() -> list[LandmarkPoint]:
    """Create a default 33-landmark skeleton."""
    return [LandmarkPoint(x=0.5, y=0.5, z=0.0, visibility=0.9) for _ in range(33)]


def test_list_supported_exercises():
    exercises = list_supported_exercises()
    assert "squats" in exercises
    assert "pushups" in exercises
    assert "bicep_curls" in exercises
    assert "lunges" in exercises
    assert "shoulder_press" in exercises


def test_get_exercise_detector():
    squat = get_exercise_detector("squats")
    assert isinstance(squat, SquatDetector)
    pushup = get_exercise_detector("pushups")
    assert isinstance(pushup, PushupDetector)
    curl = get_exercise_detector("bicep_curls")
    assert isinstance(curl, BicepCurlDetector)
    with pytest.raises(ValueError):
        get_exercise_detector("invalid_exercise")


def test_squat_rep_cycle():
    detector = SquatDetector()
    landmarks = make_dummy_landmarks()

    # Standing: hip at (0.5, 0.4), knee at (0.5, 0.7), ankle at (0.5, 0.95) -> 180 deg
    landmarks[PoseLandmark.LEFT_HIP] = LandmarkPoint(x=0.5, y=0.4)
    landmarks[PoseLandmark.LEFT_KNEE] = LandmarkPoint(x=0.5, y=0.7)
    landmarks[PoseLandmark.LEFT_ANKLE] = LandmarkPoint(x=0.5, y=0.95)
    landmarks[PoseLandmark.LEFT_SHOULDER] = LandmarkPoint(x=0.5, y=0.15)
    pose = PoseResult(landmarks=landmarks, detected=True)

    m = detector.process_frame(pose)
    assert m.rep_count == 0
    assert m.stage == RepStage.START

    # Bottom of squat: knee bent to 80 degrees
    # hip=(0.4, 0.65), knee=(0.6, 0.65), ankle=(0.6, 0.95)
    landmarks[PoseLandmark.LEFT_HIP] = LandmarkPoint(x=0.35, y=0.65)
    landmarks[PoseLandmark.LEFT_KNEE] = LandmarkPoint(x=0.6, y=0.65)
    landmarks[PoseLandmark.LEFT_ANKLE] = LandmarkPoint(x=0.6, y=0.95)
    pose_bottom = PoseResult(landmarks=landmarks, detected=True)

    m = detector.process_frame(pose_bottom)
    assert m.stage in (RepStage.TRANSITION, RepStage.INFLECTION)

    # Return to standing
    landmarks[PoseLandmark.LEFT_HIP] = LandmarkPoint(x=0.5, y=0.4)
    landmarks[PoseLandmark.LEFT_KNEE] = LandmarkPoint(x=0.5, y=0.7)
    landmarks[PoseLandmark.LEFT_ANKLE] = LandmarkPoint(x=0.5, y=0.95)
    pose_stand = PoseResult(landmarks=landmarks, detected=True)

    m = detector.process_frame(pose_stand)
    assert m.rep_count == 1
    assert m.rep_completed_now is True
    assert m.stage == RepStage.START


def test_squat_knees_caving_detected():
    detector = SquatDetector()
    landmarks = make_dummy_landmarks()

    # In squat depth with knees caving inward
    landmarks[PoseLandmark.LEFT_HIP] = LandmarkPoint(x=0.4, y=0.55)
    landmarks[PoseLandmark.RIGHT_HIP] = LandmarkPoint(x=0.6, y=0.55)
    # Knees closer together than ankles
    landmarks[PoseLandmark.LEFT_KNEE] = LandmarkPoint(x=0.48, y=0.7)
    landmarks[PoseLandmark.RIGHT_KNEE] = LandmarkPoint(x=0.52, y=0.7)
    landmarks[PoseLandmark.LEFT_ANKLE] = LandmarkPoint(x=0.35, y=0.95)
    landmarks[PoseLandmark.RIGHT_ANKLE] = LandmarkPoint(x=0.65, y=0.95)

    pose = PoseResult(landmarks=landmarks, detected=True)
    m = detector.process_frame(pose)
    assert any("caving" in err.lower() for err in m.active_errors)


def test_pushup_rep_and_sagging():
    detector = PushupDetector()
    landmarks = make_dummy_landmarks()

    # Top plank position: elbow 180
    landmarks[PoseLandmark.LEFT_SHOULDER] = LandmarkPoint(x=0.3, y=0.4)
    landmarks[PoseLandmark.LEFT_ELBOW] = LandmarkPoint(x=0.3, y=0.6)
    landmarks[PoseLandmark.LEFT_WRIST] = LandmarkPoint(x=0.3, y=0.8)
    landmarks[PoseLandmark.LEFT_HIP] = LandmarkPoint(x=0.5, y=0.45)
    landmarks[PoseLandmark.LEFT_ANKLE] = LandmarkPoint(x=0.8, y=0.5)

    pose = PoseResult(landmarks=landmarks, detected=True)
    m = detector.process_frame(pose)
    assert m.rep_count == 0

    # Bottom pushup: elbow 80 deg
    landmarks[PoseLandmark.LEFT_SHOULDER] = LandmarkPoint(x=0.3, y=0.7)
    landmarks[PoseLandmark.LEFT_ELBOW] = LandmarkPoint(x=0.2, y=0.6)
    landmarks[PoseLandmark.LEFT_WRIST] = LandmarkPoint(x=0.3, y=0.8)
    pose_down = PoseResult(landmarks=landmarks, detected=True)
    m = detector.process_frame(pose_down)
    assert m.stage in (RepStage.TRANSITION, RepStage.INFLECTION)

    # Return up
    landmarks[PoseLandmark.LEFT_SHOULDER] = LandmarkPoint(x=0.3, y=0.4)
    landmarks[PoseLandmark.LEFT_ELBOW] = LandmarkPoint(x=0.3, y=0.6)
    landmarks[PoseLandmark.LEFT_WRIST] = LandmarkPoint(x=0.3, y=0.8)
    pose_up = PoseResult(landmarks=landmarks, detected=True)
    m = detector.process_frame(pose_up)
    assert m.rep_count == 1
    assert m.rep_completed_now is True


def test_bicep_curl_rep():
    detector = BicepCurlDetector()
    landmarks = make_dummy_landmarks()

    # Arm straight down: elbow 180
    landmarks[PoseLandmark.LEFT_SHOULDER] = LandmarkPoint(x=0.5, y=0.3)
    landmarks[PoseLandmark.LEFT_ELBOW] = LandmarkPoint(x=0.5, y=0.5)
    landmarks[PoseLandmark.LEFT_WRIST] = LandmarkPoint(x=0.5, y=0.75)
    landmarks[PoseLandmark.LEFT_HIP] = LandmarkPoint(x=0.5, y=0.6)

    m = detector.process_frame(PoseResult(landmarks=landmarks, detected=True))
    assert m.rep_count == 0

    # Curled to 40 deg
    landmarks[PoseLandmark.LEFT_WRIST] = LandmarkPoint(x=0.48, y=0.35)
    m = detector.process_frame(PoseResult(landmarks=landmarks, detected=True))
    assert m.stage in (RepStage.TRANSITION, RepStage.INFLECTION)

    # Back down
    landmarks[PoseLandmark.LEFT_WRIST] = LandmarkPoint(x=0.5, y=0.75)
    m = detector.process_frame(PoseResult(landmarks=landmarks, detected=True))
    assert m.rep_count == 1
    assert m.rep_completed_now is True


def test_shoulder_press_rep():
    detector = ShoulderPressDetector()
    landmarks = make_dummy_landmarks()

    # Bottom rack position: elbows ~80 deg, wrists near shoulders
    landmarks[PoseLandmark.LEFT_SHOULDER] = LandmarkPoint(x=0.4, y=0.4)
    landmarks[PoseLandmark.RIGHT_SHOULDER] = LandmarkPoint(x=0.6, y=0.4)
    landmarks[PoseLandmark.LEFT_ELBOW] = LandmarkPoint(x=0.35, y=0.5)
    landmarks[PoseLandmark.RIGHT_ELBOW] = LandmarkPoint(x=0.65, y=0.5)
    landmarks[PoseLandmark.LEFT_WRIST] = LandmarkPoint(x=0.38, y=0.38)
    landmarks[PoseLandmark.RIGHT_WRIST] = LandmarkPoint(x=0.62, y=0.38)

    m = detector.process_frame(PoseResult(landmarks=landmarks, detected=True))
    assert m.rep_count == 0

    # Top lockout: arms straight overhead
    landmarks[PoseLandmark.LEFT_ELBOW] = LandmarkPoint(x=0.4, y=0.25)
    landmarks[PoseLandmark.RIGHT_ELBOW] = LandmarkPoint(x=0.6, y=0.25)
    landmarks[PoseLandmark.LEFT_WRIST] = LandmarkPoint(x=0.4, y=0.1)
    landmarks[PoseLandmark.RIGHT_WRIST] = LandmarkPoint(x=0.6, y=0.1)

    m = detector.process_frame(PoseResult(landmarks=landmarks, detected=True))
    assert m.stage in (RepStage.TRANSITION, RepStage.INFLECTION)

    # Lower back to start
    landmarks[PoseLandmark.LEFT_ELBOW] = LandmarkPoint(x=0.35, y=0.5)
    landmarks[PoseLandmark.RIGHT_ELBOW] = LandmarkPoint(x=0.65, y=0.5)
    landmarks[PoseLandmark.LEFT_WRIST] = LandmarkPoint(x=0.38, y=0.38)
    landmarks[PoseLandmark.RIGHT_WRIST] = LandmarkPoint(x=0.62, y=0.38)

    m = detector.process_frame(PoseResult(landmarks=landmarks, detected=True))
    assert m.rep_count == 1
    assert m.rep_completed_now is True


def test_lunge_rep():
    detector = LungeDetector()
    landmarks = make_dummy_landmarks()

    landmarks[PoseLandmark.LEFT_HIP] = LandmarkPoint(x=0.45, y=0.4)
    landmarks[PoseLandmark.LEFT_KNEE] = LandmarkPoint(x=0.45, y=0.65)
    landmarks[PoseLandmark.LEFT_ANKLE] = LandmarkPoint(x=0.45, y=0.9)
    landmarks[PoseLandmark.RIGHT_HIP] = LandmarkPoint(x=0.55, y=0.4)
    landmarks[PoseLandmark.RIGHT_KNEE] = LandmarkPoint(x=0.55, y=0.65)
    landmarks[PoseLandmark.RIGHT_ANKLE] = LandmarkPoint(x=0.55, y=0.9)

    m = detector.process_frame(PoseResult(landmarks=landmarks, detected=True))
    assert m.rep_count == 0

    landmarks[PoseLandmark.LEFT_HIP] = LandmarkPoint(x=0.35, y=0.55)
    landmarks[PoseLandmark.LEFT_KNEE] = LandmarkPoint(x=0.5, y=0.55)
    landmarks[PoseLandmark.LEFT_ANKLE] = LandmarkPoint(x=0.5, y=0.85)
    m = detector.process_frame(PoseResult(landmarks=landmarks, detected=True))
    assert m.stage in (RepStage.TRANSITION, RepStage.INFLECTION)

    landmarks[PoseLandmark.LEFT_HIP] = LandmarkPoint(x=0.45, y=0.4)
    landmarks[PoseLandmark.LEFT_KNEE] = LandmarkPoint(x=0.45, y=0.65)
    landmarks[PoseLandmark.LEFT_ANKLE] = LandmarkPoint(x=0.45, y=0.9)
    m = detector.process_frame(PoseResult(landmarks=landmarks, detected=True))
    assert m.rep_count == 1
    assert m.rep_completed_now is True
