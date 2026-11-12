"""Bicep curl detector: rep counter, elbow drift, and torso swing checks."""

from __future__ import annotations

from src.exercises.base import ExerciseDetector, ExerciseMetrics, RepStage
from src.vision.geometry import calculate_angle, calculate_vertical_angle
from src.vision.pose import PoseLandmark, PoseResult


class BicepCurlDetector(ExerciseDetector):
    """Tracks bicep curls: elbow flexion, elbow drift (pin elbows to sides), and torso swing."""

    name = "bicep_curls"
    description = "Tracks dumbbell/barbell bicep curls with elbow pinning and torso stillness checks."

    EXTENDED_ELBOW_ANGLE = 150.0
    CURLED_ELBOW_ANGLE = 50.0

    def __init__(self) -> None:
        super().__init__()
        self._initial_elbow_x: float | None = None
        self._min_elbow_angle = 180.0

    def process_frame(self, pose: PoseResult, timestamp_ms: int | None = None) -> ExerciseMetrics:
        self.frame_count += 1
        if not pose.detected or len(pose.landmarks) < 33:
            return ExerciseMetrics(
                rep_count=self.rep_count,
                stage=self.stage,
                active_errors=list(self.current_rep_errors),
                debug_info={"pose_detected": False},
            )

        lm = pose.landmarks
        # Choose side with highest confidence arm
        l_arm = (lm[PoseLandmark.LEFT_SHOULDER].visibility + lm[PoseLandmark.LEFT_WRIST].visibility) / 2
        r_arm = (lm[PoseLandmark.RIGHT_SHOULDER].visibility + lm[PoseLandmark.RIGHT_WRIST].visibility) / 2

        if l_arm >= r_arm:
            shoulder = lm[PoseLandmark.LEFT_SHOULDER]
            elbow = lm[PoseLandmark.LEFT_ELBOW]
            wrist = lm[PoseLandmark.LEFT_WRIST]
            hip = lm[PoseLandmark.LEFT_HIP]
        else:
            shoulder = lm[PoseLandmark.RIGHT_SHOULDER]
            elbow = lm[PoseLandmark.RIGHT_ELBOW]
            wrist = lm[PoseLandmark.RIGHT_WRIST]
            hip = lm[PoseLandmark.RIGHT_HIP]

        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        torso_angle = calculate_vertical_angle(shoulder, hip)

        active_frame_errors: list[str] = []

        # Check 1: Torso swing (cheating with momentum)
        if torso_angle > 18.0:
            msg = "Torso swinging (stay upright, avoid swinging)"
            active_frame_errors.append(msg)
            self.current_rep_errors.add(msg)

        # Check 2: Elbow drift (elbow should stay vertically aligned with shoulder/hip)
        # In normalized coords, x distance between shoulder and elbow
        elbow_drift = abs(elbow.x - shoulder.x)
        if elbow_drift > 0.10:
            msg = "Elbow drifting forward (keep elbows pinned to your sides)"
            active_frame_errors.append(msg)
            self.current_rep_errors.add(msg)

        # State machine
        rep_completed_now = False
        if self.stage == RepStage.START:
            if elbow_angle < 130.0:
                self.stage = RepStage.TRANSITION
                self._min_elbow_angle = elbow_angle
        elif self.stage == RepStage.TRANSITION:
            self._min_elbow_angle = min(self._min_elbow_angle, elbow_angle)
            if elbow_angle <= self.CURLED_ELBOW_ANGLE:
                self.stage = RepStage.INFLECTION
            elif elbow_angle > 140.0:
                # Dropped down without full peak contraction
                if self._min_elbow_angle > 70.0:
                    msg = "Incomplete curl range of motion (squeeze at the top)"
                    self.current_rep_errors.add(msg)
                self.rep_count += 1
                rep_completed_now = True
                self.history_errors.extend(list(self.current_rep_errors))
                self.stage = RepStage.START
                self._min_elbow_angle = 180.0
        elif self.stage == RepStage.INFLECTION:
            self._min_elbow_angle = min(self._min_elbow_angle, elbow_angle)
            if elbow_angle > 80.0:
                self.stage = RepStage.RETURN
        elif self.stage == RepStage.RETURN:
            if elbow_angle >= self.EXTENDED_ELBOW_ANGLE:
                self.rep_count += 1
                rep_completed_now = True
                self.history_errors.extend(list(self.current_rep_errors))
                self.stage = RepStage.START
                self._min_elbow_angle = 180.0

        errors_for_rep = list(self.current_rep_errors)
        if rep_completed_now:
            self.current_rep_errors.clear()

        return ExerciseMetrics(
            rep_count=self.rep_count,
            stage=self.stage,
            primary_angle=round(elbow_angle, 1),
            secondary_angle=round(torso_angle, 1),
            active_errors=active_frame_errors,
            rep_errors=errors_for_rep,
            rep_completed_now=rep_completed_now,
            debug_info={
                "elbow_drift": round(elbow_drift, 3),
                "min_elbow_angle": round(self._min_elbow_angle, 1),
            },
        )
