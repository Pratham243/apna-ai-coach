"""Shoulder press detector: rep counter, full overhead extension, and elbow alignment."""

from __future__ import annotations

from src.exercises.base import ExerciseDetector, ExerciseMetrics, RepStage
from src.vision.geometry import calculate_angle, calculate_vertical_angle
from src.vision.pose import PoseLandmark, PoseResult


class ShoulderPressDetector(ExerciseDetector):
    """Tracks overhead shoulder press: lockout extension and elbow/torso alignment."""

    name = "shoulder_press"
    description = "Tracks overhead dumbbell/barbell presses checking top lockout and torso arch."

    BOTTOM_ELBOW_ANGLE = 85.0
    TOP_LOCKOUT_ANGLE = 160.0

    def __init__(self) -> None:
        super().__init__()
        self._max_elbow_angle = 0.0

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
        l_shoulder = lm[PoseLandmark.LEFT_SHOULDER]
        r_shoulder = lm[PoseLandmark.RIGHT_SHOULDER]
        l_elbow = lm[PoseLandmark.LEFT_ELBOW]
        r_elbow = lm[PoseLandmark.RIGHT_ELBOW]
        l_wrist = lm[PoseLandmark.LEFT_WRIST]
        r_wrist = lm[PoseLandmark.RIGHT_WRIST]
        l_hip = lm[PoseLandmark.LEFT_HIP]
        r_hip = lm[PoseLandmark.RIGHT_HIP]

        l_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        r_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
        avg_elbow_angle = (l_angle + r_angle) / 2.0

        torso_angle = calculate_vertical_angle(
            ((l_shoulder.x + r_shoulder.x) / 2, (l_shoulder.y + r_shoulder.y) / 2),
            ((l_hip.x + r_hip.x) / 2, (l_hip.y + r_hip.y) / 2),
        )

        active_frame_errors: list[str] = []

        # Check 1: Excessive lower back arch (torso angle from vertical)
        if torso_angle > 18.0:
            msg = "Excessive lumbar arch (squeeze glutes, engage core)"
            active_frame_errors.append(msg)
            self.current_rep_errors.add(msg)

        # Check 2: Asymmetric elbow press / uneven press
        if abs(l_angle - r_angle) > 28.0:
            msg = "Uneven press (press both arms symmetrically)"
            active_frame_errors.append(msg)
            self.current_rep_errors.add(msg)

        # Check 3: Incomplete extension / lockout check at top
        # When wrists are above shoulders, check if wrists reach full overhead height
        avg_wrist_y = (l_wrist.y + r_wrist.y) / 2.0
        avg_shoulder_y = (l_shoulder.y + r_shoulder.y) / 2.0

        # State machine: Start at bottom (chin/shoulder height, elbow ~80-100) -> Press overhead -> Return
        rep_completed_now = False
        if self.stage == RepStage.START:
            if avg_elbow_angle > 105.0 and avg_wrist_y < avg_shoulder_y:
                self.stage = RepStage.TRANSITION
                self._max_elbow_angle = avg_elbow_angle
        elif self.stage == RepStage.TRANSITION:
            self._max_elbow_angle = max(self._max_elbow_angle, avg_elbow_angle)
            if avg_elbow_angle >= self.TOP_LOCKOUT_ANGLE and avg_wrist_y < avg_shoulder_y:
                self.stage = RepStage.INFLECTION
            elif avg_elbow_angle < 95.0:
                # Lowered back down without full overhead lockout
                if self._max_elbow_angle < 145.0:
                    msg = "Incomplete lockout at top (extend elbows overhead)"
                    self.current_rep_errors.add(msg)
                self.rep_count += 1
                rep_completed_now = True
                self.history_errors.extend(list(self.current_rep_errors))
                self.stage = RepStage.START
                self._max_elbow_angle = 0.0
        elif self.stage == RepStage.INFLECTION:
            self._max_elbow_angle = max(self._max_elbow_angle, avg_elbow_angle)
            if avg_elbow_angle < 130.0:
                self.stage = RepStage.RETURN
        elif self.stage == RepStage.RETURN:
            if avg_elbow_angle <= self.BOTTOM_ELBOW_ANGLE or (avg_wrist_y >= avg_shoulder_y - 0.05):
                self.rep_count += 1
                rep_completed_now = True
                self.history_errors.extend(list(self.current_rep_errors))
                self.stage = RepStage.START
                self._max_elbow_angle = 0.0

        errors_for_rep = list(self.current_rep_errors)
        if rep_completed_now:
            self.current_rep_errors.clear()

        return ExerciseMetrics(
            rep_count=self.rep_count,
            stage=self.stage,
            primary_angle=round(avg_elbow_angle, 1),
            secondary_angle=round(torso_angle, 1),
            active_errors=active_frame_errors,
            rep_errors=errors_for_rep,
            rep_completed_now=rep_completed_now,
            debug_info={
                "left_elbow": round(l_angle, 1),
                "right_elbow": round(r_angle, 1),
                "max_elbow_angle": round(self._max_elbow_angle, 1),
            },
        )
