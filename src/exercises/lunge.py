"""Lunge exercise detector: rep counter, knee-over-toes, and back knee depth checks."""

from __future__ import annotations

from src.exercises.base import ExerciseDetector, ExerciseMetrics, RepStage
from src.vision.geometry import calculate_angle, calculate_vertical_angle
from src.vision.pose import PoseLandmark, PoseResult


class LungeDetector(ExerciseDetector):
    """Tracks lunges: front knee angle, back knee depth, and front knee over toes."""

    name = "lunges"
    description = "Tracks forward or static lunges, validating knee tracking and back knee depth."

    STANDING_KNEE_ANGLE = 160.0
    LUNGE_DEPTH_ANGLE = 95.0

    def __init__(self) -> None:
        super().__init__()
        self._min_knee_angle = 180.0

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
        # Determine front vs back leg by checking which knee/ankle is lower or flexed
        l_knee_angle = calculate_angle(
            lm[PoseLandmark.LEFT_HIP], lm[PoseLandmark.LEFT_KNEE], lm[PoseLandmark.LEFT_ANKLE]
        )
        r_knee_angle = calculate_angle(
            lm[PoseLandmark.RIGHT_HIP], lm[PoseLandmark.RIGHT_KNEE], lm[PoseLandmark.RIGHT_ANKLE]
        )

        # The leg with the more acute knee angle in a lunge is typically the front working leg
        if l_knee_angle <= r_knee_angle:
            front_knee = lm[PoseLandmark.LEFT_KNEE]
            front_ankle = lm[PoseLandmark.LEFT_ANKLE]
            front_toe = lm[PoseLandmark.LEFT_FOOT_INDEX]
            primary_angle = l_knee_angle
            secondary_angle = r_knee_angle
        else:
            front_knee = lm[PoseLandmark.RIGHT_KNEE]
            front_ankle = lm[PoseLandmark.RIGHT_ANKLE]
            front_toe = lm[PoseLandmark.RIGHT_FOOT_INDEX]
            primary_angle = r_knee_angle
            secondary_angle = l_knee_angle

        torso_angle = calculate_vertical_angle(
            lm[PoseLandmark.LEFT_SHOULDER], lm[PoseLandmark.LEFT_HIP]
        )

        active_frame_errors: list[str] = []

        # Check 1: Front knee traveling excessively past toes
        # If facing sideways: front_toe.x and front_knee.x comparison
        toe_dist_x = front_knee.x - front_toe.x
        ankle_dist_x = front_knee.x - front_ankle.x
        if primary_angle < 110.0 and abs(toe_dist_x) > 0.09 and abs(ankle_dist_x) > 0.12:
            msg = "Front knee projecting past toes (step out further)"
            active_frame_errors.append(msg)
            self.current_rep_errors.add(msg)

        # Check 2: Back knee not lowering sufficiently (shallow lunge)
        if primary_angle < 105.0 and secondary_angle > 135.0:
            msg = "Back knee too high (lower back knee towards floor)"
            active_frame_errors.append(msg)
            self.current_rep_errors.add(msg)

        # State machine
        rep_completed_now = False
        if self.stage == RepStage.START:
            if primary_angle < 140.0:
                self.stage = RepStage.TRANSITION
                self._min_knee_angle = primary_angle
        elif self.stage == RepStage.TRANSITION:
            self._min_knee_angle = min(self._min_knee_angle, primary_angle)
            if primary_angle <= self.LUNGE_DEPTH_ANGLE:
                self.stage = RepStage.INFLECTION
            elif primary_angle > 150.0:
                if self._min_knee_angle > 110.0:
                    msg = "Lunge depth shallow (aim for 90-degree bend in front knee)"
                    self.current_rep_errors.add(msg)
                self.rep_count += 1
                rep_completed_now = True
                self.history_errors.extend(list(self.current_rep_errors))
                self.stage = RepStage.START
                self._min_knee_angle = 180.0
        elif self.stage == RepStage.INFLECTION:
            self._min_knee_angle = min(self._min_knee_angle, primary_angle)
            if primary_angle > 120.0:
                self.stage = RepStage.RETURN
        elif self.stage == RepStage.RETURN:
            if primary_angle >= self.STANDING_KNEE_ANGLE:
                self.rep_count += 1
                rep_completed_now = True
                self.history_errors.extend(list(self.current_rep_errors))
                self.stage = RepStage.START
                self._min_knee_angle = 180.0

        errors_for_rep = list(self.current_rep_errors)
        if rep_completed_now:
            self.current_rep_errors.clear()

        return ExerciseMetrics(
            rep_count=self.rep_count,
            stage=self.stage,
            primary_angle=round(primary_angle, 1),
            secondary_angle=round(secondary_angle, 1),
            active_errors=active_frame_errors,
            rep_errors=errors_for_rep,
            rep_completed_now=rep_completed_now,
            debug_info={
                "torso_angle": round(torso_angle, 1),
                "min_knee_angle": round(self._min_knee_angle, 1),
            },
        )
