"""Squat exercise detector: rep counter and form error detection."""

from __future__ import annotations

from src.exercises.base import ExerciseDetector, ExerciseMetrics, RepStage
from src.vision.geometry import calculate_angle, calculate_vertical_angle
from src.vision.pose import PoseLandmark, PoseResult


class SquatDetector(ExerciseDetector):
    """Tracks squats: knee depth, knees-caving (valgus), and back rounding."""

    name = "squats"
    description = "Bodyweight or barbell squats tracking hip depth, knee alignment, and torso angle."

    # Angles in degrees
    STANDING_KNEE_ANGLE = 160.0
    DEPTH_KNEE_ANGLE = 90.0
    SHALLOW_THRESHOLD = 100.0

    def __init__(self) -> None:
        super().__init__()
        self._min_knee_angle_this_rep = 180.0

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
        # Pick best visible side or average both
        left_vis = (lm[PoseLandmark.LEFT_HIP].visibility + lm[PoseLandmark.LEFT_KNEE].visibility) / 2
        right_vis = (lm[PoseLandmark.RIGHT_HIP].visibility + lm[PoseLandmark.RIGHT_KNEE].visibility) / 2

        if left_vis >= right_vis:
            hip = lm[PoseLandmark.LEFT_HIP]
            knee = lm[PoseLandmark.LEFT_KNEE]
            ankle = lm[PoseLandmark.LEFT_ANKLE]
            shoulder = lm[PoseLandmark.LEFT_SHOULDER]
        else:
            hip = lm[PoseLandmark.RIGHT_HIP]
            knee = lm[PoseLandmark.RIGHT_KNEE]
            ankle = lm[PoseLandmark.RIGHT_ANKLE]
            shoulder = lm[PoseLandmark.RIGHT_SHOULDER]

        knee_angle = calculate_angle(hip, knee, ankle)
        torso_angle = calculate_vertical_angle(shoulder, hip)
        hip_angle = calculate_angle(shoulder, hip, knee)

        active_frame_errors: list[str] = []

        # Check 1: Knees caving in (valgus) when facing camera
        # If both knees and ankles are well visible, check knee x separation vs ankle x separation
        l_knee = lm[PoseLandmark.LEFT_KNEE]
        r_knee = lm[PoseLandmark.RIGHT_KNEE]
        l_ankle = lm[PoseLandmark.LEFT_ANKLE]
        r_ankle = lm[PoseLandmark.RIGHT_ANKLE]

        if min(l_knee.visibility, r_knee.visibility, l_ankle.visibility, r_ankle.visibility) > 0.4:
            knee_dist = abs(l_knee.x - r_knee.x)
            ankle_dist = abs(l_ankle.x - r_ankle.x)
            # When squatting down, knees should not cave inside ankle width
            if knee_angle < 130.0 and ankle_dist > 0.08 and knee_dist < (ankle_dist * 0.72):
                msg = "Knees caving inward (push knees out)"
                active_frame_errors.append(msg)
                self.current_rep_errors.add(msg)

        # Check 2: Excessive forward lean / back rounding (torso > 45 deg from vertical)
        if knee_angle < 140.0 and torso_angle > 45.0:
            msg = "Excessive torso lean (keep chest up)"
            active_frame_errors.append(msg)
            self.current_rep_errors.add(msg)

        # State machine
        rep_completed_now = False
        if self.stage == RepStage.START:
            if knee_angle < 145.0:
                self.stage = RepStage.TRANSITION
                self._min_knee_angle_this_rep = knee_angle
        elif self.stage == RepStage.TRANSITION:
            self._min_knee_angle_this_rep = min(self._min_knee_angle_this_rep, knee_angle)
            if knee_angle <= self.DEPTH_KNEE_ANGLE:
                self.stage = RepStage.INFLECTION
            elif knee_angle > 155.0:
                # User stood back up without hitting depth
                if self._min_knee_angle_this_rep > self.SHALLOW_THRESHOLD:
                    msg = "Squat depth shallow (hips must break parallel)"
                    self.current_rep_errors.add(msg)
                self.rep_count += 1
                rep_completed_now = True
                self.history_errors.extend(list(self.current_rep_errors))
                self.stage = RepStage.START
                self._min_knee_angle_this_rep = 180.0
        elif self.stage == RepStage.INFLECTION:
            self._min_knee_angle_this_rep = min(self._min_knee_angle_this_rep, knee_angle)
            if knee_angle > 120.0:
                self.stage = RepStage.RETURN
        elif self.stage == RepStage.RETURN:
            if knee_angle >= self.STANDING_KNEE_ANGLE:
                self.rep_count += 1
                rep_completed_now = True
                self.history_errors.extend(list(self.current_rep_errors))
                self.stage = RepStage.START
                self._min_knee_angle_this_rep = 180.0

        errors_for_rep = list(self.current_rep_errors)
        if rep_completed_now:
            self.current_rep_errors.clear()

        return ExerciseMetrics(
            rep_count=self.rep_count,
            stage=self.stage,
            primary_angle=round(knee_angle, 1),
            secondary_angle=round(torso_angle, 1),
            active_errors=active_frame_errors,
            rep_errors=errors_for_rep,
            rep_completed_now=rep_completed_now,
            debug_info={
                "hip_angle": round(hip_angle, 1),
                "min_knee_angle": round(self._min_knee_angle_this_rep, 1),
            },
        )
