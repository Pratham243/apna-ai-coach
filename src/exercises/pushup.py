"""Push-up exercise detector: rep counter, elbow flare, and hip sagging checks."""

from __future__ import annotations

from src.exercises.base import ExerciseDetector, ExerciseMetrics, RepStage
from src.vision.geometry import calculate_angle
from src.vision.pose import PoseLandmark, PoseResult


class PushupDetector(ExerciseDetector):
    """Tracks push-ups: depth (elbow angle), elbow flare, and sagging hips."""

    name = "pushups"
    description = "Tracks chest-to-floor push-ups, checking elbow angles and rigid plank line."

    TOP_ELBOW_ANGLE = 160.0
    BOTTOM_ELBOW_ANGLE = 90.0

    def __init__(self) -> None:
        super().__init__()
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
        # Pick dominant side
        l_vis = (lm[PoseLandmark.LEFT_SHOULDER].visibility + lm[PoseLandmark.LEFT_ELBOW].visibility) / 2
        r_vis = (lm[PoseLandmark.RIGHT_SHOULDER].visibility + lm[PoseLandmark.RIGHT_ELBOW].visibility) / 2

        if l_vis >= r_vis:
            shoulder = lm[PoseLandmark.LEFT_SHOULDER]
            elbow = lm[PoseLandmark.LEFT_ELBOW]
            wrist = lm[PoseLandmark.LEFT_WRIST]
            hip = lm[PoseLandmark.LEFT_HIP]
            ankle = lm[PoseLandmark.LEFT_ANKLE]
        else:
            shoulder = lm[PoseLandmark.RIGHT_SHOULDER]
            elbow = lm[PoseLandmark.RIGHT_ELBOW]
            wrist = lm[PoseLandmark.RIGHT_WRIST]
            hip = lm[PoseLandmark.RIGHT_HIP]
            ankle = lm[PoseLandmark.RIGHT_ANKLE]

        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        # Plank alignment: shoulder - hip - ankle angle (should stay ~160 - 180 deg)
        plank_angle = calculate_angle(shoulder, hip, ankle)

        active_frame_errors: list[str] = []

        # Check 1: Sagging or piking hips (body line broken)
        if plank_angle < 150.0:
            msg = "Sagging hips (engage core and glutes)"
            active_frame_errors.append(msg)
            self.current_rep_errors.add(msg)
        elif plank_angle > 195.0:  # in case hips are raised too high
            msg = "Hips piked too high (keep body in straight line)"
            active_frame_errors.append(msg)
            self.current_rep_errors.add(msg)

        # Check 2: Elbow flaring excessively (shoulder-elbow angle vs torso)
        torso_elbow_angle = calculate_angle(hip, shoulder, elbow)
        if elbow_angle < 120.0 and torso_elbow_angle > 75.0:
            msg = "Elbows flaring out (tuck elbows to ~45 degrees)"
            active_frame_errors.append(msg)
            self.current_rep_errors.add(msg)

        # State machine
        rep_completed_now = False
        if self.stage == RepStage.START:
            if elbow_angle < 140.0:
                self.stage = RepStage.TRANSITION
                self._min_elbow_angle = elbow_angle
        elif self.stage == RepStage.TRANSITION:
            self._min_elbow_angle = min(self._min_elbow_angle, elbow_angle)
            if elbow_angle <= self.BOTTOM_ELBOW_ANGLE:
                self.stage = RepStage.INFLECTION
            elif elbow_angle > 150.0:
                # Returned without hitting depth
                if self._min_elbow_angle > 105.0:
                    msg = "Push-up depth shallow (lower chest to elbow height)"
                    self.current_rep_errors.add(msg)
                self.rep_count += 1
                rep_completed_now = True
                self.history_errors.extend(list(self.current_rep_errors))
                self.stage = RepStage.START
                self._min_elbow_angle = 180.0
        elif self.stage == RepStage.INFLECTION:
            self._min_elbow_angle = min(self._min_elbow_angle, elbow_angle)
            if elbow_angle > 120.0:
                self.stage = RepStage.RETURN
        elif self.stage == RepStage.RETURN:
            if elbow_angle >= self.TOP_ELBOW_ANGLE:
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
            secondary_angle=round(plank_angle, 1),
            active_errors=active_frame_errors,
            rep_errors=errors_for_rep,
            rep_completed_now=rep_completed_now,
            debug_info={
                "torso_elbow_angle": round(torso_elbow_angle, 1),
                "min_elbow_angle": round(self._min_elbow_angle, 1),
            },
        )
