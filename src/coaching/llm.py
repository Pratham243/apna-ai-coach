"""LLM coaching client using Groq Llama 3.3 70B with rule-based fallback and Langfuse tracing."""

from __future__ import annotations

import logging
import os
from typing import Any

from src.coaching.prompts import (
    COACH_BATCH_PROMPT_TEMPLATE,
    COACH_SYSTEM_PROMPT,
    VOICE_QA_SYSTEM_PROMPT,
)
from src.observability.telemetry import get_tracer

logger = logging.getLogger(__name__)

DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"

# Rule-based fallback strings if Groq is unavailable or unconfigured
RULE_BASED_FALLBACKS: dict[str, dict[str, str]] = {
    "squats": {
        "Knees caving inward (push knees out)": "Keep driving your knees outward in line with your toes.",
        "Excessive torso lean (keep chest up)": "Keep your chest proud and brace your core so you do not fold forward.",
        "Squat depth shallow (hips must break parallel)": "Sink down a couple more inches so your hips break parallel with your knees.",
        "clean": "Outstanding depth and solid knee alignment! Keep that rhythm going.",
    },
    "pushups": {
        "Sagging hips (engage core and glutes)": "Engage your core and squeeze your glutes to maintain a rigid plank line.",
        "Hips piked too high (keep body in straight line)": "Lower your hips slightly so your shoulders, hips, and ankles form a straight line.",
        "Elbows flaring out (tuck elbows to ~45 degrees)": "Tuck your elbows to a 45-degree angle from your torso to protect your shoulders.",
        "Push-up depth shallow (lower chest to elbow height)": "Lower your chest all the way down until your elbows reach ninety degrees.",
        "clean": "Excellent plank stability and crisp full-range push-ups! Solid set.",
    },
    "bicep_curls": {
        "Torso swinging (stay upright, avoid swinging)": "Stand tall and freeze your torso; eliminate momentum on every rep.",
        "Elbow drifting forward (keep elbows pinned to your sides)": "Pin your elbows firmly to your ribcage and isolate the biceps.",
        "Incomplete curl range of motion (squeeze at the top)": "Squeeze hard at the peak and lower under complete control.",
        "clean": "Perfect elbow pinning and tempo! Great bicep isolation.",
    },
    "lunges": {
        "Front knee projecting past toes (step out further)": "Take a slightly wider stride so your front shin stays nearly vertical.",
        "Back knee too high (lower back knee towards floor)": "Drop that back knee closer to the floor to fully engage your legs.",
        "Lunge depth shallow (aim for 90-degree bend in front knee)": "Commit to the depth; bend both knees down to ninety degrees.",
        "clean": "Fantastic stride length and balanced posture through the whole lunge!",
    },
    "shoulder_press": {
        "Excessive lumbar arch (squeeze glutes, engage core)": "Tuck your pelvis and brace your abs to protect your lower back overhead.",
        "Uneven press (press both arms symmetrically)": "Lock both arms overhead in sync with equal speed and height.",
        "Incomplete lockout at top (extend elbows overhead)": "Press all the way to full extension and stack the weights above your shoulders.",
        "clean": "Crisp overhead lockout and solid core bracing! Great pressing.",
    },
}


class CoachingEngine:
    """Provides conversational audio-ready coaching cues via Groq Llama 3.3 70B or rule fallback."""

    def __init__(self, api_key: str | None = None, model: str = DEFAULT_GROQ_MODEL) -> None:
        self.api_key = (api_key or os.getenv("GROQ_API_KEY", "")).strip()
        self.model = model
        self.tracer = get_tracer()
        self._groq_client: Any = None

        if self.api_key:
            try:
                from groq import Groq

                self._groq_client = Groq(api_key=self.api_key)
                logger.info("Groq client initialized with model: %s", self.model)
            except Exception as exc:
                logger.warning("Failed to initialize Groq client: %s. Using rule-based fallback.", exc)
                self._groq_client = None
        else:
            logger.info("GROQ_API_KEY not provided. Operating in rule-based coaching fallback mode.")

    @property
    def has_llm(self) -> bool:
        return self._groq_client is not None

    def generate_batch_coaching(
        self,
        exercise: str,
        reps: int,
        form_errors: list[str],
        primary_angle: float,
        secondary_angle: float,
    ) -> str:
        """Generate 1-2 sentences of coaching feedback for a 5-rep workout batch."""
        error_text = "\n".join(f"- {err}" for err in set(form_errors)) if form_errors else "None (Clean reps)"
        user_prompt = COACH_BATCH_PROMPT_TEMPLATE.format(
            exercise=exercise,
            reps=reps,
            error_list=error_text,
            primary_angle=primary_angle,
            secondary_angle=secondary_angle,
        )

        metadata = {
            "exercise": exercise,
            "reps": reps,
            "error_count": len(form_errors),
        }

        # Try Groq LLM if configured
        if self._groq_client is not None:
            try:
                with self.tracer.trace_generation(
                    name="groq_batch_coaching",
                    model=self.model,
                    input_data={"system": COACH_SYSTEM_PROMPT, "user": user_prompt},
                    metadata=metadata,
                ) as span:
                    response = self._groq_client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": COACH_SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=80,
                        temperature=0.6,
                    )
                    content = response.choices[0].message.content.strip()
                    if span and hasattr(span, "update"):
                        try:
                            span.update(output=content)
                        except Exception:
                            pass
                    return content
            except Exception as exc:
                logger.warning("Groq API call failed: %s. Falling back to rule-based coaching.", exc)

        # Fallback to rule-based string
        return self._rule_based_coaching(exercise, form_errors)

    def answer_voice_question(
        self,
        question: str,
        exercise: str,
        reps: int,
        recent_errors: list[str],
    ) -> str:
        """Answer athlete voice query during workout."""
        recent_errors_str = ", ".join(set(recent_errors)) if recent_errors else "None"
        system_prompt = VOICE_QA_SYSTEM_PROMPT.format(
            exercise=exercise,
            reps=reps,
            recent_errors=recent_errors_str,
        )

        if self._groq_client is not None:
            try:
                with self.tracer.trace_generation(
                    name="groq_voice_qa",
                    model=self.model,
                    input_data={"system": system_prompt, "question": question},
                    metadata={"exercise": exercise, "reps": reps},
                ) as span:
                    response = self._groq_client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": question},
                        ],
                        max_tokens=80,
                        temperature=0.5,
                    )
                    content = response.choices[0].message.content.strip()
                    if span and hasattr(span, "update"):
                        try:
                            span.update(output=content)
                        except Exception:
                            pass
                    return content
            except Exception as exc:
                logger.warning("Groq voice Q&A failed: %s. Using rule response.", exc)

        # Basic fallback answers for common workout questions
        q_lower = question.lower()
        if "how many" in q_lower or "rep" in q_lower:
            return f"You have completed {reps} reps of {exercise} so far. Keep pushing!"
        if "wrong" in q_lower or "form" in q_lower:
            if recent_errors:
                return f"Watch out for {recent_errors[0].lower()}. Focus on tight control."
            return "Your form looks solid and aligned right now. Stay consistent!"
        return f"Stay focused on controlled breathing and clean tempo through your {exercise}."

    def _rule_based_coaching(self, exercise: str, form_errors: list[str]) -> str:
        """Pick suitable rule-based coaching phrase."""
        exercise_rules = RULE_BASED_FALLBACKS.get(exercise, {})
        if not form_errors:
            return exercise_rules.get("clean", "Great form and solid rhythm! Keep it up.")

        # Pick most frequent or first error
        for err in form_errors:
            if err in exercise_rules:
                return exercise_rules[err]

        return f"Focus on maintaining proper alignment and control on your {exercise}."
