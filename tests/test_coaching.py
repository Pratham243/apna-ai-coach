"""Unit tests for Coaching, Telemetry, Voice, and VLM modules."""

import numpy as np
import pytest

from src.coaching.llm import CoachingEngine
from src.coaching.vlm import RateLimitError, VLMAnalyzer
from src.coaching.voice import VoiceEngine
from src.observability.telemetry import LangfuseTracer


def test_coaching_engine_rule_fallback():
    engine = CoachingEngine(api_key="")
    assert engine.has_llm is False

    feedback_clean = engine.generate_batch_coaching(
        exercise="squats",
        reps=5,
        form_errors=[],
        primary_angle=85.0,
        secondary_angle=25.0,
    )
    assert len(feedback_clean) > 0
    assert "depth" in feedback_clean.lower() or "outstanding" in feedback_clean.lower()

    feedback_error = engine.generate_batch_coaching(
        exercise="squats",
        reps=5,
        form_errors=["Knees caving inward (push knees out)"],
        primary_angle=85.0,
        secondary_angle=25.0,
    )
    assert "knees" in feedback_error.lower()


def test_voice_qa_rule_fallback():
    engine = CoachingEngine(api_key="")
    ans_reps = engine.answer_voice_question(
        question="How many reps have I done?",
        exercise="squats",
        reps=10,
        recent_errors=[],
    )
    assert "10 reps" in ans_reps

    ans_form = engine.answer_voice_question(
        question="What am I doing wrong?",
        exercise="pushups",
        reps=4,
        recent_errors=["Sagging hips (engage core and glutes)"],
    )
    assert "sagging hips" in ans_form.lower()


def test_langfuse_silent_degradation():
    tracer = LangfuseTracer()
    # Missing credentials in test env should leave it disabled
    with tracer.trace_generation("test_gen", "test_model", {"input": "test"}):
        pass
    # No exception raised


def test_vlm_rate_limiting():
    analyzer = VLMAnalyzer()
    analyzer.last_call_timestamp = 0.0  # reset

    frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(3)]
    res1 = analyzer.analyze_frames(frames, "squats")
    assert "pose looks correct" in res1 or "form issue noticed" in res1

    # Immediate second call should raise RateLimitError
    with pytest.raises(RateLimitError):
        analyzer.analyze_frames(frames, "squats")


def test_voice_engine_tts_synthesis():
    engine = VoiceEngine()
    mp3_bytes = engine.synthesize_speech("Keep your back straight")
    assert len(mp3_bytes) > 0

    b64_uri = engine.synthesize_speech_b64("Good rep")
    assert b64_uri.startswith("data:audio/mp3;base64,")
