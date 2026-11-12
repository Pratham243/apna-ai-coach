"""Coaching package combining LLM, VLM, and Voice subsystems."""

from src.coaching.llm import CoachingEngine
from src.coaching.prompts import (
    COACH_BATCH_PROMPT_TEMPLATE,
    COACH_SYSTEM_PROMPT,
    VLM_FORM_ANALYSIS_PROMPT,
    VOICE_QA_SYSTEM_PROMPT,
)
from src.coaching.vlm import RateLimitError, VLMAnalyzer
from src.coaching.voice import VoiceEngine

__all__ = [
    "COACH_BATCH_PROMPT_TEMPLATE",
    "COACH_SYSTEM_PROMPT",
    "VLM_FORM_ANALYSIS_PROMPT",
    "VOICE_QA_SYSTEM_PROMPT",
    "CoachingEngine",
    "RateLimitError",
    "VLMAnalyzer",
    "VoiceEngine",
]
