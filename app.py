"""Apna AI Coach - Real-Time In-Browser AI Gym Coach.

Webcam WebRTC -> MediaPipe Pose -> Joint Angle Geometry -> Rep Counter & Form Checker
-> Groq Llama 3.3 70B -> gTTS -> Whisper Voice Q&A -> LLaVA VLM -> Langfuse Tracing.
"""

from __future__ import annotations

import collections
import logging
import threading
import time

import av
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, WebRtcMode, webrtc_streamer

from src.coaching.llm import CoachingEngine
from src.coaching.vlm import RateLimitError, VLMAnalyzer
from src.coaching.voice import VoiceEngine
from src.exercises.base import ExerciseDetector, ExerciseMetrics
from src.exercises.factory import (
    EXERCISE_DISPLAY_NAMES,
    get_exercise_detector,
    list_supported_exercises,
)
from src.vision.overlay import annotate_frame
from src.vision.pose import PoseDetector

# Load local .env if present
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("apna-ai-coach")

# WebRTC STUN server configuration for Streamlit Cloud deployment
RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
        ]
    }
)


class WorkoutVideoProcessor(VideoProcessorBase):
    """Processes WebRTC video frames in real time with MediaPipe pose & exercise tracker."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.detector: PoseDetector | None = None
        self.exercise_detector: ExerciseDetector | None = None
        self.exercise_name: str = "squats"
        self.latest_metrics: ExerciseMetrics | None = None
        self.recent_frames: collections.deque[np.ndarray] = collections.deque(maxlen=5)
        self.recent_errors: collections.deque[str] = collections.deque(maxlen=10)
        self.pending_coaching_audio: str | None = None
        self.last_coached_rep: int = 0
        self.fps: float = 0.0
        self._last_frame_time: float = time.time()
        self._init_detector()

    def _init_detector(self) -> None:
        try:
            self.detector = PoseDetector(video_mode=True)
            self.exercise_detector = get_exercise_detector(self.exercise_name)
        except Exception as exc:
            logger.error("Failed to initialize PoseDetector in WebRTC processor: %s", exc)

    def set_exercise(self, exercise_name: str) -> None:
        with self.lock:
            if self.exercise_name != exercise_name or self.exercise_detector is None:
                self.exercise_name = exercise_name
                self.exercise_detector = get_exercise_detector(exercise_name)
                self.recent_errors.clear()
                self.last_coached_rep = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img_bgr = frame.to_ndarray(format="bgr24")
        now = time.time()
        delta = now - self._last_frame_time
        self._last_frame_time = now
        if delta > 0:
            instant_fps = 1.0 / delta
            self.fps = 0.85 * self.fps + 0.15 * instant_fps if self.fps > 0 else instant_fps

        with self.lock:
            self.recent_frames.append(img_bgr.copy())

            if self.detector is None:
                self._init_detector()

            if self.detector is not None and self.exercise_detector is not None:
                pose = self.detector.detect(img_bgr)
                metrics = self.exercise_detector.process_frame(pose)
                self.latest_metrics = metrics

                if metrics.active_errors:
                    for err in metrics.active_errors:
                        self.recent_errors.append(err)

                annotated = annotate_frame(
                    img_bgr,
                    pose,
                    self.exercise_name,
                    metrics,
                    fps=self.fps,
                )
            else:
                annotated = img_bgr

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


@st.cache_resource
def get_coaching_engine() -> CoachingEngine:
    return CoachingEngine()


@st.cache_resource
def get_voice_engine() -> VoiceEngine:
    return VoiceEngine()


@st.cache_resource
def get_vlm_analyzer() -> VLMAnalyzer:
    return VLMAnalyzer()


def main() -> None:
    st.set_page_config(
        page_title="Apna AI Coach - Real-Time AI Gym Coach",
        page_icon="💪",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    coaching_engine = get_coaching_engine()
    voice_engine = get_voice_engine()
    vlm_analyzer = get_vlm_analyzer()

    st.markdown(
        """
        <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            color: #00C896;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.05rem;
            color: #A0AEC0;
            margin-bottom: 1.5rem;
        }
        .metric-card {
            background-color: #1A202C;
            border-radius: 8px;
            padding: 1rem;
            border: 1px solid #2D3748;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="main-header">💪 Apna AI Coach</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Live in-browser webcam streaming · MediaPipe 33-point pose · '
        'Cosine geometry · Groq Llama 3.3 70B · Whisper STT · gTTS · LLaVA VLM</div>',
        unsafe_allow_html=True,
    )

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Workout Controls")
        exercise_options = list_supported_exercises()
        selected_exercise_key = st.selectbox(
            "Select Exercise",
            options=exercise_options,
            format_func=lambda x: EXERCISE_DISPLAY_NAMES.get(x, x),
            index=0,
        )

        st.markdown("---")
        st.subheader("🤖 Engine Status")
        groq_status = "🟢 Active (Llama 3.3 70B)" if coaching_engine.has_llm else "🟡 Fallback (Rule-Based)"
        st.markdown(f"**Coaching LLM:** {groq_status}")

        langfuse_status = "🟢 Active" if coaching_engine.tracer.enabled else "⚪ Disabled (Silent)"
        st.markdown(f"**Langfuse Telemetry:** {langfuse_status}")

        st.markdown("**STT:** faster-whisper (`base.en`)")
        st.markdown("**TTS:** gTTS (Google Speech)")
        st.markdown("**VLM:** LLaVA-1.5-7B (1 call/min)")

        st.markdown("---")
        st.markdown("### 📋 Exercise Guide")
        guide_text = {
            "squats": "Break parallel (hip crease below knee). Keep chest high and knees pushed out.",
            "pushups": "Lower chest to 90° elbow bend. Keep core braced; avoid sagging or piking hips.",
            "bicep_curls": "Keep elbows glued to ribs. Squeeze biceps at peak and avoid swinging torso.",
            "lunges": "Step forward into a 90° front knee bend. Lower back knee close to the floor.",
            "shoulder_press": "Lock out overhead without arching lower back. Press symmetrically.",
        }
        st.info(guide_text.get(selected_exercise_key, ""))

    col_video, col_coaching = st.columns([3, 2])

    with col_video:
        st.markdown("### 📹 Live Workout Stream")
        ctx = webrtc_streamer(
            key=f"apna-ai-coach-{selected_exercise_key}",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=WorkoutVideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        if ctx.video_processor:
            ctx.video_processor.set_exercise(selected_exercise_key)

    with col_coaching:
        st.markdown("### 🗣️ Real-Time Coaching Feedback")

        # Container for stats & spoken audio
        stats_box = st.empty()
        audio_container = st.empty()
        coaching_text_container = st.empty()

        # Check processor metrics
        if ctx.video_processor and ctx.video_processor.latest_metrics:
            m = ctx.video_processor.latest_metrics
            with stats_box.container():
                c1, c2, c3 = st.columns(3)
                c1.metric("Reps", m.rep_count)
                c2.metric("Stage", m.stage.value)
                c3.metric("Primary Angle", f"{m.primary_angle:.0f}°")

                if m.active_errors:
                    for err in m.active_errors:
                        st.error(f"⚠️ {err}")

            # Every 5 reps trigger coaching
            with ctx.video_processor.lock:
                current_reps = m.rep_count
                last_rep = ctx.video_processor.last_coached_rep
                if current_reps > 0 and current_reps % 5 == 0 and current_reps != last_rep:
                    ctx.video_processor.last_coached_rep = current_reps
                    errors_list = list(ctx.video_processor.recent_errors)
                    feedback = coaching_engine.generate_batch_coaching(
                        exercise=selected_exercise_key,
                        reps=current_reps,
                        form_errors=errors_list,
                        primary_angle=m.primary_angle,
                        secondary_angle=m.secondary_angle,
                    )
                    coaching_text_container.success(f"Coach: \"{feedback}\"")
                    try:
                        audio_b64 = voice_engine.synthesize_speech_b64(feedback)
                        audio_container.markdown(
                            f'<audio autoplay="true" src="{audio_b64}"></audio>',
                            unsafe_allow_html=True,
                        )
                    except Exception as exc:
                        logger.warning("TTS generation failed: %s", exc)

        st.markdown("---")
        st.markdown("### 🎙️ Mid-Workout Voice Q&A")
        st.caption("Press record to ask your coach a question (e.g. 'What am I doing wrong?', 'How many reps so far?').")

        audio_input = None
        if hasattr(st, "audio_input"):
            audio_input = st.audio_input("Push to talk (Record question)")
        else:
            audio_input = st.file_uploader("Upload audio question (.wav / .mp3)", type=["wav", "mp3"])

        if audio_input is not None:
            with st.spinner("Whisper transcribing your question..."):
                try:
                    audio_bytes = audio_input.read()
                    question = voice_engine.transcribe(audio_bytes)
                    if question:
                        st.markdown(f"**You asked:** *\"{question}\"*")
                        reps = (
                            ctx.video_processor.latest_metrics.rep_count
                            if (ctx.video_processor and ctx.video_processor.latest_metrics)
                            else 0
                        )
                        recent_errs = (
                            list(ctx.video_processor.recent_errors) if ctx.video_processor else []
                        )
                        with st.spinner("Coach thinking..."):
                            answer = coaching_engine.answer_voice_question(
                                question=question,
                                exercise=selected_exercise_key,
                                reps=reps,
                                recent_errors=recent_errs,
                            )
                            st.info(f"**Coach:** {answer}")
                            ans_b64 = voice_engine.synthesize_speech_b64(answer)
                            st.markdown(
                                f'<audio autoplay="true" controls src="{ans_b64}"></audio>',
                                unsafe_allow_html=True,
                            )
                    else:
                        st.warning("No speech detected in audio clip. Try speaking louder.")
                except Exception as exc:
                    st.error(f"Voice Q&A error: {exc}")

        st.markdown("---")
        st.markdown("### 👁️ Multimodal Form Analysis (LLaVA VLM)")
        st.caption("Captures 3 recent frames and passes them to the VLM. Gated to 1 call per minute.")

        can_call, remaining = vlm_analyzer.can_call()
        button_disabled = not can_call
        btn_label = "Analyze My Form" if can_call else f"Rate Limited ({remaining:.0f}s left)"

        if st.button(btn_label, disabled=button_disabled):
            frames_to_send = []
            if ctx.video_processor:
                with ctx.video_processor.lock:
                    frames_to_send = list(ctx.video_processor.recent_frames)
                    active_errs = (
                        ctx.video_processor.latest_metrics.active_errors
                        if ctx.video_processor.latest_metrics
                        else []
                    )
            else:
                active_errs = []

            if len(frames_to_send) < 3:
                # If stream hasn't produced 3 frames yet, generate blank/dummy frames for testing
                frames_to_send = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)]

            with st.spinner("Analyzing multi-frame alignment with VLM..."):
                try:
                    vlm_result = vlm_analyzer.analyze_frames(
                        frames_to_send[-3:],
                        exercise=selected_exercise_key,
                        active_errors=active_errs,
                    )
                    if "form issue" in vlm_result.lower():
                        st.error(f"🔍 **VLM Verdict:** {vlm_result}")
                    else:
                        st.success(f"🔍 **VLM Verdict:** {vlm_result}")
                except RateLimitError as rle:
                    st.warning(str(rle))
                except Exception as exc:
                    st.error(f"VLM Analysis error: {exc}")


if __name__ == "__main__":
    main()
