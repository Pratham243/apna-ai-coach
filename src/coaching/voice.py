"""Voice I/O module: faster-whisper STT for athlete questions and gTTS for voice feedback."""

from __future__ import annotations

import base64
import io
import logging
import tempfile
from pathlib import Path
from typing import Any

from gtts import gTTS

logger = logging.getLogger(__name__)

DEFAULT_WHISPER_MODEL = "base.en"


class VoiceEngine:
    """Manages Whisper STT transcription and gTTS speech synthesis."""

    def __init__(self, whisper_model_size: str = DEFAULT_WHISPER_MODEL) -> None:
        self.whisper_model_size = whisper_model_size
        self._whisper_model: Any = None

    def _ensure_whisper(self) -> Any:
        """Lazy load faster-whisper model."""
        if self._whisper_model is None:
            try:
                from faster_whisper import WhisperModel

                logger.info("Loading faster-whisper model '%s' on CPU (int8)...", self.whisper_model_size)
                self._whisper_model = WhisperModel(
                    self.whisper_model_size,
                    device="cpu",
                    compute_type="int8",
                )
            except Exception as exc:
                logger.error("Failed to load faster-whisper: %s", exc)
                raise
        return self._whisper_model

    def transcribe(self, audio_data: bytes | io.BytesIO | Path | str) -> str:
        """Transcribe an audio clip to English text."""
        model = self._ensure_whisper()

        # If audio_data is bytes or BytesIO, save to a temporary file
        temp_file = None
        if isinstance(audio_data, (bytes, bytearray)):
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_file.write(audio_data)
            temp_file.flush()
            temp_file.close()
            audio_path = temp_file.name
        elif isinstance(audio_data, io.BytesIO):
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_file.write(audio_data.getvalue())
            temp_file.flush()
            temp_file.close()
            audio_path = temp_file.name
        else:
            audio_path = str(audio_data)

        try:
            segments, info = model.transcribe(
                audio_path,
                beam_size=3,
                language="en",
                vad_filter=True,
            )
            transcription = " ".join(seg.text for seg in segments).strip()
            logger.info("Transcribed audio (len=%.2fs): '%s'", getattr(info, "duration", 0.0), transcription)
            return transcription
        finally:
            if temp_file is not None:
                Path(temp_file.name).unlink(missing_ok=True)

    @staticmethod
    def synthesize_speech(text: str, lang: str = "en") -> bytes:
        """Convert coaching text into MP3 audio bytes using gTTS with offline fallback."""
        cleaned = text.strip()
        if not cleaned:
            return b""
        fp = io.BytesIO()
        try:
            tts = gTTS(text=cleaned, lang=lang, slow=False)
            tts.write_to_fp(fp)
            fp.seek(0)
            return fp.read()
        except Exception as exc:
            logger.warning("gTTS speech synthesis failed (offline or network error: %s).", exc)
            # Minimal valid silent MP3 frame header as fallback
            return b"\xff\xfb\x90\x00" + b"\x00" * 32

    @staticmethod
    def synthesize_speech_b64(text: str, lang: str = "en") -> str:
        """Convert coaching text to base64 MP3 data URI for in-browser HTML5 audio playback."""
        mp3_bytes = VoiceEngine.synthesize_speech(text, lang=lang)
        b64 = base64.b64encode(mp3_bytes).decode("ascii")
        return f"data:audio/mp3;base64,{b64}"
