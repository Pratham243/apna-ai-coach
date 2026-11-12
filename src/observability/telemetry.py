"""Observability module integrating Langfuse SDK for tracing LLM and VLM calls."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)


class LangfuseTracer:
    """Manages Langfuse tracing. Silently degrades to a no-op if credentials are absent."""

    def __init__(self) -> None:
        self.public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
        self.secret_key = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
        self.host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com").strip()
        self.enabled = bool(self.public_key and self.secret_key)
        self._client: Any = None

        if self.enabled:
            try:
                from langfuse import Langfuse

                self._client = Langfuse(
                    public_key=self.public_key,
                    secret_key=self.secret_key,
                    host=self.host,
                )
                logger.info("Langfuse tracing enabled on host: %s", self.host)
            except Exception as exc:
                logger.warning("Failed to initialize Langfuse client: %s. Disabling tracing.", exc)
                self.enabled = False
                self._client = None
        else:
            logger.debug("Langfuse keys absent; telemetry is silently disabled.")

    @property
    def client(self) -> Any:
        return self._client

    @contextmanager
    def trace_generation(
        self,
        name: str,
        model: str,
        input_data: Any,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[Any]:
        """Context manager to trace an LLM or VLM generation."""
        if not self.enabled or self._client is None:
            yield None
            return

        ctx = None
        try:
            # Langfuse v5 observation context manager
            ctx = self._client.start_as_current_observation(
                name=name,
                as_type="generation",
                model=model,
                input=input_data,
                metadata=metadata or {},
            )
            obs = ctx.__enter__()
        except Exception as exc:
            logger.debug("Langfuse start observation failed: %s", exc)
            yield None
            return

        try:
            yield obs
        except Exception as exc:
            if hasattr(obs, "update"):
                try:
                    obs.update(level="ERROR", status_message=str(exc))
                except Exception:
                    pass
            raise
        finally:
            if ctx is not None:
                try:
                    ctx.__exit__(None, None, None)
                    self._client.flush()
                except Exception as exc:
                    logger.debug("Langfuse exit/flush error: %s", exc)


# Global singleton tracer
_tracer_instance: LangfuseTracer | None = None


def get_tracer() -> LangfuseTracer:
    """Return the global Langfuse tracer singleton."""
    global _tracer_instance
    if _tracer_instance is None:
        _tracer_instance = LangfuseTracer()
    return _tracer_instance
