"""Langfuse tracing. Each chat turn is one trace; a conversation is a Langfuse session.

The trace id is chosen before the turn runs, so user feedback (thumbs up/down) can be
attached to the exact trace that produced the reply.
"""
import logging
from typing import Any
from uuid import uuid4

from langchain_core.callbacks import BaseCallbackHandler

from .metrics import LLMMetricsCallback

logger = logging.getLogger(__name__)


class Tracer:
    def __init__(self, enabled: bool, environment: str) -> None:
        self.enabled = enabled
        self.environment = environment
        self._client: Any = None
        if enabled:
            from langfuse import get_client
            self._client = get_client()
            logger.info("Langfuse tracing enabled")
        else:
            logger.info("Langfuse tracing disabled (no LANGFUSE_PUBLIC_KEY/SECRET_KEY)")

    def new_trace_id(self) -> str:
        if self.enabled:
            from langfuse import Langfuse
            return Langfuse.create_trace_id()
        return uuid4().hex

    def run_config(self, *, trace_id: str, conversation_id: str, user_id: str, channel: str,
                   prompt_versions: dict[str, str], tags: list[str] | None = None) -> dict[str, Any]:
        """Callbacks + metadata for one graph run. LangGraph passes these down to every node and LLM call."""
        callbacks: list[BaseCallbackHandler] = [LLMMetricsCallback()]
        if self.enabled:
            from langfuse.langchain import CallbackHandler
            callbacks.append(CallbackHandler(trace_context={"trace_id": trace_id}))
        return {
            "callbacks": callbacks,
            "metadata": {
                "langfuse_session_id": conversation_id,
                "langfuse_user_id": user_id,
                "langfuse_trace_name": "chat_turn",
                "langfuse_tags": [f"channel:{channel}", f"env:{self.environment}", *(tags or []),
                                  *(f"{name}:{version}" for name, version in prompt_versions.items())],
                "channel": channel,
                **prompt_versions,
            },
        }

    def score(self, trace_id: str, name: str, value: float, comment: str | None = None) -> None:
        if self.enabled:
            self._client.create_score(trace_id=trace_id, name=name, value=value, comment=comment)

    def trace_url(self, trace_id: str) -> str | None:
        if not self.enabled:
            return None
        try:
            return self._client.get_trace_url(trace_id=trace_id)
        except Exception:
            return None

    def flush(self) -> None:
        if self.enabled:
            self._client.flush()
