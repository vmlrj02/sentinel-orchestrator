"""Prometheus metrics (served at /metrics) plus a LangChain callback that records every LLM call."""
import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult
from prometheus_client import Counter, Gauge, Histogram

TURNS = Counter("sentinel_turns_total", "Chat turns processed", ["channel", "outcome"])
TURN_LATENCY = Histogram(
    "sentinel_turn_latency_seconds", "End-to-end latency of a chat turn", ["channel"],
    buckets=(0.25, 0.5, 1, 2, 3, 5, 8, 13, 21, 34, 60),
)
QUEUE_WAIT = Histogram(
    "sentinel_turn_queue_wait_seconds", "Time a turn waited for a concurrency slot",
    buckets=(0.001, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10),
)
IN_FLIGHT = Gauge("sentinel_turns_in_flight", "Chat turns currently running")
RATE_LIMITED = Counter("sentinel_rate_limited_total", "Turns rejected by the per-user rate limit", ["channel"])
DECISIONS = Counter("sentinel_decisions_total", "Credit decisions made", ["decision"])
FEEDBACK = Counter("sentinel_feedback_total", "User feedback on replies", ["rating"])

LLM_CALLS = Counter("sentinel_llm_calls_total", "LLM calls", ["model", "outcome"])
LLM_LATENCY = Histogram(
    "sentinel_llm_latency_seconds", "LLM call latency", ["model"],
    buckets=(0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32),
)
LLM_TOKENS = Counter("sentinel_llm_tokens_total", "LLM tokens", ["model", "kind"])


class LLMMetricsCallback(AsyncCallbackHandler):
    """Times each model call and counts tokens, keyed by run id."""

    def __init__(self) -> None:
        self._started: dict[UUID, tuple[float, str]] = {}

    async def on_chat_model_start(self, serialized: dict[str, Any], messages: Any, *, run_id: UUID,
                                  metadata: dict[str, Any] | None = None, **kwargs: Any) -> None:
        model = (metadata or {}).get("ls_model_name") or "unknown"
        self._started[run_id] = (time.perf_counter(), model)

    async def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        started, model = self._started.pop(run_id, (time.perf_counter(), "unknown"))
        LLM_LATENCY.labels(model).observe(time.perf_counter() - started)
        LLM_CALLS.labels(model, "ok").inc()
        for generations in response.generations:
            for gen in generations:
                usage = getattr(getattr(gen, "message", None), "usage_metadata", None) or {}
                if usage:
                    LLM_TOKENS.labels(model, "input").inc(usage.get("input_tokens", 0))
                    LLM_TOKENS.labels(model, "output").inc(usage.get("output_tokens", 0))

    async def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        _, model = self._started.pop(run_id, (0.0, "unknown"))
        LLM_CALLS.labels(model, "error").inc()
