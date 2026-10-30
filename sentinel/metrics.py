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
INVALID_CITATIONS = Counter("sentinel_invalid_citations_total", "Citations removed because they were not retrieved")
FEEDBACK = Counter("sentinel_feedback_total", "User feedback on replies", ["rating"])

LLM_CALLS = Counter("sentinel_llm_calls_total", "LLM calls", ["model", "outcome"])
LLM_LATENCY = Histogram(
    "sentinel_llm_latency_seconds", "LLM call latency", ["model"],
    buckets=(0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32),
)
LLM_TOKENS = Counter("sentinel_llm_tokens_total", "LLM tokens", ["model", "kind"])


# On Lambda each instance has its own Prometheus registry, so metrics are also written as CloudWatch
# Embedded Metric Format log lines and aggregated by CloudWatch across instances.
_emf_enabled = False


def enable_emf() -> None:
    global _emf_enabled
    _emf_enabled = True


def _emf(metrics: dict[str, tuple[float, str]], dimensions: dict[str, str] | None = None) -> None:
    if _emf_enabled:
        from .aws import emit_emf
        emit_emf(metrics, dimensions)


def record_turn(channel: str, outcome: str, seconds: float) -> None:
    TURNS.labels(channel, outcome).inc()
    TURN_LATENCY.labels(channel).observe(seconds)
    _emf({"Turns": (1, "Count"), "TurnLatency": (round(seconds * 1000), "Milliseconds")}, {"Outcome": outcome})


def record_decision(decision: str) -> None:
    DECISIONS.labels(decision).inc()
    _emf({"Decisions": (1, "Count")}, {"Decision": decision})


def record_feedback(rating: int) -> None:
    label = "up" if rating > 0 else "down"
    FEEDBACK.labels(label).inc()
    _emf({"Feedback": (1, "Count")}, {"Rating": label})


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
        elapsed = time.perf_counter() - started
        LLM_LATENCY.labels(model).observe(elapsed)
        LLM_CALLS.labels(model, "ok").inc()
        tokens = 0
        for generations in response.generations:
            for gen in generations:
                usage = getattr(getattr(gen, "message", None), "usage_metadata", None) or {}
                if usage:
                    LLM_TOKENS.labels(model, "input").inc(usage.get("input_tokens", 0))
                    LLM_TOKENS.labels(model, "output").inc(usage.get("output_tokens", 0))
                    tokens += usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        _emf({"LLMCalls": (1, "Count"), "LLMTokens": (tokens, "Count"),
              "LLMLatency": (round(elapsed * 1000), "Milliseconds")})

    async def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        _, model = self._started.pop(run_id, (0.0, "unknown"))
        LLM_CALLS.labels(model, "error").inc()
        _emf({"LLMErrors": (1, "Count")})


def _quantile(buckets: list[tuple[float, float]], q: float) -> float | None:
    """Approximate quantile from cumulative histogram buckets (upper bound of the bucket it falls in)."""
    total = buckets[-1][1] if buckets else 0
    if not total:
        return None
    for upper, count in buckets:
        if count >= q * total:
            return upper
    return buckets[-1][0]


def snapshot() -> dict[str, Any]:
    """Live numbers for the UI's System panel. Same data Prometheus/CloudWatch would scrape."""
    from prometheus_client import REGISTRY

    turns: dict[str, float] = {}
    decisions: dict[str, float] = {}
    tokens: dict[str, float] = {}
    feedback: dict[str, float] = {}
    llm_errors = llm_calls = in_flight = rate_limited = 0.0
    latency: dict[float, float] = {}
    for metric in REGISTRY.collect():
        for s in metric.samples:
            if s.name == "sentinel_turns_total":
                turns[s.labels["outcome"]] = turns.get(s.labels["outcome"], 0) + s.value
            elif s.name == "sentinel_turn_latency_seconds_bucket" and s.labels.get("channel") != "eval":
                bound = float(s.labels["le"])
                latency[bound] = latency.get(bound, 0) + s.value
            elif s.name == "sentinel_decisions_total":
                decisions[s.labels["decision"]] = s.value
            elif s.name == "sentinel_llm_tokens_total":
                tokens[s.labels["kind"]] = tokens.get(s.labels["kind"], 0) + s.value
            elif s.name == "sentinel_llm_calls_total":
                llm_calls += s.value
                llm_errors += s.value if s.labels["outcome"] == "error" else 0
            elif s.name == "sentinel_turns_in_flight":
                in_flight = s.value
            elif s.name == "sentinel_rate_limited_total":
                rate_limited += s.value
            elif s.name == "sentinel_feedback_total":
                feedback[s.labels["rating"]] = s.value
    buckets = sorted(latency.items())
    total_turns = sum(turns.values())
    return {
        "turns": {"total": total_turns, **turns},
        "latency_s": {"p50": _quantile(buckets, 0.5), "p95": _quantile(buckets, 0.95)},
        "in_flight": in_flight,
        "rate_limited": rate_limited,
        "llm": {"calls": llm_calls, "errors": llm_errors, "tokens": tokens,
                "tokens_per_turn": round(sum(tokens.values()) / total_turns) if total_turns else None},
        "decisions": decisions,
        "feedback": feedback,
    }
