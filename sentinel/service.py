"""ChatService: the one entry point every channel (web API, Telegram, evals, load test) goes through.

It owns the production concerns around a graph run:
- per-conversation lock, so two fast messages in one chat are processed in order
- a concurrency cap (backpressure towards the LLM provider), with queue-wait metrics
- per-user rate limit, input validation, overall turn timeout
- tracing, metrics and structured logs for every turn
"""
import asyncio
import logging
import time
import weakref
from collections import OrderedDict, deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from .agent.state import REQUIRED_FIELDS, application_fields
from .config import Settings
from .logging_setup import channel_var, conversation_id_var, trace_id_var
from .metrics import IN_FLIGHT, QUEUE_WAIT, RATE_LIMITED, TURN_LATENCY, TURNS
from .observability import Tracer
from .prompts import prompt_versions

logger = logging.getLogger(__name__)

Event = dict[str, Any]


class RateLimiter:
    """Sliding window per key. In-process; with many instances this moves to Redis/DynamoDB."""

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._hits: dict[str, deque[float]] = {}

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        hits = self._hits.setdefault(key, deque())
        while hits and now - hits[0] > 60:
            hits.popleft()
        if len(hits) >= self.per_minute:
            return False
        hits.append(now)
        return True


class SeenIds:
    """Bounded set of recently seen ids, for de-duplicating retried deliveries."""

    def __init__(self, maxsize: int = 10_000) -> None:
        self._ids: OrderedDict[str, None] = OrderedDict()
        self.maxsize = maxsize

    def check_and_add(self, key: str) -> bool:
        """True if the id is new."""
        if key in self._ids:
            return False
        self._ids[key] = None
        if len(self._ids) > self.maxsize:
            self._ids.popitem(last=False)
        return True


@dataclass
class TurnResult:
    conversation_id: str
    trace_id: str
    messages: list[str] = field(default_factory=list)
    assessment: dict[str, Any] | None = None
    error: str | None = None
    explained: bool = False  # the analyst wrote this turn's reply (decision explanation or follow-up)


class ChatService:
    def __init__(self, graph: Any, settings: Settings, tracer: Tracer) -> None:
        self.graph = graph
        self.settings = settings
        self.tracer = tracer
        self._slots = asyncio.Semaphore(settings.max_concurrent_turns)
        self._locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()
        self._rate = RateLimiter(settings.rate_limit_per_minute)
        self.seen_messages = SeenIds()

    def _lock(self, conversation_id: str) -> asyncio.Lock:
        lock = self._locks.get(conversation_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[conversation_id] = lock
        return lock

    async def stream_turn(self, conversation_id: str, user_id: str, text: str, channel: str,
                          tags: list[str] | None = None) -> AsyncIterator[Event]:
        """Runs one turn and yields events: status, assessment, token, message, then done or error."""
        trace_id = self.tracer.new_trace_id()
        conversation_id_var.set(conversation_id)
        trace_id_var.set(trace_id)
        channel_var.set(channel)

        text = text.strip()
        if not text:
            yield {"type": "error", "code": "empty_message", "message": "Please type a message."}
            return
        if len(text) > self.settings.max_message_chars:
            yield {"type": "error", "code": "message_too_long",
                   "message": f"Please keep messages under {self.settings.max_message_chars} characters."}
            return
        if not self._rate.allow(user_id):
            RATE_LIMITED.labels(channel).inc()
            TURNS.labels(channel, "rate_limited").inc()
            yield {"type": "error", "code": "rate_limited",
                   "message": "You're sending messages too fast. Please wait a moment."}
            return

        started = time.perf_counter()
        outcome = "ok"
        config = {
            "configurable": {"thread_id": conversation_id},
            "recursion_limit": 10,  # interviewer -> scoring -> analyst is 3 steps; anything more is a bug
            **self.tracer.run_config(trace_id=trace_id, conversation_id=conversation_id, user_id=user_id,
                                     channel=channel, prompt_versions=prompt_versions(), tags=tags),
        }
        try:
            async with self._lock(conversation_id):
                queued = time.perf_counter()
                async with self._slots:
                    QUEUE_WAIT.observe(time.perf_counter() - queued)
                    IN_FLIGHT.inc()
                    try:
                        async with asyncio.timeout(self.settings.turn_timeout_s):
                            async for event in self._run_graph(text, config):
                                yield event
                    finally:
                        IN_FLIGHT.dec()
            yield {"type": "done", "conversation_id": conversation_id, "trace_id": trace_id,
                   "trace_url": self.tracer.trace_url(trace_id)}
        except TimeoutError:
            outcome = "timeout"
            logger.error("Turn timed out", extra={"fields": {"timeout_s": self.settings.turn_timeout_s}})
            yield {"type": "error", "code": "timeout", "message": "That took too long. Please try again.",
                   "trace_id": trace_id}
        except Exception:
            outcome = "error"
            logger.exception("Turn failed")
            yield {"type": "error", "code": "internal", "trace_id": trace_id,
                   "message": "Sorry, something went wrong on my side. Please try again."}
        finally:
            elapsed = time.perf_counter() - started
            TURNS.labels(channel, outcome).inc()
            TURN_LATENCY.labels(channel).observe(elapsed)
            logger.info("Turn finished", extra={"fields": {"outcome": outcome, "latency_ms": round(elapsed * 1000),
                                                           "user_id": user_id}})
            if self.settings.flush_traces_each_turn:
                await asyncio.to_thread(self.tracer.flush)

    async def _run_graph(self, text: str, config: dict[str, Any]) -> AsyncIterator[Event]:
        async for mode, chunk in self.graph.astream(
            {"messages": [HumanMessage(text)]}, config, stream_mode=["custom", "updates"]
        ):
            if mode == "custom":
                yield chunk
                continue
            for node, update in chunk.items():
                for message in (update or {}).get("messages", []):
                    if isinstance(message, AIMessage):
                        yield {"type": "message", "text": message.content, "node": node}

    async def run_turn(self, conversation_id: str, user_id: str, text: str, channel: str,
                       tags: list[str] | None = None) -> TurnResult:
        """Non-streaming version for channels that send whole messages (Telegram, evals)."""
        result = TurnResult(conversation_id=conversation_id, trace_id="")
        async for event in self.stream_turn(conversation_id, user_id, text, channel, tags):
            if event["type"] == "message":
                result.messages.append(event["text"])
                result.explained = event.get("node") == "analyst"
            elif event["type"] == "assessment":
                result.assessment = event["assessment"]
            elif event["type"] == "error":
                result.error = event["code"]
                result.messages.append(event["message"])
            if "trace_id" in event:
                result.trace_id = event["trace_id"]
        return result

    async def get_state(self, conversation_id: str) -> dict[str, Any]:
        snapshot = await self.graph.aget_state({"configurable": {"thread_id": conversation_id}})
        values = snapshot.values or {}
        return {
            "conversation_id": conversation_id,
            "application": application_fields(values),
            "missing": [f for f in REQUIRED_FIELDS if values.get(f) is None],
            "details_confirmed": bool(values.get("details_confirmed")),
            "assessment": values.get("assessment"),
            "messages": [{"role": "user" if isinstance(m, HumanMessage) else "assistant", "text": m.content}
                         for m in values.get("messages", [])],
        }

    async def reset(self, conversation_id: str) -> None:
        """Start a fresh application in the same conversation (Telegram /start)."""
        config = {"configurable": {"thread_id": conversation_id}}
        cleared: dict[str, Any] = {f: None for f in [*REQUIRED_FIELDS, "loan_percent_income", "assessment"]}
        async with self._lock(conversation_id):
            await self.graph.aupdate_state(
                config,
                {**cleared, "details_confirmed": False, "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]},
                as_node="interviewer",
            )

    def record_feedback(self, trace_id: str, rating: int, comment: str | None = None) -> None:
        from .metrics import FEEDBACK
        FEEDBACK.labels("up" if rating > 0 else "down").inc()
        self.tracer.score(trace_id, "user_feedback", 1.0 if rating > 0 else 0.0, comment)
