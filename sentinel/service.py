"""ChatService: the one entry point every channel (web API, Telegram, evals, load test) goes through.

It owns the production concerns around a graph run:
- per-conversation lock, so two fast messages in one chat are processed in order
- a concurrency cap (backpressure towards the LLM provider), with queue-wait metrics
- per-user rate limit, input validation, overall turn timeout
- human review: pausing while an underwriter decides, resuming the graph, notifying the applicant
- tracing, metrics and structured logs for every turn
"""
import asyncio
import logging
import time
from collections import OrderedDict, deque
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import Command

from .agent.dialogue import CHANGE_OPTIONS, clarify_message, field_for, next_slot, question_message
from .agent.graph import RESUME_OPTIONS
from .agent.state import FIELD_LABELS, REQUIRED_FIELDS, SUMMARY_LABELS, application_fields
from .commands import match_command
from .config import Settings
from .guard import only_removed, redact
from .locks import ConversationLocks, InProcessLocks
from .logging_setup import channel_var, conversation_id_var, trace_id_var
from .metrics import (
    IN_FLIGHT,
    QUEUE_WAIT,
    RATE_LIMITED,
    record_command,
    record_feedback,
    record_funnel,
    record_slot,
    record_turn,
)
from .observability import Tracer
from .prompts import prompt_versions
from .store import ApplicationRecord, ApplicationStore
from .tenants import DEFAULT_TENANT, TENANTS, get_tenant

logger = logging.getLogger(__name__)

try:  # provider errors that survive the SDK's retries and the fallback model
    from openai import APIConnectionError, APIStatusError, RateLimitError
    _PROVIDER_ERRORS: tuple[type[Exception], ...] = (RateLimitError, APIConnectionError, APIStatusError)
except ImportError:  # pragma: no cover
    _PROVIDER_ERRORS = ()

Event = dict[str, Any]
Notifier = Callable[[str, list[str]], Awaitable[None]]


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


class ReviewError(Exception):
    pass


@dataclass
class TurnResult:
    conversation_id: str
    trace_id: str
    messages: list[str] = field(default_factory=list)
    assessment: dict[str, Any] | None = None
    error: str | None = None
    explained: bool = False  # the advisor wrote this turn's reply (decision explanation or answer)
    tools: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    what_if: list[dict[str, Any]] = field(default_factory=list)
    options: list[str] = field(default_factory=list)  # quick replies offered with the last message


class ChatService:
    def __init__(self, graph: Any, settings: Settings, tracer: Tracer, store: ApplicationStore,
                 locks: ConversationLocks | None = None) -> None:
        self.graph = graph
        self.settings = settings
        self.tracer = tracer
        self.store = store
        self._slots = asyncio.Semaphore(settings.max_concurrent_turns)
        self.locks: ConversationLocks = locks or InProcessLocks()
        self._rate = RateLimiter(settings.rate_limit_per_minute)
        self._notifiers: list[Notifier] = []
        self.seen_messages = SeenIds()

    def on_notify(self, notifier: Notifier) -> None:
        """Channels register here to push messages that arrive outside a turn (underwriter decisions)."""
        self._notifiers.append(notifier)

    def _config(self, conversation_id: str, trace_id: str, user_id: str, channel: str,
                tags: list[str] | None = None) -> dict[str, Any]:
        return {
            "configurable": {"thread_id": conversation_id, "trace_id": trace_id},
            "recursion_limit": 15,  # longest legit path is ~9 steps (3 tool rounds); more is a bug
            **self.tracer.run_config(trace_id=trace_id, conversation_id=conversation_id, user_id=user_id,
                                     channel=channel, prompt_versions=prompt_versions(), tags=tags),
        }

    async def stream_turn(self, conversation_id: str, user_id: str, text: str, channel: str,
                          tenant_id: str | None = None, tags: list[str] | None = None) -> AsyncIterator[Event]:
        """Runs one turn and yields events: status, citations, tool_call, assessment, token, message,
        then done or error."""
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
        if tenant_id is not None and tenant_id not in TENANTS:
            yield {"type": "error", "code": "unknown_lender", "message": "Unknown lender."}
            return
        if not self._rate.allow(user_id):
            RATE_LIMITED.labels(channel).inc()
            record_turn(channel, "rate_limited", 0.0)
            yield {"type": "error", "code": "rate_limited",
                   "message": "You're sending messages too fast. Please wait a moment."}
            return

        started = time.perf_counter()
        outcome = "ok"
        config = self._config(conversation_id, trace_id, user_id, channel, tags)
        try:
            async with self.locks.hold(conversation_id):
                snapshot = await self.graph.aget_state(config)
                values = snapshot.values or {}
                lender = get_tenant(values.get("tenant_id") or tenant_id).name
                turns = sum(isinstance(m, HumanMessage) for m in values.get("messages", []))
                if turns == 0:
                    record_funnel("started")
                # Card, account and ID numbers are removed before the LLM, traces or logs see them
                text, removed = redact(text)
                if removed:
                    record_command("sensitive_data_removed")
                    yield {"type": "message", "node": "service", "text": (
                        "For your safety, please don't share card, bank account or ID numbers here. "
                        "I've removed that from your message and won't store it.")}
                command = match_command(text) if not only_removed(text) else None
                if only_removed(text):
                    outcome = "sensitive_data"
                elif snapshot.interrupts:
                    outcome = "under_review"
                    yield {"type": "message", "node": "service", "text": (
                        "Your application is with an underwriter right now. I'll message you here as soon as "
                        "they decide.")}
                elif command:
                    outcome = f"command_{command[0]}"
                    reply = await self._command(command[0], command[1], text, config, values, lender)
                    yield {"type": "message", "node": "service", "text": reply.content,
                           "options": reply.response_metadata.get("options", [])}
                elif turns >= self.settings.max_turns_per_conversation:
                    # Cost ceiling: no LLM call once a conversation hits the limit
                    outcome = "turn_limit"
                    yield {"type": "message", "node": "service", "text": (
                        "This conversation has reached its length limit. Please tap Start over to begin again.")}
                elif values.get("stalled_turns", 0) >= self.settings.stall_limit and text not in RESUME_OPTIONS:
                    # Several turns without progress: answer from a template instead of paying for LLM calls
                    # until the person picks a way forward
                    outcome = "stalled"
                    yield {"type": "message", "node": "service", "options": RESUME_OPTIONS, "text": (
                        f"It looks like a loan isn't what you need right now. I can help you apply for a "
                        f"personal loan or answer questions about {lender}'s loan policies. For anything else, "
                        f"please contact {lender} directly.")}
                else:
                    if values.get("stalled_turns", 0) >= self.settings.stall_limit:
                        await self.graph.aupdate_state(config, {"stalled_turns": 0}, as_node="interviewer")
                    graph_input: dict[str, Any] = {"messages": [HumanMessage(text)]}
                    if not snapshot.values.get("tenant_id"):
                        graph_input["tenant_id"] = tenant_id or DEFAULT_TENANT
                    async for event in self._run(graph_input, config):
                        yield event
            self.tracer.publish(trace_id)
            yield {"type": "done", "conversation_id": conversation_id, "trace_id": trace_id,
                   "trace_url": self.tracer.trace_url(trace_id)}
        except _PROVIDER_ERRORS as e:
            # Quota exhausted or provider down after retries and the fallback model: say so plainly
            outcome = "llm_unavailable"
            logger.error("LLM provider unavailable",
                         extra={"fields": {"error": type(e).__name__, "detail": str(e)[:300]}})
            yield {"type": "error", "code": "llm_unavailable", "trace_id": trace_id,
                   "message": "Our AI assistant is temporarily unavailable. Please try again in a few minutes."}
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
            record_turn(channel, outcome, elapsed)
            logger.info("Turn finished", extra={"fields": {"outcome": outcome, "latency_ms": round(elapsed * 1000),
                                                           "user_id": user_id}})
            if self.settings.flush_traces_each_turn:
                await asyncio.to_thread(self.tracer.flush)

    async def _run(self, graph_input: Any, config: dict[str, Any]) -> AsyncIterator[Event]:
        """One graph run under the concurrency cap and timeout, translated into channel events."""
        queued = time.perf_counter()
        async with self._slots:
            QUEUE_WAIT.observe(time.perf_counter() - queued)
            IN_FLIGHT.inc()
            try:
                async with asyncio.timeout(self.settings.turn_timeout_s):
                    async for mode, chunk in self.graph.astream(graph_input, config,
                                                                stream_mode=["custom", "updates"]):
                        if mode == "custom":
                            yield chunk
                            continue
                        for node, update in chunk.items():
                            if node == "__interrupt__":
                                yield {"type": "under_review"}
                                continue
                            for message in (update or {}).get("messages", []):
                                if isinstance(message, AIMessage) and message.content:
                                    yield {"type": "message", "text": message.content, "node": node,
                                           "options": message.response_metadata.get("options", [])}
            finally:
                IN_FLIGHT.dec()

    async def run_turn(self, conversation_id: str, user_id: str, text: str, channel: str,
                       tenant_id: str | None = None, tags: list[str] | None = None) -> TurnResult:
        """Non-streaming version for channels that send whole messages (Telegram, evals)."""
        return await _collect(conversation_id, self.stream_turn(conversation_id, user_id, text, channel,
                                                                tenant_id, tags))

    # --- Human review -----------------------------------------------------------------------------

    async def list_reviews(self, tenant_id: str | None = None) -> list[ApplicationRecord]:
        return await self.store.list(status="pending_review", tenant_id=tenant_id)

    async def submit_review(self, application_id: str, decision: Literal["approved", "declined"],
                            reviewer: str, note: str | None = None) -> TurnResult:
        """Underwriter decision: resumes the paused graph, which records it and explains it to the applicant."""
        record = await self.store.get(application_id)
        if record is None or record.status != "pending_review":
            raise ReviewError(f"{application_id} is not awaiting review")
        trace_id = self.tracer.new_trace_id()
        config = self._config(record.conversation_id, trace_id, f"underwriter-{reviewer}", "review",
                              tags=["human_review"])
        async with self.locks.hold(record.conversation_id):
            snapshot = await self.graph.aget_state(config)
            if not snapshot.interrupts:
                raise ReviewError(f"{application_id} has no paused review")
            resume: Command[Any] = Command(resume={"decision": decision, "reviewer": reviewer, "note": note})
            result = await _collect(record.conversation_id, self._run(resume, config))
        result.trace_id = trace_id
        logger.info("Review submitted", extra={"fields": {"application_id": application_id, "decision": decision,
                                                          "reviewer": reviewer}})
        for notify in self._notifiers:
            try:
                await notify(record.conversation_id, result.messages)
            except Exception:
                logger.exception("Notifier failed")
        return result

    # --- Conversation state -------------------------------------------------------------------------

    async def get_state(self, conversation_id: str) -> dict[str, Any]:
        snapshot = await self.graph.aget_state({"configurable": {"thread_id": conversation_id}})
        values = snapshot.values or {}
        missing = [f for f in REQUIRED_FIELDS if values.get(f) is None]
        if snapshot.interrupts:
            status = "under_review"
        elif values.get("final_decision"):
            status = "decided"
        elif values.get("messages") and not missing:
            status = "confirming"
        else:
            status = "collecting"
        tenant = get_tenant(values.get("tenant_id"))
        return {
            "conversation_id": conversation_id,
            "status": status,
            "lender": {"id": tenant.id, "name": tenant.name},
            "application_id": values.get("application_id"),
            "application": application_fields(values),
            "missing": missing,
            "assessment": values.get("assessment"),
            "final_decision": values.get("final_decision"),
            "messages": [{"role": "user" if isinstance(m, HumanMessage) else "assistant", "text": m.content,
                          "options": getattr(m, "response_metadata", {}).get("options", [])}
                         for m in values.get("messages", [])
                         if isinstance(m, HumanMessage) or (isinstance(m, AIMessage) and m.content)],
        }

    async def reset(self, conversation_id: str) -> None:
        """Start a fresh application in the same conversation (Telegram /start)."""
        config = {"configurable": {"thread_id": conversation_id}}
        async with self.locks.hold(conversation_id):
            snapshot = await self.graph.aget_state(config)
            if snapshot.interrupts:
                raise ReviewError("This application is with an underwriter and can't be restarted yet.")
            await self._clear(config)

    async def _clear(self, config: dict[str, Any], keep: list[BaseMessage] | None = None) -> None:
        cleared: dict[str, Any] = {f: None for f in [*REQUIRED_FIELDS, "loan_percent_income", "assessment",
                                                     "final_decision", "application_id", "applicant_question",
                                                     "pending_slot"]}
        await self.graph.aupdate_state(
            config, {**cleared, "details_confirmed": False, "stalled_turns": 0, "slot_attempts": {}, "deferred": [],
                     "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *(keep or [])]},
            as_node="interviewer",
        )

    async def _command(self, name: str, what: str, text: str, config: dict[str, Any],
                       values: dict[str, Any], lender: str) -> AIMessage:
        """Global commands: answered in code (no LLM) and saved to the conversation like any other turn."""
        record_command(name)
        updates: dict[str, Any] = {}
        decided = bool(values.get("final_decision"))
        if name == "start_over":
            reply = question_message({}, prefix="Okay, let's start again.")
            await self._clear(config, keep=[HumanMessage(text), reply])
            return reply
        if name == "clarify":
            reply = clarify_message(values)
        elif name == "help":
            reply = AIMessage(
                f"I can help you apply for a personal loan with {lender} in a few short questions, and answer "
                f"questions about {lender}'s loan policies. You can say \"start over\", \"what do you have so "
                f"far?\", \"change my income\" or \"skip\" at any time.",
                response_metadata={"options": ["Apply for a loan", "What documents do I need?"]})
        elif name == "summary":
            collected = application_fields(values)
            lines = [f"- {SUMMARY_LABELS[f]}: {v}" for f, v in collected.items() if f in SUMMARY_LABELS]
            missing = [FIELD_LABELS[f] for f in REQUIRED_FIELDS if values.get(f) is None]
            reply = AIMessage(
                ("Here's what I have so far:\n" + "\n".join(lines) if lines else "I don't have any details yet.")
                + (f"\n\nStill needed: {', '.join(missing)}." if missing else ""))
        elif name == "human":
            record_funnel("handoff_requested")
            reply = AIMessage(
                f"I can't connect you to a person from this chat yet. You can contact {lender}'s loan team "
                "directly, or carry on here and I'll do my best to help.",
                response_metadata={"options": ["Apply for a loan", "What documents do I need?"]})
        elif decided and name in ("skip", "change", "apply"):
            reply = AIMessage("Your application has already been decided. Tap Start over to begin a new one.",
                              response_metadata={"options": ["Start over"]})
        elif name == "skip":
            slot = values.get("pending_slot")
            if slot in REQUIRED_FIELDS:
                record_slot(slot, "skipped")
                updates["deferred"] = [*(values.get("deferred") or []), slot]
            merged = {**values, **updates}
            updates["pending_slot"] = next_slot(merged)
            reply = question_message(merged, prefix="No problem, we'll come back to that.")
        elif name == "change":
            field = field_for(what)
            if field is None:
                reply = AIMessage("What would you like to change?", response_metadata={"options": CHANGE_OPTIONS})
            else:
                updates.update({field: None, "details_confirmed": False, "pending_slot": field})
                reply = question_message({**values, **updates}, prefix="Sure.", slot=field)
        else:  # apply
            updates.update({"stalled_turns": 0})
            reply = question_message({**values, **updates}, prefix="Great, let's get started.")
        await self.graph.aupdate_state(config, {**updates, "messages": [HumanMessage(text), reply]},
                                       as_node="interviewer")
        return reply

    def record_feedback(self, trace_id: str, rating: int, comment: str | None = None) -> None:
        record_feedback(rating)
        self.tracer.score(trace_id, "user_feedback", 1.0 if rating > 0 else 0.0, comment)


async def _collect(conversation_id: str, events: AsyncIterator[Event]) -> TurnResult:
    result = TurnResult(conversation_id=conversation_id, trace_id="")
    async for event in events:
        kind = event["type"]
        if kind == "message":
            result.messages.append(event["text"])
            result.explained = event.get("node") == "advisor"
            result.options = event.get("options") or []
        elif kind == "assessment":
            result.assessment = event
        elif kind == "tool_call":
            result.tools.append(event["name"])
        elif kind == "citations":
            result.citations.extend(item["id"] for item in event["items"])
        elif kind == "what_if":
            result.what_if.append(event["result"])
        elif kind == "error":
            result.error = event["code"]
            result.messages.append(event["message"])
        if "trace_id" in event:
            result.trace_id = event["trace_id"]
    return result
