"""The two things the agent needs from a language model, behind one small interface.

Nodes depend on AgentLLM, not on a vendor SDK, so tests and load tests can swap in FakeAgentLLM.
"""
import asyncio
import re
from collections.abc import AsyncIterator
from typing import Any, Protocol

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from .state import REQUIRED_FIELDS, SentinelResponse


class AgentLLM(Protocol):
    model_name: str

    async def interview(self, messages: list[BaseMessage]) -> SentinelResponse:
        """Extract application fields from the conversation and write the next question."""
        ...

    def explain(self, messages: list[BaseMessage]) -> AsyncIterator[str]:
        """Stream a plain-English explanation of the credit decision."""
        ...


class OpenAIAgentLLM:
    def __init__(self, model: str, api_key: str | None, fallback_model: str | None = None,
                 timeout_s: float = 30, max_retries: int = 3) -> None:
        self.model_name = model

        def chat(name: str) -> ChatOpenAI:
            # max_retries: the SDK retries 429/5xx/connection errors with exponential backoff
            return ChatOpenAI(model=name, api_key=api_key, timeout=timeout_s, max_retries=max_retries)  # type: ignore[arg-type]

        primary = chat(model)
        self._interviewer: Runnable[Any, Any] = primary.with_structured_output(SentinelResponse)
        self._analyst: Runnable[Any, Any] = primary
        if fallback_model and fallback_model != model:
            # Provider outage or persistent errors on the primary model: same call on a second model
            backup = chat(fallback_model)
            self._interviewer = self._interviewer.with_fallbacks([backup.with_structured_output(SentinelResponse)])
            self._analyst = self._analyst.with_fallbacks([backup])

    async def interview(self, messages: list[BaseMessage]) -> SentinelResponse:
        return await self._interviewer.ainvoke(messages, config={"run_name": "interviewer_llm"})

    async def explain(self, messages: list[BaseMessage]) -> AsyncIterator[str]:
        async for chunk in self._analyst.astream(messages, config={"run_name": "analyst_llm"}):
            if isinstance(chunk.content, str) and chunk.content:
                yield chunk.content


_PAIR = re.compile(r"(\w+)=([^\s,]+)")
_YES = re.compile(r"^\s*(yes|y|yep|correct|confirm(ed)?)\b", re.I)


class FakeAgentLLM:
    """Deterministic stand-in with simulated latency. Understands `field=value` pairs and "yes".

    Used by unit tests and by the load test, which measures the system (concurrency, state,
    streaming, scoring) rather than OpenAI.
    """

    model_name = "fake"

    def __init__(self, latency_ms: int = 0) -> None:
        self.latency_s = latency_ms / 1000
        self.calls = 0

    async def interview(self, messages: list[BaseMessage]) -> SentinelResponse:
        self.calls += 1
        await asyncio.sleep(self.latency_s)
        text = next((str(m.content) for m in reversed(messages) if isinstance(m, HumanMessage)), "")
        fields: dict[str, Any] = {k: v for k, v in _PAIR.findall(text) if k in REQUIRED_FIELDS}
        if _YES.match(text):
            fields["details_confirmed"] = True
        return SentinelResponse(assistant_message="Thanks. What else can you tell me?", **fields)

    async def explain(self, messages: list[BaseMessage]) -> AsyncIterator[str]:
        self.calls += 1
        for word in "Your application has been assessed by the credit model .".split():
            await asyncio.sleep(self.latency_s / 10)
            yield word + " "
