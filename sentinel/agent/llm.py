"""The two things the agent needs from a language model, behind one small interface.

Nodes depend on AgentLLM, not on a vendor SDK, so tests and load tests can swap in FakeAgentLLM,
and each step can use a different model (model routing).
"""
import asyncio
import json
import re
from collections.abc import AsyncIterator, Sequence
from typing import Any, Protocol

from langchain_core.messages import AIMessageChunk, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from .state import REQUIRED_FIELDS, SentinelResponse


class AgentLLM(Protocol):
    model_names: dict[str, str]  # step -> model, recorded on traces

    async def interview(self, messages: list[BaseMessage]) -> SentinelResponse:
        """Extract application fields from the conversation and write the next question."""
        ...

    def advise(self, messages: list[BaseMessage], tools: Sequence[BaseTool]) -> AsyncIterator[AIMessageChunk]:
        """Stream an answer, possibly with tool calls (empty tools = must answer directly)."""
        ...


class OpenAIAgentLLM:
    def __init__(self, interviewer_model: str, advisor_model: str, api_key: str | None,
                 fallback_model: str | None = None, timeout_s: float = 30, max_retries: int = 3) -> None:
        self.model_names = {"interviewer": interviewer_model, "advisor": advisor_model}

        def chat(name: str) -> ChatOpenAI:
            # max_retries: the SDK retries 429/5xx/connection errors with exponential backoff
            return ChatOpenAI(model=name, api_key=api_key, timeout=timeout_s,  # type: ignore[arg-type]
                              max_retries=max_retries, stream_usage=True)

        self._advisor = chat(advisor_model)
        self._backup = chat(fallback_model) if fallback_model and fallback_model != advisor_model else None
        self._interviewer: Runnable[Any, Any] = chat(interviewer_model).with_structured_output(SentinelResponse)
        if fallback_model and fallback_model != interviewer_model:
            # Provider errors that survive retries: same call on a second model
            self._interviewer = self._interviewer.with_fallbacks(
                [chat(fallback_model).with_structured_output(SentinelResponse)])

    async def interview(self, messages: list[BaseMessage]) -> SentinelResponse:
        return await self._interviewer.ainvoke(messages, config={"run_name": "interviewer_llm"})

    async def advise(self, messages: list[BaseMessage], tools: Sequence[BaseTool]) -> AsyncIterator[AIMessageChunk]:
        model: Runnable[Any, Any] = self._advisor.bind_tools(tools) if tools else self._advisor
        if self._backup is not None:
            model = model.with_fallbacks([self._backup.bind_tools(tools) if tools else self._backup])
        async for chunk in model.astream(messages, config={"run_name": "advisor_llm"}):
            yield chunk


_PAIR = re.compile(r"(\w+)=([^\s,]+)")
_YES = re.compile(r"^\s*(yes|y|yep|correct|confirm(ed)?)\b", re.I)


class FakeAgentLLM:
    """Deterministic stand-in with simulated latency, for unit tests and load tests.

    interview: understands `field=value` pairs, "yes", and "question: ..." for policy questions.
    advise: "search: <query>" or "whatif: loan_amnt=8000" trigger one tool call; otherwise it answers.
    """

    def __init__(self, latency_ms: int = 0) -> None:
        self.model_names = {"interviewer": "fake", "advisor": "fake"}
        self.latency_s = latency_ms / 1000
        self.calls = 0

    async def interview(self, messages: list[BaseMessage]) -> SentinelResponse:
        self.calls += 1
        await asyncio.sleep(self.latency_s)
        text = _last_human(messages)
        fields: dict[str, Any] = {k: v for k, v in _PAIR.findall(text) if k in REQUIRED_FIELDS}
        if _YES.match(text):
            fields["details_confirmed"] = True
        if "outofscope:" in text:
            fields["out_of_scope"] = True
        if "question:" in text:
            fields["applicant_question"] = text.split("question:", 1)[1].strip()
        return SentinelResponse(**fields)

    async def advise(self, messages: list[BaseMessage], tools: Sequence[BaseTool]) -> AsyncIterator[AIMessageChunk]:
        self.calls += 1
        await asyncio.sleep(self.latency_s)
        text = _last_human(messages)
        tool_used = isinstance(messages[-1], ToolMessage)
        names = {t.name for t in tools}
        if not tool_used and "search:" in text and "search_policy" in names:
            args = {"query": text.split("search:", 1)[1].strip()}
            yield AIMessageChunk(content="", tool_call_chunks=[
                {"name": "search_policy", "args": json.dumps(args), "id": "call_1", "index": 0}])
            return
        if not tool_used and "whatif:" in text and "what_if_score" in names:
            changes = {k: float(v) for k, v in _PAIR.findall(text.split("whatif:", 1)[1])}
            yield AIMessageChunk(content="", tool_call_chunks=[
                {"name": "what_if_score", "args": json.dumps(changes), "id": "call_2", "index": 0}])
            return
        for word in "Here is what I found based on the decision record .".split():
            yield AIMessageChunk(content=word + " ")


def _last_human(messages: list[BaseMessage]) -> str:
    return next((str(m.content) for m in reversed(messages) if isinstance(m, HumanMessage)), "")
