"""Builds the runtime (LLM, checkpointer, graph, service) from settings. Shared by the API, evals and tests."""
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import MemorySaver

from .agent.graph import build_graph
from .agent.llm import AgentLLM, FakeAgentLLM, OpenAIAgentLLM
from .config import Settings
from .observability import Tracer
from .scoring.model import LocalScorer
from .service import ChatService


@dataclass
class Runtime:
    settings: Settings
    service: ChatService
    tracer: Tracer
    llm: AgentLLM


def make_llm(settings: Settings) -> AgentLLM:
    if settings.llm_provider == "fake":
        return FakeAgentLLM(latency_ms=settings.fake_llm_latency_ms)
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set (or use LLM_PROVIDER=fake)")
    return OpenAIAgentLLM(settings.openai_model, settings.openai_api_key, settings.fallback_model,
                          timeout_s=settings.llm_timeout_s)


@asynccontextmanager
async def open_checkpointer(settings: Settings) -> AsyncIterator[Any]:
    """Where conversation state lives. Durable state means any instance can serve any turn."""
    if settings.checkpointer == "memory":
        yield MemorySaver()
    elif settings.checkpointer == "sqlite":
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        Path(settings.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        async with AsyncSqliteSaver.from_conn_string(settings.sqlite_path) as saver:
            yield saver
    else:
        raise ValueError(f"Unknown CHECKPOINTER {settings.checkpointer!r}")


@asynccontextmanager
async def build_runtime(settings: Settings, llm: AgentLLM | None = None) -> AsyncIterator[Runtime]:
    llm = llm or make_llm(settings)
    tracer = Tracer(settings.langfuse_enabled, settings.environment)
    async with open_checkpointer(settings) as checkpointer:
        graph = build_graph(llm, LocalScorer(), checkpointer, settings.max_history_messages)
        try:
            yield Runtime(settings, ChatService(graph, settings, tracer), tracer, llm)
        finally:
            tracer.flush()
