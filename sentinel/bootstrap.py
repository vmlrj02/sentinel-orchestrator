"""Builds the runtime (LLM, stores, retriever, graph, service) from settings. Shared by the API, evals and tests."""
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
from .rag import HashingEmbedder, OpenAIEmbedder, Retriever, load_retriever
from .scoring.model import LocalScorer
from .service import ChatService
from .store import ApplicationStore, MemoryApplicationStore, SqliteApplicationStore


@dataclass
class Runtime:
    settings: Settings
    service: ChatService
    tracer: Tracer
    llm: AgentLLM
    retriever: Retriever


def make_llm(settings: Settings) -> AgentLLM:
    if settings.llm_provider == "fake":
        return FakeAgentLLM(latency_ms=settings.fake_llm_latency_ms)
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set (or use LLM_PROVIDER=fake)")
    return OpenAIAgentLLM(settings.interviewer_model, settings.advisor_model, settings.openai_api_key,
                          settings.fallback_model, timeout_s=settings.llm_timeout_s)


def make_retriever(settings: Settings) -> Retriever:
    embedder = HashingEmbedder() if settings.llm_provider == "fake" else OpenAIEmbedder(settings.openai_api_key)
    return load_retriever(embedder, mode=settings.retrieval_mode)  # type: ignore[arg-type]


@asynccontextmanager
async def open_stores(settings: Settings) -> AsyncIterator[tuple[Any, ApplicationStore]]:
    """Conversation checkpoints + application records. Durable state means any instance can serve any turn."""
    if settings.checkpointer == "memory":
        yield MemorySaver(), MemoryApplicationStore()
    elif settings.checkpointer == "sqlite":
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        Path(settings.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        store = await SqliteApplicationStore.open(str(Path(settings.sqlite_path).with_name("applications.db")))
        try:
            async with AsyncSqliteSaver.from_conn_string(settings.sqlite_path) as saver:
                yield saver, store
        finally:
            await store.close()
    else:
        raise ValueError(f"Unknown CHECKPOINTER {settings.checkpointer!r}")


@asynccontextmanager
async def build_runtime(settings: Settings, llm: AgentLLM | None = None,
                        retriever: Retriever | None = None) -> AsyncIterator[Runtime]:
    llm = llm or make_llm(settings)
    retriever = retriever or make_retriever(settings)
    tracer = Tracer(settings.langfuse_enabled, settings.environment, settings.public_traces)
    async with open_stores(settings) as (checkpointer, store):
        graph = build_graph(llm, LocalScorer(), retriever, store, checkpointer, settings.max_history_messages)
        try:
            yield Runtime(settings, ChatService(graph, settings, tracer, store), tracer, llm, retriever)
        finally:
            tracer.flush()
