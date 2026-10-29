import dataclasses

import pytest
from langgraph.checkpoint.memory import MemorySaver

from sentinel.agent.graph import build_graph
from sentinel.agent.llm import FakeAgentLLM
from sentinel.config import Settings, load_settings
from sentinel.observability import Tracer
from sentinel.rag import HashingEmbedder, Retriever, load_retriever
from sentinel.scoring.model import LocalScorer
from sentinel.service import ChatService
from sentinel.store import MemoryApplicationStore

# One full application in the fake LLM's field=value format
APPLICATION_TURNS = [
    "person_name=Priya person_age=29",
    "person_income=55000 person_emp_length=4 person_home_ownership=RENT",
    "loan_amnt=12000 loan_int_rate=11.2 loan_intent=EDUCATION",
    "loan_grade=B cb_person_cred_hist_length=5 cb_person_default_on_file=N",
]

# Default risk ~0.53: Aster's review band is 30-60%, so this goes to a human underwriter
REFERRED_TURNS = [
    "person_name=Asha person_age=29",
    "person_income=55000 person_emp_length=4 person_home_ownership=RENT",
    "loan_amnt=18000 loan_int_rate=13 loan_intent=MEDICAL",
    "loan_grade=C cb_person_cred_hist_length=5 cb_person_default_on_file=N",
]

_retriever: Retriever | None = None


def retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = load_retriever(HashingEmbedder())
    return _retriever


@pytest.fixture
def settings() -> Settings:
    return dataclasses.replace(
        load_settings(), llm_provider="fake", checkpointer="memory", telegram_mode="off",
        langfuse_enabled=False, fake_llm_latency_ms=0, flush_traces_each_turn=False,
        rate_limit_per_minute=1000, max_concurrent_turns=100,
    )


@pytest.fixture
def fake_llm() -> FakeAgentLLM:
    return FakeAgentLLM()


def make_service(settings: Settings, llm: FakeAgentLLM) -> ChatService:
    store = MemoryApplicationStore()
    graph = build_graph(llm, LocalScorer(), retriever(), store, MemorySaver())
    return ChatService(graph, settings, Tracer(enabled=False, environment="test"), store)


@pytest.fixture
def service(settings: Settings, fake_llm: FakeAgentLLM) -> ChatService:
    return make_service(settings, fake_llm)
