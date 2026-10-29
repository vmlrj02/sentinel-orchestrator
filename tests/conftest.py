import dataclasses

import pytest
from langgraph.checkpoint.memory import MemorySaver

from sentinel.agent.graph import build_graph
from sentinel.agent.llm import FakeAgentLLM
from sentinel.config import Settings, load_settings
from sentinel.observability import Tracer
from sentinel.scoring.model import LocalScorer
from sentinel.service import ChatService

# One full application in the fake LLM's field=value format
APPLICATION_TURNS = [
    "person_name=Priya person_age=29",
    "person_income=55000 person_emp_length=4 person_home_ownership=RENT",
    "loan_amnt=12000 loan_int_rate=11.2 loan_intent=EDUCATION",
    "loan_grade=B cb_person_cred_hist_length=5 cb_person_default_on_file=N",
]


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
    graph = build_graph(llm, LocalScorer(), MemorySaver())
    return ChatService(graph, settings, Tracer(enabled=False, environment="test"))


@pytest.fixture
def service(settings: Settings, fake_llm: FakeAgentLLM) -> ChatService:
    return make_service(settings, fake_llm)
