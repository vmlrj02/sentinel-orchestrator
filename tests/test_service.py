import asyncio
import dataclasses
import time

from sentinel.agent.llm import FakeAgentLLM
from tests.conftest import APPLICATION_TURNS, make_service


async def test_rejects_empty_and_oversized_messages(service, settings):
    assert (await service.run_turn("c", "u", "   ", "test")).error == "empty_message"
    too_long = "x" * (settings.max_message_chars + 1)
    assert (await service.run_turn("c", "u", too_long, "test")).error == "message_too_long"


async def test_rate_limit_is_per_user(settings, fake_llm):
    service = make_service(dataclasses.replace(settings, rate_limit_per_minute=2), fake_llm)
    for _ in range(2):
        assert (await service.run_turn("c", "alice", "hi", "test")).error is None
    assert (await service.run_turn("c", "alice", "hi", "test")).error == "rate_limited"
    assert (await service.run_turn("c", "bob", "hi", "test")).error is None


async def test_many_users_run_concurrently(settings):
    """100 users x 200ms LLM latency: async I/O means ~0.2s total, not 20s."""
    service = make_service(settings, FakeAgentLLM(latency_ms=200))
    started = time.perf_counter()
    results = await asyncio.gather(*(service.run_turn(f"c{i}", f"u{i}", "person_age=30", "test") for i in range(100)))
    assert all(r.error is None for r in results)
    assert time.perf_counter() - started < 2


async def test_same_conversation_turns_are_serialised(settings):
    """Two messages sent at once in one chat must not overwrite each other's fields."""
    service = make_service(settings, FakeAgentLLM(latency_ms=50))
    await asyncio.gather(*(service.run_turn("same", "u", text, "test") for text in APPLICATION_TURNS))
    state = await service.get_state("same")
    assert state["missing"] == []
    assert len(state["messages"]) == 2 * len(APPLICATION_TURNS)


async def test_concurrency_cap_queues_extra_turns(settings):
    service = make_service(dataclasses.replace(settings, max_concurrent_turns=2), FakeAgentLLM(latency_ms=100))
    started = time.perf_counter()
    await asyncio.gather(*(service.run_turn(f"c{i}", f"u{i}", "hi", "test") for i in range(6)))
    assert time.perf_counter() - started >= 0.3  # 6 turns through 2 slots = 3 waves of 100ms


async def test_turn_timeout_returns_error(settings):
    service = make_service(dataclasses.replace(settings, turn_timeout_s=0.05), FakeAgentLLM(latency_ms=500))
    result = await service.run_turn("c", "u", "hi", "test")
    assert result.error == "timeout"


async def test_reset_starts_a_fresh_application(service):
    for text in APPLICATION_TURNS:
        await service.run_turn("c", "u", text, "test")
    await service.reset("c")
    state = await service.get_state("c")
    assert state["application"] == {} and state["messages"] == []


async def test_stream_emits_tokens_then_done(service):
    for text in [*APPLICATION_TURNS]:
        await service.run_turn("c", "u", text, "test")
    events = [e async for e in service.stream_turn("c", "u", "yes", "test")]
    types = [e["type"] for e in events]
    assert types[0] == "status"
    assert "assessment" in types and "token" in types
    assert types[-1] == "done" and events[-1]["trace_id"]
