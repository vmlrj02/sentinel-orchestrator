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
    """100 users x 200ms LLM latency: async I/O means well under a second, not 20s sequentially.
    The bound is loose on purpose so a busy CI machine doesn't make it flaky."""
    service = make_service(settings, FakeAgentLLM(latency_ms=200))
    started = time.perf_counter()
    results = await asyncio.gather(*(service.run_turn(f"c{i}", f"u{i}", "person_age=30", "test") for i in range(100)))
    assert all(r.error is None for r in results)
    assert time.perf_counter() - started < 5


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


async def test_provider_outage_is_reported_clearly(service, fake_llm, monkeypatch):
    import httpx
    from openai import RateLimitError

    async def out_of_quota(messages):
        response = httpx.Response(429, request=httpx.Request("POST", "https://api.openai.com"))
        raise RateLimitError("You have no credits remaining", response=response, body=None)

    monkeypatch.setattr(fake_llm, "interview", out_of_quota)
    result = await service.run_turn("c", "u", "hello", "test")
    assert result.error == "llm_unavailable" and "temporarily unavailable" in result.messages[0]


async def test_out_of_scope_requests_get_a_clear_answer(service):
    result = await service.run_turn("oos", "u", "outofscope: I need to deposit cash", "test")
    assert "contact Aster Bank directly" in result.messages[-1]


async def test_stalled_conversation_stops_calling_the_llm(service, fake_llm):
    for text in ["I won't say", "no need", "outofscope: deposit cash"]:  # three turns, no progress
        await service.run_turn("stall", "u", text, "test")
    calls = fake_llm.calls
    for text in ["blah", "still no"]:
        events = [e async for e in service.stream_turn("stall", "u", text, "test")]
        message = next(e for e in events if e["type"] == "message")
        assert "isn't what you need" in message["text"] and message["options"]
    assert fake_llm.calls == calls  # zero tokens spent on the stalled turns

    resumed = await service.run_turn("stall", "u", "Apply for a loan", "test")  # a button resumes the chat
    assert fake_llm.calls == calls + 1 and "isn't what you need" not in resumed.messages[-1]


async def test_progress_resets_the_stall_counter(service, fake_llm):
    for text in ["hmm", "ok", "person_age=30", "hmm", "ok"]:
        await service.run_turn("progress", "u", text, "test")
    calls = fake_llm.calls
    await service.run_turn("progress", "u", "hmm", "test")
    assert fake_llm.calls == calls + 1  # only 2 non-progress turns since the last detail


async def test_conversation_turn_cap(settings, fake_llm):
    service = make_service(dataclasses.replace(settings, max_turns_per_conversation=2), fake_llm)
    for text in ["person_age=30", "person_income=50000"]:
        await service.run_turn("cap", "u", text, "test")
    calls = fake_llm.calls
    result = await service.run_turn("cap", "u", "person_emp_length=3", "test")
    assert "length limit" in result.messages[0] and fake_llm.calls == calls


async def test_name_is_not_collected(service):
    result = await service.run_turn("n", "u", "hello", "test")
    state = await service.get_state("n")
    assert "person_name" not in state["missing"] and "name" not in result.messages[-1].lower()
