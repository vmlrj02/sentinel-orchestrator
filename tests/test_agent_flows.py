"""End-to-end agent behaviour with the deterministic fake LLM: RAG, tools, human review, guards."""
from sentinel.agent.graph import MAX_TOOL_ROUNDS
from tests.conftest import APPLICATION_TURNS, REFERRED_TURNS


async def test_policy_question_mid_application_uses_rag_then_resumes(service):
    await service.run_turn("c", "u", APPLICATION_TURNS[0], "test")
    result = await service.run_turn("c", "u", "question: search: what documents do I need", "test")
    assert result.tools == ["search_policy"]
    assert result.citations and all(c.startswith("aster/") for c in result.citations)
    assert result.explained
    state = await service.get_state("c")
    assert state["status"] == "collecting" and state["application"]["person_age"] == 29


async def test_lender_is_fixed_by_the_service_not_the_model(service):
    await service.run_turn("k", "u", "question: search: fees", "test", tenant_id="kestrel")
    result = await service.run_turn("k", "u", "question: search: fees", "test", tenant_id="aster")
    assert all(c.startswith("kestrel/") for c in result.citations)  # first turn decides the lender


async def test_what_if_tool_after_decision(service):
    for text in [*APPLICATION_TURNS, "yes"]:
        await service.run_turn("w", "u", text, "test")
    events = [e async for e in service.stream_turn("w", "u", "whatif: loan_amnt=6000", "test")]
    what_if = next(e for e in events if e["type"] == "what_if")["result"]
    assert what_if["after"]["default_probability"] < what_if["before"]["default_probability"]
    assert (await service.get_state("w"))["final_decision"]["decision"] == "approved"  # unchanged


async def test_tool_errors_are_returned_to_the_model(service):
    for text in [*APPLICATION_TURNS, "yes"]:
        await service.run_turn("e", "u", text, "test")
    result = await service.run_turn("e", "u", "whatif: loan_amnt=-5", "test")
    assert result.error is None and result.explained  # model got the error text and still answered


async def test_referred_application_waits_for_underwriter(service):
    notified: list[tuple[str, list[str]]] = []

    async def notifier(conversation_id, messages):
        notified.append((conversation_id, messages))

    service.on_notify(notifier)
    for text in REFERRED_TURNS:
        await service.run_turn("r", "u", text, "test")
    result = await service.run_turn("r", "u", "yes", "test")
    assert result.assessment["policy_decision"] == "referred"
    assert "underwriter" in result.messages[-1]

    # While paused, new messages don't run the graph
    held = await service.run_turn("r", "u", "any news?", "test")
    assert "with an underwriter" in held.messages[0]

    [pending] = await service.list_reviews("aster")
    decided = await service.submit_review(pending.application_id, "declined", "maya", "income not verified")
    assert decided.explained
    assert notified and notified[0][0] == "r"

    state = await service.get_state("r")
    assert state["status"] == "decided"
    assert state["final_decision"] == {"decision": "declined", "decided_by": "underwriter:maya",
                                       "note": "income not verified", "explain": False}
    record = await service.store.get(pending.application_id)
    assert record.status == "declined" and record.review_note == "income not verified"
    assert await service.list_reviews() == []


async def test_same_applicant_different_lender_policy(service):
    """Same applicant, same model score: Aster refers to a human, Kestrel approves automatically."""
    turns = [t.replace("loan_amnt=18000 loan_int_rate=13 loan_intent=MEDICAL",
                       "loan_amnt=15000 loan_int_rate=13.5 loan_intent=PERSONAL") for t in REFERRED_TURNS]
    outcomes = {}
    for lender in ("aster", "kestrel"):
        for text in [*turns, "yes"]:
            result = await service.run_turn(lender, "u", text, "test", tenant_id=lender)
        outcomes[lender] = result.assessment["policy_decision"]
    assert outcomes == {"aster": "referred", "kestrel": "approved"}


async def test_tool_step_limit(service, fake_llm, monkeypatch):
    """A model that keeps calling tools is cut off after MAX_TOOL_ROUNDS and must answer."""
    original = fake_llm.advise

    async def always_search(messages, tools):
        if tools:
            from langchain_core.messages import AIMessageChunk
            yield AIMessageChunk(content="", tool_call_chunks=[
                {"name": "search_policy", "args": '{"query": "fees"}', "id": f"c{len(messages)}", "index": 0}])
        else:
            async for chunk in original(messages, tools):
                yield chunk

    monkeypatch.setattr(fake_llm, "advise", always_search)
    result = await service.run_turn("loop", "u", "question: anything", "test")
    assert result.tools.count("search_policy") == MAX_TOOL_ROUNDS
    assert result.explained and result.error is None
