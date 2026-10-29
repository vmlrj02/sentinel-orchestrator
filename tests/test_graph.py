from sentinel.agent.graph import route_after_interviewer, route_after_scoring, route_entry, summary_message
from sentinel.agent.state import REQUIRED_FIELDS
from tests.conftest import APPLICATION_TURNS

ALL_FIELDS = {f: "x" for f in REQUIRED_FIELDS}


def test_routing():
    assert route_entry({"messages": []}) == "interviewer"
    assert route_after_interviewer(ALL_FIELDS) == "__end__"  # complete but not confirmed
    assert route_after_interviewer({**ALL_FIELDS, "details_confirmed": True}) == "scoring"
    assert route_entry({**ALL_FIELDS, "details_confirmed": True}) == "scoring"  # retry after scoring failure
    assert route_entry({**ALL_FIELDS, "final_decision": {"decision": "approved"}}) == "advisor"
    assert route_after_interviewer({"applicant_question": "fees?"}) == "advisor"
    assert route_after_scoring({"assessment": {"policy_decision": "referred"}}) == "human_review"
    assert route_after_scoring({"assessment": {"policy_decision": "approved"}}) == "advisor"
    assert route_after_scoring({}) == "__end__"  # scoring failed: tell the user, retry next turn


def test_summary_lists_every_field_readably():
    state = {**ALL_FIELDS, "person_income": 32000.0, "loan_int_rate": 18.5,
             "cb_person_default_on_file": "Y", "loan_intent": "HOMEIMPROVEMENT"}
    text = summary_message(state)
    assert "Annual income: 32,000" in text
    assert "Interest rate (%): 18.5" in text
    assert "Previous default: Yes" in text
    assert "Purpose: Home improvement" in text
    assert "Name: Rahul Mehta" in summary_message({**state, "person_name": "Rahul Mehta"})
    assert text.count("\n- ") == len(REQUIRED_FIELDS)


async def test_full_application_shows_summary_then_scores_on_yes(service):
    for text in APPLICATION_TURNS:
        result = await service.run_turn("c1", "u1", text, "test")
    assert result.messages[-1].startswith("Here's your application:")
    assert result.assessment is None  # nothing is scored before the applicant confirms

    result = await service.run_turn("c1", "u1", "yes", "test")
    assert result.assessment["policy_decision"] == "approved"  # ~15% default risk, under Aster's 30%
    assert result.explained

    state = await service.get_state("c1")
    assert state["missing"] == []
    assert state["application"]["loan_percent_income"] == 0.22


async def test_correction_at_summary_requires_reconfirmation(service):
    for text in APPLICATION_TURNS:
        await service.run_turn("c2", "u1", text, "test")
    # Correction and "yes" in one message: the change wins and the summary is shown again
    result = await service.run_turn("c2", "u1", "yes but person_income=32000", "test")
    assert result.assessment is None
    assert "Annual income: 32,000" in result.messages[-1]

    result = await service.run_turn("c2", "u1", "yes", "test")
    assert result.assessment is not None


async def test_follow_up_questions_go_to_advisor(service, fake_llm):
    for text in [*APPLICATION_TURNS, "yes"]:
        await service.run_turn("c3", "u1", text, "test")
    calls = fake_llm.calls
    result = await service.run_turn("c3", "u1", "why?", "test")
    assert result.explained and result.assessment is None
    assert fake_llm.calls == calls + 1  # one advisor call, no re-interview, no re-scoring
