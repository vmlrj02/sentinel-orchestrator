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


def test_plain_yes_patterns():
    from sentinel.agent.graph import _PLAIN_YES
    for text in ["yes", "Yes, that's right", "yep all correct", "correct", "that's correct", "yes correct now",
                 "Looks good!", "yes please"]:
        assert _PLAIN_YES.match(text), text
    for text in ["yes but change the amount to 10000", "no", "not quite, income is 34000", "yes?? why grade E"]:
        assert not _PLAIN_YES.match(text), text



async def test_choice_questions_list_their_options(service):
    from sentinel.agent.state import CHOICES
    await service.run_turn("o", "u", "person_name=Ann person_age=30 person_income=50000 person_emp_length=3", "t")
    events = [e async for e in service.stream_turn("o", "u", "hello", "t")]
    message = next(e for e in events if e["type"] == "message")
    assert message["options"] == CHOICES["person_home_ownership"]
    assert message["text"] == ("Thanks. Do you rent or own your home?\n\n"
                               "Please reply with one of: Rent, Own, Mortgage or Other.")
    state = await service.get_state("o")
    assert state["messages"][-1]["options"] == CHOICES["person_home_ownership"]  # survives a page reload


async def test_free_text_questions_have_no_options(service):
    events = [e async for e in service.stream_turn("f", "u", "hello", "t")]
    message = next(e for e in events if e["type"] == "message")
    assert message["options"] == [] and "reply with one of" not in message["text"]


def test_parse_number():
    from sentinel.agent.validation import parse_number
    cases = {"1": 1, "12k": 12_000, "12,000": 12_000, "11.5%": 11.5, "8 lakhs": 800_000, "$5000": 5_000,
             "4 years": 4, "-10000": -10_000, "₹ 2.5 lakh": 250_000}
    for text, expected in cases.items():
        assert parse_number(text) == expected, text
    for text in ["idk", "how much ever you can give", "12 or 13", "age 30 and income 5000"]:
        assert parse_number(text) is None, text


async def test_bare_number_fills_the_question_that_was_asked(service):
    await service.run_turn("bn", "u", "person_age=30 person_income=50000 person_emp_length=3", "t")
    await service.run_turn("bn", "u", "person_home_ownership=RENT", "t")
    await service.run_turn("bn", "u", "5000", "t")  # asked: loan amount
    await service.run_turn("bn", "u", "1", "t")  # asked: interest rate
    application = (await service.get_state("bn"))["application"]
    assert application["loan_amnt"] == 5000 and application["loan_int_rate"] == 1


async def test_out_of_policy_loan_amount_is_rejected_with_the_range(service):
    await service.run_turn("lim", "u", "person_age=30 person_income=50000 person_emp_length=3", "t")
    await service.run_turn("lim", "u", "person_home_ownership=RENT", "t")
    result = await service.run_turn("lim", "u", "1", "t")
    assert result.messages[-1] == "Aster Bank lends between 1,000 and 40,000. How much would you like to borrow?"
    assert "loan_amnt" not in (await service.get_state("lim"))["application"]
