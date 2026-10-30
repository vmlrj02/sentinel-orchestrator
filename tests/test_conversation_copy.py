"""The approved conversation wording: opening, name, progress, read-back, process questions, guards."""
import pytest

from sentinel.agent.dialogue import fmt_amount, indian_grouping, playback, safe_lead
from tests.conftest import APPLICATION_TURNS


async def test_greeting_gets_the_opening_and_costs_nothing(service, fake_llm):
    result = await service.run_turn("open", "u", "hey", "t")
    text = result.messages[-1]
    assert text.startswith("Hi! I'm Sentinel, Aster Bank's loan assistant. I'll ask you 10 quick questions")
    assert "You won't need your ID, card or bank details" in text and text.endswith("(optional)")
    assert result.options == ["Skip"] and fake_llm.calls == 0


async def test_name_then_first_question(service, fake_llm):
    await service.run_turn("nm", "u", "hi", "t")
    result = await service.run_turn("nm", "u", "vimal", "t")
    assert result.messages[-1] == "Nice to meet you, Vimal. Let's start.\n\n(1 of 10) How old are you?"
    assert fake_llm.calls == 0


async def test_skipping_the_name(service):
    await service.run_turn("sk", "u", "hello", "t")
    result = await service.run_turn("sk", "u", "Skip", "t")
    assert result.messages[-1] == "No problem. Let's start.\n\n(1 of 10) How old are you?"


async def test_process_question_is_answered_without_counting_as_a_miss(service):
    await service.run_turn("pq", "u", "hi", "t")
    await service.run_turn("pq", "u", "Skip", "t")
    for _ in range(3):  # three process questions in a row: no repair, no escalation
        result = await service.run_turn("pq", "u", "process: you don't want my name?", "t")
    assert result.messages[-1] == "Fake answer about the process.\n\n(1 of 10) How old are you?"
    assert "stuck" not in result.messages[-1]


async def test_answers_are_read_back_with_progress(service):
    await service.run_turn("rb", "u", "hi", "t")
    await service.run_turn("rb", "u", "Skip", "t")
    result = await service.run_turn("rb", "u", "36", "t")
    assert result.messages[-1] == ("Got it: 36.\n\n(2 of 10) What is your annual income? "
                                   "This helps check the loan is affordable for you.")
    result = await service.run_turn("rb", "u", "9 lakhs", "t")
    assert result.messages[-1].startswith("Got it: 9 lakh a year.\n\n(3 of 10)")


async def test_summary_uses_the_applicants_number_style(service):
    for text in ["person_age=36 person_emp_length=4", "9 lakhs", "rent", "35 thousand", "11", "education", "B", "5",
                 "no"]:
        result = await service.run_turn("style", "u", text, "t")
    summary = result.messages[-1]
    assert "Annual income: 9,00,000" in summary and "Loan amount: 35,000" in summary
    assert result.options == ["Yes", "No, change something"]


async def test_decision_carries_its_reference(service):
    for text in [*APPLICATION_TURNS, "yes"]:
        result = await service.run_turn("ref", "u", text, "t")
    state = await service.get_state("ref")
    assert result.messages[-1].startswith(f"Reference: {state['application_id']}")


async def test_start_over_with_nothing_entered_restarts_straight_away(service):
    result = await service.run_turn("so", "u", "start over", "t")
    assert result.messages[-1].startswith("Okay, let's start again.")


async def test_no_continue_keeps_the_answers(service):
    await service.run_turn("cont", "u", "person_age=30 person_income=50000", "t")
    await service.run_turn("cont", "u", "start over", "t")
    result = await service.run_turn("cont", "u", "No, continue", "t")
    assert result.messages[-1].startswith("Okay, let's carry on.")
    assert (await service.get_state("cont"))["application"]["person_age"] == 30


@pytest.mark.parametrize(("lead", "kept"), [
    ("Thanks, noted.", True),
    ("You don't need your full name here; the decision only uses the 10 details.", True),
    ("How old are you?", False),                                  # code asks the questions
    ("Good news, you're likely to be approved!", False),          # no promises
    ("You could borrow up to 40000 with that income.", False),    # no amounts
    ("With 9 lakh income you'd qualify.", False),
    ("x" * 300, False),                                           # too long
])
def test_lead_guard(lead, kept):
    assert (safe_lead(lead) is not None) == kept


def test_number_formatting():
    assert indian_grouping(900000) == "9,00,000" and indian_grouping(12500000) == "1,25,00,000"
    assert indian_grouping(999) == "999"
    assert fmt_amount(900000, "indian", spoken=True) == "9 lakh"
    assert fmt_amount(950000, "indian", spoken=True) == "9,50,000"
    assert fmt_amount(900000, None) == "900,000"
    assert playback("person_home_ownership", "RENT", None) == "renting"
    assert playback("cb_person_default_on_file", "N", None) == "no previous defaults"
