"""Dialogue management: fast path, repair and escalation, commands, sensitive data, funnel metrics."""
import pytest

from sentinel import metrics
from sentinel.agent.dialogue import understand

START = "person_age=30 person_income=50000 person_emp_length=3"  # fake LLM fills these; next: home


@pytest.mark.parametrize(("slot", "text", "expected"), [
    ("person_home_ownership", "I rent", "RENT"), ("person_home_ownership", "renting", "RENT"),
    ("person_home_ownership", "my own house", "OWN"), ("person_home_ownership", "my home loan", "MORTGAGE"),
    ("person_home_ownership", "living with my parents", "OTHER"), ("person_home_ownership", "Mortgage", "MORTGAGE"),
    ("loan_intent", "for my master's", "EDUCATION"), ("loan_intent", "Home improvement", "HOMEIMPROVEMENT"),
    ("loan_intent", "hospital bills", None), ("loan_intent", "surgery", "MEDICAL"),
    ("loan_intent", "to consolidate my credit cards", None), ("loan_intent", "credit cards", "DEBTCONSOLIDATION"),
    ("loan_grade", "B", "B"), ("loan_grade", "grade c", "C"), ("loan_grade", "H", None),
    ("cb_person_default_on_file", "never", "N"), ("cb_person_default_on_file", "Yes", "Y"),
    ("cb_person_default_on_file", "nope", "N"), ("loan_amnt", "12k", 12_000), ("loan_int_rate", "11.5%", 11.5),
    ("person_home_ownership", "I rent now but I'm buying a flat next year", None),  # too much for the fast path
])
def test_fast_path_understanding(slot, text, expected):
    assert understand(slot, text) == expected


async def test_short_answers_cost_no_llm_calls(service, fake_llm):
    await service.run_turn("fast", "u", START, "t")
    calls = fake_llm.calls
    for answer in ["I rent", "12k", "11.5%", "for my degree", "B", "5 years", "never"]:
        result = await service.run_turn("fast", "u", answer, "t")
    assert result.messages[-1].startswith("That's everything. Please check your details:")
    assert fake_llm.calls == calls  # seven answers, zero LLM calls

    result = await service.run_turn("fast", "u", "Yes", "t")
    assert result.assessment["policy_decision"] == "approved"
    assert fake_llm.calls == calls + 1  # only the advisor's explanation


async def test_repair_then_escalation(service):
    first = await service.run_turn("rep", "u", "abc", "t")  # asked: age
    assert first.messages[-1] == ("Hi! I'm Sentinel, Aster Bank's loan assistant. Sorry, I didn't quite get that."
                                  "\n\n(1 of 10) How old are you?\n\nFor example: 32.")
    second = await service.run_turn("rep", "u", "abc", "t")
    assert second.messages[-1].startswith("Let me put it another way: I need your age in years")
    events = [e async for e in service.stream_turn("rep", "u", "still abc", "t")]
    message = next(e for e in events if e["type"] == "message")
    assert message["text"].startswith("We seem to be stuck on this one.")
    assert message["options"] == ["Skip for now", "Start over", "Talk to a person"]
    assert len({first.messages[-1], second.messages[-1], message["text"]}) == 3  # never the same line twice


async def test_skip_defers_a_question_until_the_end(service):
    await service.run_turn("skip", "u", START, "t")  # asked: home
    skipped = await service.run_turn("skip", "u", "Skip for now", "t")
    assert skipped.messages[-1] == ("No problem, we'll come back to that.\n\n"
                                    "(4 of 10) How much would you like to borrow?")
    for answer in ["12k", "11.5", "education", "B", "5", "no"]:
        result = await service.run_turn("skip", "u", answer, "t")
    assert "To finish your application I still need this: do you rent or own your home?" in result.messages[-1]
    result = await service.run_turn("skip", "u", "rent", "t")
    assert result.messages[-1].startswith("That's everything.")


async def test_commands_are_answered_without_the_llm(service, fake_llm):
    await service.run_turn("cmd", "u", START, "t")
    calls = fake_llm.calls
    summary = await service.run_turn("cmd", "u", "what do you have so far?", "t")
    assert "Annual income: 50000" in summary.messages[-1] and "Still needed:" in summary.messages[-1]
    change = await service.run_turn("cmd", "u", "change my income", "t")
    assert change.messages[-1].startswith("Sure.\n\n(3 of 10) What is your annual income?")
    await service.run_turn("cmd", "u", "60k", "t")
    state = await service.get_state("cmd")
    assert state["application"]["person_income"] == 60_000
    human = await service.run_turn("cmd", "u", "talk to a person", "t")
    assert "contact Aster Bank's loan team" in human.messages[-1]
    assert "start over" in (await service.run_turn("cmd", "u", "help", "t")).messages[-1]
    confirm = await service.run_turn("cmd", "u", "Start over", "t")  # answers would be lost: ask first
    assert confirm.messages[-1].startswith("Start a new application? This clears the 3 answers")
    assert confirm.options == ["Yes, start over", "No, continue"]
    restarted = await service.run_turn("cmd", "u", "Yes, start over", "t")
    assert restarted.messages[-1].startswith("Okay, let's start again. I'll ask you 10 quick questions")
    assert (await service.get_state("cmd"))["application"] == {}
    assert fake_llm.calls == calls  # every command above was free


async def test_saying_no_to_the_summary_offers_changes(service):
    await service.run_turn("no", "u", START, "t")
    for answer in ["rent", "12k", "11.5", "education", "B", "5", "no"]:
        await service.run_turn("no", "u", answer, "t")
    events = [e async for e in service.stream_turn("no", "u", "No", "t")]
    message = next(e for e in events if e["type"] == "message")
    assert message["text"].startswith("No problem. What would you like to change?")
    assert "Change loan amount" in message["options"]


async def test_card_numbers_are_removed_before_anything_sees_them(service, fake_llm):
    calls = fake_llm.calls
    result = await service.run_turn("pii", "u", "4111 1111 1111 1111", "t")
    assert "please don't share card, bank account or ID numbers" in result.messages[0]
    assert fake_llm.calls == calls  # nothing left to send to the LLM

    await service.run_turn("pii", "u", "I'm 30, card 4111-1111-1111-1111", "t")
    stored = " ".join(m["text"] for m in (await service.get_state("pii"))["messages"])
    assert "4111" not in stored and "[removed]" in stored


async def test_funnel_and_slot_metrics(service):
    before = metrics.snapshot()
    await service.run_turn("fun", "u", START, "t")
    for answer in ["rent", "12k", "11.5", "education", "B", "5", "no", "yes"]:
        await service.run_turn("fun", "u", answer, "t")
    after = metrics.snapshot()

    def delta(section: str, key: str, sub: str | None = None) -> float:
        get = (lambda s: s.get(section, {}).get(key, {}).get(sub, 0)) if sub else \
              (lambda s: s.get(section, {}).get(key, 0))
        return get(after) - get(before)

    assert delta("funnel", "started") == 1 and delta("funnel", "summary_shown") == 1
    assert delta("funnel", "confirmed") == 1
    assert delta("slots", "person_home_ownership", "filled_fast") == 1
    assert delta("slots", "person_age", "filled_llm") == 1


@pytest.mark.parametrize("text", ["what?", "what do you even mean? rent what?", "I don't understand", "huh",
                                  "what do you mean"])
async def test_clarification_explains_the_current_question(service, fake_llm, text):
    await service.run_turn(f"cl{len(text)}", "u", START, "t")  # asked: home
    calls = fake_llm.calls
    result = await service.run_turn(f"cl{len(text)}", "u", text, "t")
    assert result.messages[-1].startswith("This is about where you live")
    assert "Please reply with one of: Rent, Own, Mortgage or Other." in result.messages[-1]
    assert fake_llm.calls == calls


@pytest.mark.parametrize("text", ["what you can give me?", "the max you can give", "i said max that i am eligible for",
                                  "as much as possible", "whatever you can give"])
async def test_asking_for_the_maximum_does_not_promise_an_amount(service, fake_llm, text):
    cid = f"max{abs(hash(text))}"
    await service.run_turn(cid, "u", "person_age=36 person_income=60000 person_emp_length=4", "t")
    await service.run_turn(cid, "u", "other", "t")
    calls = fake_llm.calls
    result = await service.run_turn(cid, "u", text, "t")
    assert result.messages[-1].startswith("I can't say what you'd qualify for until your application has been")
    import re
    assert not re.search(r"\d{3,}|lakh", result.messages[-1])  # no amount quoted
    assert "loan_amnt" not in (await service.get_state(cid))["application"]
    assert fake_llm.calls == calls


async def test_frustration_gets_an_apology_and_a_recap(service, fake_llm):
    await service.run_turn("fr", "u", "person_age=36 person_income=60000 person_emp_length=4", "t")
    calls = fake_llm.calls
    result = await service.run_turn("fr", "u", "i already told you stupid", "t")
    assert result.messages[-1].startswith("Sorry, I've made this harder than it should be.")
    assert "60,000 a year" in result.messages[-1] and "Do you rent or own your home?" in result.messages[-1]
    assert "Talk to a person" in result.options and fake_llm.calls == calls


def test_homeless_is_other():
    for text in ["i live on the streets", "homeless", "no fixed address"]:
        assert understand("person_home_ownership", text) == "OTHER"
