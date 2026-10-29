from src.graph.state import REQUIRED_FIELDS
from src.graph.workflow import route_entry, route_after_sentinel

ALL_FIELDS = {f: "x" for f in REQUIRED_FIELDS}


def test_new_chat_goes_to_interview():
    assert route_entry({"messages": []}) == "sentinel"


def test_all_fields_without_confirmation_waits_for_user():
    assert route_after_sentinel(ALL_FIELDS) == "__end__"


def test_confirmed_application_goes_to_scoring():
    assert route_after_sentinel({**ALL_FIELDS, "details_confirmed": True}) == "scoring"
    assert route_entry({**ALL_FIELDS, "details_confirmed": True}) == "scoring"


def test_scored_application_goes_to_analyst():
    assert route_entry({**ALL_FIELDS, "details_confirmed": True, "is_processed": True}) == "analyst"


def test_summary_lists_every_field_readably():
    from src.graph.nodes import summary_message
    state = {**ALL_FIELDS, "person_income": 32000.0, "loan_int_rate": 18.5, "cb_person_default_on_file": "Y"}
    text = summary_message(state)
    assert "Annual income: 32,000" in text
    assert "Interest rate (%): 18.5" in text
    assert "Previous default: Yes" in text
    assert text.count("\n- ") == len(REQUIRED_FIELDS)
