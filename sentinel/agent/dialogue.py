"""Dialogue management for the application interview.

The code owns the conversation: which detail it is waiting for (the pending slot), the wording of every
question, how to recover when an answer doesn't fit, and when to offer a way out. The LLM is only used to
understand free text that the code can't parse itself (see `understand`).
"""
import re
from collections.abc import Mapping
from typing import Any

from langchain_core.messages import AIMessage

from .state import CHOICES, REQUIRED_FIELDS, missing_fields
from .validation import NUMERIC_FIELDS, parse_number

CONFIRM = "confirm"  # pending slot while the summary waits for yes/no

QUESTIONS = {
    "person_age": "How old are you?",
    "person_income": "What is your annual income?",
    "person_emp_length": "How many years have you been at your current job?",
    "person_home_ownership": "Do you rent or own your home?",
    "loan_amnt": "How much would you like to borrow?",
    "loan_int_rate": "What interest rate were you offered, in percent?",
    "loan_intent": "What is the loan for?",
    "loan_grade": "What loan grade were you given?",
    "cb_person_cred_hist_length": "How many years of credit history do you have?",
    "cb_person_default_on_file": "Have you ever defaulted on a loan before?",
}

EXAMPLES = {
    "person_age": "a number of years, like 32",
    "person_income": "a yearly amount, like 55000 or 55k",
    "person_emp_length": "a number of years, like 3 or 1.5",
    "loan_amnt": "an amount, like 12000 or 12k",
    "loan_int_rate": "a percentage, like 11.5",
    "cb_person_cred_hist_length": "a number of years, like 5",
}

ESCAPES = ["Skip for now", "Start over", "Talk to a person"]
CHANGE_OPTIONS = ["Change loan amount", "Change income", "Change interest rate", "Start over"]

# Common phrasings understood without an LLM call. Values are the model's categories.
SYNONYMS: dict[str, list[tuple[str, str]]] = {
    "person_home_ownership": [
        # Order matters: "my home loan" must hit MORTGAGE before OWN's "my home"
        (r"mortgage(d)?|(my )?home loan|housing loan", "MORTGAGE"),
        (r"rent(ing|ed|er)?|tenant|lease|paying rent", "RENT"),
        (r"own(ed|er|ing)?|(it'?s|is) mine|my (own )?(house|home|flat|apartment)|paid off", "OWN"),
        (r"other|(live|living|stay|staying) with (my )?(parents|family|friends)|hostel|dorm", "OTHER"),
    ],
    "loan_intent": [
        (r"personal|wedding|travel|holiday|vacation", "PERSONAL"),
        (r"education|study|studies|college|university|uni|school|tuition|fees|degree|master'?s|course|mba",
         "EDUCATION"),
        (r"medical|hospital|surgery|health|treatment|doctor|operation", "MEDICAL"),
        (r"venture|business|start-?up|shop|company", "VENTURE"),
        (r"home improvement|renovat\w*|repairs?|kitchen|bathroom|extension|remodel\w*", "HOMEIMPROVEMENT"),
        (r"debt( consolidation)?|consolidat\w*|credit cards?|pay off (my )?(debts|loans|cards)", "DEBTCONSOLIDATION"),
    ],
    "loan_grade": [(rf"(grade )?{g.lower()}", g) for g in "ABCDEFG"],
    "cb_person_default_on_file": [
        (r"yes|y|yeah|yep|once|twice|i have|i did|have defaulted", "Y"),
        (r"no|n|nope|never|not ever|i haven'?t|no defaults?", "N"),
    ],
}

_FILLER = re.compile(r"^\s*(i'?m|i am|it'?s|it is|i|just|um+|uh+|well|ok(ay)?|sure)\b[\s,]*", re.I)


def understand(slot: str | None, text: str) -> Any | None:
    """Fast path: parse a short answer for the pending slot in code. None means "ask the LLM"."""
    if slot is None or slot not in REQUIRED_FIELDS or len(text) > 60:
        return None
    if slot in NUMERIC_FIELDS:
        return parse_number(text)
    cleaned = text.strip().lower().rstrip(".!")
    while (stripped := _FILLER.sub("", cleaned)) != cleaned:
        cleaned = stripped
    for pattern, value in SYNONYMS.get(slot, []):
        if re.fullmatch(rf"(a |an |for |for a |for my )?({pattern})( loan)?", cleaned, re.I):
            return value
    return None


def next_slot(state: Mapping[str, Any]) -> str:
    """The next detail to ask for: missing fields in order, skipped ones last; CONFIRM when complete."""
    missing = missing_fields(state)
    if not missing:
        return CONFIRM
    deferred = state.get("deferred") or []
    return next((f for f in missing if f not in deferred), missing[0])


def question_message(state: Mapping[str, Any], prefix: str = "Thanks.", attempt: int = 0,
                     slot: str | None = None) -> AIMessage:
    """Ask for the pending slot. attempt > 0 is a repair: explain the expected answer, then offer a way out."""
    from .graph import summary_message  # summary formatting lives with the graph

    slot = slot or next_slot(state)
    if slot == CONFIRM:
        return AIMessage(summary_message(state), response_metadata={"options": ["Yes", "No"]})

    question = QUESTIONS[slot]
    deferred = state.get("deferred") or []
    if slot in deferred and all(f in deferred for f in missing_fields(state)):
        question = f"To finish your application I still need this: {question[0].lower()}{question[1:]}"
    if attempt:
        prefix = "Sorry, I didn't catch that."
    lines = [f"{prefix} {question}".strip()]
    options = list(CHOICES.get(slot, []))
    if options:
        listed = ", ".join(options[:-1]) + f" or {options[-1]}"
        lines.append(f"Please reply with one of: {listed}.")
    elif attempt:
        lines.append(f"Please answer with {EXAMPLES[slot]}.")
    if attempt >= 2:
        lines.append("Or, if you'd rather, skip this for now, start over, or ask to talk to a person.")
        options += ESCAPES
    return AIMessage("\n\n".join(lines), response_metadata={"options": options} if options else {})


def field_for(text: str) -> str | None:
    """Which field "change my income" / "update the loan amount" refers to."""
    keywords = [
        (r"age", "person_age"), (r"income|salary|earnings", "person_income"),
        (r"job|employment|years at work|work experience", "person_emp_length"),
        (r"home|housing|rent|own", "person_home_ownership"), (r"(loan )?amount|borrow", "loan_amnt"),
        (r"(interest )?rate|interest", "loan_int_rate"), (r"purpose|reason|what it'?s for", "loan_intent"),
        (r"grade", "loan_grade"), (r"credit history|history", "cb_person_cred_hist_length"),
        (r"default", "cb_person_default_on_file"),
    ]
    return next((field for pattern, field in keywords if re.search(rf"\b({pattern})\b", text, re.I)), None)

