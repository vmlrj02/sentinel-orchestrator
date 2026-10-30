"""Dialogue management for the application interview.

Code decides what happens: which detail is pending, the question wording, progress, options, how to repair
an answer that doesn't fit, and when to offer a way out. The LLM decides how to respond to what the person
actually said: its short `lead` line (acknowledging, answering a question about the process, apologising)
opens the reply, after a guard checks it. Short answers skip the LLM entirely (`understand`).
"""
import re
from collections.abc import Mapping
from typing import Any

from langchain_core.messages import AIMessage

from .state import CHOICES, REQUIRED_FIELDS, missing_fields
from .validation import NUMERIC_FIELDS, parse_number

CONFIRM = "confirm"  # pending slot while the summary waits for yes/no
NAME = "name"        # optional first step: what to call the applicant
TOTAL = len(REQUIRED_FIELDS)

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

# A short "why" on the questions people hesitate over
WHY = {
    "person_income": "This helps check the loan is affordable for you.",
    "loan_int_rate": "It's the rate on your loan offer. Not sure? Around 10 to 12% is typical.",
    "loan_grade": "It's on your loan offer or credit report, and A is the strongest. Not sure? Most people are B or C.",
    "cb_person_default_on_file": "Lenders have to check repayment history. Late payments you caught up on don't count.",
}

EXAMPLES = {
    "person_age": "32",
    "person_income": "55000, 55k or 9 lakhs",
    "person_emp_length": "3, or 1.5 for a year and a half",
    "loan_amnt": "12000 or 12k",
    "loan_int_rate": "11.5",
    "cb_person_cred_hist_length": "5",
}

# Plain-language explanations, used for "what do you mean?" and on the second miss
EXPLANATIONS = {
    "person_age": "I need your age in years, for example 32. You must be at least 18 to apply.",
    "person_income": "Your total income for a year before tax, from salary, business, pension or other sources. "
                     "For example 55000, 55k or 9 lakhs.",
    "person_emp_length": "How long you've worked for your current employer, in years. For example 3, or 1.5 for "
                         "a year and a half. If you're self-employed, how long you've run your business.",
    "person_home_ownership": "This is about where you live: do you pay rent, own your home outright, own it with a "
                             "mortgage (a home loan you're still paying), or something else, like living with family?",
    "loan_amnt": "The amount you'd like to borrow, for example 12000 or 12k.",
    "loan_int_rate": "The yearly interest rate on the loan, as a percentage, for example 11.5. If you haven't been "
                     "offered one yet, a typical rate for a good credit profile is around 10 to 12.",
    "loan_intent": "What you'll use the money for, so we can apply the right policy: everyday personal costs, "
                   "education, medical bills, a business, home improvements, or paying off other debts.",
    "loan_grade": "Your loan grade (A to G) is a rating from your credit report or loan offer; A is the strongest. "
                  "If you don't know it, B or C is common.",
    "cb_person_cred_hist_length": "How many years since you first had a credit account, like a credit card or "
                                  "loan. For example 5.",
    "cb_person_default_on_file": "Whether you've ever failed to repay a loan or credit card as agreed (a default). "
                                 "Late payments that you later caught up on don't count.",
}

ESCAPES = ["Skip for now", "Start over", "Talk to a person"]
CHANGE_OPTIONS = ["Change loan amount", "Change income", "Change interest rate", "Start over"]
SUMMARY_OPTIONS = ["Yes", "No, change something"]
RESET_OPTIONS = ["Yes, start over", "No, continue"]

# --- Reading answers in code (no LLM) ----------------------------------------------------------------------

SYNONYMS: dict[str, list[tuple[str, str]]] = {
    "person_home_ownership": [
        # Order matters: "my home loan" must hit MORTGAGE before OWN's "my home"
        (r"mortgage(d)?|(my )?home loan|housing loan", "MORTGAGE"),
        (r"rent(ing|ed|er)?|tenant|lease|paying rent", "RENT"),
        (r"own(ed|er|ing)?|(it'?s|is) mine|my (own )?(house|home|flat|apartment)|paid off", "OWN"),
        (r"other|(live|living|stay|staying) with (my )?(parents|family|friends)|hostel|dorm|homeless|"
         r"no (fixed )?(home|address)|(live|living|sleep|sleeping) (on|in) the streets?|shelter", "OTHER"),
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
_GREETING = re.compile(r"^\s*(hi+|hello+|hey+|hiya|yo|namaste|hola|good (morning|afternoon|evening)|"
                       r"start|/start|i'?d like a (personal )?loan|i want a loan|apply)\b[\s!.,]*$", re.I)
_NAME = re.compile(r"^\s*(?:(?:i'?m|i am|my name is|call me|it'?s|this is)\s+)?"
                   r"(?P<name>[a-z][a-z.'-]*(?: [a-z][a-z.'-]*){0,3})\s*[.!]?\s*$", re.I)

MAX_AMOUNT = "MAX"  # "the most you can give me": answered without quoting any amount
_MAX_INTENT = re.compile(r"(the )?(max(imum)?|most|as much as (possible|you can)|whatever|what(ever)? you can "
                         r"(give|offer|lend)|how much (can|could) (i|you) (get|give|borrow|lend)|"
                         r"what (am i|i'?m) eligible for|(max|amount) (that )?i'?m eligible for|"
                         r"what you can give( me)?)\b", re.I)


def is_greeting(text: str) -> bool:
    return bool(_GREETING.match(text))


def understand(slot: str | None, text: str) -> Any | None:
    """Fast path: parse a short answer for the pending slot in code. None means "ask the LLM"."""
    if slot is None or len(text) > 60:
        return None
    if slot == NAME:
        match = _NAME.match(text)
        if match and not is_greeting(text) and not re.search(r"\d", text):
            name = match.group("name")
            return name.title() if name.islower() else name
        return None
    if slot not in REQUIRED_FIELDS:
        return None
    if slot == "loan_amnt" and not re.search(r"\d", text) and _MAX_INTENT.search(text):
        return MAX_AMOUNT
    if slot in NUMERIC_FIELDS:
        return parse_number(text)
    cleaned = text.strip().lower().rstrip(".!")
    while (stripped := _FILLER.sub("", cleaned)) != cleaned:
        cleaned = stripped
    for pattern, value in SYNONYMS.get(slot, []):
        if re.fullmatch(rf"(a |an |for |for a |for my )?({pattern})( loan)?", cleaned, re.I):
            return value
    return None


# --- Amounts in the applicant's own style --------------------------------------------------------------------

_INDIAN = re.compile(r"lakh|lac\b|lacs|crore|₹|\brs\.?\b|inr|\b\d{1,2},\d{2},\d{3}\b", re.I)


def number_style(text: str, current: str | None) -> str | None:
    """Once someone writes "9 lakhs" or "9,00,000", amounts are read back their way for the rest of the chat."""
    return "indian" if _INDIAN.search(text) else current


def indian_grouping(value: float) -> str:
    digits = str(round(value))
    if len(digits) <= 3:
        return digits
    head, tail = digits[:-3], digits[-3:]
    groups: list[str] = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    return ",".join([head, *groups, tail] if head else [*groups, tail])


def fmt_amount(value: float, style: str | None, spoken: bool = False) -> str:
    """spoken=True for read-back ("9 lakh"); summaries keep digits ("9,00,000")."""
    if style == "indian":
        if spoken and value >= 100_000 and value % 100_000 == 0:
            return f"{value / 100_000:g} lakh"
        return indian_grouping(value)
    return f"{value:,.0f}"


def playback(field: str, value: Any, style: str | None) -> str:
    """How an answer is read back: "36", "9 lakh a year", "renting", "grade B"."""
    return {
        "person_age": lambda: f"{value:g}",
        "person_income": lambda: f"{fmt_amount(value, style, spoken=True)} a year",
        "person_emp_length": lambda: f"{value:g} year{'' if value == 1 else 's'} at your job",
        "person_home_ownership": lambda: {"RENT": "renting", "OWN": "you own your home", "MORTGAGE": "a mortgage",
                                          "OTHER": "other"}[value],
        "loan_amnt": lambda: fmt_amount(value, style, spoken=True),
        "loan_int_rate": lambda: f"{value:g}%",
        "loan_intent": lambda: {"HOMEIMPROVEMENT": "home improvement",
                                "DEBTCONSOLIDATION": "debt consolidation"}.get(value, str(value).lower()),
        "loan_grade": lambda: f"grade {value}",
        "cb_person_cred_hist_length": lambda: f"{value:g} year{'' if value == 1 else 's'} of credit history",
        "cb_person_default_on_file": lambda: "no previous defaults" if value == "N" else "a previous default",
    }[field]()


def acknowledge(filled: Mapping[str, Any], style: str | None) -> str:
    """Code-written acknowledgement when the LLM wasn't called (or its lead failed the guard)."""
    fields = [f for f in REQUIRED_FIELDS if f in filled]
    if not fields:
        return "Thanks."
    if len(fields) == 1:
        return f"Got it: {playback(fields[0], filled[fields[0]], style)}."
    labels = {"person_age": "age", "person_income": "income", "person_emp_length": "years at your job",
              "person_home_ownership": "home", "loan_amnt": "loan amount", "loan_int_rate": "interest rate",
              "loan_intent": "loan purpose", "loan_grade": "loan grade", "cb_person_cred_hist_length":
              "credit history", "cb_person_default_on_file": "default history"}
    named = [labels[f] for f in fields]
    return f"Thanks, I've noted your {', '.join(named[:-1])} and {named[-1]}."


# --- The LLM's lead line, checked before it is sent ----------------------------------------------------------

_PROMISES = re.compile(r"\b(approv\w*|eligib\w*|qualif\w*|sanction\w*|guarantee\w*|you('ll| will) get|"
                       r"we can (give|lend|offer))\b", re.I)


def safe_lead(lead: str | None) -> str | None:
    """The lead only responds to the person: no questions (code asks those), no promises about the outcome,
    no amounts, short. Anything else is dropped and a code-written acknowledgement is used instead."""
    if not lead:
        return None
    lead = lead.strip().replace("**", "")
    if (len(lead) > 280 or "?" in lead or _PROMISES.search(lead) or re.search(r"\d{4,}|\d+(\.\d+)? ?(lakh|k)\b",
                                                                              lead, re.I)):
        return None
    return lead


# --- Composing replies ---------------------------------------------------------------------------------------

def next_slot(state: Mapping[str, Any]) -> str:
    """The next detail to ask for: missing fields in order, skipped ones last; CONFIRM when complete."""
    missing = missing_fields(state)
    if not missing:
        return CONFIRM
    deferred = state.get("deferred") or []
    return next((f for f in missing if f not in deferred), missing[0])


def opening_message(lender: str, prefix: str = "Hi! I'm Sentinel, {lender}'s loan assistant.") -> AIMessage:
    return AIMessage(
        f"{prefix.format(lender=lender)} I'll ask you 10 quick questions about yourself and the loan you'd like. "
        "It takes about 2 minutes, and you'll get a decision at the end.\n\n"
        "You won't need your ID, card or bank details, so please don't share them here. You can type "
        "\"start over\", \"what do you have so far?\" or \"change my income\" at any time.\n\n"
        "First, what should I call you? (optional)",
        response_metadata={"options": ["Skip"]})


def question_message(state: Mapping[str, Any], prefix: str = "Thanks.", attempt: int = 0,
                     slot: str | None = None) -> AIMessage:
    """The reply asking for the pending detail: prefix, progress, question, why, options.
    attempt > 0 is a repair, worded differently each time, and offers a way out from the third miss."""
    from .graph import summary_message  # summary formatting lives with the graph

    slot = slot or next_slot(state)
    if slot == CONFIRM:
        return AIMessage(summary_message(state), response_metadata={"options": SUMMARY_OPTIONS})

    missing = missing_fields(state)
    number = TOTAL - len(missing) + (0 if slot not in missing else 1)
    question = QUESTIONS[slot]
    deferred = state.get("deferred") or []
    if slot in deferred and all(f in deferred for f in missing):
        question = f"To finish your application I still need this: {question[0].lower()}{question[1:]}"
    options = list(CHOICES.get(slot, []))
    hint = ("Please reply with one of: " + ", ".join(options[:-1]) + f" or {options[-1]}.") if options else ""

    if attempt == 1:
        prefix = prefix if prefix not in ("", "Thanks.") else "Sorry, I didn't quite get that."
        hint = hint or f"For example: {EXAMPLES[slot]}."
    elif attempt == 2:
        prefix = f"Let me put it another way: {EXPLANATIONS[slot]}"
    elif attempt >= 3:
        prefix = "We seem to be stuck on this one. You can answer it, skip it for now, start over, or talk to a person."
        options += ESCAPES

    body = f"({number} of {TOTAL}) {question}"
    if attempt == 0 and slot in WHY:
        body += f" {WHY[slot]}"
    parts = [p for p in (prefix.strip(), body, hint) if p]
    return AIMessage("\n\n".join(parts), response_metadata={"options": options} if options else {})


def clarify_message(state: Mapping[str, Any]) -> AIMessage:
    slot = state.get("pending_slot") or next_slot(state)
    if slot == NAME:
        return AIMessage("I just need something to call you, like your first name. It's optional, so you can "
                         "skip it.", response_metadata={"options": ["Skip"]})
    if slot == CONFIRM:
        return AIMessage("I'm checking these are the details you want me to assess. Reply Yes if they're right, "
                         "or tell me what to change.", response_metadata={"options": SUMMARY_OPTIONS})
    return question_message(state, prefix=EXPLANATIONS[slot], slot=slot)


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
