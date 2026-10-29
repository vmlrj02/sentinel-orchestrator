import json
from functools import lru_cache

import httpx
from langchain_core.messages import AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config.settings import OPENAI_MODEL, SCORING_API_URL
from .state import AgentState, SentinelResponse, REQUIRED_FIELDS, missing_fields

MODEL_FEATURES = [
    "person_age", "person_income", "person_emp_length", "person_home_ownership",
    "loan_amnt", "loan_int_rate", "loan_intent", "loan_grade", "loan_percent_income",
    "cb_person_cred_hist_length", "cb_person_default_on_file",
]

FIELD_LABELS = {
    "person_name": "full name",
    "person_age": "age in years",
    "person_income": "annual income",
    "person_emp_length": "years at current job",
    "person_home_ownership": "whether they rent, own, have a mortgage, or other",
    "loan_amnt": "loan amount requested",
    "loan_int_rate": "loan interest rate in percent",
    "loan_intent": "purpose of the loan",
    "loan_grade": "loan grade A to G",
    "cb_person_cred_hist_length": "length of credit history in years",
    "cb_person_default_on_file": "whether they have ever defaulted before",
}


@lru_cache
def get_llm():
    # Built lazily so importing the graph (e.g. in tests) needs no API key
    return ChatOpenAI(model=OPENAI_MODEL)


async def sentinel_node(state: AgentState):
    """Interviews the applicant: extracts fields from the latest message and asks for the next one."""
    missing = missing_fields(state)
    collected = {k: state.get(k) for k in MODEL_FEATURES + ["person_name"] if state.get(k) is not None}

    system_msg = SystemMessage(content=(
        "### ROLE\n"
        "You are Sentinel, a precise Loan Underwriting Assistant chatting over Telegram. "
        "Your sole objective is to collect data for specific fields.\n\n"

        "### CURRENT STATUS\n"
        f"Already collected: {json.dumps(collected) if collected else 'nothing yet'}\n"
        f"Still MISSING (in order): {[f'{f} ({FIELD_LABELS[f]})' for f in missing]}\n\n"

        "### MANDATORY RULES\n"
        "1. DATA EXTRACTION: Extract any values the user just provided into the structured schema.\n"
        "2. SINGLE QUESTION: Ask for the first MISSING field the user did not just provide, as a natural "
        "question in your own words. Never show internal field names.\n"
        "3. NO EARLY COMPLETION: Do not summarize until every field in the MISSING list is gone.\n"
        "4. VALIDATION: If the user gives an invalid value (e.g. age 'abc'), ask them to clarify.\n"
        "5. BE CONCISE: Professional, short, plain text (no markdown).\n"
        "6. CONFIRMATION: Once nothing is missing, the system shows the user a summary to confirm. "
        "Set details_confirmed=true when the user says yes to it. If they correct something, "
        "extract the corrected value instead.\n\n"

        "### FIELD GUIDELINES\n"
        "- person_home_ownership: RENT, OWN, MORTGAGE or OTHER.\n"
        "- loan_intent: PERSONAL, EDUCATION, MEDICAL, VENTURE, HOMEIMPROVEMENT or DEBTCONSOLIDATION.\n"
        "- loan_grade: A to G.\n"
        "- cb_person_default_on_file: Y or N (has the applicant defaulted before?).\n"
        "- Never ask for loan_percent_income; it is calculated."
    ))

    result: SentinelResponse = await get_llm().with_structured_output(SentinelResponse).ainvoke(
        [system_msg] + state["messages"]
    )

    # exclude_none so a turn that mentions nothing new doesn't wipe stored values
    updates = result.model_dump(exclude_none=True)
    chat_text = updates.pop("assistant_message")

    income = updates.get("person_income") or state.get("person_income")
    loan = updates.get("loan_amnt") or state.get("loan_amnt")
    if income and loan and income > 0:
        updates["loan_percent_income"] = round(loan / income, 2)

    # A correction after the summary was shown needs a fresh confirmation
    if not missing and any(state.get(f) != updates[f] for f in REQUIRED_FIELDS if f in updates):
        updates["details_confirmed"] = False

    # The summary is built here, not by the LLM, so the user always sees exactly what will be scored
    merged = {**state, **updates}
    if not missing_fields(merged) and not merged.get("details_confirmed"):
        chat_text = summary_message(merged)

    return {**updates, "messages": [AIMessage(content=chat_text)]}


SUMMARY_LABELS = {
    "person_name": "Name", "person_age": "Age", "person_income": "Annual income",
    "person_emp_length": "Years at current job", "person_home_ownership": "Home",
    "loan_amnt": "Loan amount", "loan_int_rate": "Interest rate (%)", "loan_intent": "Purpose",
    "loan_grade": "Loan grade", "cb_person_cred_hist_length": "Credit history (years)",
    "cb_person_default_on_file": "Previous default",
}


def _fmt(value) -> str:
    if isinstance(value, float) and value.is_integer():
        return f"{int(value):,}"
    if isinstance(value, (int, float)):
        return f"{value:,}"
    named = {"Y": "Yes", "N": "No", "HOMEIMPROVEMENT": "Home improvement", "DEBTCONSOLIDATION": "Debt consolidation"}
    return named.get(value, str(value).capitalize())


def summary_message(state: AgentState) -> str:
    lines = [f"- {SUMMARY_LABELS[f]}: {_fmt(state[f])}" for f in REQUIRED_FIELDS]
    return (
        "Here's your application:\n" + "\n".join(lines)
        + "\n\nPlease confirm if these details are correct (Yes/No), or tell me what to change."
    )


async def credit_scoring_node(state: AgentState):
    """Sends the collected application to the credit-risk model API."""
    payload = {"data": {k: state.get(k) for k in MODEL_FEATURES}}
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(SCORING_API_URL, json=payload)
            response.raise_for_status()
            prediction = response.json()
    except Exception as e:
        return {"messages": [AIMessage(
            content=f"I couldn't reach the scoring service ({type(e).__name__}). "
                    "Send any message to retry."
        )]}
    return {
        "messages": [AIMessage(content=f"Credit model result: {json.dumps(prediction)}")],
        "is_processed": True,
    }


async def analyst_node(state: AgentState):
    """Explains the model's decision and answers follow-up questions."""
    system_msg = SystemMessage(content=(
        "You are the Loan Analyst. The credit model has already run; its JSON result is in the conversation.\n"
        "Explain the decision to the applicant in plain English and answer any follow-up questions.\n\n"
        "GUIDE:\n"
        "- prediction 0: Approved / Low Risk\n"
        "- prediction 1: Declined / High Risk\n"
        "- probability: the estimated likelihood of default.\n"
        "- top_risk_factors: features that pushed the risk up.\n"
        "- top_strength_factors: features that pulled the risk down.\n\n"
        "Be empathetic, transparent and concise. Plain text, no markdown. "
        "Translate feature names into plain words (e.g. loan_percent_income -> loan size relative to income)."
    ))
    response = await get_llm().ainvoke([system_msg] + state["messages"])
    return {"messages": [response]}
