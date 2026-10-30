from collections.abc import Mapping
from typing import Annotated, Any, Literal, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# The credit model's inputs, and nothing more: the name is not used for the decision, so it is not
# collected (data minimisation; also one less question for people to refuse)
REQUIRED_FIELDS = [
    "person_age", "person_income", "person_emp_length",
    "person_home_ownership", "loan_amnt", "loan_int_rate", "loan_intent",
    "loan_grade", "cb_person_cred_hist_length", "cb_person_default_on_file",
]

# Plain-language meaning of each field, shown to the interviewer LLM
FIELD_LABELS = {
    "person_name": "name",
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

# Fields with a fixed set of answers: the question always lists them, and the UI offers them as buttons
CHOICES = {
    "person_home_ownership": ["Rent", "Own", "Mortgage", "Other"],
    "loan_intent": ["Personal", "Education", "Medical", "Venture", "Home improvement", "Debt consolidation"],
    "loan_grade": ["A", "B", "C", "D", "E", "F", "G"],
    "cb_person_default_on_file": ["Yes", "No"],
}

CHOICE_QUESTIONS = {
    "person_home_ownership": "Do you rent or own your home?",
    "loan_intent": "What is the loan for?",
    "loan_grade": "What loan grade were you given?",
    "cb_person_default_on_file": "Have you ever defaulted on a loan before?",
}

# Labels for the confirmation summary shown to the applicant
SUMMARY_LABELS = {
    "person_name": "Name", "person_age": "Age", "person_income": "Annual income",
    "person_emp_length": "Years at current job", "person_home_ownership": "Home",
    "loan_amnt": "Loan amount", "loan_int_rate": "Interest rate (%)", "loan_intent": "Purpose",
    "loan_grade": "Loan grade", "cb_person_cred_hist_length": "Credit history (years)",
    "cb_person_default_on_file": "Previous default",
}


class AgentState(TypedDict, total=False):
    # Personal info
    person_name: Optional[str]
    person_age: Optional[int]
    person_income: Optional[float]
    person_emp_length: Optional[float]
    person_home_ownership: Optional[str]

    # Loan info
    loan_amnt: Optional[float]
    loan_int_rate: Optional[float]
    loan_intent: Optional[str]
    loan_grade: Optional[str]
    loan_percent_income: Optional[float]  # derived, never asked

    # Credit info
    cb_person_cred_hist_length: Optional[int]
    cb_person_default_on_file: Optional[str]

    tenant_id: str                         # set by the service on the first turn, never by the LLM
    details_confirmed: bool
    applicant_question: Optional[str]      # policy question the interviewer routed to the advisor
    stalled_turns: int                     # consecutive turns with no progress (cost guard)
    application_id: Optional[str]
    assessment: Optional[dict[str, Any]]   # {"model": credit model output, "policy_decision", "policy_reason"}
    final_decision: Optional[dict[str, Any]]  # {"decision", "decided_by", "note"}
    messages: Annotated[list[BaseMessage], add_messages]


class SentinelResponse(BaseModel):
    """Extraction and response schema: the chat reply plus any fields the user just gave."""
    assistant_message: str = Field(description="The conversational chat response to the user.")
    person_name: Optional[str] = None
    person_age: Optional[int] = Field(None, description="Age in years")
    person_income: Optional[float] = Field(None, description="Annual income")
    person_emp_length: Optional[float] = Field(None, description="Years in current employment")
    loan_amnt: Optional[float] = None
    loan_int_rate: Optional[float] = Field(None, description="Interest rate in percent, e.g. 11.5")
    cb_person_cred_hist_length: Optional[int] = Field(None, description="Credit history length in years")
    person_home_ownership: Optional[Literal["RENT", "OWN", "MORTGAGE", "OTHER"]] = None
    loan_intent: Optional[Literal[
        "PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"
    ]] = None
    loan_grade: Optional[Literal["A", "B", "C", "D", "E", "F", "G"]] = None
    cb_person_default_on_file: Optional[Literal["Y", "N"]] = None
    applicant_question: Optional[str] = Field(
        None,
        description="If the applicant asked a question about the lender's rules, process, documents, rates, "
                    "fees, limits or timelines, copy the question here. Leave empty otherwise.",
    )
    out_of_scope: Optional[bool] = Field(
        None,
        description="True if the latest message asks for something other than a personal loan application "
                    "or this lender's loan policies (e.g. deposits, accounts, cards, transfers).",
    )
    details_confirmed: Optional[bool] = Field(
        None,
        description="True only when the summary has been shown and the user explicitly confirms it is "
                    "correct. False if they say something is wrong.",
    )


def missing_fields(state: Mapping[str, Any]) -> list[str]:
    return [f for f in REQUIRED_FIELDS if state.get(f) is None]


def ready_for_scoring(state: Mapping[str, Any]) -> bool:
    return not missing_fields(state) and bool(state.get("details_confirmed"))


def application_fields(state: Mapping[str, Any]) -> dict[str, Any]:
    return {f: state.get(f) for f in [*REQUIRED_FIELDS, "loan_percent_income"] if state.get(f) is not None}
