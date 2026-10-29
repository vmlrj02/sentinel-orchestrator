from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

REQUIRED_FIELDS = [
    "person_name", "person_age", "person_income", "person_emp_length",
    "person_home_ownership", "loan_amnt", "loan_int_rate", "loan_intent",
    "loan_grade", "cb_person_cred_hist_length", "cb_person_default_on_file",
]


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

    details_confirmed: bool
    is_processed: bool
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
    details_confirmed: Optional[bool] = Field(
        None,
        description="True only when every field is collected, the summary has been shown, "
                    "and the user explicitly confirms it is correct. False if they say something is wrong.",
    )


def missing_fields(state: AgentState) -> list[str]:
    return [f for f in REQUIRED_FIELDS if state.get(f) is None]


def ready_for_scoring(state: AgentState) -> bool:
    return not missing_fields(state) and bool(state.get("details_confirmed"))
