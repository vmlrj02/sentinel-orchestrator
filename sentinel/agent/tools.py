"""Tools the advisor agent can call.

Rules every tool follows:
- identity comes from graph state (InjectedState), never from model arguments: the model cannot
  choose a tenant or another applicant's data
- arguments are validated; errors come back as text the model can recover from
- every call has a timeout
"""
import asyncio
import json
from typing import Annotated, Any, Literal, Optional

from langchain_core.tools import BaseTool, ToolException, tool
from langgraph.config import get_stream_writer
from langgraph.prebuilt import InjectedState

from ..rag import Retriever
from ..scoring.model import Scorer
from ..tenants import get_tenant
from .state import application_fields

TOOL_TIMEOUT_S = 10


def make_tools(retriever: Retriever, scorer: Scorer) -> list[BaseTool]:

    @tool
    async def search_policy(query: str, state: Annotated[dict, InjectedState]) -> str:
        """Search this lender's policy documents: eligibility, borrowing limits, required documents,
        interest rates, fees, repayment, how decisions are made, appeals, and how to improve chances.
        Use it for any question about the lender's rules or process. Returns passages with ids to cite."""
        tenant_id = state["tenant_id"]
        async with asyncio.timeout(TOOL_TIMEOUT_S):
            results = await asyncio.to_thread(retriever.search, query, tenant_id)
        if not results:
            return "No relevant policy passages found."
        get_stream_writer()({"type": "citations", "items": [
            {"id": r.chunk.id, "title": r.chunk.doc_title, "section": r.chunk.section, "text": r.chunk.text,
             "score": r.score} for r in results]})
        return "\n\n".join(f"[{r.chunk.id}]\n{r.chunk.text}" for r in results)

    @tool
    async def what_if_score(
        state: Annotated[dict, InjectedState],
        loan_amnt: Optional[float] = None,
        loan_int_rate: Optional[float] = None,
        person_income: Optional[float] = None,
        loan_grade: Optional[Literal["A", "B", "C", "D", "E", "F", "G"]] = None,
        loan_intent: Optional[Literal["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT",
                                      "DEBTCONSOLIDATION"]] = None,
        person_home_ownership: Optional[Literal["RENT", "OWN", "MORTGAGE", "OTHER"]] = None,
    ) -> str:
        """Re-score the applicant's own application with some values changed, to answer "what if"
        questions such as "what if I borrowed 8000?". Only pass the values that change. The result is
        hypothetical and never changes the applicant's actual decision."""
        current = application_fields(state)
        if not current.get("person_income"):
            raise ToolException("The application is not complete yet, so it cannot be re-scored.")
        changes = {k: v for k, v in {
            "loan_amnt": loan_amnt, "loan_int_rate": loan_int_rate, "person_income": person_income,
            "loan_grade": loan_grade, "loan_intent": loan_intent, "person_home_ownership": person_home_ownership,
        }.items() if v is not None}
        if not changes:
            raise ToolException("Pass at least one value to change.")
        if any(changes.get(k, 1) <= 0 for k in ("loan_amnt", "person_income")):  # type: ignore[operator]
            raise ToolException("Loan amount and income must be positive.")

        scenario = {**current, **changes}
        scenario["loan_percent_income"] = round(scenario["loan_amnt"] / scenario["person_income"], 2)
        tenant = get_tenant(state.get("tenant_id"))
        async with asyncio.timeout(TOOL_TIMEOUT_S):
            before = state["assessment"]["model"] if state.get("assessment") else await scorer.score(current)
            after = await scorer.score(scenario)
        result: dict[str, Any] = {
            "changes": changes,
            "before": {"default_probability": before["probability"],
                       "policy_decision": tenant.decide(before["probability"], current)[0]},
            "after": {"default_probability": after["probability"],
                      "policy_decision": tenant.decide(after["probability"], scenario)[0],
                      "top_risk_factors": after["explanations"]["top_risk_factors"]},
        }
        get_stream_writer()({"type": "what_if", "result": result})
        return json.dumps(result)

    return [search_policy, what_if_score]
