"""LangGraph workflow: interviewer -> (confirmed?) -> scoring -> analyst.

Routing is plain code; the LLM only extracts fields and writes text. It never decides when to score.
"""
import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Checkpointer

from ..metrics import DECISIONS
from ..prompts import load_prompt
from ..scoring.model import Scorer
from .llm import AgentLLM
from .state import (
    FIELD_LABELS,
    REQUIRED_FIELDS,
    SUMMARY_LABELS,
    AgentState,
    application_fields,
    missing_fields,
    ready_for_scoring,
)

logger = logging.getLogger(__name__)

SCORING_FAILED = "I couldn't reach the credit model just now. Send any message to try again."


def route_entry(state: AgentState) -> str:
    if state.get("assessment"):
        return "analyst"      # decision made: answer follow-up questions
    if ready_for_scoring(state):
        return "scoring"      # e.g. retry after the credit model was unavailable
    return "interviewer"


def route_after_interviewer(state: AgentState) -> str:
    return "scoring" if ready_for_scoring(state) else END


def route_after_scoring(state: AgentState) -> str:
    return "analyst" if state.get("assessment") else END


def summary_message(state: dict[str, Any]) -> str:
    lines = [f"- {SUMMARY_LABELS[f]}: {_fmt(state[f])}" for f in REQUIRED_FIELDS]
    return (
        "Here's your application:\n" + "\n".join(lines)
        + "\n\nPlease confirm if these details are correct (Yes/No), or tell me what to change."
    )


def _fmt(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return f"{int(value):,}"
    if isinstance(value, (int, float)):
        return f"{value:,}"
    named = {"Y": "Yes", "N": "No", "HOMEIMPROVEMENT": "Home improvement", "DEBTCONSOLIDATION": "Debt consolidation"}
    if value in named:
        return named[value]
    # Enum values (RENT, MEDICAL) read better capitalised; free text like names stays as typed
    return str(value).capitalize() if str(value).isupper() else str(value)


def _recent(messages: list[BaseMessage], limit: int) -> list[BaseMessage]:
    # Context budget: the fields already extracted live in state, so old turns can be dropped
    return messages[-limit:]


def build_graph(llm: AgentLLM, scorer: Scorer, checkpointer: Checkpointer = None,
                max_history_messages: int = 30) -> Any:
    interviewer_prompt = load_prompt("interviewer")
    analyst_prompt = load_prompt("analyst")

    async def interviewer(state: AgentState) -> dict[str, Any]:
        """Extracts fields from the latest message and asks for the next one."""
        missing = missing_fields(state)
        collected = application_fields(state)
        system = interviewer_prompt.render(
            collected=json.dumps(collected) if collected else "nothing yet",
            missing=str([f"{f} ({FIELD_LABELS[f]})" for f in missing]),
        )
        result = await llm.interview([SystemMessage(system), *_recent(state["messages"], max_history_messages)])

        # exclude_none so a turn that mentions nothing new doesn't wipe stored values
        updates: dict[str, Any] = result.model_dump(exclude_none=True)
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

        if ready_for_scoring(merged):
            get_stream_writer()({"type": "status", "text": "Scoring your application..."})
            return {**updates}  # the analyst's explanation is this turn's reply
        return {**updates, "messages": [AIMessage(chat_text)]}

    async def scoring(state: AgentState) -> dict[str, Any]:
        """Sends the confirmed application to the credit model."""
        try:
            assessment = await scorer.score(application_fields(state))
        except Exception:
            logger.exception("Credit model call failed")
            return {"messages": [AIMessage(SCORING_FAILED)]}
        DECISIONS.labels(assessment.get("decision", "unknown")).inc()
        get_stream_writer()({"type": "assessment", "assessment": assessment})
        return {"assessment": assessment}

    async def analyst(state: AgentState) -> dict[str, Any]:
        """Explains the decision (streamed token by token) and answers follow-up questions."""
        system = analyst_prompt.render(
            application=json.dumps(application_fields(state)),
            assessment=json.dumps(state["assessment"]),
        )
        writer = get_stream_writer()
        parts: list[str] = []
        async for token in llm.explain([SystemMessage(system), *_recent(state["messages"], max_history_messages)]):
            parts.append(token)
            writer({"type": "token", "text": token})
        return {"messages": [AIMessage("".join(parts).replace("**", ""))]}

    builder = StateGraph(AgentState)
    builder.add_node("interviewer", interviewer)
    builder.add_node("scoring", scoring)
    builder.add_node("analyst", analyst)
    builder.add_conditional_edges(START, route_entry, ["interviewer", "scoring", "analyst"])
    builder.add_conditional_edges("interviewer", route_after_interviewer, ["scoring", END])
    builder.add_conditional_edges("scoring", route_after_scoring, ["analyst", END])
    builder.add_edge("analyst", END)
    return builder.compile(checkpointer=checkpointer)
