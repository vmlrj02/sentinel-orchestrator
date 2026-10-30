"""LangGraph workflow.

    START ─► interviewer ──(confirmed)──► scoring ──(policy: approve/decline)──► advisor ◄──► tools
                 │                            └──(policy: refer)──► human_review ──┘
                 └──(policy question)──► advisor

A workflow, not a free-roaming agent: routing is plain code. The LLM extracts fields, writes text,
and (only inside the advisor) chooses tools. Credit decisions come from the model plus the lender's
policy rules, or from a human underwriter, never from the LLM.
"""
import asyncio
import json
import logging
import re
from typing import Any
from uuid import uuid4

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    message_chunk_to_message,
)
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Checkpointer, interrupt

from ..metrics import INVALID_CITATIONS, record_decision
from ..prompts import load_prompt, prompt_versions
from ..rag import Retriever
from ..scoring.model import Scorer, get_credit_model
from ..store import ApplicationRecord, ApplicationStore, now
from ..tenants import get_tenant, max_approvable_loan
from .llm import AgentLLM
from .state import (
    CHOICE_QUESTIONS,
    CHOICES,
    FIELD_LABELS,
    REQUIRED_FIELDS,
    SUMMARY_LABELS,
    AgentState,
    application_fields,
    missing_fields,
    ready_for_scoring,
)
from .tools import make_tools

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 3
SCORING_FAILED = "I couldn't reach the credit model just now. Send any message to try again."


def route_entry(state: AgentState) -> str:
    if state.get("final_decision"):
        return "advisor"      # decided: explain and answer follow-up questions
    if ready_for_scoring(state):
        return "scoring"      # e.g. retry after the credit model was unavailable
    return "interviewer"


def route_after_interviewer(state: AgentState) -> str:
    if ready_for_scoring(state):
        return "scoring"
    if state.get("applicant_question"):
        return "advisor"
    return END


def route_after_scoring(state: AgentState) -> str:
    assessment = state.get("assessment")
    if not assessment:
        return END
    return "human_review" if assessment["policy_decision"] == "referred" else "advisor"


def route_after_advisor(state: AgentState) -> str:
    last = state["messages"][-1]
    return "tools" if isinstance(last, AIMessage) and last.tool_calls else END


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


def with_choices(text: str, state: dict[str, Any], ask_in_code: bool = False) -> AIMessage:
    """Reply asking for the next detail. If that detail has fixed answers, say so explicitly and attach
    them as quick replies, in code, so it never depends on the LLM remembering to list them.

    ask_in_code: replace the LLM's question with a fixed one for choice fields, so the question and
    its options can never disagree (the LLM may want a detail the code already considers collected)."""
    missing = missing_fields(state)
    if not missing:
        options = ["Yes", "No"]  # the confirmation summary
    elif missing[0] in CHOICES:
        options = CHOICES[missing[0]]
        if ask_in_code:
            text = f"Thanks. {CHOICE_QUESTIONS[missing[0]]}"
        listed = ", ".join(options[:-1]) + f" or {options[-1]}"
        text = f"{text.rstrip()}\n\nPlease reply with one of: {listed}."
    else:
        options = []
    return AIMessage(text, response_metadata={"options": options} if options else {})


def _recent(messages: list[BaseMessage], limit: int) -> list[BaseMessage]:
    """Context budget: keep the last `limit` messages, starting at a user message so a tool call
    is never separated from its result. Extracted fields live in state, so old turns can go."""
    window = messages[-limit:]
    while window and not isinstance(window[0], HumanMessage):
        window = window[1:]
    return window


def _dialogue(messages: list[BaseMessage]) -> list[BaseMessage]:
    """User and assistant text only: the interviewer doesn't need the advisor's tool traffic."""
    return [m for m in messages if isinstance(m, HumanMessage) or (isinstance(m, AIMessage) and m.content
                                                                     and not m.tool_calls)]


# An unambiguous yes to the summary is confirmed in code; anything fuzzier is left to the LLM
_PLAIN_YES = re.compile(
    r"^\s*(yes|yep|yeah|yup|y|correct|confirm(ed)?|that'?s (right|correct)|all (good|correct)|looks (good|right))"
    r"([\s,.!]+(it is|that'?s (right|correct|all)|all correct|correct|thanks?|please|now))*[\s.!]*$", re.I)

_CITATION = re.compile(r"\s?\[([a-z0-9_-]+/[a-z0-9_.-]+(?:#[a-z0-9-]+)?)\]")


def strip_unretrieved_citations(text: str, messages: list[BaseMessage]) -> str:
    """Keep only citations to passages search_policy actually returned in this conversation."""
    retrieved = {m.group(1) for msg in messages if isinstance(msg, ToolMessage) and msg.name == "search_policy"
                 for m in _CITATION.finditer(str(msg.content))}

    def check(match: re.Match[str]) -> str:
        if match.group(1) in retrieved:
            return match.group(0)
        INVALID_CITATIONS.inc()
        logger.warning("Removed unretrieved citation", extra={"fields": {"citation": match.group(1)}})
        return ""

    return _CITATION.sub(check, text)


def _tool_rounds_this_turn(messages: list[BaseMessage]) -> int:
    rounds = 0
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            break
        rounds += isinstance(m, AIMessage) and bool(m.tool_calls)
    return rounds


def build_graph(llm: AgentLLM, scorer: Scorer, retriever: Retriever, store: ApplicationStore,
                checkpointer: Checkpointer = None, max_history_messages: int = 30) -> Any:
    interviewer_prompt = load_prompt("interviewer")
    advisor_prompt = load_prompt("advisor")
    tools = make_tools(retriever, scorer)
    policy_tools = [t for t in tools if t.name == "search_policy"]

    async def interviewer(state: AgentState) -> dict[str, Any]:
        """Extracts fields from the latest message and asks for the next one."""
        missing = missing_fields(state)
        collected = application_fields(state)
        system = interviewer_prompt.render(
            collected=json.dumps(collected) if collected else "nothing yet",
            missing=str([f"{f} ({FIELD_LABELS[f]})" for f in missing]),
        )
        result = await llm.interview([SystemMessage(system), *_dialogue(_recent(state["messages"],
                                                                                  max_history_messages))])

        # exclude_none so a turn that mentions nothing new doesn't wipe stored values
        updates: dict[str, Any] = result.model_dump(exclude_none=True)
        chat_text = updates.pop("assistant_message")
        updates["applicant_question"] = result.applicant_question

        income = updates.get("person_income") or state.get("person_income")
        loan = updates.get("loan_amnt") or state.get("loan_amnt")
        if income and loan and income > 0:
            updates["loan_percent_income"] = round(loan / income, 2)

        # A correction after the summary was shown needs a fresh confirmation
        corrected = any(state.get(f) != updates[f] for f in REQUIRED_FIELDS if f in updates)
        if not missing and corrected:
            updates["details_confirmed"] = False
        elif not missing and _PLAIN_YES.match(str(state["messages"][-1].content)):
            updates["details_confirmed"] = True

        merged = {**state, **updates}
        if ready_for_scoring(merged):
            get_stream_writer()({"type": "status", "text": "Scoring your application..."})
            return updates  # the advisor's explanation is this turn's reply
        if result.applicant_question:
            return updates  # the advisor answers, then asks for the next detail
        # The summary is built here, not by the LLM, so the user always sees exactly what will be scored
        if not missing_fields(merged):
            chat_text = summary_message(merged)
        return {**updates, "messages": [with_choices(chat_text, merged, ask_in_code=True)]}

    async def scoring(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        """Scores the confirmed application, applies the lender's policy, and records the decision."""
        tenant = get_tenant(state.get("tenant_id"))
        application = application_fields(state)
        try:
            model_result = await scorer.score(application)
        except Exception:
            logger.exception("Credit model call failed")
            return {"messages": [AIMessage(SCORING_FAILED)]}

        decision, reason = tenant.decide(model_result["probability"], application)
        application_id = state.get("application_id") or f"app-{uuid4().hex[:10]}"
        assessment: dict[str, Any] = {"model": model_result, "policy_decision": decision, "policy_reason": reason,
                      "lender": tenant.name}
        if decision != "approved":
            # Counterfactual computed by the real model and policy, so explanations never guess
            model = get_credit_model()
            assessment["max_auto_approved_loan"] = await asyncio.to_thread(
                max_approvable_loan, tenant, application, lambda a: model.predict(a)["probability"])
        await store.upsert(ApplicationRecord(
            application_id=application_id, conversation_id=config["configurable"]["thread_id"],
            tenant_id=tenant.id, status="pending_review" if decision == "referred" else decision,
            decided_by="pending" if decision == "referred" else "policy", application=application,
            model_result=model_result, policy_reason=reason, model_version=scorer.version,
            prompt_versions=prompt_versions(), trace_id=config["configurable"].get("trace_id"),
            decided_at=None if decision == "referred" else now(),
        ))
        record_decision(decision)
        get_stream_writer()({"type": "assessment", "application_id": application_id, **assessment})

        updates: dict[str, Any] = {"assessment": assessment, "application_id": application_id}
        if decision == "referred":
            updates["messages"] = [AIMessage(
                f"Thank you. Your application ({application_id}) needs a quick look from an underwriter at "
                f"{tenant.name}: {reason}. They usually decide within {tenant.review_sla}, and I'll message you "
                "here as soon as they do.")]
        else:
            updates["final_decision"] = {"decision": decision, "decided_by": "policy", "note": reason,
                                         "explain": True}
        return updates

    async def human_review(state: AgentState) -> dict[str, Any]:
        """Pauses the graph until an underwriter decides. State is checkpointed, so this can wait days."""
        application_id, assessment = state.get("application_id"), state.get("assessment")
        assert application_id and assessment, "human_review runs only after scoring"
        review = interrupt({"application_id": application_id, "reason": assessment["policy_reason"]})
        decision, reviewer, note = review["decision"], review["reviewer"], review.get("note")
        record = await store.get(application_id)
        if record:
            record.status, record.decided_by = decision, f"underwriter:{reviewer}"
            record.review_note, record.decided_at = note, now()
            await store.upsert(record)
        record_decision(f"underwriter_{decision}")
        return {"final_decision": {"decision": decision, "decided_by": f"underwriter:{reviewer}", "note": note,
                                   "explain": True}}

    async def advisor(state: AgentState) -> dict[str, Any]:
        """Answers policy questions (RAG) and explains decisions (what-if tool). Streams its answer."""
        tenant = get_tenant(state.get("tenant_id"))
        final = state.get("final_decision")
        if final:
            situation = (f"The application has been decided: {final['decision'].upper()}, decided by "
                         f"{'an underwriter' if final['decided_by'].startswith('underwriter') else 'policy rules'}"
                         f" ({final.get('note') or 'no note'}). Model and policy record: "
                         f"{json.dumps(state.get('assessment'))}")
            closing = ("This turn: explain the decision now: the outcome, who made it, the main factors that "
                       "raised and lowered risk, and what would change the outcome (use max_auto_approved_loan "
                       "if present)." if final.get("explain")
                       else "This turn: answer the applicant's latest message.")
            available = tools
        else:
            missing = missing_fields(state)
            situation = "The application is still being collected. Nothing has been decided or scored."
            closing = (f"After answering, continue the application with one short, natural question asking for "
                       f"their {FIELD_LABELS[missing[0]]}." if missing
                       else "After answering, ask them to confirm the application summary above (Yes/No).")
            available = policy_tools
        if _tool_rounds_this_turn(state["messages"]) >= MAX_TOOL_ROUNDS:
            available = []  # step limit: answer with what has been gathered

        system = advisor_prompt.render(lender=tenant.name, situation=situation, closing_instruction=closing,
                                       application=json.dumps(application_fields(state)))
        writer = get_stream_writer()
        full: AIMessageChunk | None = None
        async for chunk in llm.advise([SystemMessage(system), *_recent(state["messages"], max_history_messages)],
                                      available):
            full = chunk if full is None else full + chunk
            if isinstance(chunk.content, str) and chunk.content:
                writer({"type": "token", "text": chunk.content})
        message = message_chunk_to_message(full) if full is not None else AIMessage("")
        if isinstance(message, AIMessage):
            message.content = strip_unretrieved_citations(str(message.content).replace("**", ""),
                                                          state["messages"])
            for call in message.tool_calls:
                writer({"type": "tool_call", "name": call["name"], "args": call["args"]})
        if not final and isinstance(message, AIMessage) and not message.tool_calls:
            message = with_choices(str(message.content), dict(state))  # it ends by asking for the next detail
        updates: dict[str, Any] = {"messages": [message], "applicant_question": None}
        if final and final.get("explain") and not getattr(message, "tool_calls", None):
            updates["final_decision"] = {**final, "explain": False}
        return updates

    builder = StateGraph(AgentState)
    builder.add_node("interviewer", interviewer)
    builder.add_node("scoring", scoring)
    builder.add_node("human_review", human_review)
    builder.add_node("advisor", advisor)
    builder.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    builder.add_conditional_edges(START, route_entry, ["interviewer", "scoring", "advisor"])
    builder.add_conditional_edges("interviewer", route_after_interviewer, ["scoring", "advisor", END])
    builder.add_conditional_edges("scoring", route_after_scoring, ["human_review", "advisor", END])
    builder.add_edge("human_review", "advisor")
    builder.add_conditional_edges("advisor", route_after_advisor, ["tools", END])
    builder.add_edge("tools", "advisor")
    return builder.compile(checkpointer=checkpointer)

