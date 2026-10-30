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
from collections.abc import Mapping
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

from ..metrics import INVALID_CITATIONS, record_decision, record_funnel, record_slot
from ..prompts import load_prompt, prompt_versions
from ..rag import Retriever
from ..scoring.model import Scorer, get_credit_model
from ..store import ApplicationRecord, ApplicationStore, now
from ..tenants import get_tenant, max_approvable_loan
from .dialogue import (
    CHANGE_OPTIONS,
    CONFIRM,
    ESCAPES,
    MAX_AMOUNT,
    NAME,
    acknowledge,
    fmt_amount,
    is_greeting,
    next_slot,
    number_style,
    opening_message,
    question_message,
    safe_lead,
    understand,
)
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
from .tools import make_tools
from .validation import NUMERIC_FIELDS, parse_number, validate

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 3
# Collected fields live in state, so the interviewer only needs the latest exchange: tokens per turn stay
# flat however long the conversation gets
INTERVIEWER_WINDOW = 6
RESUME_OPTIONS = ["Apply for a loan", "What documents do I need?"]


def out_of_scope_message(lender: str) -> AIMessage:
    return AIMessage(
        f"Sorry, I can only help with personal loan applications and questions about {lender}'s loan "
        f"policies. For anything else, like deposits or accounts, please contact {lender} directly.",
        response_metadata={"options": RESUME_OPTIONS})
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


def summary_message(state: Mapping[str, Any]) -> str:
    lines = [f"- {SUMMARY_LABELS[f]}: {_fmt(f, state[f], state.get('number_style'))}" for f in REQUIRED_FIELDS]
    return "That's everything. Please check your details:\n" + "\n".join(lines) + "\n\nIs everything correct?"


def _fmt(field: str, value: Any, style: str | None = None) -> str:
    if field in ("person_income", "loan_amnt") and isinstance(value, (int, float)):
        return fmt_amount(value, style)
    if isinstance(value, float) and value.is_integer():
        return f"{int(value):,}"
    if isinstance(value, (int, float)):
        return f"{value:,}"
    named = {"Y": "Yes", "N": "No", "HOMEIMPROVEMENT": "Home improvement", "DEBTCONSOLIDATION": "Debt consolidation"}
    if value in named:
        return named[value]
    # Enum values (RENT, MEDICAL) read better capitalised; free text stays as typed
    return str(value).capitalize() if str(value).isupper() else str(value)


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

_PLAIN_NO = re.compile(r"^\s*(no|nope|nah|not quite|not correct|incorrect|wrong)\b[\s,.!]*"
                       r"(change something|something'?s wrong)?[\s.!]*$", re.I)

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
        """Reads the latest message, updates the application, and composes the reply.

        Code decides what happens next and writes the question, progress and options. The LLM (only called
        when code can't read the message) understands the message and writes the opening `lead` line, which
        is checked by `safe_lead` before use. Greetings and short answers never reach the LLM.
        """
        text = str(state["messages"][-1].content)
        tenant = get_tenant(state.get("tenant_id"))
        slot = state.get("pending_slot") or next_slot(state)
        attempts = dict(state.get("slot_attempts") or {})
        style = number_style(text, state.get("number_style"))
        updates: dict[str, Any] = {"number_style": style, "greeted": True}
        question = out_of_scope = lead = None
        process = False
        path = "filled_fast"

        # First message: a greeting gets the opening, which explains what will happen and asks for a name
        if not state.get("greeted") and is_greeting(text):
            record_slot(NAME, "opening")
            return {**updates, "pending_slot": NAME, "messages": [opening_message(tenant.name)]}

        if slot == CONFIRM and _PLAIN_YES.match(text):
            updates["details_confirmed"] = True
            record_slot(CONFIRM, "filled_fast")
        elif slot == CONFIRM and _PLAIN_NO.match(text):
            record_slot(CONFIRM, "declined")
            return {**updates, "pending_slot": CONFIRM, "messages": [AIMessage(
                "No problem. What would you like to change? You can just tell me the right value.",
                response_metadata={"options": CHANGE_OPTIONS})]}
        elif (value := understand(slot, text)) == MAX_AMOUNT:
            # Never quote or pre-fill an amount before assessment: that would imply they qualify for something
            record_slot(slot, "reask")
            return {**updates, "pending_slot": slot, "messages": [question_message(
                state, slot=slot,
                prefix="I can't say what you'd qualify for until your application has been assessed. "
                       "Tell me the amount you need, and you'll get a decision at the end.")]}
        elif slot == NAME and value is not None:
            record_slot(NAME, "filled_fast")
            merged = {**state, "person_name": value}
            return {**updates, "person_name": value, "pending_slot": next_slot(merged),
                    "messages": [question_message(merged, prefix=f"Nice to meet you, {value}. Let's start.")]}
        elif value is not None:
            updates[slot] = value
        else:
            missing = missing_fields(state)
            collected = application_fields(state)
            system = interviewer_prompt.render(
                lender=tenant.name, review_sla=tenant.review_sla,
                collected=json.dumps(collected) if collected else "nothing yet",
                missing=str([f"{f} ({FIELD_LABELS[f]})" for f in missing]),
                asking="what to call them (optional)" if slot == NAME
                else FIELD_LABELS.get(slot, "confirmation of the summary"),
                name=state.get("person_name") or "not given",
            )
            path = "filled_llm"
            result = await llm.interview([SystemMessage(system),
                                          *_dialogue(_recent(state["messages"], INTERVIEWER_WINDOW))])
            # exclude_none so a turn that mentions nothing new doesn't wipe stored values
            updates.update({k: v for k, v in result.model_dump(exclude_none=True).items()
                            if k in REQUIRED_FIELDS or k in ("details_confirmed", "person_name")})
            question, out_of_scope = result.applicant_question, result.out_of_scope
            process, lead = bool(result.process_question), safe_lead(result.lead)
            # A bare number the LLM didn't place answers the question that was asked
            if slot in NUMERIC_FIELDS and not any(f in updates for f in REQUIRED_FIELDS) \
                    and not question and not out_of_scope and (number := parse_number(text)) is not None:
                updates[slot] = number

        updates, range_error = validate(updates, tenant)
        filled = {f: updates[f] for f in REQUIRED_FIELDS if f in updates}
        for field in filled:
            record_slot(field, path)
        income = updates.get("person_income") or state.get("person_income")
        loan = updates.get("loan_amnt") or state.get("loan_amnt")
        if income and loan and income > 0:
            updates["loan_percent_income"] = round(loan / income, 2)

        # A correction after the summary was shown needs a fresh confirmation
        changed = any(state.get(f) != v for f, v in filled.items())
        if slot == CONFIRM and changed:
            updates["details_confirmed"] = False
        elif slot == CONFIRM and _PLAIN_YES.match(text):
            updates["details_confirmed"] = True
        if updates.get("details_confirmed"):
            record_funnel("confirmed")

        # Progress = a new or changed detail, a confirmation, or a real question (about policy or the process).
        # Turns without progress are counted so the service can stop spending tokens on a stuck conversation.
        progress = changed or bool(question) or process or bool(updates.get("details_confirmed")) \
            or bool(updates.get("person_name") and updates.get("person_name") != state.get("person_name"))
        updates["stalled_turns"] = 0 if progress else state.get("stalled_turns", 0) + 1
        updates["applicant_question"] = question
        merged = {**state, **updates}
        updates["deferred"] = [f for f in (state.get("deferred") or []) if merged.get(f) is None]
        merged["deferred"] = updates["deferred"]
        intro = "" if state.get("greeted") else f"Hi! I'm Sentinel, {tenant.name}'s loan assistant. "

        if range_error or (not progress and not out_of_scope):
            # The answer didn't fit: re-ask, worded differently each time, offering a way out from the third miss
            repair_slot = slot if slot != NAME else next_slot(merged)
            attempts[repair_slot] = attempts.get(repair_slot, 0) + 1
            record_slot(repair_slot, "invalid" if range_error else "reask")
            updates["slot_attempts"] = attempts
            updates["pending_slot"] = next_slot(merged)
            if range_error:
                options = ESCAPES if attempts[repair_slot] >= 3 else []
                message = AIMessage(range_error, response_metadata={"options": options} if options else {})
            else:
                apology = "Sorry, I didn't quite get that." if attempts[repair_slot] == 1 else ""
                message = question_message(merged, prefix=intro + (lead or apology), attempt=attempts[repair_slot])
            return {**updates, "messages": [message]}
        if out_of_scope and not progress:
            return {**updates, "messages": [out_of_scope_message(tenant.name)]}

        updates["slot_attempts"] = {k: v for k, v in attempts.items() if merged.get(k) is None}
        updates["pending_slot"] = next_slot(merged)
        if ready_for_scoring(merged):
            get_stream_writer()({"type": "status", "text": "Scoring your application..."})
            return updates  # the advisor's explanation is this turn's reply
        if question:
            return updates  # the advisor answers, then the next question is appended
        if updates["pending_slot"] == CONFIRM and slot != CONFIRM:
            record_funnel("summary_shown")
        prefix = intro + (lead or (acknowledge(filled, style) if filled else "Thanks."))
        return {**updates, "messages": [question_message(merged, prefix=prefix)]}

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
            situation = "The application is still being collected. Nothing has been decided or scored."
            closing = ("Answer the question only, in two or three sentences. Do not ask for application details: "
                       "the app asks the next question after your answer.")
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
            # Back to the application: the dialogue manager appends the next question and its options
            follow_up = question_message(dict(state), prefix="")
            message = AIMessage(f"{message.content}\n\n{follow_up.content}",
                                response_metadata=follow_up.response_metadata)
        if final and final.get("explain") and isinstance(message, AIMessage) and not message.tool_calls:
            message = AIMessage(f"Reference: {state.get('application_id')}\n\n{message.content}")
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

