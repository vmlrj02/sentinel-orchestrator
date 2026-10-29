from langgraph.graph import StateGraph, START, END
from .state import AgentState, ready_for_scoring
from .nodes import sentinel_node, credit_scoring_node, analyst_node


def route_entry(state: AgentState):
    if state.get("is_processed"):
        return "analyst"      # decision made: answer follow-up questions
    if ready_for_scoring(state):
        return "scoring"      # e.g. retry after the scoring API was down
    return "sentinel"


def route_after_sentinel(state: AgentState):
    return "scoring" if ready_for_scoring(state) else END


def create_sentinel_graph(checkpointer):

    builder = StateGraph(AgentState)

    builder.add_node("sentinel", sentinel_node)
    builder.add_node("scoring", credit_scoring_node)
    builder.add_node("analyst", analyst_node)

    builder.add_conditional_edges(START, route_entry, ["sentinel", "scoring", "analyst"])
    builder.add_conditional_edges("sentinel", route_after_sentinel, ["scoring", END])
    builder.add_conditional_edges(
        "scoring", lambda s: "analyst" if s.get("is_processed") else END, ["analyst", END]
    )
    builder.add_edge("analyst", END)

    return builder.compile(checkpointer=checkpointer)
