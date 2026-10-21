"""LangGraph workflow: retrieve, grade, rewrite the query if needed, then answer."""

from collections.abc import Callable
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph


class AgentState(TypedDict, total=False):
    question: str
    query: str
    top_k: int
    attempts: int
    chunks: list
    relevant: list
    result: Any


def build_agent_graph(
    retrieve: Callable[[str, int], list],
    keep_relevant: Callable[[list], list],
    rewrite: Callable[[str, str], str],
    generate: Callable[..., Any],
    max_rewrites: int,
):
    """Compile the graph.

    Flow::

        retrieve -> grade --(relevant)--------------> generate -> END
                      |
                      +--(weak and retries left)--> rewrite -> retrieve
                      |
                      +--(weak and out of retries)-> generate -> END
    """

    def retrieve_node(state: AgentState) -> dict:
        return {"chunks": retrieve(state["query"], state["top_k"])}

    def grade_node(state: AgentState) -> dict:
        return {"relevant": keep_relevant(state["chunks"])}

    def rewrite_node(state: AgentState) -> dict:
        new_query = rewrite(state["question"], state["query"])
        return {"query": new_query, "attempts": state["attempts"] + 1}

    def generate_node(state: AgentState) -> dict:
        rewritten = state["query"] if state["query"] != state["question"] else None
        return {"result": generate(state["question"], state["relevant"], rewritten)}

    def route_after_grade(state: AgentState) -> str:
        if state["relevant"] or state["attempts"] >= max_rewrites:
            return "generate"
        return "rewrite"

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("generate", generate_node)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", route_after_grade, {"generate": "generate", "rewrite": "rewrite"})
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", END)
    return graph.compile()
