from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.orchestration.langgraph.edges import route_intent, route_judge
from app.orchestration.langgraph.nodes.intent_classifier import intent_classifier
from app.orchestration.langgraph.nodes.reasoning_agent import reasoning_agent
from app.orchestration.langgraph.nodes.judge import judge
from app.orchestration.langgraph.nodes.retriever import retriever
from app.orchestration.langgraph.nodes.synthesizer import synthesizer
from app.orchestration.langgraph.nodes.terminal_nodes import clarify_node, output_node, reject_node
from app.orchestration.langgraph.state.graph_state import GraphState

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def build_graph() -> Any:
    builder = StateGraph(GraphState)

    builder.add_node("intent_classifier", intent_classifier)
    builder.add_node("reasoning_agent", reasoning_agent)
    builder.add_node("retriever", retriever)
    builder.add_node("synthesizer", synthesizer)
    builder.add_node("judge", judge)
    builder.add_node("output", output_node)
    builder.add_node("clarify", clarify_node)
    builder.add_node("reject", reject_node)

    builder.add_edge(START, "intent_classifier")

    builder.add_conditional_edges(
        "intent_classifier",
        route_intent,
        {
            "reasoning_agent": "reasoning_agent",
            "clarify": "clarify",
            "reject": "reject",
        },
    )

    builder.add_edge("reasoning_agent", "retriever")
    builder.add_edge("retriever", "synthesizer")
    builder.add_edge("synthesizer", "judge")

    builder.add_conditional_edges(
        "judge",
        route_judge,
        {
            "output": "output",
            "reasoning_agent": "reasoning_agent",
        },
    )

    builder.add_edge("output", END)
    builder.add_edge("clarify", END)
    builder.add_edge("reject", END)

    graph = builder.compile()
    logger.info("LangGraph compiled successfully")
    return graph
