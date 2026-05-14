from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.pipeline.graph.edges import route_intent, route_judge
from app.pipeline.graph.nodes.intent_classifier import intent_classifier
from app.pipeline.graph.nodes.reasoning_agent import reasoning_agent
from app.pipeline.graph.nodes.judge import judge
from app.pipeline.graph.nodes.retriever import retriever
from app.pipeline.graph.nodes.synthesizer import synthesizer
from app.pipeline.graph.nodes.terminal_nodes import clarify_node, output_node, reject_node
from app.pipeline.graph.state.graph_state import GraphState

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def build_graph() -> Any:
    """
    Build and compile the LangGraph state graph.

    New flow:
    START → intent_classifier → [route_intent] → reasoning_agent | clarify | reject
    reasoning_agent → retriever → synthesizer → judge → [route_judge] → output | reasoning_agent (retry)
    """
    builder = StateGraph(GraphState)

    # Add all nodes
    builder.add_node("intent_classifier", intent_classifier)
    builder.add_node("reasoning_agent", reasoning_agent)
    builder.add_node("retriever", retriever)
    builder.add_node("synthesizer", synthesizer)
    builder.add_node("judge", judge)
    builder.add_node("output", output_node)
    builder.add_node("clarify", clarify_node)
    builder.add_node("reject", reject_node)

    # Build graph edges
    builder.add_edge(START, "intent_classifier")

    # Intent classification routing
    builder.add_conditional_edges(
        "intent_classifier",
        route_intent,
        {
            "reasoning_agent": "reasoning_agent",
            "clarify": "clarify",
            "reject": "reject",
        },
    )

    # Linear flow: reasoning → retrieval → synthesis → judgment
    builder.add_edge("reasoning_agent", "retriever")
    builder.add_edge("retriever", "synthesizer")
    builder.add_edge("synthesizer", "judge")

    # Judge routing: output or retry reasoning_agent
    builder.add_conditional_edges(
        "judge",
        route_judge,
        {
            "output": "output",
            "reasoning_agent": "reasoning_agent",  # Retry reasoning on failure
        },
    )

    # Terminal nodes
    builder.add_edge("output", END)
    builder.add_edge("clarify", END)
    builder.add_edge("reject", END)

    graph = builder.compile()
    logger.info("LangGraph compiled successfully with new flow: intent → reasoning → retriever → synthesizer → judge")
    return graph
