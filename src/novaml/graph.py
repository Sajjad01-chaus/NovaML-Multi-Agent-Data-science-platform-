"""LangGraph wiring."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from novaml.agents.base import AgentContext, as_node
from novaml.agents.deployer import DeployerAgent
from novaml.agents.evaluator import EvaluatorAgent
from novaml.agents.feature_engineer import FeatureEngineerAgent
from novaml.agents.human_review import HumanReviewAgent
from novaml.agents.model_selector import ModelSelectorAgent
from novaml.agents.profiler import ProfilerAgent
from novaml.agents.trainer import TrainerAgent
from novaml.state import RunState

PIPELINE = [
    ProfilerAgent,
    FeatureEngineerAgent,
    ModelSelectorAgent,
    HumanReviewAgent,
    TrainerAgent,
    EvaluatorAgent,
    DeployerAgent,
]


def build_graph(ctx: AgentContext, checkpointer: Any = None):
    g = StateGraph(RunState)
    agents = [cls() for cls in PIPELINE]
    for a in agents:
        g.add_node(a.name, as_node(a, ctx))

    g.add_edge(START, agents[0].name)
    for cur, nxt in zip(agents, agents[1:], strict=False):
        g.add_conditional_edges(cur.name, _continue_or_stop(nxt.name), [nxt.name, END])
    g.add_edge(agents[-1].name, END)
    return g.compile(checkpointer=checkpointer)


def _continue_or_stop(next_node: str):
    def route(state: RunState) -> str:
        return END if state.get("status") == "failed" else next_node

    return route
