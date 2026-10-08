"""LangGraph wiring: hub-and-spoke around a supervisor.

    profiler ─┐
    planner  ─┤
    analyst  ─┤            ┌─> trainer ─> evaluator ─> critic ─┐
    features ─┼─ supervisor┤                                    │ (tune / revise
    selector ─┤            └─> deployer ─> END                  │  features / try
    review   ─┘   ^────────────────────────────────────────────┘  other models)

Every agent returns to the supervisor, which picks the next agent from state
(see supervisor.py). The critic can send the run back for another round, so
the path through the graph is decided at runtime, within hard budgets.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from novaml.agents.analyst import AnalystAgent
from novaml.agents.base import AgentContext, as_node
from novaml.agents.critic import CriticAgent
from novaml.agents.deployer import DeployerAgent
from novaml.agents.evaluator import EvaluatorAgent
from novaml.agents.feature_engineer import FeatureEngineerAgent
from novaml.agents.human_review import HumanReviewAgent
from novaml.agents.model_selector import ModelSelectorAgent
from novaml.agents.planner import PlannerAgent
from novaml.agents.profiler import ProfilerAgent
from novaml.agents.trainer import TrainerAgent
from novaml.state import RunState
from novaml.supervisor import STEP_BUDGET_EXCEEDED, next_step

AGENTS = [
    ProfilerAgent,
    PlannerAgent,
    AnalystAgent,
    FeatureEngineerAgent,
    ModelSelectorAgent,
    HumanReviewAgent,
    TrainerAgent,
    EvaluatorAgent,
    CriticAgent,
    DeployerAgent,
]


def build_graph(ctx: AgentContext, checkpointer: Any = None):
    g = StateGraph(RunState)
    names = []
    for cls in AGENTS:
        agent = cls()
        g.add_node(agent.name, as_node(agent, ctx))
        g.add_edge(agent.name, "supervisor")
        names.append(agent.name)

    def supervisor(state: RunState) -> dict[str, Any]:
        # Out of step budget with nothing to ship: fail loudly rather than stall.
        if next_step(state, ctx.settings) == END and state.get("status") not in ("failed", "completed", "cancelled"):
            return {"status": "failed", "errors": [f"supervisor: {STEP_BUDGET_EXCEEDED} or no route"]}
        return {}

    g.add_node("supervisor", supervisor)
    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", lambda s: next_step(s, ctx.settings), [*names, END])
    # Supervisor hops count against LangGraph's recursion limit too.
    return g.compile(checkpointer=checkpointer)
