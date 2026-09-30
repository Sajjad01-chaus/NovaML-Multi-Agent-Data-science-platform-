"""Supervisor: the hub every agent returns to, deciding which agent runs next.

Routing is a pure function of state, so it is deterministic, replayable from any
checkpoint and unit-testable. Judgement lives in the agents (planner, analyst,
critic, ...), which record their decisions in state; the supervisor enforces
ordering invariants and hard budgets on top of them:

* an agent never runs before its inputs exist,
* failed runs stop immediately,
* a global step budget turns a routing bug into a graceful stop instead of a loop.
"""

from __future__ import annotations

from langgraph.graph import END

from novaml.config import Settings
from novaml.state import RunState

STEP_BUDGET_EXCEEDED = "step budget exceeded"


def next_step(state: RunState, settings: Settings) -> str:
    if state.get("status") in ("failed", "completed"):
        return END
    if len(state.get("events", [])) >= settings.max_graph_steps:
        # Out of budget: ship the best model so far if we have one.
        return "deployer" if state.get("best") else END

    if not state.get("profile"):
        return "profiler"
    if not state.get("plan"):
        return "planner"
    if state["plan"].get("run_analysis") and state.get("analysis") is None:
        return "analyst"
    if not state.get("feature_plan"):
        return "feature_engineer"
    if not state.get("candidate_models"):
        return "model_selector"
    if not state.get("approved_models"):
        return "human_review"
    if state.get("needs_training"):
        return "trainer"
    if state.get("needs_evaluation"):
        return "evaluator"
    if state.get("needs_critique"):
        return "critic"
    if state.get("accepted") and state.get("best"):
        return "deployer"
    return END
