"""Produces a validated, declarative FeaturePlan (see tools/features.py)."""

from __future__ import annotations

from typing import Any

from novaml.agents.base import Agent, AgentContext
from novaml.state import RunState
from novaml.tools.features import default_plan, validate_plan


class FeatureEngineerAgent(Agent):
    name = "feature_engineer"

    def run(self, state: RunState, ctx: AgentContext) -> dict[str, Any]:
        profile = state["profile"]
        plan, warnings = validate_plan(default_plan(profile), profile, state["target"])
        return {
            "feature_plan": plan.model_dump(),
            "feature_warnings": warnings,
            "messages": [self.say(f"feature plan: {plan.rationale}")],
        }
