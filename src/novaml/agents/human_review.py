"""Human-in-the-loop gate before the expensive training step."""

from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

from novaml.agents.base import Agent, AgentContext
from novaml.state import RunState


class HumanReviewAgent(Agent):
    name = "human_review"

    def run(self, state: RunState, ctx: AgentContext) -> dict[str, Any]:
        candidates = state["candidate_models"]
        if ctx.settings.auto_approve or state.get("auto_approve"):
            return {"approved_models": candidates, "needs_training": True, "messages": [self.say("auto-approved")]}

        # Pauses the graph; the checkpoint is persisted and the run can be resumed
        # later, from any worker, with Command(resume={"approved_models": [...]}).
        answer = interrupt(
            {
                "kind": "approve_models",
                "candidates": candidates,
                "reasoning": state.get("selection_reasoning", ""),
            }
        )
        approved = [m for m in (answer or {}).get("approved_models", []) if m in candidates]
        if not approved:
            raise ValueError("no valid models approved")
        return {
            "approved_models": approved,
            "needs_training": True,
            "status": "running",
            "messages": [self.say(f"human approved: {', '.join(approved)}")],
        }
