"""Packages the chosen pipeline as a servable bundle with a model card."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from novaml.agents.base import Agent, AgentContext
from novaml.agents.evaluator import baseline_entry
from novaml.serving.bundle import write_bundle
from novaml.state import RunState
from novaml.tools.ml import to_display


class DeployerAgent(Agent):
    name = "deployer"

    def run(self, state: RunState, ctx: AgentContext) -> dict[str, Any]:
        store = ctx.store(state)
        target, pt, metric = state["target"], state["problem_type"], state["metric"]
        best = state["best"]
        pipeline = store.load_model(best["artifact"])
        X_example = store.load_frame(state["data_refs"]["train"]).drop(columns=[target]).head(50)

        card = build_model_card(state)
        bundle = write_bundle(store.path("bundle/model.joblib").parent, pipeline, X_example, card)
        store.save_json("model_card.json", card)
        return {
            "bundle_dir": str(bundle),
            "model_card": card,
            "status": "completed",
            "messages": [
                self.say(
                    f"exported {best['model']} ({metric}={to_display(metric, pt, best['cv_score']):.4f}) to {bundle}"
                )
            ],
        }


def build_model_card(state: RunState) -> dict[str, Any]:
    pt, metric = state["problem_type"], state["metric"]
    best = state["best"]
    base = baseline_entry(state.get("leaderboard", []))
    calls = state.get("llm_calls", [])
    warnings = list(state.get("feature_warnings", []))
    lift = None
    if base:
        lift = best["cv_score"] - base["cv_score"]
        if lift <= 0:
            warnings.append("best model does not beat the naive baseline in cross-validation")
    return {
        "run_id": state["run_id"],
        "created_at": datetime.now(UTC).isoformat(),
        "model_name": best["model"],
        "params": best.get("params", {}),
        "problem_type": pt,
        "target": state["target"],
        "metric": metric,
        "cv": {
            "score": to_display(metric, pt, best["cv_score"]),
            "std": best.get("cv_std"),
            "lift_over_baseline": lift,
        },
        "holdout": state.get("holdout", {}),
        "data": {
            "train_rows": state["profile"]["n_rows"],
            "n_features": state["profile"]["n_features"],
        },
        "feature_plan": state.get("feature_plan", {}),
        "selection_reasoning": state.get("selection_reasoning", ""),
        "llm": {
            "calls": len(calls),
            "llm_decisions": sum(c.get("source") == "llm" for c in calls),
            "policy_fallbacks": sum(c.get("source") == "policy" for c in calls),
            "tokens": sum(c.get("input_tokens", 0) + c.get("output_tokens", 0) for c in calls),
            "cost_usd": round(sum(c.get("cost_usd", 0.0) for c in calls), 6),
        },
        "warnings": warnings,
    }
