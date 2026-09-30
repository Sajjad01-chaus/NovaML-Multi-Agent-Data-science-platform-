"""Critic: the reflection step. Reviews the cross-validation leaderboard and decides
whether to accept, tune, revise features, or try different model families.

It only sees cross-validation results (never the holdout), and the number of
improvement rounds is capped, so the loop is bounded and cannot overfit the test set.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from novaml.agents.base import Agent, AgentContext
from novaml.agents.evaluator import baseline_entry, best_entry
from novaml.config import Settings
from novaml.llm.gateway import ValidationFailed, as_data
from novaml.state import RunState
from novaml.tools.ml import BASELINE, METRICS, REGISTRY, to_display

Decision = Literal["accept", "tune", "revise_features", "try_other_models"]


class CriticVerdict(BaseModel):
    decision: Decision
    reasoning: str = Field(description="2-3 sentences grounded in the numbers given.")
    feature_feedback: str = Field(default="", description="For revise_features: what to change in the feature plan.")


SYSTEM = (
    "You are a critical reviewer of an AutoML run. Given cross-validation results, "
    "decide the single most valuable next action within the remaining budget: "
    "'accept' (good enough or further work unlikely to help), 'tune' (hyperparameter "
    "search on the best models), 'revise_features' (features look like the bottleneck), "
    "or 'try_other_models' (other model families may fit better). Watch for: no lift "
    "over the baseline, large train-vs-CV gaps (overfitting), high CV variance."
)


def summarize(state: RunState) -> dict[str, Any]:
    pt, metric = state["problem_type"], state["metric"]
    lb = state.get("leaderboard", [])
    base = baseline_entry(lb)
    rows = [
        {
            "round": e["round"],
            "model": e["model"],
            "mode": e.get("mode"),
            "cv": round(to_display(metric, pt, e["cv_score"]), 4),
            "cv_std": round(e["cv_std"], 4),
            "train_minus_cv": round(e["overfit_gap"], 4),
        }
        for e in lb
        if "error" not in e
    ]
    tried = sorted({e["model"] for e in lb} - {BASELINE})
    return {
        "metric": metric,
        "higher_is_better": METRICS[pt][metric][1],
        "baseline_cv": round(to_display(metric, pt, base["cv_score"]), 4) if base else None,
        "leaderboard": rows,
        "models_tried": tried,
        "models_untried": sorted(set(REGISTRY) - set(tried)),
        "feature_plan": state.get("feature_plan"),
        "rounds_used": max((e["round"] for e in lb), default=0),
        "tuned_already": any(e.get("mode") == "tune" for e in lb),
    }


def lift(state: RunState) -> float:
    lb = state.get("leaderboard", [])
    best, base = best_entry(lb), baseline_entry(lb)
    if not best or not base:
        return float("inf")
    scale = abs(base["cv_score"]) or 1.0
    return (best["cv_score"] - base["cv_score"]) / scale


def policy_verdict(state: RunState, settings: Settings) -> CriticVerdict:
    s = summarize(state)
    # The policy never picks revise_features: without a model to reason about the
    # data, a second rule-based feature plan would just repeat the first one.
    weak = lift(state) < settings.min_improvement_over_baseline
    if weak and s["models_untried"]:
        return CriticVerdict(decision="try_other_models", reasoning="Still no meaningful lift; try other model families.")
    if not s["tuned_already"]:
        return CriticVerdict(decision="tune", reasoning="Models beat the baseline; a short hyperparameter search is the cheapest remaining gain.")
    return CriticVerdict(decision="accept", reasoning="Tuned models beat the baseline; further rounds unlikely to pay off.")


def validate_verdict(v: CriticVerdict, state: RunState) -> CriticVerdict:
    s = summarize(state)
    if v.decision == "try_other_models" and not s["models_untried"]:
        raise ValidationFailed("no untried models left")
    if v.decision == "revise_features" and state.get("feature_revisions", 0) >= 2:
        raise ValidationFailed("features already revised twice")
    if v.decision == "tune" and not any(e["model"] in REGISTRY for e in state.get("leaderboard", []) if "error" not in e):
        raise ValidationFailed("nothing to tune")
    return v


class CriticAgent(Agent):
    name = "critic"

    def run(self, state: RunState, ctx: AgentContext) -> dict[str, Any]:
        s = ctx.settings
        rounds = max((e["round"] for e in state.get("leaderboard", [])), default=0)
        common = {"needs_critique": False}

        if rounds > s.max_improvement_rounds:
            v, calls = CriticVerdict(decision="accept", reasoning=f"improvement budget ({s.max_improvement_rounds} rounds) used"), []
        else:
            d = ctx.gateway(state).decide(
                agent=self.name,
                schema=CriticVerdict,
                system=SYSTEM,
                user=f"Run summary:\n{as_data(summarize(state))}\nRounds remaining after this: {s.max_improvement_rounds - rounds}",
                policy=lambda: policy_verdict(state, s),
                validate=lambda v: validate_verdict(v, state),
            )
            v, calls = d.value, [d.record]

        history = [*state.get("critiques", []), {"round": rounds, "feature_plan": state.get("feature_plan"), **v.model_dump()}]
        msg = self.say(f"round {rounds}: {v.decision}: {v.reasoning}")
        update: dict[str, Any] = {**common, "critiques": history, "llm_calls": calls, "messages": [msg]}

        if v.decision == "accept":
            update["accepted"] = True
        elif v.decision == "tune":
            ranked = sorted(
                (e for e in state["leaderboard"] if "error" not in e and e["model"] in REGISTRY),
                key=lambda e: e["cv_score"],
                reverse=True,
            )
            update.update(train_mode="tune", tune_models=list(dict.fromkeys(e["model"] for e in ranked))[:2], needs_training=True)
        elif v.decision == "revise_features":
            update.update(
                feature_plan=None,
                feature_feedback=v.feature_feedback or v.reasoning,
                feature_revisions=state.get("feature_revisions", 0) + 1,
                train_mode="fit",
                needs_training=True,
            )
        else:  # try_other_models: back through selection and (human) review
            update.update(candidate_models=[], approved_models=[], train_mode="fit", needs_training=False)
        return update
