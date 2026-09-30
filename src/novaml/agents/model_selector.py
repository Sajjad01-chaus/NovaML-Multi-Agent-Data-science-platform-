"""Chooses candidate model families from the registry (LLM with a policy fallback)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from novaml.agents.base import Agent, AgentContext
from novaml.llm.gateway import ValidationFailed, as_data
from novaml.state import RunState
from novaml.tools.ml import REGISTRY, available_models

MAX_CANDIDATES = 4


class ModelSelection(BaseModel):
    models: list[str] = Field(description="2-4 model names, chosen only from the provided registry.")
    reasoning: str = Field(description="Why these models suit this dataset, in 2-4 sentences.")


SYSTEM = (
    "You are a principal data scientist choosing which model families to train on a "
    "tabular dataset. Choose 2-4 models ONLY from the registry you are given, using "
    "the data profile (size, feature types, missingness, class balance, skew). Prefer "
    "diversity (e.g. one linear, one bagging, one boosting) unless the data argues otherwise."
)


def policy_selection(profile: dict[str, Any], exclude: list[str] | None = None) -> ModelSelection:
    n = profile["n_rows"]
    if n < 1_000:
        order = ["linear", "random_forest", "gradient_boosting", "extra_trees", "knn"]
        why = "small dataset: a regularised linear model, bagging and classic boosting"
    else:
        order = ["hist_gradient_boosting", "random_forest", "linear", "extra_trees"]
        why = "medium/large dataset: histogram boosting scales best, with RF and linear for contrast"
    picks = [m for m in order if m not in (exclude or [])][:3]
    return ModelSelection(models=picks or order[:3], reasoning=f"Policy: {why}.")


def validate_selection(sel: ModelSelection, exclude: list[str] | None = None) -> ModelSelection:
    models = [m for m in dict.fromkeys(sel.models) if m in REGISTRY and m not in (exclude or [])]
    if not models:
        raise ValidationFailed(f"no known models in {sel.models}")
    return ModelSelection(models=models[:MAX_CANDIDATES], reasoning=sel.reasoning)


class ModelSelectorAgent(Agent):
    name = "model_selector"

    def run(self, state: RunState, ctx: AgentContext) -> dict[str, Any]:
        profile = state["profile"]
        exclude = self.exclusions(state)
        user = (
            f"Problem type: {state['problem_type']}\nOptimisation metric: {state['metric']}\n"
            f"Model registry:\n{as_data(available_models())}\n"
            f"Data profile:\n{as_data(compact_profile(profile))}\n"
            + (f"Already tried (do not repeat): {exclude}\n" if exclude else "")
        )
        d = ctx.gateway(state).decide(
            agent=self.name,
            schema=ModelSelection,
            system=SYSTEM,
            user=user,
            policy=lambda: policy_selection(profile, exclude),
            validate=lambda s: validate_selection(s, exclude),
        )
        return {
            "candidate_models": d.value.models,
            "selection_reasoning": d.value.reasoning,
            "llm_calls": [d.record],
            "messages": [self.say(f"candidates ({d.source}): {', '.join(d.value.models)}")],
        }

    def exclusions(self, state: RunState) -> list[str]:
        """Model families already trained in earlier rounds (critic asked for new ones)."""
        return sorted({e["model"] for e in state.get("leaderboard", [])} & set(REGISTRY))


def compact_profile(profile: dict[str, Any], max_cols: int = 60) -> dict[str, Any]:
    """Trim the profile so prompt size stays bounded on wide datasets."""
    cols = dict(list(profile["columns"].items())[:max_cols])
    out = {k: v for k, v in profile.items() if k != "columns"}
    out["columns"] = cols
    if len(profile["columns"]) > max_cols:
        out["columns_truncated"] = len(profile["columns"]) - max_cols
    return out
