"""Produces a validated, declarative FeaturePlan (see tools/features.py).

The LLM reasons over the profile, the analyst's findings and (on a revision
round) the critic's feedback, but can only emit a FeaturePlan. The plan is
validated against real columns and executed by deterministic code, so a bad
or injected answer can at worst be ignored, never executed.
"""

from __future__ import annotations

from typing import Any

from novaml.agents.base import Agent, AgentContext
from novaml.agents.model_selector import compact_profile
from novaml.llm.gateway import as_data
from novaml.state import RunState
from novaml.tools.features import FeaturePlan, default_plan, validate_plan

SYSTEM = (
    "You are a feature engineer for a tabular AutoML system. Propose a FeaturePlan: "
    "columns to drop (identifiers, leakage, constants, near-empty), non-negative skewed "
    "numeric columns to log-transform, date-like columns to expand, and at most a few "
    "ratio features that have a plausible domain meaning. Missing values and categorical "
    "encoding are handled automatically downstream; do not plan for them. Only reference "
    "column names that exist in the profile. Ratio operands must be existing numeric "
    "columns (derived expressions such as sums are not supported). Always drop columns "
    "the analyst flags as leakage. Only drop identifiers, leakage, constant or near-empty "
    "columns: never drop a feature for weak correlation or redundancy (it may matter in "
    "interactions, and the models handle it); such drops are rejected."
)


def policy_plan(profile: dict[str, Any], analysis: dict[str, Any] | None) -> FeaturePlan:
    plan = default_plan(profile)
    leaks = [c for c in (analysis or {}).get("leakage_suspects", []) if c not in plan.drop_columns]
    if leaks:
        plan.drop_columns += leaks
        plan.rationale += f"; drop suspected leakage {leaks}"
    return plan


def droppable(profile: dict[str, Any], analysis: dict[str, Any] | None, plan: dict[str, Any] | None) -> set[str]:
    """Columns there is evidence to drop: structural junk, verified or domain-suspected leaks."""
    cols = profile["columns"]
    junk = {c for c, i in cols.items() if i.get("id_like") or i.get("constant") or i.get("missing_pct", 0) > 95}
    leaks = set((analysis or {}).get("leakage_suspects", [])) | set((plan or {}).get("suspected_leakage", []))
    return junk | (leaks & set(cols))


class FeatureEngineerAgent(Agent):
    name = "feature_engineer"

    def run(self, state: RunState, ctx: AgentContext) -> dict[str, Any]:
        profile, target = state["profile"], state["target"]
        analysis = state.get("analysis")
        feedback = state.get("feature_feedback")
        previous = [c.get("feature_plan") for c in state.get("critiques", []) if c.get("feature_plan")]

        user = f"Target: {target!r} ({state['problem_type']})\nData profile:\n{as_data(compact_profile(profile))}\n"
        if analysis:
            user += f"Analyst findings:\n{as_data(analysis)}\n"
        if feedback:
            user += f"This is a revision. Reviewer feedback on the previous plan: {feedback}\n"
            if previous:
                user += f"Previous plan:\n{as_data(previous[-1])}\n"

        d = ctx.gateway(state).decide(
            agent=self.name,
            schema=FeaturePlan,
            system=SYSTEM,
            user=user,
            policy=lambda: policy_plan(profile, analysis),
        )
        plan, warnings = validate_plan(d.value, profile, target)
        # Guardrail: drops need evidence. Evals showed the LLM discarding real signal
        # (e.g. both halves of an f3*f4 interaction, "redundant" correlated features).
        # Models cope with weak or correlated columns; discarded data is gone for good.
        if d.used_llm:
            allowed = droppable(profile, analysis, state.get("plan"))
            rejected = [c for c in plan.drop_columns if c not in allowed]
            if rejected:
                plan.drop_columns = [c for c in plan.drop_columns if c in allowed]
                warnings.append(f"kept {rejected}: no evidence they are identifiers, leakage, constant or empty")
        # Guardrail: columns the analyst verified as leaking are dropped no matter what
        # the plan says. Leakage silently inflates every downstream metric.
        forced = [c for c in (analysis or {}).get("leakage_suspects", []) if c in profile["columns"] and c not in plan.drop_columns]
        if forced and len(plan.drop_columns) + len(forced) < len(profile["columns"]):
            plan.drop_columns += forced
            warnings.append(f"dropped suspected target leakage: {forced}")
        return {
            "feature_plan": plan.model_dump(),
            "feature_warnings": warnings,
            "llm_calls": [d.record],
            "messages": [self.say(f"feature plan ({d.source}): {plan.rationale or 'no transformations'}")]
            + [self.say(f"warning: {w}") for w in warnings],
        }
