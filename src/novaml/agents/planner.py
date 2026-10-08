"""Planner: reads the data profile and commits to a run plan before any work starts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from novaml.agents.base import Agent, AgentContext
from novaml.agents.model_selector import compact_profile
from novaml.llm.gateway import as_data
from novaml.state import RunState
from novaml.tools.ml import METRICS, resolve_metric


class RunPlan(BaseModel):
    primary_metric: str = Field(default="", description="Metric to optimise; one of the allowed metrics given.")
    run_analysis: bool = Field(default=True, description="Whether an exploratory analysis step is worth its cost for this data.")
    analysis_questions: list[str] = Field(default_factory=list, description="Up to 4 concrete questions for the analyst.")
    risks: list[str] = Field(default_factory=list, description="Data risks to watch: leakage, imbalance, drift, tiny n.")
    suspected_leakage: list[str] = Field(
        default_factory=list,
        description="Columns that, from their meaning, are probably recorded after or because of the outcome.",
    )
    rationale: str = ""


SYSTEM = (
    "You are the lead data scientist planning an automated modelling run. Given a data "
    "profile, choose the metric that best reflects success for this problem (e.g. a "
    "class-balanced metric when classes are imbalanced), decide whether exploratory "
    "analysis is worth running, list the specific questions it should answer, and flag risks. "
    "Use domain knowledge to name columns that are likely recorded after or because of the "
    "outcome (post-outcome leakage); the analyst will verify them against the data."
)


def policy_plan(profile: dict[str, Any], problem_type: str) -> RunPlan:
    risks, questions = [], []
    tgt = profile["target"]
    metric = None
    if problem_type == "classification":
        share = tgt.get("minority_share", 0.5)
        if share < 0.2:
            risks.append(f"class imbalance: minority class is {share:.1%} of rows")
            metric = "balanced_accuracy" if tgt["n_unique"] > 2 else "roc_auc"
        questions.append("Which features separate the target classes most strongly?")
    else:
        if abs(tgt.get("skew") or 0) > 2:
            risks.append("target is heavily skewed; errors will be dominated by the tail")
        questions.append("Which features are most correlated with the target?")
    if profile["n_rows"] < 300:
        risks.append(f"small dataset ({profile['n_rows']} training rows): high variance estimates")
    if profile["total_missing"]:
        questions.append("Is missingness itself predictive of the target?")
    questions.append("Are any features suspiciously close to the target (possible leakage)?")
    return RunPlan(
        primary_metric=metric or resolve_metric(problem_type, None),
        run_analysis=True,
        analysis_questions=questions,
        risks=risks,
        rationale="Policy plan from profile heuristics.",
    )


class PlannerAgent(Agent):
    name = "planner"

    def run(self, state: RunState, ctx: AgentContext) -> dict[str, Any]:
        profile, pt = state["profile"], state["problem_type"]
        n_classes = profile["target"]["n_unique"] if pt == "classification" else None

        def validate(p: RunPlan) -> RunPlan:
            p.primary_metric = resolve_metric(pt, p.primary_metric, n_classes)
            p.analysis_questions = p.analysis_questions[:4]
            p.suspected_leakage = [c for c in dict.fromkeys(p.suspected_leakage) if c in profile["columns"]]
            return p

        d = ctx.gateway(state).decide(
            agent=self.name,
            schema=RunPlan,
            system=SYSTEM,
            user=(
                f"Problem type: {pt}\nAllowed metrics: {list(METRICS[pt])}\n"
                f"Data profile:\n{as_data(compact_profile(profile))}"
            ),
            policy=lambda: validate(policy_plan(profile, pt)),
            validate=validate,
        )
        plan = d.value
        msgs = [self.say(f"plan ({d.source}): optimise {plan.primary_metric}; analysis={'on' if plan.run_analysis else 'off'}")]
        msgs += [self.say(f"risk: {r}") for r in plan.risks]
        if plan.suspected_leakage:
            msgs.append(self.say(f"suspected leakage to verify: {plan.suspected_leakage}"))
        return {
            "plan": plan.model_dump(),
            "metric": plan.primary_metric,
            "llm_calls": [d.record],
            "messages": msgs,
        }
