"""Selects the best model by cross-validation, then reports holdout performance.

Selection never looks at the test split: repeated selection on holdout scores is
test-set overfitting. The holdout is used only to *report* the chosen model.
"""

from __future__ import annotations

from typing import Any

from novaml.agents.base import Agent, AgentContext
from novaml.state import RunState
from novaml.tools.ml import BASELINE, holdout_metrics, to_display


def best_entry(leaderboard: list[dict[str, Any]]) -> dict[str, Any] | None:
    ok = [e for e in leaderboard if "error" not in e and e["model"] != BASELINE]
    return max(ok, key=lambda e: e["cv_score"], default=None)


def baseline_entry(leaderboard: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next((e for e in leaderboard if e["model"] == BASELINE and "error" not in e), None)


class EvaluatorAgent(Agent):
    name = "evaluator"

    def run(self, state: RunState, ctx: AgentContext) -> dict[str, Any]:
        store = ctx.store(state)
        target, pt, metric = state["target"], state["problem_type"], state["metric"]
        lb = state.get("leaderboard", [])
        best, base = best_entry(lb), baseline_entry(lb)
        if best is None:
            raise RuntimeError("no successfully trained model to evaluate")

        test = store.load_frame(state["data_refs"]["test"])
        X, y = test.drop(columns=[target]), test[target]
        holdout = {"model": holdout_metrics(store.load_model(best["artifact"]), X, y, pt)}
        if base:
            holdout["baseline"] = holdout_metrics(store.load_model(base["artifact"]), X, y, pt)

        cv_best = to_display(metric, pt, best["cv_score"])
        msg = f"best={best['model']} (round {best['round']}) cv {metric}={cv_best:.4f}"
        if metric in holdout["model"]:
            msg += f", holdout {metric}={holdout['model'][metric]:.4f}"
        return {
            "best": {k: best[k] for k in ("model", "round", "artifact", "cv_score", "cv_std", "params") if k in best},
            "holdout": holdout,
            "messages": [self.say(msg)],
        }
