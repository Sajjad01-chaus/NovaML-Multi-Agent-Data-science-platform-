"""Cross-validates and fits each approved model (plus a baseline) as a full pipeline.

Two modes, chosen by the critic:
* fit:  default hyperparameters for every approved model
* tune: randomised search over the registry's space for the top models
"""

from __future__ import annotations

from typing import Any

from novaml.agents.base import Agent, AgentContext
from novaml.state import RunState
from novaml.tools.ml import (
    BASELINE,
    build_pipeline,
    cross_validate_pipeline,
    to_display,
    tune_pipeline,
)

IMBALANCE_SHARE = 0.2


class TrainerAgent(Agent):
    name = "trainer"

    def run(self, state: RunState, ctx: AgentContext) -> dict[str, Any]:
        s = ctx.settings
        store = ctx.store(state)
        target, pt, metric = state["target"], state["problem_type"], state["metric"]
        train = store.load_frame(state["data_refs"]["train"])
        X, y = train.drop(columns=[target]), train[target]
        rnd = _round(state)
        mode = state.get("train_mode", "fit")
        # Imbalanced targets: reweight classes so models don't just predict the majority.
        share = state["profile"]["target"].get("minority_share")
        class_weight = "balanced" if pt == "classification" and share is not None and share < IMBALANCE_SHARE else None

        if mode == "tune":
            todo = list(state.get("tune_models") or [])
        else:
            todo = list(state["approved_models"])
            if not any(e["model"] == BASELINE for e in state.get("leaderboard", [])):
                todo.insert(0, BASELINE)

        entries, msgs = [], []
        for name in todo:
            pipe = build_pipeline(name, pt, state.get("feature_plan"), s.random_state, class_weight)
            cv = dict(problem_type=pt, metric=metric, folds=s.cv_folds, seed=s.random_state, n_jobs=s.n_jobs)
            try:
                if mode == "tune":
                    pipe, params, stats = tune_pipeline(pipe, name, X, y, n_iter=s.tuning_iterations, **cv)
                else:
                    params, stats = {}, cross_validate_pipeline(pipe, X, y, **cv)
                    pipe.fit(X, y)
            except Exception as e:  # one bad model must not sink the run
                entries.append({"round": rnd, "model": name, "mode": mode, "error": f"{type(e).__name__}: {e}"[:300]})
                msgs.append(self.say(f"{name} failed: {type(e).__name__}"))
                continue
            key = store.save_model(f"models/r{rnd}_{name}.joblib", pipe)
            entries.append(
                {
                    "round": rnd,
                    "model": name,
                    "mode": mode,
                    "params": params | ({"class_weight": class_weight} if class_weight and name != BASELINE else {}),
                    "artifact": key,
                    **stats,
                    "overfit_gap": stats["train_score"] - stats["cv_score"],
                }
            )
            msgs.append(
                self.say(
                    f"[r{rnd} {mode}] {name}: cv {metric}={to_display(metric, pt, stats['cv_score']):.4f} ± {stats['cv_std']:.4f}"
                )
            )

        prior_ok = any("error" not in e and e["model"] != BASELINE for e in state.get("leaderboard", []))
        if not prior_ok and not any("error" not in e and e["model"] != BASELINE for e in entries):
            raise RuntimeError("every candidate model failed to train")
        return {
            "leaderboard": entries,
            "train_mode": "fit",
            "needs_training": False,
            "needs_evaluation": True,
            "messages": msgs,
        }


def _round(state: RunState) -> int:
    return 1 + max((e["round"] for e in state.get("leaderboard", [])), default=0)
