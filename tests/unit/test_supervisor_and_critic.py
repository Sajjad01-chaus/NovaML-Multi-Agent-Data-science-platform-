import pytest
from langgraph.graph import END

from novaml.agents.critic import CriticVerdict, policy_verdict, validate_verdict
from novaml.config import Settings
from novaml.llm.gateway import ValidationFailed
from novaml.supervisor import next_step
from novaml.tools.ml import REGISTRY

S = Settings(_env_file=None, max_graph_steps=30)


def entry(model, cv, rnd=1, mode="fit"):
    return {"round": rnd, "model": model, "mode": mode, "cv_score": cv, "cv_std": 0.01, "overfit_gap": 0.0, "artifact": "x"}


def ready(**over):
    base = {
        "profile": {"n_rows": 100},
        "plan": {"run_analysis": True},
        "analysis": {},
        "feature_plan": {"drop_columns": []},
        "candidate_models": ["linear"],
        "approved_models": ["linear"],
    }
    return base | over


def test_routing_follows_missing_inputs():
    assert next_step({}, S) == "profiler"
    assert next_step({"profile": {"n_rows": 1}}, S) == "planner"
    assert next_step(ready(analysis=None), S) == "analyst"
    assert next_step(ready(plan={"run_analysis": False}, analysis=None, feature_plan=None), S) == "feature_engineer"
    assert next_step(ready(candidate_models=[]), S) == "model_selector"
    assert next_step(ready(approved_models=[]), S) == "human_review"
    assert next_step(ready(needs_training=True), S) == "trainer"
    assert next_step(ready(needs_evaluation=True), S) == "evaluator"
    assert next_step(ready(needs_critique=True), S) == "critic"
    assert next_step(ready(accepted=True, best={"model": "linear"}), S) == "deployer"


def test_terminal_and_budget():
    assert next_step({"status": "failed"}, S) == END
    assert next_step(ready(status="completed"), S) == END
    events = [{}] * 30
    assert next_step(ready(events=events, needs_training=True), S) == END
    assert next_step(ready(events=events, needs_training=True, best={"model": "linear"}), S) == "deployer"


def critic_state(board, **over):
    return {"problem_type": "classification", "metric": "f1_macro", "leaderboard": board} | over


def test_policy_tunes_then_accepts():
    board = [entry("baseline", 0.3), entry("linear", 0.9)]
    assert policy_verdict(critic_state(board), S).decision == "tune"
    board.append(entry("linear", 0.92, rnd=2, mode="tune"))
    assert policy_verdict(critic_state(board), S).decision == "accept"


def test_policy_tries_other_models_when_no_lift():
    board = [entry("baseline", 0.5), entry("linear", 0.505)]
    assert policy_verdict(critic_state(board), S).decision == "try_other_models"


def test_validate_rejects_impossible_verdicts():
    board = [entry("baseline", 0.5)] + [entry(m, 0.6) for m in REGISTRY]
    st = critic_state(board)
    with pytest.raises(ValidationFailed):
        validate_verdict(CriticVerdict(decision="try_other_models", reasoning=""), st)
    with pytest.raises(ValidationFailed):
        validate_verdict(CriticVerdict(decision="revise_features", reasoning=""), st | {"feature_revisions": 2})
