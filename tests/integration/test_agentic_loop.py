"""The Phase 1 agentic loop: planner -> analyst (sandboxed code) -> features ->
selection -> review -> train -> evaluate -> critic, with reflection rounds."""

import numpy as np
import pandas as pd
import pytest

from novaml.llm.base import LLMError
from novaml.llm.scripted import ScriptedProvider
from novaml.service import NovaML


def nodes(state):
    return [e["node"] for e in state["events"]]


@pytest.fixture
def leaky_csv(tmp_path):
    """A churn dataset with a column that is the target in disguise."""
    rng = np.random.default_rng(1)
    n = 300
    tenure = rng.integers(0, 60, n).astype(float)
    spend = rng.gamma(2, 50, n)
    churn = (0.05 * tenure - 0.01 * spend + rng.normal(0, 1, n) < 0.5).astype(int)
    df = pd.DataFrame(
        {
            "tenure": tenure,
            "monthly_spend": spend,
            "region": rng.choice(["n", "s", "e", "w"], n),
            "cancel_code": np.where(churn == 1, 7.0, 0.0) + rng.normal(0, 0.001, n),
            "churn": churn,
        }
    )
    p = tmp_path / "leaky.csv"
    df.to_csv(p, index=False)
    return p


def test_policy_loop_tunes_then_accepts(svc, iris_csv):
    r = svc.start(iris_csv, "species")
    assert r.status == "completed", r.errors
    s = r.state
    assert nodes(s)[:3] == ["profiler", "planner", "analyst"]
    modes = {(e["round"], e["mode"]) for e in s["leaderboard"]}
    assert (1, "fit") in modes and (2, "tune") in modes
    assert [c["decision"] for c in s["critiques"]] == ["tune", "accept"]
    assert s["analysis"]["source"] == "policy" and s["analysis"]["insights"]


def test_policy_detects_leakage_and_drops_it(svc, leaky_csv):
    r = svc.start(leaky_csv, "churn")
    assert r.status == "completed", r.errors
    assert "cancel_code" in r.state["analysis"]["leakage_suspects"]
    assert "cancel_code" in r.state["feature_plan"]["drop_columns"]


def test_llm_driven_run_with_reflection(settings, leaky_csv):
    """Scripted LLM exercises every decision point, including two reflection rounds."""
    p = ScriptedProvider(
        {
            "RunPlan": [{"primary_metric": "balanced_accuracy", "run_analysis": True, "analysis_questions": ["which features leak?"], "risks": ["leakage"]}],
            "AnalystStep": [
                {"thought": "check association", "action": "run_code", "code": "print(df.corr(numeric_only=True)['churn'].round(3).to_dict())"},
                {"thought": "try to escape", "action": "run_code", "code": "import os\nprint(os.environ)"},
                {"thought": "done", "action": "finish", "insights": ["cancel_code mirrors churn"], "leakage_suspects": ["cancel_code", "ghost"]},
            ],
            "FeaturePlan": [
                {"drop_columns": ["cancel_code", "nonexistent"], "rationale": "drop leak"},
                {"drop_columns": ["cancel_code"], "ratio_features": [{"numerator": "monthly_spend", "denominator": "tenure"}], "rationale": "add spend per month of tenure"},
            ],
            "ModelSelection": [
                {"models": ["linear", "random_forest"], "reasoning": "contrast"},
                {"models": ["hist_gradient_boosting", "linear"], "reasoning": "try boosting"},
            ],
            "CriticVerdict": [
                {"decision": "revise_features", "reasoning": "weak", "feature_feedback": "add a spend/tenure ratio"},
                {"decision": "try_other_models", "reasoning": "still weak"},
            ],
        }
    )
    svc = NovaML(settings, provider=p)
    r = svc.start(leaky_csv, "churn")
    svc.close()
    assert r.status == "completed", r.errors
    s = r.state

    assert s["metric"] == "balanced_accuracy"
    # Analyst: one real sandboxed step, one blocked escape attempt, then finish.
    assert s["analysis"]["source"] == "llm" and s["analysis"]["code_steps"] == 2
    assert s["analysis"]["leakage_suspects"] == ["cancel_code"]  # 'ghost' filtered out
    # Revision round used the critic's feedback.
    assert s["feature_plan"]["ratio_features"] == [{"numerator": "monthly_spend", "denominator": "tenure"}]
    assert "cancel_code" in s["feature_plan"]["drop_columns"]
    # try_other_models excluded families already tried.
    assert s["candidate_models"] == ["hist_gradient_boosting"]
    # Round budget (2) exhausted -> accepted without another LLM call.
    assert [c["decision"] for c in s["critiques"]] == ["revise_features", "try_other_models", "accept"]
    assert nodes(s).count("trainer") == 3 and nodes(s)[-1] == "deployer"

    calls = s["llm_calls"]
    assert all(c["source"] == "llm" for c in calls)
    assert sum(c["agent"] == "critic" for c in calls) == 2
    assert s["model_card"]["llm"]["tokens"] == 100 * len(calls)

    import json
    from pathlib import Path

    steps = json.loads((Path(settings.runs_dir) / r.run_id / "analysis.json").read_text())["steps"]
    assert steps[0]["ok"] and "cancel_code" in steps[0]["output"]
    assert not steps[1]["ok"] and "PolicyViolation" in steps[1]["error"]


def test_second_human_review_when_critic_requests_new_models(settings, iris_csv):
    settings = settings.model_copy(update={"auto_approve": False})
    p = ScriptedProvider(
        {
            "ModelSelection": [{"models": ["linear"], "reasoning": "a"}, {"models": ["knn"], "reasoning": "b"}],
            "CriticVerdict": [{"decision": "try_other_models", "reasoning": "x"}, {"decision": "accept", "reasoning": "ok"}],
        }
    )
    svc = NovaML(settings, provider=p)
    r = svc.start(iris_csv, "species")
    assert r.status == "awaiting_approval" and r.pending["candidates"] == ["linear"]
    r = svc.resume(r.run_id, ["linear"])
    assert r.status == "awaiting_approval" and r.pending["candidates"] == ["knn"]
    r = svc.resume(r.run_id, ["knn"])
    svc.close()
    assert r.status == "completed", r.errors
    assert {e["model"] for e in r.state["leaderboard"]} == {"baseline", "linear", "knn"}


def test_budget_exhaustion_degrades_gracefully(settings, iris_csv):
    settings = settings.model_copy(update={"run_token_budget": 150})
    p = ScriptedProvider({"RunPlan": [{"primary_metric": "accuracy", "run_analysis": False}]}, tokens_per_call=200)
    svc = NovaML(settings, provider=p)
    r = svc.start(iris_csv, "species")
    svc.close()
    assert r.status == "completed", r.errors
    calls = r.state["llm_calls"]
    assert calls[0]["source"] == "llm"
    assert all(c["source"] == "policy" and "budget" in c["error"] for c in calls[1:])
    assert len(p.calls) == 1


def test_provider_outage_mid_run_still_completes(settings, iris_csv):
    p = ScriptedProvider({"RunPlan": [LLMError("503 overloaded")], "ModelSelection": [LLMError("timeout")]})
    svc = NovaML(settings, provider=p)
    r = svc.start(iris_csv, "species")
    svc.close()
    assert r.status == "completed", r.errors
    errs = [c["error"] for c in r.state["llm_calls"] if c["error"]]
    assert any("503" in e for e in errs) and any("timeout" in e for e in errs)
