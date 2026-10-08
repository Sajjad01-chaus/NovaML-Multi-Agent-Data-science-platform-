"""End-to-end runs through the real graph (deterministic mode, no API keys)."""

import json

import pytest
from fastapi.testclient import TestClient

from novaml.llm.scripted import ScriptedProvider
from novaml.service import InvalidResume, NovaML, RunNotFound
from novaml.serving.app import create_app


def test_classification_end_to_end(svc, iris_csv):
    r = svc.start(iris_csv, "species")
    assert r.status == "completed", r.errors
    s = r.state
    assert s["problem_type"] == "classification"
    assert s["best"]["model"] != "baseline"
    assert s["holdout"]["model"]["f1_macro"] > 0.8
    assert s["holdout"]["model"]["f1_macro"] > s["holdout"]["baseline"]["f1_macro"]
    json.dumps(s)  # state must stay serialisable (no frames/models inside)

    client = TestClient(create_app(s["bundle_dir"]))
    row = {"sepal length (cm)": 5.1, "sepal width (cm)": 3.5, "petal length (cm)": 1.4, "petal width (cm)": 0.2}
    assert client.post("/predict", json={"rows": [row]}).json()["predictions"] == ["setosa"]


def test_regression_end_to_end(svc, diabetes_csv):
    r = svc.start(diabetes_csv, "progression")
    assert r.status == "completed", r.errors
    assert r.state["problem_type"] == "regression"
    assert r.state["metric"] == "r2"
    assert r.state["holdout"]["model"]["r2"] > 0.2


def test_messy_data_end_to_end(svc, messy_csv):
    r = svc.start(messy_csv, "churn")
    assert r.status == "completed", r.errors
    s = r.state
    assert any("missing target" in m for m in s["messages"])
    assert "customer_id" in s["feature_plan"]["drop_columns"]
    card = s["model_card"]
    # No provider configured: every decision point ran its policy.
    assert card["llm"]["calls"] >= 4 and card["llm"]["llm_decisions"] == 0
    assert card["llm"]["policy_fallbacks"] == card["llm"]["calls"]


def test_bad_target_fails_cleanly(svc, iris_csv):
    r = svc.start(iris_csv, "does_not_exist")
    assert r.status == "failed"
    assert "not found" in r.errors[0]


def test_human_in_the_loop_survives_restart(settings, iris_csv):
    settings = settings.model_copy(update={"auto_approve": False})
    a = NovaML(settings, provider=None)
    r = a.start(iris_csv, "species")
    assert r.status == "awaiting_approval"
    assert r.pending["candidates"]
    a.close()

    # A different process/worker picks the run up from the on-disk checkpoint.
    b = NovaML(settings, provider=None)
    with pytest.raises(InvalidResume):
        b.resume(r.run_id, ["not_a_model"])
    chosen = r.pending["candidates"][:1]
    done = b.resume(r.run_id, chosen)
    assert done.status == "completed", done.errors
    trained = {e["model"] for e in done.state["leaderboard"]}
    assert trained == {"baseline", *chosen}
    with pytest.raises(InvalidResume):
        b.resume(r.run_id, chosen)
    with pytest.raises(RunNotFound):
        b.get("nope")
    b.close()


def test_llm_selection_is_validated(settings, iris_csv):
    provider = ScriptedProvider({"ModelSelection": [{"models": ["xgboost_ultra", "knn"], "reasoning": "r"}]})
    s = NovaML(settings, provider=provider)
    r = s.start(iris_csv, "species")
    s.close()
    assert r.status == "completed", r.errors
    assert r.state["candidate_models"] == ["knn"]
    call = next(c for c in r.state["llm_calls"] if c["agent"] == "model_selector")
    assert call["source"] == "llm"
