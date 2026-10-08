import pandas as pd
import pytest
from fastapi.testclient import TestClient

from novaml.artifacts import ArtifactStore
from novaml.serving.app import create_app
from novaml.serving.bundle import write_bundle
from novaml.tools.ml import build_pipeline


def test_artifact_keys_cannot_escape_run_dir(tmp_path):
    s = ArtifactStore(tmp_path, "run1")
    for bad in ["../x", "a/../../x", "/etc/passwd", "C:\\x", "a b"]:
        with pytest.raises(ValueError):
            s.path(bad)
    with pytest.raises(ValueError):
        ArtifactStore(tmp_path, "../evil")


def test_bundle_with_hostile_column_names_serves_safely(tmp_path):
    """Column names are data, never code (the old generator templated them into Python)."""
    evil = "x: int\nimport os; os.system('echo pwned')  #"
    df = pd.DataFrame({evil: range(60), "normal col": ["a", "b", "c"] * 20, "y": [0, 1] * 30})
    X, y = df.drop(columns=["y"]), df["y"]
    pipe = build_pipeline("linear", "classification", None, 0).fit(X, y)
    write_bundle(tmp_path / "b", pipe, X, {"model_name": "linear", "problem_type": "classification", "run_id": "r"})

    assert "import os" not in (tmp_path / "b" / "serve.py").read_text()
    client = TestClient(create_app(tmp_path / "b"))
    assert client.get("/health").json() == {"status": "ok"}

    ok = client.post("/predict", json={"rows": [{evil: 3, "normal col": "b"}, {evil: None, "normal col": "zzz"}]})
    assert ok.status_code == 200, ok.text
    body = ok.json()
    assert len(body["predictions"]) == 2 and len(body["probabilities"]) == 2

    bad = client.post("/predict", json={"rows": [{"unknown": 1}]})
    assert bad.status_code == 422
    assert client.post("/predict", json={"rows": []}).status_code == 422
