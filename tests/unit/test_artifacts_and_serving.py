import zipfile

import boto3
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from moto import mock_aws

from novaml.artifacts import LocalArtifactStore, S3ArtifactStore, materialize_bundle
from novaml.serving.app import create_app
from novaml.serving.bundle import write_bundle
from novaml.tools.ml import build_pipeline


@pytest.fixture(params=["local", "s3"])
def store(request, tmp_path, monkeypatch):
    if request.param == "local":
        yield LocalArtifactStore(tmp_path, "run1")
        return
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="novaml-test")
        yield S3ArtifactStore("novaml-test", "run1", client=client)


def test_store_roundtrips_every_type(store):
    df = pd.DataFrame({"a": [1, 2], "b": ["x", None]})
    store.save_frame("data/train.parquet", df)
    pd.testing.assert_frame_equal(store.load_frame("data/train.parquet"), df)
    store.save_json("card.json", {"k": [1, 2]})
    assert store.load_json("card.json") == {"k": [1, 2]}
    store.save_model("models/m.joblib", {"fitted": True})
    assert store.load_model("models/m.joblib") == {"fitted": True}
    assert store.exists("card.json") and not store.exists("nope.json")
    assert store.list("models/") == ["models/m.joblib"]


def test_directories_upload_download_and_zip(store, tmp_path):
    src = tmp_path / "bundle-src"
    (src / "sub").mkdir(parents=True)
    (src / "model.joblib").write_bytes(b"m")
    (src / "sub" / "schema.json").write_text("{}")
    store.put_dir("bundle", src)
    local = materialize_bundle(store)
    assert (local / "model.joblib").read_bytes() == b"m" and (local / "sub" / "schema.json").exists()
    import io

    names = zipfile.ZipFile(io.BytesIO(store.zip_dir("bundle"))).namelist()
    assert sorted(names) == ["model.joblib", "sub/schema.json"]


@pytest.mark.parametrize("bad", ["../x", "a/../../x", "/etc/passwd", r"C:\x", "a b"])
def test_keys_cannot_escape_namespace(store, bad):
    with pytest.raises(ValueError):
        store.put_bytes(bad, b"x")


def test_namespaces_are_validated(tmp_path):
    with pytest.raises(ValueError):
        LocalArtifactStore(tmp_path, "../evil")


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
