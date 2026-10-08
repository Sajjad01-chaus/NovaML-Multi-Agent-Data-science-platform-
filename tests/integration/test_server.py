"""API + queue + worker, end to end (SQLite + local artifacts; CI also runs Postgres/MinIO)."""

import io
import os
import secrets
import threading
import zipfile
from datetime import timedelta

import boto3
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import update

from novaml.agents.trainer import TrainerAgent
from novaml.server import db as dbm
from novaml.server.api import create_api
from novaml.server.worker import Worker
from novaml.serving.app import create_app


@pytest.fixture(params=["sqlite+local", "postgres+s3"])
def backend(request, settings):
    """Every server test runs on SQLite + local files, and also on real Postgres + S3
    (MinIO) when NOVAML_TEST_DATABASE_URL / NOVAML_TEST_S3_ENDPOINT are set (CI does)."""
    if request.param == "sqlite+local":
        return settings
    db_url, s3 = os.getenv("NOVAML_TEST_DATABASE_URL"), os.getenv("NOVAML_TEST_S3_ENDPOINT")
    if not (db_url and s3):
        pytest.skip("set NOVAML_TEST_DATABASE_URL and NOVAML_TEST_S3_ENDPOINT to test on Postgres + S3")
    bucket = f"novaml-test-{secrets.token_hex(4)}"
    boto3.client("s3", endpoint_url=s3, region_name="us-east-1").create_bucket(Bucket=bucket)
    database = dbm.Database(db_url.replace("postgresql://", "postgresql+psycopg://", 1))
    with database.engine.begin() as c:  # isolate tests sharing one Postgres
        c.exec_driver_sql("TRUNCATE run_events, jobs, runs")
    database.dispose()
    return settings.model_copy(
        update={"database_url": db_url, "artifact_backend": "s3", "s3_bucket": bucket, "s3_endpoint_url": s3, "s3_region": "us-east-1"}
    )


@pytest.fixture
def stack(backend):
    settings = backend.model_copy(update={"auto_approve": False, "worker_heartbeat_s": 0.2})
    database = dbm.Database(settings.sqlalchemy_url())
    client = TestClient(create_api(settings, database))
    worker = Worker(settings, database, provider=None, owner="w1")
    yield client, worker, database, settings
    worker.close()
    client.close()
    database.dispose()


def upload(client, path):
    with open(path, "rb") as f:
        r = client.post("/v1/datasets", files={"file": (path.name, f, "text/csv")})
    assert r.status_code == 201, r.text
    return r.json()


# --------------------------------------------------------------------------- queue
def test_run_and_job_are_created_atomically_and_idempotently(stack):
    _, _, database, _ = stack
    run, created = database.create_run("a" * 16, "y", None, idempotency_key="k1")
    again, created2 = database.create_run("a" * 16, "y", None, idempotency_key="k1")
    assert created and not created2 and again["id"] == run["id"]
    assert [j["kind"] for j in database.jobs_for(run["id"])] == ["start"]


def test_claim_lease_heartbeat_complete(stack):
    _, _, database, _ = stack
    run, _ = database.create_run("a" * 16, "y", None)
    job = database.claim("w1", lease_s=60)
    assert job["run_id"] == run["id"] and job["attempts"] == 1 and job["status"] == "running"
    assert database.claim("w2", lease_s=60) is None  # nothing else queued
    assert database.heartbeat(job["id"], "w1", 60)
    assert not database.heartbeat(job["id"], "intruder", 60)
    database.complete(job["id"], "w1")
    assert database.get_job(job["id"])["status"] == "done"


def test_expired_leases_are_requeued_then_dead(stack):
    _, _, database, _ = stack
    run, _ = database.create_run("a" * 16, "y", None, max_attempts=2)
    for attempt in (1, 2):
        job = database.claim(f"w{attempt}", lease_s=-1)  # lease already expired: worker "died"
        assert job["attempts"] == attempt
    database.requeue_expired()
    assert database.get_job(job["id"])["status"] == "dead"
    assert database.get_run(run["id"])["status"] == "failed"


def test_concurrent_workers_never_claim_the_same_job(stack):
    _, _, database, _ = stack
    for _ in range(20):
        database.create_run("a" * 16, "y", None)
    claimed, lock = [], threading.Lock()

    def worker(name):
        while (job := database.claim(name, 60)) is not None:
            with lock:
                claimed.append(job["id"])

    threads = [threading.Thread(target=worker, args=(f"w{i}",)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(claimed) == 20 and len(set(claimed)) == 20


# --------------------------------------------------------------------------- API flow
def test_full_flow_upload_run_approve_complete_download(stack, iris_csv, tmp_path):
    client, worker, _, _ = stack
    ds = upload(client, iris_csv)
    assert "species" in ds["columns"] and len(ds["dataset_id"]) == 16

    assert client.post("/v1/runs", json={"dataset_id": ds["dataset_id"], "target": "nope"}).status_code == 422
    body = {"dataset_id": ds["dataset_id"], "target": "species"}
    r = client.post("/v1/runs", json=body, headers={"Idempotency-Key": "abc"})
    assert r.status_code == 202 and r.json()["status"] == "queued"
    run_id = r.json()["id"]
    replay = client.post("/v1/runs", json=body, headers={"Idempotency-Key": "abc"})
    assert replay.status_code == 200 and replay.json()["id"] == run_id

    assert worker.drain() == 1
    run = client.get(f"/v1/runs/{run_id}").json()
    assert run["status"] == "awaiting_approval"
    candidates = run["detail"]["pending"]["candidates"]
    assert run["summary"]["pending"]["candidates"] == candidates
    assert client.get(f"/v1/runs/{run_id}/model").status_code == 409

    assert client.post(f"/v1/runs/{run_id}/approve", json={"models": ["xgboost_9000"]}).status_code == 422
    ok = client.post(f"/v1/runs/{run_id}/approve", json={"models": candidates[:1]})
    assert ok.status_code == 202 and ok.json()["status"] == "queued"
    assert client.post(f"/v1/runs/{run_id}/approve", json={"models": candidates[:1]}).status_code == 409

    assert worker.drain() == 1
    run = client.get(f"/v1/runs/{run_id}").json()
    assert run["status"] == "completed", run
    assert run["summary"]["best_model"] == candidates[0] and run["summary"]["cv_score"] > 0.8
    assert client.get(f"/v1/runs/{run_id}/model-card").json()["model_name"] == candidates[0]

    # The downloaded bundle is servable as-is.
    z = zipfile.ZipFile(io.BytesIO(client.get(f"/v1/runs/{run_id}/model").content))
    z.extractall(tmp_path / "bundle")
    served = TestClient(create_app(tmp_path / "bundle"))
    row = {"sepal length (cm)": 5.1, "sepal width (cm)": 3.5, "petal length (cm)": 1.4, "petal width (cm)": 0.2}
    assert served.post("/predict", json={"rows": [row]}).json()["predictions"] == ["setosa"]

    listed = client.get("/v1/runs").json()["runs"]
    assert [x["id"] for x in listed] == [run_id]


def test_event_stream_reports_progress(stack, iris_csv):
    client, worker, _, _ = stack
    ds = upload(client, iris_csv)
    run_id = client.post("/v1/runs", json={"dataset_id": ds["dataset_id"], "target": "species", "auto_approve": True}).json()["id"]
    worker.drain()
    text = client.get(f"/v1/runs/{run_id}/events").text
    events = [blk for blk in text.split("\n\n") if blk.startswith("id:")]
    kinds = [blk.split("\n")[1].removeprefix("event: ") for blk in events]
    assert kinds[0] == "status" and kinds.count("agent_done") >= 8 and "event: end" in text
    assert '"status": "completed"' in events[-1]
    ids = [int(blk.split("\n")[0].removeprefix("id: ")) for blk in events]
    assert ids == sorted(ids)
    # Reconnect with Last-Event-ID resumes after that event instead of replaying everything.
    tail = client.get(f"/v1/runs/{run_id}/events", headers={"Last-Event-ID": str(ids[-2])}).text
    assert tail.count("id: ") == 1


def test_upload_validation(stack, tmp_path):
    client, *_ = stack
    bad = tmp_path / "x.exe"
    bad.write_bytes(b"MZ")
    assert client.post("/v1/datasets", files={"file": ("x.exe", bad.read_bytes())}).status_code == 415
    assert client.post("/v1/datasets", files={"file": ("x.csv", b"\x00\x01garbage")}).status_code == 422
    assert client.get("/v1/datasets/../../etc").status_code == 404
    assert client.get("/v1/runs/nope").status_code == 404


def test_upload_size_limit(backend, tmp_path):
    s = backend.model_copy(update={"max_upload_mb": 1})
    database = dbm.Database(s.sqlalchemy_url())
    client = TestClient(create_api(s, database))
    big = "a,b\n" + "1,2\n" * 400_000  # ~1.6 MB
    assert client.post("/v1/datasets", files={"file": ("big.csv", big.encode())}).status_code == 413
    client.close()
    database.dispose()


# --------------------------------------------------------------------------- cancel
def test_cancel_queued_run(stack, iris_csv):
    client, worker, _, _ = stack
    ds = upload(client, iris_csv)
    run_id = client.post("/v1/runs", json={"dataset_id": ds["dataset_id"], "target": "species"}).json()["id"]
    assert client.post(f"/v1/runs/{run_id}/cancel").json()["status"] == "cancelled"
    assert worker.drain() == 0
    assert client.post(f"/v1/runs/{run_id}/cancel").status_code == 409


def test_cancel_running_run_stops_at_next_agent(stack, iris_csv):
    client, worker, database, _ = stack
    ds = upload(client, iris_csv)
    run_id = client.post("/v1/runs", json={"dataset_id": ds["dataset_id"], "target": "species", "auto_approve": True}).json()["id"]
    calls = {"n": 0}

    def cancel_after_two(rid):
        calls["n"] += 1
        if calls["n"] == 3:
            database.request_cancel(rid)
        return database.cancel_requested(rid)

    worker.svc.ctx.cancel_check = cancel_after_two
    worker.drain()
    run = client.get(f"/v1/runs/{run_id}").json()
    assert run["status"] == "cancelled"
    assert [e["node"] for e in run["detail"]["events"]] == ["profiler", "planner"]


# --------------------------------------------------------------------------- crash recovery
def test_worker_crash_resumes_from_last_checkpoint(stack, iris_csv, monkeypatch):
    client, worker, database, settings = stack
    ds = upload(client, iris_csv)
    run_id = client.post("/v1/runs", json={"dataset_id": ds["dataset_id"], "target": "species", "auto_approve": True}).json()["id"]

    real_run = TrainerAgent.run

    def crash(self, state, ctx):
        raise KeyboardInterrupt("simulated worker crash (process killed)")

    monkeypatch.setattr(TrainerAgent, "run", crash)
    with pytest.raises(KeyboardInterrupt):
        worker.run_once()
    job = database.jobs_for(run_id)[0]
    assert job["status"] == "running"  # orphaned: its worker is gone

    # The lease runs out (no heartbeats), and a fresh worker picks the job up.
    with database.engine.begin() as c:
        c.execute(update(dbm.jobs).where(dbm.jobs.c.id == job["id"]).values(lease_expires_at=dbm.now() - timedelta(seconds=1)))
    monkeypatch.setattr(TrainerAgent, "run", real_run)
    w2 = Worker(settings, database, provider=None, owner="w2")
    assert w2.drain() == 1
    w2.close()

    run = client.get(f"/v1/runs/{run_id}").json()
    assert run["status"] == "completed", run
    nodes = [e["node"] for e in run["detail"]["events"]]
    assert nodes.count("profiler") == 1 and nodes.count("analyst") == 1  # resumed, not restarted
    assert database.get_job(job["id"])["attempts"] == 2


def test_readyz_and_queue(stack):
    client, *_ = stack
    assert client.get("/healthz").json() == {"status": "ok"}
    r = client.get("/readyz")
    assert r.status_code == 200 and r.json()["database"] == "ok"
    assert "jobs" in client.get("/v1/queue").json()


def test_dataframe_columns_are_peeked_for_excel(stack, tmp_path):
    client, *_ = stack
    p = tmp_path / "t.xlsx"
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_excel(p, index=False)
    r = client.post("/v1/datasets", files={"file": ("t.xlsx", p.read_bytes())})
    assert r.status_code == 201 and r.json()["columns"] == ["a", "b"]


def test_python_client_end_to_end(stack, iris_csv, tmp_path):
    """The client (used by the UI) against the real API, including the SSE stream."""
    from novaml.client import NovaMLClient, NovaMLError

    client, worker, _, _ = stack
    c = NovaMLClient("http://testserver")
    c.http = client  # TestClient is an httpx.Client bound to the app
    ds = c.upload(iris_csv)
    run = c.create_run(ds["dataset_id"], "species", auto_approve=True, idempotency_key="k")
    assert c.create_run(ds["dataset_id"], "species", auto_approve=True, idempotency_key="k")["id"] == run["id"]
    worker.drain()
    kinds = [e["event"] for e in c.events(run["id"])]
    assert kinds[0] == "status" and "agent_done" in kinds and kinds[-1] == "end"
    assert c.wait(run["id"])["status"] == "completed"
    assert c.download_model(run["id"], tmp_path / "m.zip").stat().st_size > 1000
    with pytest.raises(NovaMLError) as e:
        c.approve(run["id"], ["linear"])
    assert e.value.status == 409
