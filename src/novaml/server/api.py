"""HTTP API.

The API never runs agents itself: it validates requests, writes to the run
registry and queue, and reads run state. Workers do the work. That keeps request
latency flat no matter how long a run takes, and lets each side scale on its own.

    POST /v1/datasets                 upload a CSV/XLSX/Parquet file
    POST /v1/runs                     start a run (Idempotency-Key header supported)
    GET  /v1/runs, /v1/runs/{id}      status, results, pending approval
    POST /v1/runs/{id}/approve        human-in-the-loop approval -> resume job
    POST /v1/runs/{id}/cancel
    GET  /v1/runs/{id}/events         live progress (Server-Sent Events)
    GET  /v1/runs/{id}/model          servable bundle (zip)
    GET  /v1/runs/{id}/model-card
    GET  /healthz, /readyz, /v1/queue
"""

# No `from __future__ import annotations`: FastAPI resolves body/param models at runtime.
import hashlib
import io
import json
import secrets
import time
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastapi import FastAPI, File, Header, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from novaml.artifacts import make_store
from novaml.config import Settings, get_settings
from novaml.log import configure_logging, get_logger
from novaml.server.db import Database
from novaml.service import InvalidResume, NovaML, RunNotFound

log = get_logger(__name__)

TERMINAL = {"completed", "failed", "cancelled"}
CHUNK = 1 << 20


class CreateRun(BaseModel):
    dataset_id: str = Field(pattern=r"^[a-f0-9]{16}$")
    target: str = Field(min_length=1, max_length=256)
    problem_type: Literal["classification", "regression"] | None = None
    auto_approve: bool = False


class Approve(BaseModel):
    models: list[str] = Field(min_length=1, max_length=10)


def create_api(settings: Settings | None = None, db: Database | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging()
    db = db or Database(settings.sqlalchemy_url())
    datasets = make_store(settings, "datasets")
    svc = NovaML(settings, provider=None)  # read-only use: state, validation

    app = FastAPI(title="NovaML API", version="0.3.0")
    app.state.db, app.state.svc = db, svc

    @app.on_event("shutdown")
    def _close() -> None:
        svc.close()

    # ---------------------------------------------------------------- ops
    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> JSONResponse:
        checks: dict[str, str] = {}
        try:
            db.ping()
            checks["database"] = "ok"
        except Exception as e:
            checks["database"] = f"error: {type(e).__name__}"
        try:
            datasets.exists("readyz-probe")
            checks["artifacts"] = "ok"
        except Exception as e:
            checks["artifacts"] = f"error: {type(e).__name__}"
        ok = all(v == "ok" for v in checks.values())
        return JSONResponse({"ready": ok, **checks}, status_code=200 if ok else 503)

    @app.get("/v1/queue")
    def queue() -> dict:
        return {"jobs": db.queue_depth()}

    # ---------------------------------------------------------------- datasets
    @app.post("/v1/datasets", status_code=201)
    def upload_dataset(file: UploadFile = File(...)) -> dict:
        ext = Path(file.filename or "").suffix.lower()
        if ext not in settings.allowed_extensions:
            raise HTTPException(415, f"unsupported file type {ext or '(none)'}; allowed: {list(settings.allowed_extensions)}")
        limit = settings.max_upload_mb * 1024 * 1024
        buf, size, digest = io.BytesIO(), 0, hashlib.sha256()
        while chunk := file.file.read(CHUNK):
            size += len(chunk)
            if size > limit:
                raise HTTPException(413, f"file exceeds {settings.max_upload_mb} MB")
            digest.update(chunk)
            buf.write(chunk)
        data = buf.getvalue()
        try:
            columns = _peek_columns(data, ext)
        except Exception as e:
            raise HTTPException(422, f"could not parse file as a table: {type(e).__name__}") from e
        dataset_id = secrets.token_hex(8)
        key = f"data{ext}"
        datasets.put_bytes(f"{dataset_id}/{key}", data)
        meta = {
            "dataset_id": dataset_id,
            "filename": Path(file.filename or "").name[:200],  # display only, never a path
            "ext": ext,
            "key": key,
            "size_bytes": size,
            "sha256": digest.hexdigest(),
            "columns": columns,
        }
        datasets.save_json(f"{dataset_id}/meta.json", meta)
        log.info("dataset_uploaded", dataset_id=dataset_id, size=size)
        return meta

    @app.get("/v1/datasets/{dataset_id}")
    def get_dataset(dataset_id: str) -> dict:
        return _dataset_meta(dataset_id)

    def _dataset_meta(dataset_id: str) -> dict:
        try:
            if not dataset_id.isalnum() or not datasets.exists(f"{dataset_id}/meta.json"):
                raise HTTPException(404, "dataset not found")
            return datasets.load_json(f"{dataset_id}/meta.json")
        except ValueError as e:
            raise HTTPException(404, "dataset not found") from e

    # ---------------------------------------------------------------- runs
    @app.post("/v1/runs", status_code=202)
    def create_run(body: CreateRun, response: Response, idempotency_key: str | None = Header(default=None, max_length=128)) -> dict:
        meta = _dataset_meta(body.dataset_id)
        if meta.get("columns") and body.target not in meta["columns"]:
            raise HTTPException(422, f"target {body.target!r} is not a column of this dataset")
        run, created = db.create_run(
            body.dataset_id, body.target, body.problem_type, idempotency_key,
            max_attempts=settings.job_max_attempts, auto_approve=body.auto_approve,
        )
        if created:
            db.emit(run["id"], "status", {"status": "queued"})
        else:
            response.status_code = 200  # idempotent replay: same run, no new job
        return _public(run)

    @app.get("/v1/runs")
    def list_runs(limit: int = Query(50, ge=1, le=500), status: str | None = None) -> dict:
        return {"runs": [_public(r) for r in db.list_runs(limit, status)]}

    @app.get("/v1/runs/{run_id}")
    def get_run(run_id: str) -> dict:
        run = _run_or_404(run_id)
        out = _public(run)
        detail = _detail(run_id)
        if detail:
            out["detail"] = detail
        return out

    @app.post("/v1/runs/{run_id}/approve", status_code=202)
    def approve(run_id: str, body: Approve) -> dict:
        run = _run_or_404(run_id)
        if run["status"] != "awaiting_approval":
            raise HTTPException(409, f"run is {run['status']}, not awaiting approval")
        try:
            chosen = NovaML.validate_approval(svc.get(run_id), body.models)
        except (InvalidResume, RunNotFound) as e:
            raise HTTPException(422, str(e)) from e
        if not db.request_approval_resume(run_id, chosen, settings.job_max_attempts):
            raise HTTPException(409, "run is no longer awaiting approval")
        db.emit(run_id, "approved", {"models": chosen})
        return _public(db.get_run(run_id)) | {"approved": chosen}

    @app.post("/v1/runs/{run_id}/cancel", status_code=202)
    def cancel(run_id: str) -> dict:
        run = _run_or_404(run_id)
        if not db.request_cancel(run_id):
            raise HTTPException(409, f"run is already {run['status']}")
        db.emit(run_id, "cancel_requested", {})
        return _public(db.get_run(run_id))

    @app.get("/v1/runs/{run_id}/events")
    def events(
        run_id: str,
        request: Request,
        after: int = Query(0, ge=0),
        last_event_id: str | None = Header(default=None),
        follow: bool = True,
    ) -> StreamingResponse:
        _run_or_404(run_id)
        start = int(last_event_id) if last_event_id and last_event_id.isdigit() else after

        def stream():
            last, idle = start, 0.0
            yield "retry: 2000\n\n"
            while True:
                batch = db.events_after(run_id, last)
                for e in batch:
                    last = e["id"]
                    yield f"id: {e['id']}\nevent: {e['kind']}\ndata: {json.dumps(e['payload'], default=str)}\n\n"
                run = db.get_run(run_id)
                settled = run is None or run["status"] in TERMINAL or run["status"] == "awaiting_approval"
                if not follow or (settled and not batch):
                    yield f"event: end\ndata: {json.dumps({'status': run['status'] if run else 'missing'})}\n\n"
                    return
                if not batch:
                    time.sleep(0.5)
                    idle += 0.5
                    if idle >= 15:  # keep proxies from closing an idle stream
                        idle = 0.0
                        yield ": keep-alive\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/v1/runs/{run_id}/model")
    def model(run_id: str) -> Response:
        run = _run_or_404(run_id)
        if run["status"] != "completed":
            raise HTTPException(409, f"run is {run['status']}; the model is available once it completes")
        data = make_store(settings, run_id).zip_dir("bundle")
        return Response(data, media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="novaml-{run_id}.zip"'})

    @app.get("/v1/runs/{run_id}/model-card")
    def model_card(run_id: str) -> dict:
        _run_or_404(run_id)
        store = make_store(settings, run_id)
        if not store.exists("model_card.json"):
            raise HTTPException(404, "model card not available yet")
        return store.load_json("model_card.json")

    # ---------------------------------------------------------------- helpers
    def _run_or_404(run_id: str) -> dict:
        run = db.get_run(run_id) if run_id.isalnum() else None
        if run is None:
            raise HTTPException(404, "run not found")
        return run

    def _detail(run_id: str) -> dict | None:
        try:
            res = svc.get(run_id)
        except RunNotFound:
            return None
        s = res.state
        return {
            "pending": res.pending,
            "problem_type": s.get("problem_type"),
            "metric": s.get("metric"),
            "profile": s.get("profile"),
            "plan": s.get("plan"),
            "analysis": s.get("analysis"),
            "feature_plan": s.get("feature_plan"),
            "feature_warnings": s.get("feature_warnings"),
            "leaderboard": s.get("leaderboard", []),
            "best": s.get("best"),
            "holdout": s.get("holdout"),
            "critiques": s.get("critiques", []),
            "llm_calls": s.get("llm_calls", []),
            "messages": s.get("messages", [])[-200:],
            "events": s.get("events", []),
            "errors": s.get("errors", []),
            "model_card": s.get("model_card"),
        }

    return app


def _public(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": run["id"],
        "status": run["status"],
        "dataset_id": run["dataset_id"],
        "target": run["target"],
        "problem_type": run["problem_type"],
        "cancel_requested": run["cancel_requested"],
        "summary": run.get("summary"),
        "error": run.get("error"),
        "created_at": run["created_at"].isoformat(),
        "updated_at": run["updated_at"].isoformat(),
    }


def _peek_columns(data: bytes, ext: str) -> list[str]:
    bio = io.BytesIO(data)
    if ext == ".csv":
        df = pd.read_csv(bio, nrows=5)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(bio, nrows=5)
    else:
        df = pd.read_parquet(bio)
    if df.shape[1] < 2:
        raise ValueError("need at least two columns")
    return [str(c).strip() for c in df.columns]


def app_factory() -> FastAPI:
    """`uvicorn novaml.server.api:app_factory --factory`"""
    return create_api()
