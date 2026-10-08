"""Worker process: claims jobs from the queue and drives runs through the graph.

Run N of these for throughput (`docker compose up --scale worker=N`). Each job is
leased; a heartbeat thread extends the lease while the job runs. If the process
dies, the lease expires, another worker claims the job, and the run resumes from
its last checkpoint. SIGTERM finishes the current job before exiting.
"""

from __future__ import annotations

import os
import secrets
import signal
import socket
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from novaml.artifacts import make_store
from novaml.config import Settings, get_settings
from novaml.log import configure_logging, get_logger
from novaml.server.db import Database
from novaml.service import InvalidResume, NovaML, RunNotFound, RunResult
from novaml.tools.ml import to_display

log = get_logger(__name__)


class PermanentJobError(Exception):
    """Retrying won't help (bad input, invalid approval)."""


def run_summary(res: RunResult) -> dict[str, Any]:
    s = res.state
    out: dict[str, Any] = {"status": res.status, "problem_type": s.get("problem_type"), "metric": s.get("metric")}
    if res.pending:
        out["pending"] = res.pending
    best = s.get("best")
    if best and s.get("metric") and s.get("problem_type"):
        out["best_model"] = best["model"]
        out["cv_score"] = round(to_display(s["metric"], s["problem_type"], best["cv_score"]), 4)
        out["holdout"] = s.get("holdout", {}).get("model")
    card = s.get("model_card")
    if card:
        out["llm"] = card.get("llm")
        out["warnings"] = card.get("warnings")
    if res.errors:
        out["errors"] = res.errors[-3:]
    return out


class Worker:
    def __init__(self, settings: Settings | None = None, db: Database | None = None, provider: Any = "auto", owner: str | None = None):
        self.settings = settings or get_settings()
        self.db = db or Database(self.settings.sqlalchemy_url())
        self.svc = NovaML(self.settings, provider=provider, events=self.db, cancel_check=self.db.cancel_requested)
        self.datasets = make_store(self.settings, "datasets")
        self.owner = owner or f"{socket.gethostname()}-{os.getpid()}-{secrets.token_hex(3)}"
        self._stop = threading.Event()

    def close(self) -> None:
        self.svc.close()

    def stop(self, *_: Any) -> None:
        log.info("worker_stopping", owner=self.owner)
        self._stop.set()

    def run_forever(self) -> None:
        log.info("worker_started", owner=self.owner)
        while not self._stop.is_set():
            if not self.run_once():
                self._stop.wait(self.settings.worker_poll_s)
        log.info("worker_stopped", owner=self.owner)

    def run_once(self) -> bool:
        job = self.db.claim(self.owner, self.settings.worker_lease_s)
        if job is None:
            return False
        self._process(job)
        return True

    def drain(self, timeout_s: float = 600) -> int:
        """Process jobs until the queue is empty (tests, batch runs)."""
        n, deadline = 0, time.monotonic() + timeout_s
        while time.monotonic() < deadline and self.run_once():
            n += 1
        return n

    # ------------------------------------------------------------------
    def _process(self, job: dict[str, Any]) -> None:
        run_id = job["run_id"]
        beat_stop = threading.Event()
        beat = threading.Thread(target=self._heartbeat, args=(job["id"], beat_stop), daemon=True)
        beat.start()
        log.info("job_started", job=job["id"], kind=job["kind"], run_id=run_id, attempt=job["attempts"])
        try:
            self._execute(job)
            self.db.complete(job["id"], self.owner)
        except PermanentJobError as e:
            self.db.fail(job["id"], self.owner, str(e), retry=False)
            self._finish(run_id, "failed", error=str(e))
        except Exception as e:
            status = self.db.fail(job["id"], self.owner, f"{type(e).__name__}: {e}")
            log.error("job_failed", job=job["id"], run_id=run_id, error=repr(e), next=status)
            if status == "dead":
                self._finish(run_id, "failed", error=f"{type(e).__name__}: {e}")
            else:
                self.db.set_run(run_id, status="queued")
                self.db.emit(run_id, "retrying", {"attempt": job["attempts"], "error": f"{type(e).__name__}: {e}"[:300]})
        finally:
            beat_stop.set()
            beat.join(timeout=5)

    def _heartbeat(self, job_id: int, stop: threading.Event) -> None:
        while not stop.wait(self.settings.worker_heartbeat_s):
            if not self.db.heartbeat(job_id, self.owner, self.settings.worker_lease_s):
                log.warning("lease_lost", job=job_id, owner=self.owner)
                return

    def _execute(self, job: dict[str, Any]) -> None:
        run_id, payload = job["run_id"], job["payload"]
        run = self.db.get_run(run_id)
        if run is None:
            raise PermanentJobError(f"run {run_id} does not exist")
        if run["cancel_requested"]:
            self._finish(run_id, "cancelled")
            return
        self.db.set_run(run_id, status="running")
        self.db.emit(run_id, "status", {"status": "running", "job": job["kind"], "attempt": job["attempts"]})

        try:
            if job["kind"] == "start":
                res = self._start(run_id, payload)
            elif job["kind"] == "resume":
                res = self._resume(run_id, run, payload)
            else:
                raise PermanentJobError(f"unknown job kind {job['kind']!r}")
        except (InvalidResume, RunNotFound) as e:
            raise PermanentJobError(str(e)) from e
        self._finish(run_id, res.status, res=res)

    def _start(self, run_id: str, payload: dict[str, Any]) -> RunResult:
        meta = self.datasets.load_json(f"{payload['dataset_id']}/meta.json")
        with tempfile.TemporaryDirectory(prefix="novaml-ds-") as tmp:
            path = Path(tmp) / f"data{meta['ext']}"
            path.write_bytes(self.datasets.get_bytes(f"{payload['dataset_id']}/{meta['key']}"))
            # Same run_id on a retry: the service resumes from the last checkpoint.
            return self.svc.start(
                path, payload["target"], payload.get("problem_type"), run_id=run_id, auto_approve=bool(payload.get("auto_approve"))
            )

    def _resume(self, run_id: str, run: dict[str, Any], payload: dict[str, Any]) -> RunResult:
        current = self.svc.get(run_id)
        if current.status == "awaiting_approval":
            return self.svc.resume(run_id, payload["approved_models"])
        # A previous attempt already consumed the approval and then crashed mid-run.
        return self.svc.start("", run["target"], run_id=run_id)

    def _finish(self, run_id: str, status: str, res: RunResult | None = None, error: str | None = None) -> None:
        values: dict[str, Any] = {"status": status}
        if res is not None:
            values["summary"] = run_summary(res)
            if res.errors and status in ("failed", "cancelled"):
                values["error"] = "; ".join(res.errors[-3:])[:2000]
        if error:
            values["error"] = error[:2000]
        self.db.set_run(run_id, **values)
        self.db.emit(run_id, "status", {"status": status, **({"pending": res.pending} if res and res.pending else {})})
        log.info("run_status", run_id=run_id, status=status)


def main() -> None:
    configure_logging()
    w = Worker()
    signal.signal(signal.SIGTERM, w.stop)
    signal.signal(signal.SIGINT, w.stop)
    try:
        w.run_forever()
    finally:
        w.close()
