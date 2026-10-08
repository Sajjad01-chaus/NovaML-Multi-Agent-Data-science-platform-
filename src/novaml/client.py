"""Python client for the NovaML HTTP API (used by the UI; handy in notebooks).

    c = NovaMLClient("http://localhost:8080")
    ds = c.upload("churn.csv")
    run = c.create_run(ds["dataset_id"], "churn", auto_approve=True)
    for event in c.events(run["id"]):
        print(event)
    c.download_model(run["id"], "model.zip")
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx


class NovaMLError(RuntimeError):
    def __init__(self, status: int, detail: Any):
        super().__init__(f"HTTP {status}: {detail}")
        self.status, self.detail = status, detail


class NovaMLClient:
    def __init__(self, base_url: str, timeout: float = 30.0, transport: httpx.BaseTransport | None = None):
        self.http = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout, transport=transport)

    def close(self) -> None:
        self.http.close()

    def _req(self, method: str, url: str, **kw: Any) -> Any:
        r = self.http.request(method, url, **kw)
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail")
            except ValueError:
                detail = r.text
            raise NovaMLError(r.status_code, detail)
        return r

    def upload(self, path: str | Path | None = None, data: bytes | None = None, filename: str | None = None) -> dict:
        if path is not None:
            data, filename = Path(path).read_bytes(), Path(path).name
        return self._req("POST", "/v1/datasets", files={"file": (filename or "data.csv", data or b"")}).json()

    def create_run(self, dataset_id: str, target: str, problem_type: str | None = None, auto_approve: bool = False, idempotency_key: str | None = None) -> dict:
        body = {"dataset_id": dataset_id, "target": target, "problem_type": problem_type, "auto_approve": auto_approve}
        headers = {"Idempotency-Key": idempotency_key or uuid.uuid4().hex}
        return self._req("POST", "/v1/runs", json=body, headers=headers).json()

    def get_run(self, run_id: str) -> dict:
        return self._req("GET", f"/v1/runs/{run_id}").json()

    def list_runs(self, limit: int = 50) -> list[dict]:
        return self._req("GET", "/v1/runs", params={"limit": limit}).json()["runs"]

    def approve(self, run_id: str, models: list[str]) -> dict:
        return self._req("POST", f"/v1/runs/{run_id}/approve", json={"models": models}).json()

    def cancel(self, run_id: str) -> dict:
        return self._req("POST", f"/v1/runs/{run_id}/cancel").json()

    def wait(self, run_id: str, timeout_s: float = 1800, poll_s: float = 1.0) -> dict:
        """Poll until the run settles (completed / failed / cancelled / awaiting_approval)."""
        deadline = time.monotonic() + timeout_s
        while True:
            run = self.get_run(run_id)
            if run["status"] not in ("queued", "running"):
                return run
            if time.monotonic() > deadline:
                raise TimeoutError(f"run {run_id} still {run['status']}")
            time.sleep(poll_s)

    def events(self, run_id: str, after: int = 0) -> Iterator[dict]:
        """Server-Sent Events as dicts: {"id", "event", "data"}; ends when the run settles."""
        with self.http.stream("GET", f"/v1/runs/{run_id}/events", params={"after": after}, timeout=None) as r:
            event: dict[str, Any] = {}
            for line in r.iter_lines():
                if not line:
                    if "event" in event:
                        yield event
                    event = {}
                elif line.startswith("id: "):
                    event["id"] = int(line[4:])
                elif line.startswith("event: "):
                    event["event"] = line[7:]
                elif line.startswith("data: "):
                    event["data"] = json.loads(line[6:])

    def download_model(self, run_id: str, dest: str | Path) -> Path:
        r = self._req("GET", f"/v1/runs/{run_id}/model")
        Path(dest).write_bytes(r.content)
        return Path(dest)
