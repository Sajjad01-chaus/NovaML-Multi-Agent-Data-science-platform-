"""End-to-end smoke test against a running stack (docker compose or anything else).

    python scripts/smoke.py --api http://localhost:8080

Uploads a dataset, starts a run, follows the event stream, approves when asked,
waits for completion and checks the model bundle downloads. Exit code 0 = healthy.
"""

from __future__ import annotations

import argparse
import io
import sys
import tempfile
import time
import zipfile
from pathlib import Path

from sklearn.datasets import load_breast_cancer

from novaml.client import NovaMLClient


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8080")
    ap.add_argument("--timeout", type=float, default=600)
    a = ap.parse_args()

    c = NovaMLClient(a.api, timeout=60)
    deadline = time.monotonic() + 300
    while True:  # wait for readiness
        try:
            if c.http.get("/readyz").status_code == 200:
                break
        except Exception:
            pass
        if time.monotonic() > deadline:
            print("API never became ready", file=sys.stderr)
            return 1
        time.sleep(2)

    df = load_breast_cancer(as_frame=True).frame
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "breast_cancer.csv"
        df.to_csv(path, index=False)
        ds = c.upload(path)
    print(f"uploaded {ds['dataset_id']} ({ds['size_bytes']} bytes, {len(ds['columns'])} columns)")

    run = c.create_run(ds["dataset_id"], "target")
    rid = run["id"]
    print(f"run {rid} queued")
    t0 = time.monotonic()
    last = 0
    while True:
        for ev in c.events(rid, after=last):  # resume the stream instead of replaying it
            last = ev.get("id", last)
            if ev["event"] == "agent_done":
                d = ev["data"]
                print(f"  {d['node']:<17} {d['status']:<6} {d['ms']:>8.0f} ms")
        run = c.wait(rid, timeout_s=a.timeout)
        if run["status"] == "awaiting_approval":
            models = run["summary"]["pending"]["candidates"]
            print(f"  approving {models}")
            c.approve(rid, models)
            continue
        break
    print(f"run {rid}: {run['status']} in {time.monotonic() - t0:.0f}s; summary={run.get('summary')}")
    if run["status"] != "completed":
        print(f"error: {run.get('error')}", file=sys.stderr)
        return 1
    z = zipfile.ZipFile(io.BytesIO(c.http.get(f"/v1/runs/{rid}/model").content))
    names = sorted(z.namelist())
    print("bundle:", names)
    return 0 if {"model.joblib", "input_schema.json", "model_card.json"} <= set(names) else 1


if __name__ == "__main__":
    raise SystemExit(main())
