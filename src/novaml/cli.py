"""Command line interface.

    novaml run data.csv --target churn [--problem-type classification] [--auto-approve]
    novaml resume <run_id> --approve random_forest linear
    novaml status <run_id>
    novaml serve <bundle_dir> [--port 8000]
"""

from __future__ import annotations

import argparse
import json
import sys

from novaml.config import Settings
from novaml.service import NovaML, RunResult


def _print(result: RunResult) -> int:
    s = result.state
    for m in s.get("messages", []):
        print(m)
    print(f"\nrun_id={result.run_id} status={result.status}")
    if result.pending:
        print("awaiting approval:", json.dumps(result.pending, indent=2))
        print(f"resume with: novaml resume {result.run_id} --approve {' '.join(result.pending['candidates'])}")
    if result.status == "completed":
        print("bundle:", s.get("bundle_dir"))
    for e in result.errors:
        print("ERROR:", e, file=sys.stderr)
    return 1 if result.status == "failed" else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="novaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="start a run")
    r.add_argument("dataset")
    r.add_argument("--target", required=True)
    r.add_argument("--problem-type", choices=["classification", "regression"])
    r.add_argument("--auto-approve", action="store_true", help="skip the human approval step")

    rs = sub.add_parser("resume", help="approve models for a paused run")
    rs.add_argument("run_id")
    rs.add_argument("--approve", nargs="+", required=True)

    st = sub.add_parser("status", help="show a run")
    st.add_argument("run_id")

    sv = sub.add_parser("serve", help="serve an exported model bundle")
    sv.add_argument("bundle_dir")
    sv.add_argument("--port", type=int, default=8000)

    a = p.parse_args(argv)

    if a.cmd == "serve":
        import uvicorn

        from novaml.serving.app import create_app

        uvicorn.run(create_app(a.bundle_dir), host="127.0.0.1", port=a.port)
        return 0

    settings = Settings(auto_approve=getattr(a, "auto_approve", False) or Settings().auto_approve)
    svc = NovaML(settings)
    try:
        if a.cmd == "run":
            return _print(svc.start(a.dataset, a.target, a.problem_type))
        if a.cmd == "resume":
            return _print(svc.resume(a.run_id, a.approve))
        return _print(svc.get(a.run_id))
    finally:
        svc.close()


if __name__ == "__main__":
    raise SystemExit(main())
