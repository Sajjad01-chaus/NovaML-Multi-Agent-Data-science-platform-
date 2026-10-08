"""Command line interface.

    novaml run data.csv --target churn [--problem-type classification] [--auto-approve]
    novaml resume <run_id> --approve random_forest linear
    novaml status <run_id>
    novaml serve <bundle_dir> [--port 8000]
    novaml eval [--suite core|extended|all] [--mode policy|llm|both] [--gate evals/baseline_policy.json]
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

    ev = sub.add_parser("eval", help="run the benchmark suite")
    ev.add_argument("--suite", default="core", choices=["core", "extended", "all"])
    ev.add_argument("--cases", nargs="+", help="run only these cases")
    ev.add_argument("--mode", default="policy", choices=["policy", "llm", "both"])
    ev.add_argument("--out", default="evals/results", help="report directory")
    ev.add_argument("--gate", help="baseline report JSON; exit 1 on regressions")
    ev.add_argument("--write-baseline", help="also save this run as a baseline at the given path")
    ev.add_argument("--pause", type=float, default=20.0, help="seconds between LLM cases (free-tier rate limits)")
    ev.add_argument("--compare-with", help="earlier report JSON to compare this run against")

    a = p.parse_args(argv)

    if a.cmd == "eval":
        return _eval(a)

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


def _eval(a: argparse.Namespace) -> int:
    import json
    from datetime import datetime
    from pathlib import Path

    from novaml.evals.cases import get_cases
    from novaml.evals.report import compare_markdown, gate, to_report, write_report
    from novaml.evals.runner import run_suite
    from novaml.llm.providers import build_provider
    from novaml.log import configure_logging

    configure_logging(level="WARNING")
    cases = get_cases(a.suite, a.cases)
    out = Path(a.out)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    modes = ["policy", "llm"] if a.mode == "both" else [a.mode]
    reports = {}
    for mode in modes:
        provider = None
        if mode == "llm":
            provider = build_provider(Settings())
            if provider is None:
                print("llm mode needs an LLM provider: set GROQ_API_KEY in .env", file=sys.stderr)
                return 2
        print(f"== {mode}: {len(cases)} cases ({a.suite})")

        def show(r):
            mark = "PASS" if r.passed else ("SKIP" if r.status == "skipped" else "FAIL")
            failed = [c.name for c in r.checks if not c.passed]
            print(f"  {mark} {r.case:24s} {r.eval_metric}={r.score} baseline={r.baseline_score} "
                  f"tokens={r.tokens} {r.seconds}s {failed or ''}{r.error or ''}", flush=True)

        results = run_suite(cases, mode, provider, pause_s=a.pause if mode == "llm" else 0.0, progress=show)
        report = to_report(results, mode, {"suite": a.suite})
        j, m = write_report(report, out, f"{stamp}-{mode}")
        reports[mode] = report
        s = report["summary"]
        print(f"  -> {s['cases_passed']}/{s['ran']} cases, {s['checks_passed']}/{s['checks_total']} checks; report {m}")
        if a.write_baseline:
            write_report(report, Path(a.write_baseline).parent, Path(a.write_baseline).stem)
    pair = None
    if len(reports) == 2:
        pair = (reports["policy"], reports["llm"])
    elif a.compare_with:
        pair = (json.loads(Path(a.compare_with).read_text(encoding="utf-8")), next(iter(reports.values())))
    if pair:
        cmp = out / f"{stamp}-compare.md"
        cmp.write_text(compare_markdown(*pair), encoding="utf-8")
        print(f"comparison: {cmp}")

    if a.gate:
        baseline = json.loads(Path(a.gate).read_text(encoding="utf-8"))
        current = reports.get(baseline.get("mode")) or next(iter(reports.values()))
        problems = gate(current, baseline)
        if problems:
            print("EVAL GATE FAILED:", *problems, sep="\n  - ", file=sys.stderr)
            return 1
        print(f"eval gate passed against {a.gate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
