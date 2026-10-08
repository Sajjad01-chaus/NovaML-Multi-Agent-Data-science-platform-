"""Eval reports (JSON + Markdown) and the regression gate."""

from __future__ import annotations

import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from novaml.evals.runner import CaseResult, summarize

SCORE_TOLERANCE = 0.02
"""Allowed holdout drop vs baseline; covers library-version noise across machines."""


def to_report(results: list[CaseResult], mode: str, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "mode": mode,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        **(meta or {}),
        "summary": summarize(results),
        "cases": [r.to_dict() for r in results],
    }


def write_report(report: dict[str, Any], out_dir: Path, name: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    j = out_dir / f"{name}.json"
    m = out_dir / f"{name}.md"
    j.write_text(json.dumps(report, indent=2), encoding="utf-8")
    m.write_text(markdown(report), encoding="utf-8")
    return j, m


def markdown(report: dict[str, Any]) -> str:
    s = report["summary"]
    lines = [
        f"# NovaML eval: `{report['mode']}`",
        "",
        f"{report['created_at']} · {s['cases_passed']}/{s['ran']} cases passed · "
        f"{s['checks_passed']}/{s['checks_total']} checks · decisions {s['decision_checks_passed']}/{s['decision_checks_total']} · "
        f"mean lift {s['mean_lift_over_baseline']} · {s['tokens']} tokens · {s['seconds']}s",
        "",
        "| case | pass | score | baseline | metric optimised | best model | failed checks | LLM decisions | tokens | s |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in report["cases"]:
        failed = ", ".join(f"{k['name']} ({k['detail']})" for k in c["checks"] if not k["passed"]) or ""
        mark = "skip" if c["status"] == "skipped" else ("yes" if c["passed"] else "**no**")
        dec = f"{c['llm_decisions']}/{c['llm_calls']}" if c["llm_calls"] else "-"
        lines.append(
            f"| {c['case']} | {mark} | {_f(c['score'])} | {_f(c['baseline_score'])} | {c['optimised_metric'] or '-'} | "
            f"{c['best_model'] or '-'} | {failed or c.get('error') or ''} | {dec} | {c['tokens']} | {c['seconds']} |"
        )
    return "\n".join(lines) + "\n"


def compare_markdown(a: dict[str, Any], b: dict[str, Any]) -> str:
    """Side-by-side of two runs (e.g. policy vs llm) on the same cases."""
    sa, sb = a["summary"], b["summary"]
    rows = [
        ("cases passed", f"{sa['cases_passed']}/{sa['ran']}", f"{sb['cases_passed']}/{sb['ran']}"),
        ("checks passed", f"{sa['checks_passed']}/{sa['checks_total']}", f"{sb['checks_passed']}/{sb['checks_total']}"),
        ("decision checks", f"{sa['decision_checks_passed']}/{sa['decision_checks_total']}", f"{sb['decision_checks_passed']}/{sb['decision_checks_total']}"),
        ("mean lift over baseline", _f(sa["mean_lift_over_baseline"]), _f(sb["mean_lift_over_baseline"])),
        ("tokens", str(sa["tokens"]), str(sb["tokens"])),
        ("invalid LLM outputs", _f(sa["invalid_output_rate"]), _f(sb["invalid_output_rate"])),
        ("time (s)", str(sa["seconds"]), str(sb["seconds"])),
    ]
    out = [f"| | {a['mode']} | {b['mode']} |", "|---|---|---|", *(f"| {k} | {x} | {y} |" for k, x, y in rows), ""]
    out += [f"| case | {a['mode']} score | {b['mode']} score | delta | {a['mode']} pass | {b['mode']} pass |", "|---|---|---|---|---|---|"]
    bc = {c["case"]: c for c in b["cases"]}
    for c in a["cases"]:
        d = bc.get(c["case"])
        if not d:
            continue
        delta = d["score"] - c["score"] if d["score"] is not None and c["score"] is not None else None
        out.append(f"| {c['case']} | {_f(c['score'])} | {_f(d['score'])} | {_f(delta, signed=True)} | {'yes' if c['passed'] else 'no'} | {'yes' if d['passed'] else 'no'} |")
    return "\n".join(out) + "\n"


def gate(current: dict[str, Any], baseline: dict[str, Any], tolerance: float = SCORE_TOLERANCE) -> list[str]:
    """Regressions of `current` vs `baseline`. Empty list = gate passes."""
    problems = []
    base = {c["case"]: c for c in baseline["cases"]}
    for c in current["cases"]:
        b = base.get(c["case"])
        if not b or c["status"] == "skipped":
            continue
        was = {k["name"]: k["passed"] for k in b["checks"]}
        for k in c["checks"]:
            if was.get(k["name"]) and not k["passed"]:
                problems.append(f"{c['case']}: check '{k['name']}' regressed ({k['detail']})")
        if b["status"] == "completed" and c["status"] != "completed":
            problems.append(f"{c['case']}: run {c['status']} ({c.get('error')})")
        if c["score"] is not None and b["score"] is not None and c["score"] < b["score"] - tolerance:
            problems.append(f"{c['case']}: score {c['score']:.4f} < baseline {b['score']:.4f} - {tolerance}")
    missing = set(base) - {c["case"] for c in current["cases"]}
    if missing and current.get("suite") == baseline.get("suite"):
        problems.append(f"cases missing from run: {sorted(missing)}")
    return problems


def _f(v: float | None, signed: bool = False) -> str:
    if v is None:
        return "-"
    return f"{v:+.4f}" if signed else f"{v:.4f}"
