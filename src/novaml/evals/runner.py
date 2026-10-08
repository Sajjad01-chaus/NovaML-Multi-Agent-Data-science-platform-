"""Runs eval cases through the real service and scores them.

A run is scored on two axes:
* outcome: the holdout eval metric vs the case's floor/ceiling and the baseline
* decisions: were planted leaks/IDs dropped, real signal kept, a sensible metric
  chosen, and how often LLM output had to be rejected
"""

from __future__ import annotations

import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from novaml.config import Settings
from novaml.evals.cases import EvalCase
from novaml.llm.base import LLMProvider
from novaml.service import NovaML

# Same settings for every mode so policy and LLM runs are directly comparable,
# trimmed so a full LLM suite fits the Groq free tier.
EVAL_OVERRIDES: dict[str, Any] = {
    "auto_approve": True,
    "cv_folds": 3,
    "tuning_iterations": 5,
    "max_improvement_rounds": 1,
    "analyst_max_steps": 3,
}


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class CaseResult:
    case: str
    mode: str
    status: str
    eval_metric: str
    score: float | None = None
    baseline_score: float | None = None
    optimised_metric: str | None = None
    best_model: str | None = None
    dropped: list[str] = field(default_factory=list)
    checks: list[Check] = field(default_factory=list)
    llm_calls: int = 0
    llm_decisions: int = 0
    policy_fallbacks: int = 0
    invalid_outputs: int = 0
    tokens: int = 0
    seconds: float = 0.0
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.status == "completed" and all(c.passed for c in self.checks)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["passed"] = self.passed
        return d


def score_run(case: EvalCase, mode: str, state: dict[str, Any], status: str, errors: list[str], seconds: float) -> CaseResult:
    r = CaseResult(case=case.name, mode=mode, status=status, eval_metric=case.eval_metric, seconds=round(seconds, 1))
    calls = state.get("llm_calls", [])
    r.llm_calls = len(calls)
    r.llm_decisions = sum(c.get("source") == "llm" for c in calls)
    r.policy_fallbacks = r.llm_calls - r.llm_decisions
    r.invalid_outputs = sum(str(c.get("error") or "").startswith("invalid output") for c in calls)
    r.tokens = sum(c.get("input_tokens", 0) + c.get("output_tokens", 0) for c in calls)
    if status != "completed":
        r.error = "; ".join(errors)[:500] or status
        r.checks.append(Check("completed", False, r.error))
        return r

    holdout = state.get("holdout", {})
    r.score = _round(holdout.get("model", {}).get(case.eval_metric))
    r.baseline_score = _round(holdout.get("baseline", {}).get(case.eval_metric))
    r.optimised_metric = state.get("metric")
    r.best_model = state.get("best", {}).get("model")
    r.dropped = sorted((state.get("feature_plan") or {}).get("drop_columns", []))

    r.checks.append(Check("min_score", r.score is not None and r.score >= case.min_score, f"{r.score} >= {case.min_score}"))
    if case.max_score is not None:
        r.checks.append(Check("max_score", r.score is not None and r.score <= case.max_score, f"{r.score} <= {case.max_score} (above = leakage got through)"))
    for col in case.must_drop:
        r.checks.append(Check(f"drop:{col}", col in r.dropped, "dropped" if col in r.dropped else "kept"))
    for col in case.must_keep:
        r.checks.append(Check(f"keep:{col}", col not in r.dropped, "kept" if col not in r.dropped else "dropped"))
    if case.metrics_ok:
        r.checks.append(Check("metric_choice", r.optimised_metric in case.metrics_ok, f"{r.optimised_metric} in {list(case.metrics_ok)}"))
    return r


def run_case(case: EvalCase, mode: str, provider: LLMProvider | None, work_dir: Path, base: Settings | None = None) -> CaseResult:
    settings = (base or Settings()).model_copy(update={**EVAL_OVERRIDES, "data_dir": work_dir / case.name})
    t0 = time.perf_counter()
    try:
        df = case.load()
    except Exception as e:  # e.g. no network for an OpenML case
        return CaseResult(case.name, mode, "skipped", case.eval_metric, error=f"load failed: {type(e).__name__}: {e}"[:300])
    path = work_dir / f"{case.name}.csv"
    df.to_csv(path, index=False)
    svc = NovaML(settings, provider=provider)
    try:
        res = svc.start(path, case.target, case.problem_type)
        return score_run(case, mode, res.state, res.status, res.errors, time.perf_counter() - t0)
    except Exception as e:
        r = CaseResult(case.name, mode, "crashed", case.eval_metric, seconds=round(time.perf_counter() - t0, 1))
        r.error = f"{type(e).__name__}: {e}"[:500]
        r.checks.append(Check("completed", False, r.error))
        return r
    finally:
        svc.close()


def run_suite(
    cases: list[EvalCase],
    mode: str,
    provider: LLMProvider | None,
    work_dir: Path | None = None,
    pause_s: float = 0.0,
    progress: Any = None,
) -> list[CaseResult]:
    """`pause_s` spaces LLM runs out to stay under per-minute token limits."""
    tmp = None
    if work_dir is None:
        tmp = tempfile.TemporaryDirectory(prefix="novaml-eval-")
        work_dir = Path(tmp.name)
    work_dir.mkdir(parents=True, exist_ok=True)
    results = []
    try:
        for i, case in enumerate(cases):
            if i and pause_s:
                time.sleep(pause_s)
            r = run_case(case, mode, provider, work_dir)
            results.append(r)
            if progress:
                progress(r)
    finally:
        if tmp:
            tmp.cleanup()
    return results


def summarize(results: list[CaseResult]) -> dict[str, Any]:
    ran = [r for r in results if r.status != "skipped"]
    checks = [c for r in ran for c in r.checks]
    decision = [c for c in checks if c.name.split(":")[0] in ("drop", "keep", "metric_choice", "max_score")]
    lifts = [r.score - r.baseline_score for r in ran if r.score is not None and r.baseline_score is not None]
    calls = sum(r.llm_calls for r in ran)
    return {
        "cases": len(results),
        "ran": len(ran),
        "cases_passed": sum(r.passed for r in ran),
        "checks_passed": sum(c.passed for c in checks),
        "checks_total": len(checks),
        "decision_checks_passed": sum(c.passed for c in decision),
        "decision_checks_total": len(decision),
        "mean_lift_over_baseline": _round(sum(lifts) / len(lifts)) if lifts else None,
        "llm_calls": calls,
        "llm_decision_rate": _round(sum(r.llm_decisions for r in ran) / calls) if calls else None,
        "invalid_output_rate": _round(sum(r.invalid_outputs for r in ran) / calls) if calls else None,
        "tokens": sum(r.tokens for r in ran),
        "seconds": round(sum(r.seconds for r in ran), 1),
    }


def _round(v: float | None) -> float | None:
    return None if v is None else round(float(v), 4)
