import copy

import pytest

from novaml.evals.cases import CASES, get_cases
from novaml.evals.report import compare_markdown, gate, markdown, to_report
from novaml.evals.runner import run_suite, score_run


def fake_state(score=0.9, base=0.5, dropped=(), metric="f1_macro"):
    return {
        "holdout": {"model": {"balanced_accuracy": score}, "baseline": {"balanced_accuracy": base}},
        "metric": metric,
        "best": {"model": "linear"},
        "feature_plan": {"drop_columns": list(dropped)},
        "llm_calls": [
            {"source": "llm", "input_tokens": 10, "output_tokens": 5, "error": None},
            {"source": "policy", "input_tokens": 3, "output_tokens": 0, "error": "invalid output: unknown model"},
        ],
    }


LEAK = next(c for c in CASES if c.name == "leak_missingness")


def test_score_run_checks_decisions():
    r = score_run(LEAK, "llm", fake_state(0.78, dropped=["boat", "passenger_id"]), "completed", [], 3.0)
    assert r.passed and {c.name for c in r.checks} == {
        "min_score", "max_score", "drop:boat", "drop:passenger_id", "keep:sex", "keep:pclass"
    }
    assert (r.llm_calls, r.llm_decisions, r.policy_fallbacks, r.invalid_outputs, r.tokens) == (2, 1, 1, 1, 18)


def test_leak_that_gets_through_fails_two_ways():
    r = score_run(LEAK, "llm", fake_state(0.97, dropped=["passenger_id"]), "completed", [], 3.0)
    failed = {c.name for c in r.checks if not c.passed}
    assert not r.passed and failed == {"max_score", "drop:boat"}


def test_dropping_real_signal_fails():
    r = score_run(LEAK, "llm", fake_state(0.75, dropped=["boat", "passenger_id", "sex"]), "completed", [], 3.0)
    assert [c.name for c in r.checks if not c.passed] == ["keep:sex"]


def test_failed_run_is_scored_not_raised():
    r = score_run(LEAK, "policy", {}, "failed", ["profiler: bad data"], 1.0)
    assert not r.passed and "bad data" in r.error


def report_of(*results):
    return to_report(list(results), "policy", {"suite": "core"})


def test_gate_flags_regressions_only():
    base = report_of(score_run(LEAK, "policy", fake_state(0.78, dropped=["boat", "passenger_id"]), "completed", [], 1))
    same = copy.deepcopy(base)
    assert gate(same, base) == []

    noisy = report_of(score_run(LEAK, "policy", fake_state(0.77, dropped=["boat", "passenger_id"]), "completed", [], 1))
    assert gate(noisy, base) == []  # within tolerance

    worse = report_of(score_run(LEAK, "policy", fake_state(0.70, dropped=["passenger_id"]), "completed", [], 1))
    problems = gate(worse, base)
    assert any("drop:boat" in p for p in problems) and any("score" in p for p in problems)

    crashed = report_of(score_run(LEAK, "policy", {}, "failed", ["boom"], 1))
    assert any("failed" in p for p in gate(crashed, base))


def test_gate_reports_missing_cases():
    base = report_of(score_run(LEAK, "policy", fake_state(0.78, dropped=["boat", "passenger_id"]), "completed", [], 1))
    empty = report_of()
    assert any("missing" in p for p in gate(empty, base))


def test_reports_render():
    rep = report_of(score_run(LEAK, "policy", fake_state(0.78, dropped=["boat"]), "completed", [], 1))
    md = markdown(rep)
    assert "| leak_missingness |" in md and "drop:passenger_id" in md
    other = copy.deepcopy(rep) | {"mode": "llm"}
    assert "| leak_missingness | 0.7800 | 0.7800 | +0.0000 |" in compare_markdown(rep, other)


def test_suites():
    core = get_cases("core")
    assert len(core) == 12 and all(c.suite == "core" for c in core)
    assert {c.name for c in get_cases("extended")} >= {"titanic", "adult"}
    with pytest.raises(KeyError):
        get_cases(names=["nope"])


def test_offline_cases_run_end_to_end(tmp_path):
    results = run_suite(get_cases(names=["iris", "leak_target_copy"]), "policy", None, work_dir=tmp_path)
    by = {r.case: r for r in results}
    assert by["iris"].passed, by["iris"].checks
    assert by["leak_target_copy"].passed, by["leak_target_copy"].checks
    assert "price_in_cents" in by["leak_target_copy"].dropped
