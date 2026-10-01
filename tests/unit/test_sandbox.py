import pandas as pd
import pytest

from novaml.sandbox import executor
from novaml.sandbox.executor import SubprocessSandbox
from novaml.sandbox.policy import PolicyViolation, check

DF = pd.DataFrame({"a": [1, 2, 3, 4], "b": ["x", "y", "x", "y"]})


@pytest.fixture(scope="module")
def sbx():
    return SubprocessSandbox(timeout_s=20, max_output_chars=200)


def test_runs_analysis_code(sbx):
    r = sbx.run("import numpy as np\nprint(df.groupby('b')['a'].mean().to_dict())\nprint(np.sqrt(16))", DF)
    assert r.ok, r.error
    assert "{'x': 2.0, 'y': 3.0}" in r.stdout and "4.0" in r.stdout


@pytest.mark.parametrize(
    "code",
    [
        "import os\nos.system('echo hi')",
        "import subprocess",
        "from socket import socket",
        "open('/etc/passwd').read()",
        "().__class__.__base__.__subclasses__()",
        "getattr(df, 'to_csv')('x.csv')",
        "df.to_csv('stolen.csv')",
        "pd.read_csv('/etc/passwd')",
        "df.query('a > 1')",
        "eval('1+1')",
        "__import__('os')",
        "x = '__class__'",
        "import numpy.ctypeslib",
    ],
)
def test_policy_blocks_dangerous_code(sbx, code):
    with pytest.raises(PolicyViolation):
        check(code)
    r = sbx.run(code, DF)
    assert not r.ok and r.blocked


def test_trailing_expression_is_echoed_like_a_notebook(sbx):
    r = sbx.run("x = df['a'].sum()\ndf.groupby('b')['a'].mean()", DF)
    assert r.ok and "x    2.0" in r.stdout and "y    3.0" in r.stdout


def test_realistic_frame_under_memory_limit():
    """Regression: per-core thread pools crashed the child under RLIMIT_AS on many-core hosts."""
    n = 2000
    big = pd.DataFrame({"num": range(n), "cat": ["alpha", "beta", None, "delta"] * (n // 4), "y": [0, 1] * (n // 2)})
    r = SubprocessSandbox(timeout_s=30, memory_mb=512).run("print(df.groupby('cat')['y'].mean().round(2).to_dict())", big)
    assert r.ok, r.error
    assert "alpha" in r.stdout


def test_runtime_errors_are_reported_not_raised(sbx):
    r = sbx.run("print('before')\n1/0", DF)
    assert not r.ok and "ZeroDivisionError" in r.error and "before" in r.stdout


def test_timeout_kills_runaway_code():
    r = SubprocessSandbox(timeout_s=3).run("while True:\n    pass", DF)
    assert not r.ok and "Timeout" in r.error


def test_output_is_truncated(sbx):
    r = sbx.run("print('z' * 5000)", DF)
    assert r.ok and "truncated" in r.stdout and len(r.stdout) < 400


def test_secrets_are_not_passed_to_child(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-secret")
    seen = {}
    real_run = executor.subprocess.run

    def spy(*args, **kwargs):
        seen.update(kwargs["env"])
        return real_run(*args, **kwargs)

    monkeypatch.setattr(executor.subprocess, "run", spy)
    assert SubprocessSandbox(timeout_s=20).run("print(1)", DF).ok
    assert "ANTHROPIC_API_KEY" not in seen and "GROQ_API_KEY" not in seen
