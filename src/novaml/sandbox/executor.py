"""Runs LLM-written pandas analysis code out of process.

Layers of defence:
1. Static policy (policy.py): allowlisted imports, no dunders, no I/O methods.
2. Separate interpreter in isolated mode (-I), in an empty temp dir, with a
   scrubbed environment: API keys and other secrets never reach the child.
3. Restricted builtins plus a guarded __import__ inside the child.
4. Wall-clock timeout (all platforms); memory and CPU rlimits on POSIX.
5. Output truncation, so a runaway print can't flood the LLM context.

A container backend (no network, read-only rootfs, cgroup limits) can implement
the same `Sandbox` protocol for multi-tenant production; this process-level
sandbox is the portable default.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

from novaml.sandbox.policy import (
    ALLOWED_IMPORTS,
    FORBIDDEN_ATTRS,
    FORBIDDEN_NAMES,
    PolicyViolation,
    check,
)


@dataclass
class SandboxResult:
    ok: bool
    stdout: str = ""
    error: str | None = None
    duration_ms: float = 0.0
    blocked: bool = False  # rejected by static policy, never executed


class Sandbox(Protocol):
    def run(self, code: str, df: pd.DataFrame) -> SandboxResult: ...


_RUNNER = r'''
import builtins, io, json, sys, contextlib
job = json.load(open("job.json", encoding="utf-8"))
if sys.platform != "win32":
    import resource
    mem = job["memory_mb"] * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
    cpu = int(job["timeout_s"]) + 1
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
import numpy as np
import pandas as pd
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)
df = pd.read_parquet("data.parquet")

allowed = set(job["allowed_imports"])
blocked_parts = set(job["blocked_submodules"])
real_import = builtins.__import__
def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    parts = name.split(".")
    if level or parts[0] not in allowed or any(p in blocked_parts or p.startswith("_") for p in parts[1:]):
        raise ImportError(f"import of {name!r} is not allowed")
    return real_import(name, globals, locals, fromlist, level)

safe = {k: v for k, v in vars(builtins).items() if k not in set(job["forbidden"]) and not k.startswith("_")}
safe["__import__"] = guarded_import
buf = io.StringIO()
result = {"ok": True, "error": None}
try:
    with contextlib.redirect_stdout(buf):
        exec(compile(job["code"], "<analysis>", "exec"), {"__builtins__": safe, "df": df, "pd": pd, "np": np})
except BaseException as e:
    result = {"ok": False, "error": f"{type(e).__name__}: {e}"[:1000]}
out = buf.getvalue()
limit = job["max_output"]
if len(out) > limit:
    out = out[:limit] + f"\n... [truncated {len(out) - limit} chars]"
result["stdout"] = out
with open("result.json", "w", encoding="utf-8") as f:
    json.dump(result, f)
'''

_ENV_KEEP = ("SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "TEMP", "TMP", "PATH", "LANG", "LC_ALL")


class SubprocessSandbox:
    def __init__(self, timeout_s: float = 30.0, max_output_chars: int = 4000, memory_mb: int = 1024, max_rows: int = 50_000):
        self.timeout_s = timeout_s
        self.max_output_chars = max_output_chars
        self.memory_mb = memory_mb
        self.max_rows = max_rows

    def run(self, code: str, df: pd.DataFrame) -> SandboxResult:
        t0 = time.perf_counter()
        try:
            check(code)
        except PolicyViolation as e:
            return SandboxResult(False, error=f"PolicyViolation: {e}", blocked=True)

        with tempfile.TemporaryDirectory(prefix="novaml-sbx-") as tmp:
            d = Path(tmp)
            sample = df if len(df) <= self.max_rows else df.sample(self.max_rows, random_state=0)
            sample.to_parquet(d / "data.parquet", index=False)
            (d / "runner.py").write_text(_RUNNER, encoding="utf-8")
            (d / "job.json").write_text(
                json.dumps(
                    {
                        "code": code,
                        "timeout_s": self.timeout_s,
                        "memory_mb": self.memory_mb,
                        "max_output": self.max_output_chars,
                        "allowed_imports": sorted(ALLOWED_IMPORTS),
                        "forbidden": sorted(FORBIDDEN_NAMES),
                        "blocked_submodules": sorted(FORBIDDEN_ATTRS),
                    }
                ),
                encoding="utf-8",
            )
            env = {k: os.environ[k] for k in _ENV_KEEP if k in os.environ}
            try:
                proc = subprocess.run(
                    [sys.executable, "-I", "runner.py"],
                    cwd=d,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                )
            except subprocess.TimeoutExpired:
                return SandboxResult(False, error=f"Timeout: exceeded {self.timeout_s}s", duration_ms=_ms(t0))
            res_file = d / "result.json"
            if not res_file.exists():
                err = (proc.stderr or "").strip().splitlines()[-1:] or ["no result"]
                return SandboxResult(False, error=f"Crashed: {err[0][:500]}", duration_ms=_ms(t0))
            res = json.loads(res_file.read_text(encoding="utf-8"))
        return SandboxResult(res["ok"], res.get("stdout", ""), res.get("error"), _ms(t0))


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)
