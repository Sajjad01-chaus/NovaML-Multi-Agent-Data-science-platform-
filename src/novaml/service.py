"""Run service: the single entry point used by the CLI, the UI and (later) the API.

Runs are keyed by `run_id` (= LangGraph thread id) and checkpointed to SQLite on
disk, so a run paused for human approval survives a process restart and can be
resumed from any process that points at the same database.
"""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from novaml.agents.base import AgentContext
from novaml.config import Settings, get_settings
from novaml.graph import build_graph
from novaml.llm.base import LLMProvider
from novaml.llm.providers import build_provider
from novaml.log import configure_logging, get_logger
from novaml.sandbox.executor import Sandbox

log = get_logger(__name__)


class RunNotFound(KeyError):
    pass


class InvalidResume(ValueError):
    pass


@dataclass
class RunResult:
    run_id: str
    status: str  # awaiting_approval | completed | failed
    state: dict[str, Any]
    pending: dict[str, Any] | None = None
    errors: list[str] = field(default_factory=list)


class NovaML:
    def __init__(
        self,
        settings: Settings | None = None,
        provider: LLMProvider | None | str = "auto",
        sandbox: Sandbox | None = None,
    ):
        self.settings = settings or get_settings()
        configure_logging()
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        if provider == "auto":
            provider = build_provider(self.settings)
        self.ctx = AgentContext(self.settings, provider, sandbox)  # type: ignore[arg-type]
        self._conn = sqlite3.connect(self.settings.checkpoint_path, check_same_thread=False)
        self.checkpointer = SqliteSaver(self._conn)
        self.graph = build_graph(self.ctx, self.checkpointer)

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    def start(
        self,
        dataset_path: str | Path,
        target: str,
        problem_type: str | None = None,
        run_id: str | None = None,
    ) -> RunResult:
        run_id = run_id or uuid.uuid4().hex[:12]
        initial = {
            "run_id": run_id,
            "dataset_path": str(dataset_path),
            "target": target,
            "requested_problem_type": problem_type,
            "status": "running",
        }
        log.info("run_started", run_id=run_id, target=target)
        self.graph.invoke(initial, self._config(run_id))
        return self.get(run_id)

    def resume(self, run_id: str, approved_models: list[str]) -> RunResult:
        current = self.get(run_id)
        if current.status != "awaiting_approval" or not current.pending:
            raise InvalidResume(f"run {run_id} is not waiting for approval (status={current.status})")
        allowed = set(current.pending.get("candidates", []))
        chosen = [m for m in approved_models if m in allowed]
        if not chosen:
            raise InvalidResume(f"approve at least one of {sorted(allowed)}")
        self.graph.invoke(Command(resume={"approved_models": chosen}), self._config(run_id))
        return self.get(run_id)

    def get(self, run_id: str) -> RunResult:
        snap = self.graph.get_state(self._config(run_id))
        if not snap or not snap.values:
            raise RunNotFound(run_id)
        values = dict(snap.values)
        pending = next((i.value for t in snap.tasks for i in getattr(t, "interrupts", ())), None)
        if pending is not None:
            status = "awaiting_approval"
        elif values.get("status") == "failed":
            status = "failed"
        elif not snap.next and values.get("status") == "completed":
            status = "completed"
        else:
            status = values.get("status", "running")
        return RunResult(run_id, status, values, pending, list(values.get("errors", [])))

    def _config(self, run_id: str) -> dict[str, Any]:
        return {
            "configurable": {"thread_id": run_id},
            # Each agent step is followed by a supervisor hop.
            "recursion_limit": 2 * self.settings.max_graph_steps + 5,
        }
