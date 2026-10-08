"""Run service: the single entry point used by the CLI, the UI, the API and workers.

Runs are keyed by `run_id` (= LangGraph thread id) and checkpointed after every
agent: to Postgres when `NOVAML_DATABASE_URL` points at one, otherwise to SQLite on
disk. Consequences:
* a run paused for human approval survives restarts and resumes in any process;
* a worker that crashes mid-run loses at most the agent it was executing: calling
  `start()` again with the same `run_id` continues from the last checkpoint.
"""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langgraph.types import Command

from novaml.agents.base import AgentContext, EventSink
from novaml.config import Settings, get_settings
from novaml.graph import build_graph
from novaml.llm.base import LLMProvider
from novaml.llm.providers import build_provider
from novaml.log import configure_logging, get_logger
from novaml.sandbox.executor import Sandbox

log = get_logger(__name__)

TERMINAL = ("completed", "failed", "cancelled")


class RunNotFound(KeyError):
    pass


class InvalidResume(ValueError):
    pass


@dataclass
class RunResult:
    run_id: str
    status: str  # running | awaiting_approval | completed | failed | cancelled
    state: dict[str, Any]
    pending: dict[str, Any] | None = None
    errors: list[str] = field(default_factory=list)


def open_checkpointer(settings: Settings) -> tuple[Any, Callable[[], None]]:
    """Returns (checkpointer, close)."""
    if settings.is_postgres:
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool

        pool = ConnectionPool(
            settings.libpq_url(),
            min_size=1,
            max_size=8,
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
            open=True,
        )
        saver = PostgresSaver(pool)
        saver.setup()  # idempotent: creates/migrates checkpoint tables
        return saver, pool.close

    from langgraph.checkpoint.sqlite import SqliteSaver

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.checkpoint_path, check_same_thread=False)
    return SqliteSaver(conn), conn.close


class NovaML:
    def __init__(
        self,
        settings: Settings | None = None,
        provider: LLMProvider | None | str = "auto",
        sandbox: Sandbox | None = None,
        events: EventSink | None = None,
        cancel_check: Callable[[str], bool] | None = None,
    ):
        self.settings = settings or get_settings()
        configure_logging()
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        if provider == "auto":
            provider = build_provider(self.settings)
        self.ctx = AgentContext(self.settings, provider, sandbox, events, cancel_check)  # type: ignore[arg-type]
        self.checkpointer, self._close = open_checkpointer(self.settings)
        self.graph = build_graph(self.ctx, self.checkpointer)

    def close(self) -> None:
        self._close()

    # ------------------------------------------------------------------
    def start(
        self,
        dataset_path: str | Path,
        target: str,
        problem_type: str | None = None,
        run_id: str | None = None,
        auto_approve: bool = False,
    ) -> RunResult:
        run_id = run_id or uuid.uuid4().hex[:12]
        existing = self._snapshot(run_id)
        if existing is not None:
            current = self.get(run_id)
            if current.status in TERMINAL or current.status == "awaiting_approval":
                return current  # idempotent: a retried job doesn't redo finished work
            log.info("run_recovered", run_id=run_id, next=list(existing.next))
            self.graph.invoke(None, self._config(run_id))  # continue from last checkpoint
            return self.get(run_id)

        initial = {
            "run_id": run_id,
            "dataset_path": str(dataset_path),
            "target": target,
            "requested_problem_type": problem_type,
            "auto_approve": auto_approve,
            "status": "running",
        }
        log.info("run_started", run_id=run_id, target=target)
        self.graph.invoke(initial, self._config(run_id))
        return self.get(run_id)

    def resume(self, run_id: str, approved_models: list[str]) -> RunResult:
        current = self.get(run_id)
        if current.status != "awaiting_approval" or not current.pending:
            raise InvalidResume(f"run {run_id} is not waiting for approval (status={current.status})")
        chosen = self.validate_approval(current, approved_models)
        self.graph.invoke(Command(resume={"approved_models": chosen}), self._config(run_id))
        return self.get(run_id)

    @staticmethod
    def validate_approval(current: RunResult, approved_models: list[str]) -> list[str]:
        allowed = set((current.pending or {}).get("candidates", []))
        chosen = [m for m in dict.fromkeys(approved_models) if m in allowed]
        if not chosen:
            raise InvalidResume(f"approve at least one of {sorted(allowed)}")
        return chosen

    def get(self, run_id: str) -> RunResult:
        snap = self._snapshot(run_id)
        if snap is None:
            raise RunNotFound(run_id)
        values = dict(snap.values)
        pending = next((i.value for t in snap.tasks for i in getattr(t, "interrupts", ())), None)
        if pending is not None:
            status = "awaiting_approval"
        elif values.get("status") in ("failed", "cancelled"):
            status = values["status"]
        elif not snap.next and values.get("status") == "completed":
            status = "completed"
        else:
            status = "running"
        return RunResult(run_id, status, values, pending, list(values.get("errors", [])))

    def _snapshot(self, run_id: str):
        snap = self.graph.get_state(self._config(run_id))
        return snap if snap and snap.values else None

    def _config(self, run_id: str) -> dict[str, Any]:
        return {
            "configurable": {"thread_id": run_id},
            # Each agent step is followed by a supervisor hop.
            "recursion_limit": 2 * self.settings.max_graph_steps + 5,
        }
