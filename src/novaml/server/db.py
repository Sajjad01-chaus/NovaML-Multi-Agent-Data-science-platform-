"""Run registry, job queue and event log on one relational database.

Postgres in production, SQLite for dev/tests, one code path (SQLAlchemy Core).

Why a table instead of Celery/Redis: the queue lives in the same database as the
run registry, so creating a run and enqueuing its job is one transaction (no
"run exists but job lost" window), and there is one less service to operate.
`FOR UPDATE SKIP LOCKED` lets many workers claim jobs concurrently without
blocking each other. Leases make crashes recoverable: a job whose worker stops
heart-beating is re-queued and resumed from the run's last checkpoint.
"""

from __future__ import annotations

import json
import secrets
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Engine,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    event,
    func,
    insert,
    select,
    update,
)
from sqlalchemy.exc import IntegrityError

metadata = MetaData()

runs = Table(
    "runs",
    metadata,
    Column("id", String(32), primary_key=True),
    Column("status", String(32), nullable=False),  # queued|running|awaiting_approval|completed|failed|cancelled
    Column("dataset_id", String(64), nullable=False),
    Column("target", String(256), nullable=False),
    Column("problem_type", String(32)),
    Column("idempotency_key", String(128), unique=True),
    Column("cancel_requested", Integer, nullable=False, default=0),
    Column("summary", JSON),
    Column("error", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)

jobs = Table(
    "jobs",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(32), ForeignKey("runs.id"), nullable=False),
    Column("kind", String(32), nullable=False),  # start | resume
    Column("payload", JSON, nullable=False),
    Column("status", String(16), nullable=False),  # queued | running | done | failed | dead
    Column("attempts", Integer, nullable=False, default=0),
    Column("max_attempts", Integer, nullable=False),
    Column("lease_owner", String(64)),
    Column("lease_expires_at", DateTime(timezone=True)),
    Column("last_error", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Index("ix_jobs_claim", "status", "created_at"),
)

run_events = Table(
    "run_events",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(32), ForeignKey("runs.id"), nullable=False),
    Column("kind", String(32), nullable=False),
    Column("payload", JSON, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Index("ix_run_events_run", "run_id", "id"),
)


def now() -> datetime:
    return datetime.now(UTC)


def _aware(d: datetime | None) -> datetime | None:
    # SQLite drops tzinfo; everything we store is UTC.
    return d.replace(tzinfo=UTC) if d is not None and d.tzinfo is None else d


def make_engine(url: str) -> Engine:
    if url.startswith("sqlite"):
        db_path = url.split(":///", 1)[-1]
        if db_path and db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(url, connect_args={"check_same_thread": False, "timeout": 30})

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(conn, _):  # concurrent API + worker access
            conn.isolation_level = None  # let SQLAlchemy (below) own BEGIN
            cur = conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA busy_timeout=30000")
            cur.close()

        @event.listens_for(engine, "begin")
        def _sqlite_begin_immediate(conn):
            # Take the write lock up front so claim() can't race between SELECT and UPDATE.
            conn.exec_driver_sql("BEGIN IMMEDIATE")

        return engine
    return create_engine(url, pool_pre_ping=True, pool_size=10, max_overflow=10)


class Database:
    def __init__(self, url: str):
        self.engine = make_engine(url)
        self.is_postgres = self.engine.dialect.name == "postgresql"
        metadata.create_all(self.engine)

    def dispose(self) -> None:
        self.engine.dispose()

    # ---------------------------------------------------------------- runs
    def create_run(
        self,
        dataset_id: str,
        target: str,
        problem_type: str | None,
        idempotency_key: str | None = None,
        max_attempts: int = 3,
        auto_approve: bool = False,
    ) -> tuple[dict[str, Any], bool]:
        """Creates a run and its start job atomically. Returns (run, created)."""
        if idempotency_key:
            existing = self._run_by_key(idempotency_key)
            if existing:
                return existing, False
        run_id = secrets.token_hex(6)
        t = now()
        try:
            with self.engine.begin() as c:
                c.execute(
                    insert(runs).values(
                        id=run_id, status="queued", dataset_id=dataset_id, target=target,
                        problem_type=problem_type, idempotency_key=idempotency_key,
                        cancel_requested=0, created_at=t, updated_at=t,
                    )
                )
                self._enqueue(c, run_id, "start", {"dataset_id": dataset_id, "target": target, "problem_type": problem_type, "auto_approve": auto_approve}, max_attempts)
        except IntegrityError:
            # Two requests with the same idempotency key raced; the other one won.
            existing = self._run_by_key(idempotency_key) if idempotency_key else None
            if existing:
                return existing, False
            raise
        return self.get_run(run_id), True

    def _run_by_key(self, key: str) -> dict[str, Any] | None:
        with self.engine.connect() as c:
            row = c.execute(select(runs).where(runs.c.idempotency_key == key)).mappings().first()
        return _run(row)

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self.engine.connect() as c:
            row = c.execute(select(runs).where(runs.c.id == run_id)).mappings().first()
        return _run(row)

    def list_runs(self, limit: int = 50, status: str | None = None) -> list[dict[str, Any]]:
        q = select(runs).order_by(runs.c.created_at.desc()).limit(limit)
        if status:
            q = q.where(runs.c.status == status)
        with self.engine.connect() as c:
            return [_run(r) for r in c.execute(q).mappings()]

    def set_run(self, run_id: str, **values: Any) -> None:
        with self.engine.begin() as c:
            c.execute(update(runs).where(runs.c.id == run_id).values(**values, updated_at=now()))

    def transition(self, run_id: str, from_status: str, to_status: str) -> bool:
        """Compare-and-set on run status; False if the run wasn't in `from_status`."""
        with self.engine.begin() as c:
            res = c.execute(
                update(runs).where(runs.c.id == run_id, runs.c.status == from_status).values(status=to_status, updated_at=now())
            )
            return res.rowcount == 1

    def request_approval_resume(self, run_id: str, approved: list[str], max_attempts: int = 3) -> bool:
        """awaiting_approval -> queued + resume job, atomically (double-submits get False)."""
        with self.engine.begin() as c:
            res = c.execute(
                update(runs)
                .where(runs.c.id == run_id, runs.c.status == "awaiting_approval")
                .values(status="queued", updated_at=now())
            )
            if res.rowcount != 1:
                return False
            self._enqueue(c, run_id, "resume", {"approved_models": approved}, max_attempts)
            return True

    def request_cancel(self, run_id: str) -> bool:
        with self.engine.begin() as c:
            run = c.execute(select(runs.c.status).where(runs.c.id == run_id)).first()
            if run is None or run.status in ("completed", "failed", "cancelled"):
                return False
            c.execute(update(runs).where(runs.c.id == run_id).values(cancel_requested=1, updated_at=now()))
            if run.status in ("queued", "awaiting_approval"):
                # Nothing is executing: cancel immediately and drop queued work.
                c.execute(update(runs).where(runs.c.id == run_id).values(status="cancelled"))
                c.execute(update(jobs).where(jobs.c.run_id == run_id, jobs.c.status == "queued").values(status="failed", last_error="cancelled", updated_at=now()))
            return True

    def cancel_requested(self, run_id: str) -> bool:
        with self.engine.connect() as c:
            v = c.execute(select(runs.c.cancel_requested).where(runs.c.id == run_id)).scalar()
        return bool(v)

    # ---------------------------------------------------------------- jobs
    def _enqueue(self, c, run_id: str, kind: str, payload: dict[str, Any], max_attempts: int) -> None:
        t = now()
        c.execute(
            insert(jobs).values(
                run_id=run_id, kind=kind, payload=payload, status="queued",
                attempts=0, max_attempts=max_attempts, created_at=t, updated_at=t,
            )
        )

    def claim(self, owner: str, lease_s: float) -> dict[str, Any] | None:
        """Atomically take the oldest queued job and lease it to `owner`."""
        self.requeue_expired()
        t = now()
        with self.engine.begin() as c:
            q = select(jobs.c.id).where(jobs.c.status == "queued").order_by(jobs.c.created_at, jobs.c.id).limit(1)
            if self.is_postgres:
                q = q.with_for_update(skip_locked=True)
            job_id = c.execute(q).scalar()
            if job_id is None:
                return None
            c.execute(
                update(jobs)
                .where(jobs.c.id == job_id)
                .values(status="running", attempts=jobs.c.attempts + 1, lease_owner=owner,
                        lease_expires_at=t + timedelta(seconds=lease_s), updated_at=t)
            )
            row = c.execute(select(jobs).where(jobs.c.id == job_id)).mappings().first()
        return _job(row)

    def heartbeat(self, job_id: int, owner: str, lease_s: float) -> bool:
        with self.engine.begin() as c:
            res = c.execute(
                update(jobs)
                .where(jobs.c.id == job_id, jobs.c.lease_owner == owner, jobs.c.status == "running")
                .values(lease_expires_at=now() + timedelta(seconds=lease_s), updated_at=now())
            )
            return res.rowcount == 1

    def complete(self, job_id: int, owner: str) -> None:
        with self.engine.begin() as c:
            c.execute(update(jobs).where(jobs.c.id == job_id, jobs.c.lease_owner == owner).values(status="done", lease_owner=None, updated_at=now()))

    def fail(self, job_id: int, owner: str, error: str, retry: bool = True) -> str:
        """Retry (re-queue) while attempts remain, else mark dead. Returns the new status."""
        with self.engine.begin() as c:
            job = c.execute(select(jobs).where(jobs.c.id == job_id)).mappings().first()
            if job is None:
                return "missing"
            status = "queued" if retry and job["attempts"] < job["max_attempts"] else "dead"
            c.execute(
                update(jobs).where(jobs.c.id == job_id, jobs.c.lease_owner == owner)
                .values(status=status, lease_owner=None, lease_expires_at=None, last_error=error[:2000], updated_at=now())
            )
            return status

    def requeue_expired(self) -> list[int]:
        """Jobs whose worker died (lease expired) go back to the queue, or to dead."""
        t = now()
        moved = []
        with self.engine.begin() as c:
            expired = c.execute(
                select(jobs.c.id, jobs.c.attempts, jobs.c.max_attempts, jobs.c.run_id)
                .where(jobs.c.status == "running", jobs.c.lease_expires_at < t)
            ).all()
            for j in expired:
                dead = j.attempts >= j.max_attempts
                c.execute(
                    update(jobs).where(jobs.c.id == j.id, jobs.c.status == "running")
                    .values(status="dead" if dead else "queued", lease_owner=None, lease_expires_at=None,
                            last_error="lease expired (worker lost)", updated_at=t)
                )
                if dead:
                    c.execute(update(runs).where(runs.c.id == j.run_id).values(status="failed", error="worker lost too many times", updated_at=t))
                moved.append(j.id)
        return moved

    def get_job(self, job_id: int) -> dict[str, Any] | None:
        with self.engine.connect() as c:
            return _job(c.execute(select(jobs).where(jobs.c.id == job_id)).mappings().first())

    def jobs_for(self, run_id: str) -> list[dict[str, Any]]:
        with self.engine.connect() as c:
            return [_job(r) for r in c.execute(select(jobs).where(jobs.c.run_id == run_id).order_by(jobs.c.id)).mappings()]

    def queue_depth(self) -> dict[str, int]:
        with self.engine.connect() as c:
            rows = c.execute(select(jobs.c.status, func.count()).group_by(jobs.c.status)).all()
        return {r[0]: int(r[1]) for r in rows}

    # ---------------------------------------------------------------- events
    def emit(self, run_id: str, kind: str, payload: dict[str, Any]) -> None:
        with self.engine.begin() as c:
            c.execute(insert(run_events).values(run_id=run_id, kind=kind, payload=json.loads(json.dumps(payload, default=str)), created_at=now()))

    def events_after(self, run_id: str, after_id: int = 0, limit: int = 200) -> list[dict[str, Any]]:
        with self.engine.connect() as c:
            rows = c.execute(
                select(run_events).where(run_events.c.run_id == run_id, run_events.c.id > after_id).order_by(run_events.c.id).limit(limit)
            ).mappings()
            return [dict(r) for r in rows]

    def ping(self) -> bool:
        with self.engine.connect() as c:
            c.exec_driver_sql("SELECT 1")
        return True


def _run(row) -> dict[str, Any] | None:
    if row is None:
        return None
    d = dict(row)
    d["created_at"], d["updated_at"] = _aware(d["created_at"]), _aware(d["updated_at"])
    d["cancel_requested"] = bool(d["cancel_requested"])
    return d


def _job(row) -> dict[str, Any] | None:
    if row is None:
        return None
    d = dict(row)
    for k in ("created_at", "updated_at", "lease_expires_at"):
        d[k] = _aware(d[k])
    return d
