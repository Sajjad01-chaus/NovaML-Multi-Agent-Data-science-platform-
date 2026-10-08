"""Typed runtime configuration, loaded from environment variables / `.env`.

Every knob that changes behaviour in production lives here so it can be set per
environment without code changes. Secrets are `SecretStr` so they never end up
in logs or reprs.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NOVAML_", env_file=".env", extra="ignore")

    # --- Storage -------------------------------------------------------------
    data_dir: Path = Path("var")
    """Root for run artifacts and the checkpoint database."""

    database_url: str | None = None
    """Run registry, job queue and checkpoints. `postgresql://...` in production;
    defaults to SQLite files under `data_dir` (single node / dev)."""
    artifact_backend: Literal["local", "s3"] = "local"
    s3_bucket: str | None = None
    s3_endpoint_url: str | None = None
    """Set for MinIO / R2 / other S3-compatible stores; credentials come from the
    standard AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY environment variables."""
    s3_region: str | None = None

    # --- Workers ---------------------------------------------------------------
    worker_lease_s: float = 120.0
    """A claimed job is re-queued if its worker stops heart-beating for this long."""
    worker_heartbeat_s: float = 15.0
    worker_poll_s: float = 1.0
    job_max_attempts: int = 3

    # --- Ingestion guardrails ------------------------------------------------
    max_upload_mb: int = 200
    max_rows: int = 2_000_000
    min_rows: int = 20
    allowed_extensions: tuple[str, ...] = (".csv", ".xlsx", ".xls", ".parquet")

    # --- ML ------------------------------------------------------------------
    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = 42
    n_jobs: int = 1
    tuning_iterations: int = 10
    """RandomizedSearchCV iterations when the critic asks for tuning."""

    # --- Agent loop ----------------------------------------------------------
    max_graph_steps: int = 40
    """Hard ceiling on supervisor hops, guards against routing loops."""
    max_improvement_rounds: int = 2
    """How many times the critic may send the run back for another attempt."""
    min_improvement_over_baseline: float = 0.02
    auto_approve: bool = False
    """Skip the human-in-the-loop model approval (batch / eval mode)."""
    analyst_max_steps: int = 4
    """Max code-execution steps the analyst agent may take."""

    # --- LLM -----------------------------------------------------------------
    llm_provider: Literal["auto", "groq", "anthropic", "none"] = "auto"
    """`auto` uses Groq when GROQ_API_KEY is set, otherwise deterministic policies
    (`none`: no API key needed)."""
    llm_model: str | None = None
    llm_fallback_models: list[str] | None = None
    """Tried in order when the primary model is rate limited or unavailable."""
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 2
    llm_max_output_tokens: int = 4096
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    """Anthropic effort level; agent decisions here are small, so low keeps cost down."""
    llm_server_fallbacks: bool = True
    """Anthropic server-side refusal fallback (routes a declined request to another model)."""
    run_token_budget: int = 60_000
    """Total LLM tokens a single run may spend before agents degrade to policies.
    Sized for Groq's free tier (~200K tokens/day per model): a few runs a day."""
    input_cost_per_mtok: float | None = None
    output_cost_per_mtok: float | None = None

    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    groq_api_key: SecretStr | None = Field(default=None, validation_alias="GROQ_API_KEY")

    @field_validator("anthropic_api_key", "groq_api_key", mode="before")
    @classmethod
    def _blank_key_is_none(cls, v):
        # `GROQ_API_KEY=` left empty in .env must mean "no key", not an empty key.
        if isinstance(v, SecretStr):
            v = v.get_secret_value()
        return v.strip() or None if isinstance(v, str) else v

    # --- Sandbox -------------------------------------------------------------
    sandbox_timeout_s: float = 30.0
    sandbox_max_output_chars: int = 2000
    sandbox_memory_mb: int = 1024

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    @property
    def checkpoint_path(self) -> Path:
        return self.data_dir / "checkpoints.sqlite"

    @property
    def is_postgres(self) -> bool:
        return bool(self.database_url and self.database_url.startswith(("postgres://", "postgresql")))

    def sqlalchemy_url(self) -> str:
        if not self.database_url:
            return f"sqlite:///{(self.data_dir / 'novaml.db').as_posix()}"
        url = self.database_url
        if url.startswith("postgres://"):
            url = "postgresql://" + url[len("postgres://") :]
        if url.startswith("postgresql://"):
            url = "postgresql+psycopg://" + url[len("postgresql://") :]
        return url

    def libpq_url(self) -> str:
        """Plain libpq URL for psycopg / the LangGraph Postgres checkpointer."""
        return self.sqlalchemy_url().replace("postgresql+psycopg://", "postgresql://", 1)

    def effective_provider(self) -> str:
        if self.llm_provider != "auto":
            return self.llm_provider
        return "groq" if self.groq_api_key else "none"

    def resolved_model(self) -> str | None:
        if self.llm_model:
            return self.llm_model
        return {
            "groq": "openai/gpt-oss-120b",
            "anthropic": "claude-opus-5-5",
        }.get(self.effective_provider())

    def resolved_fallback_models(self) -> list[str]:
        if self.llm_fallback_models is not None:
            return [m for m in self.llm_fallback_models if m != self.resolved_model()]
        if self.effective_provider() == "groq":
            # Separate free-tier quotas per model; order = quality, then speed.
            return [m for m in ("qwen/qwen3.8-27b", "openai/gpt-oss-20b") if m != self.resolved_model()]
        return []

    def resolved_prices(self) -> tuple[float, float]:
        """USD per million (input, output) tokens, used for cost accounting."""
        defaults = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0)}
        din, dout = defaults.get(self.resolved_model() or "", (0.0, 0.0))
        return (
            self.input_cost_per_mtok if self.input_cost_per_mtok is not None else din,
            self.output_cost_per_mtok if self.output_cost_per_mtok is not None else dout,
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
