"""Typed runtime configuration, loaded from environment variables / `.env`.

Every knob that changes behaviour in production lives here so it can be set per
environment without code changes. Secrets are `SecretStr` so they never end up
in logs or reprs.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NOVAML_", env_file=".env", extra="ignore")

    # --- Storage -------------------------------------------------------------
    data_dir: Path = Path("var")
    """Root for run artifacts and the checkpoint database."""

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
    llm_provider: Literal["anthropic", "groq", "none"] = "none"
    """`none` runs every agent on its deterministic policy (no API key needed)."""
    llm_model: str | None = None
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 2
    llm_max_output_tokens: int = 4096
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    """Anthropic effort level; agent decisions here are small, so low keeps cost down."""
    llm_server_fallbacks: bool = True
    """Anthropic server-side refusal fallback (routes a declined request to another model)."""
    run_token_budget: int = 200_000
    """Total LLM tokens a single run may spend before agents degrade to policies."""
    input_cost_per_mtok: float | None = None
    output_cost_per_mtok: float | None = None

    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    groq_api_key: SecretStr | None = Field(default=None, validation_alias="GROQ_API_KEY")

    # --- Sandbox -------------------------------------------------------------
    sandbox_timeout_s: float = 30.0
    sandbox_max_output_chars: int = 4000
    sandbox_memory_mb: int = 1024

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    @property
    def checkpoint_path(self) -> Path:
        return self.data_dir / "checkpoints.sqlite"

    def resolved_model(self) -> str | None:
        if self.llm_model:
            return self.llm_model
        return {
            "anthropic": "claude-opus-5-5",
            "groq": "llama-3.3-70b-versatile",
        }.get(self.llm_provider)

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
