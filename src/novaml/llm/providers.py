"""Concrete LLM providers. Imports are lazy so optional SDKs stay optional."""

from __future__ import annotations

from typing import Any

from novaml.config import Settings
from novaml.llm.base import LLMError, LLMProvider, T, Usage


class AnthropicProvider:
    """Claude via the official Anthropic SDK, using native structured outputs."""

    name = "anthropic"

    def __init__(self, settings: Settings, client: Any | None = None):
        import anthropic

        self._anthropic = anthropic
        self.model = settings.resolved_model() or "claude-opus-5-5"
        self.max_tokens = settings.llm_max_output_tokens
        self.effort = settings.llm_effort
        self.server_fallbacks = settings.llm_server_fallbacks
        key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else None
        self.client = client or anthropic.Anthropic(
            api_key=key, timeout=settings.llm_timeout_s, max_retries=settings.llm_max_retries
        )

    def structured(self, schema: type[T], system: str, user: str) -> tuple[T, Usage]:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=schema,
            output_config={"effort": self.effort},
        )
        try:
            if self.server_fallbacks:
                # On a safety decline the API re-runs the request on a fallback model.
                resp = self.client.beta.messages.parse(
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs
                )
            else:
                resp = self.client.messages.parse(**kwargs)
        except self._anthropic.APIError as e:
            raise LLMError(f"anthropic: {type(e).__name__}: {e}") from e

        usage = Usage(resp.usage.input_tokens, resp.usage.output_tokens)
        if resp.stop_reason == "refusal":
            raise LLMError("anthropic: request refused")
        if resp.stop_reason == "max_tokens":
            raise LLMError("anthropic: output truncated at max_tokens")
        parsed = getattr(resp, "parsed_output", None)
        if parsed is None:
            raise LLMError("anthropic: no structured output returned")
        return parsed, usage


class GroqProvider:
    """Open-weight models on Groq (free tier), via LangChain's structured output."""

    name = "groq"

    def __init__(self, settings: Settings):
        from langchain_groq import ChatGroq

        key = settings.groq_api_key.get_secret_value() if settings.groq_api_key else None
        self.model = settings.resolved_model() or "llama-3.3-70b-versatile"
        self.chat = ChatGroq(
            model=self.model,
            temperature=0,
            api_key=key,
            timeout=settings.llm_timeout_s,
            max_retries=settings.llm_max_retries,
            max_tokens=settings.llm_max_output_tokens,
        )

    def structured(self, schema: type[T], system: str, user: str) -> tuple[T, Usage]:
        try:
            out = self.chat.with_structured_output(schema, include_raw=True).invoke(
                [("system", system), ("human", user)]
            )
        except Exception as e:  # langchain wraps provider errors inconsistently
            raise LLMError(f"groq: {type(e).__name__}: {e}") from e
        meta = getattr(out.get("raw"), "usage_metadata", None) or {}
        usage = Usage(meta.get("input_tokens", 0), meta.get("output_tokens", 0))
        if out.get("parsing_error") or out.get("parsed") is None:
            raise LLMError(f"groq: unparseable output: {out.get('parsing_error')}")
        return out["parsed"], usage


def build_provider(settings: Settings) -> LLMProvider | None:
    """Returns None when running in deterministic (no-LLM) mode."""
    if settings.llm_provider == "anthropic":
        return AnthropicProvider(settings)
    if settings.llm_provider == "groq":
        return GroqProvider(settings)
    return None
