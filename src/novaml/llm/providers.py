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
    """Open-weight models on Groq's free tier, via LangChain structured output.

    Free-tier limits are per model (e.g. 30 RPM / 8K TPM / 200K TPD), so the
    provider walks a fallback chain: when a model is rate limited, retired,
    rejects the request or returns unparseable output, the next model is tried.

    * The chain is filtered once against the account's live model list, so a
      retired model costs nothing instead of a failed request on every call.
    * Tool-calling structured output is used (supported by every Groq chat model).
      When Groq rejects a generated tool call server-side (`tool_use_failed`),
      the rejected arguments are recovered and validated locally: schema
      defaults often make an "invalid" call perfectly usable.
    """

    name = "groq"

    def __init__(self, settings: Settings, chat_factory: Any | None = None, list_models: Any | None = None):
        import groq

        self._groq = groq
        self.models = [settings.resolved_model() or "openai/gpt-oss-120b", *settings.resolved_fallback_models()]
        self.model = self.models[0]
        key = settings.groq_api_key.get_secret_value() if settings.groq_api_key else None

        def default_factory(model: str):
            from langchain_groq import ChatGroq

            return ChatGroq(
                model=model,
                temperature=0,
                api_key=key,
                timeout=settings.llm_timeout_s,
                max_retries=settings.llm_max_retries,
                max_tokens=settings.llm_max_output_tokens,
            )

        def default_list_models() -> set[str]:
            client = groq.Groq(api_key=key, timeout=10, max_retries=1)
            return {m.id for m in client.models.list().data}

        self._factory = chat_factory or default_factory
        self._list_models = list_models or default_list_models
        self._chats: dict[str, Any] = {}
        self._checked = False

    def _chat(self, model: str) -> Any:
        if model not in self._chats:
            self._chats[model] = self._factory(model)
        return self._chats[model]

    def _available_chain(self) -> list[str]:
        if not self._checked:
            self._checked = True
            try:
                live = self._list_models()
            except Exception:  # listing is an optimisation; never block on it
                live = None
            if live:
                kept = [m for m in self.models if m in live]
                if kept:
                    self.models, self.model = kept, kept[0]
        return self.models

    def structured(self, schema: type[T], system: str, user: str) -> tuple[T, Usage]:
        g = self._groq
        spent = Usage()
        failures: list[str] = []
        for model in self._available_chain():
            try:
                out = self._chat(model).with_structured_output(schema, include_raw=True).invoke(
                    [("system", system), ("human", user)]
                )
            except (g.AuthenticationError, g.PermissionDeniedError) as e:
                raise LLMError(f"groq: {type(e).__name__}: check GROQ_API_KEY", usage=spent) from e
            except g.BadRequestError as e:
                recovered = _recover_tool_call(e, schema)
                if recovered is not None:
                    spent.model = model
                    return recovered, spent
                failures.append(f"{model}: {_error_code(e)}")
                continue
            except Exception as e:
                # Rate limits (429), a retired model (404) and 5xx/timeouts are
                # model-specific; each model has its own quota.
                failures.append(f"{model}: {_error_code(e)}")
                continue
            meta = getattr(out.get("raw"), "usage_metadata", None) or {}
            spent.input_tokens += meta.get("input_tokens", 0)
            spent.output_tokens += meta.get("output_tokens", 0)
            if out.get("parsing_error") or out.get("parsed") is None:
                failures.append(f"{model}: unparseable output")
                continue
            spent.model = model
            return out["parsed"], spent
        raise LLMError("groq: all models failed (" + "; ".join(failures) + ")", usage=spent)


def _error_code(e: Exception) -> str:
    body = getattr(e, "body", None)
    code = body.get("error", {}).get("code") if isinstance(body, dict) else None
    return f"{type(e).__name__}({code})" if code else type(e).__name__


def _recover_tool_call(e: Exception, schema: type[T]) -> T | None:
    """Validate the arguments Groq rejected server-side (`tool_use_failed`) ourselves."""
    import json

    from pydantic import ValidationError

    body = getattr(e, "body", None)
    err = body.get("error", {}) if isinstance(body, dict) else {}
    if err.get("code") != "tool_use_failed" or not err.get("failed_generation"):
        return None
    try:
        gen = json.loads(err["failed_generation"])
        args = gen.get("arguments", gen) if isinstance(gen, dict) else None
        if isinstance(args, str):
            args = json.loads(args)
        return schema.model_validate(args)
    except (ValueError, TypeError, ValidationError):
        return None


def build_provider(settings: Settings) -> LLMProvider | None:
    """Returns None when running in deterministic (no-LLM) mode."""
    provider = settings.effective_provider()
    if provider == "groq":
        return GroqProvider(settings)
    if provider == "anthropic":
        return AnthropicProvider(settings)
    return None
