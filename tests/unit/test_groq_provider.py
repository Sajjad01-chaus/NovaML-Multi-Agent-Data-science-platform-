import os
from types import SimpleNamespace

import groq
import httpx
import pytest
from pydantic import BaseModel, SecretStr

from novaml.config import Settings
from novaml.llm.base import LLMError
from novaml.llm.gateway import LLMGateway
from novaml.llm.providers import GroqProvider, build_provider


class Out(BaseModel):
    answer: str


def api_error(cls, status):
    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return cls(f"HTTP {status}", response=httpx.Response(status, request=req), body=None)


class FakeChat:
    """Mimics ChatGroq.with_structured_output(..., include_raw=True).invoke()."""

    def __init__(self, model, behaviour, log):
        self.model, self.behaviour, self.log = model, behaviour, log

    def with_structured_output(self, schema, include_raw):
        assert include_raw is True
        return self

    def invoke(self, messages):
        self.log.append(self.model)
        b = self.behaviour
        if isinstance(b, Exception):
            raise b
        raw = SimpleNamespace(usage_metadata={"input_tokens": 40, "output_tokens": 10})
        if b == "garbage":
            return {"raw": raw, "parsed": None, "parsing_error": ValueError("bad json")}
        return {"raw": raw, "parsed": Out(answer=f"from {self.model}"), "parsing_error": None}


def groq_settings(**kw):
    return Settings(_env_file=None, llm_provider="groq", groq_api_key=SecretStr("gsk-test"), **kw)


def provider(behaviours, **kw):
    log: list[str] = []
    p = GroqProvider(groq_settings(**kw), chat_factory=lambda m: FakeChat(m, behaviours.get(m, "ok"), log))
    return p, log


def test_defaults_target_free_tier():
    s = groq_settings()
    assert s.resolved_model() == "openai/gpt-oss-120b"
    assert s.resolved_fallback_models()[0] == "llama-3.3-70b-versatile"
    assert s.run_token_budget <= 200_000  # well inside the free daily quota
    assert s.resolved_prices() == (0.0, 0.0)


def test_auto_provider_uses_groq_only_when_key_present(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert Settings(_env_file=None).effective_provider() == "none"
    assert build_provider(Settings(_env_file=None)) is None
    monkeypatch.setenv("GROQ_API_KEY", "gsk-x")
    s = Settings(_env_file=None)
    assert s.effective_provider() == "groq"
    assert isinstance(build_provider(s), GroqProvider)


def test_primary_model_answers():
    p, log = provider({})
    value, usage = p.structured(Out, "sys", "user")
    assert value.answer == "from openai/gpt-oss-120b"
    assert usage.total == 50 and usage.model == "openai/gpt-oss-120b"
    assert log == ["openai/gpt-oss-120b"]


def test_rate_limited_model_falls_back_to_next():
    p, log = provider({"openai/gpt-oss-120b": api_error(groq.RateLimitError, 429)})
    value, usage = p.structured(Out, "s", "u")
    assert value.answer == "from llama-3.3-70b-versatile" and usage.model == "llama-3.3-70b-versatile"
    assert log == ["openai/gpt-oss-120b", "llama-3.3-70b-versatile"]


def test_retired_model_and_bad_output_fall_through():
    p, log = provider(
        {
            "openai/gpt-oss-120b": api_error(groq.NotFoundError, 404),
            "llama-3.3-70b-versatile": "garbage",
        }
    )
    value, usage = p.structured(Out, "s", "u")
    assert value.answer == "from openai/gpt-oss-20b"
    assert usage.total == 100  # the garbage attempt's tokens are still counted


def test_auth_error_stops_immediately():
    p, log = provider({"openai/gpt-oss-120b": api_error(groq.AuthenticationError, 401)})
    with pytest.raises(LLMError, match="GROQ_API_KEY"):
        p.structured(Out, "s", "u")
    assert log == ["openai/gpt-oss-120b"]


def test_all_models_exhausted_falls_back_to_policy_and_counts_tokens():
    garbage = dict.fromkeys(["openai/gpt-oss-120b", "llama-3.3-70b-versatile", "openai/gpt-oss-20b", "llama-3.1-8b-instant"], "garbage")
    p, _ = provider(garbage)
    gw = LLMGateway(p, groq_settings())
    d = gw.decide(agent="t", schema=Out, system="s", user="u", policy=lambda: Out(answer="policy"))
    assert d.source == "policy" and "all models failed" in d.record["error"]
    assert d.record["input_tokens"] + d.record["output_tokens"] == 200
    assert gw.tokens_spent == 200


def test_real_chatgroq_is_configured_without_network():
    p = GroqProvider(groq_settings(llm_timeout_s=12))
    chat = p._chat("openai/gpt-oss-120b")
    assert chat.model_name == "openai/gpt-oss-120b"
    assert chat.temperature < 1e-6  # ChatGroq maps 0 to 1e-8 internally
    assert chat.request_timeout == 12
    assert chat.groq_api_key.get_secret_value() == "gsk-test"


@pytest.mark.live
@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="needs GROQ_API_KEY")
def test_live_groq_structured_output():
    p = GroqProvider(Settings(llm_provider="groq"))
    value, usage = p.structured(Out, "Answer in one word.", "What colour is the sky on a clear day?")
    assert value.answer and usage.total > 0 and usage.model
