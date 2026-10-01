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


CHAIN = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"]


def provider(behaviours, live_models=None, **kw):
    log: list[str] = []
    p = GroqProvider(
        groq_settings(**kw),
        chat_factory=lambda m: FakeChat(m, behaviours.get(m, "ok"), log),
        list_models=lambda: live_models,  # no network in unit tests
    )
    return p, log


def test_defaults_target_free_tier():
    s = groq_settings()
    assert s.resolved_model() == "openai/gpt-oss-120b"
    assert s.resolved_fallback_models() == CHAIN[1:]
    assert s.run_token_budget <= 200_000  # well inside the free daily quota
    assert s.resolved_prices() == (0.0, 0.0)


def test_auto_provider_uses_groq_only_when_key_present(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert Settings(_env_file=None).effective_provider() == "none"
    monkeypatch.setenv("GROQ_API_KEY", "  ")  # blank line in .env
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
    assert value.answer == "from qwen/qwen3.8-27b" and usage.model == "qwen/qwen3.8-27b"
    assert log == CHAIN[:2]


def test_retired_model_and_bad_output_fall_through():
    p, log = provider(
        {
            "openai/gpt-oss-120b": api_error(groq.NotFoundError, 404),
            "qwen/qwen3.8-27b": "garbage",
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
    garbage = dict.fromkeys(CHAIN, "garbage")
    p, _ = provider(garbage)
    gw = LLMGateway(p, groq_settings())
    d = gw.decide(agent="t", schema=Out, system="s", user="u", policy=lambda: Out(answer="policy"))
    assert d.source == "policy" and "all models failed" in d.record["error"]
    assert d.record["input_tokens"] + d.record["output_tokens"] == 150
    assert gw.tokens_spent == 150


def test_chain_is_filtered_to_models_the_account_has():
    p, log = provider({}, live_models={"openai/gpt-oss-20b", "whisper-large-v3"})
    value, _ = p.structured(Out, "s", "u")
    assert value.answer == "from openai/gpt-oss-20b" and log == ["openai/gpt-oss-20b"]
    assert p.model == "openai/gpt-oss-20b"


def test_model_listing_failure_keeps_configured_chain():
    def boom():
        raise RuntimeError("network down")

    p = GroqProvider(groq_settings(), chat_factory=lambda m: FakeChat(m, "ok", []), list_models=boom)
    assert p.structured(Out, "s", "u")[1].model == CHAIN[0]


def tool_use_failed(generation: str):
    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    body = {"error": {"code": "tool_use_failed", "message": "missing properties", "failed_generation": generation}}
    return groq.BadRequestError("tool_use_failed", response=httpx.Response(400, request=req), body=body)


class Step(BaseModel):
    thought: str = ""
    code: str


def test_server_rejected_tool_call_is_recovered_locally():
    """Groq rejects calls missing a field even when our schema has a default for it."""
    gen = '{"name": "Step", "arguments": {"code": "print(df.shape)"}}'
    p, log = provider({CHAIN[0]: tool_use_failed(gen)})
    value, usage = p.structured(Step, "s", "u")
    assert value.code == "print(df.shape)" and usage.model == CHAIN[0] and log == [CHAIN[0]]


def test_unrecoverable_tool_call_falls_through():
    p, log = provider({CHAIN[0]: tool_use_failed('{"name": "Step", "arguments": {}}')})
    value, usage = p.structured(Out, "s", "u")
    assert usage.model == CHAIN[1]


def test_real_chatgroq_is_configured_without_network():
    p = GroqProvider(groq_settings(llm_timeout_s=12))
    chat = p._chat("openai/gpt-oss-120b")
    assert chat.model_name == "openai/gpt-oss-120b"
    assert chat.temperature < 1e-6  # ChatGroq maps 0 to 1e-8 internally
    assert chat.request_timeout == 12
    assert chat.groq_api_key.get_secret_value() == "gsk-test"


@pytest.mark.live
@pytest.mark.skipif(not Settings().groq_api_key, reason="needs GROQ_API_KEY (env or .env)")
def test_live_groq_structured_output():
    p = GroqProvider(Settings(llm_provider="groq"))
    value, usage = p.structured(Out, "Answer in one word.", "What colour is the sky on a clear day?")
    assert value.answer and usage.total > 0 and usage.model
