import os
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from novaml.cli import main
from novaml.config import Settings
from novaml.llm.base import LLMError
from novaml.llm.providers import AnthropicProvider, build_provider


class Out(BaseModel):
    answer: str


class FakeMessages:
    def __init__(self, resp=None, exc=None):
        self.resp, self.exc, self.kwargs = resp, exc, None

    def parse(self, **kwargs):
        self.kwargs = kwargs
        if self.exc:
            raise self.exc
        return self.resp


def fake_client(resp=None, exc=None):
    m = FakeMessages(resp, exc)
    return SimpleNamespace(messages=m, beta=SimpleNamespace(messages=m)), m


OK = Out(answer="ok")


def resp(stop="end_turn", parsed=OK):
    return SimpleNamespace(stop_reason=stop, parsed_output=parsed, usage=SimpleNamespace(input_tokens=12, output_tokens=5))


def provider(client, **kw):
    return AnthropicProvider(Settings(_env_file=None, llm_provider="anthropic", **kw), client=client)


def test_anthropic_structured_call_shape():
    client, m = fake_client(resp())
    value, usage = provider(client).structured(Out, "sys", "user")
    assert value.answer == "ok" and usage.total == 17
    assert m.kwargs["model"] == "claude-opus-5-5"
    assert m.kwargs["output_format"] is Out
    assert m.kwargs["output_config"] == {"effort": "low"}
    assert m.kwargs["fallbacks"] == "default" and m.kwargs["betas"] == ["server-side-fallback-2026-07-01"]


def test_anthropic_without_server_fallbacks_uses_ga_endpoint():
    client, m = fake_client(resp())
    provider(client, llm_server_fallbacks=False).structured(Out, "s", "u")
    assert "fallbacks" not in m.kwargs and "betas" not in m.kwargs


@pytest.mark.parametrize("stop,parsed,match", [("refusal", None, "refused"), ("max_tokens", None, "truncated"), ("end_turn", None, "no structured")])
def test_anthropic_bad_responses_raise_llm_error(stop, parsed, match):
    client, _ = fake_client(resp(stop, parsed))
    with pytest.raises(LLMError, match=match):
        provider(client).structured(Out, "s", "u")


def test_anthropic_api_errors_become_llm_errors():
    import anthropic

    client, _ = fake_client(exc=anthropic.APIConnectionError(request=SimpleNamespace(method="POST", url="u")))
    with pytest.raises(LLMError, match="APIConnectionError"):
        provider(client).structured(Out, "s", "u")


def test_build_provider_none():
    assert build_provider(Settings(_env_file=None, llm_provider="none")) is None


def test_cli_run_and_status(tmp_path, iris_csv, monkeypatch, capsys):
    monkeypatch.setenv("NOVAML_DATA_DIR", str(tmp_path / "var"))
    monkeypatch.setenv("NOVAML_LLM_PROVIDER", "none")  # never hit a real API from tests
    monkeypatch.setenv("NOVAML_CV_FOLDS", "3")
    monkeypatch.setenv("NOVAML_MAX_IMPROVEMENT_ROUNDS", "0")
    assert main(["run", str(iris_csv), "--target", "species", "--auto-approve"]) == 0
    out = capsys.readouterr().out
    assert "status=completed" in out
    run_id = out.split("run_id=")[1].split()[0]
    assert main(["status", run_id]) == 0
    assert main(["run", str(iris_csv), "--target", "nope", "--auto-approve"]) == 1


@pytest.mark.live
@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="needs ANTHROPIC_API_KEY")
def test_live_anthropic_structured_output():
    p = AnthropicProvider(Settings(llm_provider="anthropic"))
    value, usage = p.structured(Out, "Answer in one word.", "What colour is the sky on a clear day?")
    assert value.answer and usage.total > 0
