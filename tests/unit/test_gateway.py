from pydantic import BaseModel

from novaml.config import Settings
from novaml.llm.base import LLMError
from novaml.llm.gateway import LLMGateway, ValidationFailed
from novaml.llm.scripted import ScriptedProvider


class Pick(BaseModel):
    choice: str


def _gw(provider, spent=0, **kw):
    return LLMGateway(provider, Settings(_env_file=None, **kw), tokens_spent=spent)


def _decide(gw, validate=None):
    return gw.decide(agent="t", schema=Pick, system="s", user="u", policy=lambda: Pick(choice="policy"), validate=validate)


def test_no_provider_uses_policy():
    d = _decide(_gw(None))
    assert d.source == "policy" and d.value.choice == "policy" and d.record["error"] is None


def test_llm_answer_and_usage_recorded():
    p = ScriptedProvider({"Pick": [{"choice": "llm"}]}, tokens_per_call=300)
    d = _decide(_gw(p, input_cost_per_mtok=1.0, output_cost_per_mtok=2.0))
    assert d.source == "llm" and d.value.choice == "llm"
    assert d.record["input_tokens"] + d.record["output_tokens"] == 300
    assert d.record["cost_usd"] > 0


def test_provider_error_degrades_to_policy():
    p = ScriptedProvider({"Pick": [LLMError("503")]})
    d = _decide(_gw(p))
    assert d.source == "policy" and "503" in d.record["error"]


def test_invalid_output_degrades_to_policy():
    def validate(v):
        raise ValidationFailed("unknown model")

    p = ScriptedProvider({"Pick": [{"choice": "xgboost_9000"}]})
    d = _decide(_gw(p), validate)
    assert d.source == "policy" and "unknown model" in d.record["error"]


def test_budget_exhausted_skips_llm():
    p = ScriptedProvider({"Pick": [{"choice": "llm"}]})
    d = _decide(_gw(p, spent=1_000, run_token_budget=1_000))
    assert d.source == "policy" and "budget" in d.record["error"]
    assert p.calls == []


def test_prompt_carries_data_guard():
    p = ScriptedProvider({"Pick": [{"choice": "llm"}]})
    _decide(_gw(p))
    assert "untrusted" in p.calls[0]["system"]
