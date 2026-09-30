"""Headless render of the Streamlit UI against real runs (skipped if streamlit isn't installed)."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from novaml.config import get_settings  # noqa: E402
from novaml.service import NovaML  # noqa: E402

APP = str(Path(__file__).parents[2] / "ui" / "streamlit_app.py")


@pytest.fixture
def ui_env(tmp_path, monkeypatch):
    monkeypatch.setenv("NOVAML_DATA_DIR", str(tmp_path / "var"))
    monkeypatch.setenv("NOVAML_LLM_PROVIDER", "none")  # never hit a real API from tests
    monkeypatch.setenv("NOVAML_CV_FOLDS", "3")
    monkeypatch.setenv("NOVAML_MAX_IMPROVEMENT_ROUNDS", "0")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_ui_renders_approval_then_results(ui_env, iris_csv):
    svc = NovaML(get_settings(), provider=None)
    paused = svc.start(iris_csv, "species")
    assert paused.status == "awaiting_approval"

    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["result"] = paused
    at.run()
    assert not at.exception
    assert any("Human review" in i.value for i in at.info)

    done = svc.resume(paused.run_id, paused.pending["candidates"])
    svc.close()
    at.session_state["result"] = done
    at.run()
    assert not at.exception
    assert [m.label for m in at.metric][:2] == ["Best model", "CV f1_macro"]
