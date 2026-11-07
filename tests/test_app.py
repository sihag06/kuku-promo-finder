"""Headless smoke tests for the Streamlit app (no browser, no key, no network)."""
import json
from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    at = st_testing.AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    return at


def test_landing_page_renders(app):
    app.run()
    assert not app.exception
    assert app.title[0].value == "Promo Finder"


def test_demo_show_runs_in_rules_mode_and_shows_promos(app):
    app.run()
    app.sidebar.radio[0].set_value("Demo show (fictional)").run()
    app.sidebar.button[0].click().run()  # "Find promos" (rules only is the default engine)
    assert not app.exception, app.exception
    assert app.session_state["run"]["meta"]["mode"] == "mock"
    assert len(app.dataframe) >= 1 and len(app.expander) >= 1


def test_ai_mode_without_key_disables_the_run_button(app, monkeypatch):
    for env in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    app.run()
    app.sidebar.radio[1].set_value("AI models (needs a key)").run()
    assert app.sidebar.button[0].disabled
    assert any("Add a key" in w.value for w in app.sidebar.warning)


def test_typed_key_never_reaches_disk(tmp_path):
    """A key passed through config.api_keys must not appear in any dumped config."""
    from promo_finder.config import load_config
    cfg = load_config()
    cfg.llm.api_keys = {"openai": "sk-SECRET-123"}
    assert "sk-SECRET-123" not in json.dumps(cfg.model_dump(mode="json"))
    assert "sk-SECRET-123" not in repr(cfg.llm)
