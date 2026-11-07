"""LLM client + provider tests. No network: SDK clients are replaced by fakes."""
import json
from types import SimpleNamespace as NS

import pytest
from pydantic import BaseModel

from promo_finder.config import LLMConfig, load_config
from promo_finder.llm import providers
from promo_finder.llm.client import LLMClient, LLMError, _extract_json

PRESETS = {
    "openai": {"default": "openai:gpt-test", "gold": "openai:gpt-big"},
    "anthropic": {"default": "anthropic:claude-test", "gold": "anthropic:claude-big"},
    "gemini": {"default": "gemini:gem-test", "gold": "gemini:gem-big"},
}


class Answer(BaseModel):
    pick: str
    score: int


@pytest.fixture(autouse=True)
def no_keys(monkeypatch):
    for envs in providers.KEY_ENVS.values():
        for e in envs:
            monkeypatch.delenv(e, raising=False)
    providers._clients.clear()


def make(tmp_path, mode="mock", preset="openai", **kw):
    cfg = LLMConfig(mode=mode, preset=preset, presets=PRESETS,
                    prices={"openai:gpt-test": [1.0, 2.0]}, **kw)
    return LLMClient(cfg, cache_dir=tmp_path / "cache", log_dir=tmp_path / "raw")


# ---- client behaviour ------------------------------------------------------------
def test_extract_json_handles_fences_and_preamble():
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert _extract_json('Sure! Here it is: {"a": 2}') == {"a": 2}


def test_mock_mode_uses_stage_logic_and_logs(tmp_path):
    c = make(tmp_path)
    out = c.complete_json(task="miner", system="s", user="u", schema=Answer,
                          mock=lambda: Answer(pick="E001-0007", score=5), item_id="ep1")
    assert out.pick == "E001-0007"
    line = json.loads((tmp_path / "raw" / "miner.jsonl").read_text().splitlines()[0])
    assert line["item"] == "ep1" and line["model"] == "mock" and line["parsed"]["score"] == 5


def test_mock_without_fallback_fails_clearly(tmp_path):
    with pytest.raises(LLMError, match="no mock answer"):
        make(tmp_path).complete_json(task="t", system="s", user="u", schema=Answer)


def test_cache_hit_skips_the_api(tmp_path, monkeypatch):
    c = make(tmp_path, mode="live")
    calls = []

    def fake(spec, messages):
        calls.append(spec)
        return '{"pick": "E002-0001", "score": 4}', 100, 20

    monkeypatch.setattr(c, "_call_api", fake)
    a = c.complete_json(task="judge", system="s", user="u", schema=Answer)
    b = c.complete_json(task="judge", system="s", user="u", schema=Answer)
    assert a == b and calls == ["openai:gpt-test"]
    rep = c.usage_report()["judge"]
    assert rep["calls"] == 2 and rep["cached"] == 1
    assert rep["cost_usd"] == pytest.approx((100 * 1 + 20 * 2) / 1e6)


def test_invalid_json_is_retried_with_the_error(tmp_path, monkeypatch):
    c = make(tmp_path, mode="live")
    replies = iter(['{"pick": "x"}', '{"pick": "E003-0002", "score": 3}'])
    seen = []

    def fake(spec, messages):
        seen.append(list(messages))
        return next(replies), 10, 5

    monkeypatch.setattr(c, "_call_api", fake)
    assert c.complete_json(task="miner", system="s", user="u", schema=Answer).score == 3
    assert "invalid" in seen[1][-1]["content"]  # error fed back to the model
    assert c.usage_report()["miner"]["retries"] == 1


def test_gives_up_after_max_retries(tmp_path, monkeypatch):
    c = make(tmp_path, mode="live", max_retries=1)
    monkeypatch.setattr(c, "_call_api", lambda s, m: ("not json at all", 1, 1))
    with pytest.raises(LLMError, match="no valid JSON"):
        c.complete_json(task="miner", system="s", user="u", schema=Answer)


def test_missing_key_message(tmp_path):
    c = make(tmp_path, mode="live", preset="anthropic")
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY is not set"):
        c._call_api("anthropic:claude-test", [{"role": "user", "content": "hi"}])


# ---- model selection ---------------------------------------------------------------
def test_split_spec():
    assert providers.split_spec("anthropic:claude-haiku-5-5") == ("anthropic", "claude-haiku-5-5")
    assert providers.split_spec("gpt-5-mini") == ("openai", "gpt-5-mini")
    with pytest.raises(providers.ProviderError):
        providers.split_spec("acme:model")


def test_auto_preset_follows_available_key(monkeypatch):
    cfg = LLMConfig(preset="auto", presets=PRESETS)
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    assert cfg.resolved_models() == PRESETS["gemini"]


def test_auto_gold_uses_a_different_vendor(monkeypatch):
    cfg = LLMConfig(preset="auto", presets=PRESETS)
    monkeypatch.setenv("OPENAI_API_KEY", "o")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    m = cfg.resolved_models()
    assert m["default"] == "openai:gpt-test" and m["gold"] == "anthropic:claude-big"


def test_explicit_preset_and_stage_override():
    cfg = LLMConfig(preset="gemini", presets=PRESETS, models={"judge": "anthropic:claude-big"})
    assert cfg.model_for("miner") == "gemini:gem-test"
    assert cfg.model_for("judge") == "anthropic:claude-big"


def test_real_config_file_loads():
    cfg = load_config(overrides={"llm.preset": "anthropic"})
    assert cfg.llm.model_for("miner").startswith("anthropic:")
    assert cfg.llm.mode in ("mock", "live")


# ---- each vendor adapter speaks its SDK's dialect ------------------------------------
MSGS = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "U1"},
        {"role": "assistant", "content": "A1"}, {"role": "user", "content": "U2"}]


def test_openai_adapter(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    seen = {}

    def create(**kw):
        seen.update(kw)
        return NS(choices=[NS(message=NS(content='{"ok":1}'))], usage=NS(prompt_tokens=7, completion_tokens=3))

    providers._clients[("openai", None, "k")] = NS(chat=NS(completions=NS(create=create)))
    assert providers.call("openai:gpt-x", MSGS) == ('{"ok":1}', 7, 3)
    assert seen["messages"] == MSGS and seen["response_format"] == {"type": "json_object"}
    assert "temperature" not in seen


def test_anthropic_adapter_moves_system_out_of_messages(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    seen = {}

    def create(**kw):
        seen.update(kw)
        return NS(content=[NS(type="text", text='{"ok":1}')], usage=NS(input_tokens=9, output_tokens=4))

    providers._clients[("anthropic", "k")] = NS(messages=NS(create=create))
    assert providers.call("anthropic:claude-x", MSGS, temperature=0.2) == ('{"ok":1}', 9, 4)
    assert seen["system"] == "SYS" and seen["temperature"] == 0.2
    assert [m["role"] for m in seen["messages"]] == ["user", "assistant", "user"]


def test_gemini_adapter_maps_roles_and_counts_thinking(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    seen = {}

    def gen(model, contents, config):
        seen.update(model=model, contents=contents, config=config)
        return NS(text='{"ok":1}', usage_metadata=NS(prompt_token_count=11,
                  candidates_token_count=5, thoughts_token_count=2))

    providers._clients[("gemini", "k")] = NS(models=NS(generate_content=gen))
    assert providers.call("gemini:gem-x", MSGS) == ('{"ok":1}', 11, 7)
    assert seen["config"]["system_instruction"] == "SYS"
    assert seen["config"]["response_mime_type"] == "application/json"
    assert [c["role"] for c in seen["contents"]] == ["user", "model", "user"]


def test_unsupported_temperature_is_dropped_and_retried(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    attempts = []

    def create(**kw):
        attempts.append(dict(kw))
        if "temperature" in kw:
            raise ValueError("Unsupported value: 'temperature' does not support 0.2")
        return NS(choices=[NS(message=NS(content="{}"))], usage=None)

    providers._clients[("openai", None, "k")] = NS(chat=NS(completions=NS(create=create)))
    providers.call("openai:gpt-x", MSGS, temperature=0.2)
    assert len(attempts) == 2 and "temperature" not in attempts[1]


# ---- free-tier rate limits ---------------------------------------------------------------
class _RateLimited(Exception):
    status_code = 429


def test_rate_limit_waits_and_retries(tmp_path, monkeypatch):
    c = make(tmp_path, mode="live", backoff_s=0.0, rate_limit_retries=3)
    attempts = iter([_RateLimited("429 Resource exhausted"), _RateLimited("429"), ('{"pick":"x","score":1}', 5, 5)])

    def fake(spec, messages, **kw):
        nxt = next(attempts)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    monkeypatch.setattr(providers, "call", fake)
    assert c.complete_json(task="t", system="s", user="u", schema=Answer).pick == "x"
    assert c.usage_report()["_total"]["rate_limit_waits"] == 2


def test_auth_errors_are_not_retried(tmp_path, monkeypatch):
    c = make(tmp_path, mode="live", backoff_s=0.0, rate_limit_retries=3)
    calls = []

    def fake(spec, messages, **kw):
        calls.append(1)
        raise PermissionError("403 Forbidden: API key not valid")

    monkeypatch.setattr(providers, "call", fake)
    with pytest.raises(PermissionError):
        c.complete_json(task="t", system="s", user="u", schema=Answer)
    assert len(calls) == 1


# ---- busy-model fallbacks ------------------------------------------------------------
class _Busy(Exception):
    code = 503


def _fallback_client(tmp_path):
    return make(tmp_path, mode="live", backoff_s=0.0, rate_limit_retries=2,
                fallbacks={"openai:gpt-test": ["openai:gpt-alt", "openai:gpt-alt2"]})


def test_busy_model_falls_back_and_logs_who_answered(tmp_path, monkeypatch):
    c = _fallback_client(tmp_path)
    tried = []

    def fake(spec, messages, **kw):
        tried.append(spec)
        if spec == "openai:gpt-test":
            raise _Busy("503 UNAVAILABLE: high demand")
        return '{"pick":"x","score":2}', 5, 5

    monkeypatch.setattr(providers, "call", fake)
    c.complete_json(task="judge", system="s", user="u", schema=Answer, item_id="p1")
    assert tried == ["openai:gpt-test", "openai:gpt-alt"]  # no waiting: switched at once
    rep = c.usage_report()
    assert rep["_total"]["fallback_calls"] == 1 and rep["_total"]["rate_limit_waits"] == 0
    assert rep["judge"]["models"] == ["openai:gpt-alt"]
    line = json.loads((tmp_path / "raw" / "judge.jsonl").read_text().splitlines()[0])
    assert line["model"] == "openai:gpt-test" and line["answered_by"] == "openai:gpt-alt"


def test_broken_fallback_is_skipped(tmp_path, monkeypatch):
    c = _fallback_client(tmp_path)

    def fake(spec, messages, **kw):
        if spec == "openai:gpt-test":
            raise _Busy("503")
        if spec == "openai:gpt-alt":
            raise ValueError("404 model not found")  # not retryable, but only a fallback
        return '{"pick":"y","score":2}', 5, 5

    monkeypatch.setattr(providers, "call", fake)
    assert c.complete_json(task="t", system="s", user="u", schema=Answer).pick == "y"


def test_all_busy_waits_then_succeeds(tmp_path, monkeypatch):
    c = _fallback_client(tmp_path)
    state = {"n": 0}

    def fake(spec, messages, **kw):
        state["n"] += 1
        if state["n"] <= 3:  # whole first round busy
            raise _Busy("503")
        return '{"pick":"z","score":2}', 5, 5

    monkeypatch.setattr(providers, "call", fake)
    assert c.complete_json(task="t", system="s", user="u", schema=Answer).pick == "z"
    assert c.usage_report()["_total"]["rate_limit_waits"] == 1


def test_gives_up_after_all_rounds(tmp_path, monkeypatch):
    c = _fallback_client(tmp_path)
    monkeypatch.setattr(providers, "call", lambda spec, messages, **kw: (_ for _ in ()).throw(_Busy("503 busy")))
    with pytest.raises(LLMError, match="no model answered.*gpt-alt2: 503 busy"):
        c.complete_json(task="t", system="s", user="u", schema=Answer)


def test_config_gold_chain_never_uses_the_tools_models():
    cfg = load_config()
    for preset in ("gemini", "openai"):
        m = cfg.llm.presets[preset]
        tool_side = {m["default"], *cfg.llm.fallbacks.get(m["default"], [])}
        gold_side = {m["gold"], *cfg.llm.fallbacks.get(m["gold"], [])}
        assert not tool_side & gold_side, preset



# ---- daily caps and reasoning control --------------------------------------------------
class _Daily(Exception):
    code = 429


def test_daily_quota_model_is_skipped_for_the_rest_of_the_run(tmp_path, monkeypatch):
    c = _fallback_client(tmp_path)
    tried = []

    def fake(spec, messages, **kw):
        tried.append(spec)
        if spec == "openai:gpt-test":
            raise _Daily("429 RESOURCE_EXHAUSTED: requests per day limit exceeded on Free Tier")
        return '{"pick":"x","score":2}', 5, 5

    monkeypatch.setattr(providers, "call", fake)
    c.complete_json(task="t", system="s", user="u1", schema=Answer)
    c.complete_json(task="t", system="s", user="u2", schema=Answer)
    assert tried == ["openai:gpt-test", "openai:gpt-alt", "openai:gpt-alt"]  # capped model not retried


def test_all_models_capped_fails_fast_with_a_clear_message(tmp_path, monkeypatch):
    c = make(tmp_path, mode="live", backoff_s=30.0, rate_limit_retries=5)  # would wait minutes if it retried
    monkeypatch.setattr(providers, "call", lambda spec, messages, **kw: (_ for _ in ()).throw(
        _Daily("429 quota exceeded: GenerateRequestsPerDayPerProjectPerModel-FreeTier")))
    with pytest.raises(LLMError, match="Daily free-tier quota used up"):
        c.complete_json(task="t", system="s", user="u", schema=Answer)


def test_reasoning_setting_reaches_each_vendor(monkeypatch):
    seen = {}
    monkeypatch.setenv("OPENAI_API_KEY", "k"); monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    providers._clients[("openai", None, "k")] = NS(chat=NS(completions=NS(create=lambda **kw: (
        seen.__setitem__("openai", kw.get("reasoning_effort")),
        NS(choices=[NS(message=NS(content="{}"))], usage=None))[1])))
    providers._clients[("anthropic", "k")] = NS(messages=NS(create=lambda **kw: (
        seen.__setitem__("anthropic", kw.get("output_config")),
        NS(content=[NS(type="text", text="{}")], usage=NS(input_tokens=1, output_tokens=1)))[1]))
    providers._clients[("gemini", "k")] = NS(models=NS(generate_content=lambda model, contents, config: (
        seen.__setitem__("gemini", config.get("thinking_config")),
        NS(text="{}", usage_metadata=NS(prompt_token_count=1, candidates_token_count=1)))[1]))
    for spec in ("openai:a", "anthropic:b", "gemini:c"):
        providers.call(spec, MSGS, reasoning="low")
    assert seen == {"openai": "low", "anthropic": {"effort": "low"}, "gemini": {"thinking_level": "LOW"}}


def test_rejected_reasoning_setting_is_dropped(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    attempts = []

    def create(**kw):
        attempts.append(dict(kw))
        if "reasoning_effort" in kw:
            raise ValueError("Unrecognized request argument supplied: reasoning_effort")
        return NS(choices=[NS(message=NS(content="{}"))], usage=None)

    providers._clients[("openai", None, "k")] = NS(chat=NS(completions=NS(create=create)))
    providers.call("openai:old-model", MSGS, reasoning="low")
    assert len(attempts) == 2 and "reasoning_effort" not in attempts[1]


def test_reasoning_per_stage():
    cfg = LLMConfig(reasoning={"judge": "low", "default": None})
    assert cfg.reasoning_for("judge") == "low" and cfg.reasoning_for("miner") is None
    assert LLMConfig(reasoning="minimal").reasoning_for("anything") == "minimal"
