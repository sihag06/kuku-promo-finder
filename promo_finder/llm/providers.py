"""Thin adapters, one per vendor. Each takes OpenAI-style messages
([{"role": "system"|"user"|"assistant", "content": str}, ...]) and returns
(text, input_tokens, output_tokens). Nothing else in the project knows which
vendor it is talking to.

Model specs are written "provider:model", e.g.
    openai:gpt-5-mini   anthropic:claude-haiku-5-5   gemini:gemini-3.1-flash-lite
    compatible:llama3.1  (any OpenAI-compatible server at llm.base_url: Ollama,
                          OpenRouter, Groq, or an API Kuku provides)
A spec with no prefix is treated as OpenAI.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Optional

PROVIDERS = ("openai", "anthropic", "gemini", "compatible")

# Environment variables checked for each provider's key, in order.
KEY_ENVS: dict[str, tuple[str, ...]] = {
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "compatible": ("COMPATIBLE_API_KEY",),
}


class ProviderError(RuntimeError):
    pass


def split_spec(spec: str) -> tuple[str, str]:
    if ":" in spec:
        provider, model = spec.split(":", 1)
        provider = provider.strip().lower()
        if provider not in PROVIDERS:
            raise ProviderError(f"unknown provider '{provider}' in '{spec}'; use one of {PROVIDERS}")
        return provider, model.strip()
    return "openai", spec.strip()


def api_key(provider: str) -> Optional[str]:
    for env in KEY_ENVS.get(provider, ()):
        if os.environ.get(env):
            return os.environ[env]
    return None


def available_providers(extra: Optional[dict[str, str]] = None) -> list[str]:
    """Providers whose key is present (env or passed in), in preference order."""
    return [p for p in ("openai", "anthropic", "gemini") if api_key(p) or (extra or {}).get(p)]


def _need_key(provider: str, given: Optional[str] = None) -> str:
    key = given or api_key(provider)
    if not key:
        envs = " or ".join(KEY_ENVS[provider])
        raise ProviderError(f"{envs} is not set. Add it to .env, pick another provider, "
                            f"or run with llm.mode: mock to work without a key.")
    return key


# Optional request settings a given model may reject, and words its error would use.
OPTIONAL_KEYS = {
    "temperature": ("temperature",),
    "response_format": ("response_format", "response format", "json"),
    "response_mime_type": ("response_mime_type", "mime"),
    "reasoning_effort": ("reasoning_effort", "reasoning"),
    "thinking_config": ("thinking",),
    "output_config": ("output_config", "effort"),
}


def _drop_unsupported(call: Callable[[dict], Any], kwargs: dict) -> Any:
    """Some models reject temperature, JSON mode or a reasoning setting: retry without
    the culprit (up to 3 times), instead of failing the whole run."""
    for _ in range(3):
        try:
            return call(kwargs)
        except Exception as e:  # SDK-specific error types differ; inspect the message
            msg = str(e).lower()
            culprit = next((k for k, words in OPTIONAL_KEYS.items()
                            if k in kwargs and any(w in msg for w in words)), None)
            if culprit is None:
                raise
            kwargs = {kk: vv for kk, vv in kwargs.items() if kk != culprit}
    return call(kwargs)


# ---- OpenAI and OpenAI-compatible -------------------------------------------------
_clients: dict[tuple, Any] = {}


def call_openai(model: str, messages: list[dict], *, temperature: Optional[float],
                timeout: int, base_url: Optional[str] = None, compatible: bool = False,
                reasoning: Optional[str] = None, key: Optional[str] = None):
    from openai import OpenAI

    key = (key or api_key("compatible") or "not-needed") if compatible else _need_key("openai", key)
    ck = ("openai", base_url, key)
    if ck not in _clients:
        _clients[ck] = OpenAI(api_key=key, base_url=base_url, timeout=timeout)
    client = _clients[ck]
    kwargs: dict[str, Any] = {"model": model, "messages": messages,
                              "response_format": {"type": "json_object"}}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if reasoning and not compatible:
        kwargs["reasoning_effort"] = reasoning
    resp = _drop_unsupported(lambda kw: client.chat.completions.create(**kw), kwargs)
    u = getattr(resp, "usage", None)
    return (resp.choices[0].message.content or "",
            getattr(u, "prompt_tokens", 0) or 0, getattr(u, "completion_tokens", 0) or 0)


# ---- Anthropic -------------------------------------------------------------------
def call_anthropic(model: str, messages: list[dict], *, temperature: Optional[float], timeout: int,
                   reasoning: Optional[str] = None, key: Optional[str] = None):
    import anthropic

    key = _need_key("anthropic", key)
    ck = ("anthropic", key)
    if ck not in _clients:
        _clients[ck] = anthropic.Anthropic(api_key=key, timeout=timeout)
    client = _clients[ck]
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    convo = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]
    kwargs: dict[str, Any] = {"model": model, "max_tokens": 8192, "system": system, "messages": convo}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if reasoning:
        kwargs["output_config"] = {"effort": "low" if reasoning == "minimal" else reasoning}
    resp = _drop_unsupported(lambda kw: client.messages.create(**kw), kwargs)
    text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text")
    return text, resp.usage.input_tokens or 0, resp.usage.output_tokens or 0


# ---- Google Gemini ------------------------------------------------------------------
def call_gemini(model: str, messages: list[dict], *, temperature: Optional[float], timeout: int,
                reasoning: Optional[str] = None, key: Optional[str] = None):
    from google import genai

    key = _need_key("gemini", key)
    ck = ("gemini", key)
    if ck not in _clients:
        _clients[ck] = genai.Client(api_key=key, http_options={"timeout": timeout * 1000})
    client = _clients[ck]
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    contents = [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
                for m in messages if m["role"] != "system"]
    config: dict[str, Any] = {"system_instruction": system, "response_mime_type": "application/json"}
    if temperature is not None:
        config["temperature"] = temperature
    if reasoning:
        config["thinking_config"] = {"thinking_level": reasoning.upper()}

    def _go(cfg: dict):
        return client.models.generate_content(model=model, contents=contents, config=cfg)

    resp = _drop_unsupported(_go, config)
    u = resp.usage_metadata
    out_tokens = (getattr(u, "candidates_token_count", 0) or 0) + (getattr(u, "thoughts_token_count", 0) or 0)
    return resp.text or "", getattr(u, "prompt_token_count", 0) or 0, out_tokens


def call(spec: str, messages: list[dict], *, temperature: Optional[float] = None,
         timeout: int = 120, base_url: Optional[str] = None,
         reasoning: Optional[str] = None, keys: Optional[dict[str, str]] = None) -> tuple[str, int, int]:
    """reasoning: None (model default) | "minimal" | "low" | "medium" | "high": how long the
    model may think before answering. Lower = faster and cheaper, maybe less careful."""
    provider, model = split_spec(spec)
    key = (keys or {}).get(provider)  # a key passed per call (e.g. pasted in the app) beats the env
    if provider == "openai":
        return call_openai(model, messages, temperature=temperature, timeout=timeout, reasoning=reasoning,
                           key=key)
    if provider == "compatible":
        if not base_url:
            raise ProviderError("provider 'compatible' needs llm.base_url (e.g. http://localhost:11434/v1)")
        return call_openai(model, messages, temperature=temperature, timeout=timeout,
                           base_url=base_url, compatible=True, key=key)
    if provider == "anthropic":
        return call_anthropic(model, messages, temperature=temperature, timeout=timeout, reasoning=reasoning,
                              key=key)
    if provider == "gemini":
        return call_gemini(model, messages, temperature=temperature, timeout=timeout, reasoning=reasoning,
                           key=key)
    raise ProviderError(provider)
