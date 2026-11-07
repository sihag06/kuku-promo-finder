"""Load config.yaml + .env, with optional overrides from the CLI."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent


class LLMConfig(BaseModel):
    mode: str = "mock"  # mock | live
    preset: str = "auto"  # auto | openai | anthropic | gemini | compatible
    base_url: Optional[str] = None
    temperature: Optional[float] = None
    max_retries: int = 2
    timeout_s: int = 120
    concurrency: int = 4
    rate_limit_retries: int = 6   # waits on "too many requests" before giving up
    backoff_s: float = 5.0        # first wait; doubles each time, max 60s
    presets: dict[str, dict[str, str]] = Field(default_factory=dict)
    models: dict[str, str] = Field(default_factory=dict)  # overrides on top of the preset
    fallbacks: dict[str, list[str]] = Field(default_factory=dict)  # model -> models to try if it is busy
    # How long models may "think": null = model default | minimal | low | medium | high.
    # A string applies to every stage; a dict sets it per stage, e.g. {judge: low, gold: null}.
    reasoning: Optional[str | dict[str, Optional[str]]] = None
    # Keys passed in at runtime (e.g. pasted into the app). Never written to disk or logs.
    api_keys: dict[str, str] = Field(default_factory=dict, exclude=True, repr=False)

    def reasoning_for(self, stage: str) -> Optional[str]:
        r = self.reasoning
        if isinstance(r, dict):
            return r.get(stage, r.get("default"))
        return r
    prices: dict[str, list[float]] = Field(default_factory=dict)

    def active_preset(self) -> str:
        if self.preset != "auto":
            return self.preset
        from .llm.providers import available_providers

        avail = [p for p in available_providers(self.api_keys) if p in self.presets]
        return avail[0] if avail else next(iter(self.presets), "openai")

    def resolved_models(self) -> dict[str, str]:
        """stage -> "provider:model" after applying preset, auto-gold and overrides."""
        name = self.active_preset()
        models = dict(self.presets.get(name, {}))
        if self.preset == "auto":
            # Independence: if a second vendor's key exists, let it build the gold set,
            # so the checker is not just a bigger sibling of the tool's own model.
            from .llm.providers import available_providers

            others = [p for p in available_providers(self.api_keys) if p != name and p in self.presets]
            if others and "gold" in self.presets[others[0]]:
                models["gold"] = self.presets[others[0]]["gold"]
        models.update(self.models)
        if "default" not in models:
            raise ValueError(f"preset '{name}' has no 'default' model in config.yaml")
        return models

    def model_for(self, stage: str) -> str:
        m = self.resolved_models()
        return m.get(stage) or m["default"]


class PromoConfig(BaseModel):
    min_duration_s: float = 20
    max_duration_s: float = 60
    hook_window_s: float = 3.0
    max_segments: int = 3
    min_segment_s: float = 3.0
    cross_episode_span: int = 2
    overlap_dedupe: float = 0.5
    cliff_tail_s: float = 3.5           # max seconds a clip may run past its cliffhanger line
    moments_per_episode: int = 6        # hooks and cliffhangers the miner returns per episode
    candidates_per_episode: int = 4     # kept per hook-episode after pre-scoring
    max_candidates: int = 150           # overall cap before the (paid) judge stage
    max_per_hook: int = 2               # variety: one great moment can't fill the list


class RankingConfig(BaseModel):
    weights: dict[str, float] = Field(default_factory=dict)
    top_k: int = 20
    diversity_lambda: float = 0.3


class PathsConfig(BaseModel):
    cache_dir: str = ".cache/llm"
    output_dir: str = "outputs"


class Config(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    promo: PromoConfig = Field(default_factory=PromoConfig)
    ranking: RankingConfig = Field(default_factory=RankingConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)


def load_dotenv(path: Path) -> None:
    """Minimal .env reader (KEY=value lines). Existing env vars win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    """overrides use dotted keys, e.g. {"llm.provider": "openai", "llm.models.default": "gpt-5-nano"}."""
    load_dotenv(ROOT / ".env")
    path = Path(path) if path else ROOT / "config.yaml"
    data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    data = data or {}
    for dotted, value in (overrides or {}).items():
        if value is None:
            continue
        node = data
        *parents, leaf = dotted.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = value
    return Config.model_validate(data)
