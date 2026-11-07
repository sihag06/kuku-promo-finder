"""The only door to an LLM. Every stage calls `complete_json(...)` and gets back a
validated Pydantic object. Which vendor answers is decided by config, not code.

Modes
- live: real models. Each stage's model is a "provider:model" spec (OpenAI, Anthropic,
        Gemini, or any OpenAI-compatible server); see providers.py.
- mock: no network. The calling stage passes `mock=`, a function that answers with
        rule-based logic. The whole pipeline runs and is testable without a key,
        and this mode doubles as the "heuristics only" ablation baseline.

Every call (cached or not) is appended to raw_llm/<task>.jsonl: the unedited
prompts and responses the brief asks for as "raw outputs".
"""
from __future__ import annotations

import json
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, TypeVar

from pydantic import BaseModel, ValidationError

from ..config import LLMConfig
from . import providers
from .cache import DiskCache, cache_key

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


@dataclass
class TaskUsage:
    calls: int = 0
    cached: int = 0
    retries: int = 0
    failures: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    seconds: float = 0.0
    models: set = field(default_factory=set)


def _extract_json(text: str) -> Any:
    """Parse JSON even if the model wrapped it in ```json fences or added a preamble."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = min((i for i in (text.find("{"), text.find("[")) if i >= 0), default=-1)
        if start < 0:
            raise
        return json.loads(text[start:])


def is_daily_quota(e: Exception) -> bool:
    """A per-DAY cap (e.g. Gemini free tier: 20 requests/day/model): retrying today is pointless."""
    msg = str(e).lower().replace(" ", "")
    return ("perday" in msg or "requestsperday" in msg or "dailylimit" in msg) and (
        "429" in msg or "quota" in msg or "exhausted" in msg)


def is_retryable(e: Exception) -> bool:
    """Rate limits (429 / quota / resource exhausted) and transient server errors."""
    code = getattr(e, "status_code", None) or getattr(e, "code", None)
    if code in (429, 500, 502, 503, 504):
        return True
    msg = str(e).lower()
    return any(k in msg for k in ("429", "rate limit", "rate_limit", "resource exhausted",
                                  "resource_exhausted", "quota", "overloaded", "503", "unavailable"))


class LLMClient:
    def __init__(self, cfg: LLMConfig, cache_dir: str | Path = ".cache/llm",
                 log_dir: str | Path | None = None, use_cache: bool = True):
        self.cfg = cfg
        self.cache = DiskCache(cache_dir, enabled=use_cache)
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self.usage: dict[str, TaskUsage] = defaultdict(TaskUsage)
        self._lock = threading.Lock()
        self._tl = threading.local()  # which model actually answered the current call
        self._exhausted: set[str] = set()  # models whose daily quota ran out during this run

    @property
    def is_mock(self) -> bool:
        return self.cfg.mode == "mock"

    def model_for(self, stage: str) -> str:
        return "mock" if self.is_mock else self.cfg.model_for(stage)

    def _call_api(self, spec: str, messages: list[dict]) -> tuple[str, int, int]:
        """One API call that survives free-tier trouble.

        Tries the model, then each fallback in `llm.fallbacks` (a busy model is skipped
        at once instead of waited on). A model whose DAILY quota is used up is remembered
        and skipped for the rest of the run. If every model is busy or rate-limited,
        waits (5s, 10s, 20s... max 60s) and goes round again. A primary model's
        non-temporary error (bad key, bad request) is raised immediately; a broken
        fallback is skipped. If nothing answers, the error lists what each model said.
        The model that actually answered is kept in self._tl.model."""
        chain = [spec] + [m for m in self.cfg.fallbacks.get(spec, []) if m != spec]
        reasoning = self.cfg.reasoning_for(getattr(self._tl, "stage", ""))
        delay = self.cfg.backoff_s
        errors: dict[str, str] = {}
        for round_ in range(self.cfg.rate_limit_retries + 1):
            live = [m for m in chain if m not in self._exhausted]
            if not live:
                break
            for model in live:
                try:
                    out = providers.call(model, messages, temperature=self.cfg.temperature,
                                         timeout=self.cfg.timeout_s, base_url=self.cfg.base_url,
                                         reasoning=reasoning, keys=self.cfg.api_keys)
                    self._tl.model = model
                    if model != spec:
                        with self._lock:
                            self.usage["_fallbacks"].calls += 1
                            self.usage["_fallbacks"].models.add(f"{spec} -> {model}")
                    return out
                except providers.ProviderError as e:
                    if model == spec:
                        raise LLMError(str(e)) from e
                    errors[model] = str(e)[:160]
                except Exception as e:
                    errors[model] = str(e)[:160]
                    if is_daily_quota(e):
                        with self._lock:
                            self._exhausted.add(model)
                    elif model == spec and not is_retryable(e):
                        raise
            if round_ == self.cfg.rate_limit_retries or all(m in self._exhausted for m in chain):
                break
            with self._lock:
                self.usage["_rate_limit_waits"].retries += 1
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
        detail = "; ".join(f"{m}: {msg}" for m, msg in errors.items()) or "no model tried"
        hint = (" Daily free-tier quota used up: wait for the reset, add another vendor's key to .env,"
                " or enable billing." if any(m in self._exhausted for m in chain) else "")
        raise LLMError(f"no model answered for {spec}.{hint} Details: {detail}")

    def _price(self, spec: str, tin: int, tout: int) -> float:
        p = self.cfg.prices.get(spec) or self.cfg.prices.get(providers.split_spec(spec)[1])
        return 0.0 if not p else (tin * p[0] + tout * p[1]) / 1_000_000

    def _log(self, task: str, record: dict) -> None:
        if not self.log_dir:
            return
        with self._lock:
            with open(self.log_dir / f"{task}.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ---- the one public call ---------------------------------------------------
    def complete_json(self, *, task: str, system: str, user: str, schema: type[T],
                      mock: Optional[Callable[[], T]] = None, stage: Optional[str] = None,
                      item_id: str = "") -> T:
        """Ask for JSON matching `schema`; validate; retry with the error on failure."""
        spec = self.model_for(stage or task)
        schema_json = schema.model_json_schema()
        key = cache_key(model=spec, temperature=self.cfg.temperature,
                        system=system, user=user, schema=schema_json)
        t0 = time.time()
        u = self.usage[task]

        # Mock answers are free and come from code that changes while we build, so they
        # are never cached (a cached mock would hide a rule change).
        hit = None if self.is_mock else self.cache.get(key)
        if hit is not None:
            try:
                obj = schema.model_validate(hit["parsed"])
                with self._lock:
                    u.calls += 1; u.cached += 1; u.models.add(spec)
                self._log(task, {"item": item_id, "model": spec, "cached": True,
                                 "key": key, "response": hit.get("raw", ""), "parsed": hit["parsed"]})
                return obj
            except (ValidationError, KeyError):
                pass  # schema changed since cached: recompute

        if self.is_mock:
            if mock is None:
                raise LLMError(f"task '{task}' has no mock answer; set llm.mode: live")
            obj = mock()
            parsed = obj.model_dump(mode="json")
            raw, tin, tout, retries = json.dumps(parsed), len(system + user) // 4, 0, 0
        else:
            sys_msg = (system.rstrip() + "\n\nReturn ONLY a JSON object that validates against "
                       "this JSON Schema:\n" + json.dumps(schema_json))
            messages = [{"role": "system", "content": sys_msg}, {"role": "user", "content": user}]
            tin = tout = retries = 0
            last_err, raw, obj = "", "", None
            for _ in range(self.cfg.max_retries + 1):
                self._tl.model, self._tl.stage = spec, stage or task
                raw, a_in, a_out = self._call_api(spec, messages)
                tin += a_in; tout += a_out
                try:
                    obj = schema.model_validate(_extract_json(raw))
                    break
                except (json.JSONDecodeError, ValidationError) as e:
                    last_err = str(e)[:1500]
                    retries += 1
                    messages += [{"role": "assistant", "content": raw},
                                 {"role": "user", "content": "That JSON was invalid:\n" + last_err +
                                  "\nReturn the corrected JSON object only."}]
            if obj is None:
                with self._lock:
                    u.calls += 1; u.failures += 1; u.retries += retries
                self._log(task, {"item": item_id, "model": spec, "cached": False, "key": key,
                                 "error": last_err, "response": raw})
                raise LLMError(f"{task}/{item_id}: no valid JSON after {retries} retries: {last_err[:200]}")
            parsed = obj.model_dump(mode="json")

        answered_by = "mock" if self.is_mock else getattr(self._tl, "model", spec)
        cost = self._price(answered_by, tin, tout)
        with self._lock:
            u.calls += 1; u.retries += retries; u.models.add(answered_by)
            u.input_tokens += tin; u.output_tokens += tout; u.cost_usd += cost
            u.seconds += time.time() - t0
        if not self.is_mock:
            self.cache.set(key, {"raw": raw, "parsed": parsed, "model": answered_by})
        self._log(task, {"item": item_id, "model": spec, "answered_by": answered_by,
                         "cached": False, "key": key, "system": system, "user": user, "response": raw, "parsed": parsed,
                         "input_tokens": tin, "output_tokens": tout, "retries": retries})
        return obj

    def map(self, fn: Callable[[Any], Any], items: Iterable[Any], label: str = "",
            progress: Optional[Callable[[str], None]] = None) -> list[Any]:
        """Run fn over items with the configured concurrency, keeping order.
        With a label, prints "label 12/50 (1m05s)" as items finish, so long runs never look stuck."""
        items = list(items)
        workers = 1 if self.is_mock else max(1, self.cfg.concurrency)
        say = progress or (lambda m: print(m, flush=True))
        t0, done = time.time(), [0]
        every = max(1, len(items) // 10)

        def tick(x):
            out = fn(x)
            with self._lock:
                done[0] += 1
                n = done[0]
            if label and not self.is_mock and (n % every == 0 or n == len(items) or n <= 2):
                el = int(time.time() - t0)
                say(f"  {label} {n}/{len(items)} ({el // 60}m{el % 60:02d}s)")
            return out

        if workers == 1:
            return [tick(x) for x in items]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(tick, items))

    def usage_report(self) -> dict:
        out = {}
        waits = self.usage.get("_rate_limit_waits")
        fb = self.usage.get("_fallbacks")
        for task, u in self.usage.items():
            if task.startswith("_"):
                continue
            out[task] = {"calls": u.calls, "cached": u.cached, "retries": u.retries,
                         "failures": u.failures, "input_tokens": u.input_tokens,
                         "output_tokens": u.output_tokens, "cost_usd": round(u.cost_usd, 6),
                         "seconds": round(u.seconds, 1), "models": sorted(u.models)}
        out["_total"] = {k: round(sum(v[k] for v in out.values()), 6)
                         for k in ("calls", "cached", "input_tokens", "output_tokens", "cost_usd")}
        out["_total"]["rate_limit_waits"] = waits.retries if waits else 0
        out["_total"]["fallback_calls"] = fb.calls if fb else 0
        out["_total"]["fallbacks_used"] = sorted(fb.models) if fb else []
        return out


def check_models(cfg: LLMConfig) -> int:
    """`python -m promo_finder.llm.client --check`: confirm keys work and every model
    the current preset would use exists on the account, with one tiny call each."""

    class Ping(BaseModel):
        ok: bool

    print(f"keys found for: {providers.available_providers(cfg.api_keys) or 'none'}")
    print(f"preset: {cfg.preset}  ->  models:")
    failed = 0
    msgs = [{"role": "system", "content": 'Reply with the JSON object {"ok": true} and nothing else.'},
            {"role": "user", "content": "ping"}]
    for stage, spec in cfg.resolved_models().items():
        chain = [spec] + cfg.fallbacks.get(spec, [])
        usable = False
        for i, model in enumerate(chain):
            label = stage if i == 0 else "  fallback"
            try:  # direct call, no retries: we want to see each model's own status
                text, tin, tout = providers.call(model, msgs, temperature=cfg.temperature,
                                                 timeout=cfg.timeout_s, base_url=cfg.base_url,
                                                 keys=cfg.api_keys)
                Ping.model_validate(_extract_json(text))
                usable = True
                print(f"  OK    {label:<11} {model}  ({tin}+{tout} tokens)")
            except Exception as e:
                print(f"  {'BUSY' if is_retryable(e) else 'FAIL'}  {label:<11} {model}  -> {str(e)[:120]}")
        if not usable:
            failed += 1
            print(f"  !! no usable model for '{stage}' right now")
    return 1 if failed else 0


if __name__ == "__main__":
    import argparse
    import sys

    from ..config import load_config

    ap = argparse.ArgumentParser(description="LLM client utilities")
    ap.add_argument("--check", action="store_true", help="verify keys + the models this preset uses")
    ap.add_argument("--preset", default=None, help="auto | openai | anthropic | gemini")
    args = ap.parse_args()
    cfg = load_config(overrides={"llm.preset": args.preset})
    sys.exit(check_models(cfg.llm) if args.check else 0)
