"""Runs the stages in order and writes every intermediate result to the run folder.

outputs/<show-slug>/<run_id>/
  run_meta.json          config, models, timings, token + cost usage
  show_bible.json        Stage 1
  moments.json           Stage 3 (per-episode hooks / cliffhangers, with notes)
  candidates_all.json    Stage 4 (valid, de-duplicated, pruned candidates, before judging)
  rejected.json          every candidate that failed a rule, with the reason
  promos_ranked.csv      FINAL: the editor's sheet (rank, tier, segments, hook, cliffhanger, why)
  promos_ranked.md       FINAL: top list with the exact lines of every clip
  promos.json            FINAL: everything, machine-readable
  raw_llm/<task>.jsonl   every prompt and response, unedited
"""
from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from .assembler import assemble
from .config import Config
from .export import write_all
from .judge import judge_promo
from .pipeline_utils import slug
from .ranker import apply_judgement, rank
from .llm.client import LLMClient
from .miner import mine_episode
from .parser import load_show
from .show_bible import build_show_bible
from .signals import score_episode


def _git_commit() -> Optional[str]:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


class Pipeline:
    def __init__(self, cfg: Config, input_path: str, run_id: Optional[str] = None,
                 progress: Optional[Callable[[str], None]] = None, use_cache: bool = True):
        self.cfg = cfg
        self.use_cache = use_cache
        self.input_path = input_path
        self.run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
        self.progress = progress or (lambda msg: print(msg, flush=True))
        self.timings: dict[str, float] = {}

    def _stage(self, name: str, fn):
        t0 = time.time()
        self.progress(f"[{name}] ...")
        out = fn()
        self.timings[name] = round(time.time() - t0, 2)
        self.progress(f"[{name}] done in {self.timings[name]}s")
        return out

    def run(self) -> dict:
        cfg = self.cfg
        show = self._stage("parse", lambda: load_show(self.input_path))
        out = Path(cfg.paths.output_dir) / slug(show.title) / self.run_id
        out.mkdir(parents=True, exist_ok=True)
        llm = LLMClient(cfg.llm, cache_dir=cfg.paths.cache_dir, log_dir=out / "raw_llm",
                        use_cache=self.use_cache)

        bible = self._stage("show_bible", lambda: build_show_bible(show, llm))
        sigs = self._stage("signals", lambda: {ep.number: score_episode(ep) for ep in show.episodes})
        k = cfg.promo.moments_per_episode
        mined_list = self._stage("miner", lambda: llm.map(
            lambda ep: mine_episode(ep, bible, sigs[ep.number], llm, k), show.episodes,
            label="episodes mined", progress=self.progress))
        mined = {ep.number: m for ep, m in zip(show.episodes, mined_list)}
        cands, rejected = self._stage("assemble", lambda: assemble(show, mined, sigs, cfg.promo))

        (out / "show_bible.json").write_text(bible.model_dump_json(indent=2), encoding="utf-8")
        (out / "moments.json").write_text(json.dumps(
            {n: m.model_dump() for n, m in mined.items()}, indent=2, ensure_ascii=False), encoding="utf-8")
        (out / "candidates_all.json").write_text(json.dumps(
            [p.model_dump(mode="json") for p in cands], indent=2, ensure_ascii=False), encoding="utf-8")
        (out / "rejected.json").write_text(json.dumps(
            [r.__dict__ for r in rejected], indent=2, ensure_ascii=False), encoding="utf-8")

        judgements = self._stage("judge", lambda: llm.map(
            lambda p: judge_promo(p, show, bible, sigs, llm), cands,
            label="promos judged", progress=self.progress))
        judged = [apply_judgement(p, j, show, sigs, cfg.ranking) for p, j in zip(cands, judgements)]
        ranked = self._stage("rank", lambda: rank(judged, cfg.ranking))
        usage = llm.usage_report()["_total"]
        header = [f"run `{self.run_id}` | mode **{cfg.llm.mode}** | models "
                  f"{'rule-based (no API)' if cfg.llm.mode == 'mock' else cfg.llm.resolved_models()} | "
                  f"{len(ranked)} promos from {len(cands) + len(rejected)} candidates | "
                  f"LLM cost ${usage['cost_usd']:.4f}"]
        if cfg.llm.mode == "mock":
            header.append("> Mock mode: every score below is a rule-based estimate, not a model judgement.")
        write_all(ranked, show, out, cfg.ranking.top_k, header)
        meta = {
            "run_id": self.run_id, "show": show.title, "input": str(Path(self.input_path).resolve()),
            "cache": self.use_cache,
            "episodes": len(show.episodes), "runtime_min": round(show.total_duration / 60, 1),
            "mode": cfg.llm.mode,
            "models": "mock" if cfg.llm.mode == "mock" else cfg.llm.resolved_models(),
            "stage_seconds": self.timings, "llm_usage": llm.usage_report(),
            "candidates": len(cands), "rejected": len(rejected), "ranked": len(ranked),
            "tiers": {t: sum(p.tier == t for p in ranked) for t in sorted({p.tier for p in ranked})},
            "parse_warnings": [w for ep in show.episodes for w in ep.warnings],
            "git_commit": _git_commit(), "config": cfg.model_dump(mode="json"),
        }
        (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        self.progress(f"Wrote {out}")
        return {"show": show, "bible": bible, "mined": mined, "candidates": cands, "ranked": ranked,
                "rejected": rejected, "out_dir": out, "meta": meta}
