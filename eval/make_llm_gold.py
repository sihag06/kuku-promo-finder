"""Route B gold set: an independent, stronger model picks promos, blind to the tool.

    python -m eval.make_llm_gold data                     # needs a key; uses the "gold" model
    python -m eval.make_llm_gold data --episodes 2 3 5

How it stays independent of the tool being tested:
- a different (stronger, or other-vendor) model: the `gold` entry in config.yaml
- its own prompts (prompts/gold_editor_a.md and _b.md), reading the full episode script
- it never sees the tool's signals, moments, candidates, rubric or scores

Noise control: two differently-worded prompts pick independently; only promos both
agree on (same opening, or at least 50% shared footage) go into eval/gold_llm.csv.
Every pick from both runs, with its agreement status, goes to eval/gold_llm_all.csv.

Limitation (stated in the report): it reads one episode at a time, so it never
proposes cross-episode promos.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from eval.gold import GoldPromo, overlap_seconds, same_opening, save_gold
from eval.make_reading_pack import DEFAULT_EPISODES
from promo_finder import prompts
from promo_finder.config import load_config
from promo_finder.llm.client import LLMClient
from promo_finder.parser import compact_transcript, load_show
from promo_finder.schema import Episode


class GoldSegment(BaseModel):
    start_event_id: str
    end_event_id: str


class GoldPick(BaseModel):
    segments: list[GoldSegment] = Field(min_length=1, max_length=3)
    hook: str = ""
    cliffhanger: str = ""
    why: str = ""
    confidence: int = Field(default=3, ge=1, le=5)


class GoldPicks(BaseModel):
    picks: list[GoldPick] = Field(default_factory=list)


def to_gold(pick: GoldPick, ep: Episode, gid: str, source: str, min_s: float, max_s: float):
    """Turn event ids into times; return (GoldPromo, None) or (None, reason)."""
    ids = {e.id: e for e in ep.events}
    segs = []
    for s in pick.segments:
        a, b = ids.get(s.start_event_id), ids.get(s.end_event_id)
        if not a or not b:
            return None, f"unknown id in {s.start_event_id}..{s.end_event_id}"
        if b.end <= a.start:
            return None, f"segment ends before it starts ({s.start_event_id}..{s.end_event_id})"
        segs.append((ep.number, a.start, b.end))
    g = GoldPromo(id=gid, segments=segs, hook=pick.hook, cliffhanger=pick.cliffhanger, why=pick.why,
                  confidence=pick.confidence, source=source)
    if not (min_s - 1 <= g.duration <= max_s + 1):  # 1s slack: the model picks whole events
        return None, f"duration {g.duration}s outside {min_s}-{max_s}s"
    return g, None


def agree(a: GoldPromo, b: GoldPromo) -> bool:
    return same_opening(a, b) or overlap_seconds(a, b) / max(min(a.duration, b.duration), 1e-6) >= 0.5


def build_llm_gold(show, episodes: list[int], llm: LLMClient, cfg, max_picks: int = 3):
    """Both prompt variants for every episode (run in parallel, with progress), then
    keep only the picks the two variants agree on."""
    jobs = [(n, v) for n in episodes for v in ("a", "b")]

    def ask(job):
        n, variant = job
        ep = show.episode(n)
        system, user = prompts.load(f"gold_editor_{variant}", episode=str(n),
                                    transcript=compact_transcript(ep, include_camera=True),
                                    max_picks=str(max_picks))
        return llm.complete_json(task=f"gold_{variant}", stage="gold", system=system, user=user,
                                 schema=GoldPicks, item_id=f"ep{n:03d}")

    answers = dict(zip(jobs, llm.map(ask, jobs, label="gold calls")))
    agreed, everything, notes = [], [], []
    for n in episodes:
        ep = show.episode(n)
        runs = {}
        for variant in ("a", "b"):
            good = []
            for i, pick in enumerate(answers[(n, variant)].picks, 1):
                g, why = to_gold(pick, ep, f"L{n:03d}{variant}{i}", f"llm-{variant}",
                                 cfg.promo.min_duration_s, cfg.promo.max_duration_s)
                if g:
                    good.append(g)
                else:
                    notes.append(f"ep {n} variant {variant} pick {i} dropped: {why}")
            runs[variant] = good
        for g in runs["a"]:
            partner = next((b for b in runs["b"] if agree(g, b)), None)
            g.notes = f"agreed with {partner.id}" if partner else "variant A only"
            everything.append(g)
            if partner:
                agreed.append(GoldPromo(id=f"L{len(agreed) + 1:02d}", segments=g.segments, hook=g.hook,
                                        cliffhanger=g.cliffhanger, why=g.why, source="llm-agreed",
                                        confidence=round((g.confidence + partner.confidence) / 2),
                                        notes=f"A={g.id} B={partner.id}"))
        for b in runs["b"]:
            b.notes = "agreed" if any(agree(b, a) for a in runs["a"]) else "variant B only"
            everything.append(b)
    return agreed, everything, notes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--episodes", type=int, nargs="*", default=DEFAULT_EPISODES)
    ap.add_argument("--max-picks", type=int, default=3)
    ap.add_argument("--out", default="eval")
    args = ap.parse_args()
    cfg = load_config(overrides={"llm.mode": "live"})  # a rule-based gold set would grade the rules with themselves
    show = load_show(args.input)
    log = Path(cfg.paths.output_dir) / "gold_llm" / datetime.now().strftime("%Y%m%d-%H%M%S")
    llm = LLMClient(cfg.llm, cache_dir=cfg.paths.cache_dir, log_dir=log / "raw_llm")
    print(f"gold model: {cfg.llm.model_for('gold')}  (tool's default: {cfg.llm.model_for('default')})")
    print(f"{len(args.episodes) * 2} calls, {cfg.llm.concurrency} at a time; finished answers are saved, "
          f"so Ctrl+C and rerunning resumes where it stopped.", flush=True)
    agreed, everything, notes = build_llm_gold(show, args.episodes, llm, cfg, args.max_picks)
    out = Path(args.out)
    save_gold(agreed, out / "gold_llm.csv")
    save_gold(everything, out / "gold_llm_all.csv")
    (log / "notes.txt").write_text("\n".join(notes), encoding="utf-8")
    a = sum(g.source == "llm-a" for g in everything); b = sum(g.source == "llm-b" for g in everything)
    print(f"variant A {a} picks, variant B {b} picks, agreed {len(agreed)} -> {out / 'gold_llm.csv'}")
    print(f"dropped invalid picks: {len(notes)} (see {log / 'notes.txt'}); cost ${llm.usage_report()['_total']['cost_usd']:.4f}")


if __name__ == "__main__":
    main()
