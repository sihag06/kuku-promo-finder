"""Gold promos (the reference set) and the rule for "the tool found this one".

A gold CSV has one row per segment (see eval/gold_human_template.csv):
gold_id, segment, episode, start, end, hook, cliffhanger, why, confidence, notes

Matching a tool promo to a gold promo, two levels:
  strict: same opening (first segments in the same episode, starts within 3s)
          AND at least 50% of the gold promo's footage is inside the tool promo
  loose:  same opening OR at least 30% footage overlap
Strict is the headline number; loose shows "found the right scene, cut it differently".
"""
from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


def to_seconds(v: str) -> float:
    v = v.strip()
    if ":" not in v:
        return float(v)
    m, s = v.split(":", 1)
    return round(int(m) * 60 + float(s), 2)


def to_mmss(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:05.2f}"


@dataclass
class GoldPromo:
    id: str
    segments: list[tuple[int, float, float]]
    hook: str = ""
    cliffhanger: str = ""
    why: str = ""
    confidence: int = 3
    source: str = "human"
    notes: str = ""

    @property
    def duration(self) -> float:
        return round(sum(e - s for _, s, e in self.segments), 2)

    @property
    def episodes(self) -> set[int]:
        return {ep for ep, _, _ in self.segments}


def load_gold(path: str | Path, source: str = "human") -> list[GoldPromo]:
    rows: dict[str, list[dict]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r.get("gold_id") or "EXAMPLE" in (r.get("notes") or "").upper():
                continue
            rows[r["gold_id"].strip()].append(r)
    out = []
    for gid, rs in rows.items():
        rs.sort(key=lambda r: int(r.get("segment") or 1))
        segs = [(int(r["episode"]), to_seconds(r["start"]), to_seconds(r["end"])) for r in rs]
        first = rs[0]
        out.append(GoldPromo(id=gid, segments=segs, hook=first.get("hook", ""),
                             cliffhanger=first.get("cliffhanger", ""), why=first.get("why", ""),
                             confidence=int(first.get("confidence") or 3), source=source,
                             notes=first.get("notes", "")))
    return out


def save_gold(promos: list[GoldPromo], path: str | Path) -> None:
    cols = ["gold_id", "segment", "episode", "start", "end", "hook", "cliffhanger", "why", "confidence", "notes"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for g in promos:
            for i, (ep, s, e) in enumerate(g.segments, 1):
                w.writerow({"gold_id": g.id, "segment": i, "episode": ep, "start": to_mmss(s), "end": to_mmss(e),
                            "hook": g.hook, "cliffhanger": g.cliffhanger, "why": g.why,
                            "confidence": g.confidence, "notes": g.notes})


def _segs(p) -> list[tuple[int, float, float]]:
    """Accept a GoldPromo, a Promo, or a promos.json dict."""
    if isinstance(p, GoldPromo):
        return p.segments
    segs = p["segments"] if isinstance(p, dict) else [s.model_dump() for s in p.segments]
    return [(s["episode"], s["start"], s["end"]) for s in segs]


def overlap_seconds(a, b) -> float:
    total = 0.0
    for ea, sa, ta in _segs(a):
        for eb, sb, tb in _segs(b):
            if ea == eb:
                total += max(0.0, min(ta, tb) - max(sa, sb))
    return total


def same_opening(a, b, within: float = 3.0) -> bool:
    ea, sa, _ = _segs(a)[0]
    eb, sb, _ = _segs(b)[0]
    return ea == eb and abs(sa - sb) <= within


def match_level(tool, gold: GoldPromo) -> str | None:
    """'strict', 'loose' or None."""
    cover = overlap_seconds(tool, gold) / max(gold.duration, 1e-6)
    opening = same_opening(tool, gold)
    if opening and cover >= 0.5:
        return "strict"
    if opening or cover >= 0.3:
        return "loose"
    return None
