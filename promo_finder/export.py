"""Final outputs an editor can use directly.

promos_ranked.csv   one row per promo: rank, tier, every segment's episode/start/end,
                    hook, cliffhanger, why, scores, flags, rank reason (opens in Sheets/Excel)
promos_ranked.md    the top list with the exact lines of each clip, then the rest as a table
promos.json         everything, machine-readable
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .pipeline_utils import clip_lines, mmss
from .schema import Promo, Show

MAX_SEGS = 3


def seg_text(p: Promo) -> str:
    return " | ".join(f"Ep{s.episode} {mmss(s.start)}-{mmss(s.end)}" for s in p.segments)


def hook_offset(p: Promo, show: Show) -> float:
    e = show.episode(int(p.hook_event[1:4])).event(p.hook_event)
    return round(e.start - p.segments[0].start, 2)


def write_csv(promos: list[Promo], show: Show, path: Path) -> None:
    cols = ["rank", "quality_rank", "tier", "promo_id", "score", "quality", "template", "duration_s", "n_segments", "segments"]
    for i in range(1, MAX_SEGS + 1):
        cols += [f"seg{i}_episode", f"seg{i}_start", f"seg{i}_end"]
    cols += ["hook", "cliffhanger", "why", "hook_offset_s", "hook_score", "cliffhanger_score",
             "standalone_score", "emotion_score", "funnel", "spoiler_risk", "ad_safety",
             "flags", "confusions", "rank_reason"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for p in promos:
            row = {"rank": p.rank, "quality_rank": p.quality_rank, "tier": p.tier, "promo_id": p.id,
                   "score": p.scores.total, "quality": p.scores.quality,
                   "template": p.template, "duration_s": p.duration, "n_segments": len(p.segments),
                   "segments": seg_text(p), "hook": p.hook, "cliffhanger": p.cliffhanger, "why": p.why,
                   "hook_offset_s": hook_offset(p, show),
                   "hook_score": round(1 + 4 * p.scores.hook), "cliffhanger_score": round(1 + 4 * p.scores.cliffhanger),
                   "standalone_score": round(1 + 4 * p.scores.standalone),
                   "emotion_score": round(1 + 4 * p.scores.emotion), "funnel": p.scores.funnel,
                   "spoiler_risk": p.spoiler_risk, "ad_safety": p.ad_safety,
                   "flags": ", ".join(p.flags), "confusions": "; ".join(p.confusions),
                   "rank_reason": p.rank_reason}
            for i, s in enumerate(p.segments[:MAX_SEGS], 1):
                row.update({f"seg{i}_episode": s.episode, f"seg{i}_start": mmss(s.start), f"seg{i}_end": mmss(s.end)})
            w.writerow(row)


def write_markdown(promos: list[Promo], show: Show, path: Path, top_k: int, header: list[str]) -> None:
    md = [f"# Ranked promos: {show.title}", *header, "",
          "Order = suggested posting order (quality + early-episode funnel + variety). "
          "`quality #` = rank on promo quality alone. Tiers: **post first** (1-5), **test next** (6-%d), "
          "backlog, **hold** (unsafe or major spoiler)." % top_k, ""]
    for p in promos[:top_k]:
        md += [f"## #{p.rank} {p.id} | {p.tier} | {p.template} | {p.duration:.1f}s | score {p.scores.total} "
               f"| quality #{p.quality_rank} ({p.scores.quality})",
               f"**Segments:** {seg_text(p)}  ",
               f"**Hook** ({hook_offset(p, show):.1f}s in): {p.hook}  ",
               f"**Cliffhanger:** {p.cliffhanger}  ",
               f"**Why it works:** {p.why}  ",
               f"**Why this rank:** {p.rank_reason}  "]
        if p.cold_summary:
            md.append(f"**Cold viewer read:** {p.cold_summary}  ")
        if p.confusions:
            md.append(f"**Could confuse a stranger:** {'; '.join(p.confusions)}  ")
        if p.flags:
            md.append(f"**Flags:** {', '.join(p.flags)}  ")
        md += ["", "```", *clip_lines(p, show), "```", ""]
    rest = promos[top_k:]
    if rest:
        md += ["## All other candidates", "", "| # | id | tier | score | template | segments | hook |",
               "|---|---|---|---|---|---|---|"]
        md += [f"| {p.rank} | {p.id} | {p.tier} | {p.scores.total} | {p.template} | {seg_text(p)} | "
               f"{p.hook.replace('|', '/')[:70]} |" for p in rest]
    path.write_text("\n".join(md), encoding="utf-8")


def write_all(promos: list[Promo], show: Show, out: Path, top_k: int, header: list[str]) -> None:
    write_csv(promos, show, out / "promos_ranked.csv")
    write_markdown(promos, show, out / "promos_ranked.md", top_k, header)
    rows = []
    for p in promos:  # include the clip's lines so a saved run can be reviewed without the scripts
        d = p.model_dump(mode="json")
        d["clip_lines"] = clip_lines(p, show)
        rows.append(d)
    (out / "promos.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
