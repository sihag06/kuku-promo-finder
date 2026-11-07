"""Small helpers shared by the pipeline and the exporters."""
from __future__ import annotations

import re

from .schema import Promo, Show


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "show"


def mmss(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:05.2f}"


def clip_lines(p: Promo, show: Show, marks: bool = True) -> list[str]:
    """The exact script inside a promo, segment by segment (what a viewer would see).
    marks=False hides the tool's HOOK/CLIFFHANGER markers (for blind rating)."""
    out = []
    for i, s in enumerate(p.segments, 1):
        ep = show.episode(s.episode)
        out.append(f"-- segment {i}: Ep {s.episode} {mmss(s.start)}-{mmss(s.end)} ({s.duration:.1f}s)")
        for e in ep.events:
            if e.type == "camera" or e.end <= s.start or e.start >= s.end:
                continue
            mark = "" if not marks else " <== HOOK" if e.id == p.hook_event else " <== CLIFFHANGER" if e.id == p.cliff_event else ""
            who = f"{e.speaker}: " if e.type == "dialogue" else f"{e.type.upper()}: "
            out.append(f"   {mmss(e.start)} {who}{e.text}{mark}")
    return out
