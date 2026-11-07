"""Where a clip may start and end. Pure code.

A cut point is legal when it is a shot boundary (a CAMERA start/end) and no spoken
line is running across it. Clips snap to legal points so they never start mid-shot
or cut someone off mid-sentence. When no legal point exists close enough, we fall
back to a cut just outside the nearest line and flag it, instead of failing.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right

from .schema import Episode, Event, Segment

EPS = 0.05


def dialogue_spanning(ep: Episode, t: float) -> Event | None:
    """The spoken line running across time t, if any (start < t < end)."""
    for e in ep.events:
        if e.type == "dialogue" and e.start + EPS < t < e.end - EPS:
            return e
    return None


def legal_points(ep: Episode) -> list[float]:
    pts = sorted(set(ep.shot_boundaries) | {0.0, round(ep.duration, 2)})
    return [p for p in pts if dialogue_spanning(ep, p) is None]


def snap_start(ep: Episode, anchor: float, max_lead: float) -> tuple[float, bool]:
    """Latest legal point in [anchor - max_lead, anchor]. Returns (time, clean).
    clean=False means no shot boundary was close enough and we cut mid-shot."""
    pts = legal_points(ep)
    i = bisect_right(pts, anchor + EPS) - 1
    if i >= 0 and pts[i] >= anchor - max_lead - EPS:
        return pts[i], True
    t = max(0.0, anchor - 0.3)
    line = dialogue_spanning(ep, t)
    return (round(line.start, 2) if line else round(t, 2)), False


def snap_end(ep: Episode, anchor: float, max_tail: float) -> tuple[float, bool]:
    """Earliest legal point in [anchor, anchor + max_tail]. Returns (time, clean)."""
    pts = legal_points(ep)
    i = bisect_left(pts, anchor - EPS)
    if i < len(pts) and pts[i] <= anchor + max_tail + EPS:
        return pts[i], True
    t = min(ep.duration, anchor + 0.3)
    line = dialogue_spanning(ep, t)
    return (round(line.end, 2) if line else round(t, 2)), False


def snap_near(ep: Episode, target: float, lo: float, hi: float) -> float:
    """Legal point closest to target within [lo, hi]; target itself if none."""
    cands = [p for p in legal_points(ep) if lo - EPS <= p <= hi + EPS]
    return min(cands, key=lambda p: abs(p - target)) if cands else target


def make_segment(ep: Episode, start: float, end: float) -> Segment:
    inside = [e for e in ep.events if e.start >= start - EPS and e.end <= end + EPS]
    if not inside:  # degenerate: take whatever overlaps
        inside = [e for e in ep.events if e.end > start and e.start < end] or ep.events[:1]
    return Segment(episode=ep.number, start=round(start, 2), end=round(end, 2),
                   first_event=inside[0].id, last_event=inside[-1].id)


def cuts_mid_line(ep: Episode, seg: Segment) -> list[float]:
    """Cut points of this segment that fall inside a spoken line."""
    return [t for t in (seg.start, seg.end) if dialogue_spanning(ep, t) is not None]
