"""Stage 6 (also used inside Stage 4): hard rules from the brief, checked by code.

`violations()` returns the broken rules (empty list = valid). `flags()` returns
non-fatal warnings an editor should see. Nothing here is left to a model.
"""
from __future__ import annotations

from .config import PromoConfig
from .schema import Promo, Show
from .timing import EPS, cuts_mid_line


def _event(show: Show, event_id: str):
    ep_no = int(event_id[1:4])
    return show.episode(ep_no).event(event_id)


def violations(p: Promo, show: Show, cfg: PromoConfig) -> list[str]:
    v: list[str] = []
    if not p.segments:
        return ["no segments"]
    if len(p.segments) > cfg.max_segments:
        v.append(f"{len(p.segments)} segments > max {cfg.max_segments}")
    for i, s in enumerate(p.segments, 1):
        ep = show.episode(s.episode)
        if not (0 - EPS <= s.start < s.end <= ep.duration + EPS):
            v.append(f"segment {i} {s.start}-{s.end} outside episode {s.episode} (0-{ep.duration})")
        if len(p.segments) > 1 and s.duration < cfg.min_segment_s - EPS:
            v.append(f"segment {i} only {s.duration}s < {cfg.min_segment_s}s")
        if cuts_mid_line(ep, s):
            v.append(f"segment {i} cuts a spoken line at {cuts_mid_line(ep, s)}")
    if not (cfg.min_duration_s - EPS <= p.duration <= cfg.max_duration_s + EPS):
        v.append(f"duration {p.duration}s outside {cfg.min_duration_s}-{cfg.max_duration_s}s")

    first, last = p.segments[0], p.segments[-1]
    hook = _event(show, p.hook_event)
    if hook.episode != first.episode or not (first.start - EPS <= hook.start < first.end):
        v.append("hook event is not inside the first segment")
    elif hook.start - first.start > cfg.hook_window_s + EPS:
        v.append(f"hook starts {hook.start - first.start:.1f}s in (> {cfg.hook_window_s}s)")
    cliff = _event(show, p.cliff_event)
    if cliff.episode != last.episode or not (last.start - EPS <= cliff.start and cliff.end <= last.end + EPS):
        v.append("cliffhanger event is not inside the last segment")
    elif last.end - cliff.end > cfg.cliff_tail_s + EPS:
        v.append(f"clip runs {last.end - cliff.end:.1f}s past the cliffhanger (> {cfg.cliff_tail_s}s)")

    for i, a in enumerate(p.segments):
        for b in p.segments[i + 1:]:
            if a.episode == b.episode and a.start < b.end - EPS and b.start < a.end - EPS:
                v.append(f"segments overlap in episode {a.episode}")
    eps = [s.episode for s in p.segments]
    if len(set(eps)) > 1 and eps != sorted(eps):
        v.append("cross-episode segments are not in episode order")
    return v


def flags(p: Promo, show: Show) -> list[str]:
    f: list[str] = []
    hook = _event(show, p.hook_event)
    if hook.type != "dialogue":
        f.append("visual_hook")  # judged from a description: editor should check the footage
    for s in p.segments:
        bounds = set(show.episode(s.episode).shot_boundaries) | {0.0, show.episode(s.episode).duration}
        if not any(abs(s.start - b) < EPS for b in bounds) or not any(abs(s.end - b) < EPS for b in bounds):
            f.append("mid_shot_cut")
            break
    if len({s.episode for s in p.segments}) > 1:
        f.append("cross_episode")
    return f
