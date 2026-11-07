"""Stage 4: turn hook + cliffhanger moments into promo candidates. Pure code.

Four templates (see schema.PromoTemplate):
  continuous     [hook ... cliffhanger]                      one stretch
  condensed      [hook snippet] + [build-up -> cliffhanger]   same episode, skips the middle
  cold_open      [later hook] + [earlier build-up -> cliff]   flash-forward, same episode
  cross_episode  [hook in ep N] + [build-up -> cliff in N+1..N+span]

Every candidate is snapped to legal cut points (timing.py), checked against the
hard rules (validate.py), pre-scored, de-duplicated and pruned so only the best go
to the paid judge stage. Rejected candidates are kept with their reasons.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import PromoConfig
from .miner import MinedEpisode, Moment
from .schema import Episode, Event, Promo, Show
from .signals import EpisodeSignals
from .timing import make_segment, snap_end, snap_near, snap_start
from .validate import flags, violations

TEMPLATE_PENALTY = {"continuous": 0.0, "condensed": 0.03, "cold_open": 0.03, "cross_episode": 0.05}
TARGET_TOTAL = 35.0  # aim stitched promos at the middle of the 20-60s window


@dataclass
class Pick:
    ep: Episode
    event: Event
    m: Moment


@dataclass
class Rejected:
    template: str
    hook_event: str
    cliff_event: str
    reasons: list[str]


def _pre_score(h: Pick, c: Pick, template: str, sigs: dict[int, EpisodeSignals], duration: float) -> float:
    sh = sigs[h.ep.number].by_event[h.event.id].hook
    sc = sigs[c.ep.number].by_event[c.event.id].cliff
    length_fit = 1.0 - min(abs(duration - TARGET_TOTAL) / 25.0, 1.0)
    return round(0.40 * h.m.strength / 5 + 0.35 * c.m.strength / 5 + 0.10 * sh + 0.10 * sc
                 + 0.05 * length_fit - TEMPLATE_PENALTY[template], 4)


def _hook_snippet(h: Pick, cfg: PromoConfig, target: float, max_len: float):
    """A short first segment: a beat before the hook, through the hook and its reaction."""
    start, clean = snap_start(h.ep, h.event.start, max_lead=cfg.hook_window_s - 0.2)
    want = max(h.event.end + 1.0, start + target)
    end = snap_near(h.ep, want, lo=max(h.event.end, start + cfg.min_segment_s), hi=start + max_len)
    end, _ = snap_end(h.ep, end, max_tail=2.0)
    return start, end, clean


def _buildup(c: Pick, cfg: PromoConfig, length: float, not_before: float, not_after: float | None = None):
    """A segment that ends just after the cliffhanger, starting about `length` earlier."""
    end, clean = snap_end(c.ep, c.event.end, max_tail=cfg.cliff_tail_s)
    if not_after is not None:
        end = min(end, not_after)
    start = snap_near(c.ep, end - length, lo=max(not_before, 0.0), hi=c.event.start - 1.0)
    return max(start, not_before), end, clean


def build(template: str, h: Pick, c: Pick, cfg: PromoConfig) -> tuple[list, bool] | None:
    """Return ([(episode, start, end), ...], clean_cuts) or None if the shape doesn't fit."""
    if template == "continuous":
        if h.ep.number != c.ep.number or c.event.start <= h.event.start:
            return None
        s, ok1 = snap_start(h.ep, h.event.start, max_lead=cfg.hook_window_s - 0.2)
        e, ok2 = snap_end(c.ep, c.event.end, max_tail=cfg.cliff_tail_s)
        return [(h.ep, s, e)], ok1 and ok2

    if template in ("condensed", "cold_open"):
        if h.ep.number != c.ep.number:
            return None
        hs, he, ok1 = _hook_snippet(h, cfg, target=6.0, max_len=10.0)
        budget = max(cfg.min_segment_s, min(TARGET_TOTAL, cfg.max_duration_s) - (he - hs))
        if template == "condensed":  # hook first in story order, then jump ahead
            if c.event.start <= he + 5.0:
                return None  # cliff too close: that's just a continuous clip
            bs, be, ok2 = _buildup(c, cfg, budget, not_before=he + 1.0)
        else:  # cold open: the hook happens AFTER the cliffhanger; show it first
            if h.event.start <= c.event.end:
                return None
            bs, be, ok2 = _buildup(c, cfg, budget, not_before=0.0, not_after=hs)
        return [(h.ep, hs, he), (c.ep, bs, be)], ok1 and ok2

    if template == "cross_episode":
        if not (0 < c.ep.number - h.ep.number <= cfg.cross_episode_span):
            return None
        hs, he, ok1 = _hook_snippet(h, cfg, target=12.0, max_len=20.0)
        budget = max(cfg.min_segment_s, TARGET_TOTAL - (he - hs))
        bs, be, ok2 = _buildup(c, cfg, budget, not_before=0.0)
        return [(h.ep, hs, he), (c.ep, bs, be)], ok1 and ok2
    raise ValueError(template)


def extend_short(h: Pick, start: float, cfg: PromoConfig, sig: EpisodeSignals,
                 max_total: float = 45.0) -> Pick | None:
    """A strong hook paired with a too-early cliffhanger gives a clip under 20s. Instead of
    throwing the hook away, look ahead for the next unresolved line in reach (a question, an
    exclamation, or a line the rules mark as a cliffhanger) that makes a 20-45s clip."""
    ep = h.ep
    lo, hi = start + cfg.min_duration_s, start + max_total
    for e in ep.events:
        if e.type != "dialogue" or e.start <= h.event.end or e.end < lo - 0.5 or e.end > hi:
            continue
        cue = sig.by_event[e.id].cliff
        if e.text.rstrip(' "\'”').endswith(("?", "!")) or cue >= 0.35:
            m = Moment(event_id=e.id, type="mystery", quote=e.text[:100],
                       strength=max(2, min(4, round(1 + 4 * max(cue, 0.4)))),
                       why="extended: next unresolved line after a too-early cliffhanger")
            return Pick(ep, e, m)
    return None


def _overlap(a: Promo, b: Promo) -> float:
    inter = 0.0
    for x in a.segments:
        for y in b.segments:
            if x.episode == y.episode:
                inter += max(0.0, min(x.end, y.end) - max(x.start, y.start))
    return inter / max(min(a.duration, b.duration), 1e-6)


def _same_opening(a: Promo, b: Promo, within: float = 3.0) -> bool:
    """The first seconds are what a viewer sees; two clips that open on the same
    moment are the same ad, even if they end differently."""
    x, y = a.segments[0], b.segments[0]
    return a.hook_event == b.hook_event or (x.episode == y.episode and abs(x.start - y.start) <= within)


def dedupe(cands: list[Promo], threshold: float) -> tuple[list[Promo], list[Promo]]:
    """Drop a candidate when a better one opens the same way AND mostly overlaps it.
    Clips that share footage but open differently are different ads, so both stay."""
    kept: list[Promo] = []
    dropped: list[Promo] = []
    for p in sorted(cands, key=lambda p: -p.pre_score):
        dup = any(_same_opening(p, k) and _overlap(p, k) > threshold for k in kept)
        (dropped if dup else kept).append(p)
    return kept, dropped


def select(raw: list[Promo], cfg: PromoConfig) -> tuple[list[Promo], list[Rejected]]:
    """One greedy pass, best pre-score first: a candidate is dropped as a duplicate only of a
    clip that is actually KEPT, then the variety and cost quotas apply. (Doing dedupe and
    pruning in two passes lost clips that were "duplicates" of something later pruned.)"""
    rejected: list[Rejected] = []
    per_hook: dict[str, int] = {}
    per_ep: dict[int, int] = {}
    final: list[Promo] = []
    for p in sorted(raw, key=lambda p: -p.pre_score):
        ep_no = p.segments[0].episode
        reason = None
        if any(_same_opening(p, k) and _overlap(p, k) > cfg.overlap_dedupe for k in final):
            reason = "near-duplicate of a better candidate"
        elif per_hook.get(p.hook_event, 0) >= cfg.max_per_hook:
            reason = "pruned: hook moment already used enough"
        elif per_ep.get(ep_no, 0) >= cfg.candidates_per_episode:
            reason = "pruned: episode quota reached"
        elif len(final) >= cfg.max_candidates:
            reason = "pruned: overall candidate cap reached"
        if reason:
            rejected.append(Rejected(p.template, p.hook_event, p.cliff_event, [reason]))
            continue
        per_hook[p.hook_event] = per_hook.get(p.hook_event, 0) + 1
        per_ep[ep_no] = per_ep.get(ep_no, 0) + 1
        final.append(p)
    return final, rejected


def assemble(show: Show, mined: dict[int, MinedEpisode], sigs: dict[int, EpisodeSignals],
             cfg: PromoConfig) -> tuple[list[Promo], list[Rejected]]:
    picks_h: dict[int, list[Pick]] = {}
    picks_c: dict[int, list[Pick]] = {}
    for ep in show.episodes:
        md = mined.get(ep.number)
        if not md:
            continue
        picks_h[ep.number] = [Pick(ep, ep.event(m.event_id), m) for m in md.hooks]
        picks_c[ep.number] = [Pick(ep, ep.event(m.event_id), m) for m in md.cliffhangers]

    raw: list[Promo] = []
    rejected: list[Rejected] = []
    n = 0
    for ep_no, hooks in picks_h.items():
        pairs = [(h, c, t) for h in hooks for c in picks_c.get(ep_no, [])
                 for t in ("continuous", "condensed", "cold_open")]
        for later in range(ep_no + 1, ep_no + cfg.cross_episode_span + 1):
            pairs += [(h, c, "cross_episode") for h in hooks[:3] for c in picks_c.get(later, [])[:3]]
        for h, c, template in pairs:
            built = build(template, h, c, cfg)
            if built is None:
                continue
            spans, clean = built
            if template == "continuous" and spans[0][2] - spans[0][1] < cfg.min_duration_s:
                ext = extend_short(h, spans[0][1], cfg, sigs[h.ep.number])
                if ext is not None:
                    c = ext
                    end, ok = snap_end(c.ep, c.event.end, max_tail=cfg.cliff_tail_s)
                    spans, clean = [(h.ep, spans[0][1], end)], clean and ok
                    template = "continuous"
            n += 1
            segs = [make_segment(ep, s, e) for ep, s, e in spans]
            p = Promo(id=f"C{n:05d}", template=template, segments=segs,
                      hook_event=h.event.id, cliff_event=c.event.id,
                      hook=f'{h.m.type}: "{h.m.quote}"', cliffhanger=f'{c.m.type}: "{c.m.quote}"',
                      why=f"hook: {h.m.why} | cliff: {c.m.why}", hook_type=h.m.type, cliff_type=c.m.type)
            bad = violations(p, show, cfg)
            if bad:
                rejected.append(Rejected(template, h.event.id, c.event.id, bad))
                continue
            p.flags = flags(p, show) + ([] if clean else ["snapped_fallback"])
            p.pre_score = _pre_score(h, c, template, sigs, p.duration)
            raw.append(p)

    final, dropped = select(raw, cfg)
    rejected += dropped

    final.sort(key=lambda p: (p.segments[0].episode, p.segments[0].start))
    for i, p in enumerate(final, 1):
        p.id = f"P{p.segments[0].episode:03d}-{i:03d}"
    return final, rejected
