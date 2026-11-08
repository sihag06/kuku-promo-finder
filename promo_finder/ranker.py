"""Stage 7: score, order and explain. Pure code, weights in config.yaml.

Two orders, because they answer two different questions:
  quality_rank   which promos are BEST?  hook, cliffhanger, works-cold, emotion only
  rank           which to POST FIRST?    quality + early-episode funnel + rule agreement,
                                         then MMR for variety (the brief's "post first, and why")

score = sum(weight_i * component_i) - spoiler_penalty - safety_penalty
  hook, cliffhanger, standalone, emotion  from the judge (1-5 scaled to 0-1)
  funnel             earlier source episode = better: a new viewer lands on Ep 1
  signal_agreement   do the rule signals agree the hook/cliff are strong?

Posting order then applies MMR (maximal marginal relevance): each next pick trades
its score against how similar it is to promos already picked, so the top of the
list is varied (different tropes and story phases) rather than five takes of one scene.
"""
from __future__ import annotations

from .config import RankingConfig
from .judge import Judgement
from .schema import Promo, Scores, Show
from .signals import EpisodeSignals

QUALITY_KEYS = ("hook", "cliffhanger", "standalone", "emotion")
CONFUSION_STEP = 0.1  # each listed confusion beyond the first caps "works cold" 0.1 lower
SPOILER_PENALTY = {0: 0.0, 1: 0.05, 2: 0.15}
SAFETY_PENALTY = {"ok": 0.0, "review": 0.05, "unsafe": 0.30}
LABELS = {"hook": "hook", "cliffhanger": "cliffhanger", "standalone": "works cold",
          "emotion": "emotional charge", "funnel": "early-episode funnel", "signal_agreement": "rule signals agree"}


def funnel(p: Promo, episode_numbers: list[int]) -> float:
    """1.0 for a promo from the earliest episode in the input, 0.0 for the latest.
    Uses the episodes actually present (an upload may be Episodes 28-29 only), clamped to 0-1."""
    lo, hi = min(episode_numbers), max(episode_numbers)
    first = min(s.episode for s in p.segments)
    if hi == lo:
        return 1.0
    return round(min(1.0, max(0.0, 1.0 - (first - lo) / (hi - lo))), 3)


def apply_judgement(p: Promo, j: Judgement, show: Show, sigs: dict[int, EpisodeSignals],
                    cfg: RankingConfig) -> Promo:
    s01 = lambda x: (x - 1) / 4
    h_ep, c_ep = int(p.hook_event[1:4]), int(p.cliff_event[1:4])
    agree = (sigs[h_ep].by_event[p.hook_event].hook + sigs[c_ep].by_event[p.cliff_event].cliff) / 2
    # Keep the judge consistent with itself: a clip it says is confusing in many ways
    # cannot also score high on "makes sense on its own".
    standalone = min(s01(j.standalone_score), max(0.0, 1.0 - CONFUSION_STEP * max(0, len(j.confusions) - 1)))
    sc = Scores(hook=s01(j.hook_score), cliffhanger=s01(j.cliffhanger_score),
                standalone=round(standalone, 3), emotion=s01(j.emotion_score),
                funnel=funnel(p, [e.number for e in show.episodes]), signal_agreement=round(agree, 3),
                spoiler_penalty=SPOILER_PENALTY[j.spoiler_risk], safety_penalty=SAFETY_PENALTY[j.ad_safety])
    w = cfg.weights
    penalties = sc.spoiler_penalty + sc.safety_penalty
    qw = sum(w.get(k, 0) for k in QUALITY_KEYS) or 1.0
    sc.quality = round(sum(w.get(k, 0) * getattr(sc, k) for k in QUALITY_KEYS) / qw - penalties, 4)
    sc.total = round(sum(w.get(k, 0) * getattr(sc, k) for k in LABELS) - penalties, 4)
    flags = list(p.flags) + (["ad_safety_" + j.ad_safety] if j.ad_safety != "ok" else []) \
        + (["spoiler"] if j.spoiler_risk == 2 else [])
    return p.model_copy(update={
        "scores": sc, "hook": j.hook or p.hook, "cliffhanger": j.cliffhanger or p.cliffhanger,
        "why": j.why or p.why, "cold_summary": j.cold_summary, "confusions": j.confusions,
        "want_next": j.want_next, "spoiler_risk": j.spoiler_risk, "ad_safety": j.ad_safety,
        "ad_safety_reason": j.ad_safety_reason, "flags": flags})


def similarity(a: Promo, b: Promo) -> float:
    """0-1: how much two promos would feel like the same ad to a viewer."""
    sim = 0.0
    if a.hook_type and a.hook_type == b.hook_type:
        sim += 0.35
    if a.cliff_type and a.cliff_type == b.cliff_type:
        sim += 0.15
    ea, eb = a.segments[0].episode, b.segments[0].episode
    sim += 0.35 if abs(ea - eb) <= 1 else 0.15 if abs(ea - eb) <= 3 else 0.0
    if a.hook_event == b.hook_event or a.cliff_event == b.cliff_event:
        sim += 0.4
    return min(sim, 1.0)


def rank_reason(p: Promo, rank: int, w: dict[str, float], diversity_pick: bool) -> str:
    contrib = sorted(((w.get(k, 0) * getattr(p.scores, k), k) for k in LABELS), reverse=True)
    best = [LABELS[k] for _, k in contrib[:2]]
    eps = sorted({s.episode for s in p.segments})
    src = f"Ep {eps[0]}" if len(eps) == 1 else "Eps " + "+".join(map(str, eps))
    parts = [f"#{rank}: strongest on {best[0]} and {best[1]}", f"{src}, {p.template}, {p.duration:.0f}s"]
    if diversity_pick:
        parts.append("picked for variety over a similar higher scorer")
    if p.ad_safety != "ok":
        parts.append(f"ad safety: {p.ad_safety} ({p.ad_safety_reason})")
    if p.spoiler_risk:
        parts.append("spoiler risk " + ("major" if p.spoiler_risk == 2 else "minor"))
    return "; ".join(parts)


def rank(promos: list[Promo], cfg: RankingConfig) -> list[Promo]:
    """Order everything; the order IS the suggested posting order."""
    lam = cfg.diversity_lambda
    # Unsafe or major-spoiler promos go to the bottom ("hold"), whatever their score.
    postable = [p for p in promos if p.ad_safety != "unsafe" and p.spoiler_risk < 2]
    held = sorted([p for p in promos if p not in postable], key=lambda p: -p.scores.total)
    pool = sorted(postable, key=lambda p: -p.scores.total)
    ordered: list[tuple[Promo, bool]] = []
    while pool:
        if not ordered:
            pick = pool[0]
        else:
            pick = max(pool, key=lambda p: (1 - lam) * p.scores.total
                       - lam * max(similarity(p, q) for q, _ in ordered))
        diversity_pick = pick is not pool[0]
        ordered.append((pick, diversity_pick))
        pool.remove(pick)
    out = []
    for i, (p, div) in enumerate(ordered + [(p, False) for p in held], 1):
        tier = ("hold" if p in held else "post first" if i <= 5 else
                "test next" if i <= cfg.top_k else "backlog")
        if tier != "hold" and p.ad_safety == "review" and i <= 5:
            tier = "post first (check safety)"
        out.append(p.model_copy(update={"rank": i, "tier": tier,
                                        "rank_reason": rank_reason(p, i, cfg.weights, div)}))
    by_quality = sorted(out, key=lambda p: (-p.scores.quality, p.rank))
    qrank = {p.id: i for i, p in enumerate(by_quality, 1)}
    return [p.model_copy(update={"quality_rank": qrank[p.id]}) for p in out]
