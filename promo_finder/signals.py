"""Stage 2: rule-based hook / cliffhanger signals for every event. No LLM.

Why rules at all, when an LLM reads the script anyway?
- They are free, instant and explainable ("hook 0.82: violent action, shocked reaction").
- They give the LLM a second opinion, and the ranker an agreement score.
- With no API key (mock mode) they drive the whole pipeline: the heuristics-only baseline.

Each cue adds evidence; cues combine with a noisy-OR, 1 - prod(1 - w), so several
weak cues add up but the score never exceeds 1. Word lists are deliberately generic
microdrama vocabulary, not tuned to one show.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from pydantic import BaseModel

from .schema import Episode, Event

# (weight, label, pattern). Matched case-insensitively on whole words / phrases.
HOOK_ACTION = [
    (0.55, "violence", r"slap\w*|strik\w*|struck|punch\w*|hit[s]?|shov\w*|kick\w*|stab\w*|chok\w*|throttl\w*|"
                       r"backhand\w*|whip\w*|lash\w*|beat[s]?|brand\w*|grabs? (?:her|him) by|drags?|collaps\w*"),
    (0.45, "danger", r"explosion|explod\w*|shatter\w*|fire|flames?|blaze|burn\w*|blood\w*|knife|dagger|sword|"
                     r"pistol|gun\w*|blade|poison\w*|chains?|chained|noose|gallows|dungeon"),
    (0.35, "public humiliation", r"humiliat\w*|kneel\w*|on (?:her|his) knees|thrown|throws? .* (?:mud|floor|ground)|"
                                 r"spit\w*|tears? (?:off|apart)|rips?|mock\w*|laugh\w* at"),
    (0.30, "romance spike", r"kiss\w*|embrac\w*|pulls? (?:her|him) close|pinned|bare chest\w*|undress\w*"),
    (0.25, "sudden entrance", r"burst\w* (?:in|open|through)|storm\w* in|doors? (?:fly|flies|slam\w*|bang\w*)|"
                              r"barges? in"),
]
HOOK_SFX = [
    (0.45, "impact sound", r"slap|crack|thud|punch|smash|crash|shatter|gunshot|explosion|bang|scream\w*"),
    (0.30, "crowd shock", r"gasp\w*|murmur\w*|uproar"),
]
HOOK_LINE = [
    (0.45, "reveal", r"you(?:'re| are) (?:the|my|not)|i am (?:the|your)|is (?:my|your|the) (?:real|true|son|daughter|"
                     r"husband|wife|father|mother)|pregnant|alive|was never|the duke|heir|disinherit\w*|disowned?"),
    (0.40, "accusation", r"thief|liar|witch\w*|murder\w*|traitor|whore|harlot|betray\w*|you knew|how dare|"
                         r"poisoned?|stole|framed?"),
    (0.35, "rejection", r"i (?:don't|do not|never) (?:love|want)|changed my mind|marrying (?:me|her)|"
                        r"call(?:ing)? off|get out|you are nothing|worthless|plain\w*"),
    (0.30, "threat", r"i(?:'ll| will) (?:kill|destroy|ruin|make you)|you(?:'ll| will) (?:pay|die|regret|hang|burn)|"
                     r"or else|execut\w*|hang\w*"),
]
HOT_EMOTIONS = (r"furious|enraged|rage|screaming|shouting|shrieking|outraged|horrified|shocked|stunned|"
                r"devastated|venomous|vicious|cruel|menacing|threatening|seething|hysterical|sobbing|"
                r"desperate|gloating|smug|contempt\w*|disgusted|livid|icy|cold")

CLIFF_LINE = [
    (0.40, "threat/ultimatum", r"or else|if you|unless|you(?:'ll| will) (?:pay|die|regret|hang|burn|see)|"
                               r"i(?:'ll| will) (?:kill|destroy|ruin|find|make)|before (?:dawn|sunrise|nightfall)|"
                               r"tomorrow|execut\w*|hang\w*|burn\w*"),
    (0.35, "reveal pending", r"the truth|who (?:you|he|she) really|you don't know|there's something|secret|"
                             r"^(?:wait|stop|no)[\s.!?…]*$|\bno[.!…]*$"),
]
CLIFF_ACTION = [
    (0.40, "interruption", r"burst\w*|storm\w* in|doors? (?:open|fly|slam\w*|bang\w*)|freez\w*|stops? (?:dead|short|cold)|"
                           r"turns? (?:sharply|slowly|around)|a (?:voice|figure|shadow)|appears?|arriv\w*|enters?"),
    (0.35, "danger rising", r"raises? (?:the|a|his|her)|draws? (?:a|his|her) (?:sword|knife|blade|pistol)|aims?|"
                            r"lunges?|grabs?|seiz\w*|arrest\w*|guards? (?:rush|seize|grab)"),
    (0.45, "cut to black", r"cut to black|fade[s]? (?:out|to black)|smash cut|freeze frame|end on|hold on"),
]
CLIFF_SFX = [
    (0.45, "ominous cue", r"cliffhanger|ominous|swell\w*|sting|drone|silence|stops abruptly|heartbeat|thunder"),
]


def _cues(text: str, table: list[tuple[float, str, str]]) -> list[tuple[float, str]]:
    out = []
    for w, label, pat in table:
        if re.search(rf"(?<![a-z])(?:{pat})(?![a-z])", text, re.I):
            out.append((w, label))
    return out


def _noisy_or(cues: list[tuple[float, str]]) -> float:
    p = 1.0
    for w, _ in cues:
        p *= (1 - w)
    return round(1 - p, 3)


class EventSignal(BaseModel):
    event_id: str
    hook: float
    cliff: float
    hook_reasons: list[str]
    cliff_reasons: list[str]


def score_event(e: Event, ep_duration: float) -> EventSignal:
    hook: list[tuple[float, str]] = []
    cliff: list[tuple[float, str]] = []
    text = e.text
    if e.type in ("action", "camera"):
        table_w = 1.0 if e.type == "action" else 0.6  # camera lines describe, actions happen
        hook += [(w * table_w, l) for w, l in _cues(text, HOOK_ACTION)]
        cliff += [(w * table_w, l) for w, l in _cues(text, CLIFF_ACTION)]
        cliff += [(w * 0.8, l) for w, l in _cues(text, CLIFF_SFX)]  # "music hits a cliffhanger"
        if e.type == "camera" and re.search(r"extreme close|push(?:es)?[- ]in|slow[- ]motion|whip", text, re.I):
            hook.append((0.10, "intense framing"))
    elif e.type == "sfx":
        hook += _cues(text, HOOK_SFX)
        cliff += _cues(text, CLIFF_SFX)
    elif e.type == "dialogue":
        hook += _cues(text, HOOK_LINE)
        cliff += _cues(text, CLIFF_LINE)
        if text.rstrip(' "\'”').endswith("?"):  # punctuation: checked in code, not by word regex
            cliff.append((0.40, "open question"))
        emo = e.emotion or ""
        if re.search(HOT_EMOTIONS, emo, re.I):
            hook.append((0.25, f"hot delivery ({emo})"))
        if text.endswith("!"):
            hook.append((0.15, "exclamation"))
        if 1 <= len(text.split()) <= 6:
            hook.append((0.10, "short punchy line"))
    # Writers build every episode toward its last beat: strong prior for an ending.
    if ep_duration - e.end <= 8.0:
        cliff.append((0.35, "episode ending"))
    return EventSignal(event_id=e.id, hook=_noisy_or(hook), cliff=_noisy_or(cliff),
                       hook_reasons=[l for _, l in hook], cliff_reasons=[l for _, l in cliff])


@dataclass
class EpisodeSignals:
    episode: int
    by_event: dict[str, EventSignal] = field(default_factory=dict)

    def top(self, kind: str, k: int = 6, min_gap_s: float = 4.0, events: list[Event] | None = None) -> list[EventSignal]:
        """Best k events for 'hook' or 'cliff', skipping near-duplicates within min_gap_s."""
        ranked = sorted(self.by_event.values(), key=lambda s: -getattr(s, kind))
        if events is None:
            return ranked[:k]
        start = {e.id: e.start for e in events}
        picked: list[EventSignal] = []
        for s in ranked:
            if getattr(s, kind) <= 0:
                break
            if all(abs(start[s.event_id] - start[p.event_id]) >= min_gap_s for p in picked):
                picked.append(s)
            if len(picked) == k:
                break
        return picked


def score_episode(ep: Episode) -> EpisodeSignals:
    sig = EpisodeSignals(episode=ep.number)
    for e in ep.events:
        sig.by_event[e.id] = score_event(e, ep.duration)
    return sig


if __name__ == "__main__":  # python -m promo_finder.signals data --episodes 2 3
    import argparse

    from .parser import load_show

    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("-k", type=int, default=4)
    args = ap.parse_args()
    show = load_show(args.input)
    for ep in show.episodes:
        if args.episodes and ep.number not in args.episodes:
            continue
        s = score_episode(ep)
        print(f"\n=== Episode {ep.number} ({ep.duration:.0f}s)")
        for kind in ("hook", "cliff"):
            print(f"  top {kind}s:")
            for x in s.top(kind, args.k, events=ep.events):
                e = ep.event(x.event_id)
                reasons = x.hook_reasons if kind == "hook" else x.cliff_reasons
                print(f"   {getattr(x, kind):.2f} {e.start:6.1f}s {e.type:<8} {e.text[:60]!r:<64} {reasons}")
