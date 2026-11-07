"""Stage 5: judge each candidate as a stranger would see it. One LLM call per candidate.

The judge is shown ONLY the lines inside the clip (no premise, no cast list, no other
episodes): that is the honest test of "makes sense on its own". It first writes what
it thinks is happening and what confuses it, then scores. In mock mode, rule-based
estimates stand in.

A keyword ad-safety check runs in code as well as in the model, so a risky clip is
flagged even if the model misses it.
"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from . import prompts
from .llm.client import LLMClient
from .schema import Promo, Show
from .show_bible import ShowBible
from .signals import EpisodeSignals

SAFETY_PATTERNS = (r"bare[- ]chest\w*|naked|nude|undress\w*|chemise|in (?:the )?bed|bedroom|"
                   r"thrust\w*|moan\w*|seduc\w*|red-hot|branding|brand(?:s|ed)? (?:her|him)|"
                   r"torture\w*|gore|corpse|blood-soaked|soaked in blood|noose|hang(?:s|ed|ing)? (?:her|him)")
CHILD_AT_RISK = r"(?:child|boy|girl|brother|seven-year-old).{0,60}(?:strik|beat|gaol|threat|hurt|whip)"


class Judgement(BaseModel):
    cold_summary: str = ""
    confusions: list[str] = Field(default_factory=list)
    want_next: str = ""
    hook_score: int = Field(ge=1, le=5)
    cliffhanger_score: int = Field(ge=1, le=5)
    standalone_score: int = Field(ge=1, le=5)
    emotion_score: int = Field(ge=1, le=5)
    spoiler_risk: int = Field(default=0, ge=0, le=2)
    ad_safety: Literal["ok", "review", "unsafe"] = "ok"
    ad_safety_reason: str = ""
    hook: str = ""
    cliffhanger: str = ""
    why: str = ""


def _fmt(e, offset: float) -> str:
    t = f"{offset:5.1f}s"
    if e.type == "camera":
        return f"{t} [shot] {e.text}"
    if e.type == "dialogue":
        tag = f" ({e.emotion})" if e.emotion else ""
        return f'{t} {e.speaker}{tag}: "{e.text}"'
    return f"{t} {e.type.upper()}: {e.text}"


def clip_view(p: Promo, show: Show) -> tuple[str, str, str]:
    """(full clip, first 3s, last 5s) as a viewer experiences them, with clip-relative times."""
    full, first, last = [], [], []
    clock = 0.0
    for i, s in enumerate(p.segments):
        if i:
            full.append("--- cut ---")
        for e in show.episode(s.episode).events:
            if e.end <= s.start or e.start >= s.end:
                continue
            rel = clock + max(e.start, s.start) - s.start
            line = _fmt(e, rel)
            full.append(line)
            if rel < 3.0:
                first.append(line)
            if rel >= p.duration - 5.0:
                last.append(line)
        clock += s.duration
    return "\n".join(full), "\n".join(first) or "(nothing)", "\n".join(last) or "(nothing)"


def keyword_safety(text: str) -> tuple[str, str]:
    if re.search(CHILD_AT_RISK, text, re.I):
        return "review", "possible harm or threat to a child"
    m = re.search(rf"(?<![a-z])(?:{SAFETY_PATTERNS})(?![a-z])", text, re.I)
    return ("review", f"mentions '{m.group(0)}'") if m else ("ok", "")


def rule_based_judgement(p: Promo, show: Show, sigs: dict[int, EpisodeSignals], clip: str) -> Judgement:
    """Mock-mode estimates from the rule signals. Deliberately simple: it is the baseline."""
    h_ep = int(p.hook_event[1:4]); c_ep = int(p.cliff_event[1:4])
    sh = sigs[h_ep].by_event[p.hook_event].hook
    sc = sigs[c_ep].by_event[p.cliff_event].cliff
    speakers = {ln.split(":")[0].split("(")[0].strip()[6:] for ln in clip.splitlines() if '"' in ln}
    standalone = 4 - (len(p.segments) - 1) * 0.5 - (1 if len(speakers) > 4 else 0)
    hot = len(re.findall(r"\((?:[^)]*?)(?:furious|shock|horrif|cruel|desperate|outrag|scream|cold|venom)",
                         clip, re.I))
    safety, reason = keyword_safety(clip)
    to5 = lambda x: max(1, min(5, round(x)))
    return Judgement(cold_summary="(rule-based estimate: no model read this clip)",
                     hook_score=to5(1 + 4 * sh), cliffhanger_score=to5(1 + 4 * sc),
                     standalone_score=to5(standalone), emotion_score=to5(1 + hot),
                     spoiler_risk=0, ad_safety=safety, ad_safety_reason=reason,
                     hook=p.hook[:90], cliffhanger=p.cliffhanger[:90], why="rule-based: " + p.why[:120])


def judge_promo(p: Promo, show: Show, bible: ShowBible, sigs: dict[int, EpisodeSignals],
                llm: LLMClient) -> Judgement:
    clip, first3, last5 = clip_view(p, show)
    eps = sorted({s.episode for s in p.segments})
    position = (f"episode {eps[0]} of {len(show.episodes)}" if len(eps) == 1
                else f"episodes {', '.join(map(str, eps))} of {len(show.episodes)}")
    system, user = prompts.load("judge_promo", title=show.title, position=position,
                                duration=f"{p.duration:.1f}", n_segments=str(len(p.segments)),
                                first3=first3, last5=last5, clip=clip)
    j = llm.complete_json(task="judge", system=system, user=user, schema=Judgement,
                          mock=lambda: rule_based_judgement(p, show, sigs, clip), item_id=p.id)
    # Belt and braces: the code check can raise the safety level, never lower it.
    kw, reason = keyword_safety(clip)
    if kw == "review" and j.ad_safety == "ok":
        j = j.model_copy(update={"ad_safety": "review", "ad_safety_reason": f"keyword check: {reason}"})
    return j
