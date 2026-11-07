"""Stage 3: find hook and cliffhanger moments in each episode. One LLM call per episode.

The model answers with event ids only; code checks every id exists in that episode
and drops any that don't (with a note), so a hallucinated moment can never reach
the output. In mock mode the rule-based signals answer instead.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from . import prompts
from .llm.client import LLMClient
from .parser import compact_transcript
from .schema import Episode
from .show_bible import ShowBible
from .signals import EpisodeSignals

MomentType = Literal["violence", "humiliation", "reveal", "betrayal", "threat", "rescue",
                     "romance", "confrontation", "mystery", "other"]


class Moment(BaseModel):
    event_id: str
    type: MomentType = "other"
    quote: str = ""
    strength: int = Field(ge=1, le=5)
    why: str = ""


class MinedEpisode(BaseModel):
    hooks: list[Moment] = Field(default_factory=list)
    cliffhangers: list[Moment] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list, description="filled by code")


REASON_TO_TYPE = {
    "violence": "violence", "danger": "violence", "impact sound": "violence",
    "public humiliation": "humiliation", "rejection": "humiliation",
    "reveal": "reveal", "reveal pending": "reveal", "accusation": "confrontation",
    "threat": "threat", "threat/ultimatum": "threat", "danger rising": "threat",
    "romance spike": "romance", "sudden entrance": "mystery", "interruption": "mystery",
    "open question": "mystery", "ominous cue": "mystery", "cut to black": "mystery",
}


def _strength(score: float) -> int:
    return max(1, min(5, round(1 + 4 * score)))


def rule_based_moments(ep: Episode, sig: EpisodeSignals, k: int) -> MinedEpisode:
    out = MinedEpisode(notes=["rule-based moments (mock mode)"])
    for kind, target in (("hook", out.hooks), ("cliff", out.cliffhangers)):
        for s in sig.top(kind, k, events=ep.events):
            score = getattr(s, kind)
            if score < 0.25:
                continue
            reasons = s.hook_reasons if kind == "hook" else s.cliff_reasons
            mtype = next((REASON_TO_TYPE[r] for r in reasons if r in REASON_TO_TYPE), "other")
            target.append(Moment(event_id=s.event_id, type=mtype, quote=ep.event(s.event_id).text[:100],
                                 strength=_strength(score), why=", ".join(reasons)[:100]))
    return out


def build_prompt(ep: Episode, bible: ShowBible, sig: EpisodeSignals, k: int) -> tuple[str, str]:
    cast = "; ".join(f"{c.name} ({c.role})" if c.role else c.name
                     for c in bible.characters if c.importance <= 2)[:1500]
    hint_ids = {s.event_id for kind in ("hook", "cliff") for s in sig.top(kind, k, events=ep.events)}
    hints = "\n".join(
        f"{i}: {sig.by_event[i].hook:.2f} / {sig.by_event[i].cliff:.2f} "
        f"({', '.join(sig.by_event[i].hook_reasons + sig.by_event[i].cliff_reasons)})"
        for i in sorted(hint_ids)) or "(none)"
    return prompts.load("mine_moments", premise=bible.premise, cast=cast or "(unknown)",
                        episode=str(ep.number), logline=ep.logline,
                        transcript=compact_transcript(ep), hints=hints, max_items=str(k))


def validate_moments(mined: MinedEpisode, ep: Episode, k: int) -> MinedEpisode:
    valid = {e.id for e in ep.events}
    notes = list(mined.notes)
    cleaned = {}
    for name in ("hooks", "cliffhangers"):
        seen, keep = set(), []
        for m in getattr(mined, name):
            if m.event_id not in valid:
                notes.append(f"dropped {name[:-1]} with unknown id {m.event_id!r}")
            elif m.event_id in seen:
                continue
            else:
                seen.add(m.event_id)
                keep.append(m)
        cleaned[name] = sorted(keep, key=lambda m: -m.strength)[:k]
    return MinedEpisode(hooks=cleaned["hooks"], cliffhangers=cleaned["cliffhangers"], notes=notes)


def mine_episode(ep: Episode, bible: ShowBible, sig: EpisodeSignals, llm: LLMClient, k: int = 6) -> MinedEpisode:
    system, user = build_prompt(ep, bible, sig, k)
    mined = llm.complete_json(task="miner", system=system, user=user, schema=MinedEpisode,
                              mock=lambda: rule_based_moments(ep, sig, k), item_id=f"ep{ep.number:03d}")
    return validate_moments(mined, ep, k)
