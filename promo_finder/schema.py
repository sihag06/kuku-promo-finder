"""Data model shared by every pipeline stage.

Rule of the whole project: times live here as floats (seconds) and are only ever
produced by code. LLM stages refer to events by `Event.id`, never by timestamp.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, computed_field

EventType = Literal["camera", "action", "dialogue", "sfx"]


class Event(BaseModel):
    """One timed block of the screenplay (a CAMERA, ACTION, SOUND FX or dialogue line)."""

    id: str  # e.g. "E001-0042": episode 1, event 42 (stable across runs)
    episode: int
    scene: int
    index: int  # position within the episode, 0-based, in script order
    type: EventType
    text: str
    start: float  # seconds from episode start
    end: float
    speaker: Optional[str] = None  # raw label, e.g. "WOMAN IN RED"
    emotion: Optional[str] = None  # e.g. "Cold, Commanding"
    off_screen: bool = False  # (O.S.) or (V.O.)

    @computed_field
    @property
    def duration(self) -> float:
        return round(self.end - self.start, 2)


class Scene(BaseModel):
    number: int
    heading: str
    start: float
    end: float


class Episode(BaseModel):
    number: int
    source_file: str
    logline: str = ""
    continuity: str = ""
    scenes: list[Scene] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)
    # Legal cut points (seconds): every CAMERA start/end. Clips may only start or
    # end on one of these, so we never cut in the middle of a shot.
    shot_boundaries: list[float] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @computed_field
    @property
    def duration(self) -> float:
        ends = [s.end for s in self.scenes] + [e.end for e in self.events]
        return round(max(ends), 2) if ends else 0.0

    def event(self, event_id: str) -> Event:
        for e in self.events:
            if e.id == event_id:
                return e
        raise KeyError(event_id)


class Show(BaseModel):
    title: str
    episodes: list[Episode]

    def episode(self, number: int) -> Episode:
        for ep in self.episodes:
            if ep.number == number:
                return ep
        raise KeyError(number)

    @computed_field
    @property
    def total_duration(self) -> float:
        return round(sum(ep.duration for ep in self.episodes), 2)


# ---- Promo-level models (filled by later stages; defined here so every stage
# ---- agrees on one shape from day one) -------------------------------------


class Segment(BaseModel):
    episode: int
    start: float
    end: float
    first_event: str  # event id the segment opens on
    last_event: str  # event id the segment closes on

    @computed_field
    @property
    def duration(self) -> float:
        return round(self.end - self.start, 2)


class Scores(BaseModel):
    hook: float = 0.0  # all 0-1
    cliffhanger: float = 0.0
    standalone: float = 0.0
    emotion: float = 0.0
    funnel: float = 0.0
    signal_agreement: float = 0.0
    spoiler_penalty: float = 0.0
    safety_penalty: float = 0.0
    quality: float = 0.0  # how good the promo is, ignoring posting strategy (no funnel, no variety)
    total: float = 0.0    # quality + early-episode funnel + rule agreement: drives posting order


# continuous:    one stretch, hook near its start, cliffhanger at its end
# condensed:     hook snippet, then jump ahead to the build-up into a later cliffhanger
# cold_open:     flash-forward: a later hook shown first, then the earlier build-up to a cliffhanger
# cross_episode: hook in episode N, cliffhanger in episode N+1..N+span
PromoTemplate = Literal["continuous", "condensed", "cold_open", "cross_episode"]


class Promo(BaseModel):
    id: str
    template: PromoTemplate
    segments: list[Segment]
    hook_event: str
    cliff_event: str
    hook: str = ""  # human-readable hook description / quote
    cliffhanger: str = ""
    why: str = ""
    scores: Scores = Field(default_factory=Scores)
    flags: list[str] = Field(default_factory=list)  # e.g. "visual_hook", "mid_shot_cut"
    hook_type: str = ""
    cliff_type: str = ""
    pre_score: float = 0.0  # before judging: miner strengths + rule signals
    # filled by the judge (Stage 5)
    cold_summary: str = ""
    confusions: list[str] = Field(default_factory=list)
    want_next: str = ""
    spoiler_risk: int = 0
    ad_safety: str = "ok"
    ad_safety_reason: str = ""
    # filled by the ranker (Stage 7)
    tier: str = ""
    rank: Optional[int] = None          # posting order
    quality_rank: Optional[int] = None  # best-first, ignoring posting strategy
    rank_reason: str = ""

    @computed_field
    @property
    def duration(self) -> float:
        return round(sum(s.duration for s in self.segments), 2)
