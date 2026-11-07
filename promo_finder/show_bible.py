"""Stage 1: the show bible. One LLM call per show.

Who is who (with every label the script uses for them), the show's premise, its
genre tropes and its story arcs. Later stages use it to (a) tell whether a clip is
understandable to a stranger and (b) print real names instead of "WOMAN IN RED".

Guard rail: the model may only group labels that really occur in the script. Any
alias it invents is dropped by code, and any real label it forgets is added back
as its own character, so the alias map always covers every speaker exactly once.
"""
from __future__ import annotations

import re
import string
from collections import Counter, defaultdict
from typing import Optional

from pydantic import BaseModel, Field

from . import prompts
from .llm.client import LLMClient
from .schema import Show

TITLES = ("LADY", "LORD", "SIR", "DAME", "DUKE", "DUCHESS", "PRINCE", "PRINCESS",
          "KING", "QUEEN", "MISS", "MRS", "MR", "MASTER", "MISTRESS", "DOCTOR", "DR")


class Character(BaseModel):
    name: str = Field(description="canonical name, e.g. 'Clarissa'")  # kept: changing schema text invalidates cached answers
    aliases: list[str] = Field(description="speaker labels exactly as in the script")
    role: str = Field(default="", description="one line a stranger understands")
    importance: int = Field(default=2, ge=1, le=3, description="1 lead, 2 supporting, 3 minor")


class Arc(BaseModel):
    episodes: str = Field(description="range like '1-4'")
    summary: str


class ShowBible(BaseModel):
    premise: str = ""
    characters: list[Character] = Field(default_factory=list)
    relationships: list[str] = Field(default_factory=list)
    tropes: list[str] = Field(default_factory=list)
    arcs: list[Arc] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list, description="filled by code, not the model")

    def alias_map(self) -> dict[str, str]:
        return {a.upper(): c.name for c in self.characters for a in c.aliases}

    def canonical(self, label: Optional[str]) -> Optional[str]:
        if not label:
            return label
        return self.alias_map().get(label.upper(), nice(label))

    def role_of(self, name: str) -> str:
        for c in self.characters:
            if c.name == name:
                return c.role
        return ""


# ---- inputs for the model -------------------------------------------------------------
def speaker_stats(show: Show) -> dict[str, dict]:
    stats: dict[str, dict] = defaultdict(lambda: {"lines": 0, "first_ep": None, "samples": []})
    for ep in show.episodes:
        for e in ep.events:
            if e.type != "dialogue" or not e.speaker:
                continue
            s = stats[e.speaker]
            s["lines"] += 1
            s["first_ep"] = s["first_ep"] or ep.number
            if len(s["samples"]) < 2 and len(e.text.split()) >= 3:
                s["samples"].append(e.text[:90])
    return dict(sorted(stats.items(), key=lambda kv: -kv[1]["lines"]))


def build_prompt(show: Show) -> tuple[str, str]:
    loglines = "\n".join(f"{ep.number}: {ep.logline} | {ep.continuity}" for ep in show.episodes)
    speakers = "\n".join(
        f"{label} | {s['lines']} | ep {s['first_ep']} | " + " / ".join(f'"{x}"' for x in s["samples"])
        for label, s in speaker_stats(show).items())
    return prompts.load("show_bible", title=show.title, loglines=loglines, speakers=speakers)


# ---- rule-based fallback (mock mode) ------------------------------------------------
def _strip_title(label: str) -> str:
    words = re.sub(r"[^A-Z' ]", " ", label.upper()).split()
    while len(words) > 1 and words[0] in TITLES + ("THE",):
        words = words[1:]
    return " ".join(words)


def nice(label: str) -> str:
    return string.capwords(label.lower())


def rule_based_bible(show: Show) -> ShowBible:
    """Merge a titled label into its bare name only when the bare name is also used
    ('LADY MARA' -> 'MARA'). 'LORD HALDEN' and 'LADY HALDEN' stay apart:
    a shared surname is not the same person. Cannot resolve descriptions
    ('WOMAN IN RED' = Vivienne): that needs reading comprehension, which is
    exactly what the LLM version adds."""
    stats = speaker_stats(show)
    groups: dict[str, list[str]] = defaultdict(list)
    for label in stats:
        bare = _strip_title(label)
        groups[bare if bare != label and bare in stats else label].append(label)
    chars = []
    for key, labels in groups.items():
        lines = sum(stats[l]["lines"] for l in labels)
        chars.append((lines, Character(name=nice(key), aliases=labels, role="",
                                       importance=1 if lines >= 150 else 2 if lines >= 25 else 3)))
    chars.sort(key=lambda x: -x[0])
    first = show.episodes[0].logline if show.episodes else ""
    return ShowBible(premise=first, characters=[c for _, c in chars],
                     notes=["rule-based bible (mock mode): descriptive labels are not merged"])


# ---- validation: the model may only use labels that exist ---------------------------
def validate_bible(bible: ShowBible, show: Show) -> ShowBible:
    real = {label.upper() for label in speaker_stats(show)}
    used: set[str] = set()
    notes = list(bible.notes)
    kept: list[Character] = []
    for c in bible.characters:
        good, bad = [], []
        for a in c.aliases:
            A = a.strip().upper()
            if A in real and A not in used:
                good.append(A); used.add(A)
            else:
                bad.append(a)
        if bad:
            notes.append(f"dropped aliases not in script or already used for {c.name}: {bad}")
        if good:
            kept.append(c.model_copy(update={"aliases": good}))
        else:
            notes.append(f"dropped character with no real labels: {c.name}")
    stats = speaker_stats(show)
    for label in real - used:  # never leave a real speaker unmapped
        kept.append(Character(name=nice(label), aliases=[label], role="",
                              importance=3 if stats[label]["lines"] < 25 else 2))
        notes.append(f"added unmapped label: {label}")
    return bible.model_copy(update={"characters": kept, "notes": notes})


def build_show_bible(show: Show, llm: LLMClient) -> ShowBible:
    system, user = build_prompt(show)
    bible = llm.complete_json(task="show_bible", system=system, user=user, schema=ShowBible,
                              mock=lambda: rule_based_bible(show), item_id=show.title)
    return validate_bible(bible, show)


if __name__ == "__main__":  # python -m promo_finder.show_bible data [--live]
    import argparse

    from .config import load_config
    from .parser import load_show

    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--live", action="store_true", help="use a real model (needs a key)")
    args = ap.parse_args()
    cfg = load_config(overrides={"llm.mode": "live" if args.live else None})
    show = load_show(args.input)
    llm = LLMClient(cfg.llm, cache_dir=cfg.paths.cache_dir)
    b = build_show_bible(show, llm)
    print(f"Premise: {b.premise}\n")
    for c in b.characters[:25]:
        print(f"  [{c.importance}] {c.name:<22} {c.role[:50]:<50} <- {', '.join(c.aliases)}")
    print(f"\n{len(b.characters)} characters; notes:")
    for n in b.notes[:10]:
        print("  -", n)
