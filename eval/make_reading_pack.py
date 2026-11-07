"""Step 2 helper: print episodes as easy-to-read timed scripts for hand-labelling
the human gold set. Run BEFORE looking at any tool output (blind labelling).

    python -m eval.make_reading_pack data
    python -m eval.make_reading_pack data --episodes 1 3 5 --out outputs/reading_pack

Writes one markdown file per episode (gitignored: it contains Kuku's script text).
Camera lines are shown as [shot] markers: those are the legal cut points.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from promo_finder.parser import load_show

# Spread across the season so the gold set covers early, mid and late arcs.
# Episode 1 is left out on purpose: example promos from it were already discussed
# while designing the tool, so labelling it would not be blind.
DEFAULT_EPISODES = [2, 3, 5, 8, 11, 15, 20, 25, 30, 36, 42, 48]


def mmss(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:05.2f}"


def render(ep) -> str:
    out = [f"# Episode {ep.number}  ({mmss(ep.duration)})", "", f"**Logline:** {ep.logline}", ""]
    scene = None
    for e in ep.events:
        if e.scene != scene:
            scene = e.scene
            sc = next((s for s in ep.scenes if s.number == scene), None)
            out += ["", f"## Scene {scene}: {sc.heading if sc else ''}", ""]
        t = f"`{mmss(e.start)}-{mmss(e.end)}`"
        if e.type == "camera":
            out.append(f"{t} *[shot] {e.text}*")
        elif e.type == "dialogue":
            tag = f" ({e.emotion})" if e.emotion else ""
            out.append(f"{t} **{e.speaker}**{tag}: \"{e.text}\"")
        else:
            out.append(f"{t} {e.type.upper()}: {e.text}")
        out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="folder or zip of episode .txt files")
    ap.add_argument("--episodes", type=int, nargs="*", default=DEFAULT_EPISODES)
    ap.add_argument("--out", default="outputs/reading_pack")
    args = ap.parse_args()
    show = load_show(args.input)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for n in args.episodes:
        (out / f"ep{n:03d}.md").write_text(render(show.episode(n)), encoding="utf-8")
    print(f"Wrote {len(args.episodes)} episodes to {out}/  ->  now fill eval/gold_human.csv")


if __name__ == "__main__":
    main()
