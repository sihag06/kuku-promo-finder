"""Stage 0: turn screenplay .txt files into structured, timed events.

Pure code, no LLM. Designed to be forgiving: anything it cannot read becomes a
warning on the episode instead of a crash, because the interview reruns this on
a show we have never seen.

Expected block format (one per CAMERA / ACTION / SOUND FX / dialogue line):

    **LADY MARA (Confused, Breathless):**
    Start: (00:13.50) | End: (00:14.30)
    "Why?"
"""
from __future__ import annotations

import re
import tempfile
import zipfile
from pathlib import Path

from .schema import Episode, Event, Scene, Show

# --- patterns ---------------------------------------------------------------
RE_EP_NUMBER = re.compile(r"^EPISODE NUMBER:\s*(?:Episode\s*)?(\d+)", re.I)
RE_LOGLINE = re.compile(r"^EPISODE LOGLINE:\s*(.*)", re.I)
RE_CONTINUITY = re.compile(r"^CONTINUITY NOTES:\s*(.*)", re.I)
RE_SCENE = re.compile(r"^SCENE\s+(\d+)\s*$", re.I)
RE_SCENE_HEADING = re.compile(r"^Scene heading:\s*(.*)", re.I)
RE_SCENE_START = re.compile(r"^Scene start:\s*([\d:.]+)", re.I)
RE_SCENE_END = re.compile(r"^Scene end:\s*([\d:.]+)", re.I)
RE_LABEL = re.compile(r"^\*\*(.+?):?\*\*:?\s*$")  # **LABEL:**, **LABEL**: or **LABEL**
RE_TIMING = re.compile(
    r"^Start:\s*\(?\s*([\d:.]+)\s*\)?\s*\|\s*End:\s*\(?\s*([\d:.]+)\s*\)?", re.I
)
RE_FILE_EP = re.compile(r"episode[-_ ]?0*(\d+)", re.I)

SFX_LABELS = {"SOUND FX", "SFX", "SOUND", "MUSIC", "SOUND EFFECT", "SOUND EFFECTS"}
OFFSCREEN_TOKENS = {"O.S.", "V.O.", "O.C.", "OS", "VO"}
CONTINUATION_TOKENS = {"CONTINUING", "CONTINUES", "CONT'D", "CONT", "VOICE CONTINUING"}


def parse_time(value: str) -> float:
    """'01:23.50' -> 83.5 ; '1:02:03.5' -> 3723.5 ; '12.5' -> 12.5"""
    parts = value.strip().split(":")
    seconds = 0.0
    for p in parts:
        seconds = seconds * 60 + float(p)
    return round(seconds, 2)


def parse_label(label: str) -> tuple[str, str | None, str | None, bool]:
    """Split a block label into (type, speaker, emotion, off_screen).

    'MATRIARCH (O.S.) (Firm)'  -> dialogue, MATRIARCH, 'Firm', True
    'CAMERA'                   -> camera, None, None, False
    """
    label = label.strip().rstrip(":").strip()
    upper = label.upper()
    if upper == "CAMERA":
        return "camera", None, None, False
    if upper == "ACTION":
        return "action", None, None, False
    if upper in SFX_LABELS:
        return "sfx", None, None, False

    speaker = re.split(r"\s*\(", label, maxsplit=1)[0].strip()
    off_screen = False
    emotions: list[str] = []
    for group in re.findall(r"\(([^)]*)\)", label):
        for token in re.split(r"[,/]", group):
            t = token.strip()
            if not t:
                continue
            if t.upper() in OFFSCREEN_TOKENS:
                off_screen = True
            elif t.upper() in CONTINUATION_TOKENS:
                continue
            else:
                emotions.append(t)
    return "dialogue", speaker.upper(), (", ".join(emotions) or None), off_screen


def _clean_text(lines: list[str], kind: str) -> str:
    text = " ".join(l.strip() for l in lines if l.strip())
    if kind == "dialogue":
        text = text.strip().strip('"“”').strip()
    return text


def parse_episode(path: Path) -> Episode:
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    lines = raw.splitlines()

    number = None
    logline = continuity = ""
    scenes: list[Scene] = []
    events: list[Event] = []
    warnings: list[str] = []

    scene_no = 0
    cur_scene: dict | None = None
    block: dict | None = None  # {label, start, end, lines, line_no}

    def flush_block():
        nonlocal block
        if block is None:
            return
        kind, speaker, emotion, off = parse_label(block["label"])
        text = _clean_text(block["lines"], kind)
        if block["start"] is None:
            warnings.append(f"line {block['line_no']}: block '{block['label']}' has no timing; skipped")
        elif not text and kind != "camera":
            warnings.append(f"line {block['line_no']}: block '{block['label']}' has no text; skipped")
        else:
            start, end = block["start"], block["end"]
            if end < start:
                warnings.append(f"line {block['line_no']}: end {end} < start {start}; swapped")
                start, end = end, start
            events.append(
                Event(
                    id="",  # assigned after we know the episode number
                    episode=0,
                    scene=scene_no or 1,
                    index=len(events),
                    type=kind,
                    text=text,
                    start=start,
                    end=end,
                    speaker=speaker,
                    emotion=emotion,
                    off_screen=off,
                )
            )
        block = None

    def flush_scene():
        nonlocal cur_scene
        if cur_scene and cur_scene.get("start") is not None and cur_scene.get("end") is not None:
            scenes.append(Scene(**cur_scene))
        elif cur_scene:
            warnings.append(f"scene {cur_scene['number']}: missing start/end; derived from events")
            scenes.append(Scene(number=cur_scene["number"], heading=cur_scene.get("heading", ""),
                                start=-1, end=-1))
        cur_scene = None

    for i, line in enumerate(lines, start=1):
        s = line.strip()
        if not s:
            continue
        if m := RE_EP_NUMBER.match(s):
            number = int(m.group(1)); continue
        if m := RE_LOGLINE.match(s):
            logline = m.group(1).strip(); continue
        if m := RE_CONTINUITY.match(s):
            continuity = m.group(1).strip(); continue
        if s.startswith("--- SCRIPT") or s.lower().startswith("actions and dialogue"):
            flush_block(); continue
        if m := RE_SCENE.match(s):
            flush_block(); flush_scene()
            scene_no = int(m.group(1))
            cur_scene = {"number": scene_no, "heading": "", "start": None, "end": None}
            continue
        if cur_scene is not None and block is None:
            if m := RE_SCENE_HEADING.match(s):
                cur_scene["heading"] = m.group(1).strip(); continue
            if m := RE_SCENE_START.match(s):
                cur_scene["start"] = parse_time(m.group(1)); continue
            if m := RE_SCENE_END.match(s):
                cur_scene["end"] = parse_time(m.group(1)); continue
        if m := RE_LABEL.match(s):
            flush_block()
            block = {"label": m.group(1), "start": None, "end": None, "lines": [], "line_no": i}
            continue
        if block is not None:
            if block["start"] is None and (m := RE_TIMING.match(s)):
                block["start"], block["end"] = parse_time(m.group(1)), parse_time(m.group(2))
            else:
                block["lines"].append(s)
            continue
        warnings.append(f"line {i}: unrecognised text ignored: {s[:60]!r}")

    flush_block(); flush_scene()

    if number is None:
        m = RE_FILE_EP.search(path.name)
        if m:
            number = int(m.group(1))
            warnings.append("episode number taken from file name")
        else:
            raise ValueError(f"cannot determine episode number for {path.name}")

    # Fill scene bounds that were missing, from the events inside them.
    for sc in scenes:
        if sc.start < 0:
            inside = [e for e in events if e.scene == sc.number]
            sc.start = min((e.start for e in inside), default=0.0)
            sc.end = max((e.end for e in inside), default=0.0)

    for idx, e in enumerate(events):
        e.episode = number
        e.index = idx
        e.id = f"E{number:03d}-{idx:04d}"
        sc = next((x for x in scenes if x.number == e.scene), None)
        if sc and (e.start < sc.start - 0.05 or e.end > sc.end + 0.05):
            warnings.append(f"{e.id}: {e.start}-{e.end} outside scene {sc.number} ({sc.start}-{sc.end})")

    shots = sorted({round(t, 2) for e in events if e.type == "camera" for t in (e.start, e.end)})
    if not shots:  # no camera blocks: fall back to every event boundary
        shots = sorted({round(t, 2) for e in events for t in (e.start, e.end)})
        warnings.append("no CAMERA blocks; using all event boundaries as cut points")

    return Episode(
        number=number,
        source_file=path.name,
        logline=logline,
        continuity=continuity,
        scenes=scenes,
        events=events,
        shot_boundaries=shots,
        warnings=warnings,
    )


def _resolve_input(path: Path) -> tuple[Path, str]:
    """Accept a folder or a .zip; return (folder with .txt files, show title)."""
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"path not found: {path}  (check the folder/zip location; "
                                f"run `ls` to see what is there)")
    if path.is_file() and path.suffix.lower() == ".zip":
        tmp = Path(tempfile.mkdtemp(prefix="promo_show_"))
        with zipfile.ZipFile(path) as zf:
            for member in zf.infolist():  # guard against path traversal
                target = (tmp / member.filename).resolve()
                if not str(target).startswith(str(tmp.resolve())):
                    raise ValueError(f"unsafe path in zip: {member.filename}")
            zf.extractall(tmp)
        path = tmp
    txts = [p for p in path.rglob("*.txt") if not p.name.startswith("._")]
    if not txts:
        raise FileNotFoundError(f"no .txt episode files under {path}")
    folders = {p.parent for p in txts}
    title = next(iter(folders)).name if len(folders) == 1 else path.name
    return path, title


def load_show(path: str | Path, title: str | None = None) -> Show:
    folder, guessed_title = _resolve_input(Path(path))
    files = sorted(p for p in folder.rglob("*.txt") if not p.name.startswith("._"))
    episodes = [parse_episode(p) for p in files]
    episodes.sort(key=lambda e: e.number)
    seen: set[int] = set()
    for ep in episodes:
        if ep.number in seen:
            ep.warnings.append(f"duplicate episode number {ep.number}")
        seen.add(ep.number)
    return Show(title=title or guessed_title, episodes=episodes)


def save_parsed(show: Show, out_dir: str | Path) -> Path:
    """Write one JSON per episode + a summary; this is Stage 0's output."""
    out = Path(out_dir) / "parsed"
    out.mkdir(parents=True, exist_ok=True)
    for ep in show.episodes:
        (out / f"ep{ep.number:03d}.json").write_text(ep.model_dump_json(indent=2), encoding="utf-8")
    return out


def summary(show: Show) -> str:
    from collections import Counter

    types = Counter(e.type for ep in show.episodes for e in ep.events)
    durs = [ep.duration for ep in show.episodes]
    warns = [(ep.number, w) for ep in show.episodes for w in ep.warnings]
    lines = [
        f"Show: {show.title}",
        f"Episodes: {len(show.episodes)} ({show.episodes[0].number}-{show.episodes[-1].number})",
        f"Runtime: {show.total_duration / 60:.1f} min (episodes {min(durs):.1f}s-{max(durs):.1f}s)",
        f"Events: {sum(types.values())} " + ", ".join(f"{k}={v}" for k, v in types.most_common()),
        f"Warnings: {len(warns)}",
    ]
    lines += [f"  ep {n}: {w}" for n, w in warns]
    return "\n".join(lines)


def compact_transcript(ep: Episode, include_camera: bool = False) -> str:
    """The text the LLM sees: one line per event, addressed by id. Times shown for
    context only; the model is told to answer with ids."""
    out = []
    for e in ep.events:
        if e.type == "camera" and not include_camera:
            continue
        t = f"[{e.id} {e.start:06.2f}-{e.end:06.2f}]"
        if e.type == "dialogue":
            tag = f" ({e.emotion})" if e.emotion else ""
            os_ = " (O.S.)" if e.off_screen else ""
            out.append(f'{t} {e.speaker}{os_}{tag}: "{e.text}"')
        else:
            out.append(f"{t} {e.type.upper()}: {e.text}")
    return "\n".join(out)


if __name__ == "__main__":  # python -m promo_finder.parser <folder|zip> [--out outputs/x]
    import argparse

    ap = argparse.ArgumentParser(description="Parse a screenplay folder or zip (Stage 0).")
    ap.add_argument("input")
    ap.add_argument("--out", default=None, help="write parsed/epNNN.json here")
    args = ap.parse_args()
    s = load_show(args.input)
    print(summary(s))
    if args.out:
        print(f"Wrote {save_parsed(s, args.out)}")
