"""Measure a run (or compare several) against the gold sets and human ratings.

    python -m eval.evaluate --run outputs/<show>/<run_id>
    python -m eval.evaluate --run outputs/<show>/mock --run outputs/<show>/live     # ablation
    python -m eval.evaluate --make-rating-sheet outputs/<show>/<run_id>            # blind 1-5 sheet

Gold files are picked up automatically if present: eval/gold_human.csv (Route A),
eval/gold_llm.csv (Route B). Ratings: eval/human_ratings.csv.

Writes outputs/eval/<timestamp>/report.md + results.json. Sections:
 1 runs           mode, models, size, cost
 2 hard rules     every final promo re-checked against the brief's rules
 3 recall         per gold set, strict + loose, @5 / @20 / all
 4 agreement      of the tool's top 20 in labelled episodes, how many match a gold pick
 5 where misses   for each missed gold promo: ranked low? rejected (why)? never mined?
 6 ratings        human 1-5 ratings of the top 20, and judge-vs-human rank correlation
 7 run vs run     top-20 overlap and score correlation (ablation or stability)
 8 examples       unmatched top picks and missed gold promos, for the failure write-up
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from datetime import datetime
from pathlib import Path

from eval.gold import GoldPromo, load_gold, match_level, to_mmss
from promo_finder.config import PromoConfig
from promo_finder.parser import load_show
from promo_finder.schema import Promo
from promo_finder.validate import violations

GOLD_FILES = {"human": Path("eval/gold_human.csv"), "llm": Path("eval/gold_llm.csv")}
RATINGS = Path("eval/human_ratings.csv")
LLM_RATINGS = Path("eval/llm_ratings.csv")  # stand-in from eval/llm_rate.py


# ---- small stats helpers -----------------------------------------------------------
def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a: list[float], b: list[float]) -> float | None:
    if len(a) < 3 or len(a) != len(b):
        return None
    ra, rb = _ranks(a), _ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((y - mb) ** 2 for y in rb) ** 0.5
    return round(cov / (va * vb), 3) if va and vb else None


def pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100 * n / d:.0f}%)" if d else "n/a"


# ---- loading ----------------------------------------------------------------------------
class Run:
    def __init__(self, d: str | Path):
        self.dir = Path(d)
        self.meta = json.loads((self.dir / "run_meta.json").read_text())
        self.promos = json.loads((self.dir / "promos.json").read_text())
        self.rejected = json.loads((self.dir / "rejected.json").read_text())
        self.moments = json.loads((self.dir / "moments.json").read_text())
        self.label = f"{self.meta['run_id']} ({self.meta['mode']})"

    def by_quality(self) -> "Run":
        """Same run, promos re-ordered by quality alone (older runs: by total score)."""
        q = copy.copy(self)
        key = lambda p: -(p["scores"].get("quality") or p["scores"]["total"])
        q.promos = [dict(p, rank=i) for i, p in enumerate(sorted(self.promos, key=key), 1)]
        q.label = self.label.replace(")", ", quality order)")
        return q

    @property
    def models(self) -> str:
        m = self.meta["models"]
        return m if isinstance(m, str) else ", ".join(sorted(set(m.values())))


def _sig(p: dict) -> tuple:
    return tuple((s["episode"], round(s["start"], 1), round(s["end"], 1)) for s in p["segments"])


# ---- the measurements --------------------------------------------------------------
def hard_rules(run: Run, show, cfg: PromoConfig) -> dict:
    bad = []
    for p in run.promos:
        v = violations(Promo.model_validate(p), show, cfg)
        if v:
            bad.append((p["id"], v))
    reasons: dict[str, int] = {}
    for r in run.rejected:
        key = r["reasons"][0].split(" (")[0]
        key = key if not key[0].isdigit() and "s outside" not in key else "duration/timing rule"
        reasons[key] = reasons.get(key, 0) + 1
    return {"checked": len(run.promos), "violations": bad,
            "rejected_top": sorted(reasons.items(), key=lambda kv: -kv[1])[:6]}


def recall(run: Run, gold: list[GoldPromo], ks=(5, 20, None)) -> dict:
    out = {}
    for level in ("strict", "loose"):
        for k in ks:
            pool = run.promos if k is None else run.promos[:k]
            found = sum(any(_ok(match_level(p, g), level) for p in pool) for g in gold)
            out[f"{level}@{k or 'all'}"] = (found, len(gold))
    return out


def _ok(m: str | None, level: str) -> bool:
    return m == "strict" or (level == "loose" and m == "loose")


def agreement(run: Run, gold: list[GoldPromo], k: int = 20) -> tuple[int, int]:
    labelled = set().union(*(g.episodes for g in gold)) if gold else set()
    inside = [p for p in run.promos if {s["episode"] for s in p["segments"]} <= labelled][:k]
    hit = sum(any(match_level(p, g) for g in gold) for p in inside)
    return hit, len(inside)


def where_missed(run: Run, gold: list[GoldPromo], show, k: int = 20) -> list[dict]:
    """For every gold promo: the best place the tool got to."""
    rows = []
    for g in gold:
        best = None
        for p in run.promos:
            m = match_level(p, g)
            if m and (best is None or p["rank"] < best["rank"]):
                best = {"rank": p["rank"], "level": m, "id": p["id"]}
        if best and best["rank"] <= k:
            status = f"found: #{best['rank']} ({best['level']})"
        elif best:
            status = f"ranked too low: #{best['rank']} ({best['level']})"
        else:
            ep0, s0, e0 = g.segments[0]
            epL, sL, eL = g.segments[-1]

            def inside(eid: str, ep: int, s: float, e: float) -> bool:
                if int(eid[1:4]) != ep:
                    return False
                ev = show.episode(ep).event(eid)
                return s - 1 <= ev.start <= e + 1

            def opens_near(eid: str) -> bool:  # same opening as the gold promo (within 3s)
                if int(eid[1:4]) != ep0:
                    return False
                return abs(show.episode(ep0).event(eid).start - s0) <= 3.0

            rej = [r for r in run.rejected if opens_near(r["hook_event"])]
            mined_h = any(inside(m["event_id"], ep0, s0, e0) for m in run.moments.get(str(ep0), {}).get("hooks", []))
            mined_c = any(inside(m["event_id"], epL, sL, eL)
                          for m in run.moments.get(str(epL), {}).get("cliffhangers", []))
            if rej:  # a candidate opening like the gold promo existed but was dropped: say why
                rej.sort(key=lambda r: abs(show.episode(ep0).event(r["hook_event"]).start - s0))
                status = f"rejected: {rej[0]['reasons'][0][:70]} ({rej[0]['template']})"
            elif mined_h and mined_c:
                status = "both moments mined, but never paired into this shape"
            elif mined_h:
                status = "hook mined, cliffhanger not mined"
            elif mined_c:
                status = "cliffhanger mined, hook not mined"
            else:
                status = "not found: neither moment mined"
        rows.append({"gold": g.id, "segments": " | ".join(f"Ep{e} {to_mmss(s)}-{to_mmss(t)}" for e, s, t in g.segments),
                     "hook": g.hook[:60], "status": status})
    return rows


def ratings_summary(run: Run, path: Path = RATINGS) -> dict | None:
    if not path.exists():
        return None
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8")) if (r.get("overall") or "").strip()]
    rows = [r for r in rows if r.get("run_id") == run.meta["run_id"]]
    by_id = {p["id"]: p for p in run.promos}
    rows = [r for r in rows if r["promo_id"] in by_id]
    if not rows:
        return None
    overall = [int(r["overall"]) for r in rows]
    judge = [by_id[r["promo_id"]]["scores"]["total"] for r in rows]
    judge_q = [by_id[r["promo_id"]]["scores"].get("quality", 0) for r in rows]
    dims = {d: round(sum(int(r[d]) for r in rows if r.get(d)) / max(1, sum(bool(r.get(d)) for r in rows)), 2)
            for d in ("hook", "cliffhanger", "standalone")}
    low = sorted(rows, key=lambda r: int(r["overall"]))[:3]
    return {"rated": len(rows), "mean_overall": round(sum(overall) / len(overall), 2),
            "share_4plus": pct(sum(o >= 4 for o in overall), len(overall)), "dims": dims,
            "spearman_judge_vs_rater": spearman(judge, overall),
            "spearman_quality_vs_rater": spearman(judge_q, overall),
            "lowest": [(r["promo_id"], int(r["overall"]), r.get("comment", "")[:120]) for r in low]}


def compare(a: Run, b: Run, k: int = 20) -> dict:
    top_a, top_b = a.promos[:k], b.promos[:k]
    shared = sum(any(match_level(p, GoldPromo("x", [(s["episode"], s["start"], s["end"]) for s in q["segments"]]))
                     == "strict" for q in top_b) for p in top_a)
    sb = {_sig(p): p["scores"]["total"] for p in b.promos}
    common = [(p["scores"]["total"], sb[_sig(p)]) for p in a.promos if _sig(p) in sb]
    return {"top_overlap": f"{shared}/{k}", "common_promos": len(common),
            "spearman_scores": spearman([x for x, _ in common], [y for _, y in common])}


# ---- rating sheet -----------------------------------------------------------------------
def make_rating_sheet(run_dir: str, k: int = 20, data: str | None = None) -> Path:
    """Blind sheet: top-k promos in random order, no tool scores or ranks shown."""
    run = Run(run_dir)
    show = load_show(data or run.meta["input"])
    from promo_finder.pipeline_utils import clip_lines
    items = run.promos[:k]
    random.Random(7).shuffle(items)
    RATINGS.parent.mkdir(exist_ok=True)
    with open(RATINGS, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "promo_id", "segments", "clip", "hook", "cliffhanger", "standalone", "overall", "comment"])
        for p in items:
            pr = Promo.model_validate(p)
            w.writerow([run.meta["run_id"], p["id"],
                        " | ".join(f"Ep{s['episode']} {to_mmss(s['start'])}-{to_mmss(s['end'])}" for s in p["segments"]),
                        "\n".join(clip_lines(pr, show, marks=False)), "", "", "", "", ""])
    return RATINGS


# ---- report -------------------------------------------------------------------------------
def build_report(runs: list[Run], golds: dict[str, list[GoldPromo]], k: int = 20,
                 data: str | None = None) -> tuple[str, dict]:
    show = load_show(data or runs[0].meta["input"])
    cfg = PromoConfig(**runs[0].meta["config"]["promo"])
    res: dict = {"generated": datetime.now().isoformat(timespec="seconds"), "runs": {}}
    md = [f"# Evaluation: {runs[0].meta['show']}", f"Generated {res['generated']}", ""]

    md += ["## 1. Runs", "", "| run | mode | models | promos | LLM calls | cost (USD) |", "|---|---|---|---|---|---|"]
    for r in runs:
        u = r.meta["llm_usage"]["_total"]
        md.append(f"| {r.label} | {r.meta['mode']} | {r.models} | {len(r.promos)} | {u['calls']} | {u['cost_usd']:.4f} |")

    md += ["", "## 2. Hard rules (every final promo re-checked)", ""]
    for r in runs:
        hr = hard_rules(r, show, cfg)
        res["runs"].setdefault(r.label, {})["hard_rules"] = hr
        md.append(f"- **{r.label}**: {pct(hr['checked'] - len(hr['violations']), hr['checked'])} pass. "
                  f"Rejected before ranking, top reasons: " + "; ".join(f"{a} ({b})" for a, b in hr["rejected_top"]))
        for pid, v in hr["violations"][:5]:
            md.append(f"  - VIOLATION {pid}: {v}")

    for gname, gold in golds.items():
        eps = sorted(set().union(*(g.episodes for g in gold)))
        md += ["", f"## 3. Recall vs {gname} gold ({len(gold)} promos, episodes {eps})", "",
               "| run | strict@5 | strict@20 | strict@all | loose@20 | loose@all | agreement@20 in labelled eps |",
               "|---|---|---|---|---|---|---|"]
        for r in [x for run in runs for x in (run, run.by_quality())]:
            rc = recall(r, gold)
            ag = agreement(r, gold, k)
            res["runs"].setdefault(r.label, {})[f"recall_{gname}"] = {kk: list(v) for kk, v in rc.items()}
            res["runs"][r.label][f"agreement_{gname}"] = list(ag)
            md.append(f"| {r.label} | {pct(*rc['strict@5'])} | {pct(*rc['strict@20'])} | {pct(*rc['strict@all'])} | "
                      f"{pct(*rc['loose@20'])} | {pct(*rc['loose@all'])} | {pct(*ag)} |")
        md += ["", f"### Where each {gname} gold promo ended up ({runs[-1].label})", "",
               "| gold | segments | hook | status (posting order) | best quality rank |", "|---|---|---|---|---|"]
        wm = where_missed(runs[-1], gold, show, k)
        qrun = runs[-1].by_quality()
        for x, g in zip(wm, gold):
            ranks = [p["rank"] for p in qrun.promos if match_level(p, g)]
            x["quality_rank"] = min(ranks) if ranks else None
        res["runs"][runs[-1].label][f"where_{gname}"] = wm
        md += [f"| {x['gold']} | {x['segments']} | {x['hook'].replace('|', '/')} | {x['status']} | "
               f"{'#' + str(x['quality_rank']) if x['quality_rank'] else '-'} |" for x in wm]
    if not golds:
        md += ["", "## 3. Recall", "", "_No gold file found (eval/gold_human.csv or eval/gold_llm.csv)._"]

    md += ["", "## 6. Ratings of the top 20", ""]
    for who, path in (("human", RATINGS), ("LLM stand-in (independent model, blind)", LLM_RATINGS)):
        for r in runs:
            rs = ratings_summary(r, path)
            res["runs"][r.label][f"ratings_{who.split()[0]}"] = rs
            if not rs:
                continue
            md.append(f"- **{r.label}, {who}**: {rs['rated']} rated; mean overall {rs['mean_overall']}/5; "
                      f"rated 4+: {rs['share_4plus']}; means {rs['dims']}; rank agreement (Spearman) "
                      f"tool score vs rater {rs['spearman_judge_vs_rater']}, quality score vs rater "
                      f"{rs['spearman_quality_vs_rater']}")
            md += [f"  - lowest: {pid} rated {o}/5: {c}" for pid, o, c in rs["lowest"]]
    if not any(md_line.startswith("- **") and "rated" in md_line for md_line in md[-len(runs) * 8:]):
        md.append("- no ratings yet: `python -m eval.llm_rate <run dir>` (AI stand-in) or "
                  "`python -m eval.evaluate --make-rating-sheet <run dir>` (human)")

    if len(runs) > 1:
        md += ["", "## 7. Run vs run", "", "| A | B | top-20 overlap | common promos | score Spearman |",
               "|---|---|---|---|---|"]
        for b in runs[1:]:
            c = compare(runs[0], b, k)
            res.setdefault("compare", {})[f"{runs[0].label} vs {b.label}"] = c
            md.append(f"| {runs[0].label} | {b.label} | {c['top_overlap']} | {c['common_promos']} | {c['spearman_scores']} |")

    main = runs[-1]
    md += ["", f"## 8. Examples for the failure write-up ({main.label})", "",
           "Top-20 promos in labelled episodes that match NO gold pick (possible bad picks, or gold gaps):", ""]
    all_gold = [g for gs in golds.values() for g in gs]
    labelled = set().union(*(g.episodes for g in all_gold)) if all_gold else set()
    for p in main.promos[:k]:
        if {s["episode"] for s in p["segments"]} <= labelled and not any(match_level(p, g) for g in all_gold):
            md.append(f"- #{p['rank']} {p['id']} ({p['template']}, {p['duration']:.0f}s): hook {p['hook'][:70]!r} / "
                      f"cliff {p['cliffhanger'][:60]!r}; flags {p['flags']}")
    md += ["", "_Add your own reading of these (why each is good or bad) below._", ""]
    return "\n".join(md), res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=[], help="run folder (repeat to compare)")
    ap.add_argument("--gold", action="append", default=[], help="extra gold csv: name=path")
    ap.add_argument("--make-rating-sheet", metavar="RUN_DIR")
    ap.add_argument("-k", type=int, default=20)
    ap.add_argument("--data", help="script folder/zip, if it moved since the run (default: path in run_meta)")
    ap.add_argument("--out", default="outputs/eval")
    args = ap.parse_args()
    if args.make_rating_sheet:
        print(f"Wrote {make_rating_sheet(args.make_rating_sheet, args.k, args.data)}: fill hook/cliffhanger/standalone/overall 1-5")
        return
    if not args.run:
        ap.error("give at least one --run")
    runs = [Run(d) for d in args.run]
    golds = {n: load_gold(p, n) for n, p in GOLD_FILES.items() if p.exists()}
    for spec in args.gold:
        name, path = spec.split("=", 1)
        golds[name] = load_gold(path, name)
    md, res = build_report(runs, golds, args.k, args.data)
    out = Path(args.out) / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(md, encoding="utf-8")
    (out / "results.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(md)
    print(f"\nWrote {out / 'report.md'}")


if __name__ == "__main__":
    main()
