"""Stand-in for human ratings: an independent model rates a run's top promos, blind.

    python -m eval.llm_rate outputs/<show>/<run_id>            # top 20, uses the `gold` model

Blind in the same way as the human sheet: the rater sees only each clip's lines (no rank,
no scores, no reasons from the tool), in shuffled order, through its own prompt
(prompts/rater.md). It writes eval/llm_ratings.csv in the same format as
eval/human_ratings.csv, which eval/evaluate.py reports as "LLM ratings".

This is a proxy, not a person: report it as such. Its value is a precision estimate plus
a check of whether the tool's own judge ranks clips the way an independent reader does.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from pydantic import BaseModel, Field

from promo_finder import prompts
from promo_finder.config import load_config
from promo_finder.llm.client import LLMClient

OUT = Path("eval/llm_ratings.csv")


class Rating(BaseModel):
    hook: int = Field(ge=1, le=5)
    cliffhanger: int = Field(ge=1, le=5)
    standalone: int = Field(ge=1, le=5)
    overall: int = Field(ge=1, le=5)
    comment: str = ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("-k", type=int, default=20)
    args = ap.parse_args()
    run = Path(args.run_dir)
    meta = json.loads((run / "run_meta.json").read_text())
    promos = json.loads((run / "promos.json").read_text())[: args.k]
    random.Random(7).shuffle(promos)

    cfg = load_config(overrides={"llm.mode": "live"})
    llm = LLMClient(cfg.llm, cache_dir=cfg.paths.cache_dir, log_dir=run / "raw_llm")
    print(f"rater model: {cfg.llm.model_for('gold')} (tool's judge: {cfg.llm.model_for('judge')}); "
          f"{len(promos)} promos, blind and shuffled")

    def rate(p: dict) -> Rating:
        clip = "\n".join(line.replace(" <== HOOK", "").replace(" <== CLIFFHANGER", "")
                         for line in p["clip_lines"])  # hide the tool's own markers
        system, user = prompts.load("rater", clip=clip)
        return llm.complete_json(task="rater", stage="gold", system=system, user=user,
                                 schema=Rating, item_id=p["id"])

    ratings = llm.map(rate, promos, label="promos rated")
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "promo_id", "hook", "cliffhanger", "standalone", "overall", "comment"])
        for p, r in zip(promos, ratings):
            w.writerow([meta["run_id"], p["id"], r.hook, r.cliffhanger, r.standalone, r.overall, r.comment])
    cost = llm.usage_report()["_total"]["cost_usd"]
    print(f"Wrote {OUT}; cost ${cost:.4f}. Now rerun eval.evaluate to see section 6.")


if __name__ == "__main__":
    main()
