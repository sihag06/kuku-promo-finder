"""One command to run the promo finder on a show.

    python run.py data                         # no key needed (rule-based mock mode)
    python run.py data --live                  # real models, vendor picked from .env
    python run.py show.zip --live --preset anthropic
    python run.py data --live --model gemini:gemini-3.8-flash   # one model for every stage
"""
from __future__ import annotations

import argparse
import sys

from promo_finder.config import load_config
from promo_finder.llm.client import LLMError
from promo_finder.pipeline import Pipeline


def main() -> int:
    ap = argparse.ArgumentParser(description="Find ranked promo clips in a microdrama screenplay.")
    ap.add_argument("input", help="folder or .zip of episode .txt files")
    ap.add_argument("--live", action="store_true", help="call real models (needs a key in .env)")
    ap.add_argument("--preset", help="auto | openai | anthropic | gemini | compatible")
    ap.add_argument("--model", help='override every stage, e.g. "anthropic:claude-haiku-5-5"')
    ap.add_argument("--config", help="path to an alternative config.yaml")
    ap.add_argument("--run-id", help="name of the output folder (default: timestamp)")
    ap.add_argument("--no-cache", action="store_true",
                    help="ignore saved answers (e.g. to test whether the judge gives stable scores)")
    args = ap.parse_args()

    overrides = {"llm.mode": "live" if args.live else None, "llm.preset": args.preset}
    if args.model:
        overrides["llm.models"] = {"default": args.model, "show_bible": args.model,
                                   "miner": args.model, "judge": args.model}
    cfg = load_config(args.config, overrides)
    try:
        Pipeline(cfg, args.input, run_id=args.run_id, use_cache=not args.no_cache).run()
    except (LLMError, FileNotFoundError) as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
