# Evaluation playbook (run in this order)

Run folders below are `outputs/i-married-a-broken-soldier-he-was-a-duke/<run-id>`; shortened to `<show>/<run-id>`.

| # | Command | What it gives you | Cost |
|---|---|---|---|
| 0 | Fill `eval/gold_human.csv` (see LABELLING_GUIDE.md) **before step 3** | Route A gold: your blind picks | your time, ~40 min |
| 1 | `python -m promo_finder.llm.client --check` | confirms your key and model names | < $0.01 |
| 2 | `python run.py data --run-id rules` | rules-only baseline (no API) | free |
| 3 | `python run.py data --live --run-id live` | the real run | ~$0.35-0.60 (gpt-5-mini) |
| 4 | `python -m eval.make_llm_gold data` | Route B gold from the stronger `gold` model, two prompts, agreed picks only | ~$1 (gpt-5.4, 24 calls) |
| 5 | `python -m eval.evaluate --run <show>/rules --run <show>/live` | report: rules, recall, where misses were lost, rules-vs-AI ablation | free |
| 6 | `python -m eval.evaluate --make-rating-sheet <show>/live` then fill `eval/human_ratings.csv` | blind 1-5 ratings of the top 20 (shuffled, no scores shown) | ~20 min |
| 7 | rerun step 5 | adds precision + judge-vs-human correlation | free |
| 8 | `python run.py data --live --no-cache --run-id live-2` then `python -m eval.evaluate --run <show>/live --run <show>/live-2` | judge stability: same input twice, do scores and the top 20 hold? | ~$0.35-0.60 |
| 9 (optional) | `python run.py data --live --model openai:gpt-5.4 --run-id live-strong` | "stronger model" swap the brief mentions | ~$3 |

Notes
- **Two keys (recommended setup):** with OPENAI_API_KEY and GEMINI_API_KEY both in `.env`, `preset: auto` runs the
  tool on OpenAI (gpt-5-mini) and builds the gold set with Gemini: a different vendor checks the tool. On a paid
  OpenAI key, raise `concurrency` to 10 in config.yaml for a ~2-3x faster run.
- **Faster, cheaper runs:** `reasoning: low` in config.yaml lets models think less. Measure it: run once with
  `null` and once with `low` (different --run-id), then compare both with eval/evaluate.py.
- **Gemini free tier caps some models at 20 requests/day.** The tool notices, skips that model for the run and
  uses its fallbacks; finished answers are cached, so rerunning the next day resumes.
- **Gemini key only (free tier):** `preset: auto` picks Gemini by itself: gemini-3.1-flash-lite for the tool,
  gemini-3.8-flash for the gold set. Costs above become $0 within free-tier limits. Free tiers cap requests per
  minute: the tool waits and retries automatically; if you still see many waits, set `concurrency: 2` in config.yaml.
- Step 9 uses the same model as the Route B gold, so compare it against **human** gold only.
- With a second vendor's key in `.env`, `preset: auto` builds the gold with that vendor automatically (more independent).
- The report lands in `outputs/eval/<timestamp>/report.md`. Section 8 lists top picks that match no gold
  promo: read them and write one line each on why they are good or bad. That is the failure analysis.
- Gold files, ratings and reports quote the scripts: they are gitignored; share them via the Drive folder.
