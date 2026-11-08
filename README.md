# Promo Finder

Reads a microdrama screenplay and finds the best 20–60 second promo clips in it: the episode,
the start and end of every segment, the hook, the cliffhanger, one line on why it works, and a
ranked posting order. Built for the Kuku AI Generalist assignment (Problem 1).

- **Live app:** [kuku-promo-finder.streamlit.app](https://kuku-promo-finder-hfjphhfrtjvbvvfcq9uev8.streamlit.app/)
- **Walkthrough video (3 min):** [Google Drive](https://drive.google.com/file/d/13Dw0ZphtwPwrmaa-AQfd5iqCKZMn5wh4/view?usp=sharing)
- **Solution doc (2 pages):** `docs/solution_doc.pdf`
- **Raw outputs from the full run:** [Google Drive](https://drive.google.com/file/d/1-LQm_GJ-cK1uMJ6OKX9BCN7Ts5bDO3WP/view?usp=sharing)

> The input pack (Kuku's scripts) is confidential and is **not** in this repo. The app ships with
> a short, original, fictional demo show (`samples/demo_show/`) so anyone can try it.

## Results on the 50-episode input pack

| | Rules only (no AI) | AI (gpt-5-mini, judge) |
|---|---|---|
| Promos breaking a brief rule (20–60s, hook ≤ 3s, no mid-line cut, ≤ 3 segments) | 0 / 150 | 0 / 150 |
| My 15 hand-picked promos found anywhere in the list | 9 (60%) | 12 (80%) |
| My hand-picked promos found in the top 20 | 2 (13%) | 5 (33%) |
| Top-20 promos that match a human or GPT-5.4 pick | 12 / 20 | 16 / 20 |
| Top 20 rated 4+ / 5 by a blind independent rater (GPT-5.4, AI stand-in for a human) | – | 13 / 20 (65%), mean 3.6 |
| Cost and time per 50-episode show | $0, 1 s | $0.72, ~17 min at 4 parallel calls (reruns ~$0) |

Full method and failure analysis: `docs/solution_doc.pdf` and `eval/`.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q                       # 105 tests, no key or network needed

python run.py samples/demo_show           # rules only: free, offline, ~1 s
streamlit run app.py                      # the editor's review screen
```

Run on a real show (a folder or .zip of episode `.txt` files):

```bash
cp .env.example .env                      # add ANY one key: OpenAI, Anthropic or Gemini
python -m promo_finder.llm.client --check # confirms the key and every configured model
python run.py path/to/show --live         # writes outputs/<show>/<timestamp>/
```

## How it works

```
scripts ─► parse ─► show bible ─► signals ─► mine moments ─► assemble ─► judge ─► validate ─► rank ─► editor review
  .txt     code       AI (1)        code      AI (1/episode)    code    AI (1/clip)   code       code      human (app)
```

The one rule that makes it trustworthy: **the AI never writes a timestamp.** It points at lines by id
(`E007-0042`); code turns ids into times, snaps every cut to a camera change, never cuts a spoken line,
and checks every rule from the brief. AI does only what needs judgement:

| Stage | Who | What |
|---|---|---|
| Parse | code | screenplay → timed events (camera, action, dialogue, SFX); camera changes = legal cut points |
| Show bible | AI, once per show | who is who across label variants (e.g. a character introduced as a description, named later) |
| Signals | code | free rule scores for hooks (slaps, reveals, accusations) and cliffhangers (open questions, threats, episode endings) |
| Mine moments | AI, per episode | up to 6 hooks and 6 cliffhangers, by line id; invented ids are dropped by code |
| Assemble | code | 4 shapes: continuous, condensed, cold open (flash-forward), cross-episode; too-short clips extend to the next open line |
| Judge | AI, per clip | sees **only the clip's lines** (the honest "makes sense to a stranger" test); scores hook (first 3 s), cliffhanger (last 5 s), works-cold, emotion, spoiler risk, ad safety |
| Rank | code | `quality_rank` (best promos) and posting order (quality + early-episode funnel + variety); weights in `config.yaml` |
| Review | human | Streamlit app: read each clip's lines, accept / reject / needs-edit; acceptance rate is the live success metric |

## Swap the model (one line)

Every AI call goes through `promo_finder/llm/client.py`. Each stage's model is a `provider:model` string in
`config.yaml`:

```yaml
llm:
  mode: live
  preset: auto          # whichever key is in .env: openai | anthropic | gemini | compatible
  models: {judge: anthropic:claude-sonnet-5-5}    # optional per-stage override
  base_url: null        # for preset: compatible (Ollama, OpenRouter, or an API you provide)
```

Or from the command line: `python run.py show/ --live --model gemini:gemini-3.6-flash`.
Busy or rate-limited models fall back to the next model in `llm.fallbacks`; daily free-tier caps are detected;
every answer is cached, so a rerun costs nothing. `reasoning: low` makes models think less (faster, cheaper).

## Outputs (`outputs/<show>/<run_id>/`)

| File | What |
|---|---|
| `promos_ranked.csv` | the editor's sheet: rank, quality rank, tier, every segment's episode/start/end, hook, cliffhanger, why, scores, flags |
| `promos_ranked.md` | top 20 with the exact lines of each clip, then the rest |
| `promos.json` | everything, machine-readable |
| `rejected.json` | every candidate that broke a rule or was pruned, with the reason |
| `raw_llm/*.jsonl` | every prompt and response, unedited |
| `run_meta.json` | models, tokens, cost, seconds per stage, config |

## Evaluate

```bash
python run.py data --run-id rules                        # baseline without AI
python run.py data --live --run-id live
python -m eval.make_llm_gold data --episodes 2 3 5 ...   # independent GPT-5.4 picks, two prompts, agreed only
python -m eval.llm_rate outputs/<show>/live              # blind 1-5 ratings of the top 20 (AI stand-in)
python -m eval.evaluate --run outputs/<show>/rules --run outputs/<show>/live
```

The report covers: rule re-check, recall against human and GPT picks (strict and loose), where every missed
pick was lost (never mined, rejected and why, or ranked low), ratings, and run-vs-run comparisons.
`eval/LABELLING_GUIDE.md` explains how the 15 human picks were made (blind, before any tool output).

## Repo layout

```
run.py, app.py, config.yaml     CLI, Streamlit app, all settings
promo_finder/                   parser, show_bible, signals, miner, timing, assembler, judge,
                                validate, ranker, export, pipeline, uploads, llm/{client,providers,cache}
prompts/                        every prompt, as plain files
eval/                           gold sets, LLM gold, LLM rater, evaluation report
samples/                        fictional demo show (also the "new show" rehearsal)
tests/                          105 tests, all offline
```

## Known limits

- Judges text, not video: visual hooks are flagged (`visual_hook`) for an editor to check the footage.
- Assumes script timestamps match the final cut.
- Only 150 candidates (max 4 per episode) are judged, to control cost: 3 of 15 human picks were cut there.
- Order *inside* the top 20 does not agree with the blind rater (Spearman ≈ −0.2, n = 20): trust the shortlist, not #1 vs #4.
- Stitched clips (cold open, cross-episode) get the lowest ratings when the jump changes scene or cast.
- The cold-viewer judge ranks late-season plot reveals low; a fan of the show rates them high. Only a live A/B test settles which works better as an ad.
