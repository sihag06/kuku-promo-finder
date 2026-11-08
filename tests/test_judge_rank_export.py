"""Judge, ranker, exports and an end-to-end mock run. Synthetic data only."""
import csv
import json
from pathlib import Path

import pytest

from promo_finder.config import LLMConfig, RankingConfig, load_config
from promo_finder.export import write_csv
from promo_finder.judge import Judgement, clip_view, judge_promo, keyword_safety
from promo_finder.llm.client import LLMClient
from promo_finder.pipeline import Pipeline
from promo_finder.ranker import apply_judgement, rank, similarity
from promo_finder.schema import Promo, Scores, Segment, Show
from promo_finder.show_bible import ShowBible
from promo_finder.signals import score_episode

from test_assembly import LINES, line, make_episode

FIX = Path(__file__).parent / "fixtures" / "mini_show"
W = load_config().ranking.weights


@pytest.fixture
def show():
    return Show(title="t", episodes=[make_episode(1, lines=LINES), make_episode(2, lines=LINES)])


def promo(show, segs, hook="Thief!", cliff="Where is he?", **kw):
    ep_h, ep_c = show.episode(segs[0][0]), show.episode(segs[-1][0])
    return Promo(id=kw.pop("id", "P1"), template=kw.pop("template", "continuous"),
                 hook_event=line(ep_h, hook).id, cliff_event=line(ep_c, cliff).id,
                 segments=[Segment(episode=e, start=s, end=t, first_event="", last_event="") for e, s, t in segs],
                 **kw)


# ---- judge ------------------------------------------------------------------------------
def test_clip_view_is_clip_relative_and_splits_first_and_last(show):
    p = promo(show, [(1, 8.0, 20.0), (2, 32.0, 44.0)], template="cross_episode")
    full, first3, last5 = clip_view(p, show)
    assert "--- cut ---" in full
    assert '"Thief!"' in first3 and "Where is he?" not in first3   # Thief! at 2.5s of the clip
    assert "Where is he?" in last5                                 # 12s + 7s = 19s of a 24s clip
    assert "  2.5s" in full                                        # times are clip-relative


def test_keyword_safety():
    assert keyword_safety("She lies in bed in a silk chemise.")[0] == "review"
    assert keyword_safety("Her brother laughs at the market.")[0] == "ok"
    assert keyword_safety("They threaten to send her seven-year-old brother to the gaol")[0] == "review"


def live_llm(tmp_path, answer: Judgement):
    llm = LLMClient(LLMConfig(mode="live", preset="openai", presets={"openai": {"default": "openai:x"}}),
                    cache_dir=tmp_path)
    llm._call_api = lambda spec, msgs: (answer.model_dump_json(), 50, 20)
    return llm


def test_judge_sees_only_the_clip(show, tmp_path):
    p = promo(show, [(1, 8.0, 44.0)])
    seen = {}
    llm = live_llm(tmp_path, Judgement(hook_score=4, cliffhanger_score=4, standalone_score=4, emotion_score=3))
    orig = llm._call_api
    llm._call_api = lambda spec, msgs: (seen.setdefault("user", msgs[-1]["content"]), orig(spec, msgs))[1]
    bible = ShowBible(premise="SECRET PREMISE", characters=[])
    judge_promo(p, show, bible, {1: score_episode(show.episode(1))}, llm)
    assert "SECRET PREMISE" not in seen["user"] and "Liar!" not in seen["user"]  # nothing outside the clip


def test_code_safety_check_can_raise_but_not_lower(show, tmp_path):
    ep = show.episode(1)
    ep.events[0].text = "Vivienne in a silk chemise on the bed"  # inside the clip
    p = promo(show, [(1, 0.0, 44.0)], hook="Thief!")
    sigs = {1: score_episode(ep)}
    j = judge_promo(p, show, ShowBible(), sigs,
                    live_llm(tmp_path, Judgement(hook_score=3, cliffhanger_score=3, standalone_score=3,
                                                 emotion_score=3, ad_safety="ok")))
    assert j.ad_safety == "review" and "keyword" in j.ad_safety_reason


# ---- ranker -----------------------------------------------------------------------------
def judged(show, segs, h, c, s, e=3, spoiler=0, safety="ok", **kw):
    p = promo(show, segs, **kw)
    sigs = {n: score_episode(show.episode(n)) for n in (1, 2)}
    j = Judgement(hook_score=h, cliffhanger_score=c, standalone_score=s, emotion_score=e,
                  spoiler_risk=spoiler, ad_safety=safety)
    return apply_judgement(p, j, show, sigs, RankingConfig(weights=W))


def test_score_formula_and_penalties(show):
    best = judged(show, [(1, 8.0, 44.0)], 5, 5, 5, 5)
    worst = judged(show, [(1, 8.0, 44.0)], 1, 1, 1, 1)
    assert best.scores.hook == 1.0 and worst.scores.hook == 0.0
    assert best.scores.total > worst.scores.total
    flagged = judged(show, [(1, 8.0, 44.0)], 5, 5, 5, 5, spoiler=1, safety="review")
    assert flagged.scores.total == pytest.approx(best.scores.total - 0.05 - 0.05, abs=1e-3)
    assert "ad_safety_review" in flagged.flags


def test_unsafe_and_major_spoilers_are_held_at_the_bottom(show):
    a = judged(show, [(1, 8.0, 44.0)], 5, 5, 5, safety="unsafe", id="A")
    b = judged(show, [(2, 8.0, 44.0)], 2, 2, 2, id="B")
    c = judged(show, [(2, 56.0, 76.0)], 5, 5, 5, spoiler=2, hook="Liar!", cliff="Who are you?", id="C")
    out = rank([a, b, c], RankingConfig(weights=W, top_k=20))
    assert out[0].id == "B"  # the only postable promo leads, despite a lower score
    assert {p.id: p.tier for p in out}["A"] == "hold" and {p.id: p.tier for p in out}["C"] == "hold"


def test_diversity_lifts_a_different_promo_above_a_near_copy(show):
    top = judged(show, [(1, 8.0, 44.0)], 5, 5, 4, id="TOP", hook_type="humiliation")
    twin = judged(show, [(1, 8.0, 40.0)], 5, 5, 4, id="TWIN", hook_type="humiliation")
    other = judged(show, [(2, 56.0, 76.0)], 5, 4, 4, hook="Liar!", cliff="Who are you?", id="OTHER",
                   hook_type="reveal")
    assert similarity(top, twin) > similarity(top, other)
    assert twin.scores.total > other.scores.total
    out = rank([top, twin, other], RankingConfig(weights=W, diversity_lambda=0.3))
    assert [p.id for p in out] == ["TOP", "OTHER", "TWIN"]
    assert "variety" in out[1].rank_reason


# ---- export + end to end -------------------------------------------------------------------
def test_csv_has_every_segment_with_times(show, tmp_path):
    p = judged(show, [(1, 8.0, 20.0), (2, 32.0, 44.0)], 4, 4, 4, template="cross_episode")
    p = rank([p], RankingConfig(weights=W))[0]
    write_csv([p], show, tmp_path / "x.csv")
    row = next(csv.DictReader(open(tmp_path / "x.csv")))
    assert (row["seg1_episode"], row["seg1_start"], row["seg1_end"]) == ("1", "00:08.00", "00:20.00")
    assert (row["seg2_episode"], row["seg2_start"], row["seg2_end"]) == ("2", "00:32.00", "00:44.00")
    assert row["hook_offset_s"] == "2.5" and row["rank"] == "1"


def test_end_to_end_mock_run_writes_every_output(tmp_path):
    cfg = load_config(overrides={"llm.mode": "mock", "paths.output_dir": str(tmp_path / "out"),
                                 "paths.cache_dir": str(tmp_path / "cache")})
    res = Pipeline(cfg, str(FIX), run_id="t", progress=lambda m: None).run()
    out = res["out_dir"]
    for f in ("promos_ranked.csv", "promos_ranked.md", "promos.json", "candidates_all.json",
              "rejected.json", "run_meta.json", "show_bible.json", "moments.json"):
        assert (out / f).exists(), f
    meta = json.loads((out / "run_meta.json").read_text())
    assert meta["mode"] == "mock" and meta["episodes"] == 2
    assert (out / "raw_llm" / "miner.jsonl").exists()


def test_many_confusions_cap_works_cold_score(show):
    p = promo(show, [(1, 8.0, 44.0)])
    sigs = {1: score_episode(show.episode(1)), 2: score_episode(show.episode(2))}
    clear = apply_judgement(p, Judgement(hook_score=4, cliffhanger_score=4, standalone_score=5, emotion_score=3,
                                         confusions=["who is she?"]), show, sigs, RankingConfig(weights=W))
    muddled = apply_judgement(p, Judgement(hook_score=4, cliffhanger_score=4, standalone_score=5, emotion_score=3,
                                           confusions=["a", "b", "c", "d", "e"]), show, sigs, RankingConfig(weights=W))
    assert clear.scores.standalone == 1.0 and muddled.scores.standalone == pytest.approx(0.6)
    assert muddled.scores.quality < clear.scores.quality


def test_quality_rank_ignores_the_early_episode_preference(show):
    early = judged(show, [(1, 8.0, 44.0)], 4, 4, 4, id="EARLY")
    late = judged(show, [(2, 56.0, 76.0)], 5, 5, 5, hook="Liar!", cliff="Who are you?", id="LATE")
    late.scores.funnel = 0.0; early.scores.funnel = 1.0
    early.scores.total, late.scores.total = 0.90, 0.80  # funnel bonus lifts the weaker early promo
    out = {p.id: p for p in rank([early, late], RankingConfig(weights=W))}
    assert out["EARLY"].rank == 1 and out["LATE"].quality_rank == 1  # post early first, but late is better


def test_funnel_uses_the_episodes_actually_uploaded():
    """Regression: an upload of Episodes 28-29 gave funnel = -27 (formula assumed episode 1 start)."""
    from promo_finder.ranker import funnel
    p28 = Promo(id="a", template="continuous", hook_event="E028-0001", cliff_event="E028-0002",
                segments=[Segment(episode=28, start=0, end=30, first_event="", last_event="")])
    p29 = p28.model_copy(update={"segments": [Segment(episode=29, start=0, end=30, first_event="", last_event="")]})
    assert funnel(p28, [28, 29]) == 1.0 and funnel(p29, [28, 29]) == 0.0
    assert funnel(p28, [28]) == 1.0                       # single episode: no preference
    assert funnel(p28, list(range(1, 51))) == pytest.approx(1 - 27 / 49, abs=1e-3)  # full show unchanged
