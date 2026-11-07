"""Evaluation code: gold IO, matching, Spearman, LLM-gold agreement, report end to end."""
import json
from pathlib import Path

import pytest

from eval.evaluate import Run, build_report, spearman
from eval.gold import GoldPromo, load_gold, match_level, save_gold
from eval.make_llm_gold import GoldPick, GoldPicks, GoldSegment, build_llm_gold
from promo_finder.config import LLMConfig, load_config
from promo_finder.llm.client import LLMClient
from promo_finder.pipeline import Pipeline
from promo_finder.schema import Show

from test_assembly import LINES, line, make_episode

FIX = Path(__file__).parent / "fixtures" / "mini_show"


def test_spearman():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    assert spearman([1, 2], [1, 2]) is None  # too few points to mean anything


def test_gold_roundtrip_skips_example_rows(tmp_path):
    g = [GoldPromo("G1", [(2, 4.6, 15.8), (4, 127.4, 151.4)], hook="h", cliffhanger="c", confidence=5)]
    save_gold(g, tmp_path / "g.csv")
    with open(tmp_path / "g.csv", "a") as f:
        f.write("G00,1,99,00:01.00,00:24.00,x,y,z,4,MADE-UP EXAMPLE - delete\n")
    back = load_gold(tmp_path / "g.csv")
    assert len(back) == 1 and back[0].segments == g[0].segments and back[0].duration == pytest.approx(35.2)


def test_match_levels():
    gold = GoldPromo("G", [(1, 10.0, 40.0)])
    tool = lambda segs: {"segments": [{"episode": e, "start": s, "end": t} for e, s, t in segs]}
    assert match_level(tool([(1, 9.0, 38.0)]), gold) == "strict"      # same opening, 93% covered
    assert match_level(tool([(1, 20.0, 45.0)]), gold) == "loose"      # different opening, 67% covered
    assert match_level(tool([(1, 11.0, 18.0)]), gold) == "loose"      # same opening, little footage
    assert match_level(tool([(1, 60.0, 90.0)]), gold) is None
    assert match_level(tool([(2, 10.0, 40.0)]), gold) is None         # other episode


def test_llm_gold_keeps_only_agreed_picks_and_drops_bad_ids(tmp_path):
    ep = make_episode(2, lines=LINES)
    show = Show(title="t", episodes=[ep])
    first, mid, late = line(ep, "Thief!").id, line(ep, "Where is he?").id, line(ep, "Run!").id
    liar = line(ep, "Liar!").id
    seg = lambda a, b: GoldSegment(start_event_id=a, end_event_id=b)
    answers = {
        "gold_a": GoldPicks(picks=[GoldPick(segments=[seg(first, mid)], confidence=5),
                                   GoldPick(segments=[seg(liar, late)], confidence=3),
                                   GoldPick(segments=[seg("E002-9999", mid)])]),        # invented id
        "gold_b": GoldPicks(picks=[GoldPick(segments=[seg(first, mid)], confidence=4)]),
    }
    cfg = load_config()
    llm = LLMClient(LLMConfig(mode="live", preset="openai",
                              presets={"openai": {"default": "openai:small", "gold": "openai:big"}}),
                    cache_dir=tmp_path)
    used = []

    def fake(spec, msgs):
        used.append(spec)
        variant = "gold_a" if "senior promo editor" in msgs[0]["content"] else "gold_b"
        return answers[variant].model_dump_json(), 10, 10

    llm._call_api = fake
    agreed, everything, notes = build_llm_gold(show, [2], llm, cfg)
    assert set(used) == {"openai:big"}                      # the gold model, not the tool's model
    assert len(agreed) == 1 and agreed[0].confidence in (4, 5)
    assert any("unknown id" in n for n in notes)
    assert {g.notes for g in everything} >= {"variant A only", "agreed"}


def test_report_end_to_end_on_a_mock_run(tmp_path):
    cfg = load_config(overrides={"llm.mode": "mock", "paths.output_dir": str(tmp_path / "out"),
                                 "paths.cache_dir": str(tmp_path / "c")})
    out = Pipeline(cfg, str(FIX), run_id="r", progress=lambda m: None).run()["out_dir"]
    run = Run(out)
    # sanity: a gold set made of the run's own top promo must be found at rank 1
    if run.promos:
        p = run.promos[0]
        gold = [GoldPromo("S", [(s["episode"], s["start"], s["end"]) for s in p["segments"]])]
        md, res = build_report([run], {"self": gold})
        assert "1/1 (100%)" in md and "found: #1 (strict)" in md
    md, res = build_report([run, run], {})
    assert "100%) pass" in md and "Run vs run" in md
    assert json.dumps(res, default=str)


def test_llm_ratings_are_reported_as_a_stand_in(tmp_path, monkeypatch):
    import csv as _csv
    import eval.evaluate as ev
    cfg = load_config(overrides={"llm.mode": "mock", "paths.output_dir": str(tmp_path / "out"),
                                 "paths.cache_dir": str(tmp_path / "c")})
    run = Run(Pipeline(cfg, str(FIX), run_id="r", progress=lambda m: None).run()["out_dir"])
    rated = tmp_path / "llm_ratings.csv"
    with open(rated, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["run_id", "promo_id", "hook", "cliffhanger", "standalone", "overall", "comment"])
        for i, p in enumerate(run.promos):
            w.writerow(["r", p["id"], 4, 3, 4, 5 - (i % 3), "fine"])
    monkeypatch.setattr(ev, "LLM_RATINGS", rated)
    monkeypatch.setattr(ev, "RATINGS", tmp_path / "none.csv")
    md, res = build_report([run], {})
    if run.promos:
        assert "LLM stand-in" in md and "mean overall" in md
