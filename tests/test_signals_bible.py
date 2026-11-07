"""Signals (Stage 2) and show bible (Stage 1) tests, on synthetic data only."""
from pathlib import Path

import pytest

from promo_finder.config import LLMConfig
from promo_finder.llm.client import LLMClient
from promo_finder.parser import load_show
from promo_finder.schema import Event
from promo_finder.show_bible import (Character, ShowBible, build_show_bible, rule_based_bible,
                                     speaker_stats, validate_bible)
from promo_finder.signals import score_episode, score_event

FIX = Path(__file__).parent / "fixtures" / "mini_show"


def ev(type_, text, start=10.0, end=12.0, emotion=None, speaker=None):
    return Event(id="E001-0001", episode=1, scene=1, index=1, type=type_, text=text,
                 start=start, end=end, emotion=emotion, speaker="X" if type_ == "dialogue" else speaker)


# ---- signals -------------------------------------------------------------------------
def test_slap_is_a_strong_hook_and_neutral_line_is_not():
    slap = score_event(ev("action", "The merchant slaps the baker across the face."), 120)
    calm = score_event(ev("action", "She walks to the window and looks outside."), 120)
    assert slap.hook >= 0.5 and "violence" in slap.hook_reasons
    assert calm.hook == 0.0


def test_hot_accusation_beats_plain_line():
    hot = score_event(ev("dialogue", "You knew all along!", emotion="Furious"), 120)
    plain = score_event(ev("dialogue", "The weather is fine today in the village."), 120)
    assert hot.hook > 0.5 > plain.hook


def test_word_boundaries_avoid_false_matches():
    # 'hither' must not count as 'hit'; 'waiting' must not count as 'wait'
    assert score_event(ev("action", "She beckons him hither."), 120).hook == 0.0
    assert "reveal pending" not in score_event(ev("dialogue", "Damien is waiting for me."), 120).cliff_reasons
    assert "reveal pending" not in score_event(ev("dialogue", "No one will recognize me."), 120).cliff_reasons
    assert "reveal pending" in score_event(ev("dialogue", "No..."), 120).cliff_reasons


def test_question_and_episode_end_make_a_cliffhanger():
    end_q = score_event(ev("dialogue", "Where is my brother?", start=115, end=118), 120)
    mid_q = score_event(ev("dialogue", "Where is my brother?", start=40, end=43), 120)
    assert end_q.cliff > mid_q.cliff > 0
    assert "episode ending" in end_q.cliff_reasons


def test_cliffhanger_cue_in_camera_text_counts():
    s = score_event(ev("camera", "Close-up as music hits an abrupt, suspenseful cliffhanger."), 120)
    assert "ominous cue" in s.cliff_reasons


def test_scores_stay_in_range_on_fixture():
    show = load_show(FIX)
    for ep in show.episodes:
        for s in score_episode(ep).by_event.values():
            assert 0.0 <= s.hook <= 1.0 and 0.0 <= s.cliff <= 1.0
    ep = show.episode(1)
    sig = score_episode(ep)
    top2 = [ep.event(s.event_id).text for s in sig.top("hook", 2)]  # no de-duplication
    assert any(t.startswith("A merchant slaps") for t in top2) and "Thief!" in top2
    # with de-duplication, the slap (1.0s) is folded into "Thief!" (3.2s): same moment
    deduped = [ep.event(s.event_id).text for s in sig.top("hook", 2, events=ep.events)]
    assert deduped[0] == "Thief!" and not any(t.startswith("A merchant") for t in deduped)


# ---- show bible ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def show():
    return load_show(FIX)


def test_speaker_stats(show):
    st = speaker_stats(show)
    assert st["BAKER"]["lines"] == 3 and st["BAKER"]["first_ep"] == 1


def test_rule_based_merges_title_only_when_bare_name_exists(show, monkeypatch):
    import promo_finder.show_bible as sb
    fake = {"LADY MARA": {"lines": 9}, "MARA": {"lines": 30},
            "LORD HALDEN": {"lines": 4}, "LADY HALDEN": {"lines": 3}}
    monkeypatch.setattr(sb, "speaker_stats", lambda s: fake)
    groups = {c.name: sorted(c.aliases) for c in rule_based_bible(show).characters}
    assert groups["Mara"] == ["LADY MARA", "MARA"]
    assert groups["Lord Halden"] == ["LORD HALDEN"] and groups["Lady Halden"] == ["LADY HALDEN"]


def test_validation_drops_invented_aliases_and_adds_missing_labels(show):
    llm_said = ShowBible(premise="p", characters=[
        Character(name="Baker", aliases=["BAKER", "THE BAKER GIRL"], role="accused baker"),  # invented alias
        Character(name="Ghost", aliases=["GHOST"]),                                         # invented person
    ])
    b = validate_bible(llm_said, show)
    amap = b.alias_map()
    assert amap["BAKER"] == "Baker" and "THE BAKER GIRL" not in amap and "GHOST" not in amap
    assert set(amap) == set(speaker_stats(show))  # every real speaker mapped exactly once
    assert any("dropped" in n for n in b.notes) and any("added unmapped" in n for n in b.notes)


def test_alias_used_twice_is_kept_once(show):
    b = validate_bible(ShowBible(characters=[Character(name="A", aliases=["BAKER"]),
                                             Character(name="B", aliases=["BAKER", "MERCHANT"])]), show)
    assert b.alias_map()["BAKER"] == "A" and b.alias_map()["MERCHANT"] == "B"


def test_build_in_mock_mode_is_not_cached(show, tmp_path):
    llm = LLMClient(LLMConfig(mode="mock", presets={"openai": {"default": "openai:x"}}),
                    cache_dir=tmp_path / "c", log_dir=tmp_path / "raw")
    b = build_show_bible(show, llm)
    assert b.canonical("BAKER") == "Baker"
    assert not any((tmp_path / "c").rglob("*.json"))  # mock answers never hit the cache
    assert (tmp_path / "raw" / "show_bible.jsonl").exists()


def test_live_mode_uses_model_answer_but_still_validates(show, tmp_path, monkeypatch):
    cfg = LLMConfig(mode="live", preset="openai", presets={"openai": {"default": "openai:x"}})
    llm = LLMClient(cfg, cache_dir=tmp_path / "c")
    answer = ShowBible(premise="A baker is framed.", characters=[
        Character(name="Mara", aliases=["BAKER"], role="the framed baker", importance=1),
        Character(name="Invented", aliases=["NOBODY"])]).model_dump_json()
    monkeypatch.setattr(llm, "_call_api", lambda spec, msgs: (answer, 100, 50))
    b = build_show_bible(show, llm)
    assert b.canonical("BAKER") == "Mara" and b.role_of("Mara") == "the framed baker"
    assert "NOBODY" not in b.alias_map()
