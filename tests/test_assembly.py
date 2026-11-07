"""Timing, rule checks, mining validation and candidate assembly on synthetic episodes."""
import pytest

from promo_finder.assembler import assemble, build, dedupe, Pick
from promo_finder.config import LLMConfig, PromoConfig
from promo_finder.llm.client import LLMClient
from promo_finder.miner import MinedEpisode, Moment, mine_episode, validate_moments
from promo_finder.schema import Episode, Event, Promo, Segment, Show
from promo_finder.show_bible import ShowBible
from promo_finder.signals import score_episode
from promo_finder.timing import legal_points, snap_end, snap_start
from promo_finder.validate import flags, violations

CFG = PromoConfig()


def make_episode(n: int, duration: float = 120.0, lines: dict | None = None) -> Episode:
    """Camera shots every 4s; dialogue lines at given {start: (end, text)}."""
    events, idx = [], 0
    t = 0.0
    while t < duration:
        events.append(dict(type="camera", text="shot", start=t, end=min(t + 4, duration)))
        t += 4
    for start, (end, text) in (lines or {}).items():
        events.append(dict(type="dialogue", text=text, start=start, end=end, speaker="A"))
    events.sort(key=lambda e: (e["start"], e["type"] != "camera"))
    evs = [Event(id=f"E{n:03d}-{i:04d}", episode=n, scene=1, index=i, **e) for i, e in enumerate(events)]
    shots = sorted({x for e in evs if e.type == "camera" for x in (e.start, e.end)})
    return Episode(number=n, source_file="x", events=evs, shot_boundaries=shots)


def line(ep: Episode, text: str) -> Event:
    return next(e for e in ep.events if e.text == text)


LINES = {10.5: (12.0, "Thief!"), 15.0: (17.0, "You knew?"), 39.0: (41.0, "Where is he?"),
         60.0: (61.0, "Liar!"), 70.0: (73.0, "Who are you?"), 115.0: (117.0, "Run!")}


# ---- timing ----------------------------------------------------------------------------
def test_snap_start_takes_latest_shot_before_hook():
    ep = make_episode(1, lines=LINES)
    assert snap_start(ep, 10.5, max_lead=2.8) == (8.0, True)  # shot at 8.0, hook at 10.5


def test_cut_points_never_split_a_line():
    ep = make_episode(1, lines={38.5: (41.0, "A long line across the 40s shot cut.")})
    assert 40.0 not in legal_points(ep)
    end, clean = snap_end(ep, 38.9, max_tail=5.5)
    assert end == 44.0 and clean  # skips 40.0 because the line is still running


def test_fallback_when_no_shot_is_close_enough():
    ep = make_episode(1, lines={2.5: (3.0, "x")})
    ep.shot_boundaries = [0.0, 120.0]
    start, clean = snap_start(ep, 50.0, max_lead=2.8)
    assert not clean and start == pytest.approx(49.7)


# ---- rule checks ---------------------------------------------------------------------
def promo(segs, hook, cliff, template="continuous"):
    return Promo(id="X", template=template, hook_event=hook, cliff_event=cliff,
                 segments=[Segment(episode=e, start=s, end=t, first_event="", last_event="") for e, s, t in segs])


@pytest.fixture
def show():
    eps = [make_episode(1, lines=LINES), make_episode(2, lines=LINES)]
    return Show(title="t", episodes=eps)


def test_valid_continuous_passes(show):
    ep = show.episode(1)
    p = promo([(1, 8.0, 44.0)], line(ep, "Thief!").id, line(ep, "Where is he?").id)
    assert violations(p, show, CFG) == []


@pytest.mark.parametrize("segs, hook, cliff, expect", [
    ([(1, 8.0, 20.0)], "Thief!", "You knew?", "duration"),
    ([(1, 4.0, 44.0)], "Thief!", "Where is he?", "hook starts"),
    ([(1, 8.0, 40.0)], "Thief!", "Where is he?", "cuts a spoken line"),
    ([(1, 8.0, 52.0)], "Thief!", "Where is he?", "past the cliffhanger"),
    ([(1, 12.0, 44.0)], "Thief!", "Where is he?", "not inside the first segment"),
])
def test_rule_violations(show, segs, hook, cliff, expect):
    ep = show.episode(1)
    p = promo(segs, line(ep, hook).id, line(ep, cliff).id)
    assert any(expect in v for v in violations(p, show, CFG)), violations(p, show, CFG)


def test_cross_episode_must_run_forward(show):
    p = promo([(2, 8.0, 20.0), (1, 32.0, 44.0)], line(show.episode(2), "Thief!").id,
              line(show.episode(1), "Where is he?").id, "cross_episode")
    assert any("episode order" in v for v in violations(p, show, CFG))


def test_flags(show):
    ep = show.episode(1)
    p = promo([(1, 8.0, 20.0), (2, 32.0, 44.0)], line(ep, "Thief!").id,
              line(show.episode(2), "Where is he?").id, "cross_episode")
    assert "cross_episode" in flags(p, show) and "visual_hook" not in flags(p, show)


# ---- mining ----------------------------------------------------------------------------
def test_unknown_event_ids_are_dropped(show):
    ep = show.episode(1)
    real = line(ep, "Thief!").id
    m = MinedEpisode(hooks=[Moment(event_id=real, strength=5), Moment(event_id="E001-9999", strength=5),
                            Moment(event_id=real, strength=4)])
    v = validate_moments(m, ep, 6)
    assert [h.event_id for h in v.hooks] == [real] and any("E001-9999" in n for n in v.notes)


def test_live_miner_cannot_smuggle_a_fake_moment(show, tmp_path, monkeypatch):
    ep = show.episode(1)
    llm = LLMClient(LLMConfig(mode="live", preset="openai", presets={"openai": {"default": "openai:x"}}),
                    cache_dir=tmp_path)
    fake = MinedEpisode(hooks=[Moment(event_id="E001-0500", strength=5, quote="invented slap")],
                        cliffhangers=[Moment(event_id=line(ep, "Who are you?").id, strength=4)])
    monkeypatch.setattr(llm, "_call_api", lambda s, m: (fake.model_dump_json(), 10, 10))
    out = mine_episode(ep, ShowBible(), score_episode(ep), llm)
    assert out.hooks == [] and len(out.cliffhangers) == 1


# ---- assembly --------------------------------------------------------------------------
def mined_for(ep, hooks, cliffs):
    return MinedEpisode(hooks=[Moment(event_id=line(ep, h).id, strength=5, quote=h) for h in hooks],
                        cliffhangers=[Moment(event_id=line(ep, c).id, strength=4, quote=c) for c in cliffs])


def test_build_shapes(show):
    e1, e2 = show.episode(1), show.episode(2)
    pk = lambda ep, t: Pick(ep, line(ep, t), Moment(event_id=line(ep, t).id, strength=5))
    assert build("continuous", pk(e1, "Thief!"), pk(e1, "Where is he?"), CFG)[0] == [(e1, 8.0, 44.0)]
    assert build("continuous", pk(e1, "Where is he?"), pk(e1, "Thief!"), CFG) is None  # cliff before hook
    cold = build("cold_open", pk(e1, "Liar!"), pk(e1, "Where is he?"), CFG)[0]
    assert cold[0][1] > cold[1][2] - 1e-6  # the later hook is shown first
    cross = build("cross_episode", pk(e1, "Thief!"), pk(e2, "Who are you?"), CFG)[0]
    assert [x[0].number for x in cross] == [1, 2]


def test_assemble_produces_only_valid_candidates(show):
    sigs = {ep.number: score_episode(ep) for ep in show.episodes}
    mined = {1: mined_for(show.episode(1), ["Thief!", "Liar!"], ["Where is he?", "Who are you?", "Run!"]),
             2: mined_for(show.episode(2), ["Thief!"], ["Where is he?", "Who are you?"])}
    # small show: default quotas (2 per hook, 4 per episode) would prune the cross-episode variant
    cfg = PromoConfig(max_per_hook=3, candidates_per_episode=8)
    cands, rejected = assemble(show, mined, sigs, cfg)
    assert cands, rejected
    for p in cands:
        assert violations(p, show, CFG) == []
        assert CFG.min_duration_s <= p.duration <= CFG.max_duration_s
    assert {"continuous", "cross_episode"} <= {p.template for p in cands}
    per_hook = {}
    for p in cands:
        per_hook[p.hook_event] = per_hook.get(p.hook_event, 0) + 1
    assert max(per_hook.values()) <= cfg.max_per_hook
    assert all(r.reasons for r in rejected)


def test_dedupe_keeps_the_better_of_two_overlapping(show):
    a = promo([(1, 8.0, 44.0)], "E001-0001", "E001-0002"); a.pre_score = 0.9
    b = promo([(1, 8.0, 40.0)], "E001-0001", "E001-0002"); b.pre_score = 0.5
    c = promo([(1, 60.0, 90.0)], "E001-0003", "E001-0002"); c.pre_score = 0.4
    # d shares a's footage but opens elsewhere (different hook): a different ad, kept
    d = promo([(2, 0.0, 10.0), (1, 20.0, 44.0)], "E002-0001", "E001-0002"); d.pre_score = 0.3
    kept, dropped = dedupe([b, a, c, d], 0.5)
    assert a in kept and c in kept and d in kept and dropped == [b]


def test_too_short_clip_is_extended_to_the_next_open_line():
    lines = {10.5: (12.0, "Thief!"), 16.0: (17.5, "Fine."), 33.0: (35.0, "Where is my brother?")}
    ep = make_episode(1, lines=lines)
    show = Show(title="t", episodes=[ep])
    sigs = {1: score_episode(ep)}
    early = line(ep, "Fine.")  # a cliffhanger only 9s after the hook: clip would be too short
    mined = {1: MinedEpisode(hooks=[Moment(event_id=line(ep, "Thief!").id, strength=5, quote="Thief!")],
                             cliffhangers=[Moment(event_id=early.id, strength=4, quote="Fine.")])}
    cands, rejected = assemble(show, mined, sigs, CFG)
    cont = [p for p in cands if p.template == "continuous"]
    assert cont and cont[0].cliff_event == line(ep, "Where is my brother?").id
    assert CFG.min_duration_s <= cont[0].duration <= 45 and violations(cont[0], show, CFG) == []


def test_duplicate_of_a_pruned_clip_is_not_lost():
    """Regression: A and B open the same way and A scores higher, but A is pruned because its
    hook is over quota. B only 'duplicates' a clip that was never kept, so B must survive."""
    from promo_finder.assembler import select
    mk = lambda pid, hook, segs, score: Promo(
        id=pid, template="continuous", hook_event=hook, cliff_event="E001-0001", pre_score=score,
        segments=[Segment(episode=1, start=s, end=e, first_event="", last_event="") for s, e in segs])
    x1 = mk("X1", "E001-0009", [(60.0, 90.0)], 0.99)   # uses hook E001-0009's only slot
    a = mk("A", "E001-0009", [(8.0, 44.0)], 0.90)      # same hook: pruned by the per-hook quota
    b = mk("B", "E001-0003", [(8.0, 40.0)], 0.80)      # same opening as A, different hook
    final, rejected = select([b, a, x1], PromoConfig(max_per_hook=1))
    assert [p.id for p in final] == ["X1", "B"]       # old two-pass logic lost B as A's "duplicate"
    assert [r.reasons[0] for r in rejected] == ["pruned: hook moment already used enough"]
