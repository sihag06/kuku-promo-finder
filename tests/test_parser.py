"""Parser tests on a tiny synthetic show (no Kuku content in the repo)."""
import shutil
from pathlib import Path

import pytest

from promo_finder.parser import compact_transcript, load_show, parse_label, parse_time

FIX = Path(__file__).parent / "fixtures" / "mini_show"


def test_parse_time():
    assert parse_time("00:00.00") == 0.0
    assert parse_time("01:23.50") == 83.5
    assert parse_time("02:18.00") == 138.0
    assert parse_time("1:02:03.5") == 3723.5


@pytest.mark.parametrize(
    "label, expected",
    [
        ("CAMERA", ("camera", None, None, False)),
        ("ACTION", ("action", None, None, False)),
        ("SOUND FX", ("sfx", None, None, False)),
        ("MATRIARCH (Cold, Commanding)", ("dialogue", "MATRIARCH", "Cold, Commanding", False)),
        ("MATRIARCH (O.S.) (Firm)", ("dialogue", "MATRIARCH", "Firm", True)),
        ("FATHER (Solemn, O.S.)", ("dialogue", "FATHER", "Solemn", True)),
        ("DAMIEN (O.S. / Continues)", ("dialogue", "DAMIEN", None, True)),
        ("DAMIEN'S MOTHER (V.O.)", ("dialogue", "DAMIEN'S MOTHER", None, True)),
        ("LADY MARA", ("dialogue", "LADY MARA", None, False)),
    ],
)
def test_parse_label(label, expected):
    assert parse_label(label) == expected


@pytest.fixture(scope="module")
def show():
    return load_show(FIX)


def test_episodes_sorted_and_counted(show):
    assert [e.number for e in show.episodes] == [1, 2]


def test_header_fields(show):
    ep = show.episode(1)
    assert ep.logline.startswith("A baker is accused")
    assert "Synthetic" in ep.continuity


def test_scenes(show):
    ep = show.episode(1)
    assert [(s.number, s.start, s.end) for s in ep.scenes] == [(1, 0.0, 20.0), (2, 20.0, 30.0)]
    assert ep.scenes[1].heading == "INT. BAKERY – NIGHT"
    assert ep.duration == 30.0


def test_events_and_ids(show):
    ep = show.episode(1)
    types = [e.type for e in ep.events]
    assert types == ["camera", "action", "dialogue", "camera", "dialogue", "dialogue",
                     "camera", "sfx", "camera", "dialogue"]
    assert ep.events[0].id == "E001-0000"
    assert [e.index for e in ep.events] == list(range(len(ep.events)))
    assert ep.event("E001-0009").scene == 2


def test_dialogue_details(show):
    ep = show.episode(1)
    slap_line = ep.events[2]
    assert (slap_line.speaker, slap_line.emotion, slap_line.text) == ("MERCHANT", "Furious, Shouting", "Thief!")
    multi = ep.events[4]
    assert multi.text == "I took nothing. You know that."  # multi-line joined, quotes stripped
    assert multi.off_screen and multi.emotion == "Solemn"
    assert ep.events[5].off_screen and ep.events[5].emotion is None
    assert ep.events[9].off_screen  # V.O.


def test_bad_block_becomes_warning_not_crash(show):
    ep = show.episode(1)
    assert any("BROKEN BLOCK" in w and "no timing" in w for w in ep.warnings)


def test_swapped_times_and_colon_outside_label(show):
    ep = show.episode(2)
    assert ep.events[0].type == "camera"  # '**CAMERA**:' still recognised
    line = ep.events[1]
    assert (line.start, line.end) == (58.0, 62.0)
    assert any("swapped" in w for w in ep.warnings)


def test_shot_boundaries_are_camera_edges(show):
    assert show.episode(1).shot_boundaries == [0.0, 5.0, 12.0, 20.0, 30.0]


def test_compact_transcript_uses_ids_and_hides_camera(show):
    text = compact_transcript(show.episode(1))
    assert "[E001-0002" in text and 'MERCHANT (Furious, Shouting): "Thief!"' in text
    assert "CAMERA" not in text


def test_zip_input(tmp_path):
    archive = shutil.make_archive(str(tmp_path / "mini"), "zip", FIX.parent, FIX.name)
    show = load_show(archive)
    assert len(show.episodes) == 2 and show.title == "mini_show"


def test_empty_folder_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_show(tmp_path)
