"""Upload handling for the app: .txt files, .zip archives, any mix; unsafe zips refused."""
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pytest

from promo_finder.parser import load_show
from promo_finder.uploads import save_uploads

FIX = Path(__file__).parent / "fixtures" / "mini_show"


@dataclass
class Up:  # what Streamlit's UploadedFile gives us: a name and bytes
    name: str
    data: bytes

    def getvalue(self) -> bytes:
        return self.data


def zipped(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for n, b in files.items():
            zf.writestr(n, b)
    return buf.getvalue()


EP1, EP2 = (FIX / "episode-001-mini.txt").read_bytes(), (FIX / "episode-002-mini.txt").read_bytes()


def test_loose_txt_files():
    show = load_show(save_uploads([Up("episode-001-mini.txt", EP1), Up("episode-002-mini.txt", EP2)]))
    assert [e.number for e in show.episodes] == [1, 2]


def test_zip_and_txt_mixed():
    show = load_show(save_uploads([Up("first.zip", zipped({"show/ep1.txt": EP1})), Up("ep2.txt", EP2)]))
    assert [e.number for e in show.episodes] == [1, 2]


def test_browser_path_is_ignored():
    folder = save_uploads([Up("../../evil/episode-001-mini.txt", EP1)])
    assert (folder / "episode-001-mini.txt").exists()


def test_unsafe_zip_refused():
    with pytest.raises(ValueError, match="unsafe path"):
        save_uploads([Up("bad.zip", zipped({"../../escape.txt": EP1}))])


def test_upload_without_episodes_refused():
    with pytest.raises(ValueError, match="no episode"):
        save_uploads([Up("notes.zip", zipped({"readme.md": b"hi"}))])
