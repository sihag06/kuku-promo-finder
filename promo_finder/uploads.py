"""Turn browser uploads (episode .txt files and/or .zip archives, any mix) into one folder."""
from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path


def save_uploads(uploads) -> Path:
    """Accept any mix of episode .txt files and .zip archives; return one folder with all of them."""
    tmp = Path(tempfile.mkdtemp(prefix="upload_")) / "Uploaded episodes"
    tmp.mkdir(parents=True)
    for up in uploads:
        name = Path(up.name).name  # never trust a path from the browser
        if name.lower().endswith(".zip"):
            zpath = tmp.parent / name
            zpath.write_bytes(up.getvalue())
            with zipfile.ZipFile(zpath) as zf:
                for m in zf.infolist():  # refuse paths that escape the folder
                    if not str((tmp / m.filename).resolve()).startswith(str(tmp.resolve())):
                        raise ValueError(f"unsafe path in zip: {m.filename}")
                zf.extractall(tmp / Path(name).stem)
        elif name.lower().endswith(".txt"):
            (tmp / name).write_bytes(up.getvalue())
    if not list(tmp.rglob("*.txt")):
        raise ValueError("no episode .txt files found in the upload")
    return tmp
