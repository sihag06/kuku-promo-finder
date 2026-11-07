"""Disk cache for LLM responses: same model + same prompt -> same answer, for free.

Makes reruns cheap and deterministic, and means a prompt tweak only re-pays for the
stage whose prompt changed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional


def cache_key(**parts: Any) -> str:
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class DiskCache:
    def __init__(self, directory: str | Path, enabled: bool = True):
        self.dir = Path(directory)
        self.enabled = enabled
        if enabled:
            self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.dir / key[:2] / f"{key}.json"

    def get(self, key: str) -> Optional[dict]:
        if not self.enabled:
            return None
        p = self._path(key)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None  # corrupt entry: treat as a miss

    def set(self, key: str, value: dict) -> None:
        if not self.enabled:
            return
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        tmp.replace(p)
