"""Load prompt templates from prompts/<name>.md.

Prompts live in plain files (not inside code) so they can be read, reviewed and
versioned on their own. Each file has a `## system` and a `## user` section;
placeholders use $name (string.Template), so JSON braces in examples are safe.
"""
from __future__ import annotations

import re
from pathlib import Path
from string import Template

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load(name: str, **values: str) -> tuple[str, str]:
    text = (PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8")
    parts = re.split(r"^##\s+(system|user)\s*$", text, flags=re.M | re.I)
    sections = {parts[i].lower(): parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}
    if "system" not in sections or "user" not in sections:
        raise ValueError(f"prompts/{name}.md needs '## system' and '## user' sections")
    return (Template(sections["system"]).substitute(values),
            Template(sections["user"]).substitute(values))
