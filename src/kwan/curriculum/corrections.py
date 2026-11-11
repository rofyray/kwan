"""Corrections for misprints in the curriculum PDFs (`data/curriculum/corrections.json`).

The PDFs print some indicator codes wrongly: a wrong digit, a missing part, the same code used for two
different indicators, or a statement repeated on a duplicated page. Each entry in the corrections file
names the code as printed, where it sits, what to do with it and why. The corrected code is canonical;
the printed code is kept as an alias of the indicator when no other indicator owns that code.

Actions:
    recode        the statement is a real indicator whose code is `corrected`
    merge         the statement repeats the indicator `corrected`; its exemplars are added to that one
    drop_heading  the line is not an indicator; the text under it stays with the indicator it belongs to

This module needs only the standard library, so any part of Kwan can use it to map draft keys.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kwan._paths import data_dir

ACTIONS = ("recode", "merge", "drop_heading")


@dataclass
class Correction:
    """One entry of the corrections file."""

    printed: str
    corrected: str | None
    action: str = "recode"
    doc: str | None = None
    occurrence: int | None = None  # 1-based count of the printed code within the document; None = every one
    alias: bool = True  # keep the printed code as an alias of the corrected indicator
    draft_key: str | None = None  # the key the drafting agents used for this indicator
    page: int | None = None
    source: str = ""
    reason: str = ""


@dataclass
class Corrections:
    """All corrections of one subject."""

    entries: list[Correction] = field(default_factory=list)
    sub_strand_names: dict[str, str] = field(default_factory=dict)  # "B4.1.4" -> name, where no heading is printed
    draft_keys: dict[str, str] = field(default_factory=dict)  # draft key -> canonical code

    def find(self, printed: str, doc: str | None, occurrence: int) -> Correction | None:
        """The entry for the `occurrence`-th time `printed` appears in `doc`, if there is one."""
        for c in self.entries:
            if c.printed != printed or (c.doc is not None and doc is not None and c.doc != doc):
                continue
            if c.occurrence is None or c.occurrence == occurrence:
                return c
        return None

    def has(self, printed: str, doc: str | None = None) -> bool:
        """True when some entry talks about this printed code."""
        return any(c.printed == printed and (c.doc is None or doc is None or c.doc == doc) for c in self.entries)


def corrections_from_dict(data: dict[str, Any]) -> Corrections:
    """Build `Corrections` from the JSON content of a corrections file."""
    entries = []
    for raw in data.get("corrections", []):
        action = raw.get("action", "recode")
        if action not in ACTIONS:
            raise ValueError(f"unknown correction action {action!r} for {raw.get('printed')!r}")
        entries.append(
            Correction(
                printed=raw["printed"],
                corrected=raw.get("corrected"),
                action=action,
                doc=raw.get("doc"),
                occurrence=raw.get("occurrence"),
                alias=bool(raw.get("alias", action == "recode")),
                draft_key=raw.get("draft_key"),
                page=raw.get("page"),
                source=raw.get("source", ""),
                reason=raw.get("reason", ""),
            )
        )
    return Corrections(
        entries=entries,
        sub_strand_names=dict(data.get("sub_strand_names", {})),
        draft_keys=dict(data.get("draft_keys", {})),
    )


def load_corrections(path: Path | str | None = None) -> Corrections:
    """Read a corrections file; a missing file means no corrections."""
    path = Path(path) if path else data_dir() / "curriculum" / "corrections.json"
    if not path.exists():
        return Corrections()
    return corrections_from_dict(json.loads(path.read_text(encoding="utf-8")))


def draft_key_map(path: Path | str | None = None) -> dict[str, str]:
    """`{draft key: canonical bare code}` for every draft key that is not already a canonical code.

    Draft files under `data/contexts/drafts/` are keyed by the code as printed (plus `<code>_dup` for a
    second use of one code). A key that is missing from this map is already canonical.
    """
    corr = load_corrections(path)
    out = dict(corr.draft_keys)
    for c in corr.entries:
        if c.action == "recode" and c.corrected and c.draft_key and c.draft_key != c.corrected:
            out.setdefault(c.draft_key, c.corrected)
    return out
