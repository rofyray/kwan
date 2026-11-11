"""Draft practice settings: read the draft files and merge them into `data/contexts/<subject>.json`.

The draft files (`data/contexts/drafts/<subject>.*.json`) were written offline by a language model
and hold, per indicator, one short label and two or three settings:

    {"meta": {"drafted_by": "<model id> (...)", "source_file": "raw/m46.txt", ...},
     "indicators": {"B4.1.1.1.2": {"short_label": "...", "settings": ["...", "..."]}}}

Draft files are keyed by the code as printed in the curriculum PDF. Some printed codes are
misprints, so every key goes through `data/curriculum/corrections.json` (its `draft_keys` map)
to get the canonical code. When that file is missing the printed code is used as it is and the merge
says so: run the merge again once the corrections file exists.

The merge never marks anything as reviewed and never drops a teacher's decision:

    - a draft that is already in the file (same text) is left alone, with its review status;
    - a draft that a teacher edited or rejected stays edited or rejected;
    - an unreviewed draft whose text is no longer in the draft files is removed (it was rewritten);
    - a draft that fails the automatic checks is not merged and is listed in the result.

Only the standard library is used.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kwan._paths import data_dir
from kwan.context import checks
from kwan.context.settings import (
    DEFAULT_SUBJECT,
    bare_code,
    canonical_id,
    clean_text,
    code_sort_key,
    contexts_dir,
    load_label_reviews,
    text_hash,
)

DUP_SUFFIX = "_dup"


def drafts_dir() -> Path:
    """`data/contexts/drafts/`."""
    return contexts_dir() / "drafts"


def corrections_path() -> Path:
    """`data/curriculum/corrections.json` (written by the curriculum pipeline)."""
    return data_dir() / "curriculum" / "corrections.json"


def structure_path(subject: str = DEFAULT_SUBJECT) -> Path:
    """`data/curriculum/<subject>.structure.json` (the list of indicators, without curriculum text)."""
    return data_dir() / "curriculum" / f"{subject}.structure.json"


def model_id(meta: Mapping[str, Any]) -> str:
    """The model id from a draft file's `drafted_by` ("claude-x (offline draft...)" -> "claude-x")."""
    words = str(meta.get("drafted_by", "")).split()
    return words[0] if words else "unknown"


def load_draft_key_map(path: Path | None = None) -> dict[str, str] | None:
    """`{draft key: canonical code}` from the corrections file, or None when the file does not exist.

    The `draft_keys` map of the file is used when present. Otherwise the map is built from the
    entries of `corrections` that carry a `draft_key` and a `corrected` code.
    """
    path = Path(path) if path else corrections_path()
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, Mapping) and isinstance(raw.get("draft_keys"), Mapping):
        return {str(k): str(v) for k, v in raw["draft_keys"].items()}
    entries = raw.get("corrections", []) if isinstance(raw, Mapping) else raw
    out: dict[str, str] = {}
    for entry in entries:
        key, corrected = entry.get("draft_key"), entry.get("corrected")
        if key and corrected and key != corrected:
            out[str(key)] = str(corrected)
    return out


def load_structure_codes(path: Path | None = None, subject: str = DEFAULT_SUBJECT) -> list[str] | None:
    """The bare codes of the structure file in curriculum order, or None when the file does not exist."""
    path = Path(path) if path else structure_path(subject)
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    records = raw.get("indicators", raw.get("nodes", [])) if isinstance(raw, Mapping) else raw
    return [str(r["code"]) for r in records]


@dataclass
class DraftSet:
    """All drafts, keyed by canonical id, plus what the reader noticed."""

    subject: str = DEFAULT_SUBJECT
    items: dict[str, dict[str, Any]] = field(default_factory=dict)  # id -> {short_label, settings, source, key, file}
    mapping: str = "provisional"  # "corrections" when the corrections file was used
    remapped: dict[str, str] = field(default_factory=dict)  # draft key -> canonical code, where they differ
    problems: list[str] = field(default_factory=list)

    @property
    def n_settings(self) -> int:
        return sum(len(v["settings"]) for v in self.items.values())

    def labels(self) -> dict[str, str]:
        """`{bare canonical code: draft short label}`."""
        return {bare_code(k): str(v["short_label"]) for k, v in self.items.items()}


def load_drafts(
    directory: Path | None = None, corrections: Path | None = None, subject: str = DEFAULT_SUBJECT
) -> DraftSet:
    """Read every draft file of a subject and map its keys to canonical ids."""
    directory = Path(directory) if directory else drafts_dir()
    key_map = load_draft_key_map(corrections)
    out = DraftSet(subject=subject, mapping="corrections" if key_map is not None else "provisional")
    for path in sorted(directory.glob(f"{subject}.*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        source = f"draft:{model_id(doc.get('meta', {}))}"
        for key, value in doc.get("indicators", {}).items():
            code = (key_map or {}).get(key, key)
            if code != key:
                out.remapped[key] = code
            elif key.endswith(DUP_SUFFIX):
                out.problems.append(f"{path.name}: {key} has no canonical code yet (kept under its draft key)")
            indicator = canonical_id(code, subject)
            if indicator in out.items:
                other = out.items[indicator]
                out.problems.append(
                    f"{path.name}: {key} maps to {code}, which {other['file']} already gave as {other['key']} (skipped)"
                )
                continue
            out.items[indicator] = {
                "short_label": clean_text(str(value.get("short_label", ""))),
                "settings": [clean_text(str(s)) for s in value.get("settings", [])],
                "source": source,
                "key": key,
                "file": path.name,
            }
    out.items = {k: out.items[k] for k in sorted(out.items, key=code_sort_key)}
    return out


def current_labels(drafts: DraftSet | None = None, reviews_path: Path | None = None) -> dict[str, dict[str, Any]]:
    """`{bare code: {"short_label", "reviewed", "reviewer", "reviewed_at"}}`: reviewed labels win over drafts."""
    drafts = drafts if drafts is not None else load_drafts()
    out: dict[str, dict[str, Any]] = {
        code: {"short_label": label, "reviewed": False, "reviewer": None, "reviewed_at": None}
        for code, label in drafts.labels().items()
    }
    for code, review in load_label_reviews(reviews_path).items():
        out[code] = {
            "short_label": review["short_label"],
            "reviewed": True,
            "reviewer": review.get("reviewer"),
            "reviewed_at": review.get("reviewed_at"),
        }
    return out


@dataclass
class MergeResult:
    """What a merge did."""

    mapping: str = "provisional"
    indicators: int = 0
    added: int = 0
    kept: int = 0
    removed: int = 0
    kept_reviewed: int = 0
    failing: list[tuple[str, str, list[checks.Failure]]] = field(default_factory=list)  # (id, text, why)
    remapped: dict[str, str] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    not_in_structure: list[str] = field(default_factory=list)
    no_drafts: list[str] = field(default_factory=list)

    def lines(self) -> list[str]:
        """A short report for the command line."""
        out = [
            f"mapping of draft keys: {self.mapping}"
            + ("" if self.mapping == "corrections" else " (no corrections file found: run this again when it exists)"),
            f"indicators with settings: {self.indicators}",
            f"drafts added: {self.added}, already there: {self.kept}, stale drafts removed: {self.removed}, "
            f"reviewed entries kept: {self.kept_reviewed}",
        ]
        if self.remapped:
            out.append(
                "draft keys mapped to canonical codes: " + ", ".join(f"{k} -> {v}" for k, v in self.remapped.items())
            )
        if self.failing:
            out.append(f"NOT merged, {len(self.failing)} drafts fail the automatic checks:")
            out += [f"  {i}: {t[:70]}... [{'; '.join(str(f) for f in why)}]" for i, t, why in self.failing]
        if self.not_in_structure:
            out.append("drafts for codes that are not in the structure file: " + ", ".join(self.not_in_structure))
        if self.no_drafts:
            out.append("indicators in the structure file with no draft: " + ", ".join(self.no_drafts))
        out += [f"problem: {p}" for p in self.problems]
        return out


def _valid_code(code: str) -> bool:
    return re.fullmatch(r"[A-Z]+\d+(\.\d+){4}", code) is not None


def merge_drafts(doc: dict[str, Any], drafts: DraftSet, structure_codes: list[str] | None = None) -> MergeResult:
    """Merge drafts into a settings document in place. Safe to run again and again.

    `doc` is the content of `data/contexts/<subject>.json` (see `kwan.context.settings`).
    """
    result = MergeResult(mapping=drafts.mapping, remapped=dict(drafts.remapped), problems=list(drafts.problems))
    settings: dict[str, list[dict[str, Any]]] = doc.setdefault("settings", {})
    rejected: dict[str, list[dict[str, Any]]] = doc.setdefault("rejected", {})

    for indicator, item in drafts.items.items():
        code = bare_code(indicator)
        if not _valid_code(code):
            result.problems.append(f"{indicator}: not a full indicator code, its drafts were not merged")
            continue
        grade = checks.grade_of(indicator)
        existing = settings.get(indicator, [])
        # hashes a teacher has already dealt with: reviewed as is, edited, or rejected
        handled = {text_hash(e["text"]) for e in existing if e.get("reviewed")}
        handled |= {str(e["edited_from"]) for e in existing if e.get("edited_from")}
        handled |= {text_hash(e["text"]) for e in rejected.get(indicator, [])}
        draft_hashes = {text_hash(t) for t in item["settings"]}
        merged: list[dict[str, Any]] = []
        for e in existing:  # keep the order of the file: reviewed entries and still-current drafts stay in place
            if e.get("reviewed") or e.get("lang", "en") != "en":
                merged.append(e)
                result.kept_reviewed += 1 if e.get("reviewed") else 0
                continue
            h = text_hash(e["text"])
            if h not in draft_hashes or h in handled:
                continue
            siblings = [x["text"] for x in merged if x.get("lang", "en") == "en"]
            failures = checks.check_setting(e["text"], grade, siblings=siblings)
            if failures:
                result.failing.append((indicator, e["text"], failures))
                continue
            merged.append(e)
            result.kept += 1
        seen = {text_hash(e["text"]) for e in merged}
        for text in item["settings"]:
            h = text_hash(text)
            if h in handled or h in seen:
                continue
            siblings = [e["text"] for e in merged if e.get("lang", "en") == "en"]
            failures = checks.check_setting(text, grade, siblings=siblings)
            if failures:
                result.failing.append((indicator, text, failures))
                continue
            merged.append(
                {
                    "text": text,
                    "lang": "en",
                    "reviewed": False,
                    "source": item["source"],
                    "reviewer": None,
                    "reviewed_at": None,
                }
            )
            seen.add(h)
            result.added += 1
        kept_ids = {id(e) for e in merged}
        result.removed += sum(1 for e in existing if id(e) not in kept_ids)
        if merged:
            settings[indicator] = merged
        elif indicator in settings:
            del settings[indicator]

    # unreviewed drafts under ids that the draft files no longer produce (for example after a remap)
    for indicator in [k for k in settings if k not in drafts.items]:
        keep = [e for e in settings[indicator] if e.get("reviewed")]
        result.removed += len(settings[indicator]) - len(keep)
        if keep:
            settings[indicator] = keep
            result.kept_reviewed += len(keep)
            result.problems.append(f"{indicator}: reviewed settings under an id that no draft file maps to (kept)")
        else:
            del settings[indicator]

    result.indicators = len(settings)
    if structure_codes is not None:
        known = set(structure_codes)
        result.not_in_structure = sorted(
            (bare_code(k) for k in settings if bare_code(k) not in known), key=code_sort_key
        )
        have = {bare_code(k) for k in settings}
        result.no_drafts = [c for c in structure_codes if c not in have]
    return result
