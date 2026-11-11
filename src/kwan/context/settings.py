"""Practice settings per indicator: the data file and the pick that goes with a recommendation (PRD 8.6).

`data/contexts/<subject>.json` holds the settings written for Kwan, one list per indicator:

    {"schema_version": 1, "subject": "math",
     "settings": {"math:B4.1.1.1.2": [{"text": "...", "lang": "en", "reviewed": false,
                                       "source": "draft:<model id>", "reviewer": null, "reviewed_at": null}]},
     "rejected": {"math:B4.1.1.1.2": [{"text": "...", "reviewer": "...", "reviewed_at": "...", "note": "..."}]},
     "rejected_labels": {"B4.1.1.1.2": [{"short_label": "...", "reviewer": "...", "reviewed_at": "..."}]}}

An entry is used at run time only when a teacher has reviewed it (`reviewed: true`). A setting that a
teacher edited also carries `edited_from` (the hash of the draft it replaced) and `region` (where the
reviewer teaches). Rejected drafts are moved out of `settings`, so nothing that reads `settings` can
ship them.

`ContextStore.pick` chooses the setting shown with a recommendation, in this order:

    1. reviewed settings of the indicator in the language asked for;
    2. reviewed English settings of the indicator;
    3. (only with `include_drafts=True`) drafts in the language asked for, then English drafts;
    4. the v1 bank by strand and lower or upper primary (`kwan.context.bank`).

Inside the first group that has anything, the choice is `crc32(bare code) % number of options`, so the
same skill always shows the same setting and the pick does not change between bare and namespaced ids.

Only the standard library is used.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import unicodedata
import zlib
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from kwan._paths import data_dir
from kwan.context import bank as _bank

SCHEMA_VERSION = 1
DEFAULT_SUBJECT = "math"
LOCAL_LANGS = ("tw", "ee", "gaa")  # Twi, Ewe, Ga (Basic 1 to Basic 3)
LANG_NAMES = {"en": "English", "tw": "Twi", "ee": "Ewe", "gaa": "Ga"}
LOCAL_LANG_MAX_GRADE = 3

Entry = dict[str, Any]

# ----------------------------------------------------------------------------- ids and hashes


def bare_code(indicator_id: str) -> str:
    """The code without the subject: `bare_code("math:B4.1.1.1.2") -> "B4.1.1.1.2"`."""
    return indicator_id.rpartition(":")[2]


def canonical_id(indicator_id: str, subject: str = DEFAULT_SUBJECT) -> str:
    """The namespaced id: `canonical_id("B4.1.1.1.2") -> "math:B4.1.1.1.2"`. A namespaced id is unchanged."""
    return indicator_id if ":" in indicator_id else f"{subject}:{indicator_id}"


def code_sort_key(indicator_id: str) -> tuple[Any, ...]:
    """Sort key that puts `B4.1.1.1.2` before `B4.1.1.1.10` and `B4` before `B10`."""
    parts = re.split(r"(\d+)", bare_code(indicator_id))
    return tuple(int(p) if p.isdigit() else p for p in parts)


def clean_text(text: str) -> str:
    """Text as it is stored: Unicode NFC, one line, single spaces, no spaces at the ends."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def text_hash(text: str) -> str:
    """A short stable hash of a text (16 hex digits of SHA-256 of the cleaned text).

    It names a draft on a review sheet and ties a translation to the English text it was made from:
    when the English text changes, its hash changes and the old translation no longer matches.
    """
    return hashlib.sha256(clean_text(text).encode("utf-8")).hexdigest()[:16]


# ----------------------------------------------------------------------------- data files


def contexts_dir() -> Path:
    """`data/contexts/` of this checkout (or under `KWAN_DATA_DIR`)."""
    return data_dir() / "contexts"


def contexts_path(subject: str = DEFAULT_SUBJECT) -> Path:
    """Path of the settings file of a subject, for example `data/contexts/math.json`."""
    return contexts_dir() / f"{subject}.json"


def label_reviews_path() -> Path:
    """Path of `data/contexts/label_reviews.json` (short labels a teacher has reviewed)."""
    return contexts_dir() / "label_reviews.json"


def translations_dir() -> Path:
    """`data/contexts/translations/`."""
    return contexts_dir() / "translations"


def translations_path(lang: str, directory: Path | None = None) -> Path:
    """Path of the translation file of one language, for example `.../translations/tw.json`."""
    return (Path(directory) if directory else translations_dir()) / f"{lang}.json"


def write_json(path: Path, doc: Any) -> None:
    """Write a JSON file in one step (a crash cannot leave half a file)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def empty_contexts(subject: str = DEFAULT_SUBJECT) -> dict[str, Any]:
    """A settings document with nothing in it."""
    return {"schema_version": SCHEMA_VERSION, "subject": subject, "settings": {}, "rejected": {}, "rejected_labels": {}}


def load_contexts(path: Path | None = None, subject: str = DEFAULT_SUBJECT) -> dict[str, Any]:
    """Read a settings file. A missing file gives an empty document."""
    path = Path(path) if path else contexts_path(subject)
    doc = empty_contexts(subject)
    if path.exists():
        doc.update(json.loads(path.read_text(encoding="utf-8")))
    if doc.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: schema_version {doc.get('schema_version')!r} is not supported")
    return doc


def save_contexts(doc: Mapping[str, Any], path: Path | None = None) -> None:
    """Write a settings file with indicators in curriculum order."""
    out = dict(doc)
    for key in ("settings", "rejected", "rejected_labels"):
        block = out.get(key) or {}
        out[key] = {k: block[k] for k in sorted(block, key=code_sort_key) if block[k]}
    write_json(Path(path) if path else contexts_path(str(out.get("subject", DEFAULT_SUBJECT))), out)


def load_label_reviews(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """`{bare canonical code: {"short_label", "reviewer", "reviewed_at", ...}}`; empty when there is no file."""
    path = Path(path) if path else label_reviews_path()
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {bare_code(k): v for k, v in raw.items()}


def save_label_reviews(reviews: Mapping[str, Mapping[str, Any]], path: Path | None = None) -> None:
    """Write the reviewed labels in curriculum order."""
    ordered = {k: dict(reviews[k]) for k in sorted(reviews, key=code_sort_key)}
    write_json(Path(path) if path else label_reviews_path(), ordered)


def empty_translations(lang: str, subject: str = DEFAULT_SUBJECT) -> dict[str, Any]:
    """A translation document with nothing in it."""
    return {"schema_version": SCHEMA_VERSION, "subject": subject, "lang": lang, "translations": {}, "rejected": {}}


def load_translations(lang: str, directory: Path | None = None, subject: str = DEFAULT_SUBJECT) -> dict[str, Any]:
    """Read `translations/<lang>.json`. A missing file gives an empty document.

    {"schema_version": 1, "subject": "math", "lang": "tw",
     "translations": {"math:B1.1.1.1.1": [{"source_hash": "<hash of the English text>", "text": "...",
                      "reviewed": false, "source": "machine:khaya", "reviewer": null, "reviewed_at": null}]}}
    """
    path = translations_path(lang, directory)
    doc = empty_translations(lang, subject)
    if path.exists():
        doc.update(json.loads(path.read_text(encoding="utf-8")))
    if doc.get("lang") != lang:
        raise ValueError(f"{path}: holds language {doc.get('lang')!r}, expected {lang!r}")
    return doc


def save_translations(doc: Mapping[str, Any], directory: Path | None = None) -> None:
    """Write a translation file with indicators in curriculum order."""
    out = dict(doc)
    for key in ("translations", "rejected"):
        block = out.get(key) or {}
        out[key] = {k: block[k] for k in sorted(block, key=code_sort_key) if block[k]}
    write_json(translations_path(str(out["lang"]), directory), out)


def live_translations(
    settings: Mapping[str, Sequence[Mapping[str, Any]]],
    translations: Mapping[str, Sequence[Mapping[str, Any]]],
    lang: str,
) -> dict[str, list[Entry]]:
    """Translations that still match an English setting, as entries `{"text", "lang", "reviewed", ...}`.

    A translation counts as reviewed only when a teacher approved the translation AND the English
    setting it was made from. A translation whose `source_hash` matches no current English text is
    stale and is left out.
    """
    out: dict[str, list[Entry]] = {}
    for indicator, items in translations.items():
        english = {text_hash(str(e["text"])): e for e in settings.get(indicator, ()) if e.get("lang", "en") == "en"}
        for item in items:
            src = english.get(str(item.get("source_hash")))
            if src is None:
                continue
            out.setdefault(indicator, []).append(
                {
                    "text": item["text"],
                    "lang": lang,
                    "reviewed": bool(item.get("reviewed")) and bool(src.get("reviewed")),
                    "source_hash": item.get("source_hash"),
                }
            )
    return out


# ----------------------------------------------------------------------------- the store


def _bank_from(bank: Mapping[Any, Sequence[str]] | None) -> dict[tuple[int, str], list[str]]:
    """The strand x band bank as `{(strand, band): [settings]}` from either key style."""
    if bank is None:
        return {key: list(options) for key, options in _bank.CONTEXTS.items()}
    out: dict[tuple[int, str], list[str]] = {}
    for key, options in bank.items():
        if isinstance(key, str):
            strand, _, band = key.partition(":")
            out[(int(strand), band)] = list(options)
        else:
            out[(int(key[0]), str(key[1]))] = list(options)
    return out


class ContextStore:
    """Practice settings for one subject, with the pick rule of PRD 8.6.

    It implements the `ContextSource` protocol used by the recommender:
    `pick(indicator_id, strand, grade, lang="en", include_drafts=False) -> str`.
    """

    def __init__(
        self,
        settings: Mapping[str, Iterable[Mapping[str, Any]]] | None = None,
        bank: Mapping[Any, Sequence[str]] | None = None,
        subject: str = DEFAULT_SUBJECT,
    ) -> None:
        """`settings` is `{indicator id: [{"text", "lang", "reviewed"}, ...]}` (ids bare or namespaced).

        `bank` is the fallback by strand and band, keyed `"1:lower"` (as in a bundle) or `(1, "lower")`;
        with `None` the v1 bank in `kwan.context.bank` is used.
        """
        self.subject = subject
        self._settings: dict[str, list[Entry]] = {}
        for indicator, entries in (settings or {}).items():
            kept = [dict(e) for e in entries if e.get("text")]
            if kept:
                self._settings.setdefault(canonical_id(indicator, subject), []).extend(kept)
        self._bank = _bank_from(bank)

    # -- construction

    @classmethod
    def from_file(
        cls,
        path: Path | None = None,
        translations: Path | None | bool = None,
        langs: Sequence[str] = LOCAL_LANGS,
        subject: str = DEFAULT_SUBJECT,
    ) -> ContextStore:
        """Build the store from `data/contexts/<subject>.json` and the translation files next to it.

        `translations` is the directory of the translation files; by default the `translations/`
        folder beside the settings file. Pass `False` to leave translations out.
        """
        path = Path(path) if path else contexts_path(subject)
        doc = load_contexts(path, subject)
        settings: dict[str, list[Entry]] = {k: [dict(e) for e in v] for k, v in doc["settings"].items()}
        if translations is not False:
            directory = Path(translations) if isinstance(translations, (str, Path)) else path.parent / "translations"
            for lang in langs:
                if not translations_path(lang, directory).exists():
                    continue
                tdoc = load_translations(lang, directory, subject)
                for indicator, entries in live_translations(doc["settings"], tdoc["translations"], lang).items():
                    settings.setdefault(indicator, []).extend(entries)
        return cls(settings, None, str(doc.get("subject", subject)))

    @classmethod
    def from_bundle(cls, bundle: Mapping[str, Any]) -> ContextStore:
        """Build the store from a bundle's `contexts` and `context_bank` dictionaries.

        A bundle without `context_bank` falls back to the v1 bank in `kwan.context.bank`.
        """
        return cls(
            bundle.get("contexts") or {}, bundle.get("context_bank"), str(bundle.get("subject", DEFAULT_SUBJECT))
        )

    # -- lookup

    def entries(self, indicator_id: str) -> list[Entry]:
        """Every stored entry of an indicator (reviewed or not, all languages). Accepts bare codes."""
        return list(self._settings.get(canonical_id(indicator_id, self.subject), ()))

    def indicators(self) -> list[str]:
        """The ids that have at least one stored entry, in curriculum order."""
        return sorted(self._settings, key=code_sort_key)

    def choose(
        self, indicator_id: str, strand: int, grade: int, lang: str = "en", include_drafts: bool = False
    ) -> dict[str, Any]:
        """The pick with where it came from: `{"text", "tier", "lang", "reviewed"}`.

        `tier` is `"reviewed"`, `"draft"` or `"bank"`. `text` is empty only when the bank has nothing
        for that strand and band.
        """
        entries = self._settings.get(canonical_id(indicator_id, self.subject), ())
        code = bare_code(indicator_id)
        tiers: list[tuple[str, str, bool]] = [("reviewed", lang, True)]
        if lang != "en":
            tiers.append(("reviewed", "en", True))
        if include_drafts:
            tiers.append(("draft", lang, False))
            if lang != "en":
                tiers.append(("draft", "en", False))
        for tier, tier_lang, reviewed in tiers:
            options = [
                str(e["text"])
                for e in entries
                if e.get("lang", "en") == tier_lang and bool(e.get("reviewed")) is reviewed
            ]
            if options:
                text = options[zlib.crc32(code.encode()) % len(options)]
                return {"text": text, "tier": tier, "lang": tier_lang, "reviewed": reviewed}
        options = self._bank.get((int(strand), _bank.band(int(grade))), [])
        text = options[zlib.crc32(code.encode()) % len(options)] if options else ""
        return {"text": text, "tier": "bank", "lang": "en", "reviewed": True}

    def pick(self, indicator_id: str, strand: int, grade: int, lang: str = "en", include_drafts: bool = False) -> str:
        """The practice setting to show for an indicator (see the module docstring for the order)."""
        return str(self.choose(indicator_id, strand, grade, lang, include_drafts)["text"])

    # -- export

    def bundle_contexts(self, include_drafts: bool = False) -> dict[str, list[dict[str, Any]]]:
        """The `contexts` block of a bundle: `{id: [{"text", "lang", "reviewed"}]}`.

        Only reviewed entries by default, which is what a shipped bundle holds.
        """
        out: dict[str, list[dict[str, Any]]] = {}
        for indicator in self.indicators():
            kept = [
                {"text": e["text"], "lang": e.get("lang", "en"), "reviewed": bool(e.get("reviewed"))}
                for e in self._settings[indicator]
                if include_drafts or e.get("reviewed")
            ]
            if kept:
                out[indicator] = kept
        return out

    def bundle_bank(self) -> dict[str, list[str]]:
        """The `context_bank` block of a bundle: `{"1:lower": [...], ...}`."""
        return {f"{strand}:{band}": list(options) for (strand, band), options in sorted(self._bank.items())}
