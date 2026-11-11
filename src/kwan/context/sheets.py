"""Teacher review sheets for practice settings, short labels and translations (PRD 8.4 and 8.5).

Export writes CSV files that open cleanly in Google Sheets, Excel and LibreOffice (UTF-8 with a byte
order mark, one header row). Import reads a filled-in sheet back and applies the decisions.

A settings sheet has one row per draft setting and one row per short label:

    kind              "setting" or "label"
    indicator         the indicator code, for example B4.1.1.1.2
    class             Basic 1 to Basic 6
    short_label       Kwan's short name for the skill
    exemplar_summary  what the curriculum asks for; filled only when the local full-text file exists
    draft_text        the draft setting (or the draft label on a "label" row)
    decision          the reviewer fills in: accept, edit or reject (blank = not reviewed yet)
    edited_text       the corrected text when the decision is edit
    reviewer_id       a short code for the reviewer (not a name, not an email address)
    region_taught     the region where the reviewer teaches
    note              optional
    row_id            names the row for the import; do not change it

A translation sheet has `lang` and `english_text` in place of `exemplar_summary`.

What the import does
    accept   the draft becomes `reviewed: true` with the reviewer id and the date
    edit     the edited text replaces the draft and becomes `reviewed: true`; it must pass the
             automatic checks, otherwise the row is refused with the reason
    reject   the draft is moved out of the live settings and kept in a `rejected` list
    blank    nothing happens; the row stays pending

Importing the same sheet twice changes nothing the second time. A row that disagrees with a decision
already stored is reported as a conflict and left alone, unless the import is run with `override`.

Only the standard library is used.
"""

from __future__ import annotations

import copy
import csv
import datetime as _dt
import io
import json
import re
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kwan._paths import data_dir
from kwan.context import checks
from kwan.context.drafts import DraftSet, load_drafts
from kwan.context.settings import (
    DEFAULT_SUBJECT,
    LANG_NAMES,
    LOCAL_LANGS,
    bare_code,
    canonical_id,
    clean_text,
    code_sort_key,
    contexts_path,
    label_reviews_path,
    load_contexts,
    load_label_reviews,
    load_translations,
    save_contexts,
    save_label_reviews,
    save_translations,
    text_hash,
    translations_dir,
    translations_path,
)

SETTING_COLUMNS = (
    "kind",
    "indicator",
    "class",
    "short_label",
    "exemplar_summary",
    "draft_text",
    "decision",
    "edited_text",
    "reviewer_id",
    "region_taught",
    "note",
    "row_id",
)
TRANSLATION_COLUMNS = (
    "kind",
    "indicator",
    "class",
    "lang",
    "short_label",
    "english_text",
    "draft_text",
    "decision",
    "edited_text",
    "reviewer_id",
    "region_taught",
    "note",
    "row_id",
)
DECISIONS = {
    "accept": "accept",
    "accepted": "accept",
    "edit": "edit",
    "edited": "edit",
    "reject": "reject",
    "rejected": "reject",
}
DEFAULT_BATCH_SIZE = 100
REVIEWER_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-]{1,23}")
SUMMARY_CHARS = 300


class SheetError(ValueError):
    """A review sheet cannot be read."""


def review_dir() -> Path:
    """`build/review/`: where sheets are written by default (git-ignored)."""
    return data_dir().parent / "build" / "review"


def full_text_path(subject: str = DEFAULT_SUBJECT) -> Path:
    """`build/curriculum/<subject>.full.json`: the local file that holds the curriculum text."""
    return data_dir().parent / "build" / "curriculum" / f"{subject}.full.json"


def today() -> str:
    """Today's date (UTC) as `YYYY-MM-DD`."""
    return _dt.datetime.now(_dt.UTC).date().isoformat()


# ----------------------------------------------------------------------------- reading and writing CSV


def write_sheet(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    """Write a CSV that spreadsheet programs open cleanly: UTF-8 with a byte order mark."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore", lineterminator="\r\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def read_sheet(path: Path) -> list[dict[str, str]]:
    """Read a review sheet. Column names are matched without regard to case, spaces or underscores."""
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise SheetError(
            f"{path}: the file is not UTF-8. In Excel choose 'CSV UTF-8'; in Google Sheets use "
            "File > Download > Comma-separated values."
        ) from exc
    lines = text.splitlines()
    if not lines:
        raise SheetError(f"{path}: the file is empty")
    delimiter = max(",;\t", key=lines[0].count)
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter)
    try:
        header = [re.sub(r"[\s\-]+", "_", h.strip().lower()) for h in next(reader)]
    except StopIteration as exc:
        raise SheetError(f"{path}: the file is empty") from exc
    if "decision" not in header:
        raise SheetError(f"{path}: there is no 'decision' column. Keep the first row of the sheet as it was exported.")
    if "row_id" not in header and not {"indicator", "draft_text"} <= set(header):
        raise SheetError(
            f"{path}: the 'row_id' column is missing. Keep the columns of the sheet as they were exported."
        )
    rows = []
    for values in reader:
        if not any(v.strip() for v in values):
            continue
        rows.append({h: (values[i] if i < len(values) else "") for i, h in enumerate(header)})
    return rows


# ----------------------------------------------------------------------------- export


def exemplar_summaries(path: Path | None = None, subject: str = DEFAULT_SUBJECT) -> dict[str, str]:
    """`{bare code: one line on what the curriculum asks}` from the local full-text file, if it exists.

    The text belongs to NaCCA. It goes only into review sheets under `build/`, which are not committed.
    """
    path = Path(path) if path else full_text_path(subject)
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    records = raw.get("indicators", raw.get("nodes", [])) if isinstance(raw, Mapping) else raw
    out: dict[str, str] = {}
    for record in records:
        parts = [clean_text(str(record.get("description") or ""))]
        exemplars = record.get("exemplars") or []
        if exemplars:
            first = exemplars[0]
            parts.append("Example: " + clean_text(str(first.get("text", "") if isinstance(first, Mapping) else first)))
        summary = " ".join(p for p in parts if p)
        if len(summary) > SUMMARY_CHARS:
            summary = summary[: SUMMARY_CHARS - 3].rstrip() + "..."
        if summary:
            out[str(record["code"])] = summary
    return out


def _spread(indicators: Sequence[str]) -> list[str]:
    """Indicators in a fixed, mixed order: classes take turns, and inside a class the order is by crc32.

    A first batch drawn from this order is a fair sample of all six classes and all strands.
    """
    by_grade: dict[int, list[str]] = {}
    for indicator in indicators:
        by_grade.setdefault(checks.grade_of(indicator), []).append(indicator)
    for items in by_grade.values():
        items.sort(key=lambda i: (zlib.crc32(bare_code(i).encode()), i))
    out: list[str] = []
    queues = [by_grade[g] for g in sorted(by_grade)]
    while any(queues):
        for q in queues:
            if q:
                out.append(q.pop(0))
    return out


def setting_row_id(indicator: str, text: str) -> str:
    """Row id of a draft setting."""
    return f"s|{indicator}|{text_hash(text)}"


def label_row_id(indicator: str, label: str) -> str:
    """Row id of a draft label."""
    return f"l|{indicator}|{text_hash(label)}"


def translation_row_id(lang: str, indicator: str, source_hash: str, text: str) -> str:
    """Row id of a draft translation."""
    return f"t|{lang}|{indicator}|{source_hash}|{text_hash(text)}"


def pending_review_groups(
    doc: Mapping[str, Any],
    drafts: DraftSet,
    label_reviews: Mapping[str, Any],
    summaries: Mapping[str, str] | None = None,
    kinds: Sequence[str] = ("label", "setting"),
    grades: Sequence[int] | None = None,
) -> list[list[dict[str, str]]]:
    """Rows that still need a decision, grouped by indicator (the label row first, then its settings)."""
    summaries = summaries or {}
    subject = str(doc.get("subject", DEFAULT_SUBJECT))
    draft_labels = drafts.labels()
    rejected_labels = doc.get("rejected_labels", {})
    indicators = set(doc.get("settings", {})) | {canonical_id(c, subject) for c in draft_labels}
    groups = []
    for indicator in _spread(sorted(indicators, key=code_sort_key)):
        code = bare_code(indicator)
        grade = checks.grade_of(indicator)
        if grades and grade not in grades:
            continue
        reviewed_label = label_reviews.get(code)
        label = str(reviewed_label["short_label"]) if reviewed_label else draft_labels.get(code, "")
        base = {
            "indicator": code,
            "class": f"Basic {grade}",
            "short_label": label,
            "exemplar_summary": summaries.get(code, ""),
        }
        rows: list[dict[str, str]] = []
        if "label" in kinds and not reviewed_label and code in draft_labels:
            rejected = {text_hash(str(r["short_label"])) for r in rejected_labels.get(code, [])}
            if text_hash(label) not in rejected:
                rows.append({**base, "kind": "label", "draft_text": label, "row_id": label_row_id(indicator, label)})
        if "setting" in kinds:
            for entry in doc.get("settings", {}).get(indicator, []):
                if entry.get("reviewed") or entry.get("lang", "en") != "en":
                    continue
                text = str(entry["text"])
                rows.append({**base, "kind": "setting", "draft_text": text, "row_id": setting_row_id(indicator, text)})
        if rows:
            groups.append(rows)
    return groups


def pending_translation_groups(
    doc: Mapping[str, Any],
    tdoc: Mapping[str, Any],
    labels: Mapping[str, str],
    include_drafts: bool = False,
) -> list[list[dict[str, str]]]:
    """Machine-drafted translations that still need a decision, grouped by indicator.

    Only translations of reviewed English settings are listed, unless `include_drafts` is set.
    """
    lang = str(tdoc["lang"])
    groups = []
    for indicator in sorted(tdoc.get("translations", {}), key=code_sort_key):
        english = {text_hash(str(e["text"])): e for e in doc.get("settings", {}).get(indicator, [])}
        code = bare_code(indicator)
        rows = []
        for item in tdoc["translations"][indicator]:
            src = english.get(str(item.get("source_hash")))
            if item.get("reviewed") or src is None or not (include_drafts or src.get("reviewed")):
                continue
            rows.append(
                {
                    "kind": "translation",
                    "indicator": code,
                    "class": f"Basic {checks.grade_of(indicator)}",
                    "lang": lang,
                    "short_label": labels.get(code, ""),
                    "english_text": str(src["text"]),
                    "draft_text": str(item["text"]),
                    "row_id": translation_row_id(lang, indicator, str(item["source_hash"]), str(item["text"])),
                }
            )
        if rows:
            groups.append(rows)
    return groups


def make_batches(
    groups: Sequence[list[dict[str, str]]], batch_size: int = DEFAULT_BATCH_SIZE
) -> list[list[dict[str, str]]]:
    """Cut the groups into sheets of about `batch_size` settings; an indicator is never split."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    batches: list[list[dict[str, str]]] = []
    current: list[dict[str, str]] = []
    count = 0
    for rows in groups:
        current.extend(rows)
        count += sum(1 for r in rows if r["kind"] != "label")
        if count >= batch_size or len(current) >= 2 * batch_size:
            batches.append(current)
            current, count = [], 0
    if current:
        batches.append(current)
    return batches


@dataclass
class ExportResult:
    """What an export wrote."""

    files: list[tuple[Path, int, int]] = field(default_factory=list)  # (path, setting or translation rows, label rows)
    batches_total: int = 0
    pending_rows: int = 0
    pending_labels: int = 0
    with_exemplars: bool = False

    def lines(self) -> list[str]:
        """A short report for the command line."""
        out = [
            f"waiting for review: {self.pending_rows} rows and {self.pending_labels} short labels, "
            f"in {self.batches_total} sheets"
        ]
        out += [f"wrote {path} ({n} rows, {labels} labels)" for path, n, labels in self.files]
        if not self.files:
            out.append("nothing to export")
        if self.with_exemplars:
            out.append(
                "These sheets hold NaCCA curriculum text. Use them for review only; do not commit or publish them."
            )
        return out


def export_review(
    out_dir: Path | None = None,
    batch: int = 1,
    batch_size: int = DEFAULT_BATCH_SIZE,
    all_batches: bool = False,
    lang: str = "en",
    grades: Sequence[int] | None = None,
    kinds: Sequence[str] = ("label", "setting"),
    include_drafts: bool = False,
    exemplars: bool = True,
    contexts: Path | None = None,
    drafts: DraftSet | None = None,
    label_reviews: Path | None = None,
    translations: Path | None = None,
    full_text: Path | None = None,
    subject: str = DEFAULT_SUBJECT,
) -> ExportResult:
    """Write review sheets for what is still pending.

    With `lang="en"` the sheets hold draft settings and short labels. With `lang` set to `tw`, `ee`
    or `gaa` they hold machine-drafted translations. `batch` is 1-based and counts over what is
    pending now, so after an import batch 1 is the next sheet. `all_batches` writes every sheet.
    """
    out_dir = Path(out_dir) if out_dir else review_dir()
    doc = load_contexts(contexts, subject)
    drafts = drafts if drafts is not None else load_drafts(subject=subject)
    reviews = load_label_reviews(label_reviews)
    result = ExportResult()
    if lang == "en":
        summaries = exemplar_summaries(full_text, subject) if exemplars else {}
        result.with_exemplars = bool(summaries)
        groups = pending_review_groups(doc, drafts, reviews, summaries, kinds, grades)
        columns: Sequence[str] = SETTING_COLUMNS
        stem = f"{subject}.review"
    else:
        tdoc = load_translations(lang, translations, subject)
        labels = {**drafts.labels(), **{c: str(r["short_label"]) for c, r in reviews.items()}}
        groups = pending_translation_groups(doc, tdoc, labels, include_drafts)
        if grades:
            groups = [g for g in groups if int(g[0]["class"].split()[-1]) in grades]
        columns = TRANSLATION_COLUMNS
        stem = f"{subject}.translations.{lang}"
    batches = make_batches(groups, batch_size)
    result.batches_total = len(batches)
    result.pending_rows = sum(1 for g in groups for r in g if r["kind"] != "label")
    result.pending_labels = sum(1 for g in groups for r in g if r["kind"] == "label")
    wanted = range(1, len(batches) + 1) if all_batches else [batch]
    for number in wanted:
        if not 1 <= number <= len(batches):
            continue
        rows = batches[number - 1]
        path = out_dir / f"{stem}.batch-{number:02d}-of-{len(batches):02d}.csv"
        write_sheet(path, columns, rows)
        n_labels = sum(1 for r in rows if r["kind"] == "label")
        result.files.append((path, len(rows) - n_labels, n_labels))
    return result


# ----------------------------------------------------------------------------- import


@dataclass
class ImportResult:
    """What an import did, row by row."""

    applied: dict[str, dict[str, int]] = field(default_factory=dict)  # kind -> decision -> rows
    unchanged: int = 0
    pending: int = 0
    errors: list[tuple[int, str, str]] = field(default_factory=list)  # (sheet line, indicator, message)
    conflicts: list[tuple[int, str, str]] = field(default_factory=list)
    dry_run: bool = False

    @property
    def ok(self) -> bool:
        return not self.errors and not self.conflicts

    def count(self, kind: str, decision: str) -> int:
        """Rows of a kind ("setting", "label", "translation") that got a decision applied."""
        return self.applied.get(kind, {}).get(decision, 0)

    def _add(self, kind: str, decision: str) -> None:
        self.applied.setdefault(kind, {}).setdefault(decision, 0)
        self.applied[kind][decision] += 1

    def lines(self) -> list[str]:
        """A short report for the command line."""
        out = []
        for kind in ("setting", "label", "translation"):
            if kind in self.applied:
                d = self.applied[kind]
                out.append(
                    f"{kind}s: {d.get('accept', 0)} accepted, {d.get('edit', 0)} edited, {d.get('reject', 0)} rejected"
                )
        out.append(f"already applied (no change): {self.unchanged}; left blank (still pending): {self.pending}")
        out += [f"REFUSED line {line} ({indicator}): {msg}" for line, indicator, msg in self.errors]
        out += [f"CONFLICT line {line} ({indicator}): {msg}" for line, indicator, msg in self.conflicts]
        if self.dry_run:
            out.append("dry run: nothing was written")
        return out


class _Refused(Exception):
    """This row cannot be applied; the message says why."""


class _Conflict(Exception):
    """This row disagrees with a decision that is already stored."""


def _fail_text(failures: Sequence[checks.Failure]) -> str:
    return "; ".join(str(f) for f in failures)


def _review_fields(reviewer: str, date: str, region: str) -> dict[str, Any]:
    out: dict[str, Any] = {"reviewer": reviewer, "reviewed_at": date}
    if region:
        out["region"] = region
    return out


def _apply_setting(
    doc: dict[str, Any], indicator: str, h: str, row: Mapping[str, str], decision: str, date: str, override: bool
) -> str:
    """Apply one decision on an English setting. Returns "applied" or "unchanged"."""
    settings = doc.setdefault("settings", {})
    rejected = doc.setdefault("rejected", {})
    entries: list[dict[str, Any]] = settings.setdefault(indicator, [])
    gone: list[dict[str, Any]] = rejected.setdefault(indicator, [])
    reviewer, region = row["reviewer_id"], row.get("region_taught", "")
    edited = clean_text(row.get("edited_text", ""))
    if decision == "edit" and edited and text_hash(edited) == h:
        decision = "accept"  # the "edit" changed nothing

    as_edit = next((e for e in entries if e.get("edited_from") == h), None)
    as_reject = next((e for e in gone if text_hash(e["text"]) == h), None)
    as_draft = next((e for e in entries if not e.get("edited_from") and text_hash(e["text"]) == h), None)
    if as_edit is not None:
        state = "edit"
    elif as_reject is not None:
        state = "reject"
    elif as_draft is not None:
        state = "accept" if as_draft.get("reviewed") else "pending"
    else:
        raise _Refused("this draft is no longer in the settings file (it may have been rewritten); export a new sheet")

    if state != "pending":
        same = state == decision and (decision != "edit" or (as_edit is not None and as_edit["text"] == edited))
        if same:
            return "unchanged"
        if not override:
            who = (as_edit or as_reject or as_draft or {}).get("reviewer")
            raise _Conflict(f"already {state}ed by {who}; this row says {decision}. Nothing changed.")
        original = clean_text(row.get("draft_text", ""))
        if text_hash(original) != h:
            raise _Refused("cannot override: the draft_text column no longer holds the original draft")
        fresh = {
            "text": original,
            "lang": "en",
            "reviewed": False,
            "source": "draft",
            "reviewer": None,
            "reviewed_at": None,
        }
        if as_edit is not None:
            fresh["source"] = as_edit.get("edited_from_source", "draft")
            entries[entries.index(as_edit)] = fresh
        elif as_reject is not None:
            fresh["source"] = as_reject.get("source", "draft")
            gone.remove(as_reject)
            entries.append(fresh)
        else:
            assert as_draft is not None
            as_draft.update(reviewed=False, reviewer=None, reviewed_at=None)
            as_draft.pop("region", None)
            fresh = as_draft
        as_draft = fresh

    assert as_draft is not None
    grade = checks.grade_of(indicator)
    siblings = [e["text"] for e in entries if e is not as_draft and e.get("lang", "en") == "en"]
    if decision == "accept":
        failures = checks.check_setting(as_draft["text"], grade, siblings)
        if failures:
            raise _Refused(f"cannot accept, the draft fails the automatic checks: {_fail_text(failures)}")
        as_draft.update(reviewed=True, **_review_fields(reviewer, date, region))
    elif decision == "edit":
        if not edited:
            raise _Refused("the decision is edit but the edited_text column is empty")
        failures = checks.check_setting(edited, grade, siblings)
        if failures:
            raise _Refused(f"the edited text fails the automatic checks: {_fail_text(failures)}")
        entries[entries.index(as_draft)] = {
            "text": edited,
            "lang": "en",
            "reviewed": True,
            "source": "teacher-edit",
            **_review_fields(reviewer, date, region),
            "edited_from": h,
            "edited_from_source": as_draft.get("source"),
        }
    else:
        entries.remove(as_draft)
        record = {"text": as_draft["text"], "source": as_draft.get("source"), **_review_fields(reviewer, date, region)}
        if row.get("note", "").strip():
            record["note"] = clean_text(row["note"])
        gone.append(record)
    return "applied"


def _apply_label(
    doc: dict[str, Any],
    reviews: dict[str, dict[str, Any]],
    draft_labels: Mapping[str, str],
    indicator: str,
    h: str,
    row: Mapping[str, str],
    decision: str,
    date: str,
    override: bool,
) -> str:
    """Apply one decision on a short label. Returns "applied" or "unchanged"."""
    code = bare_code(indicator)
    gone: list[dict[str, Any]] = doc.setdefault("rejected_labels", {}).setdefault(code, [])
    reviewer, region = row["reviewer_id"], row.get("region_taught", "")
    edited = clean_text(row.get("edited_text", ""))
    if decision == "edit" and edited and text_hash(edited) == h:
        decision = "accept"
    review = reviews.get(code)
    as_reject = next((r for r in gone if text_hash(str(r["short_label"])) == h), None)
    draft = draft_labels.get(code)
    if review is not None and review.get("draft_hash") == h:
        state = "accept" if text_hash(str(review["short_label"])) == h else "edit"
    elif review is not None:
        raise _Conflict(
            f"this label was already reviewed by {review.get('reviewer')} from another draft. Nothing changed."
        )
    elif as_reject is not None:
        state = "reject"
    elif draft is not None and text_hash(draft) == h:
        state = "pending"
    else:
        raise _Refused("this draft label is no longer current (it may have been rewritten); export a new sheet")

    if state != "pending":
        same = state == decision and (decision != "edit" or (review is not None and review["short_label"] == edited))
        if same:
            return "unchanged"
        if not override:
            who = (review or as_reject or {}).get("reviewer")
            raise _Conflict(f"already {state}ed by {who}; this row says {decision}. Nothing changed.")
        if draft is None or text_hash(draft) != h:
            raise _Refused("cannot override: the draft label is no longer current")
        reviews.pop(code, None)
        if as_reject is not None:
            gone.remove(as_reject)
    assert draft is not None
    if decision == "reject":
        record = {"short_label": draft, **_review_fields(reviewer, date, region)}
        if row.get("note", "").strip():
            record["note"] = clean_text(row["note"])
        gone.append(record)
        return "applied"
    label = draft if decision == "accept" else edited
    if not label:
        raise _Refused("the decision is edit but the edited_text column is empty")
    failures = checks.check_label(label)
    if failures:
        raise _Refused(f"the label fails the automatic checks: {_fail_text(failures)}")
    reviews[code] = {"short_label": label, **_review_fields(reviewer, date, region), "draft_hash": h}
    return "applied"


def _apply_translation(
    doc: Mapping[str, Any],
    tdoc: dict[str, Any],
    indicator: str,
    source_hash: str,
    h: str,
    row: Mapping[str, str],
    decision: str,
    date: str,
    override: bool,
) -> str:
    """Apply one decision on a translation. Returns "applied" or "unchanged"."""
    entries: list[dict[str, Any]] = tdoc.setdefault("translations", {}).setdefault(indicator, [])
    gone: list[dict[str, Any]] = tdoc.setdefault("rejected", {}).setdefault(indicator, [])
    reviewer, region = row["reviewer_id"], row.get("region_taught", "")
    edited = clean_text(row.get("edited_text", ""))
    if decision == "edit" and edited and text_hash(edited) == h:
        decision = "accept"
    english = next(
        (e for e in doc.get("settings", {}).get(indicator, []) if text_hash(str(e["text"])) == source_hash), None
    )
    if english is None:
        raise _Refused("the English setting has changed since this sheet was exported; the translation must be redone")

    def mine(e: Mapping[str, Any]) -> bool:
        return e.get("source_hash") == source_hash

    as_edit = next((e for e in entries if mine(e) and e.get("edited_from") == h), None)
    as_reject = next((e for e in gone if mine(e) and text_hash(e["text"]) == h), None)
    as_draft = next((e for e in entries if mine(e) and not e.get("edited_from") and text_hash(e["text"]) == h), None)
    if as_edit is not None:
        state = "edit"
    elif as_reject is not None:
        state = "reject"
    elif as_draft is not None:
        state = "accept" if as_draft.get("reviewed") else "pending"
    else:
        raise _Refused("this translation draft is no longer in the translation file; export a new sheet")

    if state != "pending":
        same = state == decision and (decision != "edit" or (as_edit is not None and as_edit["text"] == edited))
        if same:
            return "unchanged"
        if not override:
            who = (as_edit or as_reject or as_draft or {}).get("reviewer")
            raise _Conflict(f"already {state}ed by {who}; this row says {decision}. Nothing changed.")
        original = clean_text(row.get("draft_text", ""))
        if text_hash(original) != h:
            raise _Refused("cannot override: the draft_text column no longer holds the original draft")
        fresh = {
            "source_hash": source_hash,
            "text": original,
            "reviewed": False,
            "source": "machine",
            "reviewer": None,
            "reviewed_at": None,
        }
        if as_edit is not None:
            fresh["source"] = as_edit.get("edited_from_source", "machine")
            entries[entries.index(as_edit)] = fresh
        elif as_reject is not None:
            fresh["source"] = as_reject.get("source", "machine")
            gone.remove(as_reject)
            entries.append(fresh)
        else:
            assert as_draft is not None
            as_draft.update(reviewed=False, reviewer=None, reviewed_at=None)
            as_draft.pop("region", None)
            fresh = as_draft
        as_draft = fresh

    assert as_draft is not None
    if decision == "reject":
        entries.remove(as_draft)
        record = {
            "source_hash": source_hash,
            "text": as_draft["text"],
            "source": as_draft.get("source"),
            **_review_fields(reviewer, date, region),
        }
        if row.get("note", "").strip():
            record["note"] = clean_text(row["note"])
        gone.append(record)
        return "applied"
    text = as_draft["text"] if decision == "accept" else edited
    if not text:
        raise _Refused("the decision is edit but the edited_text column is empty")
    failures = checks.check_translation(text, str(english["text"]))
    if failures:
        raise _Refused(f"the translation fails the automatic checks: {_fail_text(failures)}")
    if decision == "accept":
        as_draft.update(reviewed=True, **_review_fields(reviewer, date, region))
    else:
        entries[entries.index(as_draft)] = {
            "source_hash": source_hash,
            "text": edited,
            "reviewed": True,
            "source": "teacher-edit",
            **_review_fields(reviewer, date, region),
            "edited_from": h,
            "edited_from_source": as_draft.get("source"),
        }
    return "applied"


def _row_identity(row: Mapping[str, str], subject: str) -> tuple[str, ...]:
    """`(kind, ...)` parsed from `row_id`, or rebuilt from the other columns when `row_id` is gone."""
    row_id = row.get("row_id", "").strip()
    if row_id:
        parts = tuple(row_id.split("|"))
        if (parts[0] in ("s", "l") and len(parts) == 3) or (parts[0] == "t" and len(parts) == 5):
            return parts
        raise _Refused(f"the row_id {row_id!r} was changed; restore it from the exported sheet")
    kind = row.get("kind", "").strip().lower()
    indicator = canonical_id(row.get("indicator", "").strip(), subject)
    draft = clean_text(row.get("draft_text", ""))
    if not row.get("indicator", "").strip() or not draft:
        raise _Refused("the row has no row_id and no indicator and draft_text to find it by")
    if kind == "label":
        return ("l", indicator, text_hash(draft))
    if kind == "translation":
        english = clean_text(row.get("english_text", ""))
        return ("t", row.get("lang", "").strip(), indicator, text_hash(english), text_hash(draft))
    return ("s", indicator, text_hash(draft))


def import_review(
    path: Path,
    date: str | None = None,
    override: bool = False,
    dry_run: bool = False,
    contexts: Path | None = None,
    drafts: DraftSet | None = None,
    label_reviews: Path | None = None,
    translations: Path | None = None,
    subject: str = DEFAULT_SUBJECT,
) -> ImportResult:
    """Apply the decisions of a filled-in review sheet. Importing the same sheet again changes nothing.

    `date` is stored as `reviewed_at` (default: today, UTC). Rows that cannot be applied are listed in
    the result and every other row is still applied. With `dry_run` nothing is written.
    """
    rows = read_sheet(Path(path))
    date = date or today()
    contexts_file = Path(contexts) if contexts else contexts_path(subject)
    reviews_file = Path(label_reviews) if label_reviews else label_reviews_path()
    tdir = Path(translations) if translations else translations_dir()
    doc = load_contexts(contexts_file, subject)
    reviews = load_label_reviews(reviews_file)
    before = (copy.deepcopy(doc), copy.deepcopy(reviews))
    draft_labels: Mapping[str, str] | None = None
    tdocs: dict[str, dict[str, Any]] = {}
    tdocs_before: dict[str, dict[str, Any]] = {}
    result = ImportResult(dry_run=dry_run)

    for line, row in enumerate(rows, start=2):
        indicator_shown = row.get("indicator", "").strip() or "?"
        raw_decision = row.get("decision", "").strip().lower()
        if not raw_decision:
            result.pending += 1
            continue
        try:
            decision = DECISIONS.get(raw_decision)
            if decision is None:
                raise _Refused(f"unknown decision {row.get('decision', '').strip()!r}; write accept, edit or reject")
            reviewer = row.get("reviewer_id", "").strip()
            if not REVIEWER_ID.fullmatch(reviewer):
                raise _Refused(
                    "reviewer_id must be a short code of letters and digits (2 to 24 characters), "
                    "not a name or an email address"
                )
            clean_row = {**row, "reviewer_id": reviewer, "region_taught": clean_text(row.get("region_taught", ""))}
            identity = _row_identity(row, subject)
            if identity[0] == "s":
                kind = "setting"
                status = _apply_setting(doc, identity[1], identity[2], clean_row, decision, date, override)
            elif identity[0] == "l":
                kind = "label"
                if draft_labels is None:
                    draft_labels = (drafts if drafts is not None else load_drafts(subject=subject)).labels()
                status = _apply_label(
                    doc, reviews, draft_labels, identity[1], identity[2], clean_row, decision, date, override
                )
            else:
                kind = "translation"
                lang = identity[1]
                if lang not in LOCAL_LANGS:
                    raise _Refused(
                        f"unknown language {lang!r}; Kwan has {', '.join(LANG_NAMES[x] for x in LOCAL_LANGS)}"
                    )
                if lang not in tdocs:
                    tdocs[lang] = load_translations(lang, tdir, subject)
                    tdocs_before[lang] = copy.deepcopy(tdocs[lang])
                status = _apply_translation(
                    doc, tdocs[lang], identity[2], identity[3], identity[4], clean_row, decision, date, override
                )
        except _Refused as exc:
            result.errors.append((line, indicator_shown, str(exc)))
            continue
        except _Conflict as exc:
            result.conflicts.append((line, indicator_shown, str(exc)))
            continue
        if status == "unchanged":
            result.unchanged += 1
        else:
            result._add(kind, decision)

    if not dry_run:
        for key in ("settings", "rejected", "rejected_labels"):
            doc[key] = {k: v for k, v in doc.get(key, {}).items() if v}
        if _strip_empty(doc) != _strip_empty(before[0]):
            save_contexts(doc, contexts_file)
        if reviews != before[1]:
            save_label_reviews(reviews, reviews_file)
        for lang, tdoc in tdocs.items():
            if _strip_empty(tdoc) != _strip_empty(tdocs_before[lang]):
                save_translations(tdoc, tdir)
    return result


def _strip_empty(doc: Mapping[str, Any]) -> dict[str, Any]:
    """A document without its empty lists, so that an import that changed nothing rewrites nothing."""
    out = dict(doc)
    for key in ("settings", "rejected", "rejected_labels", "translations"):
        if isinstance(out.get(key), Mapping):
            out[key] = {k: v for k, v in out[key].items() if v}
    return out


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "SETTING_COLUMNS",
    "TRANSLATION_COLUMNS",
    "ExportResult",
    "ImportResult",
    "SheetError",
    "exemplar_summaries",
    "export_review",
    "import_review",
    "make_batches",
    "pending_review_groups",
    "pending_translation_groups",
    "read_sheet",
    "review_dir",
    "translations_path",
    "write_sheet",
]
