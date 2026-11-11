"""Coverage of practice settings, short labels and translations, reported as it truly is.

`contexts_block()` builds the `contexts` block of `data/eval_results.json`. Every number is counted
from the committed files: nothing is marked as reviewed unless a teacher's decision was imported.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kwan.context import checks
from kwan.context.drafts import DraftSet, load_drafts, load_structure_codes
from kwan.context.settings import (
    DEFAULT_SUBJECT,
    LOCAL_LANGS,
    bare_code,
    load_contexts,
    load_label_reviews,
    load_translations,
)
from kwan.context.sheets import today
from kwan.context.translate import translation_coverage


def contexts_block(
    contexts: Path | None = None,
    drafts: DraftSet | None = None,
    label_reviews: Path | None = None,
    translations: Path | None = None,
    structure: Path | None = None,
    subject: str = DEFAULT_SUBJECT,
) -> dict[str, Any]:
    """The `contexts` results block: indicators, drafts, check results, reviewed counts, translations."""
    doc = load_contexts(contexts, subject)
    drafts = drafts if drafts is not None else load_drafts(subject=subject)
    settings = doc["settings"]
    structure_codes = load_structure_codes(structure, subject)
    if structure_codes is not None:
        codes, source = list(structure_codes), "structure file"
    else:
        codes = sorted({bare_code(k) for k in settings} | set(drafts.labels()))
        source = "draft files (the structure file was not there)"
    n = len(codes)
    english = [e for entries in settings.values() for e in entries if e.get("lang", "en") == "en"]
    reviewed = [e for e in english if e.get("reviewed")]
    with_reviewed = {bare_code(k) for k, entries in settings.items() if any(e.get("reviewed") for e in entries)}
    with_any = {bare_code(k) for k, entries in settings.items() if entries}
    report = checks.check_contexts(settings)
    labels = drafts.labels()
    label_fail = sum(1 for label in labels.values() if checks.check_label(label))
    reviews = load_label_reviews(label_reviews)
    block: dict[str, Any] = {
        "subject": subject,
        "generated_on": today(),
        "indicators": n,
        "indicators_counted_from": source,
        "indicators_with_a_setting": len(with_any & set(codes)) if structure_codes is not None else len(with_any),
        "settings": len(english),
        "draft_settings_unreviewed": len(english) - len(reviewed),
        "draft_sources": sorted({str(e.get("source")) for e in english if not e.get("reviewed")}),
        "settings_passing_automatic_checks": report.passed,
        "settings_failing_automatic_checks": report.failed,
        "check_failures": {k: v for k, v in report.by_check.items()},
        "reviewed_settings": len(reviewed),
        "teacher_edited_settings": sum(1 for e in reviewed if e.get("edited_from")),
        "rejected_settings": sum(len(v) for v in doc.get("rejected", {}).values()),
        "indicators_with_reviewed_setting": len(with_reviewed),
        "reviewed_coverage_pct": round(100.0 * len(with_reviewed) / n, 1) if n else 0.0,
        "reviewers": len({str(e.get("reviewer")) for e in reviewed if e.get("reviewer")}),
        "short_labels": len(labels),
        "short_labels_failing_checks": label_fail,
        "reviewed_labels": len(reviews),
        "translations": {},
        "note": "Reviewed means a teacher's decision was imported from a review sheet. "
        "Drafts are not shown by default.",
    }
    for lang in LOCAL_LANGS:
        block["translations"][lang] = translation_coverage(doc, load_translations(lang, translations, subject))
    return block


def coverage_lines(block: dict[str, Any]) -> list[str]:
    """The block as plain lines for the command line."""
    out = [
        f"indicators: {block['indicators']} (from the {block['indicators_counted_from']})",
        f"indicators with at least one setting on file: {block['indicators_with_a_setting']}",
        f"settings on file: {block['settings']} "
        f"({block['draft_settings_unreviewed']} unreviewed drafts, {block['reviewed_settings']} reviewed, "
        f"{block['rejected_settings']} rejected and set aside)",
        f"settings passing the automatic checks: {block['settings_passing_automatic_checks']} of {block['settings']}",
        f"indicators with a reviewed setting: {block['indicators_with_reviewed_setting']} of {block['indicators']} "
        f"({block['reviewed_coverage_pct']}%)",
        f"short labels: {block['short_labels']} drafted, {block['reviewed_labels']} reviewed",
    ]
    for lang, t in block["translations"].items():
        out.append(
            f"translations {lang}: {t['unreviewed_drafts']} unreviewed drafts, {t['reviewed_translations']} reviewed, "
            f"{t['indicators_with_shipping_translation']} of {t['indicators_b1_b3']} Basic 1 to 3 indicators covered "
            f"({t['coverage_pct']}%)"
        )
    return out
