"""Short labels (PRD task 2.7): Kwan-written names for indicators, at most 60 characters.

Labels are drafted by the drafting agents (`data/contexts/drafts/<subject>.*.json`, keyed by the code
as printed) and become reviewed only through `data/contexts/label_reviews.json`, which the context
package writes when a teacher review is imported. This module never marks a label as reviewed itself.
"""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path
from typing import Any

MAX_LABEL = 60


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def placeholder(code: str) -> str:
    """The label used when nobody has written one yet."""
    return f"Skill {code}"


def read_drafts(pattern: Path | str) -> dict[str, str]:
    """`{draft key: short_label}` from every draft file that matches `pattern`."""
    out: dict[str, str] = {}
    for name in sorted(glob.glob(str(pattern))):
        data = json.loads(Path(name).read_text(encoding="utf-8"))
        entries = data.get("indicators", data) if isinstance(data, dict) else {}
        for key, value in entries.items():
            if isinstance(value, dict) and isinstance(value.get("short_label"), str) and value["short_label"].strip():
                out[key] = value["short_label"].strip()
    return out


def read_reviews(path: Path | str) -> dict[str, dict[str, Any]]:
    """`{canonical code or id: {"short_label", "reviewer", "reviewed_at"}}`; empty when the file is absent."""
    path = Path(path)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    data = data.get("labels", data) if isinstance(data, dict) else {}
    return {k: v for k, v in data.items() if isinstance(v, dict) and str(v.get("short_label", "")).strip()}


def resolve_keys(drafts: dict[str, str], records: list[dict[str, Any]], draft_keys: dict[str, str]) -> dict[str, str]:
    """Map draft keys to canonical codes: a canonical code first, then the corrections map, then an alias."""
    canonical = {r["code"] for r in records}
    alias = {a: r["code"] for r in records for a in r.get("aliases", []) if a != r["code"]}
    out: dict[str, str] = {}
    for key, label in drafts.items():
        bare = key.split(":", 1)[-1]
        code = bare if bare in canonical else draft_keys.get(bare) or alias.get(bare)
        if code in canonical:
            out.setdefault(code, label)
    return out


def apply_labels(
    records: list[dict[str, Any]],
    drafts: dict[str, str],
    reviews: dict[str, dict[str, Any]],
    draft_keys: dict[str, str],
) -> dict[str, list[str]]:
    """Set `short_label` and `label_reviewed` on every record.

    A draft label that repeats any indicator statement word for word is not used (committing it would
    commit the statement): the indicator gets the placeholder and is listed under
    `same_as_description` as well as `missing`, so the drafting agents can reword it.

    Returns a report: codes left with a placeholder, codes whose label is longer than `MAX_LABEL`, draft
    keys that match no indicator, codes whose draft repeated the statement, and the codes that carry a
    reviewed label.
    """
    by_code = resolve_keys(drafts, records, draft_keys)
    statements = {_norm(str(r.get("description", ""))) for r in records} - {""}
    canonical = {r["code"] for r in records}
    alias = {a for r in records for a in r.get("aliases", [])}
    report: dict[str, list[str]] = {
        "missing": [],
        "too_long": [],
        "reviewed": [],
        "unmatched_draft_keys": [],
        "same_as_description": [],
    }
    for key in drafts:
        bare = key.split(":", 1)[-1]
        if bare not in canonical and bare not in draft_keys and bare not in alias:
            report["unmatched_draft_keys"].append(key)
    for r in records:
        review = reviews.get(r["code"]) or reviews.get(r["id"])
        if review:
            r["short_label"], r["label_reviewed"] = str(review["short_label"]).strip(), True
            report["reviewed"].append(r["code"])
        elif r["code"] in by_code and _norm(by_code[r["code"]]) not in statements:
            r["short_label"], r["label_reviewed"] = by_code[r["code"]], False
        else:
            if r["code"] in by_code:
                report["same_as_description"].append(r["code"])
            r["short_label"], r["label_reviewed"] = placeholder(r["code"]), False
            report["missing"].append(r["code"])
        if len(r["short_label"]) > MAX_LABEL:
            report["too_long"].append(r["code"])
    return report
