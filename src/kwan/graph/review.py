"""Teacher review of proposed edges (PRD Phase 7, task 7.6).

`export_review` writes a sheet a teacher can fill in: one row per proposed edge with both skills,
the model probability, and empty `decision` and `reviewer` columns. `import_review` reads the filled
sheet back and records each decision (accepted or rejected, who, when) in the proposed edge file.

The sheet goes under `build/` by default, because it holds the curriculum descriptions when they
are on this machine, and those must not be committed. Only accepted edges enter the shipped graph.
"""

from __future__ import annotations

import csv
import os
import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx

from kwan import ids
from kwan._paths import data_dir
from kwan.graph.augment import (
    STATUSES,
    _status,
    build_dir,
    load_texts,
    node_texts,
    review_counts,
    rule_graph,
    today,
)

COLUMNS = (
    "a_id",
    "b_id",
    "a_grade",
    "b_grade",
    "a_label",
    "b_label",
    "a_description",
    "b_description",
    "probability",
    "decision",
    "reviewer",
    "note",
)
QUESTION_FOR_REVIEWER = "Must a pupil be able to do A before they can learn B? Write accepted or rejected."
_ACCEPT = {"accepted", "accept", "yes", "y"}
_REJECT = {"rejected", "reject", "no", "n"}

__all__ = [
    "COLUMNS",
    "ImportReport",
    "ReviewError",
    "acceptance",
    "default_sheet_path",
    "export_review",
    "import_review",
]


class ReviewError(ValueError):
    """A review sheet that cannot be applied. `problems` lists every row that is wrong."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("review sheet not applied:\n" + "\n".join(f"  {p}" for p in problems))
        self.problems = problems


def default_sheet_path(subject: str = ids.DEFAULT_SUBJECT) -> Path:
    """`build/review/<subject>.augmented.review.csv` (git-ignored)."""
    return build_dir() / "review" / f"{subject}.augmented.review.csv"


def _inside(path: Path, folder: Path) -> bool:
    try:
        path.resolve().relative_to(folder.resolve())
    except ValueError:
        return False
    return True


def export_review(
    G: nx.DiGraph,
    doc: Mapping[str, Any],
    out: str | os.PathLike[str] | None = None,
    texts: Mapping[str, Mapping[str, str]] | None = None,
    status: str = "pending",
    sample: int | None = None,
    seed: int = 0,
) -> tuple[Path, int, bool]:
    """Write the review sheet. Returns `(path, rows, has_descriptions)`.

    `texts` is the output of `kwan.graph.augment.load_texts`; with None it is read from the default
    full-text file. The description columns are filled only when that text is on this machine;
    otherwise they are empty and the sheet shows the short labels. `status` picks which edges to
    list (`pending`, `accepted`, `rejected` or `all`). `sample` draws that many edges at random
    (fixed by `seed`), for the 100-edge acceptance sample. A sheet with descriptions is refused
    inside the committed data directory.
    """
    if status != "all" and status not in STATUSES:
        raise ValueError(f"unknown status {status!r}: choose from all, {', '.join(STATUSES)}")
    subject = doc.get("subject") or G.graph.get("subject") or ids.DEFAULT_SUBJECT
    path = Path(out) if out is not None else default_sheet_path(subject)
    if texts is None:
        texts = load_texts(subject=subject)
    has_text = bool(texts)
    if has_text and _inside(path, data_dir()):
        raise ValueError(f"{path} is inside the committed data directory; a sheet with descriptions goes under build/")
    info = node_texts(G, texts)
    resolver = ids.Resolver(G)
    edges = [e for e in doc.get("edges") or [] if status == "all" or _status(e) == status]
    edges.sort(key=lambda e: (-float(e.get("weight", 0.0)), e["source"], e["target"]))
    if sample is not None and sample < len(edges):
        edges = random.Random(seed).sample(edges, sample)
        edges.sort(key=lambda e: (-float(e.get("weight", 0.0)), e["source"], e["target"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for e in edges:
            a, b = resolver.get(e["source"]), resolver.get(e["target"])
            if a is None or b is None:
                continue  # an edge whose indicator left the curriculum; `kwan graph augment --dry-run` reports it
            review = (e.get("provenance") or {}).get("review") or {}
            decided = _status(e) != "pending"
            writer.writerow(
                {
                    "a_id": e["source"],
                    "b_id": e["target"],
                    "a_grade": G.nodes[a].get("grade", ""),
                    "b_grade": G.nodes[b].get("grade", ""),
                    "a_label": G.nodes[a].get("short_label", ""),
                    "b_label": G.nodes[b].get("short_label", ""),
                    "a_description": info[a]["text"] if has_text else "",
                    "b_description": info[b]["text"] if has_text else "",
                    "probability": f"{float(e.get('weight', 0.0)):.4f}",
                    "decision": _status(e) if decided else "",
                    "reviewer": (review.get("by") or "") if decided else "",
                    "note": "",
                }
            )
            rows += 1
    return path, rows, has_text


@dataclass
class ImportReport:
    """What `import_review` did."""

    accepted: int = 0
    rejected: int = 0
    blank: int = 0
    unchanged: int = 0
    refused: list[dict[str, str]] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)


def _decision(raw: str) -> str | None:
    text = (raw or "").strip().lower()
    if not text:
        return None
    if text in _ACCEPT:
        return "accepted"
    if text in _REJECT:
        return "rejected"
    raise ValueError(f"decision {raw!r} is not 'accepted' or 'rejected'")


def import_review(
    G: nx.DiGraph,
    doc: Mapping[str, Any],
    sheet: str | os.PathLike[str],
    reviewer: str | None = None,
    at: str | None = None,
) -> tuple[dict[str, Any], ImportReport]:
    """Apply a filled review sheet. Returns the updated document and a report; `doc` is not changed.

    Every row is checked before anything is applied: both ids must be indicators of the graph, the
    edge must be one of the proposed edges, the decision must be `accepted`, `rejected` or blank,
    and a row with a decision needs a reviewer id (its own `reviewer` cell, or the `reviewer`
    argument). Any problem raises `ReviewError` listing all of them, and nothing is applied.

    Rows with a blank decision are left alone. An accepted edge that would close a cycle with the
    rule edges and the edges already accepted is refused: it stays as it was and is listed in
    `report.refused`. Accepted edges are tried from the highest probability down.
    """
    at = at or today()
    resolver = ids.Resolver(G)
    edges = [dict(e) for e in doc.get("edges") or []]
    by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for e in edges:
        a, b = resolver.get(str(e.get("source"))), resolver.get(str(e.get("target")))
        if a is not None and b is not None:
            by_pair[(a, b)] = e
    problems: list[str] = []
    decisions: dict[tuple[str, str], tuple[str, str]] = {}
    report = ImportReport()
    with open(sheet, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in ("a_id", "b_id", "decision") if c not in (reader.fieldnames or [])]
        if missing:
            raise ReviewError([f"the sheet has no column {c!r}" for c in missing])
        for line, row in enumerate(reader, start=2):
            raw_a, raw_b = (row.get("a_id") or "").strip(), (row.get("b_id") or "").strip()
            if not raw_a and not raw_b and not (row.get("decision") or "").strip():
                continue  # an empty line
            a, b = resolver.get(raw_a), resolver.get(raw_b)
            if a is None or b is None:
                unknown = raw_a if a is None else raw_b
                problems.append(f"row {line}: unknown indicator {unknown!r}")
                continue
            if (a, b) not in by_pair:
                problems.append(f"row {line}: {raw_a} -> {raw_b} is not a proposed edge")
                continue
            try:
                decision = _decision(row.get("decision") or "")
            except ValueError as exc:
                problems.append(f"row {line}: {exc}")
                continue
            if decision is None:
                report.blank += 1
                continue
            who = (row.get("reviewer") or "").strip() or (reviewer or "").strip()
            if not who:
                problems.append(f"row {line}: a decision needs a reviewer id (fill the reviewer cell or pass one)")
                continue
            if (a, b) in decisions and decisions[(a, b)][0] != decision:
                problems.append(f"row {line}: {raw_a} -> {raw_b} has two different decisions in this sheet")
                continue
            decisions[(a, b)] = (decision, who)
    if problems:
        raise ReviewError(problems)

    # the graph the accepted edges must fit into: rule edges plus edges accepted earlier and not re-decided here
    H = rule_graph(G)
    for (a, b), e in by_pair.items():
        if _status(e) == "accepted" and (a, b) not in decisions and not nx.has_path(H, b, a):
            H.add_edge(a, b, type="augmented")

    def record(e: dict[str, Any], decision: str, who: str) -> None:
        review = (e.get("provenance") or {}).get("review") or {}
        if review.get("status") == decision and review.get("by") == who:
            report.unchanged += 1
            return
        e["provenance"] = {**(e.get("provenance") or {}), "review": {"status": decision, "by": who, "at": at}}
        if decision == "accepted":
            report.accepted += 1
        else:
            report.rejected += 1

    ordered = sorted(decisions.items(), key=lambda kv: (-float(by_pair[kv[0]].get("weight", 0.0)), kv[0]))
    for (a, b), (decision, who) in ordered:
        e = by_pair[(a, b)]
        if decision == "rejected":
            record(e, decision, who)
        elif a == b or nx.has_path(H, b, a):
            report.refused.append(
                {"source": e["source"], "target": e["target"], "reason": "accepting it would close a cycle"}
            )
        else:
            H.add_edge(a, b, type="augmented")
            record(e, decision, who)
    out = {**doc, "edges": edges}
    report.counts = review_counts(out)
    return out, report


def acceptance(doc: Mapping[str, Any]) -> dict[str, Any]:
    """Teacher acceptance so far: accepted / (accepted + rejected), or "not yet measured" with no review."""
    counts = review_counts(doc)
    reviewed = counts["accepted"] + counts["rejected"]
    if reviewed == 0:
        return {
            "reviewed": 0,
            "teacher_acceptance": None,
            "status": "not yet measured: no teacher has reviewed an edge",
        }
    return {
        "reviewed": reviewed,
        "teacher_acceptance": round(counts["accepted"] / reviewed, 4),
        "status": "measured" if reviewed >= 100 else f"partly measured: {reviewed} of the 100-edge sample reviewed",
    }
