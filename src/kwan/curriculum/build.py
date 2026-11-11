"""The curriculum build (PRD task 2.6): parse the PDFs and write the two indicator files.

    data/curriculum/<subject>.structure.json   committed: codes, numbers, names and Kwan-written labels
    build/curriculum/<subject>.full.json       git-ignored: the same records plus the NaCCA text

NaCCA's descriptions, content-standard text and exemplars are copyright, so they stay out of every
committed file until NaCCA gives written permission (PRD 5.3, decision D5). Two switches control this:

    KWAN_CURRICULUM_TEXT=full|labels     what downstream steps use (default: labels)
    data/curriculum/permissions.json      "nacca_text_commit_allowed": true lets the build write the full
                                          text to a committed file as well
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kwan._paths import repo_root
from kwan.curriculum.corrections import Corrections, draft_key_map, load_corrections
from kwan.curriculum.document import ParseResult, parse_pages
from kwan.curriculum.exemplars import load_terms
from kwan.curriculum.families import Family, load_family
from kwan.curriculum.labels import apply_labels, read_drafts, read_reviews
from kwan.curriculum.model import PageText

TEXT_ENV = "KWAN_CURRICULUM_TEXT"
STRUCTURE_FIELDS = (
    "id",
    "subject",
    "code",
    "grade",
    "strand",
    "strand_name",
    "sub_strand",
    "sub_strand_name",
    "content_standard",
    "cs_key",
    "indicator_no",
    "short_label",
    "label_reviewed",
    "aliases",
)
# fields that hold NaCCA sentences: never in a committed file while permission is missing
TEXT_FIELDS = ("description", "content_standard_text", "exemplars")
DEFAULT_PERMISSIONS = {
    "nacca_text_commit_allowed": False,
    "granted_by": None,
    "granted_on": None,
    "evidence": None,
    "note": (
        "NaCCA curriculum text (indicator descriptions, content-standard text, exemplars) is copyright "
        "NaCCA, Ministry of Education 2019. Kwan commits codes, numbers, names and its own short labels "
        "only. Set nacca_text_commit_allowed to true only after NaCCA gives written permission, and record "
        "who granted it, when, and where the letter is kept (docs/governance.md)."
    ),
}


def root_dir(root: Path | str | None = None) -> Path:
    """The repository root (or the current directory outside a checkout)."""
    return Path(root) if root else (repo_root() or Path.cwd())


def text_mode() -> str:
    """`"full"` or `"labels"`, from the `KWAN_CURRICULUM_TEXT` environment variable (default labels)."""
    value = os.environ.get(TEXT_ENV, "labels").strip().lower()
    if value not in ("full", "labels"):
        raise ValueError(f"{TEXT_ENV} must be 'full' or 'labels', not {value!r}")
    return value


def permissions_path(root: Path | str | None = None) -> Path:
    """Where the permissions file lives."""
    return root_dir(root) / "data" / "curriculum" / "permissions.json"


def load_permissions(root: Path | str | None = None) -> dict[str, Any]:
    """The permissions file; a missing file means nothing is allowed."""
    path = permissions_path(root)
    if not path.exists():
        return dict(DEFAULT_PERMISSIONS)
    return {**DEFAULT_PERMISSIONS, **json.loads(path.read_text(encoding="utf-8"))}


def commit_allowed(root: Path | str | None = None) -> bool:
    """True only when the permissions file says NaCCA text may be committed."""
    return load_permissions(root).get("nacca_text_commit_allowed") is True


def load_indicators(
    family: Family | None = None, mode: str | None = None, root: Path | str | None = None
) -> tuple[list[dict[str, Any]], str]:
    """The indicator records downstream steps should use, and the text mode they are in.

    In `full` mode the records come from the full file (the git-ignored build output, or the committed
    copy once permission allows one). When no full file exists, or the mode is `labels`, the committed
    structure file is returned and the mode is `"labels"`.
    """
    family = family or load_family()
    mode = mode or text_mode()
    base = root_dir(root)
    if mode == "full":
        for key in ("full_file", "committed_full_file"):
            path = family.path(key, base)
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8")), "full"
    return json.loads(family.path("structure_file", base).read_text(encoding="utf-8")), "labels"


def strip_text(record: dict[str, Any]) -> dict[str, Any]:
    """The committed form of a record: structure fields only, in a fixed order."""
    return {k: record[k] for k in STRUCTURE_FIELDS if k in record}


# ---- extraction -------------------------------------------------------------------------------------


def find_pdf(family: Family, doc_id: str, root: Path | str | None = None) -> Path | None:
    """The local PDF of a document: `raw/curriculum/<file>` first, then the family's known copies."""
    base = root_dir(root)
    doc = next(d for d in family.documents if d.id == doc_id)
    for candidate in [Path("raw") / "curriculum" / doc.file, *(Path(p) for p in doc.local_copies)]:
        if (base / candidate).exists():
            return base / candidate
    return None


def pdftotext(pdf: Path, out: Path) -> bool:
    """Run `pdftotext -layout`; False when the tool is not installed or fails."""
    exe = shutil.which("pdftotext")
    if exe is None:
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.run([exe, "-layout", str(pdf), str(out)], capture_output=True).returncode == 0  # noqa: S603


def find_text(family: Family, doc_id: str, root: Path | str | None = None) -> Path | None:
    """The `pdftotext -layout` dump of a document, made from the PDF when it is not there yet."""
    base = root_dir(root)
    doc = next(d for d in family.documents if d.id == doc_id)
    made = base / "build" / "curriculum" / "text" / f"{doc.id}.txt"
    pdf = find_pdf(family, doc_id, base)
    if made.exists() and (pdf is None or made.stat().st_mtime >= pdf.stat().st_mtime):
        return made
    if pdf is not None and pdftotext(pdf, made):
        return made
    for candidate in doc.text_copies:
        if (base / candidate).exists():
            return base / candidate
    return None


def extract(family: Family, engine: str = "pdfplumber", root: Path | str | None = None) -> list[PageText]:
    """All pages of all documents of a family, through one extraction path."""
    base = root_dir(root)
    pages: list[PageText] = []
    for doc in family.documents:
        if engine == "pdfplumber":
            from kwan.curriculum import pdf as pdf_path

            source = find_pdf(family, doc.id, base)
            if source is None:
                raise FileNotFoundError(f"no PDF for {doc.id}: run `kwan curriculum fetch` first")
            pages += pdf_path.extract_pages(source, family, doc.id, base / "build" / "curriculum" / "cache")
        elif engine == "pdftotext":
            from kwan.curriculum import text as text_path

            source = find_text(family, doc.id, base)
            if source is None:
                raise FileNotFoundError(f"no text dump for {doc.id}: install pdftotext (poppler) or fetch the PDF")
            pages += text_path.extract_pages(source, family, doc.id)
        else:
            raise ValueError(f"unknown extraction engine {engine!r}; use pdfplumber or pdftotext")
    return pages


def parse_family(
    family: Family, engine: str = "pdfplumber", root: Path | str | None = None, corrections: Corrections | None = None
) -> ParseResult:
    """Extract and parse a family with its corrections file and the Ghana term list."""
    base = root_dir(root)
    if corrections is None:
        corrections = load_corrections(family.path("corrections", base))
    return parse_pages(extract(family, engine, base), family, corrections, load_terms())


# ---- build ------------------------------------------------------------------------------------------


@dataclass
class BuildResult:
    """What one build wrote and found."""

    records: list[dict[str, Any]]
    parse: ParseResult
    labels: dict[str, list[str]]
    engine: str
    written: list[Path] = field(default_factory=list)


def write_json(path: Path, data: Any) -> None:
    """Write JSON with a stable layout (UTF-8, one-space indent, trailing newline)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def build(
    family: Family | None = None, engine: str = "pdfplumber", root: Path | str | None = None, write: bool = True
) -> BuildResult:
    """Parse the family, attach short labels and write the structure file and the full file."""
    family = family or load_family()
    base = root_dir(root)
    corrections_path = family.path("corrections", base)
    parsed = parse_family(family, engine, base)
    records = parsed.records
    drafts = read_drafts(base / family.paths["label_drafts"])
    reviews = read_reviews(family.path("label_reviews", base))
    labels = apply_labels(records, drafts, reviews, draft_key_map(corrections_path))
    full = [{**strip_text(r), **{k: r[k] for k in r if k not in STRUCTURE_FIELDS}} for r in records]
    result = BuildResult(records=full, parse=parsed, labels=labels, engine=engine)
    if not write:
        return result
    if not permissions_path(base).exists():
        write_json(permissions_path(base), DEFAULT_PERMISSIONS)
        result.written.append(permissions_path(base))
    targets = [
        (family.path("structure_file", base), [strip_text(r) for r in full]),
        (family.path("full_file", base), full),
        (
            family.path("full_file", base).with_name(f"{family.subject}.parse.json"),
            {"engine": engine, "log": parsed.log, "sub_strand_sources": parsed.sub_strand_sources, "labels": labels},
        ),
    ]
    if commit_allowed(base):
        targets.append((family.path("committed_full_file", base), full))
    for path, data in targets:
        write_json(path, data)
        result.written.append(path)
    return result
