"""Licensing mode (PRD 5.3, decision D5, task 2.6): no NaCCA sentence in any committed curriculum file.

Also the build on a synthetic document: structure file, full file, permissions flag, text mode, labels.
"""

import json
import re
from pathlib import Path

import pytest

from kwan.curriculum import build
from kwan.curriculum.families import family_from_dict, load_family

from .curriculum_helpers import MATH, layout_row

ROOT = Path(__file__).resolve().parent.parent
FAM = load_family()
COMMITTED_DIR = ROOT / "data" / "curriculum"
TEXT_KEYS = set(build.TEXT_FIELDS) | {"exemplar", "content_standard_text", "indicator_description"}
MIN_WORDS = 7  # a run of fewer words ("Compare and order fractions") is a skill name, not a sentence


def norm(text):
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def committed_files():
    """Every committed data file that could carry curriculum text: data/**, the shipped bundle, the QA report."""
    roots = [ROOT / "data", ROOT / "src" / "kwan" / "bundles", ROOT / "docs" / "qa"]
    files = sorted(p for root in roots for p in root.rglob("*") if p.is_file() and p.name != ".DS_Store")
    report = FAM.path("qa_report", ROOT)
    return files + ([report] if report.exists() and report not in files else [])


def keys_in(node):
    if isinstance(node, dict):
        for k, v in node.items():
            yield k
            yield from keys_in(v)
    elif isinstance(node, list):
        for v in node:
            yield from keys_in(v)


def sentences(text):
    """Pieces of a NaCCA text that are long enough to recognise."""
    for piece in re.split(r"(?<=[.;?!])\s+", text):
        if len(norm(piece).split()) >= MIN_WORDS:
            yield norm(piece)


def nacca_strings(full):
    """Normalised NaCCA sentences to search for inside committed files: {sentence: where it comes from}."""
    out = {}
    for r in full:
        if len(norm(r["description"]).split()) >= MIN_WORDS:
            out[norm(r["description"])] = f"description of {r['code']}"
        if len(norm(r["content_standard_text"]).split()) >= MIN_WORDS:
            out[norm(r["content_standard_text"])] = f"content standard of {r['code']}"
        for e in r["exemplars"]:
            for s in sentences(e["text"]):
                out[s] = f"exemplar of {r['code']}"
    return out


def whole_values(node):
    """Every string value in a JSON document."""
    if isinstance(node, dict):
        for v in node.values():
            yield from whole_values(v)
    elif isinstance(node, list):
        for v in node:
            yield from whole_values(v)
    elif isinstance(node, str):
        yield node


pytestmark = pytest.mark.skipif(build.commit_allowed(ROOT), reason="NaCCA has given permission: text may be committed")


def test_permission_flag_is_off_and_no_full_text_file_is_committed():
    perms = json.loads((COMMITTED_DIR / "permissions.json").read_text(encoding="utf-8"))
    assert perms["nacca_text_commit_allowed"] is False
    assert not FAM.path("committed_full_file", ROOT).exists()
    assert FAM.paths["full_file"].startswith("build/")  # the text-bearing file lives in the git-ignored build folder


def test_committed_files_have_no_text_fields():
    """Runs everywhere, with or without the PDFs: no committed JSON has a field that holds NaCCA text."""
    structure = json.loads(FAM.path("structure_file", ROOT).read_text(encoding="utf-8"))
    assert all(tuple(r) == build.STRUCTURE_FIELDS for r in structure)
    for path in COMMITTED_DIR.rglob("*.json"):
        found = set(keys_in(json.loads(path.read_text(encoding="utf-8")))) & TEXT_KEYS
        assert not found, f"{path.name} has text fields {sorted(found)}"


def test_no_nacca_sentence_in_any_committed_file():
    """With the build output at hand, look for every description, standard and exemplar sentence."""
    full_path = FAM.path("full_file", ROOT)
    if not full_path.exists():
        pytest.skip("build/curriculum/math.full.json is not built here; the schema check above still ran")
    full = json.loads(full_path.read_text(encoding="utf-8"))
    needles = nacca_strings(full)
    assert len(needles) > 500
    # any description, however short, must not appear word for word as a committed value
    exact = {norm(r["description"]): r["code"] for r in full if r["description"].strip()}
    for path in committed_files():
        hay = norm(path.read_text(encoding="utf-8"))
        hits = sorted({where for text, where in needles.items() if text in hay})
        assert not hits, f"{path.relative_to(ROOT)} repeats NaCCA text: {hits[:10]}"
        if path.suffix == ".json":
            copies = sorted(
                {exact[norm(v)] for v in whole_values(json.loads(path.read_text(encoding="utf-8"))) if norm(v) in exact}
            )
            assert not copies, f"{path.relative_to(ROOT)} holds a description word for word: {copies}"


def test_short_labels_are_not_copies_of_the_descriptions():
    full_path = FAM.path("full_file", ROOT)
    if not full_path.exists():
        pytest.skip("build/curriculum/math.full.json is not built here")
    copies = [
        r["code"]
        for r in json.loads(full_path.read_text(encoding="utf-8"))
        if norm(r["short_label"]) == norm(r["description"])
    ]
    assert not copies


def test_the_check_itself_catches_a_leak(tmp_path):
    full = [
        {
            "code": "B1.1.1.1.1",
            "description": "Sort a set of made-up objects by size",
            "content_standard_text": "",
            "exemplars": [{"text": "Short one. Ask pupils to line up the made-up objects from small to big."}],
        }
    ]
    needles = nacca_strings(full)
    assert norm("Sort a set of made-up objects by size") in needles
    assert any("line up the made up objects" in n for n in needles) and "short one" not in needles
    leaked = norm(json.dumps({"reason": "It says: sort a set of MADE-UP objects, by size."}))
    assert any(n in leaked for n in needles)
    assert not any(n in norm(json.dumps({"short_label": "Sort objects by size"})) for n in needles)


# ---- the build on a synthetic document --------------------------------------------------------------


def synthetic_root(tmp_path, allowed=None, reviews=None):
    fam = family_from_dict(
        "synthetic-math", {**MATH, "documents": [{"id": "doc1", "grades": [4], "text_copies": ["raw/doc1.txt"]}]}
    )
    header = layout_row("INDICATORS AND EXEMPLARS", "CONTENT STANDARDS", "CORE COMPETENCIES")
    pages = [
        ["BASIC 4".center(80)],
        [
            "Strand 1: NUMBER".center(90),
            "Sub-strand 1: Counting things".center(90),
            header,
            layout_row("B4.1.1.1.1 Count the made-up beads aloud", "B4.1.1.1", "Learners develop:"),
            layout_row("E.g. 1. Kofi counts twelve beads for one cedi", "Know made-up numbers"),
            layout_row("B4.1.1.6.2 Order the made-up beads by size"),
            layout_row("E.g. 1. Line the beads up from small to big"),
        ],
    ]
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw/doc1.txt").write_text("\f".join("\n".join(p) for p in pages) + "\f", encoding="utf-8")
    (tmp_path / "data/curriculum").mkdir(parents=True)
    (tmp_path / "data/curriculum/math.corrections.json").write_text(
        json.dumps(
            {
                "corrections": [
                    {"printed": "B4.1.1.6.2", "corrected": "B4.1.1.1.2", "draft_key": "B4.1.1.6.2", "reason": "digit"}
                ]
            }
        )
    )
    (tmp_path / "data/contexts/drafts").mkdir(parents=True)
    drafts = {
        "meta": {},
        "indicators": {
            "B4.1.1.1.1": {"short_label": "Count beads aloud"},
            "B4.1.1.6.2": {"short_label": "Order beads by size"},
        },
    }
    (tmp_path / "data/contexts/drafts/math.B4.json").write_text(json.dumps(drafts))
    if reviews is not None:
        (tmp_path / "data/contexts/math.label_reviews.json").write_text(json.dumps(reviews))
    if allowed is not None:
        (tmp_path / "data/curriculum/permissions.json").write_text(json.dumps({"nacca_text_commit_allowed": allowed}))
    return fam


def test_build_writes_a_text_free_structure_file_and_a_full_file_under_build(tmp_path):
    fam = synthetic_root(tmp_path)
    result = build.build(fam, engine="pdftotext", root=tmp_path)
    structure = json.loads((tmp_path / "data/curriculum/math.structure.json").read_text())
    full = json.loads((tmp_path / "build/curriculum/math.full.json").read_text())
    assert [r["id"] for r in structure] == [r["id"] for r in full] == ["math:B4.1.1.1.1", "math:B4.1.1.1.2"]
    assert all(tuple(r) == build.STRUCTURE_FIELDS for r in structure)
    assert structure[1] == {
        "id": "math:B4.1.1.1.2",
        "subject": "math",
        "code": "B4.1.1.1.2",
        "grade": 4,
        "strand": 1,
        "strand_name": "Number",
        "sub_strand": 1,
        "sub_strand_name": "Counting things",
        "content_standard": "B4.1.1.1",
        "cs_key": "1.1.1",
        "indicator_no": 2,
        "short_label": "Order beads by size",  # found through the printed code the draft was keyed by
        "label_reviewed": False,
        "aliases": ["B4.1.1.1.2", "B4.1.1.6.2"],
    }
    assert full[0]["description"] == "Count the made-up beads aloud"
    assert full[0]["content_standard_text"] == "Know made-up numbers"
    assert full[0]["exemplars"] == [
        {"text": "Kofi counts twelve beads for one cedi", "ghana_context": True, "page": 2, "marked": True}
    ]
    assert full[0]["page"] == 2 and list(full[0])[: len(build.STRUCTURE_FIELDS)] == list(build.STRUCTURE_FIELDS)
    committed = " ".join(p.read_text() for p in (tmp_path / "data/curriculum").rglob("*.json"))
    assert "made-up" not in committed and "twelve beads" not in committed
    # the permissions file is created closed, and no text-bearing file lands under data/
    assert json.loads((tmp_path / "data/curriculum/permissions.json").read_text())["nacca_text_commit_allowed"] is False
    assert not (tmp_path / "data/curriculum/math.full.json").exists()
    assert result.labels == {
        "missing": [],
        "too_long": [],
        "reviewed": [],
        "unmatched_draft_keys": [],
        "same_as_description": [],
    }


def test_flipping_the_permission_flag_commits_the_full_text(tmp_path):
    fam = synthetic_root(tmp_path, allowed=True)
    build.build(fam, engine="pdftotext", root=tmp_path)
    committed = json.loads((tmp_path / "data/curriculum/math.full.json").read_text())
    assert committed[0]["description"] == "Count the made-up beads aloud"
    assert build.commit_allowed(tmp_path) is True


@pytest.mark.parametrize("value", ["yes", 1, "true", None])
def test_only_a_real_true_opens_the_permission(tmp_path, value):
    synthetic_root(tmp_path, allowed=value)
    assert build.commit_allowed(tmp_path) is False


def test_reviewed_labels_come_only_from_the_review_file(tmp_path):
    reviews = {
        "B4.1.1.1.2": {
            "short_label": "Put beads in order of size",
            "reviewer": "teacher-01",
            "reviewed_at": "2026-11-12",
        }
    }
    fam = synthetic_root(tmp_path, reviews=reviews)
    result = build.build(fam, engine="pdftotext", root=tmp_path)
    a, b = json.loads((tmp_path / "data/curriculum/math.structure.json").read_text())
    assert (a["short_label"], a["label_reviewed"]) == ("Count beads aloud", False)
    assert (b["short_label"], b["label_reviewed"]) == ("Put beads in order of size", True)
    assert result.labels["reviewed"] == ["B4.1.1.1.2"]


def test_indicator_without_a_label_gets_a_placeholder_and_is_reported(tmp_path):
    fam = synthetic_root(tmp_path)
    (tmp_path / "data/contexts/drafts/math.B4.json").write_text(
        json.dumps({"indicators": {"B4.1.1.1.1": {"short_label": "x" * 61}, "B9.9.9.9.9": {"short_label": "Nothing"}}})
    )
    result = build.build(fam, engine="pdftotext", root=tmp_path)
    b = json.loads((tmp_path / "data/curriculum/math.structure.json").read_text())[1]
    assert (b["short_label"], b["label_reviewed"]) == ("Skill B4.1.1.1.2", False)
    assert result.labels == {
        "missing": ["B4.1.1.1.2"],
        "too_long": ["B4.1.1.1.1"],
        "reviewed": [],
        "unmatched_draft_keys": ["B9.9.9.9.9"],
        "same_as_description": [],
    }


def test_a_draft_label_that_repeats_the_statement_is_not_committed(tmp_path):
    fam = synthetic_root(tmp_path)
    (tmp_path / "data/contexts/drafts/math.B4.json").write_text(
        json.dumps({"indicators": {"B4.1.1.1.1": {"short_label": "Count the made-up beads aloud."}}})
    )
    result = build.build(fam, engine="pdftotext", root=tmp_path)
    a = json.loads((tmp_path / "data/curriculum/math.structure.json").read_text())[0]
    assert (a["short_label"], a["label_reviewed"]) == ("Skill B4.1.1.1.1", False)
    assert result.labels["same_as_description"] == ["B4.1.1.1.1"] and "B4.1.1.1.1" in result.labels["missing"]


def test_text_mode_switch(tmp_path, monkeypatch):
    fam = synthetic_root(tmp_path)
    monkeypatch.delenv(build.TEXT_ENV, raising=False)
    assert build.text_mode() == "labels"
    build.build(fam, engine="pdftotext", root=tmp_path)
    records, mode = build.load_indicators(fam, root=tmp_path)
    assert mode == "labels" and "description" not in records[0]
    monkeypatch.setenv(build.TEXT_ENV, "full")
    records, mode = build.load_indicators(fam, root=tmp_path)
    assert mode == "full" and records[0]["description"] == "Count the made-up beads aloud"
    (tmp_path / "build/curriculum/math.full.json").unlink()  # a fresh clone has no build output: fall back to labels
    records, mode = build.load_indicators(fam, root=tmp_path)
    assert mode == "labels" and "description" not in records[0]
    monkeypatch.setenv(build.TEXT_ENV, "everything")
    with pytest.raises(ValueError, match="KWAN_CURRICULUM_TEXT"):
        build.text_mode()
