"""The committed curriculum files: counts equal the hand count, ids are sound, corrections are consistent.

The last tests parse the real PDFs and need files under raw/ (marked slow, skipped when they are missing).
"""

import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path

import pytest

from kwan.curriculum import build
from kwan.curriculum.corrections import ACTIONS, draft_key_map, load_corrections
from kwan.curriculum.document import is_truncated
from kwan.curriculum.families import load_family
from kwan.curriculum.labels import MAX_LABEL, read_drafts, read_reviews

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "curriculum"
FAM = load_family()
CODE = re.compile(r"B([1-6])\.([1-4])\.(\d)\.(\d)\.(\d{1,2})")


@pytest.fixture(scope="module")
def structure():
    return json.loads((DATA / "math.structure.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def expected():
    return json.loads((DATA / "expected_counts.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def corrections():
    return json.loads((DATA / "corrections.json").read_text(encoding="utf-8"))


def table(records):
    return dict(Counter((str(r["grade"]), str(r["strand"]), str(r["sub_strand"])) for r in records))


def flat(expected):
    return {(g, s, ss): n for g, a in expected["counts"].items() for s, b in a.items() for ss, n in b.items()}


# ---- counts (PRD task 2.3) --------------------------------------------------------------------------


def test_hand_count_adds_up(expected):
    for grade, strands in expected["counts"].items():
        assert sum(sum(v.values()) for v in strands.values()) == expected["per_grade"][grade], grade
    assert sum(expected["per_grade"].values()) == expected["total"]
    assert all(n > 0 for n in flat(expected).values())


def test_structure_matches_the_hand_count_exactly(structure, expected):
    assert table(structure) == flat(expected)
    assert len(structure) == expected["total"]
    assert {str(g): n for g, n in Counter(r["grade"] for r in structure).items()} == expected["per_grade"]


def test_gaps_in_the_pdf_are_recorded_and_not_filled(structure, expected):
    codes = {r["code"] for r in structure}
    standards = {r["content_standard"] for r in structure}
    sub_strands = {r["content_standard"].rsplit(".", 1)[0] for r in structure}
    gaps = [g for note in expected["notes"] for g in note.get("gap", [])]
    assert "B4.1.3.2.2" in gaps and "B5.3.4.2.1" in gaps and "B6.3.2" in gaps
    for gap in gaps:
        assert gap not in codes and gap not in standards and gap not in sub_strands, gap
    # and every jump in the numbering of the committed file is one of the recorded gaps
    by_cs = {}
    for r in structure:
        by_cs.setdefault(r["content_standard"], []).append(r["indicator_no"])
    missing = {f"{cs}.{i}" for cs, nos in by_cs.items() for i in range(1, max(nos) + 1) if i not in nos}
    assert missing <= set(gaps)


# ---- records ----------------------------------------------------------------------------------------


def test_record_schema_and_order(structure):
    for r in structure:
        assert tuple(r) == build.STRUCTURE_FIELDS, r["id"]
        g, s, ss, cs, no = (int(x) for x in CODE.fullmatch(r["code"]).groups())
        assert r["id"] == f"math:{r['code']}" and r["subject"] == "math"
        assert (r["grade"], r["strand"], r["sub_strand"], r["indicator_no"]) == (g, s, ss, no)
        assert r["content_standard"] == f"B{g}.{s}.{ss}.{cs}" and r["cs_key"] == f"{s}.{ss}.{cs}"
        assert r["strand_name"] == FAM.strands[s]
        assert r["sub_strand_name"] and not r["sub_strand_name"].startswith("Sub-strand ")
        assert r["aliases"][0] == r["code"]
    keys = [
        (r["grade"], r["strand"], r["sub_strand"], int(r["cs_key"].split(".")[-1]), r["indicator_no"])
        for r in structure
    ]
    assert keys == sorted(keys)


def test_ids_and_aliases_are_unique(structure):
    ids = [r["id"] for r in structure]
    assert len(ids) == len(set(ids))
    names = [a for r in structure for a in r["aliases"]]
    assert len(names) == len(set(names)), [n for n, c in Counter(names).items() if c > 1]


def test_sub_strand_names_are_one_per_sub_strand(structure):
    names = {}
    for r in structure:
        names.setdefault((r["grade"], r["strand"], r["sub_strand"]), set()).add(r["sub_strand_name"])
    assert all(len(v) == 1 for v in names.values())


# ---- corrections ------------------------------------------------------------------------------------


def test_corrections_are_complete_and_point_at_real_indicators(structure, corrections):
    codes = {r["code"]: r for r in structure}
    for c in corrections["corrections"]:
        assert c["action"] in ACTIONS
        assert c["reason"].strip() and c["source"].strip() and c["doc"] in {d.id for d in FAM.documents}
        assert isinstance(c["page"], int)
        if c["action"] == "drop_heading":
            assert c["corrected"] is None
            continue
        assert c["corrected"] in codes, c
        aliases = codes[c["corrected"]]["aliases"]
        if c["action"] == "recode" and c.get("alias") and c["printed"] != c["corrected"]:
            assert c["printed"] in aliases
        else:
            assert c["printed"] not in aliases[1:]


def test_the_known_misprints_are_fixed(structure):
    by = {r["code"]: r for r in structure}
    # a Basic 3 statement printed with a Basic 4 code: both indicators exist, and neither owns the other's code
    assert by["B3.1.1.3.2"]["aliases"] == ["B3.1.1.3.2"]
    assert by["B4.1.1.3.2"]["aliases"] == ["B4.1.1.3.2"]
    assert by["B3.1.1.3.2"]["short_label"] != by["B4.1.1.3.2"]["short_label"]
    # digit misprints keep the printed code as an alias
    assert by["B6.1.2.6.1"]["aliases"] == ["B6.1.2.6.1", "B6.1.1.6.1"]
    assert by["B6.1.2.6.3"]["aliases"] == ["B6.1.2.6.3", "B6.1.1.4.3"]
    assert by["B5.3.2.2.2"]["aliases"] == ["B5.3.2.2.2", "B5.3.3.2.2"]
    assert by["B5.3.2.2.4"]["aliases"] == ["B5.3.2.2.4", "B5.3.3.2.4"]
    assert by["B6.4.2.2.3"]["aliases"] == ["B6.4.2.2.3", "P6.4.2.2.3"]
    assert "B6.4.1.2.2" in by and "B2.3.2.1.1" in by and "B6.2.2.1.1" in by
    for code in ("B5.2.3.1.3", "B5.3.2.2.3", "B5.3.4.1.1", "B5.4.1.1.1"):  # no space after the code in print
        assert code in by
    # the code that is printed for two indicators: the first keeps it, the second has its own id
    assert by["B6.3.3.5.1"]["aliases"] == ["B6.3.3.5.1"]
    assert by["B6.3.3.6.1"]["aliases"] == ["B6.3.3.6.1"]
    assert by["B6.3.3.6.2"]["aliases"] == ["B6.3.3.6.2", "B6.3.3.5.2"]
    assert "B6.3.3.5.2" not in by
    for wrong in ("B6.1.1.6.1", "B6.1.1.4.3", "B5.3.3.2.2", "B5.3.3.2.4"):
        assert wrong not in by


def test_every_draft_key_reaches_exactly_one_indicator(structure):
    drafts = read_drafts(ROOT / FAM.paths["label_drafts"])
    if not drafts:
        pytest.skip("no draft files in this checkout")
    codes = {r["code"] for r in structure}
    alias = {a: r["code"] for r in structure for a in r["aliases"]}
    keys = draft_key_map(DATA / "corrections.json")
    assert keys["B6.3.3.5.1_dup"] == "B6.3.3.6.1"
    reached = []
    for key in drafts:
        code = key if key in codes else keys.get(key) or alias.get(key)
        assert code in codes, f"draft key {key} matches no indicator"
        reached.append(code)
    assert len(reached) == len(set(reached))
    assert load_corrections(DATA / "corrections.json").find("B6.3.3.5.1", "m46", 2).corrected == "B6.3.3.6.1"


# ---- short labels (PRD task 2.7) --------------------------------------------------------------------


def test_short_labels_are_short_and_reviewed_only_through_the_review_file(structure):
    reviews = read_reviews(ROOT / FAM.paths["label_reviews"])
    for r in structure:
        assert r["short_label"].strip() and len(r["short_label"]) <= MAX_LABEL, r["code"]
        assert "—" not in r["short_label"]
        assert r["label_reviewed"] is (r["code"] in reviews or r["id"] in reviews), r["code"]


def test_results_block_agrees_with_the_committed_files(structure, expected):
    results = json.loads((ROOT / "data" / "eval_results.json").read_text(encoding="utf-8"))
    block = results.get("curriculum")
    if block is None:
        pytest.skip("kwan curriculum qa has not written its results block yet")
    assert block["indicators"] == len(structure) == block["expected"] == expected["total"]
    assert block["per_grade"] == expected["per_grade"] and block["counts_match"] is True
    assert block["truncation_rate"] == round(block["truncated"] / block["indicators"], 4) < 0.005
    assert block["share_with_exemplar"] == round(block["with_exemplar"] / block["indicators"], 4) >= 0.9
    assert block["duplicate_ids"] == 0
    assert block["labels"]["reviewed"] == sum(r["label_reviewed"] for r in structure)
    sources = json.loads((DATA / "sources.json").read_text(encoding="utf-8"))["sources"]
    assert {s["id"]: s["sha256"] for s in block["sources"]} == {s["id"]: s["sha256"] for s in sources}


def test_sources_file_records_url_size_hash_and_date():
    sources = json.loads((DATA / "sources.json").read_text(encoding="utf-8"))["sources"]
    assert [s["id"] for s in sources] == [d.id for d in FAM.documents]
    for s, d in zip(sources, FAM.documents):
        assert s["url"] == d.url and s["url"].startswith("https://curriculumresources.edu.gh/")
        assert s["size_bytes"] > 100_000 and re.fullmatch(r"[0-9a-f]{64}", s["sha256"])
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", s["fetched_at"])


# ---- the real PDFs (need raw/) ----------------------------------------------------------------------

HAVE_PDFS = all(build.find_pdf(FAM, d.id, ROOT) is not None for d in FAM.documents)
HAVE_TEXT = all(any((ROOT / t).exists() for t in d.text_copies) for d in FAM.documents) or (
    HAVE_PDFS and shutil.which("pdftotext") is not None
)


def cache_is_warm():
    from kwan.curriculum import pdf

    return HAVE_PDFS and all(
        (
            pdf.default_cache_dir()
            / f"{pdf.sha256_of(build.find_pdf(FAM, d.id, ROOT))[:16]}.v{pdf.CACHE_VERSION}.words.json"
        ).exists()
        for d in FAM.documents
    )


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_PDFS, reason="needs the curriculum PDFs under raw/ (kwan curriculum fetch)")
def test_local_pdfs_match_the_recorded_hashes():
    from kwan.curriculum.pdf import sha256_of

    sources = {s["id"]: s for s in json.loads((DATA / "sources.json").read_text(encoding="utf-8"))["sources"]}
    for d in FAM.documents:
        path = build.find_pdf(FAM, d.id, ROOT)
        assert sha256_of(path) == sources[d.id]["sha256"] and path.stat().st_size == sources[d.id]["size_bytes"]


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_PDFS, reason="needs the curriculum PDFs under raw/ (kwan curriculum fetch)")
@pytest.mark.skipif(
    not (os.environ.get("KWAN_RUN_SLOW") == "1" or cache_is_warm()),
    reason="slow: reading the PDFs with pdfplumber takes minutes without the word cache; set KWAN_RUN_SLOW=1",
)
def test_pdfplumber_parse_matches_hand_count_and_committed_structure(structure, expected):
    pytest.importorskip("pdfplumber")
    result = build.build(FAM, engine="pdfplumber", root=ROOT, write=False)
    assert not result.parse.errors()
    assert table(result.records) == flat(expected)
    assert [build.strip_text(r) for r in result.records] == structure  # the committed file is what the parser gives
    n = len(result.records)
    truncated = [r["code"] for r in result.records if is_truncated(r["description"])]
    assert len(truncated) / n < 0.005, truncated  # PRD acceptance: truncation below 0.5%
    assert sum(1 for r in result.records if r["exemplars"]) / n >= 0.9  # PRD acceptance: exemplars for 90% or more
    assert all(r["content_standard_text"] and r["page"] > 0 for r in result.records)
    assert result.labels["missing"] == result.labels["same_as_description"]  # the only gaps are blocked repeats
    assert not result.labels["unmatched_draft_keys"]


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_TEXT, reason="needs the pdftotext dumps under raw/ or pdftotext and the PDFs")
def test_pdftotext_fallback_finds_the_same_indicators(structure, expected):
    result = build.parse_family(FAM, engine="pdftotext", root=ROOT)
    assert not result.errors()
    assert table(result.records) == flat(expected)
    assert [r["id"] for r in result.records] == [r["id"] for r in structure]
    assert [r["aliases"] for r in result.records] == [r["aliases"] for r in structure]
    assert sum(1 for r in result.records if r["exemplars"]) / len(result.records) >= 0.9
