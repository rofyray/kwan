"""Draft files, the corrections mapping and the merge into the settings file; then the committed file itself."""

import json

import pytest

from kwan.context import checks, drafts
from kwan.context.settings import canonical_id, load_contexts, save_contexts, text_hash
from tests.test_context_support import MODEL, SYNTHETIC, draft_set


def write_draft_file(directory, indicators, name="math.test.json", drafted_by="test-model (offline draft)"):
    directory.mkdir(parents=True, exist_ok=True)
    doc = {"meta": {"subject": "math", "drafted_by": drafted_by, "drafted_on": "2026-11-10"}, "indicators": indicators}
    (directory / name).write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")


def write_corrections(path, draft_keys=None, corrections=None):
    doc = {"schema_version": 1, "subject": "math", "corrections": corrections or []}
    if draft_keys is not None:
        doc["draft_keys"] = draft_keys
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")


def indicators_from(data=None):
    data = SYNTHETIC if data is None else data
    return {code: {"short_label": label, "settings": list(settings)} for code, (label, settings) in data.items()}


# ----------------------------------------------------------------------------- reading drafts


def test_load_drafts_maps_printed_keys_through_the_corrections_file(tmp_path):
    inds = indicators_from()
    inds["B6.3.3.5.1_dup"] = inds.pop("B6.1.4.1.1")  # a second use of a printed code, keyed with _dup
    inds["B5.3.3.2.2"] = inds.pop("B5.1.3.1.1")  # a misprinted sub-strand digit
    write_draft_file(tmp_path / "drafts", inds)
    corrections = tmp_path / "corrections.json"
    write_corrections(corrections, {"B6.3.3.5.1_dup": "B6.3.3.6.1", "B5.3.3.2.2": "B5.3.2.2.2"})
    ds = drafts.load_drafts(tmp_path / "drafts", corrections)
    assert ds.mapping == "corrections"
    assert ds.remapped == {"B6.3.3.5.1_dup": "B6.3.3.6.1", "B5.3.3.2.2": "B5.3.2.2.2"}
    assert "math:B6.3.3.6.1" in ds.items and "math:B5.3.2.2.2" in ds.items
    assert "math:B6.3.3.5.1_dup" not in ds.items
    assert ds.items["math:B6.3.3.6.1"]["source"] == "draft:test-model"
    assert ds.items["math:B6.3.3.6.1"]["key"] == "B6.3.3.5.1_dup"
    assert ds.n_settings == 16 and ds.problems == []
    assert list(ds.items)[:2] == ["math:B1.1.1.1.1", "math:B2.1.2.1.1"]  # curriculum order


def test_load_drafts_without_a_corrections_file_is_provisional_and_says_so(tmp_path):
    inds = indicators_from()
    inds["B6.3.3.5.1_dup"] = inds.pop("B6.1.4.1.1")
    write_draft_file(tmp_path / "drafts", inds)
    ds = drafts.load_drafts(tmp_path / "drafts", tmp_path / "no-such-corrections.json")
    assert ds.mapping == "provisional" and ds.remapped == {}
    assert "math:B6.3.3.5.1_dup" in ds.items
    assert any("B6.3.3.5.1_dup" in p and "no canonical code yet" in p for p in ds.problems)


def test_corrections_without_a_draft_keys_map_are_read_from_the_entries(tmp_path):
    path = tmp_path / "corrections.json"
    write_corrections(
        path,
        corrections=[
            {"printed": "B6.1.1.6.1", "corrected": "B6.1.2.6.1", "draft_key": "B6.1.1.6.1", "action": "recode"},
            {"printed": "B3.1.2.3", "corrected": None, "action": "drop_heading"},
            {"printed": "B4.1.1.3.2", "corrected": "B3.1.1.3.2", "draft_key": "B3.1.1.3.2", "action": "recode"},
        ],
    )
    assert drafts.load_draft_key_map(path) == {"B6.1.1.6.1": "B6.1.2.6.1"}
    assert drafts.load_draft_key_map(tmp_path / "missing.json") is None


def test_two_draft_files_that_map_to_the_same_code_keep_the_first_and_report(tmp_path):
    write_draft_file(tmp_path / "drafts", indicators_from({"B1.1.1.1.1": SYNTHETIC["B1.1.1.1.1"]}), "math.a.json")
    write_draft_file(tmp_path / "drafts", {"B1.1.1.1.1": {"short_label": "Other", "settings": ["x"]}}, "math.b.json")
    ds = drafts.load_drafts(tmp_path / "drafts", tmp_path / "none.json")
    assert ds.items["math:B1.1.1.1.1"]["file"] == "math.a.json"
    assert len(ds.problems) == 1 and "math.b.json" in ds.problems[0] and "skipped" in ds.problems[0]


def test_model_id_comes_from_drafted_by():
    assert drafts.model_id({"drafted_by": "claude-x-1 (offline draft, not yet reviewed)"}) == "claude-x-1"
    assert drafts.model_id({}) == "unknown"


# ----------------------------------------------------------------------------- merging


def test_merge_adds_every_passing_draft_as_unreviewed_with_the_model_as_source():
    doc = {"schema_version": 1, "subject": "math", "settings": {}}
    result = drafts.merge_drafts(doc, draft_set())
    assert (result.added, result.kept, result.removed, result.failing) == (16, 0, 0, [])
    assert result.indicators == 6
    for code, (_, settings) in SYNTHETIC.items():
        entries = doc["settings"][canonical_id(code)]
        assert [e["text"] for e in entries] == settings
        for e in entries:
            assert e == {
                "text": e["text"],
                "lang": "en",
                "reviewed": False,
                "source": MODEL,
                "reviewer": None,
                "reviewed_at": None,
            }


def test_merge_is_idempotent_and_keeps_teacher_decisions(tmp_path):
    doc = {"schema_version": 1, "subject": "math", "settings": {}}
    ds = draft_set()
    drafts.merge_drafts(doc, ds)
    a, b, c = SYNTHETIC["B3.1.4.1.1"][1]
    entries = doc["settings"]["math:B3.1.4.1.1"]
    entries[0].update(reviewed=True, reviewer="KA01", reviewed_at="2026-11-20")
    edited = "Kwame has two GH₵ 5 notes and one GH₵ 1 coin in Keta. How much money does he have?"
    entries[1] = {
        "text": edited,
        "lang": "en",
        "reviewed": True,
        "source": "teacher-edit",
        "reviewer": "KA01",
        "reviewed_at": "2026-11-20",
        "edited_from": text_hash(b),
    }
    doc["rejected"] = {"math:B3.1.4.1.1": [{"text": c, "reviewer": "KA01", "reviewed_at": "2026-11-20"}]}
    doc["settings"]["math:B3.1.4.1.1"] = entries[:2]
    doc["settings"]["math:B1.1.1.1.1"].append({"text": "Twi text.", "lang": "tw", "reviewed": True})
    before = json.dumps(doc, sort_keys=True)
    result = drafts.merge_drafts(doc, ds)
    assert json.dumps(doc, sort_keys=True) == before
    assert (result.added, result.removed, result.failing) == (0, 0, [])
    assert result.kept_reviewed == 3
    assert [e["text"] for e in doc["settings"]["math:B3.1.4.1.1"]] == [a, edited]  # the rejected c did not come back
    # save and load, then merge again: still nothing changes
    path = tmp_path / "math.json"
    save_contexts(doc, path)
    loaded = load_contexts(path)
    drafts.merge_drafts(loaded, ds)
    assert json.dumps(loaded["settings"], sort_keys=True) == json.dumps(doc["settings"], sort_keys=True)


def test_rewritten_drafts_replace_the_old_unreviewed_text():
    doc = {"schema_version": 1, "subject": "math", "settings": {}}
    drafts.merge_drafts(doc, draft_set())
    data = {code: (label, list(settings)) for code, (label, settings) in SYNTHETIC.items()}
    old = data["B2.1.2.1.1"][1][0]
    new = "Yaw sells 12 balls of kenkey in Tema. Then he sells 30 more. How many does he sell?"
    data["B2.1.2.1.1"][1][0] = new
    result = drafts.merge_drafts(doc, draft_set(data))
    assert (result.added, result.removed, result.kept) == (1, 1, 15)
    texts = [e["text"] for e in doc["settings"]["math:B2.1.2.1.1"]]
    assert new in texts and old not in texts


def test_failing_drafts_are_not_merged_and_are_reported():
    data = {code: (label, list(settings)) for code, (label, settings) in SYNTHETIC.items()}
    data["B4.1.2.1.1"][1][1] = "A crate holds 12 bottles of Coke. Dede in Nungua has 6 crates. How many bottles?"
    data["B4.1.2.1.1"][1][2] = data["B4.1.2.1.1"][1][0].replace("6 full", "7 full")  # a near copy of the first
    doc = {"schema_version": 1, "subject": "math", "settings": {}}
    result = drafts.merge_drafts(doc, draft_set(data))
    assert result.added == 14 and len(result.failing) == 2
    assert {why[0].check for _, _, why in result.failing} == {"banned", "duplicate"}
    assert len(doc["settings"]["math:B4.1.2.1.1"]) == 1
    assert any("NOT merged" in line for line in result.lines())


def test_merge_reports_against_the_structure_file():
    doc = {"schema_version": 1, "subject": "math", "settings": {}}
    structure = [c for c in SYNTHETIC if c != "B6.1.4.1.1"] + ["B6.9.9.9.9"]
    result = drafts.merge_drafts(doc, draft_set(), structure)
    assert result.not_in_structure == ["B6.1.4.1.1"]
    assert result.no_drafts == ["B6.9.9.9.9"]
    lines = "\n".join(result.lines())
    assert "not in the structure file: B6.1.4.1.1" in lines and "no draft: B6.9.9.9.9" in lines


def test_merge_never_marks_anything_reviewed():
    doc = {"schema_version": 1, "subject": "math", "settings": {}}
    drafts.merge_drafts(doc, draft_set())
    assert not any(e["reviewed"] for entries in doc["settings"].values() for e in entries)


def test_load_structure_codes_reads_a_list_or_a_wrapped_list(tmp_path):
    path = tmp_path / "math.structure.json"
    path.write_text(json.dumps([{"code": "B1.1.1.1.1"}, {"code": "B1.1.1.1.2"}]), encoding="utf-8")
    assert drafts.load_structure_codes(path) == ["B1.1.1.1.1", "B1.1.1.1.2"]
    path.write_text(json.dumps({"indicators": [{"code": "B2.1.1.1.1"}]}), encoding="utf-8")
    assert drafts.load_structure_codes(path) == ["B2.1.1.1.1"]
    assert drafts.load_structure_codes(tmp_path / "missing.json") is None


def test_current_labels_prefers_reviewed_labels(tmp_path):
    reviews = tmp_path / "label_reviews.json"
    reviews.write_text(
        json.dumps({"B1.1.1.1.1": {"short_label": "Count to 20", "reviewer": "KA01", "reviewed_at": "2026-11-20"}}),
        encoding="utf-8",
    )
    labels = drafts.current_labels(draft_set(), reviews)
    assert labels["B1.1.1.1.1"] == {
        "short_label": "Count to 20",
        "reviewed": True,
        "reviewer": "KA01",
        "reviewed_at": "2026-11-20",
    }
    assert labels["B2.1.2.1.1"] == {
        "short_label": SYNTHETIC["B2.1.2.1.1"][0],
        "reviewed": False,
        "reviewer": None,
        "reviewed_at": None,
    }


# ----------------------------------------------------------------------------- the committed files (the PRD gate)


@pytest.fixture(scope="module")
def committed():
    return load_contexts()


def test_every_committed_setting_passes_the_automatic_checks(committed):
    """PRD Phase 8 gate: the checks run on every setting in data/contexts/math.json."""
    report = checks.check_contexts(committed["settings"])
    assert report.total >= 800
    assert report.failed == 0, [(i, t, [str(f) for f in why]) for i, t, why, *_ in report.failures]


def test_committed_file_has_the_documented_schema(committed):
    assert committed["schema_version"] == 1 and committed["subject"] == "math"
    keys = {"text", "lang", "reviewed", "source", "reviewer", "reviewed_at"}
    for indicator, entries in committed["settings"].items():
        assert indicator.startswith("math:B") and checks.grade_of(indicator) in range(1, 7)
        assert 1 <= len(entries) <= 3
        for e in entries:
            assert keys <= set(e), (indicator, e)
            assert e["lang"] == "en"
            assert e["source"].startswith("draft:") or e["source"] == "teacher-edit"
            if not e["reviewed"]:
                assert e["reviewer"] is None and e["reviewed_at"] is None and e["source"].startswith("draft:")
            else:
                assert e["reviewer"] and e["reviewed_at"]


def test_committed_file_matches_the_structure_file_and_the_corrections():
    codes = drafts.load_structure_codes()
    if codes is None:
        pytest.skip("data/curriculum/math.structure.json is not there yet")
    doc = load_contexts()
    have = {k.split(":")[1] for k in doc["settings"]}
    assert have == set(codes)
    key_map = drafts.load_draft_key_map()
    if key_map:
        for printed, corrected in key_map.items():
            assert canonical_id(corrected) in doc["settings"], (printed, corrected)
            if printed != corrected:
                assert canonical_id(printed) not in doc["settings"], printed


def test_committed_drafts_are_in_the_file_and_nothing_is_reviewed_without_a_reviewer(committed):
    ds = drafts.load_drafts()
    assert ds.mapping == "corrections", "data/curriculum/corrections.json should exist by now"
    assert ds.n_settings == 828 and len(ds.items) == 276
    assert set(ds.items) == set(committed["settings"])
    for indicator, item in ds.items.items():
        on_file = {text_hash(e["text"]) for e in committed["settings"][indicator]}
        on_file |= {e.get("edited_from") for e in committed["settings"][indicator]}
        on_file |= {text_hash(e["text"]) for e in committed.get("rejected", {}).get(indicator, [])}
        for text in item["settings"]:
            assert text_hash(text) in on_file, (indicator, text)


def test_committed_short_labels_pass_their_checks():
    labels = drafts.load_drafts().labels()
    bad = {
        code: [str(f) for f in checks.check_label(label)] for code, label in labels.items() if checks.check_label(label)
    }
    assert bad == {}
    assert all(len(label) <= 60 for label in labels.values())
