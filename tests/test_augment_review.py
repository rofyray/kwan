"""Phase 7 teacher review: the sheet goes out, decisions come back (task 7.6). Synthetic graph only."""

import csv
import json

import networkx as nx
import pytest

from kwan.graph import _cli_augment as cli
from kwan.graph import augment, review
from kwan.graph.schema import save_graph
from tests.test_augment import EDGES, M, _aug, _doc, make_graph


@pytest.fixture
def G():
    return make_graph()


@pytest.fixture
def doc():
    return _doc(
        _aug("B1.1.1.1.2", "B2.1.2.1.1", 0.93),
        _aug("B1.1.1.1.1", "B2.3.1.1.1", 0.88),
        _aug("B1.1.2.1.1", "B2.2.1.1.1", 0.81),
        _aug("B1.3.1.1.1", "B2.3.1.1.1", 0.80),
    )


def _rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=review.COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def test_export_without_full_text_has_labels_blank_descriptions_and_blank_decisions(G, doc, tmp_path):
    path, n, has_text = review.export_review(G, doc, out=tmp_path / "sheet.csv", texts={})
    rows = _rows(path)
    assert (n, has_text) == (4, False) and len(rows) == 4
    assert tuple(rows[0]) == review.COLUMNS
    assert [r["probability"] for r in rows] == ["0.9300", "0.8800", "0.8100", "0.8000"]  # most likely first
    first = rows[0]
    assert first["a_id"] == M("B1.1.1.1.2") and first["b_id"] == M("B2.1.2.1.1")
    assert first["a_label"] == "Count objects to 100" and first["b_label"] == "Add numbers to 100"
    assert (first["a_grade"], first["b_grade"]) == ("1", "2")
    assert all(r["a_description"] == r["b_description"] == r["decision"] == r["reviewer"] == "" for r in rows)


def test_export_fills_descriptions_only_when_the_full_text_is_on_this_machine(G, doc, tmp_path):
    texts = {"B1.1.1.1.2": {"description": "A made-up description", "exemplar": ""}}
    path, _, has_text = review.export_review(G, doc, out=tmp_path / "sheet.csv", texts=texts)
    first = _rows(path)[0]
    assert has_text and first["a_description"] == "A made-up description"
    assert first["b_description"] == "Add numbers to 100"  # no text for this one: its label stands in


def test_a_sheet_with_descriptions_is_refused_inside_the_committed_data_directory(G, doc, tmp_path, monkeypatch):
    monkeypatch.setattr(review, "data_dir", lambda: tmp_path / "data")
    texts = {"B1.1.1.1.2": {"description": "A made-up description", "exemplar": ""}}
    with pytest.raises(ValueError, match="build/"):
        review.export_review(G, doc, out=tmp_path / "data" / "sheet.csv", texts=texts)
    path, _, _ = review.export_review(G, doc, out=tmp_path / "data" / "sheet.csv", texts={})  # labels only is fine
    assert path.is_file()


def test_default_sheet_path_is_under_build():
    assert review.default_sheet_path().parts[-3:] == ("build", "review", "math.augmented.review.csv")


def test_export_can_draw_a_fixed_random_sample(G, doc, tmp_path):
    one = _rows(review.export_review(G, doc, out=tmp_path / "a.csv", texts={}, sample=2, seed=7)[0])
    two = _rows(review.export_review(G, doc, out=tmp_path / "b.csv", texts={}, sample=2, seed=7)[0])
    assert len(one) == 2 and one == two


def test_round_trip_records_decisions_with_reviewer_and_date(G, doc, tmp_path):
    path, _, _ = review.export_review(G, doc, out=tmp_path / "sheet.csv", texts={})
    rows = _rows(path)
    rows[0].update(decision="accepted", reviewer="teacher-07")
    rows[1].update(decision="Rejected", reviewer="teacher-07")
    rows[2].update(decision="yes")  # no reviewer in the cell: the default reviewer is used
    _write(path, rows)
    new_doc, report = review.import_review(G, doc, path, reviewer="teacher-09", at="2026-11-20")
    assert (report.accepted, report.rejected, report.blank, report.refused) == (2, 1, 1, [])
    assert report.counts == {"pending": 1, "accepted": 2, "rejected": 1}
    by_pair = {(e["source"], e["target"]): e["provenance"]["review"] for e in new_doc["edges"]}
    assert by_pair[(M("B1.1.1.1.2"), M("B2.1.2.1.1"))] == {"status": "accepted", "by": "teacher-07", "at": "2026-11-20"}
    assert by_pair[(M("B1.1.1.1.1"), M("B2.3.1.1.1"))] == {"status": "rejected", "by": "teacher-07", "at": "2026-11-20"}
    assert by_pair[(M("B1.1.2.1.1"), M("B2.2.1.1.1"))]["by"] == "teacher-09"
    assert by_pair[(M("B1.3.1.1.1"), M("B2.3.1.1.1"))] == {"status": "pending", "by": None, "at": None}
    assert augment.review_counts(doc) == {"pending": 4, "accepted": 0, "rejected": 0}  # the input is not changed

    saved = augment.save_augmented(new_doc, tmp_path / "math.augmented.json")
    assert augment.review_counts(augment.load_augmented(saved)) == report.counts
    again, second = review.import_review(G, augment.load_augmented(saved), path, reviewer="teacher-09", at="2026-12-01")
    assert (second.accepted, second.rejected, second.unchanged) == (0, 0, 3)  # importing twice changes nothing
    assert again["edges"] == augment.load_augmented(saved)["edges"]


def test_import_accepts_bare_codes(G, doc, tmp_path):
    sheet = tmp_path / "sheet.csv"
    _write(sheet, [{"a_id": "B1.1.1.1.2", "b_id": "B2.1.2.1.1", "decision": "accepted", "reviewer": "t-1"}])
    new_doc, report = review.import_review(G, doc, sheet)
    assert report.accepted == 1 and augment.accepted_edges(new_doc)[0]["source"] == M("B1.1.1.1.2")


def test_import_checks_every_row_before_applying_anything(G, doc, tmp_path):
    sheet = tmp_path / "sheet.csv"
    _write(
        sheet,
        [
            {"a_id": M("B1.1.1.1.2"), "b_id": M("B2.1.2.1.1"), "decision": "accepted", "reviewer": "t-1"},
            {"a_id": "math:B9.9.9.9.9", "b_id": M("B2.1.2.1.1"), "decision": "accepted", "reviewer": "t-1"},
            {"a_id": M("B1.1.1.1.1"), "b_id": M("B4.1.2.1.1"), "decision": "accepted", "reviewer": "t-1"},
            {"a_id": M("B1.1.1.1.1"), "b_id": M("B2.3.1.1.1"), "decision": "maybe", "reviewer": "t-1"},
            {"a_id": M("B1.1.2.1.1"), "b_id": M("B2.2.1.1.1"), "decision": "rejected", "reviewer": ""},
        ],
    )
    with pytest.raises(review.ReviewError) as err:
        review.import_review(G, doc, sheet)
    problems = err.value.problems
    assert len(problems) == 4
    assert "row 3: unknown indicator 'math:B9.9.9.9.9'" in problems
    assert any("row 4" in p and "not a proposed edge" in p for p in problems)
    assert any("row 5" in p and "'maybe'" in p for p in problems)
    assert any("row 6" in p and "reviewer id" in p for p in problems)
    assert augment.review_counts(doc) == {"pending": 4, "accepted": 0, "rejected": 0}


def test_import_refuses_a_sheet_without_the_needed_columns(G, doc, tmp_path):
    sheet = tmp_path / "sheet.csv"
    sheet.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(review.ReviewError, match="no column 'a_id'"):
        review.import_review(G, doc, sheet)


def test_import_refuses_an_accepted_edge_that_would_close_a_cycle(G, tmp_path):
    # B1.1.2.1.1 -> B2.1.2.1.1 -> B3.1.2.1.1 are rule edges; the two proposals below cannot both be accepted
    doc = _doc(
        _aug("B3.1.2.1.1", "B2.2.1.1.1", 0.9),
        _aug("B2.2.1.1.1", "B2.1.2.1.1", 0.7),
        _aug("B1.1.1.1.1", "B2.3.1.1.1", 0.6),
    )
    sheet = tmp_path / "sheet.csv"
    _write(
        sheet,
        [{"a_id": e["source"], "b_id": e["target"], "decision": "accepted", "reviewer": "t-1"} for e in doc["edges"]],
    )
    new_doc, report = review.import_review(G, doc, sheet, at="2026-11-20")
    assert report.accepted == 2
    assert report.refused == [
        {"source": M("B2.2.1.1.1"), "target": M("B2.1.2.1.1"), "reason": "accepting it would close a cycle"}
    ]
    status = {(e["source"], e["target"]): e["provenance"]["review"]["status"] for e in new_doc["edges"]}
    assert status[(M("B2.2.1.1.1"), M("B2.1.2.1.1"))] == "pending"  # the refused edge stays as it was
    H, merge = augment.merge_accepted(G, new_doc)
    assert nx.is_directed_acyclic_graph(H) and len(merge.inserted) == 2 and cli.check_augmented(G, new_doc) == []


def test_import_refuses_an_edge_against_a_rule_edge_path(G, tmp_path):
    doc = _doc(_aug("B4.1.2.1.1", "B1.1.2.1.1", 0.9))
    sheet = tmp_path / "sheet.csv"
    _write(sheet, [{"a_id": M("B4.1.2.1.1"), "b_id": M("B1.1.2.1.1"), "decision": "accepted", "reviewer": "t-1"}])
    new_doc, report = review.import_review(G, doc, sheet)
    assert report.accepted == 0 and len(report.refused) == 1
    assert augment.accepted_edges(new_doc) == []


def test_a_reviewer_can_change_an_earlier_decision(G, doc, tmp_path):
    sheet = tmp_path / "sheet.csv"
    row = {"a_id": M("B1.1.1.1.2"), "b_id": M("B2.1.2.1.1"), "decision": "accepted", "reviewer": "t-1"}
    _write(sheet, [row])
    first, _ = review.import_review(G, doc, sheet, at="2026-11-20")
    _write(sheet, [{**row, "decision": "rejected", "reviewer": "t-2"}])
    second, report = review.import_review(G, first, sheet, at="2026-11-21")
    assert report.rejected == 1 and report.counts == {"pending": 3, "accepted": 0, "rejected": 1}
    assert second["edges"][0]["provenance"]["review"] == {"status": "rejected", "by": "t-2", "at": "2026-11-21"}


def test_two_different_decisions_for_one_edge_in_one_sheet_are_an_error(G, doc, tmp_path):
    sheet = tmp_path / "sheet.csv"
    row = {"a_id": M("B1.1.1.1.2"), "b_id": M("B2.1.2.1.1"), "decision": "accepted", "reviewer": "t-1"}
    _write(sheet, [row, {**row, "decision": "rejected"}])
    with pytest.raises(review.ReviewError, match="two different decisions"):
        review.import_review(G, doc, sheet)


def test_acceptance_is_not_measured_until_a_teacher_reviews(doc):
    out = review.acceptance(doc)
    assert out["teacher_acceptance"] is None and out["reviewed"] == 0 and "not yet measured" in out["status"]
    reviewed = _doc(
        _aug("B1.1.1.1.2", "B2.1.2.1.1", 0.9, "accepted", "t-1"),
        _aug("B1.1.1.1.1", "B2.3.1.1.1", 0.8, "rejected", "t-1"),
        _aug("B1.1.2.1.1", "B2.2.1.1.1", 0.7, "accepted", "t-1"),
        _aug("B1.3.1.1.1", "B2.3.1.1.1", 0.6),
    )
    out = review.acceptance(reviewed)
    assert out["teacher_acceptance"] == pytest.approx(0.6667) and out["reviewed"] == 3
    assert out["status"].startswith("partly measured")


def test_cli_round_trip(G, doc, tmp_path, capsys):
    graph, aug, sheet = tmp_path / "math.graph.json", tmp_path / "math.augmented.json", tmp_path / "sheet.csv"
    save_graph(G, graph)
    augment.save_augmented(doc, aug)
    common = ["--graph", str(graph), "--augmented", str(aug)]
    full = ["--full-text", str(tmp_path / "none.json")]
    assert cli.main(["review-export", "--out", str(sheet), *common, *full]) == 0
    assert "short labels only" in capsys.readouterr().out
    rows = _rows(sheet)
    rows[0]["decision"] = "accepted"
    rows[3]["decision"] = "rejected"
    _write(sheet, rows)
    assert cli.main(["review-import", str(sheet), *common]) == 1  # no reviewer id anywhere
    assert "reviewer id" in capsys.readouterr().err
    assert augment.review_counts(augment.load_augmented(aug))["pending"] == 4
    assert cli.main(["review-import", str(sheet), "--reviewer", "teacher-03", "--date", "2026-11-20", *common]) == 0
    saved = augment.load_augmented(aug)
    assert augment.review_counts(saved) == {"pending": 2, "accepted": 1, "rejected": 1}
    assert saved["edges"][0]["provenance"]["review"] == {"status": "accepted", "by": "teacher-03", "at": "2026-11-20"}
    assert json.loads(aug.read_text(encoding="utf-8"))["run"] == {"provider": "fake"}  # the run record is kept
    assert cli.main(["review-export", "--out", str(sheet), "--status", "all", *common, *full]) == 0
    assert [r["decision"] for r in _rows(sheet)] == ["accepted", "", "", "rejected"]
    assert len(EDGES) == G.number_of_edges()
