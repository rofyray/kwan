"""`kwan context ...` commands and the `contexts` results block, on a synthetic data folder."""

import csv
import json

import pytest

from kwan import cli
from kwan.context import coverage
from kwan.context.settings import load_contexts
from tests.test_context_support import SYNTHETIC, draft_set, make_world


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A synthetic data folder that the CLI sees as `data/` (through KWAN_DATA_DIR)."""
    w = make_world(tmp_path)
    monkeypatch.setenv("KWAN_DATA_DIR", str(w.data))
    drafts_dir = w.data / "contexts" / "drafts"
    drafts_dir.mkdir(parents=True)
    indicators = {code: {"short_label": label, "settings": list(s)} for code, (label, s) in SYNTHETIC.items()}
    meta = {"subject": "math", "drafted_by": "test-model (offline draft)", "drafted_on": "2026-11-10"}
    (drafts_dir / "math.test.json").write_text(json.dumps({"meta": meta, "indicators": indicators}), encoding="utf-8")
    (w.data / "curriculum").mkdir()
    (w.data / "curriculum" / "corrections.json").write_text(json.dumps({"draft_keys": {}}), encoding="utf-8")
    structure = [{"code": code} for code in SYNTHETIC]
    (w.data / "curriculum" / "math.structure.json").write_text(json.dumps(structure), encoding="utf-8")
    return w


def run(capsys, *argv):
    code = cli.main(["context", *argv])
    out = capsys.readouterr()
    return code, out.out, out.err


def test_context_is_registered_with_its_commands():
    parser, skipped = cli.build_parser()
    assert "kwan.context._cli" not in skipped
    helptext = parser.format_help()
    assert "context" in helptext
    sub = next(a for a in parser._subparsers._group_actions if "context" in a.choices).choices["context"]
    commands = next(a for a in sub._subparsers._group_actions).choices
    assert set(commands) == {"check", "merge-drafts", "review-export", "review-import", "translate-draft", "coverage"}


def test_check_prints_a_per_check_summary_and_passes_on_clean_data(world, capsys):
    code, out, _ = run(capsys, "check")
    assert code == 0
    assert "16 settings checked: 16 pass, 0 fail (100.0% pass)" in out
    for name in ("form", "length", "em_dash", "reading_level", "banned", "currency", "ghana_term", "duplicate"):
        assert f"  {name:<14}" in out
    assert "6 short labels checked: 6 pass, 0 fail" in out


def test_check_fails_and_names_the_setting_when_one_is_bad(world, capsys):
    doc = load_contexts(world.contexts)
    doc["settings"]["math:B1.1.1.1.1"][0]["text"] = "Kwame buys 4 tins of Milo in Ho. How much does he pay?"
    world.contexts.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    code, out, _ = run(capsys, "check")
    assert code == 1
    assert "16 settings checked: 15 pass, 1 fail" in out
    assert "banned" in out and "Milo" in out and "math:B1.1.1.1.1 [0]" in out


def test_check_can_look_at_the_draft_files_and_the_bank(world, capsys):
    code, out, _ = run(capsys, "check", "--drafts")
    assert code == 0 and "draft files (corrections mapping)" in out and "16 settings checked" in out
    code, out, _ = run(capsys, "check", "--bank")
    assert "v1 bank" in out and "21 settings checked" in out


def test_merge_drafts_is_rerunnable_and_marks_nothing_reviewed(world, capsys):
    world.contexts.unlink()
    code, out, _ = run(capsys, "merge-drafts")
    assert code == 0
    assert "mapping of draft keys: corrections" in out and "drafts added: 16" in out
    assert "Nothing was marked as reviewed" in out
    doc = load_contexts(world.contexts)
    assert len(doc["settings"]) == 6 and not any(e["reviewed"] for v in doc["settings"].values() for e in v)
    code, out, _ = run(capsys, "merge-drafts")
    assert code == 0 and "drafts added: 0, already there: 16" in out
    (world.data / "curriculum" / "corrections.json").unlink()
    code, out, _ = run(capsys, "merge-drafts", "--dry-run")
    assert "provisional" in out and "run this again when it exists" in out and "nothing was written" in out


def test_export_then_import_through_the_cli(world, capsys, tmp_path):
    out_dir = tmp_path / "sheets"
    code, out, _ = run(capsys, "review-export", "--out-dir", str(out_dir), "--batch-size", "10")
    assert code == 0
    assert "waiting for review: 16 rows and 6 short labels, in 2 sheets" in out
    path = next(out_dir.glob("math.review.batch-01-of-02.csv"))
    first = SYNTHETIC["B1.1.1.1.1"][1][0]
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert any(r["draft_text"] == first for r in rows)
    for row in rows:  # a teacher accepts the first setting of B1.1.1.1.1
        if row["kind"] == "setting" and row["draft_text"] == first:
            row.update(decision="accept", reviewer_id="KA01", region_taught="Volta")
    filled = tmp_path / "filled.csv"
    with filled.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    code, out, _ = run(capsys, "review-import", str(filled), "--date", "2026-11-20")
    assert code == 0, out
    assert "settings: 1 accepted, 0 edited, 0 rejected" in out
    entry = world.entry("B1.1.1.1.1", first)
    assert entry["reviewed"] is True and entry["reviewer"] == "KA01" and entry["reviewed_at"] == "2026-11-20"
    code, out, _ = run(capsys, "review-import", str(filled), "--date", "2026-11-21")
    assert code == 0 and "already applied (no change): 1" in out
    code, _, err = run(capsys, "review-import", str(tmp_path / "missing.csv"))
    assert code == 2 and "missing.csv" in err


def test_translate_draft_without_a_key_explains(world, capsys, monkeypatch):
    monkeypatch.delenv("KHAYA_API_KEY", raising=False)
    doc = load_contexts(world.contexts)
    doc["settings"]["math:B1.1.1.1.1"][0].update(reviewed=True, reviewer="KA01", reviewed_at="2026-11-20")
    world.contexts.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    code, _, err = run(capsys, "translate-draft", "--lang", "tw")
    assert code == 2 and "KHAYA_API_KEY" in err
    assert not (world.translations / "tw.json").exists()


def test_coverage_block_is_honest_and_is_written_under_the_contexts_key(world, capsys, tmp_path):
    results = tmp_path / "eval_results.json"
    results.write_text(json.dumps({"other": {"keep": 1}}), encoding="utf-8")
    code, out, _ = run(capsys, "coverage", "--out", str(results))
    assert code == 0
    assert "indicators: 6 (from the structure file)" in out
    assert "indicators with a reviewed setting: 0 of 6 (0.0%)" in out
    assert "translations tw: 0 unreviewed drafts, 0 reviewed, 0 of 3 Basic 1 to 3 indicators covered (0.0%)" in out
    written = json.loads(results.read_text(encoding="utf-8"))
    assert written["other"] == {"keep": 1}
    block = written["contexts"]
    assert block["indicators"] == 6 and block["settings"] == 16 and block["draft_settings_unreviewed"] == 16
    assert block["settings_passing_automatic_checks"] == 16 and block["settings_failing_automatic_checks"] == 0
    assert block["reviewed_settings"] == 0 and block["indicators_with_reviewed_setting"] == 0
    assert block["reviewed_coverage_pct"] == 0.0 and block["reviewed_labels"] == 0
    assert block["short_labels"] == 6 and block["draft_sources"] == ["draft:test-model"]
    assert set(block["translations"]) == {"tw", "ee", "gaa"}
    assert block["translations"]["ee"]["coverage_pct"] == 0.0


def test_coverage_counts_reviews_once_they_exist(world):
    doc = load_contexts(world.contexts)
    doc["settings"]["math:B1.1.1.1.1"][0].update(reviewed=True, reviewer="KA01", reviewed_at="2026-11-20")
    doc["settings"]["math:B2.1.2.1.1"][0].update(reviewed=True, reviewer="YB02", reviewed_at="2026-11-20")
    world.contexts.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    block = coverage.contexts_block(contexts=world.contexts, drafts=draft_set(), label_reviews=world.labels)
    assert block["reviewed_settings"] == 2 and block["indicators_with_reviewed_setting"] == 2
    assert block["reviewed_coverage_pct"] == 33.3 and block["reviewers"] == 2
    assert block["draft_settings_unreviewed"] == 14


def test_committed_results_block_matches_the_committed_files():
    """The numbers in data/eval_results.json come from the files, never from a hand."""
    from kwan.eval import results

    stored = results.load().get("contexts")
    if stored is None:
        pytest.skip("the contexts block has not been written yet")
    fresh = coverage.contexts_block()
    for key in ("indicators", "settings", "reviewed_settings", "reviewed_labels", "settings_passing_automatic_checks"):
        assert stored[key] == fresh[key], key
    assert stored["reviewed_settings"] == sum(
        1 for v in load_contexts()["settings"].values() for e in v if e["reviewed"]
    )
