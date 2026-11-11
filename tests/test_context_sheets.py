"""Teacher review sheets (PRD 8.4): export, fill in, import, and import again."""

import csv
import io
import json

import pytest

from kwan.context import checks, sheets
from kwan.context.settings import canonical_id, save_translations, text_hash
from tests.test_context_support import MODEL, SYNTHETIC, make_world

N_SETTINGS = sum(len(s) for _, s in SYNTHETIC.values())  # 16
N_LABELS = len(SYNTHETIC)  # 6


def export_all(world, **kw):
    """Export everything into one sheet and return (path, rows)."""
    result = sheets.export_review(out_dir=world.out, batch_size=1000, **world.kwargs(), **kw)
    assert len(result.files) == 1
    path = result.files[0][0]
    return path, sheets.read_sheet(path)


def fill(path, decisions, out=None, **defaults):
    """Fill in a sheet: `decisions` maps draft text -> dict of column values. Returns the new path."""
    rows = sheets.read_sheet(path)
    for row in rows:
        update = decisions.get(row["draft_text"])
        if update:
            row.update({"reviewer_id": "KA01", "region_taught": "Volta", **defaults, **update})
    out = out or path.with_name("filled.csv")
    with out.open("w", encoding="utf-8", newline="") as f:  # no byte order mark, like a Google Sheets download
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return out


def texts(code):
    return SYNTHETIC[code][1]


# ----------------------------------------------------------------------------- export


def test_export_writes_utf8_with_bom_and_the_documented_columns(tmp_path):
    world = make_world(tmp_path)
    path, rows = export_all(world)
    raw = path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    assert path.parent == world.out and path.name == "math.review.batch-01-of-01.csv"
    header = next(csv.reader(io.StringIO(raw.decode("utf-8-sig"))))
    assert tuple(header) == sheets.SETTING_COLUMNS
    assert header[:2] == ["kind", "indicator"] and header[-1] == "row_id"
    assert "GH₵ 15" in raw.decode("utf-8-sig")
    assert b"\r\n" in raw


def test_export_has_one_row_per_draft_setting_and_one_per_short_label(tmp_path):
    world = make_world(tmp_path)
    _, rows = export_all(world)
    settings = [r for r in rows if r["kind"] == "setting"]
    labels = [r for r in rows if r["kind"] == "label"]
    assert (len(settings), len(labels)) == (N_SETTINGS, N_LABELS)
    assert sorted(r["draft_text"] for r in settings) == sorted(t for _, s in SYNTHETIC.values() for t in s)
    assert {r["indicator"]: r["draft_text"] for r in labels} == {c: label for c, (label, _) in SYNTHETIC.items()}
    for r in rows:
        assert r["short_label"] == SYNTHETIC[r["indicator"]][0]
        assert r["class"] == f"Basic {checks.grade_of(r['indicator'])}"
        assert r["decision"] == r["edited_text"] == r["reviewer_id"] == r["region_taught"] == r["note"] == ""
    assert len({r["row_id"] for r in rows}) == len(rows)
    # the label row of an indicator comes right before its settings
    for i, r in enumerate(rows):
        if r["kind"] == "label":
            assert rows[i + 1]["kind"] == "setting" and rows[i + 1]["indicator"] == r["indicator"]


def test_exemplar_summary_is_filled_only_when_the_full_text_file_exists(tmp_path):
    world = make_world(tmp_path)
    _, rows = export_all(world, full_text=tmp_path / "missing.json")
    assert {r["exemplar_summary"] for r in rows} == {""}

    full = tmp_path / "math.full.json"
    made_up = [
        {
            "code": "B1.1.1.1.1",
            "description": "Made-up description for the test.",
            "exemplars": [{"text": "E1 " * 200}],
        },
        {"code": "B2.1.2.1.1", "description": "Another made-up description.", "exemplars": []},
    ]
    full.write_text(json.dumps(made_up), encoding="utf-8")
    result = sheets.export_review(out_dir=world.out, batch_size=1000, full_text=full, **world.kwargs())
    assert result.with_exemplars and any("NaCCA" in line for line in result.lines())
    rows = sheets.read_sheet(result.files[0][0])
    by_code = {r["indicator"]: r["exemplar_summary"] for r in rows}
    assert by_code["B1.1.1.1.1"].startswith("Made-up description for the test. Example: E1")
    assert len(by_code["B1.1.1.1.1"]) <= 300 and by_code["B1.1.1.1.1"].endswith("...")
    assert by_code["B2.1.2.1.1"] == "Another made-up description."
    assert by_code["B4.1.2.1.1"] == ""
    off = sheets.export_review(out_dir=world.out, batch_size=1000, full_text=full, exemplars=False, **world.kwargs())
    assert not off.with_exemplars
    assert {r["exemplar_summary"] for r in sheets.read_sheet(off.files[0][0])} == {""}


def test_batches_cover_every_row_once_and_never_split_an_indicator(tmp_path):
    world = make_world(tmp_path)
    result = sheets.export_review(out_dir=world.out, batch_size=5, all_batches=True, **world.kwargs())
    assert result.batches_total == len(result.files) == 3
    assert (result.pending_rows, result.pending_labels) == (N_SETTINGS, N_LABELS)
    seen = []
    for path, n_settings, n_labels in result.files:
        rows = sheets.read_sheet(path)
        assert (n_settings, n_labels) == (
            sum(r["kind"] == "setting" for r in rows),
            sum(r["kind"] == "label" for r in rows),
        )
        seen.append({r["indicator"] for r in rows})
        assert f"-of-{result.batches_total:02d}" in path.name
    assert sum(n for _, n, _ in result.files) == N_SETTINGS
    assert set.union(*seen) == set(SYNTHETIC)
    assert sum(len(s) for s in seen) == len(SYNTHETIC)  # no indicator in two sheets
    # the first sheet mixes classes instead of starting with Basic 1 only
    first = sheets.read_sheet(result.files[0][0])
    assert len({r["class"] for r in first}) >= 2
    # one sheet on request, and the order is the same every time
    one = sheets.export_review(out_dir=tmp_path / "again", batch=2, batch_size=5, **world.kwargs())
    assert [p.name for p, _, _ in one.files] == ["math.review.batch-02-of-03.csv"]
    assert sheets.read_sheet(one.files[0][0]) == sheets.read_sheet(result.files[1][0])
    none = sheets.export_review(out_dir=tmp_path / "again", batch=9, batch_size=5, **world.kwargs())
    assert none.files == [] and "nothing to export" in none.lines()


def test_export_can_be_narrowed_to_a_class_or_a_kind(tmp_path):
    world = make_world(tmp_path)
    _, rows = export_all(world, grades=[3, 4])
    assert {r["class"] for r in rows} == {"Basic 3", "Basic 4"}
    _, rows = export_all(world, kinds=("setting",))
    assert {r["kind"] for r in rows} == {"setting"} and len(rows) == N_SETTINGS
    _, rows = export_all(world, kinds=("label",))
    assert {r["kind"] for r in rows} == {"label"} and len(rows) == N_LABELS


# ----------------------------------------------------------------------------- round trip


def test_an_untouched_sheet_reviews_nothing(tmp_path):
    world = make_world(tmp_path)
    path, rows = export_all(world)
    before = world.contexts.read_bytes()
    result = sheets.import_review(path, **world.kwargs())
    assert result.ok and result.pending == len(rows) and result.applied == {} and result.unchanged == 0
    assert world.contexts.read_bytes() == before
    assert not world.labels.exists()
    assert not any(e["reviewed"] for entries in world.doc()["settings"].values() for e in entries)


def test_round_trip_accept_edit_reject_and_blank(tmp_path):
    world = make_world(tmp_path)
    path, _ = export_all(world)
    a, b, c = texts("B3.1.4.1.1")
    edited = "Kwame has two GH₵ 5 notes and one GH₵ 1 coin in Keta. How much money does he have?"
    label = SYNTHETIC["B3.1.4.1.1"][0]
    other_label = SYNTHETIC["B1.1.1.1.1"][0]
    filled = fill(
        path,
        {
            a: {"decision": "accept"},
            b: {"decision": "Edit", "edited_text": "  " + edited + " "},
            c: {"decision": "REJECT", "note": "The price of koko is too low for Hohoe."},
            label: {"decision": "accepted"},
            other_label: {"decision": "edit", "edited_text": "Count up to 20 objects"},
        },
    )
    result = sheets.import_review(filled, date="2026-11-20", **world.kwargs())
    assert result.ok, result.lines()
    assert (result.count("setting", "accept"), result.count("setting", "edit"), result.count("setting", "reject")) == (
        1,
        1,
        1,
    )
    assert (result.count("label", "accept"), result.count("label", "edit")) == (1, 1)
    assert result.pending == N_SETTINGS + N_LABELS - 5

    doc = world.doc()
    entries = doc["settings"]["math:B3.1.4.1.1"]
    assert [e["text"] for e in entries] == [a, edited]
    accepted, changed = entries
    assert accepted == {
        "text": a,
        "lang": "en",
        "reviewed": True,
        "source": MODEL,
        "reviewer": "KA01",
        "reviewed_at": "2026-11-20",
        "region": "Volta",
    }
    assert changed["reviewed"] is True and changed["source"] == "teacher-edit"
    assert (changed["reviewer"], changed["reviewed_at"], changed["region"]) == ("KA01", "2026-11-20", "Volta")
    assert changed["edited_from"] == text_hash(b) and changed["edited_from_source"] == MODEL
    assert doc["rejected"]["math:B3.1.4.1.1"] == [
        {
            "text": c,
            "source": MODEL,
            "reviewer": "KA01",
            "reviewed_at": "2026-11-20",
            "region": "Volta",
            "note": "The price of koko is too low for Hohoe.",
        }
    ]
    # everything else is untouched and still unreviewed
    for indicator, items in doc["settings"].items():
        if indicator != "math:B3.1.4.1.1":
            assert all(not e["reviewed"] and e["reviewer"] is None for e in items)
    assert world.label_reviews() == {
        "B1.1.1.1.1": {
            "short_label": "Count up to 20 objects",
            "reviewer": "KA01",
            "reviewed_at": "2026-11-20",
            "region": "Volta",
            "draft_hash": text_hash(other_label),
        },
        "B3.1.4.1.1": {
            "short_label": label,
            "reviewer": "KA01",
            "reviewed_at": "2026-11-20",
            "region": "Volta",
            "draft_hash": text_hash(label),
        },
    }


def test_import_is_idempotent(tmp_path):
    world = make_world(tmp_path)
    path, _ = export_all(world)
    a, b, c = texts("B4.1.2.1.1")
    edited = "Musah plants 9 rows of yam near Yendi with 7 in each row. How many yams does he plant?"
    filled = fill(
        path,
        {
            a: {"decision": "accept"},
            b: {"decision": "edit", "edited_text": edited},
            c: {"decision": "reject"},
            SYNTHETIC["B4.1.2.1.1"][0]: {"decision": "accept"},
            SYNTHETIC["B5.1.3.1.1"][0]: {"decision": "reject"},
        },
    )
    first = sheets.import_review(filled, date="2026-11-20", **world.kwargs())
    assert first.ok and sum(sum(d.values()) for d in first.applied.values()) == 5
    snapshot = (world.contexts.read_bytes(), world.labels.read_bytes())
    stamp = (world.contexts.stat().st_mtime_ns, world.labels.stat().st_mtime_ns)

    second = sheets.import_review(filled, date="2027-01-01", **world.kwargs())  # a later day, same sheet
    assert second.ok and second.applied == {} and second.unchanged == 5
    assert (world.contexts.read_bytes(), world.labels.read_bytes()) == snapshot
    assert (world.contexts.stat().st_mtime_ns, world.labels.stat().st_mtime_ns) == stamp  # not even rewritten
    assert world.entry("B4.1.2.1.1", a)["reviewed_at"] == "2026-11-20"


def test_reviewed_and_rejected_rows_leave_the_next_export(tmp_path):
    world = make_world(tmp_path)
    path, _ = export_all(world)
    a, b, c = texts("B6.1.4.1.1")
    label = SYNTHETIC["B6.1.4.1.1"][0]
    rejected_label = SYNTHETIC["B2.1.2.1.1"][0]
    filled = fill(
        path,
        {
            a: {"decision": "accept"},
            b: {"decision": "reject"},
            label: {"decision": "accept"},
            rejected_label: {"decision": "reject"},
        },
    )
    assert sheets.import_review(filled, **world.kwargs()).ok
    _, rows = export_all(world)
    drafts_left = [r["draft_text"] for r in rows]
    assert a not in drafts_left and b not in drafts_left and c in drafts_left
    assert label not in drafts_left and rejected_label not in drafts_left
    assert len(rows) == N_SETTINGS + N_LABELS - 4
    # the reviewed label is now the label shown on the remaining rows of that indicator
    assert {r["short_label"] for r in rows if r["indicator"] == "B6.1.4.1.1"} == {label}
    assert world.doc()["rejected_labels"]["B2.1.2.1.1"][0]["short_label"] == rejected_label


# ----------------------------------------------------------------------------- refusals


def test_edited_text_that_fails_the_checks_is_refused_with_the_reason(tmp_path):
    world = make_world(tmp_path)
    path, _ = export_all(world)
    a, b, c = texts("B3.1.4.1.1")
    filled = fill(
        path,
        {
            a: {"decision": "edit", "edited_text": "Fati pays with a GH₵ 3 note in Wa. How much change does she get?"},
            b: {"decision": "edit", "edited_text": "Tom has 5 dollars. How much is left?"},
            c: {"decision": "accept"},
        },
    )
    result = sheets.import_review(filled, **world.kwargs())
    assert not result.ok and len(result.errors) == 2
    messages = {msg for _, _, msg in result.errors}
    assert any("the edited text fails the automatic checks" in m and 'no "GH₵ 3" note' in m for m in messages)
    assert any("dollars" in m and "ghana_term" in m for m in messages)
    assert all(indicator == "B3.1.4.1.1" for _, indicator, _ in result.errors)
    assert all(line >= 2 for line, _, _ in result.errors)
    assert any(line.startswith("REFUSED line") for line in result.lines())
    # the refused rows stay as unreviewed drafts; the good row is applied
    assert world.entry("B3.1.4.1.1", a)["reviewed"] is False
    assert world.entry("B3.1.4.1.1", b)["reviewed"] is False
    assert world.entry("B3.1.4.1.1", c)["reviewed"] is True
    assert result.count("setting", "accept") == 1


def test_edit_that_copies_another_setting_of_the_indicator_is_refused(tmp_path):
    world = make_world(tmp_path)
    path, _ = export_all(world)
    a, b, _ = texts("B1.1.1.1.1")
    result = sheets.import_review(fill(path, {b: {"decision": "edit", "edited_text": a}}), **world.kwargs())
    assert len(result.errors) == 1 and "duplicate" in result.errors[0][2]


@pytest.mark.parametrize(
    "update, expected",
    [
        ({"decision": "edit", "edited_text": ""}, "edited_text column is empty"),
        ({"decision": "maybe"}, "unknown decision"),
        ({"decision": "accept", "reviewer_id": ""}, "reviewer_id must be a short code"),
        ({"decision": "accept", "reviewer_id": "ama@example.org"}, "not a name or an email address"),
        ({"decision": "accept", "reviewer_id": "Ama Mensah"}, "reviewer_id must be a short code"),
        ({"decision": "accept", "row_id": "s|math:B1.1.1.1.1"}, "row_id"),
        ({"decision": "accept", "row_id": "s|math:B1.1.1.1.1|0000000000000000"}, "no longer in the settings file"),
    ],
)
def test_rows_that_cannot_be_applied_are_refused(tmp_path, update, expected):
    world = make_world(tmp_path)
    path, _ = export_all(world)
    a = texts("B1.1.1.1.1")[0]
    before = world.contexts.read_bytes()
    result = sheets.import_review(fill(path, {a: update}), **world.kwargs())
    assert len(result.errors) == 1 and expected in result.errors[0][2], result.errors
    assert world.contexts.read_bytes() == before


def test_label_edit_longer_than_60_characters_is_refused(tmp_path):
    world = make_world(tmp_path)
    path, _ = export_all(world)
    label = SYNTHETIC["B1.1.1.1.1"][0]
    result = sheets.import_review(fill(path, {label: {"decision": "edit", "edited_text": "x" * 61}}), **world.kwargs())
    assert len(result.errors) == 1 and "limit for a label is 60" in result.errors[0][2]
    assert world.label_reviews() == {}


def test_a_second_reviewer_who_disagrees_is_a_conflict_until_override(tmp_path):
    world = make_world(tmp_path)
    path, _ = export_all(world)
    a = texts("B2.1.2.1.1")[0]
    assert sheets.import_review(fill(path, {a: {"decision": "accept"}}), date="2026-11-20", **world.kwargs()).ok

    other = fill(path, {a: {"decision": "reject", "reviewer_id": "YB02"}}, out=path.with_name("second.csv"))
    result = sheets.import_review(other, date="2026-11-25", **world.kwargs())
    assert not result.ok and len(result.conflicts) == 1 and result.errors == []
    assert "already accepted by KA01" in result.conflicts[0][2]
    assert world.entry("B2.1.2.1.1", a)["reviewer"] == "KA01"  # nothing changed

    same = fill(path, {a: {"decision": "accept", "reviewer_id": "YB02"}}, out=path.with_name("third.csv"))
    agreed = sheets.import_review(same, **world.kwargs())
    assert agreed.ok and agreed.unchanged == 1
    assert world.entry("B2.1.2.1.1", a)["reviewer"] == "KA01"

    forced = sheets.import_review(other, date="2026-11-25", override=True, **world.kwargs())
    assert forced.ok and forced.count("setting", "reject") == 1
    assert world.entry("B2.1.2.1.1", a) is None
    assert world.doc()["rejected"]["math:B2.1.2.1.1"][0]["reviewer"] == "YB02"


def test_override_can_replace_an_edit_with_another_edit(tmp_path):
    world = make_world(tmp_path)
    path, _ = export_all(world)
    a = texts("B2.1.2.1.1")[1]
    one = "A trotro in Kumasi has 14 people. 6 more get in. How many people are in it now?"
    two = "A trotro in Kumasi has 14 people. 4 more get in. How many people are in it now?"
    assert sheets.import_review(fill(path, {a: {"decision": "edit", "edited_text": one}}), **world.kwargs()).ok
    again = fill(path, {a: {"decision": "edit", "edited_text": two}}, out=path.with_name("second.csv"))
    assert len(sheets.import_review(again, **world.kwargs()).conflicts) == 1
    assert sheets.import_review(again, override=True, **world.kwargs()).ok
    live = [e["text"] for e in world.doc()["settings"]["math:B2.1.2.1.1"]]
    assert two in live and one not in live and a not in live
    assert world.entry("B2.1.2.1.1", two)["edited_from"] == text_hash(a)


def test_dry_run_reports_and_writes_nothing(tmp_path):
    world = make_world(tmp_path)
    path, _ = export_all(world)
    before = world.contexts.read_bytes()
    filled = fill(
        path, {texts("B1.1.1.1.1")[0]: {"decision": "accept"}, SYNTHETIC["B1.1.1.1.1"][0]: {"decision": "accept"}}
    )
    result = sheets.import_review(filled, dry_run=True, **world.kwargs())
    assert result.ok and result.count("setting", "accept") == 1 and result.count("label", "accept") == 1
    assert "dry run: nothing was written" in result.lines()
    assert world.contexts.read_bytes() == before and not world.labels.exists()


# ----------------------------------------------------------------------------- sheets as spreadsheets save them


def _rewrite(path, out, delimiter=",", header=None, drop=(), encoding="utf-8"):
    rows = sheets.read_sheet(path)
    names = [c for c in rows[0] if c not in drop]
    shown = [header.get(c, c) if header else c for c in names]
    buf = io.StringIO(newline="")
    writer = csv.writer(buf, delimiter=delimiter)
    writer.writerow(shown)
    for row in rows:
        writer.writerow([row[c] for c in names])
    out.write_bytes(buf.getvalue().encode(encoding))
    return out


def test_import_reads_semicolons_renamed_headers_and_a_missing_row_id(tmp_path):
    world = make_world(tmp_path)
    path, _ = export_all(world)
    a, b, _ = texts("B5.1.3.1.1") + [None]
    filled = fill(path, {a: {"decision": "accept"}, SYNTHETIC["B5.1.3.1.1"][0]: {"decision": "accept"}})
    odd = _rewrite(
        filled,
        tmp_path / "odd.csv",
        delimiter=";",
        header={"reviewer_id": "Reviewer ID", "draft_text": "Draft text", "region_taught": "Region taught"},
        drop=("row_id",),
    )
    result = sheets.import_review(odd, **world.kwargs())
    assert result.ok, result.lines()
    assert world.entry("B5.1.3.1.1", a)["reviewed"] is True
    assert world.entry("B5.1.3.1.1", b)["reviewed"] is False
    assert "B5.1.3.1.1" in world.label_reviews()


def test_unreadable_sheets_raise_a_clear_error(tmp_path):
    world = make_world(tmp_path)
    path, _ = export_all(world)
    filled = fill(path, {texts("B3.1.4.1.1")[0]: {"decision": "accept"}})
    latin = _rewrite(filled, tmp_path / "latin.csv", encoding="utf-16")
    with pytest.raises(sheets.SheetError, match="not UTF-8"):
        sheets.import_review(latin, **world.kwargs())
    no_decision = _rewrite(filled, tmp_path / "nodecision.csv", drop=("decision",))
    with pytest.raises(sheets.SheetError, match="no 'decision' column"):
        sheets.import_review(no_decision, **world.kwargs())
    bare = _rewrite(filled, tmp_path / "bare.csv", drop=("row_id", "draft_text"))
    with pytest.raises(sheets.SheetError, match="row_id"):
        sheets.import_review(bare, **world.kwargs())
    empty = tmp_path / "empty.csv"
    empty.write_bytes(b"")
    with pytest.raises(sheets.SheetError, match="empty"):
        sheets.import_review(empty, **world.kwargs())


def test_a_cedi_sign_lost_by_a_spreadsheet_is_caught_by_the_checks(tmp_path):
    world = make_world(tmp_path)
    path, _ = export_all(world)
    a = texts("B3.1.4.1.1")[0]
    broken = a.replace("GH₵", "GH?")
    result = sheets.import_review(fill(path, {a: {"decision": "edit", "edited_text": broken}}), **world.kwargs())
    assert len(result.errors) == 1 and "fails the automatic checks" in result.errors[0][2]


# ----------------------------------------------------------------------------- translations


def _with_translations(tmp_path):
    english = texts("B1.1.1.1.1")
    world = make_world(tmp_path, reviewed=(english[0], english[1]))
    items = {
        canonical_id("B1.1.1.1.1"): [
            # placeholders, not real Twi: the tests look only at the review mechanics
            {
                "source_hash": text_hash(english[0]),
                "text": "Tw-one 5 xx 2 xx.",
                "reviewed": False,
                "source": "machine:khaya",
            },
            {
                "source_hash": text_hash(english[1]),
                "text": "Tw-two 3 xx 4 xx.",
                "reviewed": False,
                "source": "machine:khaya",
            },
            {
                "source_hash": text_hash(english[2]),
                "text": "Tw-three 6 xx 3 xx.",
                "reviewed": False,
                "source": "machine:khaya",
            },
        ]
    }
    save_translations({"schema_version": 1, "subject": "math", "lang": "tw", "translations": items}, world.translations)
    return world, english


def test_translation_sheet_lists_only_translations_of_reviewed_english(tmp_path):
    world, english = _with_translations(tmp_path)
    result = sheets.export_review(out_dir=world.out, lang="tw", **world.kwargs())
    path = result.files[0][0]
    assert path.name == "math.translations.tw.batch-01-of-01.csv"
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
    rows = sheets.read_sheet(path)
    assert tuple(rows[0]) == sheets.TRANSLATION_COLUMNS
    assert [r["english_text"] for r in rows] == english[:2]
    assert [r["draft_text"] for r in rows] == ["Tw-one 5 xx 2 xx.", "Tw-two 3 xx 4 xx."]
    assert {r["kind"] for r in rows} == {"translation"} and {r["lang"] for r in rows} == {"tw"}
    more = sheets.export_review(out_dir=world.out, lang="tw", include_drafts=True, **world.kwargs())
    assert len(sheets.read_sheet(more.files[0][0])) == 3
    nothing = sheets.export_review(out_dir=world.out, lang="ee", **world.kwargs())
    assert nothing.files == [] and nothing.pending_rows == 0


def test_translation_round_trip_is_idempotent_and_checks_the_numbers(tmp_path):
    world, english = _with_translations(tmp_path)
    result = sheets.export_review(out_dir=world.out, lang="tw", include_drafts=True, **world.kwargs())
    path = result.files[0][0]
    filled = fill(
        path,
        {
            "Tw-one 5 xx 2 xx.": {"decision": "accept"},
            "Tw-two 3 xx 4 xx.": {"decision": "edit", "edited_text": "Tw-two better 3 xx 4 xx."},
            "Tw-three 6 xx 3 xx.": {"decision": "reject", "note": "Not how a teacher says it."},
        },
        reviewer_id="TW07",
        region_taught="Ashanti",
    )
    first = sheets.import_review(filled, date="2026-12-01", **world.kwargs())
    assert first.ok, first.lines()
    assert [first.count("translation", d) for d in ("accept", "edit", "reject")] == [1, 1, 1]
    tdoc = world.tdoc("tw")
    live = tdoc["translations"]["math:B1.1.1.1.1"]
    assert [(t["text"], t["reviewed"], t["reviewer"]) for t in live] == [
        ("Tw-one 5 xx 2 xx.", True, "TW07"),
        ("Tw-two better 3 xx 4 xx.", True, "TW07"),
    ]
    assert live[1]["edited_from"] == text_hash("Tw-two 3 xx 4 xx.") and live[1]["source"] == "teacher-edit"
    assert live[0]["reviewed_at"] == "2026-12-01" and live[0]["region"] == "Ashanti"
    assert tdoc["rejected"]["math:B1.1.1.1.1"][0]["note"] == "Not how a teacher says it."
    snapshot = (world.translations / "tw.json").read_bytes()
    second = sheets.import_review(filled, date="2027-02-02", **world.kwargs())
    assert second.ok and second.unchanged == 3 and second.applied == {}
    assert (world.translations / "tw.json").read_bytes() == snapshot
    # the English settings file was not touched by a translation import
    assert world.entry("B1.1.1.1.1", english[0])["reviewer"] == "T1"

    wrong = fill(
        path,
        {"Tw-one 5 xx 2 xx.": {"decision": "edit", "edited_text": "Tw-one 5 xx 9 xx."}},
        out=path.with_name("wrong.csv"),
        reviewer_id="TW07",
    )
    bad = sheets.import_review(wrong, override=True, **world.kwargs())
    assert len(bad.errors) == 1 and "numbers must be the same" in bad.errors[0][2]


def test_translation_of_a_changed_english_text_is_refused(tmp_path):
    world, english = _with_translations(tmp_path)
    result = sheets.export_review(out_dir=world.out, lang="tw", **world.kwargs())
    filled = fill(result.files[0][0], {"Tw-one 5 xx 2 xx.": {"decision": "accept"}})
    doc = world.doc()
    doc["settings"]["math:B1.1.1.1.1"][0]["text"] = "Ama has 6 oranges at Makola Market. She gets 2 more. How many now?"
    world.contexts.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    refused = sheets.import_review(filled, **world.kwargs())
    assert len(refused.errors) == 1 and "English setting has changed" in refused.errors[0][2]
    assert not world.tdoc("tw")["translations"]["math:B1.1.1.1.1"][0]["reviewed"]
