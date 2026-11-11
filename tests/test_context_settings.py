"""Practice-setting selection (PRD 8.6): reviewed settings first, then the v1 bank; drafts only on request."""

import inspect
import json
import zlib

import pytest

from kwan.context import bank
from kwan.context.settings import (
    ContextStore,
    bare_code,
    canonical_id,
    clean_text,
    code_sort_key,
    contexts_path,
    live_translations,
    load_contexts,
    save_contexts,
    save_translations,
    text_hash,
)

CODE = "B2.1.2.1.1"
ID = "math:" + CODE


def entry(text, lang="en", reviewed=False):
    return {"text": text, "lang": lang, "reviewed": reviewed}


def crc_pick(options, code=CODE):
    return options[zlib.crc32(code.encode()) % len(options)]


# ----------------------------------------------------------------------------- equal to v1 without reviews


def test_empty_store_equals_the_v1_bank_on_every_node_of_the_v1_graph(G_v1):
    store = ContextStore()
    assert G_v1.number_of_nodes() > 200
    for node, data in G_v1.nodes(data=True):
        expected = bank.practice_setting(bare_code(node), data["strand"], data["grade"])
        assert expected
        assert store.pick(node, data["strand"], data["grade"]) == expected
        assert store.pick(canonical_id(bare_code(node)), data["strand"], data["grade"]) == expected
        assert store.pick(node, data["strand"], data["grade"], lang="tw") == expected


def test_committed_settings_have_no_reviews_so_picks_equal_the_v1_bank_on_every_node(G):
    """Today no teacher has reviewed anything: the committed file must not change a single pick."""
    doc = load_contexts()
    if any(e.get("reviewed") for entries in doc["settings"].values() for e in entries):
        pytest.skip("teachers have reviewed settings; picks are allowed to differ from v1 now")
    store = ContextStore.from_file()
    for node, data in G.nodes(data=True):
        expected = bank.practice_setting(bare_code(node), data["strand"], data["grade"])
        for lang in ("en", "tw", "ee", "gaa"):
            assert store.pick(node, data["strand"], data["grade"], lang=lang) == expected


def test_store_from_a_bundle_without_reviews_equals_the_v1_bank(G):
    bundle = {"subject": "math", "contexts": {}, "context_bank": bank.bank_dict()}
    store = ContextStore.from_bundle(bundle)
    no_bank = ContextStore.from_bundle({"contexts": {}})
    for node, data in G.nodes(data=True):
        expected = bank.practice_setting(bare_code(node), data["strand"], data["grade"])
        assert store.pick(node, data["strand"], data["grade"]) == expected
        assert no_bank.pick(node, data["strand"], data["grade"]) == expected


def test_unreviewed_drafts_are_never_picked_by_default():
    store = ContextStore({ID: [entry("Draft one in Tema."), entry("Draft two in Wa.")]})
    assert store.pick(ID, 1, 2) == bank.practice_setting(CODE, 1, 2)
    assert store.choose(ID, 1, 2)["tier"] == "bank"


# ----------------------------------------------------------------------------- order of the tiers


def test_reviewed_english_wins_over_the_bank():
    reviewed = ["Reviewed A in Ho.", "Reviewed B in Wa.", "Reviewed C in Ada."]
    store = ContextStore({ID: [entry("Draft in Tema."), *[entry(t, reviewed=True) for t in reviewed]]})
    assert store.pick(ID, 1, 2) == crc_pick(reviewed)
    assert store.choose(ID, 1, 2) == {"text": crc_pick(reviewed), "tier": "reviewed", "lang": "en", "reviewed": True}


def test_requested_language_first_then_reviewed_english_then_bank():
    store = ContextStore(
        {
            ID: [
                entry("English reviewed in Ho.", reviewed=True),
                entry("Twi reviewed.", lang="tw", reviewed=True),
                entry("Ewe draft.", lang="ee"),
            ]
        }
    )
    assert store.pick(ID, 1, 2, lang="tw") == "Twi reviewed."
    assert store.pick(ID, 1, 2, lang="ee") == "English reviewed in Ho."  # the Ewe draft is not reviewed
    assert store.pick(ID, 1, 2, lang="gaa") == "English reviewed in Ho."
    assert store.pick(ID, 1, 2) == "English reviewed in Ho."
    assert store.pick("math:B2.1.2.1.2", 1, 2, lang="tw") == bank.practice_setting("B2.1.2.1.2", 1, 2)


def test_drafts_only_with_include_drafts_and_after_reviewed_settings():
    drafts = ["Draft one in Tema.", "Draft two in Wa."]
    only_drafts = ContextStore({ID: [entry(t) for t in drafts]})
    assert only_drafts.pick(ID, 1, 2, include_drafts=True) == crc_pick(drafts)
    assert only_drafts.choose(ID, 1, 2, include_drafts=True)["tier"] == "draft"
    assert only_drafts.choose(ID, 1, 2, include_drafts=True)["reviewed"] is False

    mixed = ContextStore({ID: [*[entry(t) for t in drafts], entry("Reviewed in Ho.", reviewed=True)]})
    assert mixed.pick(ID, 1, 2, include_drafts=True) == "Reviewed in Ho."

    langs = ContextStore({ID: [entry("English draft in Ho."), entry("Twi draft.", lang="tw")]})
    assert langs.pick(ID, 1, 2, lang="tw", include_drafts=True) == "Twi draft."
    assert langs.pick(ID, 1, 2, lang="ee", include_drafts=True) == "English draft in Ho."
    assert langs.pick(ID, 1, 2, lang="tw") == bank.practice_setting(CODE, 1, 2)


def test_pick_is_crc32_of_the_bare_code_whatever_id_is_passed():
    reviewed = [f"Reviewed {i} in Ho." for i in range(7)]
    for code in ("B1.1.1.1.1", "B4.3.3.2.5", "B6.4.2.2.3"):
        store = ContextStore({code: [entry(t, reviewed=True) for t in reviewed]})  # bare keys are accepted
        expected = reviewed[zlib.crc32(code.encode()) % 7]
        assert store.pick(code, 1, 4) == expected
        assert store.pick("math:" + code, 1, 4) == expected
        assert [store.pick(code, 1, 4) for _ in range(3)] == [expected] * 3


def test_empty_bank_cell_gives_an_empty_string_like_v1():
    store = ContextStore({}, bank={"1:lower": ["Only this one in Ho."]})
    assert store.pick("B1.1.1.1.1", 1, 1) == "Only this one in Ho."
    assert store.pick("B4.1.1.1.1", 1, 4) == ""
    assert bank.practice_setting("B1.1.1.1.1", 9, 1) == ""
    assert ContextStore().pick("B1.9.1.1.1", 9, 1) == ""


def test_bank_accepts_both_key_styles():
    a = ContextStore({}, bank={"3:upper": ["X in Ho.", "Y in Wa."]})
    b = ContextStore({}, bank={(3, "upper"): ["X in Ho.", "Y in Wa."]})
    assert a.pick("B5.3.1.1.1", 3, 5) == b.pick("B5.3.1.1.1", 3, 5) == crc_pick(["X in Ho.", "Y in Wa."], "B5.3.1.1.1")
    assert a.bundle_bank() == {"3:upper": ["X in Ho.", "Y in Wa."]}


def test_pick_has_the_signature_of_the_context_source_protocol():
    params = inspect.signature(ContextStore.pick).parameters
    assert list(params) == ["self", "indicator_id", "strand", "grade", "lang", "include_drafts"]
    assert params["lang"].default == "en" and params["include_drafts"].default is False


# ----------------------------------------------------------------------------- bank helpers


def test_bank_dict_matches_the_v1_bank():
    d = bank.bank_dict()
    assert set(d) == {f"{s}:{b}" for s in (1, 2, 3, 4) for b in ("lower", "upper")}
    assert sum(len(v) for v in d.values()) == 21
    assert d["1:lower"] == bank.CONTEXTS[(1, "lower")]
    assert ContextStore().bundle_bank() == d


# ----------------------------------------------------------------------------- bundles


def test_bundle_contexts_hold_only_reviewed_entries_and_round_trip():
    store = ContextStore(
        {
            ID: [entry("Draft in Tema."), entry("Reviewed in Ho.", reviewed=True), entry("Twi.", "tw", True)],
            "math:B3.1.1.1.1": [entry("Only a draft in Wa.")],
        }
    )
    contexts = store.bundle_contexts()
    assert contexts == {
        ID: [
            {"text": "Reviewed in Ho.", "lang": "en", "reviewed": True},
            {"text": "Twi.", "lang": "tw", "reviewed": True},
        ]
    }
    assert set(store.bundle_contexts(include_drafts=True)) == {ID, "math:B3.1.1.1.1"}
    again = ContextStore.from_bundle({"subject": "math", "contexts": contexts, "context_bank": store.bundle_bank()})
    for lang in ("en", "tw", "ee"):
        assert again.pick(ID, 1, 2, lang=lang) == store.pick(ID, 1, 2, lang=lang)
    assert again.pick("B3.1.1.1.1", 1, 3) == bank.practice_setting("B3.1.1.1.1", 1, 3)
    json.dumps(contexts)  # plain data


# ----------------------------------------------------------------------------- files


def _write(tmp_path, settings, translations=None):
    path = tmp_path / "contexts" / "math.json"
    save_contexts({"schema_version": 1, "subject": "math", "settings": settings}, path)
    for lang, items in (translations or {}).items():
        save_translations(
            {"schema_version": 1, "subject": "math", "lang": lang, "translations": items}, path.parent / "translations"
        )
    return path


def test_from_file_uses_reviewed_translations_of_reviewed_english(tmp_path):
    english = "Yaw sells 12 balls of kenkey in Tema. How many are left?"
    draft = "A trotro in Kumasi has 14 people. How many now?"
    path = _write(
        tmp_path,
        {ID: [entry(english, reviewed=True), entry(draft)]},
        {
            "tw": {
                ID: [
                    {"source_hash": text_hash(english), "text": "Twi of reviewed.", "reviewed": True},
                    {"source_hash": text_hash(draft), "text": "Twi of a draft.", "reviewed": True},
                    {"source_hash": "0" * 16, "text": "Twi of a text that changed.", "reviewed": True},
                ]
            },
            "ee": {ID: [{"source_hash": text_hash(english), "text": "Ewe machine draft.", "reviewed": False}]},
        },
    )
    store = ContextStore.from_file(path)
    assert store.pick(ID, 1, 2, lang="tw") == "Twi of reviewed."
    assert store.pick(ID, 1, 2, lang="ee") == english  # the Ewe draft is not reviewed
    assert store.pick(ID, 1, 2, lang="ee", include_drafts=True) == english  # reviewed English beats a draft
    texts = {e["text"] for e in store.entries(ID)}
    assert "Twi of a text that changed." not in texts  # stale: its English text is gone
    twi_of_draft = next(e for e in store.entries(ID) if e["text"] == "Twi of a draft.")
    assert twi_of_draft["reviewed"] is False  # its English text has not been reviewed
    assert ContextStore.from_file(path, translations=False).pick(ID, 1, 2, lang="tw") == english


def test_live_translations_drop_stale_entries():
    settings = {ID: [entry("English in Ho.", reviewed=True)]}
    items = {
        ID: [
            {"source_hash": text_hash("English in Ho."), "text": "A", "reviewed": True},
            {"source_hash": text_hash("Old English in Ho."), "text": "B", "reviewed": True},
        ],
        "math:B9.1.1.1.1": [{"source_hash": "x", "text": "C", "reviewed": True}],
    }
    assert live_translations(settings, items, "tw") == {
        ID: [{"text": "A", "lang": "tw", "reviewed": True, "source_hash": text_hash("English in Ho.")}]
    }


def test_load_contexts_of_a_missing_file_is_empty_and_save_sorts_by_code(tmp_path):
    path = tmp_path / "math.json"
    assert load_contexts(path)["settings"] == {}
    save_contexts(
        {
            "schema_version": 1,
            "subject": "math",
            "settings": {
                "math:B4.1.1.1.10": [entry("x")],
                "math:B10.1.1.1.1": [entry("x")],
                "math:B4.1.1.1.2": [entry("x")],
                "math:B1.1.1.1.1": [],
            },
        },
        path,
    )
    raw = path.read_text(encoding="utf-8")
    assert raw.endswith("\n") and "GH\\u" not in raw
    assert list(load_contexts(path)["settings"]) == ["math:B4.1.1.1.2", "math:B4.1.1.1.10", "math:B10.1.1.1.1"]
    path.write_text('{"schema_version": 99, "settings": {}}', encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        load_contexts(path)


def test_contexts_path_follows_the_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("KWAN_DATA_DIR", str(tmp_path))
    assert contexts_path() == tmp_path.resolve() / "contexts" / "math.json"


# ----------------------------------------------------------------------------- small helpers


def test_text_hash_ignores_spacing_and_changes_with_the_text():
    a = text_hash("Ama has 3 yams in Ho.")
    assert a == text_hash("  Ama has 3  yams in Ho.\n")
    assert a != text_hash("Ama has 4 yams in Ho.")
    assert len(a) == 16 and int(a, 16) >= 0
    assert clean_text(" Ama  has\n3 yams ") == "Ama has 3 yams"


def test_id_helpers():
    assert bare_code("math:B4.1.1.1.2") == bare_code("B4.1.1.1.2") == "B4.1.1.1.2"
    assert canonical_id("B4.1.1.1.2") == canonical_id("math:B4.1.1.1.2") == "math:B4.1.1.1.2"
    codes = ["math:B4.1.1.1.10", "math:B4.1.1.1.2", "math:B10.1.1.1.1", "math:B1.4.1.1.1"]
    assert sorted(codes, key=code_sort_key) == [
        "math:B1.4.1.1.1",
        "math:B4.1.1.1.2",
        "math:B4.1.1.1.10",
        "math:B10.1.1.1.1",
    ]
