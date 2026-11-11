"""Local-language drafts (PRD 8.5): the translation files, the Khaya client with a fake transport, coverage."""

import json

import pytest

from kwan.context import translate
from kwan.context.settings import load_translations, save_translations, text_hash
from tests.test_context_support import SYNTHETIC, make_world

B1 = SYNTHETIC["B1.1.1.1.1"][1]
B2 = SYNTHETIC["B2.1.2.1.1"][1]


class FakeTransport:
    """Records every request and answers like the documented API."""

    def __init__(self, status=200, body=None, answers=None):
        self.calls = []
        self.status = status
        self.body = body
        self.answers = answers or (lambda text, lang: f"[{lang}] {text}")

    def __call__(self, url, headers, body, timeout):
        payload = json.loads(body.decode("utf-8"))
        self.calls.append((url, dict(headers), payload, timeout))
        if self.body is not None:
            return self.status, self.body
        return self.status, json.dumps(self.answers(payload["in"], payload["lang"])).encode("utf-8")


# ----------------------------------------------------------------------------- the client


def test_client_sends_the_documented_request(monkeypatch):
    monkeypatch.delenv(translate.KHAYA_KEY_ENV, raising=False)
    transport = FakeTransport()
    client = translate.KhayaClient(api_key="test-key", transport=transport)
    out = client.translate("Ama has 5 oranges.", "tw")
    assert out == "[en-tw] Ama has 5 oranges."
    url, headers, payload, timeout = transport.calls[0]
    assert url == "https://translation-api.ghananlp.org/v1/translate"
    assert headers["Ocp-Apim-Subscription-Key"] == "test-key"
    assert headers["Content-Type"] == "application/json"
    assert payload == {"in": "Ama has 5 oranges.", "lang": "en-tw"}
    assert timeout == 30.0


def test_client_reads_the_key_from_the_environment_and_refuses_without_one(monkeypatch):
    monkeypatch.delenv(translate.KHAYA_KEY_ENV, raising=False)
    with pytest.raises(translate.KhayaKeyMissing, match="KHAYA_API_KEY"):
        translate.KhayaClient(transport=FakeTransport())
    monkeypatch.setenv(translate.KHAYA_KEY_ENV, " from-env ")
    transport = FakeTransport()
    translate.KhayaClient(transport=transport).translate("Hello.", "ee")
    assert transport.calls[0][1]["Ocp-Apim-Subscription-Key"] == "from-env"
    assert transport.calls[0][2]["lang"] == "en-ee"


@pytest.mark.parametrize("lang", ["tw", "ee", "gaa"])
def test_the_three_languages_make_the_expected_pair(lang):
    transport = FakeTransport()
    translate.KhayaClient(api_key="k", transport=transport).translate("Hello.", lang)
    assert transport.calls[0][2]["lang"] == f"en-{lang}"


def test_client_refuses_an_unknown_language():
    with pytest.raises(ValueError, match="unknown target language"):
        translate.KhayaClient(api_key="k", transport=FakeTransport()).translate("Hello.", "fr")


def test_client_accepts_a_json_object_with_the_text_under_a_known_key():
    transport = FakeTransport(body=json.dumps({"translatedText": "  Twi text here. "}).encode())
    assert translate.KhayaClient(api_key="k", transport=transport).translate("Hi.", "tw") == "Twi text here."


@pytest.mark.parametrize(
    "status, body, message",
    [
        (401, b'{"statusCode": 401, "message": "Access denied"}', "refused the key"),
        (403, b"", "refused the key"),
        (429, b"", "too many requests"),
        (500, b"server error", "HTTP 500"),
    ],
)
def test_client_turns_http_errors_into_clear_messages(status, body, message):
    client = translate.KhayaClient(api_key="k", transport=FakeTransport(status=status, body=body))
    with pytest.raises(translate.KhayaError, match=message):
        client.translate("Hi.", "tw")


@pytest.mark.parametrize("body", [b"not json", b'{"something": 1}', b'""', b"[1, 2]", b'{"out": ""}'])
def test_client_names_the_unverified_contract_on_a_surprising_answer(body):
    client = translate.KhayaClient(api_key="k", transport=FakeTransport(body=body))
    with pytest.raises(translate.KhayaContractError, match="API contract not verified"):
        client.translate("Hi.", "tw")


# ----------------------------------------------------------------------------- drafting runs


def test_machine_draft_translates_reviewed_b1_to_b3_english_only(tmp_path):
    world = make_world(tmp_path, reviewed=(B1[0], B1[1], SYNTHETIC["B4.1.2.1.1"][1][0]))
    transport = FakeTransport()
    client = translate.KhayaClient(api_key="k", transport=transport)
    sleeps = []
    run = translate.machine_draft(
        "tw", client=client, contexts=world.contexts, translations=world.translations, delay=0.5, sleep=sleeps.append
    )
    assert (run.wanted, run.already, run.made, run.errors) == (2, 0, 2, [])
    assert [c[2]["in"] for c in transport.calls] == [B1[0], B1[1]]  # B4 is never sent; unreviewed B1 is not sent
    assert sleeps == [0.5]
    tdoc = world.tdoc("tw")
    assert tdoc["lang"] == "tw" and tdoc["subject"] == "math"
    items = tdoc["translations"]["math:B1.1.1.1.1"]
    assert [i["source_hash"] for i in items] == [text_hash(B1[0]), text_hash(B1[1])]
    assert all(i["reviewed"] is False and i["reviewer"] is None and i["source"] == "machine:khaya" for i in items)
    assert items[0]["text"] == f"[en-tw] {B1[0]}"
    assert "All machine drafts are unreviewed" in run.lines()[-1]


def test_machine_draft_is_idempotent_and_respects_the_limit(tmp_path):
    world = make_world(tmp_path, reviewed=tuple(B1) + tuple(B2))
    transport = FakeTransport()
    client = translate.KhayaClient(api_key="k", transport=transport)
    first = translate.machine_draft(
        "ee", client=client, limit=3, contexts=world.contexts, translations=world.translations, delay=0
    )
    assert (first.wanted, first.made, first.skipped_limit) == (5, 3, 2)
    second = translate.machine_draft(
        "ee", client=client, contexts=world.contexts, translations=world.translations, delay=0
    )
    assert (second.wanted, second.already, second.made) == (5, 3, 2)
    third = translate.machine_draft(
        "ee", client=client, contexts=world.contexts, translations=world.translations, delay=0
    )
    assert (third.already, third.made) == (5, 0)
    assert len(transport.calls) == 5
    assert sum(len(v) for v in world.tdoc("ee")["translations"].values()) == 5


def test_machine_draft_with_include_drafts_sends_unreviewed_english_too(tmp_path):
    world = make_world(tmp_path)
    client = translate.KhayaClient(api_key="k", transport=FakeTransport())
    nothing = translate.machine_draft("gaa", client=client, contexts=world.contexts, translations=world.translations)
    assert (nothing.wanted, nothing.made) == (0, 0)
    assert not (world.translations / "gaa.json").exists()
    run = translate.machine_draft(
        "gaa", client=client, include_drafts=True, contexts=world.contexts, translations=world.translations, delay=0
    )
    assert run.wanted == len(B1) + len(B2) + len(SYNTHETIC["B3.1.4.1.1"][1]) == run.made


def test_machine_draft_stops_at_the_first_api_error_and_saves_nothing_broken(tmp_path):
    world = make_world(tmp_path, reviewed=tuple(B1))
    client = translate.KhayaClient(api_key="k", transport=FakeTransport(status=401, body=b"{}"))
    run = translate.machine_draft(
        "tw", client=client, contexts=world.contexts, translations=world.translations, delay=0
    )
    assert run.made == 0 and len(run.errors) == 1 and "refused the key" in run.errors[0]
    assert not (world.translations / "tw.json").exists()
    echo = translate.KhayaClient(api_key="k", transport=FakeTransport(answers=lambda t, lang: t))
    run = translate.machine_draft("tw", client=echo, contexts=world.contexts, translations=world.translations, delay=0)
    assert run.made == 0 and all("unchanged" in e for e in run.errors) and len(run.errors) == 3


def test_machine_draft_without_a_key_fails_clearly(tmp_path, monkeypatch):
    monkeypatch.delenv(translate.KHAYA_KEY_ENV, raising=False)
    world = make_world(tmp_path, reviewed=tuple(B1))
    with pytest.raises(translate.KhayaKeyMissing):
        translate.machine_draft("tw", contexts=world.contexts, translations=world.translations)


def test_a_changed_english_text_invalidates_its_unreviewed_draft(tmp_path):
    world = make_world(tmp_path, reviewed=tuple(B1))
    client = translate.KhayaClient(api_key="k", transport=FakeTransport())
    translate.machine_draft("tw", client=client, contexts=world.contexts, translations=world.translations, delay=0)
    doc = world.doc()
    doc["settings"]["math:B1.1.1.1.1"][0]["text"] = "Ama has 7 oranges at Makola Market. She gets 2 more. How many now?"
    world.contexts.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    run = translate.machine_draft(
        "tw", client=client, contexts=world.contexts, translations=world.translations, delay=0
    )
    assert run.stale_removed == 1 and run.made == 1 and run.already == 2
    hashes = {i["source_hash"] for i in world.tdoc("tw")["translations"]["math:B1.1.1.1.1"]}
    assert hashes == {text_hash(e["text"]) for e in doc["settings"]["math:B1.1.1.1.1"]}


def test_dry_run_calls_the_api_but_writes_nothing(tmp_path):
    world = make_world(tmp_path, reviewed=tuple(B1))
    transport = FakeTransport()
    client = translate.KhayaClient(api_key="k", transport=transport)
    run = translate.machine_draft(
        "tw", client=client, dry_run=True, contexts=world.contexts, translations=world.translations, delay=0
    )
    assert run.made == 3 and len(transport.calls) == 3 and run.dry_run
    assert not (world.translations / "tw.json").exists()


# ----------------------------------------------------------------------------- files and coverage


def test_translation_file_round_trip_and_language_check(tmp_path):
    doc = {"schema_version": 1, "subject": "math", "lang": "tw", "translations": {"math:B1.1.1.1.1": [{"text": "x"}]}}
    save_translations(doc, tmp_path)
    loaded = load_translations("tw", tmp_path)
    assert loaded["translations"] == doc["translations"] and loaded["rejected"] == {}
    assert load_translations("ee", tmp_path) == {
        "schema_version": 1,
        "subject": "math",
        "lang": "ee",
        "translations": {},
        "rejected": {},
    }
    (tmp_path / "gaa.json").write_text(json.dumps({"schema_version": 1, "lang": "tw", "translations": {}}))
    with pytest.raises(ValueError, match="holds language"):
        load_translations("gaa", tmp_path)


def test_translation_coverage_counts_only_what_would_ship():
    doc = {
        "settings": {
            "math:B1.1.1.1.1": [{"text": "A", "reviewed": True}, {"text": "B", "reviewed": False}],
            "math:B2.1.1.1.1": [{"text": "C", "reviewed": True}],
            "math:B3.1.1.1.1": [{"text": "D", "reviewed": False}],
            "math:B4.1.1.1.1": [{"text": "E", "reviewed": True}],
        }
    }
    tdoc = {
        "translations": {
            "math:B1.1.1.1.1": [
                {"source_hash": text_hash("A"), "text": "a", "reviewed": True},
                {"source_hash": text_hash("B"), "text": "b", "reviewed": True},  # English not reviewed
                {"source_hash": "0" * 16, "text": "old", "reviewed": True},  # stale
            ],
            "math:B2.1.1.1.1": [{"source_hash": text_hash("C"), "text": "c", "reviewed": False}],
            "math:B4.1.1.1.1": [{"source_hash": text_hash("E"), "text": "e", "reviewed": True}],
        }
    }
    cov = translate.translation_coverage(doc, tdoc)
    assert cov == {
        "indicators_b1_b3": 3,
        "unreviewed_drafts": 1,
        "reviewed_translations": 2,
        "shipping_translations": 1,
        "stale_translations": 1,
        "indicators_with_shipping_translation": 1,
        "coverage_pct": 33.3,
    }
    assert translate.translation_coverage({"settings": {}}, {"translations": {}})["coverage_pct"] == 0.0


def test_committed_translation_coverage_is_reported_as_it_is():
    """Today: no translation exists. The report must say zero, not hide it."""
    from kwan.context.settings import load_contexts

    doc = load_contexts()
    for lang in translate.LOCAL_LANGS:
        cov = translate.translation_coverage(doc, load_translations(lang))
        assert cov["indicators_b1_b3"] == 93
        assert cov["indicators_with_shipping_translation"] <= cov["indicators_b1_b3"]
        if cov["reviewed_translations"] == 0:
            assert cov["coverage_pct"] == 0.0
