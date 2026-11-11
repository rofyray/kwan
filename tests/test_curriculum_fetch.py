"""`kwan curriculum fetch`: sources.json, reuse of local copies and the hash-change stop. No network."""

import hashlib
import json
import os

import pytest

from kwan.curriculum import fetch as fetch_mod
from kwan.curriculum.families import family_from_dict

from .curriculum_helpers import MATH

DOCS = [
    {
        "id": "doc1",
        "grades": [1],
        "url": "https://example.org/one.pdf",
        "file": "one.pdf",
        "local_copies": ["raw/old1.pdf"],
    },
    {"id": "doc2", "grades": [4], "url": "https://example.org/two.pdf", "file": "two.pdf"},
]
FAM = family_from_dict("synthetic-math", {**MATH, "documents": DOCS})


def fake_fetcher(content_by_url, calls):
    def fetcher(url, dest):
        calls.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content_by_url[url])

    return fetcher


def sources(root):
    return {s["id"]: s for s in json.loads((root / "data/curriculum/sources.json").read_text())["sources"]}


def test_fetch_downloads_records_and_then_reuses(tmp_path):
    calls = []
    content = {"https://example.org/one.pdf": b"first pdf", "https://example.org/two.pdf": b"second pdf!"}
    report = fetch_mod.fetch(FAM, root=tmp_path, fetcher=fake_fetcher(content, calls), today="2026-01-02")
    assert calls == ["https://example.org/one.pdf", "https://example.org/two.pdf"]
    assert (tmp_path / "raw/curriculum/one.pdf").read_bytes() == b"first pdf"
    rec = sources(tmp_path)
    assert rec["doc2"] == {
        "id": "doc2",
        "family": "synthetic-math",
        "subject": "math",
        "url": "https://example.org/two.pdf",
        "file": "two.pdf",
        "path": "raw/curriculum/two.pdf",
        "size_bytes": 11,
        "sha256": hashlib.sha256(b"second pdf!").hexdigest(),
        "fetched_at": "2026-01-02",
    }
    assert report.ok() and "downloaded" in report.text()
    # a second run finds the files and downloads nothing; the fetch date stays
    report = fetch_mod.fetch(FAM, root=tmp_path, fetcher=fake_fetcher(content, calls), today="2026-03-04")
    assert len(calls) == 2
    assert [r["action"] for r in report.rows] == ["already there", "already there"]
    assert sources(tmp_path)["doc1"]["fetched_at"] == "2026-01-02"


def test_fetch_reuses_a_local_copy_instead_of_downloading(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw/old1.pdf").write_bytes(b"kept from v1")
    calls = []
    content = {"https://example.org/two.pdf": b"second"}
    report = fetch_mod.fetch(FAM, root=tmp_path, fetcher=fake_fetcher(content, calls), today="2026-01-02")
    assert calls == ["https://example.org/two.pdf"]
    assert report.rows[0]["action"] == os.path.join("copied from raw", "old1.pdf")
    assert (tmp_path / "raw/curriculum/one.pdf").read_bytes() == b"kept from v1"


def test_changed_hash_fails_with_a_report_and_changes_nothing(tmp_path):
    calls = []
    old = {"https://example.org/one.pdf": b"first pdf", "https://example.org/two.pdf": b"second pdf"}
    fetch_mod.fetch(FAM, root=tmp_path, fetcher=fake_fetcher(old, calls), today="2026-01-02")
    before = (tmp_path / "data/curriculum/sources.json").read_text()
    new = {**old, "https://example.org/two.pdf": b"a new edition"}
    with pytest.raises(fetch_mod.HashChanged) as err:
        fetch_mod.fetch(FAM, root=tmp_path, refresh=True, fetcher=fake_fetcher(new, calls), today="2026-05-06")
    text = str(err.value)
    assert "HASH CHANGED for doc2" in text
    assert hashlib.sha256(b"second pdf").hexdigest() in text and hashlib.sha256(b"a new edition").hexdigest() in text
    assert "doc1" in text  # the report still lists every document
    assert (tmp_path / "data/curriculum/sources.json").read_text() == before
    assert (tmp_path / "raw/curriculum/two.pdf").read_bytes() == b"second pdf"  # the pinned file is kept
    assert (tmp_path / "raw/curriculum/two.pdf.new").read_bytes() == b"a new edition"  # kept apart for a look
    # accepting the change records the new hash and replaces the file
    report = fetch_mod.fetch(
        FAM, root=tmp_path, refresh=True, accept_changes=True, fetcher=fake_fetcher(new, calls), today="2026-05-06"
    )
    assert not report.ok()
    assert sources(tmp_path)["doc2"]["sha256"] == hashlib.sha256(b"a new edition").hexdigest()
    assert sources(tmp_path)["doc2"]["fetched_at"] == "2026-05-06"
    assert (tmp_path / "raw/curriculum/two.pdf").read_bytes() == b"a new edition"


def test_local_file_that_no_longer_matches_the_record_is_caught_without_any_download(tmp_path):
    calls = []
    content = {"https://example.org/one.pdf": b"first pdf", "https://example.org/two.pdf": b"second pdf"}
    fetch_mod.fetch(FAM, root=tmp_path, fetcher=fake_fetcher(content, calls), today="2026-01-02")
    (tmp_path / "raw/curriculum/one.pdf").write_bytes(b"edited by hand")
    with pytest.raises(fetch_mod.HashChanged, match="HASH CHANGED for doc1"):
        fetch_mod.fetch(FAM, root=tmp_path, fetcher=fake_fetcher(content, calls))
    assert len(calls) == 2


def test_failed_download_leaves_no_file_and_no_record(tmp_path):
    def broken(url, dest):
        raise OSError("no network")

    with pytest.raises(OSError):
        fetch_mod.fetch(FAM, root=tmp_path, fetcher=broken)
    assert not (tmp_path / "data/curriculum/sources.json").exists()
    assert not (tmp_path / "raw/curriculum/one.pdf").exists()


def test_sources_of_other_families_are_kept(tmp_path):
    path = tmp_path / "data/curriculum/sources.json"
    path.parent.mkdir(parents=True)
    other = {"id": "x", "family": "another-family", "sha256": "abc"}
    path.write_text(json.dumps({"schema_version": 1, "sources": [other]}))
    content = {"https://example.org/one.pdf": b"1", "https://example.org/two.pdf": b"2"}
    fetch_mod.fetch(FAM, root=tmp_path, fetcher=fake_fetcher(content, []))
    rows = json.loads(path.read_text())["sources"]
    assert rows[0] == other and [r["id"] for r in rows] == ["x", "doc1", "doc2"]


@pytest.mark.network
@pytest.mark.skipif(
    os.environ.get("KWAN_RUN_NETWORK") != "1", reason="network: set KWAN_RUN_NETWORK=1 to download the PDFs"
)
def test_real_download_matches_the_committed_hashes(tmp_path):
    from pathlib import Path

    from kwan.curriculum.families import load_family

    repo = Path(__file__).resolve().parent.parent
    target = tmp_path / "data/curriculum"
    target.mkdir(parents=True)
    (target / "sources.json").write_text((repo / "data/curriculum/sources.json").read_text())
    report = fetch_mod.fetch(load_family(), root=tmp_path, refresh=True)  # raises HashChanged on any difference
    assert report.ok() and [r["action"] for r in report.rows] == ["downloaded", "downloaded"]
