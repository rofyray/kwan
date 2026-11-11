"""`kwan curriculum fetch` (PRD task 2.1): download the curriculum PDFs and pin their hashes.

Each PDF of a family goes to `raw/curriculum/<file>` (git-ignored). Its URL, size, SHA-256 and the date
it was fetched are recorded in `data/curriculum/sources.json` (committed). When a file's hash differs
from the recorded one, the command stops with a report and changes nothing: a new edition of the
curriculum means the indicator counts and corrections must be checked again by a person.

A PDF that is already on disk is not downloaded again (pass `refresh=True` to download anyway). A fresh
download is first written beside the old file as `<file>.new` and replaces it only when its hash is the
recorded one (or when the change is accepted), so a changed document never silently replaces the pinned one.
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kwan.curriculum.build import root_dir, write_json
from kwan.curriculum.families import Family
from kwan.curriculum.pdf import sha256_of

USER_AGENT = "kwan-curriculum-fetch/1.0 (+https://github.com/rofyray/kwan)"
TIMEOUT = 120


class HashChanged(RuntimeError):
    """A PDF no longer matches the SHA-256 recorded in sources.json."""


@dataclass
class FetchReport:
    """What `fetch` did for each document."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    changed: list[dict[str, Any]] = field(default_factory=list)

    def ok(self) -> bool:
        """True when no hash changed."""
        return not self.changed

    def text(self) -> str:
        """A plain-text report for the terminal."""
        lines = []
        for r in self.rows:
            lines.append(f"{r['id']}: {r['action']}  {r['size_bytes']} bytes  sha256 {r['sha256']}")
            lines.append(f"    {r['path']}")
        for c in self.changed:
            lines += [
                f"HASH CHANGED for {c['id']} ({c['url']})",
                f"    recorded  {c['recorded_sha256']}  ({c['recorded_size']} bytes, fetched {c['recorded_at']})",
                f"    now       {c['sha256']}  ({c['size_bytes']} bytes)",
                "    The curriculum document is not the one Kwan was built from. Check the indicator counts",
                "    (data/curriculum/expected_counts.json) and corrections against the new file, then run",
                "    `kwan curriculum fetch --accept-changes` to record the new hash.",
            ]
        return "\n".join(lines)


def sources_path(root: Path | str | None = None) -> Path:
    """Where sources.json lives."""
    return root_dir(root) / "data" / "curriculum" / "sources.json"


def load_sources(root: Path | str | None = None) -> dict[str, Any]:
    """The recorded sources, or an empty record."""
    path = sources_path(root)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"schema_version": 1, "sources": []}


def download(url: str, dest: Path) -> None:
    """Download `url` to `dest` (through a temporary file, so a failed download leaves nothing behind)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response, open(tmp, "wb") as f:  # noqa: S310
        shutil.copyfileobj(response, f)
    tmp.replace(dest)


def fetch(
    family: Family,
    root: Path | str | None = None,
    refresh: bool = False,
    accept_changes: bool = False,
    fetcher: Any = download,
    today: str | None = None,
) -> FetchReport:
    """Get every PDF of the family and update sources.json.

    Raises `HashChanged` (after writing nothing) when a file differs from its recorded hash, unless
    `accept_changes` is true. `fetcher(url, dest)` does the download and can be replaced in tests.
    """
    base = root_dir(root)
    recorded = load_sources(base)
    known = {(s.get("family"), s["id"]): s for s in recorded.get("sources", [])}
    today = today or dt.date.today().isoformat()
    report = FetchReport()
    rows: dict[tuple[str | None, str], dict[str, Any]] = dict(known)
    for doc in family.documents:
        dest = base / "raw" / "curriculum" / doc.file
        candidate = dest
        local = next((base / c for c in doc.local_copies if (base / c).exists()), None)
        if dest.exists() and not refresh:
            action = "already there"
        elif local is not None and not refresh:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local, dest)
            action = f"copied from {local.relative_to(base)}"
        else:
            candidate = dest.with_name(dest.name + ".new")  # kept apart until its hash is checked
            fetcher(doc.url, candidate)
            action = "downloaded"
        row = {
            "id": doc.id,
            "family": family.name,
            "subject": family.subject,
            "url": doc.url,
            "file": doc.file,
            "path": str(dest.relative_to(base)),
            "size_bytes": candidate.stat().st_size,
            "sha256": sha256_of(candidate),
            "fetched_at": today,
        }
        old = known.get((family.name, doc.id))
        differs = bool(old) and old.get("sha256") != row["sha256"]
        if old and not differs and action != "downloaded":
            row["fetched_at"] = old.get("fetched_at", today)
        if differs:
            report.changed.append(
                {
                    **row,
                    "path": str(candidate.relative_to(base)),
                    "recorded_sha256": old.get("sha256"),
                    "recorded_size": old.get("size_bytes"),
                    "recorded_at": old.get("fetched_at"),
                }
            )
        if candidate != dest and (not differs or accept_changes):
            candidate.replace(dest)
        report.rows.append({**row, "action": action})
        rows[(family.name, doc.id)] = row
    if report.changed and not accept_changes:
        raise HashChanged(report.text())
    write_json(sources_path(base), {"schema_version": 1, "sources": list(rows.values())})
    return report
