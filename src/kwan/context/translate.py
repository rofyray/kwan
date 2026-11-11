"""Machine first drafts of local-language settings with GhanaNLP's Khaya API (PRD 8.5).

Kwan offers practice settings in Twi (`tw`), Ewe (`ee`) and Ga (`gaa`) for Basic 1 to Basic 3. The
first draft of a translation comes from a machine. It is stored with `reviewed: false` and is never
shown until a teacher who reads and writes the language well has approved it on a review sheet.

What is known about the Khaya API, and how
    Request (checked on 2026-11-10 against GhanaNLP's own OpenAPI file in the public repository
    GhanaNLP/ghananlp-translation-api, and against the live host, which answers 401 "missing
    subscription key" on this path and 404 on a made-up path):

        POST https://translation-api.ghananlp.org/v1/translate
        Ocp-Apim-Subscription-Key: <key>
        Content-Type: application/json
        {"in": "<English text>", "lang": "en-tw"}

    Response: NOT checked against a real call, because no key was available when this was written.
    The published file does not describe the response. Community examples read the body as JSON and
    get the translated text as a string. `KhayaClient` therefore accepts a JSON string, or an object
    with the text under one of a few likely keys, and raises `KhayaContractError` for anything else,
    so that a surprise is seen at once and nothing wrong is saved.

    Language pairs: `en-tw` is the documented example. `en-ee` and `en-gaa` follow the same pattern
    but were not confirmed with a real call either.

The key is read from the environment variable `KHAYA_API_KEY`. It is never written to a file.

Only the standard library is used.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kwan.context import checks
from kwan.context.settings import (
    DEFAULT_SUBJECT,
    LANG_NAMES,
    LOCAL_LANG_MAX_GRADE,
    LOCAL_LANGS,
    clean_text,
    code_sort_key,
    load_contexts,
    load_translations,
    save_translations,
    text_hash,
)

KHAYA_URL = "https://translation-api.ghananlp.org/v1/translate"
KHAYA_KEY_ENV = "KHAYA_API_KEY"
KHAYA_KEY_HEADER = "Ocp-Apim-Subscription-Key"
KHAYA_SOURCE = "machine:khaya"
_RESPONSE_KEYS = ("out", "translation", "translated_text", "translatedText", "text", "result")

# (url, headers, body, timeout in seconds) -> (HTTP status, response body)
Transport = Callable[[str, Mapping[str, str], bytes, float], tuple[int, bytes]]


class KhayaError(RuntimeError):
    """A call to the Khaya API failed."""


class KhayaKeyMissing(KhayaError):
    """No API key was given and `KHAYA_API_KEY` is not set."""


class KhayaContractError(KhayaError):
    """The API answered in a shape this client does not know. The response format is not verified."""


def urllib_transport(url: str, headers: Mapping[str, str], body: bytes, timeout: float) -> tuple[int, bytes]:
    """Send one POST with the standard library and return `(status, body)`. HTTP errors do not raise."""
    request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 (fixed https URL)
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except urllib.error.URLError as exc:
        raise KhayaError(f"cannot reach the Khaya API: {exc.reason}") from exc


class KhayaClient:
    """A small client for the Khaya translation API.

    `transport` does the HTTP call; tests pass a fake one, so no test touches the network.
    """

    def __init__(
        self,
        api_key: str | None = None,
        transport: Transport | None = None,
        url: str = KHAYA_URL,
        timeout: float = 30.0,
    ) -> None:
        key = api_key if api_key is not None else os.environ.get(KHAYA_KEY_ENV, "")
        if not key.strip():
            raise KhayaKeyMissing(
                f"no Khaya API key: set the environment variable {KHAYA_KEY_ENV}. "
                "Keys are given out by GhanaNLP (https://ghananlp.org)."
            )
        self._key = key.strip()
        self._transport = transport or urllib_transport
        self.url = url
        self.timeout = timeout

    def translate(self, text: str, target: str, source: str = "en") -> str:
        """Translate one text, for example `translate("Count the oranges.", "tw")`."""
        if target not in LOCAL_LANGS:
            raise ValueError(f"unknown target language {target!r}; Kwan uses {', '.join(LOCAL_LANGS)}")
        body = json.dumps({"in": text, "lang": f"{source}-{target}"}, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json", "Cache-Control": "no-cache", KHAYA_KEY_HEADER: self._key}
        status, raw = self._transport(self.url, headers, body, self.timeout)
        shown = raw[:200].decode("utf-8", errors="replace")
        if status in (401, 403):
            raise KhayaError(f"the Khaya API refused the key (HTTP {status}). Check {KHAYA_KEY_ENV}.")
        if status == 429:
            raise KhayaError("the Khaya API says too many requests (HTTP 429). Wait and run the command again.")
        if status != 200:
            raise KhayaError(f"the Khaya API answered HTTP {status}: {shown}")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise KhayaContractError(f"API contract not verified: the answer is not JSON: {shown}") from exc
        out: Any = payload
        if isinstance(payload, Mapping):
            out = next((payload[k] for k in _RESPONSE_KEYS if isinstance(payload.get(k), str)), None)
        if not isinstance(out, str) or not out.strip():
            raise KhayaContractError(
                "API contract not verified: expected the translated text as a JSON string, got: " + shown
            )
        return clean_text(out)


@dataclass
class DraftRun:
    """What a machine-draft run did."""

    lang: str
    wanted: int = 0
    already: int = 0
    made: int = 0
    skipped_limit: int = 0
    failed_checks: int = 0
    stale_removed: int = 0
    errors: list[str] = field(default_factory=list)
    dry_run: bool = False

    def lines(self) -> list[str]:
        """A short report for the command line."""
        name = LANG_NAMES.get(self.lang, self.lang)
        out = [
            f"{name}: {self.wanted} English settings to translate, {self.already} already drafted, "
            f"{self.made} new machine drafts" + (" (dry run, not saved)" if self.dry_run else ""),
        ]
        if self.skipped_limit:
            out.append(f"{self.skipped_limit} left for the next run (limit reached)")
        if self.failed_checks:
            out.append(f"{self.failed_checks} new drafts fail an automatic check; the reviewer will have to edit them")
        if self.stale_removed:
            out.append(f"{self.stale_removed} unreviewed drafts removed because their English text changed")
        out += [f"error: {e}" for e in self.errors]
        out.append("All machine drafts are unreviewed. Nothing ships until a fluent teacher approves it.")
        return out


def sources_to_translate(
    doc: Mapping[str, Any], include_drafts: bool = False, max_grade: int = LOCAL_LANG_MAX_GRADE
) -> list[tuple[str, str]]:
    """`(indicator id, English text)` pairs that should have a translation, in curriculum order.

    Only reviewed English settings of Basic 1 to Basic 3 by default: translating a draft that a
    teacher may still change wastes the translator's time. `include_drafts` adds unreviewed drafts.
    """
    out = []
    for indicator in sorted(doc.get("settings", {}), key=code_sort_key):
        if checks.grade_of(indicator) > max_grade:
            continue
        for entry in doc["settings"][indicator]:
            if entry.get("lang", "en") == "en" and (include_drafts or entry.get("reviewed")):
                out.append((indicator, str(entry["text"])))
    return out


def machine_draft(
    lang: str,
    client: KhayaClient | None = None,
    limit: int | None = None,
    include_drafts: bool = False,
    delay: float = 1.0,
    dry_run: bool = False,
    contexts: Path | None = None,
    translations: Path | None = None,
    subject: str = DEFAULT_SUBJECT,
    sleep: Callable[[float], None] = time.sleep,
) -> DraftRun:
    """Machine-draft the missing translations of one language and save them as unreviewed.

    A text that already has a translation (drafted, reviewed or rejected) is not sent again, so the
    command can be run many times. `limit` caps the number of API calls in one run and `delay` is the
    pause between calls. Unreviewed drafts whose English text has changed are removed.
    """
    if lang not in LOCAL_LANGS:
        raise ValueError(f"unknown language {lang!r}; Kwan uses {', '.join(LOCAL_LANGS)}")
    doc = load_contexts(contexts, subject)
    tdoc = load_translations(lang, translations, subject)
    run = DraftRun(lang=lang, dry_run=dry_run)
    wanted = sources_to_translate(doc, include_drafts)
    run.wanted = len(wanted)

    # drop unreviewed drafts that no longer match any English text
    for indicator in list(tdoc["translations"]):
        live = {text_hash(str(e["text"])) for e in doc.get("settings", {}).get(indicator, [])}
        keep = [t for t in tdoc["translations"][indicator] if t.get("reviewed") or t.get("source_hash") in live]
        run.stale_removed += len(tdoc["translations"][indicator]) - len(keep)
        tdoc["translations"][indicator] = keep

    todo: list[tuple[str, str, str]] = []
    for indicator, text in wanted:
        h = text_hash(text)
        have = [*tdoc["translations"].get(indicator, []), *tdoc.get("rejected", {}).get(indicator, [])]
        if any(t.get("source_hash") == h for t in have):
            run.already += 1
        else:
            todo.append((indicator, text, h))
    if limit is not None and len(todo) > limit:
        run.skipped_limit = len(todo) - limit
        todo = todo[:limit]
    if todo and client is None:
        client = KhayaClient()
    for n, (indicator, text, h) in enumerate(todo):
        assert client is not None
        if n and delay > 0:
            sleep(delay)
        try:
            translated = client.translate(text, lang)
        except KhayaError as exc:
            run.errors.append(f"{indicator}: {exc}")
            break  # a failing key or a changed API will fail for every text: stop at the first error
        if text_hash(translated) == h:
            run.errors.append(f"{indicator}: the API returned the English text unchanged; not saved")
            continue
        if checks.check_translation(translated, text):
            run.failed_checks += 1
        tdoc["translations"].setdefault(indicator, []).append(
            {
                "source_hash": h,
                "text": translated,
                "reviewed": False,
                "source": KHAYA_SOURCE,
                "reviewer": None,
                "reviewed_at": None,
            }
        )
        run.made += 1
    if not dry_run and (run.made or run.stale_removed):
        save_translations(tdoc, translations)
    return run


def translation_coverage(
    doc: Mapping[str, Any], tdoc: Mapping[str, Any], max_grade: int = LOCAL_LANG_MAX_GRADE
) -> dict[str, Any]:
    """Honest coverage numbers of one language over the Basic 1 to Basic 3 indicators.

    An indicator is covered when it has at least one translation that ships: the translation is
    reviewed, its English text is reviewed, and the hashes still match.
    """
    indicators = [i for i in doc.get("settings", {}) if checks.grade_of(i) <= max_grade]
    machine = reviewed = stale = shipping = 0
    covered = set()
    for indicator, items in tdoc.get("translations", {}).items():
        if checks.grade_of(indicator) > max_grade:
            continue
        english = {text_hash(str(e["text"])): e for e in doc.get("settings", {}).get(indicator, [])}
        for item in items:
            src = english.get(str(item.get("source_hash")))
            if src is None:
                stale += 1
                continue
            if item.get("reviewed"):
                reviewed += 1
                if src.get("reviewed"):
                    shipping += 1
                    covered.add(indicator)
            else:
                machine += 1
    n = len(indicators)
    return {
        "indicators_b1_b3": n,
        "unreviewed_drafts": machine,
        "reviewed_translations": reviewed,
        "shipping_translations": shipping,
        "stale_translations": stale,
        "indicators_with_shipping_translation": len(covered),
        "coverage_pct": round(100.0 * len(covered) / n, 1) if n else 0.0,
    }
