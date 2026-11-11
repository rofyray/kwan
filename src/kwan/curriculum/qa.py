"""`kwan curriculum qa` (PRD task 2.8): check the parsed curriculum and write an HTML report.

The report shows counts against the hand count, the truncation check, duplicate ids, indicators without
exemplars, the corrections applied, how the pdfplumber path compares with the pdftotext path, and 20
random indicators with page numbers for a person to check against the PDF.

The committed report (`docs/qa/curriculum-<subject>.html`) never shows NaCCA sentences: it identifies
indicators by code, Kwan-written short label and page number. `full=True` writes a second, text-bearing
report under `build/` (git-ignored) for the person doing the spot check.
"""
# ruff: noqa: E501  (HTML and CSS templates: long lines are clearer than broken strings)

from __future__ import annotations

import datetime as dt
import html
import json
import random
import re
import zlib
from collections import Counter
from pathlib import Path
from typing import Any

from kwan.curriculum.build import commit_allowed, parse_family, root_dir
from kwan.curriculum.document import STOP_WORDS, is_truncated, last_word
from kwan.curriculum.families import Family, load_family
from kwan.curriculum.fetch import load_sources
from kwan.curriculum.labels import MAX_LABEL

RESULTS_KEY = "curriculum"
SAMPLE_SIZE = 20
SAMPLE_SEED = "kwan-curriculum-qa"


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def expected_counts(family: Family, root: Path) -> dict[str, Any]:
    """The hand count file of the family."""
    return json.loads(family.path("expected_counts", root).read_text(encoding="utf-8"))


def count_table(records: list[dict[str, Any]]) -> dict[tuple[str, str, str], int]:
    """Indicators per (grade, strand, sub-strand), with string keys like the expected-counts file."""
    return dict(Counter((str(r["grade"]), str(r["strand"]), str(r["sub_strand"])) for r in records))


def compare_counts(records: list[dict[str, Any]], expected: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per (grade, strand, sub-strand) that is expected or found."""
    found = count_table(records)
    want = {(g, s, ss): n for g, a in expected["counts"].items() for s, b in a.items() for ss, n in b.items()}
    names = {(str(r["grade"]), str(r["strand"]), str(r["sub_strand"])): r for r in records}
    rows = []
    for key in sorted(set(found) | set(want), key=lambda k: tuple(int(x) for x in k)):
        r = names.get(key)
        rows.append(
            {
                "grade": int(key[0]),
                "strand": int(key[1]),
                "sub_strand": int(key[2]),
                "strand_name": r["strand_name"] if r else "",
                "sub_strand_name": r["sub_strand_name"] if r else "",
                "expected": want.get(key, 0),
                "found": found.get(key, 0),
            }
        )
    return rows


def compare_engines(primary: list[dict[str, Any]], other: list[dict[str, Any]]) -> dict[str, Any]:
    """How two extraction paths differ: ids, descriptions and exemplar counts."""
    a = {r["code"]: r for r in primary}
    b = {r["code"]: r for r in other}
    both = sorted(set(a) & set(b))
    differ = [c for c in both if _norm(a[c]["description"]) != _norm(b[c]["description"])]
    exemplar_differ = [c for c in both if len(a[c]["exemplars"]) != len(b[c]["exemplars"])]
    return {
        "same_ids": set(a) == set(b),
        "only_in_primary": sorted(set(a) - set(b)),
        "only_in_other": sorted(set(b) - set(a)),
        "descriptions_equal": len(both) - len(differ),
        "descriptions_differ": differ,
        "exemplar_counts_differ": exemplar_differ,
    }


def engine_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Headline numbers of one extraction path."""
    truncated = [r["code"] for r in records if is_truncated(r["description"])]
    return {
        "indicators": len(records),
        "truncated": len(truncated),
        "truncated_codes": truncated,
        "with_exemplar": sum(1 for r in records if r["exemplars"]),
        "exemplars": sum(len(r["exemplars"]) for r in records),
    }


def compute(family: Family | None = None, root: Path | str | None = None, today: str | None = None) -> dict[str, Any]:
    """Run every check on the built files and return the full QA data (with NaCCA text in `sample`)."""
    family = family or load_family()
    base = root_dir(root)
    full_path = family.path("full_file", base)
    if not full_path.exists():
        raise FileNotFoundError(f"{full_path} is missing: run `kwan curriculum parse` first")
    records: list[dict[str, Any]] = json.loads(full_path.read_text(encoding="utf-8"))
    structure: list[dict[str, Any]] = json.loads(family.path("structure_file", base).read_text(encoding="utf-8"))
    meta_path = full_path.with_name(f"{family.subject}.parse.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    engine = meta.get("engine", "pdfplumber")
    expected = expected_counts(family, base)

    rows = compare_counts(records, expected)
    n = len(records)
    ids = Counter(r["id"] for r in records)
    names = Counter(a for r in records for a in set(r["aliases"]) | {r["code"]})
    truncated = [r for r in records if is_truncated(r["description"])]
    without = [r for r in records if not r["exemplars"]]
    exemplars = [e for r in records for e in r["exemplars"]]

    other_engine = "pdftotext" if engine == "pdfplumber" else "pdfplumber"
    engines: dict[str, Any] = {engine: engine_summary(records)}
    try:
        other = parse_family(family, other_engine, base).records
        engines[other_engine] = engine_summary(other)
        engines["comparison"] = compare_engines(records, other)
    except (FileNotFoundError, ImportError, OSError) as exc:
        engines["comparison"] = {"unavailable": f"{other_engine} path could not run: {exc}"}

    rng = random.Random(zlib.crc32(SAMPLE_SEED.encode()))
    sample = sorted(rng.sample(records, min(SAMPLE_SIZE, n)), key=lambda r: records.index(r))
    sources = [s for s in load_sources(base).get("sources", []) if s.get("family") == family.name]
    return {
        "family": family.name,
        "subject": family.subject,
        "title": family.title,
        "generated_on": today or dt.date.today().isoformat(),
        "engine": engine,
        "other_engine": other_engine,
        "records": records,
        "structure_matches_full": [strip["id"] for strip in structure] == [r["id"] for r in records],
        "rows": rows,
        "indicators": n,
        "expected": expected["total"],
        "counts_match": all(r["expected"] == r["found"] for r in rows) and n == expected["total"],
        "per_grade": {str(g): c for g, c in sorted(Counter(r["grade"] for r in records).items())},
        "expected_per_grade": expected.get("per_grade", {}),
        "per_grade_strand": {
            str(g): {
                str(s): c
                for (gg, s), c in sorted(Counter((r["grade"], r["strand"]) for r in records).items())
                if gg == g
            }
            for g in sorted({r["grade"] for r in records})
        },
        "gaps": [x for x in expected.get("notes", []) if x.get("gap")],
        "truncated": truncated,
        "duplicate_ids": sorted(i for i, c in ids.items() if c > 1),
        "alias_clashes": sorted(a for a, c in names.items() if c > 1),
        "without_exemplar": without,
        "exemplars": len(exemplars),
        "exemplars_ghana": sum(1 for e in exemplars if e["ghana_context"]),
        "indicators_with_ghana_exemplar": sum(1 for r in records if any(e["ghana_context"] for e in r["exemplars"])),
        "labels": {
            "reviewed": sum(1 for r in records if r["label_reviewed"]),
            "draft": sum(1 for r in records if not r["label_reviewed"] and not r["short_label"].startswith("Skill ")),
            "missing": [r["code"] for r in records if r["short_label"].startswith("Skill ")],
            "too_long": [r["code"] for r in records if len(r["short_label"]) > MAX_LABEL],
        },
        "log": meta.get("log", []),
        "sub_strand_sources": meta.get("sub_strand_sources", {}),
        "engines": engines,
        "sample": sample,
        "sources": sources,
        "commit_allowed": commit_allowed(base),
    }


def results_block(qa: dict[str, Any]) -> dict[str, Any]:
    """The numbers that go into `data/eval_results.json` under the key `curriculum`."""
    n = qa["indicators"] or 1
    comparison = qa["engines"].get("comparison", {})
    engines = {
        k: {kk: vv for kk, vv in v.items() if kk != "truncated_codes"}
        for k, v in qa["engines"].items()
        if k != "comparison"
    }
    if "unavailable" not in comparison:
        engines["same_ids"] = comparison["same_ids"]
        engines["descriptions_equal"] = comparison["descriptions_equal"]
        engines["descriptions_differ"] = len(comparison["descriptions_differ"])
        engines["exemplar_counts_differ"] = len(comparison["exemplar_counts_differ"])
    return {
        "family": qa["family"],
        "subject": qa["subject"],
        "generated_on": qa["generated_on"],
        "engine": qa["engine"],
        "indicators": qa["indicators"],
        "expected": qa["expected"],
        "counts_match": qa["counts_match"],
        "per_grade": qa["per_grade"],
        "per_grade_strand": qa["per_grade_strand"],
        "truncated": len(qa["truncated"]),
        "truncation_rate": round(len(qa["truncated"]) / n, 4),
        "truncated_codes": [r["code"] for r in qa["truncated"]],
        "with_exemplar": qa["indicators"] - len(qa["without_exemplar"]),
        "share_with_exemplar": round((qa["indicators"] - len(qa["without_exemplar"])) / n, 4),
        "without_exemplar_codes": [r["code"] for r in qa["without_exemplar"]],
        "exemplars": qa["exemplars"],
        "exemplars_ghana_context": qa["exemplars_ghana"],
        "indicators_with_ghana_exemplar": qa["indicators_with_ghana_exemplar"],
        "duplicate_ids": len(qa["duplicate_ids"]),
        "corrections_applied": sum(1 for e in qa["log"] if e["kind"].startswith("correction:")),
        "labels": {
            "draft": qa["labels"]["draft"],
            "reviewed": qa["labels"]["reviewed"],
            "missing": len(qa["labels"]["missing"]),
            "over_60_characters": len(qa["labels"]["too_long"]),
        },
        "extraction": engines,
        "sources": [{k: s.get(k) for k in ("id", "file", "size_bytes", "sha256", "fetched_at")} for s in qa["sources"]],
        "nacca_text_committed": qa["commit_allowed"],
    }


# ---- HTML -------------------------------------------------------------------------------------------

CSS = """
:root{--bg:#fbfaf7;--fg:#1d1b16;--muted:#6b665c;--line:#ddd8cc;--card:#fff;--ok:#1f6f43;--bad:#a4281c;--warn:#8a5a00;--accent:#7a4b12}
@media (prefers-color-scheme:dark){:root{--bg:#16150f;--fg:#ece8dd;--muted:#a39d8f;--line:#3a372e;--card:#1f1e17;--ok:#6fcf97;--bad:#ff8a7a;--warn:#e9b860;--accent:#e2b072}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1060px;margin:0 auto;padding:28px 16px 64px}h1{font-size:1.6rem;margin:0 0 4px}h2{font-size:1.15rem;margin:36px 0 8px;padding-top:12px;border-top:1px solid var(--line)}
p{margin:6px 0 10px}.muted{color:var(--muted)}code{font:13px ui-monospace,SFMono-Regular,Menlo,monospace}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:18px 0}
.tile{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px}.tile b{display:block;font-size:1.45rem;font-variant-numeric:tabular-nums}
.tile span{color:var(--muted);font-size:.85rem}.ok{color:var(--ok)}.bad{color:var(--bad)}.warn{color:var(--warn)}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:.9rem;background:var(--card)}
th,td{border:1px solid var(--line);padding:5px 8px;text-align:left;vertical-align:top}th{background:transparent;color:var(--muted);font-weight:600}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}.text{color:var(--accent)}details{margin:8px 0}summary{cursor:pointer}
"""


def _e(value: Any) -> str:
    return html.escape(str(value))


def _table(headers: list[str], rows: list[list[Any]], numeric: tuple[int, ...] = ()) -> str:
    head = "".join(
        f'<th class="n">{_e(h)}</th>' if i in numeric else f"<th>{_e(h)}</th>" for i, h in enumerate(headers)
    )
    body = []
    for row in rows:
        cells = "".join(f'<td class="n">{c}</td>' if i in numeric else f"<td>{c}</td>" for i, c in enumerate(row))
        body.append(f"<tr>{cells}</tr>")
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def _mark(ok: bool, yes: str = "OK", no: str = "CHECK") -> str:
    return f'<span class="ok">{yes}</span>' if ok else f'<span class="bad">{no}</span>'


def _where(r: dict[str, Any]) -> str:
    printed = f" (printed page {_e(r['printed_page'])})" if r.get("printed_page") else ""
    return f"{_e(r.get('doc', ''))}, PDF page {_e(r.get('page', ''))}{printed}"


def render(qa: dict[str, Any], full: bool = False) -> str:
    """The HTML report. With `full=False` (the committed report) no NaCCA sentence is written."""
    n = qa["indicators"] or 1
    rate = len(qa["truncated"]) / n
    share = (qa["indicators"] - len(qa["without_exemplar"])) / n
    out: list[str] = []
    add = out.append
    add(f"<h1>Curriculum check: {_e(qa['title'])}</h1>")
    mode = (
        "This copy shows curriculum text and must stay out of git."
        if full
        else (
            "This report holds no curriculum text. Indicators are named by code, by Kwan's own short label and by page."
        )
    )
    add(
        f'<p class="muted">Made on {_e(qa["generated_on"])} by <code>kwan curriculum qa</code> from the '
        f"{_e(qa['engine'])} extraction. {mode}</p>"
    )
    tiles = [
        (qa["indicators"], f"indicators found (hand count: {qa['expected']})", qa["counts_match"]),
        (
            f"{rate:.2%}",
            f"descriptions that stop mid-sentence ({len(qa['truncated'])}); target below 0.50%",
            rate < 0.005,
        ),
        (f"{share:.1%}", "indicators with at least one exemplar; target 90% or more", share >= 0.9),
        (qa["exemplars"], f"exemplars in all, {qa['exemplars_ghana']} of them with Ghanaian context", True),
        (len(qa["duplicate_ids"]), "duplicate ids", not qa["duplicate_ids"] and not qa["alias_clashes"]),
        (
            qa["labels"]["reviewed"],
            f"short labels reviewed by a teacher ({qa['labels']['draft']} are still drafts)",
            True,
        ),
    ]
    add(
        '<div class="tiles">'
        + "".join(
            f'<div class="tile"><b class="{"ok" if ok else "bad"}">{_e(v)}</b><span>{_e(t)}</span></div>'
            for v, t, ok in tiles
        )
        + "</div>"
    )

    add("<h2>1. Sources</h2>")
    if qa["sources"]:
        add(
            _table(
                ["Document", "File", "Size (bytes)", "SHA-256", "Fetched"],
                [
                    [
                        _e(s["id"]),
                        _e(s["file"]),
                        _e(s["size_bytes"]),
                        f"<code>{_e(s['sha256'])}</code>",
                        _e(s["fetched_at"]),
                    ]
                    for s in qa["sources"]
                ],
                numeric=(2,),
            )
        )
    else:
        add('<p class="warn">No sources are recorded yet. Run <code>kwan curriculum fetch</code>.</p>')

    add("<h2>2. Counts against the hand count</h2>")
    add(
        f"<p>The hand count is <code>data/curriculum/expected_counts.json</code>. The parser must match it exactly. "
        f"Result: {_mark(qa['counts_match'], 'every row matches', 'some rows differ')}.</p>"
    )
    grades = sorted(qa["per_grade"], key=int)
    add(
        _table(
            ["Grade", "Found", "Hand count", ""],
            [
                [
                    f"Basic {g}",
                    qa["per_grade"][g],
                    qa["expected_per_grade"].get(g, ""),
                    _mark(qa["per_grade"][g] == qa["expected_per_grade"].get(g)),
                ]
                for g in grades
            ]
            + [["All", qa["indicators"], qa["expected"], _mark(qa["indicators"] == qa["expected"])]],
            numeric=(1, 2),
        )
    )
    add("<details open><summary>Every grade, strand and sub-strand</summary>")
    add(
        _table(
            ["Grade", "Strand", "Sub-strand", "Found", "Hand count", ""],
            [
                [
                    f"Basic {r['grade']}",
                    f"{r['strand']} {_e(r['strand_name'])}",
                    f"{r['sub_strand']} {_e(r['sub_strand_name'])}",
                    r["found"],
                    r["expected"],
                    _mark(r["found"] == r["expected"]),
                ]
                for r in qa["rows"]
            ],
            numeric=(3, 4),
        )
    )
    add("</details>")
    if qa["gaps"]:
        add("<p>Numbers that the PDF itself skips. No indicator is made up for them.</p>")
        add(
            _table(
                ["Where", "Missing from the PDF", "Note"],
                [[_e(g["where"]), _e(", ".join(g["gap"])), _e(g["note"])] for g in qa["gaps"]],
            )
        )

    add("<h2>3. Truncation check</h2>")
    add(
        f"<p>A description counts as cut short when it is empty, ends with a comma or hyphen, or its last word is one of: "
        f"<code>{_e(' '.join(sorted(STOP_WORDS)))}</code>. Found: {len(qa['truncated'])} of {qa['indicators']} ({rate:.2%}).</p>"
    )
    if qa["truncated"]:
        rows = [
            [
                f"<code>{_e(r['code'])}</code>",
                _e(r["short_label"]),
                _where(r),
                f"<code>{_e(last_word(r['description']))}</code>",
            ]
            + ([f'<span class="text">{_e(r["description"])}</span>'] if full else [])
            for r in qa["truncated"]
        ]
        add(_table(["Code", "Short label", "Where", "Last word"] + (["Description"] if full else []), rows))
        add("<p>Open the PDF at each page to see whether the statement really ends there in print.</p>")

    add("<h2>4. Duplicate ids</h2>")
    if qa["duplicate_ids"] or qa["alias_clashes"]:
        add(
            f'<p class="bad">Duplicate ids: {_e(", ".join(qa["duplicate_ids"]) or "none")}. '
            f"Codes or aliases used by more than one indicator: {_e(', '.join(qa['alias_clashes']) or 'none')}.</p>"
        )
    else:
        add("<p>None. Every id is used once and no alias points at two indicators.</p>")

    add("<h2>5. Indicators without exemplars</h2>")
    if qa["without_exemplar"]:
        add(
            _table(
                ["Code", "Short label", "Where"],
                [[f"<code>{_e(r['code'])}</code>", _e(r["short_label"]), _where(r)] for r in qa["without_exemplar"]],
            )
        )
    else:
        add(
            f"<p>None. All {qa['indicators']} indicators have at least one exemplar "
            f"({qa['exemplars']} exemplars; {qa['indicators_with_ghana_exemplar']} indicators have an exemplar with Ghanaian context).</p>"
        )

    add("<h2>6. Corrections and warnings</h2>")
    fixes = [x for x in qa["log"] if x["kind"].startswith("correction:")]
    add("<p>Misprints in the PDFs and what the parser did, from <code>data/curriculum/corrections.json</code>.</p>")
    add(
        _table(
            ["Canonical code", "Action", "Document", "PDF page", "What and why"],
            [
                [
                    f"<code>{_e(x['code'])}</code>",
                    _e(x["kind"].split(":", 1)[1]),
                    _e(x["doc"]),
                    _e(x["page"]),
                    _e(x["detail"]),
                ]
                for x in fixes
            ],
            numeric=(3,),
        )
    )
    other = [x for x in qa["log"] if not x["kind"].startswith("correction:")]
    if other:
        add(
            "<p>Other things the parser noticed. A content-standard mismatch means the code printed in the left column "
            "does not fit the indicator beside it; each one was checked when the corrections were written.</p>"
        )
        add(
            _table(
                ["Level", "Kind", "Code", "Document", "PDF page", "Detail"],
                [
                    [
                        _mark(x["level"] != "error", _e(x["level"]), "error"),
                        _e(x["kind"]),
                        f"<code>{_e(x['code'])}</code>",
                        _e(x["doc"]),
                        _e(x["page"]),
                        _e(x["detail"]),
                    ]
                    for x in other
                ],
                numeric=(4,),
            )
        )

    add("<h2>7. pdfplumber against pdftotext</h2>")
    cmp_ = qa["engines"].get("comparison", {})
    if "unavailable" in cmp_:
        add(f'<p class="warn">{_e(cmp_["unavailable"])}</p>')
    else:
        a, b = qa["engine"], qa["other_engine"]
        add(
            f"<p>The files are built from the <b>{_e(a)}</b> path. The <b>{_e(b)}</b> path is the fallback and is run here only to compare.</p>"
        )
        add(
            _table(
                ["", a, b],
                [
                    [_e(label), qa["engines"][a][key], qa["engines"][b][key]]
                    for label, key in (
                        ("Indicators found", "indicators"),
                        ("Descriptions cut short", "truncated"),
                        ("Indicators with an exemplar", "with_exemplar"),
                        ("Exemplars", "exemplars"),
                    )
                ],
                numeric=(1, 2),
            )
        )
        add(
            f"<p>Same set of ids: {_mark(cmp_['same_ids'], 'yes', 'no')}. Descriptions that read the same: "
            f"{cmp_['descriptions_equal']} of {qa['indicators']}.</p>"
        )
        for title, key in (
            (f"Only in {a}", "only_in_primary"),
            (f"Only in {b}", "only_in_other"),
            ("Descriptions that differ", "descriptions_differ"),
            ("Exemplar counts that differ", "exemplar_counts_differ"),
        ):
            if cmp_[key]:
                add(f"<p>{_e(title)} ({len(cmp_[key])}): <code>{_e(', '.join(cmp_[key]))}</code></p>")

    add("<h2>8. Sub-strand names</h2>")
    add(
        "<p>Where each name comes from. A name marked corrections is written by Kwan because the PDF prints no clean heading.</p>"
    )
    seen: dict[str, dict[str, Any]] = {}
    for r in qa["records"]:
        seen.setdefault(r["content_standard"].rsplit(".", 1)[0], r)
    add(
        _table(
            ["Sub-strand", "Name", "Source"],
            [
                [f"<code>{_e(k)}</code>", _e(r["sub_strand_name"]), _e(qa["sub_strand_sources"].get(k, ""))]
                for k, r in seen.items()
            ],
        )
    )

    add(f"<h2>9. Spot check: {len(qa['sample'])} random indicators</h2>")
    add(
        "<p>Open the PDF at the page shown and check that the statement under that code says what the short label says, "
        "and that the number of exemplars is right. The sample is the same on every run until the indicator list changes.</p>"
    )
    headers = ["Code", "Short label", "Where", "Words", "Exemplars", "With Ghanaian context"]
    rows = []
    for r in qa["sample"]:
        row = [
            f"<code>{_e(r['code'])}</code>",
            _e(r["short_label"]),
            _where(r),
            len(r["description"].split()),
            len(r["exemplars"]),
            sum(1 for x in r["exemplars"] if x["ghana_context"]),
        ]
        if full:
            first = r["exemplars"][0]["text"] if r["exemplars"] else ""
            row += [f'<span class="text">{_e(r["description"])}</span>', f'<span class="text">{_e(first[:400])}</span>']
        rows.append(row)
    add(_table(headers + (["Description", "First exemplar"] if full else []), rows, numeric=(3, 4, 5)))

    add("<h2>10. Short labels</h2>")
    lab = qa["labels"]
    add(
        f"<p>{lab['reviewed']} reviewed, {lab['draft']} drafts waiting for a teacher's review, {len(lab['missing'])} missing, "
        f"{len(lab['too_long'])} longer than {MAX_LABEL} characters.</p>"
    )
    if lab["missing"] or lab["too_long"]:
        add(
            f'<p class="bad">Missing: <code>{_e(", ".join(lab["missing"]) or "none")}</code>. Too long: <code>{_e(", ".join(lab["too_long"]) or "none")}</code>.</p>'
        )
    title = f"Curriculum check: {qa['subject']}"
    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{_e(title)}</title><style>{CSS}</style></head><body><main>\n"
        + "\n".join(out)
        + "\n</main></body></html>\n"
    )


def run(
    family: Family | None = None,
    root: Path | str | None = None,
    out: Path | str | None = None,
    full: bool = False,
    results: Path | str | None | bool = None,
    today: str | None = None,
) -> dict[str, Any]:
    """Compute the checks, write the report(s) and the results block; returns the QA data.

    `results` is the results file to update (default `data/eval_results.json` under the root); pass
    False to leave it alone.
    """
    family = family or load_family()
    base = root_dir(root)
    qa = compute(family, base, today)
    target = Path(out) if out else family.path("qa_report", base)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render(qa, full=False), encoding="utf-8")
    qa["written"] = [target]
    if full:
        extra = family.path("full_file", base).with_name(f"qa-{family.subject}.full.html")
        extra.write_text(render(qa, full=True), encoding="utf-8")
        qa["written"].append(extra)
    if results is not False:
        from kwan.eval import results as results_file

        path = Path(results) if results else base / "data" / "eval_results.json"
        results_file.update(RESULTS_KEY, results_block(qa), path=path)
        qa["written"].append(path)
    return qa
