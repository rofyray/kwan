"""Command-line entries for the curriculum pipeline: `kwan curriculum fetch|parse|qa`.

Nothing heavy is imported here; each handler imports what it needs when the command runs.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from typing import Any

EXTRA = "curriculum"


def _family(args: argparse.Namespace) -> Any:
    from kwan.curriculum.families import load_family

    return load_family(args.family, args.families_file)


def _fetch(args: argparse.Namespace) -> int:
    from kwan.curriculum import fetch

    family = _family(args)
    try:
        report = fetch.fetch(family, root=args.root, refresh=args.refresh, accept_changes=args.accept_changes)
    except fetch.HashChanged as exc:
        print(str(exc), file=sys.stderr)
        print("kwan curriculum fetch: stopped, nothing was recorded.", file=sys.stderr)
        return 1
    print(report.text())
    print(f"Recorded in {fetch.sources_path(args.root)}")
    return 0


def _parse(args: argparse.Namespace) -> int:
    from kwan.curriculum import build

    family = _family(args)
    engine = args.engine
    try:
        result = build.build(family, engine=engine, root=args.root)
    except ImportError as exc:
        if engine != "pdfplumber" or args.no_fallback:
            raise
        print(f"pdfplumber is not available ({exc}); falling back to pdftotext.", file=sys.stderr)
        result = build.build(family, engine="pdftotext", root=args.root)
    grades = Counter(r["grade"] for r in result.records)
    print(
        f"{len(result.records)} indicators ({result.engine}): "
        + ", ".join(f"B{g} {n}" for g, n in sorted(grades.items()))
    )
    exemplars = sum(len(r["exemplars"]) for r in result.records)
    print(f"{exemplars} exemplars; {sum(1 for r in result.records if not r['exemplars'])} indicators without one")
    labels = result.labels
    print(
        f"short labels: {len(labels['reviewed'])} reviewed, {len(labels['missing'])} missing"
        + (f" ({', '.join(labels['missing'])})" if labels["missing"] else "")
    )
    if labels["same_as_description"]:
        same = ", ".join(labels["same_as_description"])
        print(f"draft labels that repeat the statement word for word (not used): {same}")
    if labels["unmatched_draft_keys"]:
        print(f"draft keys that match no indicator: {', '.join(labels['unmatched_draft_keys'])}")
    for path in result.written:
        print(f"wrote {path}")
    errors = result.parse.errors()
    for e in errors:
        print(f"ERROR {e['kind']} {e['code']} ({e['doc']} page {e['page']}): {e['detail']}", file=sys.stderr)
    return 1 if errors else 0


def _qa(args: argparse.Namespace) -> int:
    from kwan.curriculum import qa

    family = _family(args)
    results: Any = False if args.no_results else args.results
    data = qa.run(family, root=args.root, out=args.out, full=args.full, results=results)
    n = data["indicators"] or 1
    rate = len(data["truncated"]) / n
    share = (data["indicators"] - len(data["without_exemplar"])) / n
    verdict = "match" if data["counts_match"] else "MISMATCH"
    print(f"indicators: {data['indicators']} (hand count {data['expected']}): {verdict}")
    print(f"descriptions cut short: {len(data['truncated'])} ({rate:.2%}; target below 0.50%)")
    print(f"indicators with an exemplar: {share:.1%} (target 90% or more)")
    print(f"duplicate ids: {len(data['duplicate_ids'])}")
    for path in data["written"]:
        print(f"wrote {path}")
    ok = (
        data["counts_match"]
        and rate < 0.005
        and share >= 0.9
        and not data["duplicate_ids"]
        and not data["alias_clashes"]
    )
    if not ok:
        print("kwan curriculum qa: at least one acceptance check failed.", file=sys.stderr)
    return 0 if ok or args.no_fail else 1


def register(subparsers: Any) -> None:
    """Add `curriculum` (with `fetch`, `parse` and `qa`) to the `kwan` command."""
    parser = subparsers.add_parser(
        "curriculum", help="fetch the curriculum PDFs, parse them into indicator files and check the result"
    )
    parser.set_defaults(func=lambda args: parser.print_help(), extra=EXTRA)
    sub = parser.add_subparsers(dest="curriculum_command", metavar="STEP")

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--family", default=None, help="document family in families.yaml (default: primary mathematics)")
        p.add_argument("--families-file", default=None, help="another families.yaml to read")
        p.add_argument("--root", default=None, help="repository root (default: found from the package)")
        p.set_defaults(extra=EXTRA)

    p = sub.add_parser("fetch", help="download the PDFs to raw/curriculum/ and record URL, size, SHA-256 and date")
    common(p)
    p.add_argument("--refresh", action="store_true", help="download again even when the file is already on disk")
    p.add_argument("--accept-changes", action="store_true", help="record a changed hash instead of failing")
    p.set_defaults(func=_fetch)

    p = sub.add_parser(
        "parse", help="write data/curriculum/<subject>.structure.json and build/curriculum/<subject>.full.json"
    )
    common(p)
    p.add_argument("--engine", choices=("pdfplumber", "pdftotext"), default="pdfplumber", help="extraction path")
    p.add_argument(
        "--no-fallback", action="store_true", help="fail instead of using pdftotext when pdfplumber is missing"
    )
    p.set_defaults(func=_parse)

    p = sub.add_parser("qa", help="check the parsed curriculum and write the HTML report")
    common(p)
    p.add_argument("--out", default=None, help="where to write the report (default: docs/qa/curriculum-<subject>.html)")
    p.add_argument(
        "--full", action="store_true", help="also write a report with curriculum text under build/ (never commit it)"
    )
    p.add_argument("--results", default=None, help="results file to update (default: data/eval_results.json)")
    p.add_argument("--no-results", action="store_true", help="do not touch the results file")
    p.add_argument("--no-fail", action="store_true", help="exit 0 even when an acceptance check fails")
    p.set_defaults(func=_qa)
