"""Command-line entries for practice settings: `kwan context <command>`.

    kwan context check            run the automatic checks on every committed setting
    kwan context merge-drafts     merge the draft files into data/contexts/math.json
    kwan context review-export    write review sheets (CSV) for teachers
    kwan context review-import    apply the decisions of a filled-in review sheet
    kwan context translate-draft  machine-draft local-language translations with the Khaya API
    kwan context coverage         say how much is drafted, checked, reviewed and translated

Nothing here needs more than the standard library.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

SUBJECT_HELP = "subject (default: math)"


def _check(args: argparse.Namespace) -> int:
    from kwan.context import bank, checks, drafts, settings

    failed = False
    if args.bank:
        texts = {f"B{2 if band == 'lower' else 5}.{strand}": list(v) for (strand, band), v in bank.CONTEXTS.items()}
        report = checks.check_texts(texts)
        print("v1 bank (fallback by strand and band):")
    elif args.drafts:
        ds = drafts.load_drafts(subject=args.subject)
        report = checks.check_texts({k: v["settings"] for k, v in ds.items.items()})
        print(f"draft files ({ds.mapping} mapping):")
    else:
        path = Path(args.path) if args.path else settings.contexts_path(args.subject)
        doc = settings.load_contexts(path, args.subject)
        report = checks.check_contexts(doc["settings"])
        print(f"{path}:")
    print("\n".join(report.summary_lines()))
    if report.failures and (args.verbose or len(report.failures) <= 20):
        for indicator, i, text, why in report.failures:
            print(f"  {indicator} [{i}] {text}")
            for f in why:
                print(f"      {f}")
    elif report.failures:
        print("  (run with --verbose to list every failing setting)")
    failed = failed or bool(report.failures)

    if not args.bank:
        labels = drafts.current_labels(drafts.load_drafts(subject=args.subject))
        bad = [(code, v["short_label"], checks.check_label(v["short_label"])) for code, v in labels.items()]
        bad = [b for b in bad if b[2]]
        print(f"{len(labels)} short labels checked: {len(labels) - len(bad)} pass, {len(bad)} fail")
        for code, label, why in bad:
            print(f"  {code} {label!r}: " + "; ".join(str(f) for f in why))
        failed = failed or bool(bad)
    return 1 if failed else 0


def _merge(args: argparse.Namespace) -> int:
    from kwan.context import drafts, settings

    path = Path(args.path) if args.path else settings.contexts_path(args.subject)
    ds = drafts.load_drafts(
        Path(args.drafts_dir) if args.drafts_dir else None,
        Path(args.corrections) if args.corrections else None,
        args.subject,
    )
    doc = settings.load_contexts(path, args.subject)
    result = drafts.merge_drafts(doc, ds, drafts.load_structure_codes(subject=args.subject))
    print("\n".join(result.lines()))
    if args.dry_run:
        print("dry run: nothing was written")
    else:
        settings.save_contexts(doc, path)
        print(f"wrote {path}")
    print("Nothing was marked as reviewed. Reviews come only from `kwan context review-import`.")
    return 1 if result.failing else 0


def _export(args: argparse.Namespace) -> int:
    from kwan.context import sheets

    kinds = {"settings": ("setting",), "labels": ("label",)}.get(args.only, ("label", "setting"))
    result = sheets.export_review(
        out_dir=Path(args.out_dir) if args.out_dir else None,
        batch=args.batch,
        batch_size=args.batch_size,
        all_batches=args.all,
        lang=args.lang,
        grades=args.grade or None,
        kinds=kinds,
        include_drafts=args.include_drafts,
        exemplars=not args.no_exemplars,
        subject=args.subject,
    )
    print("\n".join(result.lines()))
    return 0


def _import(args: argparse.Namespace) -> int:
    from kwan.context import sheets

    try:
        result = sheets.import_review(
            Path(args.csv), date=args.date, override=args.override, dry_run=args.dry_run, subject=args.subject
        )
    except (sheets.SheetError, FileNotFoundError) as exc:
        print(f"kwan context review-import: {exc}", file=sys.stderr)
        return 2
    print("\n".join(result.lines()))
    return 0 if result.ok else 1


def _translate(args: argparse.Namespace) -> int:
    from kwan.context import translate

    try:
        run = translate.machine_draft(
            args.lang,
            limit=args.limit,
            include_drafts=args.include_drafts,
            delay=args.delay,
            dry_run=args.dry_run,
            subject=args.subject,
        )
    except translate.KhayaError as exc:
        print(f"kwan context translate-draft: {exc}", file=sys.stderr)
        return 2
    print("\n".join(run.lines()))
    return 1 if run.errors else 0


def _coverage(args: argparse.Namespace) -> int:
    from kwan.context import coverage

    block = coverage.contexts_block(subject=args.subject)
    print("\n".join(coverage.coverage_lines(block)))
    if args.out:
        from kwan.eval import results

        results.update("contexts", block, path=args.out)
        print(f"wrote the 'contexts' block to {args.out}")
    return 0


def register(subparsers: Any) -> None:
    """Add `context` and its commands to the `kwan` command."""
    ctx = subparsers.add_parser("context", help="practice settings: checks, review sheets, translations, coverage")
    commands = ctx.add_subparsers(dest="context_command", metavar="COMMAND", required=True)

    p = commands.add_parser("check", help="run the automatic checks on every committed setting and short label")
    p.add_argument("--path", metavar="PATH", help="settings file (default: data/contexts/<subject>.json)")
    p.add_argument("--drafts", action="store_true", help="check the draft files instead of the settings file")
    p.add_argument("--bank", action="store_true", help="check the v1 fallback bank instead (for information)")
    p.add_argument("-v", "--verbose", action="store_true", help="list every failing setting")
    p.add_argument("--subject", default="math", help=SUBJECT_HELP)
    p.set_defaults(func=_check)

    p = commands.add_parser("merge-drafts", help="merge data/contexts/drafts/*.json into the settings file")
    p.add_argument("--path", metavar="PATH", help="settings file (default: data/contexts/<subject>.json)")
    p.add_argument("--drafts-dir", metavar="DIR", help="folder with the draft files")
    p.add_argument("--corrections", metavar="PATH", help="corrections file (default: data/curriculum/corrections.json)")
    p.add_argument("--dry-run", action="store_true", help="show what would change and write nothing")
    p.add_argument("--subject", default="math", help=SUBJECT_HELP)
    p.set_defaults(func=_merge)

    p = commands.add_parser("review-export", help="write review sheets (CSV) for what still needs a teacher")
    p.add_argument("--out-dir", metavar="DIR", help="where to write (default: build/review)")
    p.add_argument(
        "--batch", type=int, default=1, help="which sheet to write, counted over what is pending (default: 1)"
    )
    p.add_argument("--batch-size", type=int, default=100, help="settings per sheet (default: 100)")
    p.add_argument("--all", action="store_true", help="write every sheet")
    p.add_argument(
        "--lang", default="en", choices=("en", "tw", "ee", "gaa"), help="en: settings and labels; else translations"
    )
    p.add_argument("--grade", type=int, action="append", choices=range(1, 7), help="only this class (repeat for more)")
    p.add_argument("--only", choices=("settings", "labels"), help="leave out the other kind of row")
    p.add_argument(
        "--include-drafts", action="store_true", help="translations: also those of unreviewed English drafts"
    )
    p.add_argument("--no-exemplars", action="store_true", help="leave the exemplar summary column empty")
    p.add_argument("--subject", default="math", help=SUBJECT_HELP)
    p.set_defaults(func=_export)

    p = commands.add_parser("review-import", help="apply the decisions of a filled-in review sheet")
    p.add_argument("csv", help="the sheet, saved as CSV")
    p.add_argument("--date", metavar="YYYY-MM-DD", help="review date to store (default: today)")
    p.add_argument("--override", action="store_true", help="let this sheet replace decisions that are already stored")
    p.add_argument("--dry-run", action="store_true", help="show what would change and write nothing")
    p.add_argument("--subject", default="math", help=SUBJECT_HELP)
    p.set_defaults(func=_import)

    p = commands.add_parser(
        "translate-draft", help="machine-draft translations with the Khaya API (needs KHAYA_API_KEY)"
    )
    p.add_argument("--lang", required=True, choices=("tw", "ee", "gaa"), help="Twi, Ewe or Ga")
    p.add_argument("--limit", type=int, default=25, help="most API calls in this run (default: 25)")
    p.add_argument("--delay", type=float, default=1.0, help="seconds between calls (default: 1)")
    p.add_argument("--include-drafts", action="store_true", help="also translate unreviewed English drafts")
    p.add_argument("--dry-run", action="store_true", help="call the API but save nothing")
    p.add_argument("--subject", default="math", help=SUBJECT_HELP)
    p.set_defaults(func=_translate)

    p = commands.add_parser("coverage", help="how much is drafted, checked, reviewed and translated")
    p.add_argument("--out", metavar="PATH", help="write the 'contexts' block into this results file")
    p.add_argument("--subject", default="math", help=SUBJECT_HELP)
    p.set_defaults(func=_coverage)
