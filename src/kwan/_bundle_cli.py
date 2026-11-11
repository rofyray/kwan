"""`kwan bundle`: write `kwan-bundle.<subject>.json` from the committed data files (PRD 9.4)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from kwan.bundle import TEXT_MODES, BundleError, build_bundle, bundle_file_name, dumps_bundle

DEFAULT_OUT = "build/bundles"


def _bundle(args: argparse.Namespace) -> int:
    from kwan.learner.item_bank import ItemBank

    params = provenance = None
    if args.params:
        from kwan.bundle import load_params_file

        params, provenance = load_params_file(args.params)
    item_bank = None
    if args.item_bank:
        import json

        with open(args.item_bank, encoding="utf-8") as f:
            item_bank = ItemBank.from_dict(json.load(f))
    try:
        doc = build_bundle(
            subject=args.subject,
            text_mode=args.text,
            params=params,
            params_provenance=provenance,
            item_bank=item_bank,
            out=args.out,
            include_drafts=not args.reviewed_only,
            bundle_version=args.bundle_version,
        )
    except (BundleError, FileNotFoundError) as exc:
        print(f"kwan bundle: {exc}", file=sys.stderr)
        return 1
    target = Path(args.out)
    if target.is_dir() or not target.suffix:
        target = target / bundle_file_name(args.subject)
    size = len(dumps_bundle(doc).encode("utf-8"))
    prov = doc["provenance"]
    print(
        f"wrote {target} ({size:,} bytes): {doc['bundle_version']}, text {doc['text_mode']}, "
        f"{prov['graph']['nodes']} skills, {prov['graph']['edges']} links, "
        f"{prov['contexts']['reviewed']} reviewed and {prov['contexts']['drafts']} draft settings, "
        f"params from {prov['params'].get('source', 'defaults')}, commit {doc['source_commit'][:12]}"
    )
    return 0


def register(subparsers: Any) -> None:
    """Add `bundle` to the `kwan` command."""
    p = subparsers.add_parser("bundle", help="write kwan-bundle.<subject>.json from the committed data files")
    p.add_argument("--subject", default="math", help="subject to bundle (default: math)")
    p.add_argument(
        "--text",
        choices=TEXT_MODES,
        default="labels",
        help="labels: Kwan's short labels (what the project ships); full: NaCCA text, only with recorded permission",
    )
    p.add_argument(
        "--out", default=DEFAULT_OUT, metavar="PATH", help=f"directory or file to write (default: {DEFAULT_OUT})"
    )
    p.add_argument(
        "--params",
        metavar="FILE",
        help="fitted parameters JSON (default: data/params/<subject>.params.json if present)",
    )
    p.add_argument("--item-bank", metavar="FILE", help="item-difficulty bank JSON to include")
    p.add_argument("--reviewed-only", action="store_true", help="leave unreviewed practice settings out of the bundle")
    p.add_argument(
        "--bundle-version", metavar="TEXT", help="override the bundle version (default: <subject>-<package version>)"
    )
    p.set_defaults(func=_bundle, extra=None)
