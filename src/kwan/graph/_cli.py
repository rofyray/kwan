"""Command-line entries for the curriculum graph: `kwan graph build`, `validate` and `export`.

The decision-model commands (`kwan graph augment`, `review-export`, `review-import`) live in
`kwan.graph._cli_augment`; they are added here when that module is present.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import Any


def _build(args: argparse.Namespace) -> int:
    from kwan.graph import build

    try:
        result = build.build(
            structure=args.structure,
            out=args.out,
            full_text=args.full_text,
            full_out=args.full_out,
            augmented=args.augmented,
            at=args.date,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"kwan graph build: {exc}", file=sys.stderr)
        return 1
    s = build.summary(result.graph)
    types = ", ".join(f"{k} {v}" for k, v in s["edges_by_type"].items()) or "no edges"
    print(f"Structure: {result.structure_path}")
    print(f"Graph: {s['nodes']} nodes, {s['edges']} edges ({types}), {s['roots']} roots, "
          f"longest path {s['longest_path_edges']} edges")  # fmt: skip
    if result.augmented is not None:
        a = result.augmented
        print(f"Augmented edges: {a['merged']} merged of {a['accepted']} accepted "
              f"({a['pending']} pending and {a['rejected']} rejected left out)")  # fmt: skip
        for item in a["skipped"]:
            print(f"  skipped {item['source']} -> {item['target']}: {item['reason']}")
    else:
        print("Augmented edges: no augmented file, rule edges only")
    print(f"Wrote {result.path} (labels mode)")
    if result.full_path is not None:
        print(f"Wrote {result.full_path} (full-text mode, not for commit); "
              f"{len(result.without_text)} nodes have no curriculum text")  # fmt: skip
    else:
        print("No full-text file found, so no full-text graph was written")
    return 0


def _validate(args: argparse.Namespace) -> int:
    from kwan.graph import validate

    try:
        stats = validate.validate(
            graph=args.graph, report=args.report, results=args.results, write_results=not args.no_results
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"kwan graph validate: {exc}", file=sys.stderr)
        return 1
    for name, passed in stats["checks"].items():
        print(f"{'pass' if passed else 'FAIL'}  {name.replace('_', ' ')}")
    print(f"{stats['nodes']} nodes, {stats['edges']} edges, {stats['roots']} roots, "
          f"{len(stats['isolated_nodes'])} isolated, longest path {stats['longest_path_edges']} edges "
          f"(weighted {stats['longest_path_weighted']})")  # fmt: skip
    print(f"Report: {args.report or validate.default_report_path(stats['subject'])}")
    return 0 if stats["ok"] else 1


def _export(args: argparse.Namespace) -> int:
    from kwan.graph import build, export, schema

    path = args.graph or (build.default_paths()["full_out"] if args.full else None)
    try:
        G = schema.load_graph(path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"kwan graph export: {exc}", file=sys.stderr)
        return 1
    if not args.full and G.graph.get("text_mode") == "full":
        G = schema.strip_text(G)  # full text leaves the build folder only when it is asked for
    out = export.write_html(G, args.out, grades=args.grades, inline=args.inline)
    note = " (full-text mode, do not commit)" if G.graph.get("text_mode") == "full" else " (labels mode)"
    print(f"Wrote {out}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges{note}")
    return 0


def register(subparsers: Any) -> None:
    """Add `graph` (with `build`, `validate`, `export` and the decision-model commands) to `kwan`."""
    graph = subparsers.add_parser("graph", help="build, check and draw the curriculum knowledge graph")
    commands = graph.add_subparsers(dest="graph_command", metavar="ACTION", required=True)

    p = commands.add_parser("build", help="build the graph from the curriculum structure file")
    p.add_argument(
        "--structure", metavar="PATH", help="structure records (default data/curriculum/math.structure.json)"
    )
    p.add_argument("--out", metavar="PATH", help="labels-mode graph to write (default data/graph/math.graph.json)")
    p.add_argument("--full-text", metavar="PATH", help="full-text records (default build/curriculum/math.full.json)")
    p.add_argument("--full-out", metavar="PATH", help="full-text graph (default build/graph/math.graph.full.json)")
    p.add_argument(
        "--augmented", metavar="PATH", help="reviewed augmented edges (default data/graph/math.augmented.json)"
    )
    p.add_argument(
        "--date", metavar="YYYY-MM-DD", help="date for new rule edges (default today; old edges keep theirs)"
    )
    p.set_defaults(func=_build)

    p = commands.add_parser("validate", help="check the graph and write docs/qa/graph-math.md")
    p.add_argument("--graph", metavar="PATH", help="graph file to check (default data/graph/math.graph.json)")
    p.add_argument("--report", metavar="PATH", help="Markdown report to write (default docs/qa/graph-math.md)")
    p.add_argument("--results", metavar="PATH", help="results file to update (default data/eval_results.json)")
    p.add_argument("--no-results", action="store_true", help="do not touch the results file")
    p.set_defaults(func=_validate)

    p = commands.add_parser("export", help="write an interactive graph.html (needs pyvis)")
    p.add_argument("--out", metavar="PATH", required=True, help="HTML file to write, for example docs/graph.html")
    p.add_argument("--graph", metavar="PATH", help="graph file to draw (default data/graph/math.graph.json)")
    p.add_argument("--grades", metavar="N", type=int, nargs="+", help="draw only these grades, for example 4 5 6")
    p.add_argument("--full", action="store_true", help="show curriculum descriptions (full-text graph; do not commit)")
    p.add_argument("--inline", action="store_true", help="put the drawing library inside the page (works offline)")
    p.set_defaults(func=_export, extra="demo")

    try:
        augment = importlib.import_module("kwan.graph._cli_augment")
    except ImportError:  # the decision-model commands are not part of this install
        return
    augment.register_into(commands)
