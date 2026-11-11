"""Check a curriculum graph and write the validation report `docs/qa/graph-math.md`.

The report holds codes and numbers only. It never quotes a curriculum sentence, so it is safe to
commit whatever text mode the checked graph is in.

Three checks must pass: the graph is a DAG, no edge points to a node that is not listed, and every
node can be reached from a root (a node with nothing before it). The rest of the report describes
the graph: longest path, degrees by grade, edges by type and by provenance, isolated nodes.
"""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Hashable, Mapping
from pathlib import Path
from typing import Any

import networkx as nx

from kwan import ids
from kwan._paths import data_dir
from kwan.graph import schema

RESULTS_KEY = "graph"

__all__ = ["RESULTS_KEY", "analyse", "default_report_path", "render_report", "results_block", "validate"]


def default_report_path(subject: str = ids.DEFAULT_SUBJECT) -> Path:
    """`docs/qa/graph-<subject>.md` next to the data directory."""
    return data_dir().parent / "docs" / "qa" / f"graph-{subject}.md"


def _code(n: Hashable) -> str:
    return ids.code_of(str(n))


def _codes(nodes: Any) -> list[str]:
    return [_code(n) for n in sorted(nodes, key=schema.sort_key)]


def _mean(values: list[int]) -> float:
    return round(sum(values) / len(values), 2) if values else 0.0


def analyse(G: nx.DiGraph) -> dict[str, Any]:
    """Every number of the validation report, as plain JSON types.

    Load the graph with `strict=False` to have edges that point to unlisted nodes counted here
    instead of raising at load time.
    """
    listed = [n for n, d in G.nodes(data=True) if d]  # a node that only an edge mentions has no attributes
    unlisted = set(G) - set(listed)
    orphan_edges = [(a, b) for a, b in G.edges if a in unlisted or b in unlisted]
    is_dag = nx.is_directed_acyclic_graph(G)
    roots = [n for n in listed if G.in_degree(n) == 0]
    leaves = [n for n in listed if G.out_degree(n) == 0]
    isolated = [n for n in listed if G.degree(n) == 0]
    reached: set[Hashable] = set(roots)
    stack = list(roots)
    while stack:
        for nxt in G.successors(stack.pop()):
            if nxt not in reached:
                reached.add(nxt)
                stack.append(nxt)
    unreachable = [n for n in listed if n not in reached]

    longest_edges = longest_weighted = None
    longest_path: list[str] = []
    cycle: list[str] = []
    if is_dag:
        longest_edges = nx.dag_longest_path_length(G, weight=None)
        longest_weighted = round(float(nx.dag_longest_path_length(G, weight="weight")), 4)
        longest_path = [_code(n) for n in nx.dag_longest_path(G, weight=None)]
    else:
        found = nx.find_cycle(G)
        cycle = [_code(a) for a, _ in found] + [_code(found[-1][1])]

    grade_of = {n: G.nodes[n].get("grade") for n in listed}
    by_grade: dict[str, dict[str, Any]] = {}
    for grade in sorted({g for g in grade_of.values() if g is not None}):
        members = [n for n in listed if grade_of[n] == grade]
        ins = [G.in_degree(n) for n in members]
        outs = [G.out_degree(n) for n in members]
        by_grade[str(grade)] = {
            "nodes": len(members),
            "roots": sum(1 for x in ins if x == 0),
            "leaves": sum(1 for x in outs if x == 0),
            "edges_in": sum(ins),
            "edges_out": sum(outs),
            "in_degree_mean": _mean(ins),
            "in_degree_max": max(ins, default=0),
            "out_degree_mean": _mean(outs),
            "out_degree_max": max(outs, default=0),
            "edges_inside": sum(1 for a, b in G.edges if grade_of.get(a) == grade and grade_of.get(b) == grade),
            "edges_to_later_grades": sum(
                1 for a, b in G.edges if grade_of.get(a) == grade and (grade_of.get(b) or 0) > grade
            ),
        }
    strand_names: dict[str, str] = {}
    by_strand: Counter[str] = Counter()
    for n in listed:
        strand = G.nodes[n].get("strand")
        if strand is not None:
            by_strand[str(strand)] += 1
            strand_names.setdefault(str(strand), str(G.nodes[n].get("strand_name") or ""))

    by_type: Counter[str] = Counter()
    by_provenance: Counter[str] = Counter()
    by_review: Counter[str] = Counter()
    no_provenance = 0
    for *_, d in G.edges(data=True):
        by_type[str(d.get("type"))] += 1
        prov = d.get("provenance")
        if isinstance(prov, Mapping) and prov.get("by"):
            by_provenance[str(prov["by"])] += 1
            if isinstance(prov.get("review"), Mapping):
                by_review[str(prov["review"].get("status"))] += 1
        else:
            no_provenance += 1
    backward = [
        (a, b)
        for a, b in G.edges
        if grade_of.get(a) is not None and grade_of.get(b) is not None and grade_of[a] > grade_of[b]
    ]
    checks = {"is_dag": is_dag, "no_orphan_edges": not orphan_edges, "all_reachable_from_a_root": not unreachable}
    return {
        "subject": G.graph.get("subject") or ids.DEFAULT_SUBJECT,
        "schema_version": G.graph.get("schema_version"),
        "text_mode": G.graph.get("text_mode"),
        "ok": all(checks.values()),
        "checks": checks,
        "nodes": len(listed),
        "edges": G.number_of_edges(),
        "edges_by_type": dict(sorted(by_type.items())),
        "edges_by_provenance": dict(sorted(by_provenance.items())),
        "edges_without_provenance": no_provenance,
        "augmented_edges_by_review": dict(sorted(by_review.items())),
        "orphan_edges": [[str(a), str(b)] for a, b in orphan_edges],
        "roots": len(roots),
        "root_codes": _codes(roots),
        "leaves": len(leaves),
        "isolated_nodes": _codes(isolated),
        "unreachable_from_a_root": _codes(unreachable),
        "components": nx.number_weakly_connected_components(G) if len(G) else 0,
        "longest_path_edges": longest_edges,
        "longest_path_weighted": longest_weighted,
        "longest_path": longest_path,
        "cycle": cycle,
        "edges_to_an_earlier_grade": [[_code(a), _code(b)] for a, b in backward],
        "nodes_by_strand": {k: {"name": strand_names[k], "nodes": v} for k, v in sorted(by_strand.items())},
        "by_grade": by_grade,
    }


def results_block(stats: Mapping[str, Any], graph_file: str | None = None, report: str | None = None) -> dict[str, Any]:
    """The key numbers of `analyse`, as stored under `graph` in `data/eval_results.json`."""
    block: dict[str, Any] = {
        "graph_file": graph_file,
        "report": report,
        "subject": stats["subject"],
        "schema_version": stats["schema_version"],
        "text_mode": stats["text_mode"],
        "ok": stats["ok"],
        "is_dag": stats["checks"]["is_dag"],
        "nodes": stats["nodes"],
        "edges": stats["edges"],
        "edges_by_type": stats["edges_by_type"],
        "edges_by_provenance": stats["edges_by_provenance"],
        "orphan_edges": len(stats["orphan_edges"]),
        "roots": stats["roots"],
        "leaves": stats["leaves"],
        "isolated_nodes": len(stats["isolated_nodes"]),
        "unreachable_from_a_root": len(stats["unreachable_from_a_root"]),
        "components": stats["components"],
        "longest_path_edges": stats["longest_path_edges"],
        "longest_path_weighted": stats["longest_path_weighted"],
        "edges_to_an_earlier_grade": len(stats["edges_to_an_earlier_grade"]),
        "nodes_by_grade": {g: row["nodes"] for g, row in stats["by_grade"].items()},
        "nodes_by_strand": {k: row["nodes"] for k, row in stats["nodes_by_strand"].items()},
    }
    return block


def _table(header: list[str], rows: list[list[Any]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return out


def _passed(flag: bool) -> str:
    return "pass" if flag else "FAIL"


def _code_list(codes: list[str], limit: int = 60) -> str:
    if not codes:
        return "none"
    shown = ", ".join(f"`{c}`" for c in codes[:limit])
    return shown if len(codes) <= limit else f"{shown} and {len(codes) - limit} more"


def render_report(stats: Mapping[str, Any], graph_file: str | None = None) -> str:
    """The Markdown text of the validation report. Codes and numbers only."""
    subject = stats["subject"]
    checks = stats["checks"]
    lines = [
        f"# Graph validation: {subject}",
        "",
        "This page is written by `kwan graph validate`. Do not edit it by hand: rebuild the graph with",
        "`kwan graph build` and run `kwan graph validate` again. It lists codes and numbers only.",
        "",
    ]
    if graph_file:
        lines += [
            f"Graph file: `{graph_file}` (schema version {stats['schema_version']}, {stats['text_mode']} mode).",
            "",
        ]
    lines += [
        "## Result",
        "",
        f"**{'All checks pass.' if stats['ok'] else 'At least one check fails.'}**",
        "",
        "| Check | Result | Detail |",
        "|---|---|---|",
        f"| The graph has no cycle (it is a DAG) | {_passed(checks['is_dag'])} | "
        + ("no cycle found" if checks["is_dag"] else "cycle: " + " -> ".join(f"`{c}`" for c in stats["cycle"]))
        + " |",
        f"| No edge points to a node that is not listed | {_passed(checks['no_orphan_edges'])} | "
        f"orphan edges: {len(stats['orphan_edges'])} |",
        f"| Every node can be reached from a root | {_passed(checks['all_reachable_from_a_root'])} | "
        f"nodes not reached: {len(stats['unreachable_from_a_root'])} |",
        "",
        "## Size",
        "",
        *_table(
            ["Measure", "Value"],
            [
                ["Nodes (indicators)", stats["nodes"]],
                ["Edges (prerequisite links)", stats["edges"]],
                ["Roots (nothing comes before)", stats["roots"]],
                ["Leaves (nothing comes after)", stats["leaves"]],
                ["Isolated nodes (no edge at all)", len(stats["isolated_nodes"])],
                ["Connected parts", stats["components"]],
                ["Edges that point to an earlier grade", len(stats["edges_to_an_earlier_grade"])],
            ],
        ),
        "",
    ]
    if stats["nodes_by_strand"]:
        rows = [[f"{k}: {row['name']}".rstrip(": "), row["nodes"]] for k, row in stats["nodes_by_strand"].items()]
        lines += ["Nodes by strand:", "", *_table(["Strand", "Nodes"], rows), ""]
    lines += ["## Longest path", ""]
    if checks["is_dag"]:
        lines += [
            f"The longest chain of prerequisites has **{stats['longest_path_edges']} edges**. "
            f"Counted by weight (a bridge edge counts 0.5 and an augmented edge counts its probability), "
            f"the heaviest chain weighs **{stats['longest_path_weighted']}**.",
            "",
            "Longest chain by number of edges: " + " -> ".join(f"`{c}`" for c in stats["longest_path"]),
            "",
        ]
    else:
        lines += ["Not measured, because the graph has a cycle.", ""]
    lines += [
        "## Degree by grade",
        "",
        "In-degree is the number of edges that arrive at a node (its direct prerequisites). "
        "Out-degree is the number of edges that leave it.",
        "",
        *_table(
            ["Grade", "Nodes", "Roots", "Leaves", "Edges in", "Mean in", "Max in", "Edges out", "Mean out", "Max out"],
            [
                [
                    f"B{g}",
                    r["nodes"],
                    r["roots"],
                    r["leaves"],
                    r["edges_in"],
                    f"{r['in_degree_mean']:.2f}",
                    r["in_degree_max"],
                    r["edges_out"],
                    f"{r['out_degree_mean']:.2f}",
                    r["out_degree_max"],
                ]
                for g, r in stats["by_grade"].items()
            ],
        ),
        "",
        "Where the edges of each grade go:",
        "",
        *_table(
            ["Grade", "Edges inside the grade", "Edges to a later grade"],
            [[f"B{g}", r["edges_inside"], r["edges_to_later_grades"]] for g, r in stats["by_grade"].items()],
        ),
        "",
        "## Edges by type",
        "",
        *_table(["Type", "Edges"], [[k, v] for k, v in stats["edges_by_type"].items()] or [["(none)", 0]]),
        "",
        "## Edges by provenance",
        "",
        "Provenance says what made an edge: a rule (`rule:<name>`) or the decision model, whose edges "
        "enter the graph only after a reviewer accepts them.",
        "",
        *_table(
            ["Made by", "Edges"], [[f"`{k}`", v] for k, v in stats["edges_by_provenance"].items()] or [["(none)", 0]]
        ),
        "",
        f"Edges without a provenance record: {stats['edges_without_provenance']}.",
        "",
    ]
    if stats["augmented_edges_by_review"]:
        rows = [[k, v] for k, v in stats["augmented_edges_by_review"].items()]
        lines += ["Reviewed edges by review status:", "", *_table(["Status", "Edges"], rows), ""]
    lines += [
        "## Isolated nodes",
        "",
        "An isolated node has no edge. The rules give it none when its content standard has one indicator, "
        "appears in one grade only and is the only content standard of its sub-strand in that grade.",
        "",
        f"Isolated nodes ({len(stats['isolated_nodes'])}): {_code_list(stats['isolated_nodes'])}",
        "",
        "## Roots",
        "",
        f"Roots ({stats['roots']}): {_code_list(stats['root_codes'])}",
        "",
    ]
    problems: list[str] = []
    if stats["orphan_edges"]:
        problems += ["Orphan edges: " + ", ".join(f"`{a}` -> `{b}`" for a, b in stats["orphan_edges"][:60]), ""]
    if stats["unreachable_from_a_root"]:
        problems += ["Nodes not reached from a root: " + _code_list(stats["unreachable_from_a_root"]), ""]
    if stats["edges_to_an_earlier_grade"]:
        pairs = ", ".join(f"`{a}` -> `{b}`" for a, b in stats["edges_to_an_earlier_grade"][:60])
        problems += ["Edges that point to an earlier grade: " + pairs, ""]
    if problems:
        lines += ["## Problems to look at", "", *problems]
    return "\n".join(lines).rstrip("\n") + "\n"


def _shown(path: Path) -> str:
    """The path relative to the repository when it is inside it, with forward slashes."""
    root = data_dir().parent
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def validate(
    graph: str | os.PathLike[str] | None = None,
    report: str | os.PathLike[str] | None = None,
    results: str | os.PathLike[str] | None = None,
    write_results: bool = True,
) -> dict[str, Any]:
    """Check a graph file, write the Markdown report and merge the key numbers into the results file.

    Paths left as None take the standard locations: `data/graph/math.graph.json`,
    `docs/qa/graph-math.md` and `data/eval_results.json`. Returns the numbers of `analyse`.
    """
    graph_path = Path(graph) if graph else schema.default_graph_path()
    G = schema.load_graph(graph_path, strict=False)
    stats = analyse(G)
    report_path = Path(report) if report else default_report_path(stats["subject"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(stats, graph_file=_shown(graph_path)), encoding="utf-8")
    if write_results:
        from kwan.eval import results as results_file

        block = results_block(stats, graph_file=_shown(graph_path), report=_shown(report_path))
        # worked out now, not at import time, so that KWAN_DATA_DIR is respected
        results_file.update(RESULTS_KEY, block, path=Path(results) if results else data_dir() / "eval_results.json")
    return stats
