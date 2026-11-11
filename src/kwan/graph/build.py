"""Build the Kwan curriculum knowledge graph (a DAG) from indicator records.

Three rules make the edges. Each one reads only the position of an indicator in the curriculum:

* `sequence`: inside one content standard, indicator i comes before indicator i+1 (weight 1.0).
* `vertical`: the last indicator of a content standard in one grade comes before the first indicator
  of the same content standard in the next grade that has it (weight 1.0).
* `bridge`: inside one grade and sub-strand, the last indicator of a content standard comes before
  the first indicator of the next content standard (weight 0.5).

`build_graph`, `save` and `load` are the v1 interface (bare codes, no provenance): `save` writes the v1
node-link file, `load` reads one and also any v2 file (through `kwan.graph.schema.load_graph`).
`DEFAULT_PATH` is the committed v2 graph. `build_rule_graph` and `build` make the schema v2 graph:
canonical ids, a provenance record on every edge, optional curriculum text, and the accepted edges
of the decision model.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx

from kwan import ids
from kwan._paths import data_dir
from kwan.graph import schema

DATA_DIR = data_dir()
DEFAULT_PATH = schema.default_graph_path()  # the committed v2 graph; works from any working directory
RULE_WEIGHT = {"sequence": 1.0, "vertical": 1.0, "bridge": 0.5}


def _cs_order(key: str) -> list[int]:
    return [int(x) for x in key.split(".")]


def rule_edges(indicators: Iterable[Mapping[str, Any]]) -> list[tuple[Any, Any, str, float]]:
    """The rule edges `(source id, target id, type, weight)` of a list of indicator records.

    A record needs `id`, `grade` and `cs_key` ("strand.sub_strand.content_standard"). The order of the
    records inside a content standard is the teaching order.
    """
    by_cs: dict[tuple[int, str], list[Any]] = defaultdict(list)  # (grade, cs_key) -> ordered indicator ids
    for r in indicators:
        by_cs[(r["grade"], r["cs_key"])].append(r["id"])
    out: list[tuple[Any, Any, str, float]] = []
    seen: set[tuple[Any, Any]] = set()

    def add(a: Any, b: Any, kind: str) -> None:
        seen.add((a, b))
        out.append((a, b, kind, RULE_WEIGHT[kind]))

    # 1) sequence edges inside a content standard: indicator i -> i+1
    for members in by_cs.values():
        for a, b in zip(members, members[1:]):
            add(a, b, "sequence")
    # 2) vertical edges: last indicator of cs in grade g -> first indicator of same cs in the next grade that has it
    last_grade = max((g for g, _ in by_cs), default=0)
    for (g, key), members in by_cs.items():
        for g2 in range(g + 1, max(last_grade, 6) + 1):
            if (g2, key) in by_cs:
                add(members[-1], by_cs[(g2, key)][0], "vertical")
                break
    # 3) sub-strand bridge: same grade, same sub-strand, consecutive content standards (weaker)
    sub: dict[tuple[int, str, str], list[list[Any]]] = defaultdict(list)
    for (g, key), members in sorted(by_cs.items(), key=lambda kv: (kv[0][0], _cs_order(kv[0][1]))):
        s, ss, _ = key.split(".")
        sub[(g, s, ss)].append(members)
    for groups in sub.values():
        for a, b in zip(groups, groups[1:]):
            if (a[-1], b[0]) not in seen:
                add(a[-1], b[0], "bridge")
    return out


def build_graph(indicators: Sequence[Mapping[str, Any]]) -> nx.DiGraph:
    """The v1 graph: node keys are the record ids as given, edges carry only `type` and `weight`."""
    G = nx.DiGraph()
    for r in indicators:
        G.add_node(r["id"], **r)
    for a, b, kind, weight in rule_edges(indicators):
        G.add_edge(a, b, type=kind, weight=weight)
    assert nx.is_directed_acyclic_graph(G), "curriculum graph must be a DAG"
    return G


def save(G: nx.DiGraph, path: str | os.PathLike[str]) -> None:
    """Write a graph as a v1 file (networkx node-link JSON). Committed v2 graphs are written by
    `kwan.graph.schema.save_graph`, so this takes no default path."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nx.node_link_data(G, edges="edges"), f, ensure_ascii=False, indent=1)


def load(path: str | os.PathLike[str] = DEFAULT_PATH) -> nx.DiGraph:
    """Read a graph file: a v1 file (networkx node-link JSON, bare ids) as it is, or a v2 file through
    `kwan.graph.schema.load_graph`. The default is the committed v2 graph."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, Mapping) and data.get("schema_version") is None and ("directed" in data or "links" in data):
        return nx.node_link_graph(data, edges="edges" if "edges" in data else "links")
    return schema.graph_from_dict(data)


# ---------------------------------------------------------------- schema v2


def today() -> str:
    """Today's date (UTC) as `YYYY-MM-DD`, the form used in edge provenance."""
    return dt.datetime.now(dt.UTC).date().isoformat()


def structure_from_v1(
    indicators: Iterable[Mapping[str, Any]], subject: str = ids.DEFAULT_SUBJECT
) -> list[dict[str, Any]]:
    """Structure records (ids and numbers only, no curriculum text) from v1 indicator records.

    The short label is the placeholder `"Skill <code>"`. This is a migration helper: it lets the v2
    rules run on a v1 indicator list.
    """
    out = []
    for r in indicators:
        code = ids.code_of(r["id"])
        out.append(
            {
                "id": ids.make(subject, code),
                "subject": subject,
                "code": code,
                "grade": r["grade"],
                "strand": r["strand"],
                "strand_name": r.get("strand_name", ""),
                "sub_strand": r["sub_strand"],
                "sub_strand_name": r.get("sub_strand_name", ""),
                "content_standard": r["content_standard"],
                "cs_key": r["cs_key"],
                "indicator_no": r["indicator_no"],
                "short_label": r.get("short_label") or f"Skill {code}",
                "label_reviewed": bool(r.get("label_reviewed", False)),
                "aliases": list(r.get("aliases") or [code]),
            }
        )
    return out


def read_records(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Read a structure file or a full-text file: a JSON list of records, or an object that holds one."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, Mapping):
        for key in ("indicators", "records", "nodes"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a list of indicator records")
    return data


def _normalise(record: Mapping[str, Any], subject: str | None) -> dict[str, Any]:
    raw = record.get("id") or record.get("code")
    if not raw:
        raise ValueError("indicator record without an 'id' or a 'code'")
    rec_subject, code = ids.split(str(raw), record.get("subject") or subject or ids.DEFAULT_SUBJECT)
    assert rec_subject is not None
    rec = {**record, "id": ids.make(rec_subject, code), "subject": rec_subject, "code": code}
    for key in ("grade", "cs_key"):
        if key not in rec:
            raise ValueError(f"indicator record {rec['id']} has no '{key}'")
    rec["short_label"] = rec.get("short_label") or f"Skill {code}"
    aliases = [a for a in rec.get("aliases") or [] if isinstance(a, str) and a]
    rec["aliases"] = aliases if code in aliases else [code, *aliases]
    return rec


def _teaching_order(rec: Mapping[str, Any]) -> tuple[Any, ...]:
    return (rec["grade"], _cs_order(rec["cs_key"]), rec.get("indicator_no", 0), schema.sort_key(rec["id"]))


def build_rule_graph(
    records: Iterable[Mapping[str, Any]],
    at: str | None = None,
    text_mode: str | None = None,
    subject: str | None = None,
) -> nx.DiGraph:
    """The schema v2 graph of a list of indicator records, with the three rule edges.

    Node keys are canonical ids (`math:B4.1.1.1.2`). Every edge has `type`, `weight` and
    `provenance = {"by": "rule:<type>", "at": <date>, "review": None}`; `at` defaults to today.
    `text_mode="labels"` leaves the curriculum text out even when the records carry it; `"full"`
    keeps it; `None` picks `"full"` only when a record has a description. The records are put in
    curriculum order first, so the result does not depend on the order of the input.
    """
    recs = sorted((_normalise(r, subject) for r in records), key=_teaching_order)
    seen: set[str] = set()
    for r in recs:
        if r["id"] in seen:
            raise ValueError(f"indicator {r['id']} is listed twice")
        seen.add(r["id"])
    if text_mode is None:
        text_mode = "full" if any(r.get("description") not in (None, "", r["short_label"]) for r in recs) else "labels"
    if text_mode not in schema.TEXT_MODES:
        raise ValueError(f"unknown text_mode {text_mode!r}")
    subjects = {r["subject"] for r in recs}
    if len(subjects) > 1:
        raise ValueError(f"one graph holds one subject, got {sorted(subjects)}")
    G = nx.DiGraph(
        schema_version=schema.SCHEMA_VERSION,
        subject=subjects.pop() if subjects else (subject or ids.DEFAULT_SUBJECT),
        text_mode=text_mode,
    )
    for r in recs:
        data = {k: v for k, v in r.items() if k != "id"}
        if text_mode == "labels":
            for k in schema.TEXT_KEYS:
                data.pop(k, None)
        data["description"] = data.get("description") or data["short_label"]
        G.add_node(r["id"], **data)
    at = at or today()
    for a, b, kind, weight in rule_edges(recs):
        G.add_edge(a, b, type=kind, weight=weight, provenance={"by": f"rule:{kind}", "at": at, "review": None})
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("the curriculum graph must be a DAG; the rule edges made a cycle")
    return G


def merge_full_text(G: nx.DiGraph, full_records: Iterable[Mapping[str, Any]]) -> tuple[nx.DiGraph, list[str]]:
    """A full-text copy of `G` with the curriculum text of `full_records`, and the nodes that got none.

    Records are matched by id, bare code or alias. A node without a record keeps its short label as
    its description.
    """
    H = G.copy()
    H.graph["text_mode"] = "full"
    resolver = ids.Resolver(H)
    got: set[str] = set()
    for rec in full_records:
        key = resolver.get(str(rec.get("id") or rec.get("code") or ""))
        if key is None:
            continue
        for k in schema.TEXT_KEYS:
            if rec.get(k) not in (None, "", []):
                H.nodes[key][k] = rec[k]
        if rec.get("description"):
            got.add(key)
    return H, [n for n in H if n not in got]


def review_status(edge: Mapping[str, Any]) -> str:
    """`accepted`, `rejected` or `pending` (also when the edge has no review record)."""
    prov = edge.get("provenance")
    review = prov.get("review") if isinstance(prov, Mapping) else None
    status = review.get("status") if isinstance(review, Mapping) else None
    return status if status in ("accepted", "rejected") else "pending"


def merge_augmented(G: nx.DiGraph, edges: Iterable[Mapping[str, Any]]) -> tuple[nx.DiGraph, dict[str, Any]]:
    """A copy of `G` plus the accepted augmented edges, and a record of what happened.

    Only edges whose review status is `accepted` are merged; pending and rejected edges are left out.
    An accepted edge is skipped, with a reason, when it names an unknown indicator, links an indicator
    to itself, has no weight, repeats an edge that is already there, or would close a cycle. Edges are
    tried from the highest weight down, so when two accepted edges cannot both stay the stronger wins.
    """
    H = G.copy()
    resolver = ids.Resolver(H)
    counts: Counter[str] = Counter()
    skipped: list[dict[str, Any]] = []
    accepted: list[tuple[float, Any, Any, Mapping[str, Any]]] = []
    for e in edges:
        status = review_status(e)
        counts[status] += 1
        if status != "accepted":
            continue
        a, b = resolver.get(str(e.get("source"))), resolver.get(str(e.get("target")))
        weight = e.get("weight")

        def skip(reason: str, e: Mapping[str, Any] = e) -> None:
            skipped.append({"source": e.get("source"), "target": e.get("target"), "reason": reason})

        if a is None or b is None:
            skip("unknown indicator")
        elif a == b:
            skip("links an indicator to itself")
        elif not isinstance(weight, int | float) or isinstance(weight, bool):
            skip("no weight")
        else:
            accepted.append((float(weight), a, b, e))
    accepted.sort(key=lambda t: (-t[0], schema.sort_key(t[1]), schema.sort_key(t[2])))
    merged = 0
    for weight, a, b, e in accepted:
        if H.has_edge(a, b):
            kind = H.edges[a, b].get("type")
            skipped.append({"source": a, "target": b, "reason": f"already linked ({kind} edge)"})
        elif nx.has_path(H, b, a):
            skipped.append({"source": a, "target": b, "reason": "would close a cycle"})
        else:
            prov = dict(e.get("provenance") or {})
            prov.setdefault("by", "model:unknown")
            prov.setdefault("at", None)
            H.add_edge(a, b, type="augmented", weight=weight, provenance=prov)
            merged += 1
    report = {
        "merged": merged,
        "accepted": counts["accepted"],
        "pending": counts["pending"],
        "rejected": counts["rejected"],
        "skipped": skipped,
    }
    return H, report


def keep_dates(G: nx.DiGraph, previous: nx.DiGraph) -> int:
    """Give each rule edge of `G` the date it has in `previous`, when `previous` has the same edge.

    A rebuild then changes only the lines of edges that are new. Returns how many dates were kept.
    """
    kept = 0
    for a, b, data in G.edges(data=True):
        old = previous.get_edge_data(a, b)
        prov = data.get("provenance") or {}
        if not old or old.get("type") != data.get("type") or data.get("type") not in RULE_WEIGHT:
            continue
        old_prov = old.get("provenance") or {}
        if old_prov.get("by") == prov.get("by") and old_prov.get("at"):
            data["provenance"] = {**prov, "at": old_prov["at"]}
            kept += 1
    return kept


def summary(G: nx.DiGraph) -> dict[str, Any]:
    """Node and edge counts of a graph, for the build message."""
    dag = nx.is_directed_acyclic_graph(G)
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "edges_by_type": dict(sorted(Counter(d.get("type") for *_, d in G.edges(data=True)).items())),
        "roots": sum(1 for n in G if G.in_degree(n) == 0),
        "longest_path_edges": nx.dag_longest_path_length(G, weight=None) if dag else None,
        "text_mode": G.graph.get("text_mode"),
    }


@dataclass
class BuildResult:
    """What `build` made: the labels graph, the full-text graph when there was text, and where they went."""

    graph: nx.DiGraph
    path: Path
    structure_path: Path
    full_graph: nx.DiGraph | None = None
    full_path: Path | None = None
    without_text: list[str] = field(default_factory=list)
    augmented: dict[str, Any] | None = None
    dates_kept: int = 0


def default_paths(subject: str = ids.DEFAULT_SUBJECT) -> dict[str, Path]:
    """The standard file locations of one subject, worked out from the data directory."""
    data = data_dir()
    build_dir = data.parent / "build"
    return {
        "structure": data / "curriculum" / f"{subject}.structure.json",
        "out": schema.default_graph_path(subject),
        "augmented": data / "graph" / f"{subject}.augmented.json",
        "full_text": build_dir / "curriculum" / f"{subject}.full.json",
        "full_out": build_dir / "graph" / f"{subject}.graph.full.json",
    }


def _inside(path: Path, folder: Path) -> bool:
    try:
        path.resolve().relative_to(folder.resolve())
    except ValueError:
        return False
    return True


def build(
    structure: str | os.PathLike[str] | None = None,
    out: str | os.PathLike[str] | None = None,
    full_text: str | os.PathLike[str] | None = None,
    full_out: str | os.PathLike[str] | None = None,
    augmented: str | os.PathLike[str] | None = None,
    at: str | None = None,
    subject: str = ids.DEFAULT_SUBJECT,
) -> BuildResult:
    """Build the graph of a subject from its structure file and write it.

    Steps: the rule edges from the structure records; the accepted augmented edges when the augmented
    file exists; the labels graph to `out`; and, when the full-text file exists, the same graph with
    the curriculum text to `full_out`. Paths left as None take the standard locations (see
    `default_paths`). The full-text graph is never written inside the data directory, because that
    directory is committed. With `at=None` an edge that was already in `out` keeps its date.
    """
    paths = default_paths(subject)
    structure_path = Path(structure) if structure else paths["structure"]
    out_path = Path(out) if out else paths["out"]
    full_text_path = Path(full_text) if full_text else paths["full_text"]
    full_out_path = Path(full_out) if full_out else paths["full_out"]
    augmented_path = Path(augmented) if augmented else paths["augmented"]
    if not structure_path.is_file():
        raise FileNotFoundError(f"no structure file at {structure_path}; make it with `kwan curriculum build`")

    G = build_rule_graph(read_records(structure_path), at=at, text_mode="labels", subject=subject)
    result = BuildResult(graph=G, path=out_path, structure_path=structure_path)
    if at is None and out_path.is_file():
        try:
            result.dates_kept = keep_dates(G, schema.load_graph(out_path, strict=False))
        except (OSError, ValueError, KeyError, TypeError):
            pass  # an unreadable old file gives no dates to keep
    if augmented_path.is_file():
        with open(augmented_path, encoding="utf-8") as f:
            proposed = json.load(f)
        edges = proposed.get("edges", []) if isinstance(proposed, Mapping) else proposed
        G, result.augmented = merge_augmented(G, edges)
        result.graph = G
    schema.save_graph(G, out_path)
    if full_text_path.is_file():
        if _inside(full_out_path, data_dir()):
            raise ValueError(f"{full_out_path} is inside the committed data directory; full text goes under build/")
        result.full_graph, result.without_text = merge_full_text(G, read_records(full_text_path))
        result.full_path = schema.save_graph(result.full_graph, full_out_path)
    return result


def main(argv: Sequence[str] | None = None) -> nx.DiGraph:
    """Build the graph from an indicators file and print the summary line: `INDICATORS_JSON GRAPH_JSON`."""
    import sys

    src, out = sys.argv[1:3] if argv is None else argv[:2]
    with open(src, encoding="utf-8") as f:
        G = build_graph(json.load(f))
    save(G, out)
    print(
        G.number_of_nodes(),
        "nodes",
        G.number_of_edges(),
        "edges",
        Counter(d["type"] for *_, d in G.edges(data=True)),
        "roots",
        sum(1 for n in G if G.in_degree(n) == 0),
        "longest path",
        nx.dag_longest_path_length(G),
    )
    return G


if __name__ == "__main__":
    main()
