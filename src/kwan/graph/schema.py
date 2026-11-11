"""Graph schema v2: load, save and convert the curriculum knowledge graph.

A v2 graph file looks like this:

    {"schema_version": 2, "subject": "math", "text_mode": "labels",
     "nodes": [{"id": "math:B4.1.1.1.2", "subject": "math", "code": "B4.1.1.1.2", "grade": 4, ...}],
     "edges": [{"source": "math:B3.1.2.1.1", "target": "math:B4.1.2.1.1", "type": "vertical", "weight": 1.0,
                "provenance": {"by": "rule:vertical", "at": "2026-11-10", "review": null}}]}

`text_mode` says what the node text is. In `"labels"` mode the file holds only codes, numbers, names
and the short labels written for Kwan; the curriculum sentences are left out. In `"full"` mode the
nodes also carry the curriculum text (`description`, `content_standard_text`, `exemplars`, `page`);
such files are build outputs and are never committed.

A loaded graph always has a `description` on every node: the curriculum description in full mode,
otherwise the short label. The loader also reads v1 files (networkx node-link data with bare codes
as node keys) and leaves their keys and attributes as they are.

This module needs only the standard library and networkx.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Hashable, Mapping
from pathlib import Path
from typing import Any

import networkx as nx

from kwan import ids
from kwan._paths import data_dir

SCHEMA_VERSION = 2
RULE_TYPES = ("sequence", "vertical", "bridge")
EDGE_TYPES = (*RULE_TYPES, "augmented")
TEXT_MODES = ("labels", "full")
# node attributes that hold curriculum text; they exist only in full-text graphs
TEXT_KEYS = ("description", "content_standard_text", "exemplars", "page")
NODE_KEY_ORDER = (
    "id",
    "subject",
    "code",
    "grade",
    "strand",
    "strand_name",
    "sub_strand",
    "sub_strand_name",
    "content_standard",
    "cs_key",
    "indicator_no",
    "short_label",
    "label_reviewed",
    "aliases",
    *TEXT_KEYS,
)
EDGE_KEY_ORDER = ("source", "target", "type", "weight", "provenance")
PROVENANCE_KEY_ORDER = ("by", "at", "review")
REVIEW_KEY_ORDER = ("status", "by", "at")

__all__ = [
    "EDGE_TYPES",
    "RULE_TYPES",
    "SCHEMA_VERSION",
    "TEXT_KEYS",
    "default_graph_path",
    "dump_graph",
    "dumps_graph",
    "graph_from_dict",
    "load_graph",
    "save_graph",
    "sort_key",
    "strip_text",
]

_NUMBER = re.compile(r"(\d+)")


def default_graph_path(subject: str = ids.DEFAULT_SUBJECT) -> Path:
    """Where the committed labels-mode graph of a subject lives: `data/graph/<subject>.graph.json`."""
    return data_dir() / "graph" / f"{subject}.graph.json"


def sort_key(node_id: Hashable) -> tuple[Any, ...]:
    """Natural order for ids: by subject, then by the numbers in the code (so B4.1.2 sorts before B4.1.10)."""
    text = str(node_id)
    subject, sep, code = text.partition(ids.SEP)
    if not sep:
        subject, code = "", text
    parts = tuple((0, int(p), "") if p.isdigit() else (1, 0, p) for p in _NUMBER.split(code) if p not in ("", "."))
    return (subject, parts, text)


def _ordered(d: Mapping[str, Any], first: tuple[str, ...]) -> dict[str, Any]:
    out = {k: d[k] for k in first if k in d}
    out.update({k: d[k] for k in sorted(d) if k not in out})
    return out


def _clean_provenance(prov: Any) -> Any:
    if not isinstance(prov, Mapping):
        return prov
    out = _ordered(prov, PROVENANCE_KEY_ORDER)
    out.setdefault("review", None)
    if isinstance(out["review"], Mapping):
        out["review"] = _ordered(out["review"], REVIEW_KEY_ORDER)
    return out


def _default_provenance(edge_type: Any) -> dict[str, Any]:
    by = f"rule:{edge_type}" if edge_type in RULE_TYPES else "unknown"
    return {"by": by, "at": None, "review": None}


# ---------------------------------------------------------------- reading


def _from_v1(d: Mapping[str, Any]) -> nx.DiGraph:
    edges_key = "edges" if "edges" in d else "links"
    G = nx.node_link_graph({**d, "directed": True, "multigraph": False}, edges=edges_key)
    G.graph.setdefault("subject", ids.DEFAULT_SUBJECT)
    G.graph["schema_version"] = 1
    full = False
    for n, data in G.nodes(data=True):
        label = data.get("short_label")
        if data.get("description"):
            full = full or data["description"] != label
        else:
            data["description"] = label or f"Skill {ids.code_of(str(n))}"
    G.graph["text_mode"] = d.get("text_mode") or ("full" if full else "labels")
    return G


def _from_v2(d: Mapping[str, Any], strict: bool) -> nx.DiGraph:
    subject = d.get("subject") or ids.DEFAULT_SUBJECT
    nodes = d.get("nodes") or []
    mode = d.get("text_mode")
    if mode is None:
        mode = "full" if any(r.get("description") not in (None, "", r.get("short_label")) for r in nodes) else "labels"
    if mode not in TEXT_MODES:
        raise ValueError(f"unknown text_mode {mode!r} (expected one of {', '.join(TEXT_MODES)})")
    G = nx.DiGraph(schema_version=SCHEMA_VERSION, subject=subject, text_mode=mode)
    for rec in nodes:
        if "id" not in rec:
            raise ValueError("graph node without an 'id'")
        key = rec["id"]
        if key in G:
            raise ValueError(f"graph node {key!r} is listed twice")
        data = {k: v for k, v in rec.items() if k != "id"}
        node_subject, code = ids.split(key, subject)
        data.setdefault("subject", node_subject)
        data.setdefault("code", code)
        data.setdefault("short_label", f"Skill {code}")
        data.setdefault("aliases", [code])
        if mode == "labels":
            for k in TEXT_KEYS:
                data.pop(k, None)
        if not data.get("description"):
            data["description"] = data["short_label"]
        G.add_node(key, **data)
    for e in d.get("edges") or d.get("links") or []:
        a, b = e.get("source"), e.get("target")
        missing = [x for x in (a, b) if x not in G]
        if missing:
            if strict:
                raise ValueError(f"edge {a!r} -> {b!r} points to unknown node {missing[0]!r}")
            for x in missing:
                G.add_node(x)
        data = {k: v for k, v in e.items() if k not in ("source", "target")}
        data["weight"] = float(data.get("weight", 1.0))
        if strict and data.get("type") not in EDGE_TYPES:
            raise ValueError(f"edge {a!r} -> {b!r} has unknown type {data.get('type')!r}")
        data["provenance"] = _clean_provenance(data.get("provenance") or _default_provenance(data.get("type")))
        G.add_edge(a, b, **data)
    return G


def graph_from_dict(d: Mapping[str, Any], strict: bool = True) -> nx.DiGraph:
    """Build the graph from parsed JSON: a v2 graph, a v1 node-link file, or a bundle that holds a graph.

    With `strict=True` (the default) an edge that points to a node that is not listed, or that has an
    unknown type, raises `ValueError`. With `strict=False` such edges are kept, so that
    `kwan.graph.validate` can report them.
    """
    if not isinstance(d, Mapping):
        raise ValueError("a graph file must hold a JSON object")
    version = d.get("schema_version")
    inner = d.get("graph")
    if isinstance(inner, Mapping) and "nodes" in inner and "nodes" not in d:  # a bundle: the graph is one section
        merged = {"schema_version": SCHEMA_VERSION, "subject": d.get("subject"), "text_mode": d.get("text_mode")}
        return graph_from_dict({**merged, **inner}, strict=strict)
    if version is None and ("directed" in d or "multigraph" in d or "links" in d):
        return _from_v1(d)
    if version == 1:
        return _from_v1(d)
    if version not in (None, SCHEMA_VERSION):
        raise ValueError(f"graph schema_version {version!r} is not supported (this Kwan reads versions 1 and 2)")
    if "nodes" not in d:
        raise ValueError("not a Kwan graph: no 'nodes' list")
    return _from_v2(d, strict)


def _packaged_bundle() -> dict[str, Any] | None:
    try:
        from importlib import resources

        ref = resources.files("kwan").joinpath("bundles", f"kwan-bundle.{ids.DEFAULT_SUBJECT}.json")
        return json.loads(ref.read_text(encoding="utf-8")) if ref.is_file() else None
    except (OSError, ValueError, ModuleNotFoundError):
        return None


def load_graph(path: str | os.PathLike[str] | None = None, strict: bool = True) -> nx.DiGraph:
    """Load a graph file (schema v2 or v1). With no path, load the default committed maths graph.

    The default is `data/graph/math.graph.json`; in an install without the repository data it is the
    graph inside the bundle shipped with the package.
    """
    if path is None:
        default = default_graph_path()
        if not default.is_file():
            bundle = _packaged_bundle()
            if bundle is not None:
                return graph_from_dict(bundle, strict=strict)
            raise FileNotFoundError(f"no graph at {default}; build it with `kwan graph build`")
        path = default
    with open(path, encoding="utf-8") as f:
        return graph_from_dict(json.load(f), strict=strict)


# ---------------------------------------------------------------- writing


def _canonical_ids(G: nx.DiGraph, subject: str) -> dict[Hashable, str]:
    out: dict[Hashable, str] = {}
    for n, data in G.nodes(data=True):
        text = str(n)
        out[n] = text if ids.is_namespaced(text) else ids.make(data.get("subject") or subject, text)
    if len(set(out.values())) != len(out):
        raise ValueError("two nodes of the graph share one canonical id")
    return out


def dump_graph(G: nx.DiGraph) -> dict[str, Any]:
    """The schema v2 dictionary of a graph, with nodes and edges in a stable, sorted order.

    Node keys are written as canonical ids, so dumping a v1 graph (bare codes) upgrades it to v2. In
    labels mode the curriculum text attributes are left out. `graph_from_dict(dump_graph(G))` gives
    back an equal graph for any v2 graph.
    """
    subject = G.graph.get("subject") or ids.DEFAULT_SUBJECT
    mode = G.graph.get("text_mode") or "labels"
    if mode not in TEXT_MODES:
        raise ValueError(f"unknown text_mode {mode!r}")
    canon = _canonical_ids(G, subject)
    nodes = []
    for n, data in G.nodes(data=True):
        node_subject, code = ids.split(canon[n])
        rec: dict[str, Any] = {"id": canon[n], "subject": node_subject, "code": code}
        rec.update({k: v for k, v in data.items() if k != "id"})
        rec.setdefault("short_label", f"Skill {code}")
        rec.setdefault("aliases", [code])
        if mode == "labels":
            for k in TEXT_KEYS:
                rec.pop(k, None)
        nodes.append(_ordered(rec, NODE_KEY_ORDER))
    nodes.sort(key=lambda r: sort_key(r["id"]))
    edges = []
    for a, b, data in G.edges(data=True):
        rec = {"source": canon[a], "target": canon[b], **data}
        rec["weight"] = float(rec.get("weight", 1.0))
        rec["provenance"] = _clean_provenance(rec.get("provenance") or _default_provenance(rec.get("type")))
        edges.append(_ordered(rec, EDGE_KEY_ORDER))
    edges.sort(key=lambda e: (sort_key(e["source"]), sort_key(e["target"]), str(e.get("type"))))
    return {"schema_version": SCHEMA_VERSION, "subject": subject, "text_mode": mode, "nodes": nodes, "edges": edges}


def dumps_graph(G: nx.DiGraph) -> str:
    """The file text of a graph: one node or edge per line, so that a change shows as a one-line diff."""
    d = dump_graph(G)

    def line(obj: Any) -> str:
        return json.dumps(obj, ensure_ascii=False)

    out = ["{"]
    out += [f' "{k}": {line(d[k])},' for k in ("schema_version", "subject", "text_mode")]
    for name, last in (("nodes", False), ("edges", True)):
        rows = [f"  {line(item)}" for item in d[name]]
        body = ",\n".join(rows)
        out.append(f' "{name}": [\n{body}\n ]' if rows else f' "{name}": []')
        if not last:
            out[-1] += ","
    out.append("}")
    return "\n".join(out) + "\n"


def save_graph(G: nx.DiGraph, path: str | os.PathLike[str]) -> Path:
    """Write the graph as schema v2 JSON (see `dumps_graph`). The file is replaced in one step."""
    path = Path(path)
    text = dumps_graph(G)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)
    return path


def strip_text(G: nx.DiGraph) -> nx.DiGraph:
    """A labels-mode copy of a graph: the curriculum text is removed and `description` is the short label."""
    H = G.copy()
    H.graph["text_mode"] = "labels"
    for n, data in H.nodes(data=True):
        for k in TEXT_KEYS:
            data.pop(k, None)
        data["description"] = data.get("short_label") or f"Skill {ids.code_of(str(n))}"
    return H
