"""The committed labels-mode graph `data/graph/math.graph.json`: licensing, shape, and freshness.

The file holds codes, numbers, names and short labels only. The curriculum sentences (descriptions,
content-standard text, exemplars) must never be in it. These tests are skipped until the file exists.
"""

import json
from pathlib import Path

import networkx as nx
import pytest

from kwan import ids
from kwan.graph import build, schema, validate

ROOT = Path(__file__).resolve().parent.parent
GRAPH = ROOT / "data" / "graph" / "math.graph.json"
STRUCTURE = ROOT / "data" / "curriculum" / "math.structure.json"
AUGMENTED = ROOT / "data" / "graph" / "math.augmented.json"
FULL_TEXT = ROOT / "build" / "curriculum" / "math.full.json"
REPORT = ROOT / "docs" / "qa" / "graph-math.md"
# every key a committed node may have: structure fields and the short label written for Kwan
ALLOWED_NODE_KEYS = {
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
}
FORBIDDEN_KEYS = {"description", "content_standard_text", "exemplars", "exemplar", "page", "text"}

pytestmark = pytest.mark.skipif(not GRAPH.exists(), reason="data/graph/math.graph.json is not built yet")


@pytest.fixture(scope="module")
def data():
    return json.loads(GRAPH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def G():
    return schema.load_graph(GRAPH)


def strings(value):
    """Every string inside a JSON value."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from strings(v)


def squash(text):
    return " ".join(text.lower().split())


# ---------------------------------------------------------------- licensing


def test_committed_graph_is_in_labels_mode(data):
    assert data["schema_version"] == 2 and data["subject"] == "math" and data["text_mode"] == "labels"
    assert set(data) == {"schema_version", "subject", "text_mode", "nodes", "edges"}


def test_committed_nodes_have_no_curriculum_text_fields(data):
    for node in data["nodes"]:
        assert not set(node) & FORBIDDEN_KEYS, node["id"]
        assert set(node) <= ALLOWED_NODE_KEYS, (node["id"], sorted(set(node) - ALLOWED_NODE_KEYS))


def test_committed_edges_have_only_the_schema_fields(data):
    for e in data["edges"]:
        assert set(e) == {"source", "target", "type", "weight", "provenance"}, e
        assert set(e["provenance"]) <= {"by", "at", "review"} | {"model", "model_version"}, e
        assert not any(len(s) > 80 for s in strings(e["provenance"])), e  # no free text rides along on an edge


def test_committed_strings_are_short_names_and_labels_not_sentences(data):
    for node in data["nodes"]:
        assert isinstance(node["short_label"], str) and 0 < len(node["short_label"]) <= 80, node["id"]
        for key in ("strand_name", "sub_strand_name"):
            assert len(node.get(key, "")) <= 80, (node["id"], key)
        assert all(isinstance(a, str) and len(a) <= 40 for a in node["aliases"]), node["id"]


@pytest.mark.skipif(not FULL_TEXT.exists(), reason="no local full-text build to compare with")
def test_no_curriculum_sentence_of_the_full_text_build_is_in_the_committed_graph(data):
    committed = {squash(s) for s in strings(data)}
    blob = squash(GRAPH.read_text(encoding="utf-8"))
    for rec in build.read_records(FULL_TEXT):
        for key in ("description", "content_standard_text", "exemplars"):
            for sentence in strings(rec.get(key)):
                s = squash(sentence)
                if len(s) < 40:
                    continue  # too short to tell apart from a label
                assert s not in committed, (rec.get("id"), key)
                assert s not in blob, (rec.get("id"), key)


def test_loaded_graph_describes_each_node_with_its_short_label(G):
    assert G.graph["text_mode"] == "labels"
    for n, d in G.nodes(data=True):
        assert d["description"] == d["short_label"], n
        assert "exemplars" not in d and "content_standard_text" not in d, n


@pytest.mark.skipif(not REPORT.exists(), reason="docs/qa/graph-math.md is not written yet")
def test_validation_report_quotes_no_label_and_no_long_sentence(data):
    report = REPORT.read_text(encoding="utf-8")
    assert "—" not in report
    for node in data["nodes"]:
        label = node["short_label"]
        if len(label) >= 12 and not label.startswith("Skill "):
            assert label not in report, node["id"]


# ---------------------------------------------------------------- shape


def test_committed_ids_are_canonical_and_match_the_fields(data):
    seen = set()
    for node in data["nodes"]:
        assert node["id"] == ids.make("math", node["code"]) and node["subject"] == "math"
        assert node["id"] not in seen
        seen.add(node["id"])
        assert node["code"] in node["aliases"]
        g, s, ss, cs, i = (int(x) for x in node["code"][1:].split("."))
        assert (node["grade"], node["strand"], node["sub_strand"], node["indicator_no"]) == (g, s, ss, i), node["id"]
        assert node["content_standard"] == f"B{g}.{s}.{ss}.{cs}" and node["cs_key"] == f"{s}.{ss}.{cs}", node["id"]
        assert 1 <= node["grade"] <= 6


def test_every_committed_edge_has_provenance_and_a_known_type(data):
    nodes = {n["id"] for n in data["nodes"]}
    for e in data["edges"]:
        assert e["source"] in nodes and e["target"] in nodes, e
        assert e["type"] in schema.EDGE_TYPES, e
        prov = e["provenance"]
        assert prov["by"] and prov["at"], e
        if e["type"] == "augmented":
            assert not prov["by"].startswith("rule:") and prov["review"]["status"] == "accepted", e
            assert 0.0 < e["weight"] <= 1.0, e
        else:
            assert prov == {"by": f"rule:{e['type']}", "at": prov["at"], "review": None}, e
            assert e["weight"] == build.RULE_WEIGHT[e["type"]], e


def test_committed_graph_passes_the_validation_checks(G):
    stats = validate.analyse(G)
    assert stats["ok"], stats["checks"]
    assert nx.is_directed_acyclic_graph(G) and stats["orphan_edges"] == [] and stats["unreachable_from_a_root"] == []
    assert stats["edges_without_provenance"] == 0


def test_rule_edges_never_point_to_an_earlier_grade(G):
    for a, b, d in G.edges(data=True):
        if d["type"] != "augmented":
            assert G.nodes[a]["grade"] <= G.nodes[b]["grade"], (a, b)


def test_committed_file_is_in_its_own_stable_layout(G):
    assert schema.dumps_graph(G) == GRAPH.read_text(encoding="utf-8")


def test_resolver_accepts_every_bare_code_and_alias(G, data):
    resolver = ids.Resolver(G)
    for node in data["nodes"]:
        assert resolver.resolve(node["code"]) == node["id"] == resolver.resolve(node["id"])
        for alias in node["aliases"]:
            assert resolver.resolve(alias) in G


# ---------------------------------------------------------------- freshness


@pytest.mark.skipif(not STRUCTURE.exists(), reason="no structure file to rebuild from")
def test_committed_graph_is_what_the_builder_makes_from_the_structure_file(G, data):
    rebuilt = build.build_rule_graph(build.read_records(STRUCTURE), at="2000-01-01", text_mode="labels")
    if AUGMENTED.exists():
        proposed = json.loads(AUGMENTED.read_text(encoding="utf-8"))
        rebuilt, _ = build.merge_augmented(rebuilt, proposed.get("edges", []))
    fresh = schema.dump_graph(rebuilt)
    assert fresh["nodes"] == data["nodes"], "run `kwan graph build`: the structure file changed"
    as_set = lambda edges: {(e["source"], e["target"], e["type"], e["weight"], e["provenance"]["by"]) for e in edges}
    assert as_set(fresh["edges"]) == as_set(data["edges"]), "run `kwan graph build`: the edges changed"
