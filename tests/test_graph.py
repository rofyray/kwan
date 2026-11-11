"""Knowledge graph: edge rules on tiny hand-built curricula, persistence, then invariants on the committed graph."""

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import networkx as nx
import pytest

from kwan.graph import build as graph
from kwan.graph import schema

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
GRAPH_V1 = FIXTURES / "graph_v1.json"
INDICATORS_V1 = FIXTURES / "indicators_v1.json"
NODE_ATTRS = {
    "id",
    "grade",
    "strand",
    "strand_name",
    "sub_strand",
    "content_standard",
    "cs_key",
    "indicator_no",
    "description",
}  # PRD 5.1
STRANDS = {1: "Number", 2: "Algebra", 3: "Geometry and Measurement", 4: "Data"}
WEIGHT = {"sequence": 1.0, "vertical": 1.0, "bridge": 0.5}


def ind(code, description=None):
    """An indicator record, as parse_curriculum writes it, from a code like "B4.1.1.1.2"."""
    g, s, ss, cs, i = (int(x) for x in code[1:].split("."))
    return {
        "id": code,
        "grade": g,
        "strand": s,
        "strand_name": STRANDS[s],
        "sub_strand": ss,
        "content_standard": f"B{g}.{s}.{ss}.{cs}",
        "cs_key": f"{s}.{ss}.{cs}",
        "indicator_no": i,
        "description": description or f"Indicator {code}",
    }


def build(*codes):
    return graph.build_graph([ind(c) for c in codes])


def edges(G):
    return {(a, b): (d["type"], d["weight"]) for a, b, d in G.edges(data=True)}


# ---------------------------------------------------------------- edge rules


def test_nodes_carry_the_indicator_record():
    rec = ind("B4.1.1.1.2", "Made-up label for one indicator")
    G = graph.build_graph([rec])
    assert list(G.nodes) == ["B4.1.1.1.2"] and G.nodes["B4.1.1.1.2"] == rec
    assert G.number_of_edges() == 0


def test_empty_curriculum_gives_an_empty_graph():
    G = graph.build_graph([])
    assert G.number_of_nodes() == 0 and G.number_of_edges() == 0


def test_sequence_edges_chain_the_indicators_of_a_content_standard():
    G = build("B1.1.1.1.1", "B1.1.1.1.2", "B1.1.1.1.3")
    assert edges(G) == {
        ("B1.1.1.1.1", "B1.1.1.1.2"): ("sequence", 1.0),
        ("B1.1.1.1.2", "B1.1.1.1.3"): ("sequence", 1.0),
    }


def test_sequence_follows_the_given_order_even_with_a_missing_number():
    G = build("B1.1.1.1.1", "B1.1.1.1.3", "B1.1.1.1.4")  # indicator 2 was not extracted
    assert edges(G) == {
        ("B1.1.1.1.1", "B1.1.1.1.3"): ("sequence", 1.0),
        ("B1.1.1.1.3", "B1.1.1.1.4"): ("sequence", 1.0),
    }


def test_no_sequence_edge_between_different_strands_sub_strands_or_grades():
    G = build("B1.1.1.1.1", "B1.2.1.1.1", "B1.1.2.1.1", "B1.3.3.2.1")
    assert G.number_of_edges() == 0


def test_vertical_edge_links_last_indicator_to_first_of_the_next_grade():
    G = build("B1.1.1.1.1", "B1.1.1.1.2", "B2.1.1.1.1", "B2.1.1.1.2")
    assert edges(G) == {
        ("B1.1.1.1.1", "B1.1.1.1.2"): ("sequence", 1.0),
        ("B2.1.1.1.1", "B2.1.1.1.2"): ("sequence", 1.0),
        ("B1.1.1.1.2", "B2.1.1.1.1"): ("vertical", 1.0),
    }


def test_vertical_edge_skips_a_grade_that_lacks_the_content_standard():
    G = build("B1.2.1.1.1", "B2.3.1.1.1", "B3.2.1.1.1")  # B2 has no 2.1.1
    assert edges(G) == {("B1.2.1.1.1", "B3.2.1.1.1"): ("vertical", 1.0)}


def test_vertical_edge_goes_only_to_the_nearest_later_grade():
    G = build("B1.1.1.1.1", "B2.1.1.1.1", "B4.1.1.1.1", "B6.1.1.1.1")
    assert edges(G) == {
        ("B1.1.1.1.1", "B2.1.1.1.1"): ("vertical", 1.0),
        ("B2.1.1.1.1", "B4.1.1.1.1"): ("vertical", 1.0),
        ("B4.1.1.1.1", "B6.1.1.1.1"): ("vertical", 1.0),
    }


def test_vertical_edge_needs_the_same_cs_key():
    G = build("B1.1.1.1.1", "B2.1.1.2.1", "B3.1.2.1.1", "B4.2.1.1.1")
    assert G.number_of_edges() == 0


def test_bridge_links_consecutive_content_standards_with_half_weight():
    G = build("B1.1.2.1.1", "B1.1.2.1.2", "B1.1.2.2.1", "B1.1.2.2.2", "B1.1.2.3.1")
    assert edges(G) == {
        ("B1.1.2.1.1", "B1.1.2.1.2"): ("sequence", 1.0),
        ("B1.1.2.2.1", "B1.1.2.2.2"): ("sequence", 1.0),
        ("B1.1.2.1.2", "B1.1.2.2.1"): ("bridge", 0.5),  # last of c -> first of c+1
        ("B1.1.2.2.2", "B1.1.2.3.1"): ("bridge", 0.5),
    }


def test_bridge_order_is_numeric_and_independent_of_input_order():
    G = build("B1.1.1.3.1", "B1.1.1.1.1", "B1.1.1.2.1")
    assert edges(G) == {("B1.1.1.1.1", "B1.1.1.2.1"): ("bridge", 0.5), ("B1.1.1.2.1", "B1.1.1.3.1"): ("bridge", 0.5)}


def test_bridge_stays_inside_one_grade_and_sub_strand():
    G = build(
        "B1.1.1.1.1",
        "B1.1.2.2.1",  # other sub-strand
        "B1.2.1.2.1",  # other strand
        "B2.1.1.2.1",
    )  # other grade
    assert G.number_of_edges() == 0


def test_all_three_edge_types_together():
    G = build("B1.1.1.1.1", "B1.1.1.1.2", "B1.1.1.2.1", "B2.1.1.1.1", "B2.1.1.2.1")
    assert edges(G) == {
        ("B1.1.1.1.1", "B1.1.1.1.2"): ("sequence", 1.0),
        ("B1.1.1.1.2", "B2.1.1.1.1"): ("vertical", 1.0),
        ("B1.1.1.2.1", "B2.1.1.2.1"): ("vertical", 1.0),
        ("B1.1.1.1.2", "B1.1.1.2.1"): ("bridge", 0.5),
        ("B2.1.1.1.1", "B2.1.1.2.1"): ("bridge", 0.5),
    }
    assert nx.is_directed_acyclic_graph(G)
    assert [n for n in G if G.in_degree(n) == 0] == ["B1.1.1.1.1"]


def test_no_duplicate_bridge_when_an_edge_already_exists():
    # Cannot happen with well-formed ids: node "y" is listed under two content standards, so the
    # vertical edge x -> y is already there when the bridge x -> y is considered. The guard keeps it.
    x = ind("B1.1.1.1.1")
    y_next_grade = {**ind("B2.1.1.1.1"), "id": "y"}
    y_same_grade = {**ind("B1.1.1.2.1"), "id": "y"}
    G = graph.build_graph([x, y_next_grade, y_same_grade])
    assert edges(G) == {("B1.1.1.1.1", "y"): ("vertical", 1.0)}


def test_build_graph_rejects_a_cycle():
    a_first = {**ind("B1.1.1.1.1"), "id": "a"}
    b = {**ind("B1.1.1.1.2"), "id": "b"}
    a_again = {**ind("B1.1.1.2.1"), "id": "a"}  # sequence a -> b, then bridge b -> a
    with pytest.raises(AssertionError, match="DAG"):
        graph.build_graph([a_first, b, a_again])


# ---------------------------------------------------------------- save / load


def test_save_load_round_trip(tmp_path):
    G = graph.build_graph(
        [ind("B1.1.1.1.1", "Count the beads: “how many?”"), ind("B1.1.1.1.2"), ind("B1.1.1.2.1"), ind("B2.1.1.1.1")]
    )
    path = tmp_path / "graph.json"
    graph.save(G, path)
    H = graph.load(path)
    assert isinstance(H, nx.DiGraph)
    assert list(H.nodes) == list(G.nodes)
    # node-link loading turns "id" into the node key, so a loaded node has every attribute but "id"
    assert all({**H.nodes[n], "id": n} == G.nodes[n] for n in G)
    assert all("id" not in H.nodes[n] for n in H)
    assert edges(H) == edges(G) and len(edges(G)) == 3
    assert nx.node_link_data(H, edges="edges") == nx.node_link_data(G, edges="edges")


def test_saved_file_is_node_link_json_with_an_edges_key(tmp_path):
    G = graph.build_graph([ind("B1.1.1.1.1", "Count the beads: “how many?”"), ind("B1.1.1.1.2")])
    path = tmp_path / "graph.json"
    graph.save(G, path)
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    assert data["directed"] is True and data["multigraph"] is False
    assert [n["id"] for n in data["nodes"]] == ["B1.1.1.1.1", "B1.1.1.1.2"]
    assert data["edges"] == [{"type": "sequence", "weight": 1.0, "source": "B1.1.1.1.1", "target": "B1.1.1.1.2"}]
    assert "links" not in data
    assert "“how many?”" in text  # UTF-8, not \u escapes


def test_load_accepts_a_string_path(tmp_path):
    path = tmp_path / "g.json"
    graph.save(build("B1.1.1.1.1", "B1.1.1.1.2"), str(path))
    assert graph.load(str(path)).number_of_edges() == 1


def test_default_path_is_the_committed_v2_graph():
    assert Path(graph.DEFAULT_PATH) == DATA / "graph" / "math.graph.json" == schema.default_graph_path()
    assert Path(graph.DEFAULT_PATH).is_absolute() and Path(graph.DEFAULT_PATH).is_file()
    assert not (DATA / "graph.json").exists() and not (DATA / "indicators.json").exists()  # the v1 files are gone


def test_load_without_argument_works_from_another_directory_and_reads_the_v2_graph(tmp_path, monkeypatch, G):
    monkeypatch.chdir(tmp_path)
    assert not Path("data").exists()
    H = graph.load()
    assert H.graph["schema_version"] == 2 and all(str(n).startswith("math:") for n in H)
    assert set(H.nodes) == set(G.nodes) and set(H.edges) == set(G.edges)
    assert all(H.nodes[n] == G.nodes[n] for n in H)


def test_load_reads_the_frozen_v1_file_as_it_is(G_v1):
    H = graph.load(GRAPH_V1)
    assert list(H.nodes) == list(G_v1.nodes) and "schema_version" not in H.graph
    assert nx.node_link_data(H, edges="edges") == nx.node_link_data(G_v1, edges="edges")


def test_save_takes_no_default_path(tmp_path):
    with pytest.raises(TypeError):
        graph.save(build("B1.1.1.1.1"))


def test_cli_builds_the_frozen_v1_graph_and_prints_the_summary(tmp_path):
    out = tmp_path / "graph.json"
    done = subprocess.run(
        [sys.executable, "-m", "kwan.graph.build", str(INDICATORS_V1), str(out)],
        capture_output=True,
        text=True,
        check=True,
        cwd=tmp_path,
    )
    expected = "266 nodes 287 edges Counter({'sequence': 150, 'vertical': 82, 'bridge': 55}) roots 16 longest path 36"
    assert done.stdout.strip() in (expected, expected + ".0")  # PRD prints 36.0 (float edge weights)
    assert json.loads(out.read_text(encoding="utf-8")) == json.loads(GRAPH_V1.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- the frozen v1 graph (tests/fixtures/graph_v1.json)
# The v1 file with every NaCCA description replaced by the Kwan short label: 266 nodes, 287 edges, bare ids.


def test_frozen_v1_graph_is_a_dag(G_v1):
    assert isinstance(G_v1, nx.DiGraph) and nx.is_directed_acyclic_graph(G_v1)


def test_frozen_v1_graph_size(G_v1):
    assert G_v1.number_of_nodes() == 266
    assert G_v1.number_of_edges() == 287


def test_frozen_v1_edge_types_and_weights(G_v1):
    assert Counter(d["type"] for *_, d in G_v1.edges(data=True)) == {"sequence": 150, "vertical": 82, "bridge": 55}
    for a, b, d in G_v1.edges(data=True):
        assert set(d) == {"type", "weight"}, (a, b)
        assert d["weight"] == WEIGHT[d["type"]], (a, b)


def test_frozen_v1_roots_and_longest_path(G_v1):
    assert sum(1 for n in G_v1 if G_v1.in_degree(n) == 0) == 16
    assert nx.dag_longest_path_length(G_v1) == 36  # weighted: a bridge counts 0.5
    assert nx.dag_longest_path_length(G_v1, weight=None) == 38  # the longest chain counted in edges


def test_frozen_v1_nodes_have_the_full_attribute_set(G_v1):
    # in a loaded graph the PRD "id" field is the node key; the other eight fields are attributes
    for n, d in G_v1.nodes(data=True):
        assert set(d) | {"id"} == NODE_ATTRS and "id" not in d, n
        g, s, ss, cs, i = (int(x) for x in n[1:].split("."))
        assert (d["grade"], d["strand"], d["sub_strand"], d["indicator_no"]) == (g, s, ss, i), n
        assert d["content_standard"] == f"B{g}.{s}.{ss}.{cs}" and d["cs_key"] == f"{s}.{ss}.{cs}", n
        assert d["strand_name"] == STRANDS[d["strand"]]
        assert isinstance(d["description"], str) and d["description"]
        assert all(isinstance(d[k], int) for k in ("grade", "strand", "sub_strand", "indicator_no")), n


def test_frozen_v1_file_stores_every_node_with_all_nine_fields():
    data = json.loads(GRAPH_V1.read_text(encoding="utf-8"))
    assert set(data) == {"directed", "multigraph", "graph", "nodes", "edges"}
    assert len(data["nodes"]) == 266 and len(data["edges"]) == 287
    assert all(set(node) == NODE_ATTRS for node in data["nodes"])
    assert all(set(edge) == {"type", "weight", "source", "target"} for edge in data["edges"])


def test_frozen_v1_graph_nodes_are_the_committed_indicators(G_v1, indicators_v1):
    assert list(G_v1.nodes) == [r["id"] for r in indicators_v1]
    assert [{"id": r["id"], **G_v1.nodes[r["id"]]} for r in indicators_v1] == indicators_v1


def test_frozen_v1_edges_obey_the_edge_rules(G_v1):
    for a, b, d in G_v1.edges(data=True):
        x, y = G_v1.nodes[a], G_v1.nodes[b]
        if d["type"] == "sequence":
            assert x["content_standard"] == y["content_standard"] and x["indicator_no"] < y["indicator_no"], (a, b)
        elif d["type"] == "vertical":
            assert x["cs_key"] == y["cs_key"] and x["grade"] < y["grade"], (a, b)
        else:
            assert d["type"] == "bridge"
            assert (x["grade"], x["strand"], x["sub_strand"]) == (y["grade"], y["strand"], y["sub_strand"]), (a, b)
            assert int(x["cs_key"].split(".")[-1]) < int(y["cs_key"].split(".")[-1]), (a, b)


def test_frozen_v1_edges_never_point_to_an_earlier_grade(G_v1):
    assert all(G_v1.nodes[a]["grade"] <= G_v1.nodes[b]["grade"] for a, b in G_v1.edges)


def test_graph_rebuilt_from_the_v1_indicators_equals_the_frozen_file(indicators_v1):
    rebuilt = nx.node_link_data(graph.build_graph(indicators_v1), edges="edges")
    committed = json.loads(GRAPH_V1.read_text(encoding="utf-8"))
    assert rebuilt == committed


def test_rebuilt_v1_graph_saves_byte_for_byte(tmp_path, indicators_v1):
    path = tmp_path / "graph.json"
    graph.save(graph.build_graph(indicators_v1), path)
    assert path.read_bytes() == GRAPH_V1.read_bytes()
