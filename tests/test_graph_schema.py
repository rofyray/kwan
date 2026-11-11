"""Graph schema v2: reading and writing, the labels and full text modes, and reading v1 files."""

import json

import networkx as nx
import pytest

from kwan.graph import schema

STRANDS = {1: "Number", 2: "Algebra", 3: "Geometry and Measurement", 4: "Data"}


def rec(code, **extra):
    """A structure record (no curriculum text) from a code like "B4.1.1.1.2"."""
    g, s, ss, cs, i = (int(x) for x in code[1:].split("."))
    return {
        "id": f"math:{code}",
        "subject": "math",
        "code": code,
        "grade": g,
        "strand": s,
        "strand_name": STRANDS[s],
        "sub_strand": ss,
        "sub_strand_name": f"Sub-strand {s}.{ss}",
        "content_standard": f"B{g}.{s}.{ss}.{cs}",
        "cs_key": f"{s}.{ss}.{cs}",
        "indicator_no": i,
        "short_label": f"Test skill {code}",
        "label_reviewed": False,
        "aliases": [code],
        **extra,
    }


def edge(a, b, kind="sequence", weight=1.0, at="2026-11-10", review=None, by=None):
    return {
        "source": f"math:{a}",
        "target": f"math:{b}",
        "type": kind,
        "weight": weight,
        "provenance": {"by": by or f"rule:{kind}", "at": at, "review": review},
    }


def v2_dict(text_mode="labels"):
    extra = {}
    if text_mode == "full":
        extra = {
            "description": "A made-up full description for the test",
            "content_standard_text": "A made-up content standard",
            "exemplars": [{"text": "A made-up exemplar", "ghana_context": False}],
            "page": 12,
        }
    return {
        "schema_version": 2,
        "subject": "math",
        "text_mode": text_mode,
        "nodes": [rec("B1.1.1.1.1", **extra), rec("B1.1.1.1.2", **extra), rec("B2.1.1.1.1", **extra)],
        "edges": [
            edge("B1.1.1.1.1", "B1.1.1.1.2"),
            edge(
                "B1.1.1.1.1",
                "B2.1.1.1.1",
                "augmented",
                0.83,
                by="model:test",
                review={"status": "accepted", "by": "teacher-1", "at": "2026-11-11"},
            ),
            edge("B1.1.1.1.2", "B2.1.1.1.1", "vertical"),
        ],  # in file order: by source, then target
    }


# ---------------------------------------------------------------- reading v2


def test_graph_from_dict_reads_a_v2_graph():
    G = schema.graph_from_dict(v2_dict())
    assert isinstance(G, nx.DiGraph)
    assert list(G.nodes) == ["math:B1.1.1.1.1", "math:B1.1.1.1.2", "math:B2.1.1.1.1"]
    assert G.graph == {"schema_version": 2, "subject": "math", "text_mode": "labels"}
    node = G.nodes["math:B1.1.1.1.2"]
    assert (
        "id" not in node and node["code"] == "B1.1.1.1.2" and node["grade"] == 1 and node["aliases"] == ["B1.1.1.1.2"]
    )
    assert G.edges["math:B1.1.1.1.2", "math:B2.1.1.1.1"] == {
        "type": "vertical",
        "weight": 1.0,
        "provenance": {"by": "rule:vertical", "at": "2026-11-10", "review": None},
    }
    assert G.edges["math:B1.1.1.1.1", "math:B2.1.1.1.1"]["provenance"]["review"]["status"] == "accepted"


def test_every_node_has_a_description_the_short_label_in_labels_mode():
    G = schema.graph_from_dict(v2_dict())
    assert all(d["description"] == d["short_label"] for _, d in G.nodes(data=True))
    assert not any(k in d for _, d in G.nodes(data=True) for k in ("exemplars", "content_standard_text", "page"))


def test_full_mode_keeps_the_curriculum_text():
    G = schema.graph_from_dict(v2_dict("full"))
    assert G.graph["text_mode"] == "full"
    node = G.nodes["math:B1.1.1.1.1"]
    assert node["description"] == "A made-up full description for the test"
    assert node["exemplars"][0]["text"] == "A made-up exemplar" and node["page"] == 12


def test_a_file_marked_labels_never_yields_curriculum_text():
    d = v2_dict("full")
    d["text_mode"] = "labels"
    G = schema.graph_from_dict(d)
    assert all(n["description"] == n["short_label"] and "exemplars" not in n for _, n in G.nodes(data=True))


def test_text_mode_is_inferred_when_the_file_does_not_say():
    labels, full = v2_dict(), v2_dict("full")
    del labels["text_mode"], full["text_mode"]
    assert schema.graph_from_dict(labels).graph["text_mode"] == "labels"
    assert schema.graph_from_dict(full).graph["text_mode"] == "full"


def test_missing_node_fields_get_safe_defaults():
    G = schema.graph_from_dict({"schema_version": 2, "nodes": [{"id": "math:B1.1.1.1.1", "grade": 1}], "edges": []})
    assert G.nodes["math:B1.1.1.1.1"] == {
        "grade": 1,
        "subject": "math",
        "code": "B1.1.1.1.1",
        "short_label": "Skill B1.1.1.1.1",
        "aliases": ["B1.1.1.1.1"],
        "description": "Skill B1.1.1.1.1",
    }


def test_strict_loading_rejects_orphan_edges_duplicates_and_unknown_types():
    orphan = v2_dict()
    orphan["edges"].append(edge("B1.1.1.1.1", "B9.9.9.9.9"))
    with pytest.raises(ValueError, match="unknown node"):
        schema.graph_from_dict(orphan)
    assert schema.graph_from_dict(orphan, strict=False).has_edge("math:B1.1.1.1.1", "math:B9.9.9.9.9")
    twice = v2_dict()
    twice["nodes"].append(rec("B1.1.1.1.1"))
    with pytest.raises(ValueError, match="listed twice"):
        schema.graph_from_dict(twice)
    odd = v2_dict()
    odd["edges"][0]["type"] = "guess"
    with pytest.raises(ValueError, match="unknown type"):
        schema.graph_from_dict(odd)


@pytest.mark.parametrize(
    "bad", [{"schema_version": 3, "nodes": []}, {"hello": 1}, [], {"schema_version": 2, "text_mode": "x", "nodes": []}]
)
def test_unreadable_input_raises_valueerror(bad):
    with pytest.raises(ValueError):
        schema.graph_from_dict(bad)


def test_a_bundle_with_a_graph_section_loads():
    d = v2_dict()
    bundle = {"schema_version": 2, "bundle_version": "math-test", "subject": "math", "text_mode": "labels",
              "graph": {"nodes": d["nodes"], "edges": d["edges"]}, "params": {}}  # fmt: skip
    G = schema.graph_from_dict(bundle)
    assert G.number_of_nodes() == 3 and G.number_of_edges() == 3 and G.graph["text_mode"] == "labels"


# ---------------------------------------------------------------- writing and round trips


@pytest.mark.parametrize("mode", ["labels", "full"])
def test_dump_and_load_round_trip(mode):
    d = v2_dict(mode)
    G = schema.graph_from_dict(d)
    assert schema.dump_graph(G) == d
    again = schema.graph_from_dict(schema.dump_graph(G))
    assert dict(again.nodes(data=True)) == dict(G.nodes(data=True))
    assert {(a, b): x for a, b, x in again.edges(data=True)} == {(a, b): x for a, b, x in G.edges(data=True)}
    assert again.graph == G.graph


@pytest.mark.parametrize("mode", ["labels", "full"])
def test_save_and_load_round_trip_on_disk(tmp_path, mode):
    G = schema.graph_from_dict(v2_dict(mode))
    path = schema.save_graph(G, tmp_path / "deep" / "math.graph.json")
    assert json.loads(path.read_text(encoding="utf-8")) == v2_dict(mode)
    H = schema.load_graph(path)
    assert dict(H.nodes(data=True)) == dict(G.nodes(data=True)) and H.graph == G.graph
    assert schema.save_graph(H, tmp_path / "again.json").read_bytes() == path.read_bytes()


def test_labels_mode_output_holds_no_curriculum_text():
    G = schema.graph_from_dict(v2_dict("full"))
    G.graph["text_mode"] = "labels"
    text = schema.dumps_graph(G)
    assert "made-up" not in text
    assert all(not set(node) & set(schema.TEXT_KEYS) for node in json.loads(text)["nodes"])
    stripped = schema.strip_text(schema.graph_from_dict(v2_dict("full")))
    assert stripped.graph["text_mode"] == "labels" and "made-up" not in schema.dumps_graph(stripped)
    assert all(d["description"] == d["short_label"] for _, d in stripped.nodes(data=True))


def test_output_is_sorted_and_does_not_depend_on_insertion_order():
    d = v2_dict()
    shuffled = {**d, "nodes": d["nodes"][::-1], "edges": d["edges"][::-1]}
    assert schema.dumps_graph(schema.graph_from_dict(shuffled)) == schema.dumps_graph(schema.graph_from_dict(d))


def test_node_order_is_numeric_not_alphabetic():
    codes = ["B4.1.10.1.1", "B4.1.2.1.1", "B10.1.1.1.1", "B4.1.2.1.10", "B4.1.2.1.9"]
    G = schema.graph_from_dict({"schema_version": 2, "nodes": [rec(c) for c in codes], "edges": []})
    assert [n["code"] for n in schema.dump_graph(G)["nodes"]] == [
        "B4.1.2.1.1",
        "B4.1.2.1.9",
        "B4.1.2.1.10",
        "B4.1.10.1.1",
        "B10.1.1.1.1",
    ]


def test_file_layout_is_one_node_or_edge_per_line_with_stable_key_order():
    text = schema.dumps_graph(schema.graph_from_dict(v2_dict()))
    lines = text.splitlines()
    assert text.endswith("}\n") and lines[1:4] == [
        ' "schema_version": 2,',
        ' "subject": "math",',
        ' "text_mode": "labels",',
    ]
    node_lines = [ln for ln in lines if ln.startswith('  {"id"')]
    edge_lines = [ln for ln in lines if ln.startswith('  {"source"')]
    assert len(node_lines) == 3 and len(edge_lines) == 3
    assert list(json.loads(node_lines[0].rstrip(","))) == list(rec("B1.1.1.1.1"))
    first_edge = json.loads(edge_lines[0].rstrip(","))
    assert list(first_edge) == ["source", "target", "type", "weight", "provenance"]
    assert list(first_edge["provenance"]) == ["by", "at", "review"]
    assert "—" not in text


def test_empty_graph_round_trip(tmp_path):
    G = schema.graph_from_dict(
        {"schema_version": 2, "subject": "math", "text_mode": "labels", "nodes": [], "edges": []}
    )
    H = schema.load_graph(schema.save_graph(G, tmp_path / "empty.json"))
    assert H.number_of_nodes() == 0 and H.graph["schema_version"] == 2


def test_non_ascii_text_is_written_as_it_is(tmp_path):
    G = schema.graph_from_dict(
        {"schema_version": 2, "nodes": [rec("B1.1.1.1.1", short_label="Count GH₵ coins")], "edges": []}
    )
    path = schema.save_graph(G, tmp_path / "g.json")
    assert "GH₵" in path.read_text(encoding="utf-8")


# ---------------------------------------------------------------- v1 files


def v1_file(tmp_path, edges_key="edges"):
    """A v1 node-link file as `kwan.graph.build.save` wrote it: bare codes, descriptions, no provenance."""
    nodes = [
        {"id": c, "grade": int(c[1]), "strand": 1, "strand_name": "Number", "sub_strand": 1,
         "content_standard": c[:8], "cs_key": "1.1.1", "indicator_no": int(c[-1]), "description": f"Made-up text {c}"}
        for c in ("B1.1.1.1.1", "B1.1.1.1.2", "B2.1.1.1.1")
    ]  # fmt: skip
    links = [
        {"type": "sequence", "weight": 1.0, "source": "B1.1.1.1.1", "target": "B1.1.1.1.2"},
        {"type": "vertical", "weight": 1.0, "source": "B1.1.1.1.2", "target": "B2.1.1.1.1"},
    ]
    path = tmp_path / "graph.json"
    path.write_text(json.dumps({"directed": True, "multigraph": False, "graph": {}, "nodes": nodes, edges_key: links}))
    return path


@pytest.mark.parametrize("edges_key", ["edges", "links"])
def test_load_graph_reads_a_v1_file_and_keeps_bare_ids(tmp_path, edges_key):
    G = schema.load_graph(v1_file(tmp_path, edges_key))
    assert list(G.nodes) == ["B1.1.1.1.1", "B1.1.1.1.2", "B2.1.1.1.1"]
    assert G.graph["schema_version"] == 1 and G.graph["subject"] == "math" and G.graph["text_mode"] == "full"
    assert G.nodes["B1.1.1.1.2"]["description"] == "Made-up text B1.1.1.1.2" and "id" not in G.nodes["B1.1.1.1.2"]
    assert G.edges["B1.1.1.1.2", "B2.1.1.1.1"] == {"type": "vertical", "weight": 1.0}


def test_v1_file_loads_the_same_as_with_the_v1_loader(tmp_path):
    from kwan.graph import build

    path = v1_file(tmp_path)
    old, new = build.load(path), schema.load_graph(path)
    assert dict(new.nodes(data=True)) == dict(old.nodes(data=True))
    assert list(new.edges(data=True)) == list(old.edges(data=True))


def test_dumping_a_v1_graph_upgrades_it_to_v2(tmp_path):
    d = schema.dump_graph(schema.load_graph(v1_file(tmp_path)))
    assert d["schema_version"] == 2 and d["text_mode"] == "full"
    assert [n["id"] for n in d["nodes"]] == ["math:B1.1.1.1.1", "math:B1.1.1.1.2", "math:B2.1.1.1.1"]
    assert d["nodes"][0]["code"] == "B1.1.1.1.1" and d["nodes"][0]["aliases"] == ["B1.1.1.1.1"]
    assert d["edges"][-1] == {
        "source": "math:B1.1.1.1.2",
        "target": "math:B2.1.1.1.1",
        "type": "vertical",
        "weight": 1.0,
        "provenance": {"by": "rule:vertical", "at": None, "review": None},
    }
    G2 = schema.graph_from_dict(d)
    assert G2.graph["schema_version"] == 2 and nx.is_directed_acyclic_graph(G2)


def test_load_graph_without_a_path_reads_the_default_file(tmp_path, monkeypatch):
    monkeypatch.setenv("KWAN_DATA_DIR", str(tmp_path))
    assert schema.default_graph_path() == tmp_path / "graph" / "math.graph.json"
    schema.save_graph(schema.graph_from_dict(v2_dict()), schema.default_graph_path())
    assert list(schema.load_graph().nodes) == ["math:B1.1.1.1.1", "math:B1.1.1.1.2", "math:B2.1.1.1.1"]


def test_schema_module_needs_only_the_standard_library_and_networkx():
    import subprocess
    import sys

    code = (
        "import sys, kwan.graph.schema, kwan.graph.build, kwan.graph.validate, kwan.graph._cli\n"
        "heavy = {'pyvis', 'pandas', 'numpy', 'yaml', 'sklearn', 'httpx'} & set(sys.modules)\n"
        "assert not heavy, heavy"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
