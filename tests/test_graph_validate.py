"""Graph validation: the numbers of the report on tiny graphs, the Markdown page, the results block, the command."""

import json

import networkx as nx
import pytest

from kwan import cli
from kwan.graph import build, schema, validate

STRANDS = {1: "Number", 2: "Algebra", 3: "Geometry and Measurement", 4: "Data"}
DAY = "2026-11-10"


def rec(code, **extra):
    g, s, ss, cs, i = (int(x) for x in code[1:].split("."))
    return {
        "id": f"math:{code}",
        "grade": g,
        "strand": s,
        "strand_name": STRANDS[s],
        "sub_strand": ss,
        "sub_strand_name": f"Sub-strand {s}.{ss}",
        "content_standard": f"B{g}.{s}.{ss}.{cs}",
        "cs_key": f"{s}.{ss}.{cs}",
        "indicator_no": i,
        "short_label": f"Test skill {code}",
        **extra,
    }


def tiny():
    """Eight indicators, seven edges of all four kinds, one isolated node.

    B1.1.1.1.1 -> B1.1.1.1.2 (sequence) -> B1.1.1.2.1 (bridge, 0.5)
    B1.1.1.1.2 -> B2.1.1.1.1 (vertical) -> B2.1.1.1.2 (sequence) -> B2.1.1.2.1 (bridge, 0.5)
    B1.1.1.2.1 -> B2.1.1.2.1 (vertical);  B1.4.1.1.1 -> B2.1.1.2.1 (augmented, 0.8);  B2.3.1.1.1 is isolated
    """
    codes = [
        "B1.1.1.1.1",
        "B1.1.1.1.2",
        "B1.1.1.2.1",
        "B2.1.1.1.1",
        "B2.1.1.1.2",
        "B2.1.1.2.1",
        "B1.4.1.1.1",
        "B2.3.1.1.1",
    ]
    G = build.build_rule_graph([rec(c) for c in codes], at=DAY)
    proposed = {
        "source": "B1.4.1.1.1",
        "target": "B2.1.1.2.1",
        "weight": 0.8,
        "provenance": {"by": "model:test-1", "at": DAY, "review": {"status": "accepted", "by": "t1", "at": DAY}},
    }
    G, _ = build.merge_augmented(G, [proposed])
    return G


def test_analyse_numbers_on_a_tiny_graph():
    s = validate.analyse(tiny())
    assert s["ok"] is True
    assert s["checks"] == {"is_dag": True, "no_orphan_edges": True, "all_reachable_from_a_root": True}
    assert (s["nodes"], s["edges"]) == (8, 7)
    assert s["edges_by_type"] == {"augmented": 1, "bridge": 2, "sequence": 2, "vertical": 2}
    assert s["edges_by_provenance"] == {"model:test-1": 1, "rule:bridge": 2, "rule:sequence": 2, "rule:vertical": 2}
    assert s["edges_without_provenance"] == 0 and s["augmented_edges_by_review"] == {"accepted": 1}
    assert s["roots"] == 3 and s["root_codes"] == ["B1.1.1.1.1", "B1.4.1.1.1", "B2.3.1.1.1"]
    assert s["leaves"] == 2 and s["isolated_nodes"] == ["B2.3.1.1.1"]  # B2.1.1.2.1 and the isolated node
    assert s["unreachable_from_a_root"] == [] and s["orphan_edges"] == [] and s["edges_to_an_earlier_grade"] == []
    assert s["components"] == 2
    assert s["longest_path_edges"] == 4
    assert s["longest_path"] == ["B1.1.1.1.1", "B1.1.1.1.2", "B2.1.1.1.1", "B2.1.1.1.2", "B2.1.1.2.1"]
    assert s["longest_path_weighted"] == 3.5  # sequence 1 + vertical 1 + sequence 1 + bridge 0.5
    assert s["nodes_by_strand"] == {
        "1": {"name": "Number", "nodes": 6},
        "3": {"name": "Geometry and Measurement", "nodes": 1},
        "4": {"name": "Data", "nodes": 1},
    }


def test_analyse_degree_by_grade():
    by_grade = validate.analyse(tiny())["by_grade"]
    assert by_grade["1"] == {
        "nodes": 4,
        "roots": 2,
        "leaves": 0,
        "edges_in": 2,
        "edges_out": 5,
        "in_degree_mean": 0.5,
        "in_degree_max": 1,
        "out_degree_mean": 1.25,
        "out_degree_max": 2,
        "edges_inside": 2,
        "edges_to_later_grades": 3,
    }
    assert by_grade["2"] == {
        "nodes": 4,
        "roots": 1,
        "leaves": 2,
        "edges_in": 5,
        "edges_out": 2,
        "in_degree_mean": 1.25,
        "in_degree_max": 3,
        "out_degree_mean": 0.5,
        "out_degree_max": 1,
        "edges_inside": 2,
        "edges_to_later_grades": 0,
    }


def test_weighted_longest_path_counts_a_bridge_as_half():
    G = build.build_rule_graph([rec(c) for c in ["B1.1.1.1.1", "B1.1.1.2.1", "B1.1.1.3.1"]], at=DAY)
    s = validate.analyse(G)
    assert s["longest_path_edges"] == 2 and s["longest_path_weighted"] == 1.0
    assert s["longest_path"] == ["B1.1.1.1.1", "B1.1.1.2.1", "B1.1.1.3.1"]


def test_a_cycle_fails_the_dag_and_reachability_checks():
    G = tiny()
    G.add_edge("math:B2.1.1.1.2", "math:B1.1.1.1.1", type="augmented", weight=0.5, provenance={"by": "model:x"})
    s = validate.analyse(G)
    assert s["ok"] is False and s["checks"]["is_dag"] is False
    assert s["longest_path_edges"] is None and s["longest_path_weighted"] is None
    assert s["cycle"][0] == s["cycle"][-1] and len(s["cycle"]) == 5
    assert s["edges_to_an_earlier_grade"] == [["B2.1.1.1.2", "B1.1.1.1.1"]]
    assert "B1.1.1.1.1" in s["unreachable_from_a_root"]  # the cycle has no way in from a root
    report = validate.render_report(s)
    assert "At least one check fails." in report and "FAIL" in report and "## Problems to look at" in report


def test_orphan_edges_are_counted_when_the_file_is_loaded_leniently(tmp_path):
    d = schema.dump_graph(tiny())
    d["edges"].append({"source": "math:B1.1.1.1.1", "target": "math:B7.1.1.1.1", "type": "vertical", "weight": 1.0})
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(d), encoding="utf-8")
    s = validate.validate(path, report=tmp_path / "report.md", write_results=False)
    assert s["ok"] is False and s["checks"]["no_orphan_edges"] is False
    assert s["orphan_edges"] == [["math:B1.1.1.1.1", "math:B7.1.1.1.1"]] and s["nodes"] == 8
    assert "orphan edges: 1" in (tmp_path / "report.md").read_text(encoding="utf-8")


def test_empty_graph_is_analysed_without_error():
    s = validate.analyse(nx.DiGraph(subject="math"))
    assert s["nodes"] == 0 and s["edges"] == 0 and s["ok"] is True and s["longest_path_edges"] == 0
    assert "# Graph validation: math" in validate.render_report(s)


def test_report_shows_the_numbers_and_only_codes():
    G, _ = build.merge_full_text(
        tiny(), [{"id": f"math:{c}", "description": f"Made-up sentence {c}"} for c in ["B1.1.1.1.1"]]
    )
    s = validate.analyse(G)
    report = validate.render_report(s, graph_file="data/graph/math.graph.json")
    assert "**All checks pass.**" in report
    assert "| Nodes (indicators) | 8 |" in report and "| Edges (prerequisite links) | 7 |" in report
    assert "| B1 | 4 | 2 | 0 | 2 | 0.50 | 1 | 5 | 1.25 | 2 |" in report
    assert "| B2 | 4 | 1 | 2 | 5 | 1.25 | 3 | 2 | 0.50 | 1 |" in report
    assert "| augmented | 1 |" in report and "| `rule:bridge` | 2 |" in report and "| `model:test-1` | 1 |" in report
    assert "has **4 edges**" in report and "weighs **3.5**" in report
    assert "Isolated nodes (1): `B2.3.1.1.1`" in report
    assert "Made-up" not in report and "Test skill" not in report  # codes and numbers only
    assert "—" not in report and "–" not in report


def test_results_block_holds_the_key_numbers():
    block = validate.results_block(
        validate.analyse(tiny()), graph_file="data/graph/math.graph.json", report="docs/qa/graph-math.md"
    )
    assert block == {
        "graph_file": "data/graph/math.graph.json",
        "report": "docs/qa/graph-math.md",
        "subject": "math",
        "schema_version": 2,
        "text_mode": "labels",
        "ok": True,
        "is_dag": True,
        "nodes": 8,
        "edges": 7,
        "edges_by_type": {"augmented": 1, "bridge": 2, "sequence": 2, "vertical": 2},
        "edges_by_provenance": {"model:test-1": 1, "rule:bridge": 2, "rule:sequence": 2, "rule:vertical": 2},
        "orphan_edges": 0,
        "roots": 3,
        "leaves": 2,
        "isolated_nodes": 1,
        "unreachable_from_a_root": 0,
        "components": 2,
        "longest_path_edges": 4,
        "longest_path_weighted": 3.5,
        "edges_to_an_earlier_grade": 0,
        "nodes_by_grade": {"1": 4, "2": 4},
        "nodes_by_strand": {"1": 6, "3": 1, "4": 1},
    }
    json.dumps(block)


def test_validate_writes_the_report_and_merges_only_its_own_results_key(tmp_path):
    graph_path = schema.save_graph(tiny(), tmp_path / "math.graph.json")
    results = tmp_path / "eval_results.json"
    results.write_text(json.dumps({"latency": {"p95_ms": 1.5}}), encoding="utf-8")
    stats = validate.validate(graph_path, report=tmp_path / "qa" / "graph-math.md", results=results)
    assert stats["ok"] and (tmp_path / "qa" / "graph-math.md").read_text(encoding="utf-8").startswith(
        "# Graph validation"
    )
    stored = json.loads(results.read_text(encoding="utf-8"))
    assert set(stored) == {"latency", "graph"} and stored["latency"] == {"p95_ms": 1.5}
    assert stored["graph"]["nodes"] == 8 and stored["graph"]["edges_by_type"]["augmented"] == 1


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A made-up repository layout with the tiny graph at the default location."""
    monkeypatch.setenv("KWAN_DATA_DIR", str(tmp_path / "data"))
    schema.save_graph(tiny(), schema.default_graph_path())
    return tmp_path


def test_cli_graph_validate_uses_the_standard_locations(repo, capsys):
    assert cli.main(["graph", "validate"]) == 0
    out = capsys.readouterr().out
    assert "pass  is dag" in out and "8 nodes, 7 edges, 3 roots, 1 isolated, longest path 4 edges (weighted 3.5)" in out
    report = (repo / "docs" / "qa" / "graph-math.md").read_text(encoding="utf-8")
    assert "Graph file: `data/graph/math.graph.json` (schema version 2, labels mode)." in report
    block = json.loads((repo / "data" / "eval_results.json").read_text(encoding="utf-8"))["graph"]
    assert block["graph_file"] == "data/graph/math.graph.json" and block["report"] == "docs/qa/graph-math.md"


def test_cli_graph_validate_returns_an_error_code_when_a_check_fails(repo, capsys):
    d = schema.dump_graph(tiny())
    d["edges"].append({"source": "math:B2.1.1.1.2", "target": "math:B1.1.1.1.1", "type": "augmented", "weight": 0.5})
    schema.default_graph_path().write_text(json.dumps(d), encoding="utf-8")
    assert cli.main(["graph", "validate", "--no-results"]) == 1
    assert "FAIL  is dag" in capsys.readouterr().out and not (repo / "data" / "eval_results.json").exists()
