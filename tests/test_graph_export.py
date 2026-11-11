"""Graph export: the interactive HTML page holds every node, styles augmented edges, and respects the text mode."""

import json
import re

import pytest

pytest.importorskip("pyvis", reason="the graph page needs pyvis (pip install 'kwan[demo]')")

from kwan import cli  # noqa: E402
from kwan.graph import build, export, schema  # noqa: E402

STRANDS = {1: "Number", 2: "Algebra", 3: "Geometry and Measurement", 4: "Data"}
DAY = "2026-11-10"
CODES = ["B1.1.1.1.1", "B1.1.1.1.2", "B1.1.1.2.1", "B2.1.1.1.1", "B2.2.1.1.1", "B3.3.1.1.1", "B4.4.1.1.1"]


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


def graph(full=False):
    G = build.build_rule_graph([rec(c) for c in CODES], at=DAY)
    proposed = {
        "source": "B2.2.1.1.1",
        "target": "B3.3.1.1.1",
        "weight": 0.8,
        "provenance": {"by": "model:test-1", "at": DAY, "review": {"status": "accepted", "by": "t1", "at": DAY}},
    }
    G, _ = build.merge_augmented(G, [proposed])
    if full:
        G, _ = build.merge_full_text(
            G, [{"id": f"math:{c}", "description": f"Made-up sentence for {c}"} for c in CODES]
        )
    return G


def page_data(page, name):
    """The node or edge list that pyvis wrote into the page."""
    match = re.search(rf"{name} = new vis\.DataSet\((\[.*?\])\);", page, re.S)
    assert match, f"no {name} data in the page"
    return json.loads(match.group(1))


def test_page_contains_every_node_id():
    G = graph()
    page = export.graph_html(G)
    assert page.lstrip().startswith("<html>") and "vis-network" in page
    assert all(f'"{n}"' in page for n in G.nodes)
    nodes = {n["id"]: n for n in page_data(page, "nodes")}
    assert set(G.nodes) <= set(nodes)
    assert {n for n in nodes if n.startswith("heading:")} == {"heading:B1", "heading:B2", "heading:B3", "heading:B4"}


def test_nodes_are_coloured_and_shaped_by_strand_and_labelled_with_the_code():
    nodes = {n["id"]: n for n in page_data(export.graph_html(graph()), "nodes")}
    for strand, code in [(1, "B1.1.1.1.1"), (2, "B2.2.1.1.1"), (3, "B3.3.1.1.1"), (4, "B4.4.1.1.1")]:
        node = nodes[f"math:{code}"]
        colour, shape = export.STRAND_STYLE[strand]
        assert node["color"]["background"] == colour and node["shape"] == shape and node["label"] == code
        assert node["kwan_strand"] == strand and node["kwan_grade"] == int(code[1])
    assert (
        len({c for c, _ in export.STRAND_STYLE.values()}) == 4
        and len({s for _, s in export.STRAND_STYLE.values()}) == 4
    )


def test_only_augmented_edges_are_dashed():
    G = graph()
    edges = page_data(export.graph_html(G), "edges")
    assert len(edges) == G.number_of_edges() == 4
    kinds = {(e["from"], e["to"]): e for e in edges}
    assert kinds[("math:B2.2.1.1.1", "math:B3.3.1.1.1")]["kwan_type"] == "augmented"
    for e in edges:
        assert bool(e["dashes"]) == (e["kwan_type"] == "augmented"), e
        assert e["arrows"] == "to"
    assert {e["kwan_type"] for e in edges} == {"sequence", "vertical", "bridge", "augmented"}


def test_page_has_grade_and_strand_filters_and_a_legend():
    page = export.graph_html(graph())
    for g in (1, 2, 3, 4):
        assert f'data-kwan-grade="{g}"' in page
    assert 'data-kwan-grade="5"' not in page
    for s, name in STRANDS.items():
        assert f'data-kwan-strand="{s}"' in page and name in page
    assert "stroke-dasharray" in page and "augmented" in page and "nodes.update" in page
    assert "7 indicators, 4 prerequisite links" in page
    assert "bootstrap" not in page and "—" not in page


def test_labels_mode_tooltip_shows_code_and_label_only():
    page = export.graph_html(graph())
    node = {n["id"]: n for n in page_data(page, "nodes")}["math:B1.1.1.1.2"]
    assert node["title"] == "B1.1.1.1.2\nTest skill B1.1.1.1.2\nB1, Number"
    assert "shown with short labels" in page


def test_description_appears_only_in_full_text_mode():
    full = graph(full=True)
    page = export.graph_html(full)
    title = {n["id"]: n for n in page_data(page, "nodes")}["math:B1.1.1.1.2"]["title"]
    assert title.startswith("B1.1.1.1.2\nTest skill B1.1.1.1.2\n") and title.endswith("Made-up sentence for B1.1.1.1.2")
    assert "shown with full curriculum text" in page
    relabelled = full.copy()
    relabelled.graph["text_mode"] = "labels"  # even with the text on the nodes, labels mode never shows it
    assert "Made-up" not in export.graph_html(relabelled)
    assert "Made-up" not in export.graph_html(schema.strip_text(full))


def test_node_tooltip_wraps_long_text():
    data = {
        "code": "B1.1.1.1.1",
        "short_label": "Short label",
        "grade": 1,
        "strand_name": "Number",
        "description": "word " * 60,
    }
    lines = export.node_tooltip("math:B1.1.1.1.1", data, "full").split("\n")
    assert lines[:3] == ["B1.1.1.1.1", "Short label", "B1, Number"] and max(len(x) for x in lines) <= 64
    assert export.node_tooltip("math:B1.1.1.1.1", data, "labels") == "B1.1.1.1.1\nShort label\nB1, Number"


def test_grades_argument_keeps_only_those_grades():
    page = export.graph_html(graph(), grades=[1, 2])
    ids_in_page = {n["id"] for n in page_data(page, "nodes")}
    assert ids_in_page == {f"math:{c}" for c in CODES if c[1] in "12"} | {"heading:B1", "heading:B2"}
    assert 'data-kwan-grade="3"' not in page and "5 indicators" in page


def test_layout_puts_grades_in_columns_and_content_standards_in_rows():
    G = graph()
    pos = export.layout(G)
    assert set(pos) == set(G.nodes) and len(set(pos.values())) == len(pos)
    x = {c: pos[f"math:{c}"][0] for c in CODES}
    y = {c: pos[f"math:{c}"][1] for c in CODES}
    assert x["B1.1.1.1.1"] < x["B1.1.1.1.2"] < x["B2.1.1.1.1"] < x["B3.3.1.1.1"] < x["B4.4.1.1.1"]
    assert y["B1.1.1.1.1"] == y["B1.1.1.1.2"] == y["B2.1.1.1.1"] < y["B1.1.1.2.1"] < y["B2.2.1.1.1"] < y["B3.3.1.1.1"]
    assert export.layout(G) == pos


def test_inline_page_embeds_the_drawing_library():
    remote, inline = export.graph_html(graph()), export.graph_html(graph(), inline=True)
    assert "cdnjs.cloudflare.com" in remote and "cdnjs.cloudflare.com" not in inline
    assert len(inline) > len(remote) + 100_000


def test_page_is_the_same_on_every_run_and_a_v1_graph_can_be_drawn(tmp_path):
    assert export.graph_html(graph()) == export.graph_html(graph())
    v1 = build.build_graph([{**rec(c), "id": c, "description": f"Made-up v1 text {c}"} for c in CODES])
    page = export.graph_html(v1)
    assert all(f'"{c}"' in page for c in CODES) and "Made-up" not in page  # no text_mode on the graph: labels


def test_write_html_and_cli_export(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("KWAN_DATA_DIR", str(tmp_path / "data"))
    schema.save_graph(graph(), schema.default_graph_path())
    out = tmp_path / "site" / "graph.html"
    assert cli.main(["graph", "export", "--out", str(out)]) == 0
    page = out.read_text(encoding="utf-8")
    assert all(f'"math:{c}"' in page for c in CODES)
    assert "7 nodes, 4 edges (labels mode)" in capsys.readouterr().out
    assert export.write_html(graph(), tmp_path / "again.html").read_text(encoding="utf-8") == page


def test_cli_export_strips_text_unless_full_is_asked_for(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("KWAN_DATA_DIR", str(tmp_path / "data"))
    full_path = schema.save_graph(graph(full=True), tmp_path / "build" / "graph" / "math.graph.full.json")
    assert cli.main(["graph", "export", "--graph", str(full_path), "--out", str(tmp_path / "labels.html")]) == 0
    assert "Made-up" not in (tmp_path / "labels.html").read_text(encoding="utf-8")
    assert cli.main(["graph", "export", "--full", "--out", str(tmp_path / "full.html")]) == 0
    assert "Made-up sentence for B1.1.1.1.1" in (tmp_path / "full.html").read_text(encoding="utf-8")
    assert "do not commit" in capsys.readouterr().out
