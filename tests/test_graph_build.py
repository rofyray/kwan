"""Schema v2 builder: rule edges with provenance, the v1 edge set, full text, augmented edges, `kwan graph build`."""

import json
import random
from collections import Counter
from pathlib import Path

import networkx as nx
import pytest

from kwan import cli
from kwan.graph import build, schema

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
V1_INDICATORS = FIXTURES / "indicators_v1.json"  # the frozen v1 files, descriptions replaced by short labels
V1_GRAPH = FIXTURES / "graph_v1.json"
STRANDS = {1: "Number", 2: "Algebra", 3: "Geometry and Measurement", 4: "Data"}
WEIGHT = {"sequence": 1.0, "vertical": 1.0, "bridge": 0.5}
DAY = "2026-11-10"


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


def v2(*codes, at=DAY):
    return build.build_rule_graph([rec(c) for c in codes], at=at)


def edges(G):
    """{(source code, target code): (type, weight)}"""
    return {(a.split(":")[-1], b.split(":")[-1]): (d["type"], d["weight"]) for a, b, d in G.edges(data=True)}


def augmented(a, b, weight=0.8, status="accepted"):
    review = None if status is None else {"status": status, "by": "teacher-1", "at": "2026-11-11"}
    return {
        "source": f"math:{a}",
        "target": f"math:{b}",
        "type": "augmented",
        "weight": weight,
        "provenance": {"by": "model:test-1", "at": DAY, "review": review},
    }


# ---------------------------------------------------------------- rule edges on tiny records


def test_nodes_use_canonical_ids_and_carry_the_record():
    G = v2("B4.1.1.1.2")
    assert list(G.nodes) == ["math:B4.1.1.1.2"] and G.number_of_edges() == 0
    assert G.nodes["math:B4.1.1.1.2"] == {
        **{k: v for k, v in rec("B4.1.1.1.2").items() if k != "id"},
        "description": "Test skill B4.1.1.1.2",
    }
    assert G.graph == {"schema_version": 2, "subject": "math", "text_mode": "labels"}


def test_bare_codes_and_missing_labels_are_filled_in():
    record = {k: v for k, v in rec("B4.1.1.1.2").items() if k not in ("subject", "code", "short_label", "aliases")}
    record["id"] = "B4.1.1.1.2"
    G = build.build_rule_graph([record], at=DAY)
    node = G.nodes["math:B4.1.1.1.2"]
    assert node["subject"] == "math" and node["code"] == "B4.1.1.1.2" and node["aliases"] == ["B4.1.1.1.2"]
    assert node["short_label"] == "Skill B4.1.1.1.2" == node["description"]


def test_printed_code_aliases_are_kept_after_the_bare_code():
    G = build.build_rule_graph([rec("B6.1.2.6.1", aliases=["B6.1.1.6.1"])], at=DAY)
    assert G.nodes["math:B6.1.2.6.1"]["aliases"] == ["B6.1.2.6.1", "B6.1.1.6.1"]


def test_sequence_edges_chain_a_content_standard():
    assert edges(v2("B1.1.1.1.1", "B1.1.1.1.2", "B1.1.1.1.3")) == {
        ("B1.1.1.1.1", "B1.1.1.1.2"): ("sequence", 1.0),
        ("B1.1.1.1.2", "B1.1.1.1.3"): ("sequence", 1.0),
    }


def test_sequence_skips_over_a_missing_indicator_number():
    assert edges(v2("B1.1.1.1.1", "B1.1.1.1.3", "B1.1.1.1.4")) == {
        ("B1.1.1.1.1", "B1.1.1.1.3"): ("sequence", 1.0),
        ("B1.1.1.1.3", "B1.1.1.1.4"): ("sequence", 1.0),
    }


def test_vertical_edge_goes_to_the_nearest_later_grade_that_has_the_content_standard():
    assert edges(v2("B1.2.1.1.1", "B1.2.1.1.2", "B2.3.1.1.1", "B3.2.1.1.1", "B6.2.1.1.1")) == {
        ("B1.2.1.1.1", "B1.2.1.1.2"): ("sequence", 1.0),
        ("B1.2.1.1.2", "B3.2.1.1.1"): ("vertical", 1.0),  # B2 has no 2.1.1
        ("B3.2.1.1.1", "B6.2.1.1.1"): ("vertical", 1.0),
    }


def test_bridge_links_consecutive_content_standards_of_a_sub_strand_at_half_weight():
    assert edges(v2("B1.1.2.1.1", "B1.1.2.1.2", "B1.1.2.2.1", "B1.1.2.10.1", "B1.2.2.3.1", "B2.1.2.3.1")) == {
        ("B1.1.2.1.1", "B1.1.2.1.2"): ("sequence", 1.0),
        ("B1.1.2.1.2", "B1.1.2.2.1"): ("bridge", 0.5),
        ("B1.1.2.2.1", "B1.1.2.10.1"): ("bridge", 0.5),  # numeric order: 10 comes after 2
    }


def test_no_edge_across_strands_or_sub_strands():
    assert v2("B1.1.1.1.1", "B1.2.1.1.1", "B1.1.2.1.1", "B2.3.3.2.1").number_of_edges() == 0


def test_result_does_not_depend_on_input_order():
    codes = ["B1.1.1.1.1", "B1.1.1.1.2", "B1.1.1.2.1", "B2.1.1.1.1", "B2.1.1.2.1", "B2.1.1.2.2", "B3.1.1.2.1"]
    expected = schema.dumps_graph(v2(*codes))
    rng = random.Random(7)
    for _ in range(5):
        shuffled = codes[:]
        rng.shuffle(shuffled)
        assert schema.dumps_graph(v2(*shuffled)) == expected


def test_empty_input_gives_an_empty_graph():
    G = build.build_rule_graph([], at=DAY)
    assert G.number_of_nodes() == 0 and G.graph["text_mode"] == "labels"


def test_duplicate_and_incomplete_records_are_rejected():
    with pytest.raises(ValueError, match="listed twice"):
        build.build_rule_graph([rec("B1.1.1.1.1"), {**rec("B1.1.1.1.1"), "id": "B1.1.1.1.1"}], at=DAY)
    with pytest.raises(ValueError, match="no 'cs_key'"):
        build.build_rule_graph([{"id": "math:B1.1.1.1.1", "grade": 1}], at=DAY)
    with pytest.raises(ValueError, match="one subject"):
        build.build_rule_graph(
            [rec("B1.1.1.1.1"), {**rec("B1.1.1.1.2"), "id": "english:B1.1.1.1.2", "subject": "english"}]
        )


# ---------------------------------------------------------------- provenance


def test_every_rule_edge_has_provenance_with_rule_name_and_date():
    G = v2("B1.1.1.1.1", "B1.1.1.1.2", "B1.1.1.2.1", "B2.1.1.1.1", "B2.1.1.2.1")
    assert Counter(d["type"] for *_, d in G.edges(data=True)) == {"sequence": 1, "vertical": 2, "bridge": 2}
    for a, b, d in G.edges(data=True):
        assert set(d) == {"type", "weight", "provenance"}, (a, b)
        assert d["provenance"] == {"by": f"rule:{d['type']}", "at": DAY, "review": None}, (a, b)
        assert d["weight"] == WEIGHT[d["type"]]


def test_default_date_is_today_as_an_iso_date():
    import datetime as dt

    G = build.build_rule_graph([rec("B1.1.1.1.1"), rec("B1.1.1.1.2")])
    at = G.edges["math:B1.1.1.1.1", "math:B1.1.1.1.2"]["provenance"]["at"]
    assert at == build.today() and dt.date.fromisoformat(at)


# ---------------------------------------------------------------- the v1 edge set


def random_curriculum(seed):
    """Indicator codes of a small made-up curriculum, in curriculum order."""
    rng = random.Random(seed)
    codes = []
    for g in range(1, 7):
        for s in range(1, 4):
            for ss in range(1, 3):
                for cs in sorted(rng.sample(range(1, 6), rng.randint(0, 3))):
                    codes += [f"B{g}.{s}.{ss}.{cs}.{i}" for i in range(1, rng.randint(1, 4) + 1)]
    return codes


@pytest.mark.parametrize("seed", range(8))
def test_v2_rules_give_the_same_edges_as_the_v1_builder_on_made_up_curricula(seed):
    codes = random_curriculum(seed)
    v1_records = [{"id": c, "grade": int(c[1]), "cs_key": ".".join(c[1:].split(".")[1:4])} for c in codes]
    old = build.build_graph(v1_records)
    new = v2(*codes)
    assert [n.split(":")[1] for n in new.nodes] == list(old.nodes)
    assert edges(new) == {(a, b): (d["type"], d["weight"]) for a, b, d in old.edges(data=True)}
    assert nx.is_directed_acyclic_graph(new)


def test_v2_rules_on_the_v1_indicator_set_give_exactly_the_v1_edge_set():
    indicators = json.loads(V1_INDICATORS.read_text(encoding="utf-8"))
    v1 = json.loads(V1_GRAPH.read_text(encoding="utf-8"))
    G = build.build_rule_graph(build.structure_from_v1(indicators), at=DAY)
    assert [n.split(":")[1] for n in G.nodes] == [n["id"] for n in v1["nodes"]]
    assert edges(G) == {(e["source"], e["target"]): (e["type"], e["weight"]) for e in v1["edges"]}
    assert G.number_of_edges() == len(v1["edges"]) == 287
    assert Counter(d["type"] for *_, d in G.edges(data=True)) == {"sequence": 150, "vertical": 82, "bridge": 55}
    assert all(
        d["provenance"] == {"by": f"rule:{d['type']}", "at": DAY, "review": None} for *_, d in G.edges(data=True)
    )


def test_structure_from_v1_drops_the_curriculum_text():
    v1 = {"id": "B4.1.1.1.2", "grade": 4, "strand": 1, "strand_name": "Number", "sub_strand": 1,
          "content_standard": "B4.1.1.1", "cs_key": "1.1.1", "indicator_no": 2, "description": "Made-up sentence"}  # fmt: skip
    (out,) = build.structure_from_v1([v1])
    assert out == {**rec("B4.1.1.1.2"), "sub_strand_name": "", "short_label": "Skill B4.1.1.1.2"}
    assert "Made-up" not in json.dumps(out)


# ---------------------------------------------------------------- full text


def full_records():
    return [
        {"id": "math:B1.1.1.1.1", "description": "Made-up description one", "content_standard_text": "Made-up standard",
         "exemplars": [{"text": "Made-up exemplar", "ghana_context": True}], "page": 3},
        {"id": "B1.1.1.1.9", "description": "Made-up description two"},  # matched through an alias
        {"id": "math:B9.9.9.9.9", "description": "No such node"},
    ]  # fmt: skip


def test_merge_full_text_adds_text_to_a_copy_and_reports_nodes_without_any():
    G = build.build_rule_graph(
        [rec("B1.1.1.1.1"), rec("B1.1.1.1.2", aliases=["B1.1.1.1.9"]), rec("B1.1.1.1.3")], at=DAY
    )
    H, without = build.merge_full_text(G, full_records())
    assert H.graph["text_mode"] == "full" and G.graph["text_mode"] == "labels"
    assert H.nodes["math:B1.1.1.1.1"]["description"] == "Made-up description one"
    assert (
        H.nodes["math:B1.1.1.1.1"]["exemplars"][0]["ghana_context"] is True and H.nodes["math:B1.1.1.1.1"]["page"] == 3
    )
    assert H.nodes["math:B1.1.1.1.2"]["description"] == "Made-up description two"
    assert H.nodes["math:B1.1.1.1.3"]["description"] == "Test skill B1.1.1.1.3" and without == ["math:B1.1.1.1.3"]
    assert G.nodes["math:B1.1.1.1.1"]["description"] == "Test skill B1.1.1.1.1"  # the labels graph is untouched
    assert list(H.edges(data=True)) == list(G.edges(data=True)) and "math:B9.9.9.9.9" not in H


def test_labels_mode_build_ignores_text_in_the_records():
    with_text = [rec("B1.1.1.1.1", description="Made-up description", exemplars=[{"text": "Made-up exemplar"}])]
    labels = build.build_rule_graph(with_text, at=DAY, text_mode="labels")
    assert "Made-up" not in schema.dumps_graph(labels) and labels.nodes["math:B1.1.1.1.1"]["description"].startswith(
        "Test"
    )
    assert build.build_rule_graph(with_text, at=DAY).graph["text_mode"] == "full"


# ---------------------------------------------------------------- augmented edges


def chain():
    """B1.1.1.1.1 -> .2 -> .3 (sequence) -> B2.1.1.1.1 (vertical), plus an unrelated B1.2.1.1.1."""
    return v2("B1.1.1.1.1", "B1.1.1.1.2", "B1.1.1.1.3", "B2.1.1.1.1", "B1.2.1.1.1")


def test_only_accepted_augmented_edges_are_merged():
    proposed = [
        augmented("B1.2.1.1.1", "B2.1.1.1.1", 0.9, "accepted"),
        augmented("B1.2.1.1.1", "B1.1.1.1.3", 0.8, "pending"),
        augmented("B1.2.1.1.1", "B1.1.1.1.2", 0.7, "rejected"),
        augmented("B1.2.1.1.1", "B1.1.1.1.1", 0.6, None),  # no review record counts as pending
    ]
    G = chain()
    H, report = build.merge_augmented(G, proposed)
    added = set(H.edges) - set(G.edges)
    assert added == {("math:B1.2.1.1.1", "math:B2.1.1.1.1")} and G.number_of_edges() == 3
    assert H.edges["math:B1.2.1.1.1", "math:B2.1.1.1.1"] == {
        "type": "augmented",
        "weight": 0.9,
        "provenance": {
            "by": "model:test-1",
            "at": DAY,
            "review": {"status": "accepted", "by": "teacher-1", "at": "2026-11-11"},
        },
    }
    assert report == {"merged": 1, "accepted": 1, "pending": 2, "rejected": 1, "skipped": []}


def test_cycle_guard_skips_an_accepted_edge_that_would_close_a_cycle():
    H, report = build.merge_augmented(chain(), [augmented("B2.1.1.1.1", "B1.1.1.1.1", 0.95)])
    assert nx.is_directed_acyclic_graph(H) and H.number_of_edges() == 3
    assert report["merged"] == 0 and report["accepted"] == 1
    assert report["skipped"] == [
        {"source": "math:B2.1.1.1.1", "target": "math:B1.1.1.1.1", "reason": "would close a cycle"}
    ]


def test_of_two_accepted_edges_that_cannot_both_stay_the_stronger_wins():
    weak, strong = augmented("B2.1.1.1.1", "B1.2.1.1.1", 0.6), augmented("B1.2.1.1.1", "B1.1.1.1.1", 0.9)
    for order in ([weak, strong], [strong, weak]):
        H, report = build.merge_augmented(chain(), order)
        assert H.has_edge("math:B1.2.1.1.1", "math:B1.1.1.1.1") and not H.has_edge("math:B2.1.1.1.1", "math:B1.2.1.1.1")
        assert nx.is_directed_acyclic_graph(H) and report["merged"] == 1
        assert [s["reason"] for s in report["skipped"]] == ["would close a cycle"]


def test_accepted_edges_with_other_problems_are_skipped_with_a_reason():
    proposed = [
        augmented("B1.1.1.1.1", "B1.1.1.1.2", 0.9),  # the rule already links them
        augmented("B1.1.1.1.1", "B9.9.9.9.9", 0.9),
        augmented("B1.2.1.1.1", "B1.2.1.1.1", 0.9),
        augmented("B1.2.1.1.1", "B1.1.1.1.1", None),
        augmented("B1.2.1.1.1", "B1.1.1.1.3", 0.7),
        augmented("B1.2.1.1.1", "B1.1.1.1.3", 0.6),  # the same edge proposed twice
    ]
    H, report = build.merge_augmented(chain(), proposed)
    assert report["merged"] == 1 and H.edges["math:B1.2.1.1.1", "math:B1.1.1.1.3"]["weight"] == 0.7
    assert H.edges["math:B1.1.1.1.1", "math:B1.1.1.1.2"]["type"] == "sequence"
    assert sorted(s["reason"] for s in report["skipped"]) == [
        "already linked (augmented edge)",
        "already linked (sequence edge)",
        "links an indicator to itself",
        "no weight",
        "unknown indicator",
    ]


def test_augmented_edges_may_name_indicators_by_bare_code_or_alias():
    G = build.build_rule_graph([rec("B1.1.1.1.1"), rec("B1.2.1.1.1", aliases=["B1.2.1.1.9"])], at=DAY)
    edge = {**augmented("x", "y"), "source": "B1.2.1.1.9", "target": "B1.1.1.1.1"}
    H, report = build.merge_augmented(G, [edge])
    assert report["merged"] == 1 and H.has_edge("math:B1.2.1.1.1", "math:B1.1.1.1.1")


# ---------------------------------------------------------------- build() and `kwan graph build`


@pytest.fixture
def workspace(tmp_path):
    """Structure, full-text and augmented files of a five-indicator curriculum, and the paths `build` needs."""
    codes = ["B1.1.1.1.1", "B1.1.1.1.2", "B1.1.1.1.3", "B2.1.1.1.1", "B1.2.1.1.1"]
    paths = {
        "structure": tmp_path / "data" / "curriculum" / "math.structure.json",
        "out": tmp_path / "data" / "graph" / "math.graph.json",
        "augmented": tmp_path / "data" / "graph" / "math.augmented.json",
        "full_text": tmp_path / "build" / "curriculum" / "math.full.json",
        "full_out": tmp_path / "build" / "graph" / "math.graph.full.json",
    }
    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    paths["structure"].write_text(json.dumps([rec(c) for c in codes]), encoding="utf-8")
    full = [
        rec(c, description=f"Made-up description {c}", exemplars=[{"text": "Made-up exemplar"}], page=1) for c in codes
    ]
    paths["full_text"].write_text(json.dumps(full), encoding="utf-8")
    proposed = [
        augmented("B1.2.1.1.1", "B2.1.1.1.1", 0.9, "accepted"),
        augmented("B1.2.1.1.1", "B1.1.1.1.2", 0.8, "pending"),
        augmented("B1.2.1.1.1", "B1.1.1.1.3", 0.7, "rejected"),
        augmented("B2.1.1.1.1", "B1.1.1.1.1", 0.6, "accepted"),  # would close a cycle
    ]
    paths["augmented"].write_text(json.dumps({"edges": proposed}), encoding="utf-8")
    return paths


def test_build_writes_the_labels_graph_and_the_full_text_graph(workspace):
    result = build.build(**workspace, at=DAY)
    labels = json.loads(workspace["out"].read_text(encoding="utf-8"))
    full = json.loads(workspace["full_out"].read_text(encoding="utf-8"))
    assert labels["text_mode"] == "labels" and full["text_mode"] == "full"
    assert "Made-up" not in workspace["out"].read_text(encoding="utf-8")
    assert all(n["description"].startswith("Made-up description") and n["exemplars"] for n in full["nodes"])
    assert labels["edges"] == full["edges"] and [n["id"] for n in labels["nodes"]] == [n["id"] for n in full["nodes"]]
    assert Counter(e["type"] for e in labels["edges"]) == {"sequence": 2, "vertical": 1, "augmented": 1}
    assert all(e["provenance"]["by"] and "review" in e["provenance"] for e in labels["edges"])
    assert result.augmented["merged"] == 1 and result.augmented["pending"] == 1 and result.augmented["rejected"] == 1
    assert [s["reason"] for s in result.augmented["skipped"]] == ["would close a cycle"]
    assert result.without_text == [] and result.full_path == workspace["full_out"]
    assert nx.is_directed_acyclic_graph(schema.load_graph(workspace["out"]))


def test_build_without_optional_files_writes_rule_edges_only(workspace):
    workspace["augmented"].unlink()
    workspace["full_text"].unlink()
    result = build.build(**workspace, at=DAY)
    assert result.augmented is None and result.full_graph is None and not workspace["full_out"].exists()
    types = Counter(d["type"] for *_, d in schema.load_graph(workspace["out"]).edges(data=True))
    assert types == {"sequence": 2, "vertical": 1}


def test_rebuild_keeps_the_date_of_edges_that_were_already_there(workspace):
    build.build(**workspace, at="2026-01-05")
    first = workspace["out"].read_bytes()
    records = json.loads(workspace["structure"].read_text(encoding="utf-8")) + [rec("B2.1.1.1.2")]
    workspace["structure"].write_text(json.dumps(records), encoding="utf-8")
    result = build.build(**workspace)  # no date: today for new edges, the old date for old ones
    G = schema.load_graph(workspace["out"])
    assert result.dates_kept == 3
    assert G.edges["math:B1.1.1.1.1", "math:B1.1.1.1.2"]["provenance"]["at"] == "2026-01-05"
    assert G.edges["math:B2.1.1.1.1", "math:B2.1.1.1.2"]["provenance"]["at"] == build.today()
    workspace["structure"].write_text(json.dumps(records[:-1]), encoding="utf-8")
    build.build(**workspace)
    assert workspace["out"].read_bytes() == first  # same input, same bytes, whatever the day


def test_build_refuses_to_put_full_text_in_the_committed_data_folder(workspace, monkeypatch):
    monkeypatch.setenv("KWAN_DATA_DIR", str(workspace["out"].parent.parent))
    with pytest.raises(ValueError, match="committed data directory"):
        build.build(**{**workspace, "full_out": workspace["out"].parent / "math.graph.full.json"}, at=DAY)


def test_build_without_a_structure_file_says_what_to_do(tmp_path):
    with pytest.raises(FileNotFoundError, match="no structure file"):
        build.build(structure=tmp_path / "nothing.json", out=tmp_path / "g.json")


def test_default_paths_follow_the_data_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("KWAN_DATA_DIR", str(tmp_path / "data"))
    paths = build.default_paths()
    assert paths["structure"] == tmp_path / "data" / "curriculum" / "math.structure.json"
    assert paths["out"] == tmp_path / "data" / "graph" / "math.graph.json"
    assert paths["augmented"] == tmp_path / "data" / "graph" / "math.augmented.json"
    assert paths["full_text"] == tmp_path / "build" / "curriculum" / "math.full.json"
    assert paths["full_out"] == tmp_path / "build" / "graph" / "math.graph.full.json"


def test_read_records_accepts_a_list_or_an_object_that_holds_one(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps([rec("B1.1.1.1.1")]))
    (tmp_path / "b.json").write_text(json.dumps({"schema_version": 1, "indicators": [rec("B1.1.1.1.1")]}))
    (tmp_path / "c.json").write_text(json.dumps({"hello": 1}))
    assert build.read_records(tmp_path / "a.json") == build.read_records(tmp_path / "b.json") == [rec("B1.1.1.1.1")]
    with pytest.raises(ValueError):
        build.read_records(tmp_path / "c.json")


def test_cli_graph_build_uses_the_standard_locations(workspace, monkeypatch, capsys):
    monkeypatch.setenv("KWAN_DATA_DIR", str(workspace["out"].parent.parent))
    assert cli.main(["graph", "build", "--date", DAY]) == 0
    out = capsys.readouterr().out
    assert "5 nodes, 4 edges (augmented 1, sequence 2, vertical 1)" in out
    assert "1 merged of 2 accepted (1 pending and 1 rejected left out)" in out and "would close a cycle" in out
    assert workspace["out"].exists() and workspace["full_out"].exists() and "—" not in out


def test_cli_graph_build_fails_cleanly_without_a_structure_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("KWAN_DATA_DIR", str(tmp_path))
    assert cli.main(["graph", "build"]) == 1
    assert "no structure file" in capsys.readouterr().err
