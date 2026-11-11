"""Phase 7 core: candidate pairs, the cache, cycle-safe insertion, the proposed edge file, impact and the CLI.

Everything here runs with the fake provider and a stand-in embedder. The graph is a tiny made-up
curriculum; no curriculum sentence and no dataset row is used.
"""

import json

import networkx as nx
import pytest

from kwan.graph import _cli_augment as cli
from kwan.graph import augment, candidates
from kwan.graph.providers import get_provider
from kwan.graph.providers.fake import FakeDecisionModel
from kwan.graph.schema import graph_from_dict, save_graph

RULES_YAML = """
math:
  rules:
    - {from: 1, to: 3, why: test}
"""


def _node(code, grade, strand, label):
    parts = code.split(".")
    return {
        "id": f"math:{code}",
        "subject": "math",
        "code": code,
        "grade": grade,
        "strand": strand,
        "strand_name": {1: "Number", 2: "Algebra", 3: "Geometry and Measurement"}[strand],
        "sub_strand": int(parts[2]),
        "content_standard": ".".join(parts[:4]),
        "cs_key": ".".join(parts[1:4]),
        "indicator_no": int(parts[4]),
        "short_label": label,
        "aliases": [code],
    }


def _edge(a, b, kind="sequence"):
    return {
        "source": f"math:{a}",
        "target": f"math:{b}",
        "type": kind,
        "weight": 1.0,
        "provenance": {"by": f"rule:{kind}", "at": "2026-11-10", "review": None},
    }


NODES = [
    _node("B1.1.1.1.1", 1, 1, "Count objects to 20"),
    _node("B1.1.1.1.2", 1, 1, "Count objects to 100"),
    _node("B1.1.2.1.1", 1, 1, "Add numbers to 20"),
    _node("B2.1.2.1.1", 2, 1, "Add numbers to 100"),
    _node("B3.1.2.1.1", 3, 1, "Add numbers to 1000"),
    _node("B4.1.2.1.1", 4, 1, "Add numbers to 10000"),
    _node("B2.2.1.1.1", 2, 2, "Extend number patterns by adding"),
    _node("B2.3.1.1.1", 2, 3, "Measure length by counting units"),
    _node("B1.3.1.1.1", 1, 3, "Compare lengths of objects"),
]
EDGES = [
    _edge("B1.1.1.1.1", "B1.1.1.1.2"),
    _edge("B1.1.2.1.1", "B2.1.2.1.1", "vertical"),
    _edge("B2.1.2.1.1", "B3.1.2.1.1", "vertical"),
    _edge("B3.1.2.1.1", "B4.1.2.1.1", "vertical"),
]


def make_graph():
    return graph_from_dict(
        {"schema_version": 2, "subject": "math", "text_mode": "labels", "nodes": NODES, "edges": EDGES}
    )


def M(code):
    return f"math:{code}"


@pytest.fixture
def G():
    return make_graph()


@pytest.fixture
def rules(tmp_path):
    path = tmp_path / "rules.yaml"
    path.write_text(RULES_YAML, encoding="utf-8")
    return candidates.load_cross_strand_rules(path)


def _candidates(G, rules, embedder=None, top_k=20):
    nodes = {n: {"grade": d["grade"], "strand": d["strand"]} for n, d in G.nodes(data=True)}
    texts = {n: d["short_label"] for n, d in G.nodes(data=True)}
    return candidates.candidate_pairs(nodes, texts, embedder or candidates.HashingEmbedder(), rules, G.edges(), top_k)


# ---------------------------------------------------------------- 7.1 candidates


def test_candidates_respect_grade_window_strand_rules_and_existing_edges(G, rules):
    cands = _candidates(G, rules)
    pairs = {(c.source, c.target) for c in cands}
    assert pairs, "the toy graph should give candidates"
    for a, b in pairs:
        ga, gb = G.nodes[a]["grade"], G.nodes[b]["grade"]
        assert ga <= gb <= ga + 2
        assert not G.has_edge(a, b) and not G.has_edge(b, a)
        sa, sb = G.nodes[a]["strand"], G.nodes[b]["strand"]
        assert sa == sb or (sa, sb) == (1, 3)
    assert (M("B1.1.1.1.1"), M("B2.3.1.1.1")) in pairs  # Number -> Measurement is listed
    assert (M("B1.1.1.1.1"), M("B2.2.1.1.1")) not in pairs  # Number -> Algebra is not listed in this test file
    assert (M("B1.3.1.1.1"), M("B2.1.2.1.1")) not in pairs  # Measurement -> Number is not listed
    assert (M("B1.1.1.1.1"), M("B4.1.2.1.1")) not in pairs  # three grades apart
    assert (M("B2.1.2.1.1"), M("B1.1.2.1.1")) not in pairs  # a higher grade never feeds a lower one


def test_candidates_keep_top_k_per_target_most_similar_first(G, rules):
    cands = _candidates(G, rules, top_k=2)
    by_target = {}
    for c in cands:
        by_target.setdefault(c.target, []).append(c)
    assert by_target and all(len(v) <= 2 for v in by_target.values())
    for group in by_target.values():
        assert [c.rank for c in group] == list(range(1, len(group) + 1))
        assert [c.similarity for c in group] == sorted((c.similarity for c in group), reverse=True)
    full = _candidates(G, rules, top_k=20)
    assert len(full) > len(cands)


def test_candidates_are_deterministic_and_carry_signals(G, rules):
    one, two = _candidates(G, rules), _candidates(G, rules)
    assert one == two
    names = [k for k, _ in one[0].signals]
    assert names == ["sim", "rank", "z", "margin", "z_rev", "central"]
    assert all(c.signals[0][1] == pytest.approx(c.similarity, abs=1e-6) for c in one)


def test_embedder_is_injected(G, rules):
    class Stub:
        name = "stub"

        def __init__(self):
            self.seen = []

        def encode(self, texts):
            self.seen.extend(texts)
            return [[1.0, float(len(t))] for t in texts]

    stub = Stub()
    cands = _candidates(G, rules, embedder=stub)
    assert len(stub.seen) == G.number_of_nodes()
    assert cands


def test_similarity_table_matches_plain_cosine():
    table = candidates.similarity_table([[1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    assert table[0][0] == pytest.approx(1.0)
    assert table[0][1] == pytest.approx(2**-0.5)
    assert table[2][0] == 0.0  # a zero vector is similar to nothing


def test_committed_rules_file_lists_the_prd_pairs():
    rules = candidates.load_cross_strand_rules()
    assert rules.allows(1, 1) and rules.allows(1, 3) and rules.allows(1, 4) and rules.allows(1, 2)
    assert not rules.allows(4, 1) and not rules.allows(3, 1)
    junyi = candidates.load_cross_strand_rules(section="junyi_2015")
    assert junyi.allows("arithmetic", "algebra") and not junyi.allows("calculus", "arithmetic")


def test_rule_without_from_or_to_is_an_error(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("math:\n  rules:\n    - {from: 1}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="from"):
        candidates.load_cross_strand_rules(path)


# ---------------------------------------------------------------- 7.2 question, protocol, cache


def _q(a="math:B1.1.1.1.1", b="math:B2.1.2.1.1", **kw):
    base = dict(a_id=a, a_text="Count things", a_grade=1, b_id=b, b_text="Add things", b_grade=2)
    return augment.PairQuestion(**{**base, **kw})


def test_prompt_holds_every_input_the_prd_lists():
    q = _q(a_label="Count", b_label="Add", a_exemplar="Count stones", b_exemplar="Add stones")
    text = augment.render_prompt(q)
    for piece in ("B1.1.1.1.1", "B2.1.2.1.1", "Basic 1", "Basic 2", "Count things", "Add things", "Count stones"):
        assert piece in text
    assert "Must a pupil be able to do A before they can learn B?" in text
    assert '{"probability"' in text
    assert "—" not in text


def test_cache_calls_the_provider_once_and_survives_a_restart(tmp_path):
    path = tmp_path / "augment" / "cache.jsonl"
    fake = FakeDecisionModel(default=0.7)
    model = augment.CachedDecisionModel(fake, path)
    assert model.prerequisite_probability(_q()) == 0.7
    assert model.prerequisite_probability(_q()) == 0.7
    assert fake.asked == [("math:B1.1.1.1.1", "math:B2.1.2.1.1")]
    assert (model.calls, model.hits) == (1, 1)

    fresh_fake = FakeDecisionModel(default=0.1)  # would answer differently if it were asked
    again = augment.CachedDecisionModel(fresh_fake, path)
    assert again.prerequisite_probability(_q()) == 0.7
    assert fresh_fake.asked == [] and again.spend()["calls"] == 0 and again.spend()["cache_hits"] == 1


def test_cache_line_has_provider_model_input_hash_and_output_but_no_prompt_text(tmp_path):
    path = tmp_path / "cache.jsonl"
    model = augment.CachedDecisionModel(FakeDecisionModel(default=0.25), path)
    model.prerequisite_probability(_q())
    (line,) = path.read_text(encoding="utf-8").splitlines()
    row = json.loads(line)
    assert row["provider"] == "fake" and row["model"] == "fixed-outputs" and row["probability"] == 0.25
    assert len(row["key"]) == 64 and len(row["input_hash"]) == 64
    assert "Count things" not in line


def test_cache_key_depends_on_provider_model_and_prompt(tmp_path):
    a = augment.CachedDecisionModel(FakeDecisionModel(), tmp_path / "c.jsonl")
    other = FakeDecisionModel()
    other.model = "another-model"
    b = augment.CachedDecisionModel(other, tmp_path / "c.jsonl")
    assert a.key(_q()) == a.key(_q())
    assert a.key(_q()) != b.key(_q())
    assert a.key(_q()) != a.key(_q(b_text="Subtract things"))


def test_cache_only_mode_never_calls_the_provider(tmp_path):
    fake = FakeDecisionModel(default=0.5)
    model = augment.CachedDecisionModel(fake, tmp_path / "c.jsonl", cache_only=True)
    assert not model.cached(_q())
    with pytest.raises(augment.CacheMiss):
        model.prerequisite_probability(_q())
    assert fake.asked == [] and not (tmp_path / "c.jsonl").exists()


def test_cache_refuses_a_number_that_is_not_a_probability(tmp_path):
    model = augment.CachedDecisionModel(FakeDecisionModel(default=1.5), tmp_path / "c.jsonl")
    with pytest.raises(ValueError, match="probability"):
        model.prerequisite_probability(_q())


def test_cache_skips_a_half_written_line(tmp_path):
    path = tmp_path / "c.jsonl"
    model = augment.CachedDecisionModel(FakeDecisionModel(default=0.4), path)
    model.prerequisite_probability(_q())
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"key": "abc", "proba')
    again = augment.CachedDecisionModel(FakeDecisionModel(default=0.9), path)
    assert again.prerequisite_probability(_q()) == 0.4


def test_spend_is_logged(tmp_path):
    class Paid(FakeDecisionModel):
        last_cost_usd = 0.002

    model = augment.CachedDecisionModel(Paid(default=0.5), tmp_path / "cache.jsonl")
    model.prerequisite_probability(_q())
    model.prerequisite_probability(_q(b_text="Other"))
    model.prerequisite_probability(_q())
    row = model.write_spend("test run")
    assert row["calls"] == 2 and row["cache_hits"] == 1 and row["cost_usd"] == pytest.approx(0.004)
    logged = [json.loads(x) for x in (tmp_path / "spend.jsonl").read_text(encoding="utf-8").splitlines()]
    assert logged[-1]["run"] == "test run" and logged[-1]["provider"] == "fake"


def test_fake_provider_is_fixed_and_registered():
    fake = get_provider("fake", outputs={("a", "b"): 0.9})
    assert fake.prerequisite_probability(_q("a", "b")) == 0.9
    first = fake.prerequisite_probability(_q("a", "c"))
    assert 0.0 <= first < 1.0 and first == FakeDecisionModel().prerequisite_probability(_q("a", "c"))
    with pytest.raises(ValueError, match="unknown provider"):
        get_provider("nope")


# ---------------------------------------------------------------- 7.5 cycle-safe insertion


def _chain(*names):
    H = nx.DiGraph()
    H.add_nodes_from(names)
    return H


def test_insertion_goes_from_the_highest_probability_down_and_never_closes_a_cycle():
    H = _chain("a", "b", "c")
    P = augment.Proposal
    report = augment.cycle_safe_insert(H, [P("c", "a", 0.7), P("a", "b", 0.9), P("b", "c", 0.8)])
    assert [(p.source, p.target) for p in report.inserted] == [("a", "b"), ("b", "c")]
    assert [(p.source, p.target, why) for p, why in report.skipped] == [("c", "a", "cycle")]
    assert nx.is_directed_acyclic_graph(H)
    assert H.edges["a", "b"]["type"] == "augmented" and H.edges["a", "b"]["weight"] == 0.9


def test_insertion_skips_an_edge_that_repeats_a_path_of_one_or_two_edges_but_not_three():
    H = nx.DiGraph([("a", "b"), ("b", "c"), ("c", "d")])
    P = augment.Proposal
    report = augment.cycle_safe_insert(H, [P("a", "b", 0.9), P("a", "c", 0.9), P("a", "d", 0.9)])
    assert {(p.source, p.target): why for p, why in report.skipped} == {
        ("a", "b"): "duplicate_path",
        ("a", "c"): "duplicate_path",
    }
    assert [(p.source, p.target) for p in report.inserted] == [("a", "d")]
    assert report.reasons() == {"duplicate_path": 2}


def test_insertion_reports_unknown_nodes_and_self_loops_and_two_way_pairs():
    H = _chain("a", "b")
    P = augment.Proposal
    report = augment.cycle_safe_insert(H, [P("a", "zz", 0.9), P("a", "a", 0.9), P("a", "b", 0.6), P("b", "a", 0.8)])
    assert {(p.source, p.target): why for p, why in report.skipped} == {
        ("a", "zz"): "unknown_node",
        ("a", "a"): "self_loop",
        ("a", "b"): "cycle",  # the reverse edge had the higher probability and went in first
    }
    assert list(H.edges()) == [("b", "a")]


def test_insertion_does_not_depend_on_input_order():
    P = augment.Proposal
    props = [P("a", "b", 0.5), P("b", "a", 0.5), P("b", "c", 0.5), P("c", "a", 0.5)]
    H1, H2 = _chain("a", "b", "c"), _chain("a", "b", "c")
    augment.cycle_safe_insert(H1, props)
    augment.cycle_safe_insert(H2, list(reversed(props)))
    assert sorted(H1.edges()) == sorted(H2.edges())


# ---------------------------------------------------------------- 7.4 proposing


def _questions(G, rules):
    texts = augment.node_texts(G)
    return augment.build_questions(G, _candidates(G, rules), texts)


def test_proposed_edges_are_all_pending_and_carry_model_provenance(G, rules):
    questions = _questions(G, rules)
    wanted = (M("B1.1.1.1.2"), M("B2.1.2.1.1"))
    assert wanted in {(q.a_id, q.b_id) for q in questions}
    fake = FakeDecisionModel(outputs={wanted: 0.93}, default=0.1)
    before = sorted(G.edges())
    run = augment.propose_edges(G, questions, fake, threshold=0.8, at="2026-11-10")
    assert sorted(G.edges()) == before  # the graph is not changed
    assert run.questions == len(questions) and run.above_threshold == 1
    assert run.edges == [
        {
            "source": wanted[0],
            "target": wanted[1],
            "type": "augmented",
            "weight": 0.93,
            "provenance": {
                "by": "model:fake/fixed-outputs",
                "at": "2026-11-10",
                "review": {"status": "pending", "by": None, "at": None},
            },
        }
    ]


def test_proposals_never_make_a_cycle_whatever_the_model_says(G, rules):
    questions = _questions(G, rules)
    run = augment.propose_edges(G, questions, FakeDecisionModel(default=0.99), threshold=0.5)
    H = G.copy()
    H.add_edges_from((e["source"], e["target"]) for e in run.edges)
    assert run.edges and nx.is_directed_acyclic_graph(H)
    assert all(e["provenance"]["review"]["status"] == "pending" for e in run.edges)
    assert run.above_threshold == len(run.edges) + len(run.report.skipped)


def test_node_texts_use_full_text_when_it_is_there_and_labels_otherwise(G, tmp_path):
    full = tmp_path / "math.full.json"
    full.write_text(
        json.dumps(
            [
                {
                    "id": "math:B1.1.1.1.1",
                    "description": "A made-up description",
                    "exemplars": [{"text": "A  made-up\nexemplar", "ghana_context": False}, {"text": "second"}],
                }
            ]
        ),
        encoding="utf-8",
    )
    texts = augment.node_texts(G, augment.load_texts(full))
    assert texts[M("B1.1.1.1.1")] == {
        "label": "Count objects to 20",
        "text": "A made-up description",
        "exemplar": "A made-up exemplar",
    }
    assert texts[M("B1.1.1.1.2")] == {"label": "", "text": "Count objects to 100", "exemplar": ""}
    assert augment.load_texts(tmp_path / "missing.json") == {}


# ---------------------------------------------------------------- the proposed edge file


def _doc(*edges):
    return {"schema_version": 1, "subject": "math", "run": {"provider": "fake"}, "edges": list(edges)}


def _aug(a, b, w, status="pending", by=None):
    return {
        "source": M(a),
        "target": M(b),
        "type": "augmented",
        "weight": w,
        "provenance": {
            "by": "model:fake/fixed-outputs",
            "at": "2026-11-10",
            "review": {"status": status, "by": by, "at": "2026-11-11" if by else None},
        },
    }


def test_saved_file_holds_ids_weights_and_provenance_only(tmp_path):
    edge = {**_aug("B1.1.1.1.2", "B2.1.2.1.1", 0.91234), "a_description": "text that must not be committed"}
    path = augment.save_augmented(_doc(_aug("B1.1.1.1.1", "B1.1.2.1.1", 0.5), edge), tmp_path / "math.augmented.json")
    raw = path.read_text(encoding="utf-8")
    assert "must not be committed" not in raw
    doc = augment.load_augmented(path)
    assert [e["weight"] for e in doc["edges"]] == [0.9123, 0.5]  # highest probability first
    assert all(set(e) == {"source", "target", "type", "weight", "provenance"} for e in doc["edges"])
    assert augment.review_counts(doc) == {"pending": 2, "accepted": 0, "rejected": 0}
    assert raw.count("\n") == len(doc["edges"]) + 7  # one edge per line


def test_missing_file_is_an_empty_document(tmp_path):
    doc = augment.load_augmented(tmp_path / "none.json")
    assert doc["edges"] == [] and augment.accepted_edges(doc) == []


def test_a_rerun_keeps_decisions_already_made():
    old = _doc(
        _aug("B1.1.1.1.2", "B2.1.2.1.1", 0.9, "accepted", "t-01"),
        _aug("B1.1.1.1.1", "B1.1.2.1.1", 0.8, "rejected", "t-01"),
        _aug("B1.3.1.1.1", "B2.3.1.1.1", 0.7),
    )
    new = [
        _aug("B1.1.1.1.2", "B2.1.2.1.1", 0.95),
        _aug("B1.1.1.1.1", "B1.1.2.1.1", 0.85),
        _aug("B1.1.2.1.1", "B2.2.1.1.1", 0.6),
    ]
    merged = {(e["source"], e["target"]): e for e in augment.keep_reviews(new, old)}
    assert merged[(M("B1.1.1.1.2"), M("B2.1.2.1.1"))]["provenance"]["review"]["status"] == "accepted"
    assert merged[(M("B1.1.1.1.2"), M("B2.1.2.1.1"))]["weight"] == 0.9
    assert merged[(M("B1.1.1.1.1"), M("B1.1.2.1.1"))]["provenance"]["review"]["status"] == "rejected"
    assert (M("B1.3.1.1.1"), M("B2.3.1.1.1")) not in merged  # an undecided old proposal the new run dropped
    assert merged[(M("B1.1.2.1.1"), M("B2.2.1.1.1"))]["provenance"]["review"]["status"] == "pending"


def test_merge_accepted_adds_only_accepted_edges_and_refuses_cycles(G):
    doc = _doc(
        _aug("B1.1.1.1.2", "B2.1.2.1.1", 0.9, "accepted", "t-01"),
        _aug("B1.3.1.1.1", "B2.3.1.1.1", 0.8),  # pending
        _aug("B1.1.1.1.1", "B1.1.2.1.1", 0.7, "rejected", "t-01"),
        _aug("B4.1.2.1.1", "B1.1.2.1.1", 0.6, "accepted", "t-01"),  # would close a cycle
        {**_aug("B1.1.1.1.1", "B2.2.1.1.1", 0.5, "accepted", "t-02"), "source": "B1.1.1.1.1"},  # bare code
    )
    H, report = augment.merge_accepted(G, doc)
    assert G.number_of_edges() == len(EDGES)
    added = sorted((a, b) for a, b, d in H.edges(data=True) if d["type"] == "augmented")
    assert added == [(M("B1.1.1.1.1"), M("B2.2.1.1.1")), (M("B1.1.1.1.2"), M("B2.1.2.1.1"))]
    assert report.reasons() == {"cycle": 1}
    assert H.edges[M("B1.1.1.1.2"), M("B2.1.2.1.1")]["provenance"]["review"]["by"] == "t-01"
    assert nx.is_directed_acyclic_graph(H)


# ---------------------------------------------------------------- 7.7 impact


def test_impact_with_no_accepted_edge_says_so_and_runs_nothing(G):
    calls = []
    out = augment.impact(G, edges=[], simulate=lambda graph: calls.append(graph))
    assert out == {"status": "no accepted edges yet", "accepted_edges": 0, "merged_edges": 0}
    assert calls == []


def test_impact_compares_the_simulator_with_and_without_a_synthetic_accepted_edge(G):
    seen = []

    def simulate(graph):
        seen.append(graph.number_of_edges())
        return {"beginner": {"kwan": (10.0 + graph.number_of_edges(), 1.0), "random": (5.0, 1.0)}}

    out = augment.impact(G, edges=[_aug("B1.1.1.1.2", "B2.1.2.1.1", 0.9)], simulate=simulate)
    assert seen == [len(EDGES), len(EDGES) + 1]
    assert out["status"] == "measured" and out["accepted_edges"] == 1 and out["merged_edges"] == 1
    assert out["without_accepted_edges"]["beginner"] == {"kwan": 14.0, "random": 5.0}
    assert out["difference"]["beginner"] == {"kwan": 1.0, "random": 0.0}


def test_impact_runs_the_real_simulator_on_a_synthetic_accepted_edge(G):
    out = augment.impact(G, edges=[_aug("B1.1.1.1.2", "B2.1.2.1.1", 0.9)], n_learners=2, steps=12)
    assert out["status"] == "measured" and out["merged_edges"] == 1
    assert set(out["without_accepted_edges"]) == set(out["with_accepted_edges"]) == set(out["difference"])
    for by_policy in out["with_accepted_edges"].values():
        assert "kwan" in by_policy


def test_impact_reports_an_accepted_edge_that_cannot_be_added(G):
    out = augment.impact(G, edges=[_aug("B4.1.2.1.1", "B1.1.2.1.1", 0.9)], simulate=lambda graph: {})
    assert out["status"] == "no accepted edge could be added" and out["skipped"] == {"cycle": 1}


# ---------------------------------------------------------------- CLI


@pytest.fixture
def files(tmp_path, G):
    graph = tmp_path / "math.graph.json"
    save_graph(G, graph)
    rules = tmp_path / "rules.yaml"
    rules.write_text(RULES_YAML, encoding="utf-8")
    return {"graph": str(graph), "rules": str(rules), "out": str(tmp_path / "math.augmented.json"), "dir": tmp_path}


def _common(files):
    full = str(files["dir"] / "no-full-text.json")
    return ["--graph", files["graph"], "--rules", files["rules"], "--out", files["out"], "--full-text", full]


def test_dry_run_walks_the_pipeline_and_writes_nothing(files, capsys):
    before = sorted(p.name for p in files["dir"].iterdir())
    code = cli.main(["augment", "--dry-run", *_common(files), "--cache", str(files["dir"] / "cache.jsonl")])
    out = capsys.readouterr().out
    assert code == 0
    assert "nothing is written" in out and "candidate pairs:" in out and "no cycles" in out
    assert sorted(p.name for p in files["dir"].iterdir()) == before


def test_dry_run_reports_a_real_provider_that_is_not_set_up(files, capsys, monkeypatch):
    monkeypatch.delenv("KWAN_LLM_ENDPOINT", raising=False)
    monkeypatch.delenv("KWAN_LLM_MODEL", raising=False)
    code = cli.main(["augment", "--dry-run", "--provider", "llm-json", *_common(files)])
    out = capsys.readouterr().out
    assert code == 0 and "not set up" in out and "fake provider" in out


def test_dry_run_fails_on_a_proposed_edge_file_with_a_cycle_or_an_unknown_id(files, capsys):
    bad = _doc(_aug("B4.1.2.1.1", "B1.1.2.1.1", 0.9, "accepted", "t-1"), _aug("B9.9.9.9.9", "B1.1.2.1.1", 0.8))
    augment.save_augmented(bad, files["out"])
    code = cli.main(["augment", "--dry-run", *_common(files)])
    err = capsys.readouterr().err
    assert code == 1 and "closes a cycle" in err and "unknown indicator" in err


def test_check_augmented_flags_text_keys_and_reviews_without_a_reviewer(G):
    edge = {**_aug("B1.1.1.1.2", "B2.1.2.1.1", 0.9, "accepted"), "description": "x"}
    problems = cli.check_augmented(G, _doc(edge))
    assert any("unexpected keys" in p for p in problems) and any("without a reviewer" in p for p in problems)
    assert cli.check_augmented(G, _doc(_aug("B1.1.1.1.2", "B2.1.2.1.1", 0.9))) == []


def test_fake_provider_cannot_write_real_proposals(files, capsys):
    code = cli.main(["augment", "--provider", "fake", *_common(files)])
    assert code == 2 and "fake provider" in capsys.readouterr().err
    assert not (files["dir"] / "math.augmented.json").exists()


def test_results_block_reports_review_as_not_yet_measured(G, monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "_junyi_file", lambda: tmp_path / "no_junyi.json")
    monkeypatch.setattr(cli, "_run_file", lambda: tmp_path / "no_run.json")
    doc = _doc(_aug("B1.1.1.1.2", "B2.1.2.1.1", 0.9), _aug("B1.1.1.1.1", "B2.3.1.1.1", 0.85))
    block = cli.results_block(G, doc)
    assert block["review"]["pending"] == 2 and block["review"]["accepted"] == 0
    assert block["review"]["teacher_acceptance"] is None and "not yet measured" in block["review"]["status"]
    assert block["impact_in_simulator"]["status"] == "no accepted edges yet"
    assert block["nacca"]["proposed_edges"] == 2 and block["nacca"]["acyclic"] is True
    assert block["nacca"]["proposed_edges_by_strand_pair"] == {
        "Number -> Geometry and Measurement": 1,
        "Number -> Number": 1,
    }
    assert block["junyi_benchmark"] is None and block["meets_precision_bar"] is False


def test_register_into_adds_the_three_commands():
    import argparse

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()
    cli.register_into(sub)
    assert {"augment", "review-export", "review-import"} <= set(sub.choices)
    args = parser.parse_args(["augment", "--dry-run"])
    assert args.dry_run and callable(args.func)


def test_committed_proposed_edge_file_is_valid_and_has_no_curriculum_text():
    from kwan.graph import schema

    path = augment.default_augmented_path()
    if not path.is_file() or not schema.default_graph_path().is_file():
        pytest.skip("the committed graph or proposed edge file is not there yet")
    doc = augment.load_augmented(path)
    G = schema.load_graph()
    assert cli.check_augmented(G, doc) == []
    for e in doc["edges"]:
        assert set(e) == {"source", "target", "type", "weight", "provenance"}
        review = e["provenance"]["review"]
        assert review["status"] == "pending" or review["by"], "a decision needs a reviewer id"
