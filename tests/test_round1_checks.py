"""The eight Round 1 checks and the API round trip (the former root `test_kwan.py`), now against the package.

They run on the committed v2 graph and on the frozen v1 graph: the product still passes its first tests.
"""

import networkx as nx
import pytest

from kwan import ids
from kwan.eval import simulate
from kwan.learner.elo import READY, EloLearner, LearnerState
from kwan.recommend.recommender import Recommender

FIRST = "B1.1.1.1.1"  # a starting skill with no prerequisites


@pytest.fixture(scope="module", params=["v2", "v1"])
def graph(request, G, G_v1):
    return G if request.param == "v2" else G_v1


@pytest.fixture(scope="module")
def first(graph):
    return ids.Resolver(graph).resolve(FIRST)


def test_indicators_parsed(graph):
    assert graph.number_of_nodes() >= 250
    assert {d["grade"] for _, d in graph.nodes(data=True)} == {1, 2, 3, 4, 5, 6}
    assert all(d["description"] for _, d in graph.nodes(data=True))


def test_graph_is_dag(graph):
    assert nx.is_directed_acyclic_graph(graph)
    assert graph.number_of_edges() >= 250


def test_learner_updates_move_mastery(graph, first):
    L = EloLearner(G=graph)
    s = LearnerState("t")
    m0 = L.mastery(s, first)
    L.update(s, first, True)
    assert L.mastery(s, first) > m0
    L.update(s, first, False)
    L.update(s, first, False)
    assert L.mastery(s, first) < 0.95


def test_streak_mastery(graph, first):
    L = EloLearner(G=graph)
    s = LearnerState("t")
    for _ in range(3):
        L.update(s, first, True)
    assert L.mastery(s, first) >= 0.95


def test_recommend_returns_ready_nodes_with_rationale(graph):
    L = EloLearner(G=graph)
    R = Recommender(graph, L)
    s = LearnerState("t")
    out = R.recommend(s, k=3)
    assert 1 <= len(out["recommendations"]) <= 3
    for r in out["recommendations"]:
        assert r["rationale"] and r["indicator"] in graph
        assert r["ghana_context"]
        assert all(L.mastery(s, p) >= READY for p in graph.predecessors(r["indicator"]))


def test_latency_under_5ms(graph):
    L = EloLearner(G=graph)
    R = Recommender(graph, L)
    s = LearnerState("t")
    assert R.recommend(s)["latency_ms"] < 5


def test_kwan_beats_baselines_on_gapped_learners(graph):
    res = simulate.compare(graph, n_learners=20, profile="gapped_p4")
    assert res["kwan"][0] > res["syllabus"][0] and res["kwan"][0] > res["random"][0]


def test_api_roundtrip():
    from fastapi.testclient import TestClient

    from kwan.api import app as api

    api.STATES.clear()
    c = TestClient(api.app)
    assert c.get("/health").json()["ok"]
    r = c.post("/attempt", json={"learner_id": "t", "indicator": FIRST, "correct": True})
    assert r.status_code == 200 and 0 <= r.json()["new_mastery"] <= 1
    rec = c.get("/recommend/t?k=2").json()
    assert len(rec["recommendations"]) == 2
    assert c.get("/recommend/t?goal=NOPE").status_code == 404
    api.STATES.clear()
