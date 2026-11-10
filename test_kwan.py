import json, networkx as nx, pytest, graph, engine, simulate

@pytest.fixture(scope="module")
def G(): return graph.load("data/graph.json")

def test_indicators_parsed():
    recs = json.load(open("data/indicators.json"))
    assert len(recs) >= 250
    grades = {r["grade"] for r in recs}; assert grades == {1, 2, 3, 4, 5, 6}
    assert all(r["description"] for r in recs)

def test_graph_is_dag(G):
    assert nx.is_directed_acyclic_graph(G); assert G.number_of_edges() >= 250

def test_learner_updates_move_mastery(G):
    L = engine.EloLearner(G=G); s = engine.LearnerState("t"); n = "B1.1.1.1.1"
    m0 = L.mastery(s, n); L.update(s, n, True); assert L.mastery(s, n) > m0
    L.update(s, n, False); L.update(s, n, False); assert L.mastery(s, n) < 0.95

def test_streak_mastery(G):
    L = engine.EloLearner(G=G); s = engine.LearnerState("t"); n = "B1.1.1.1.1"
    for _ in range(3): L.update(s, n, True)
    assert L.mastery(s, n) >= 0.95

def test_recommend_returns_ready_nodes_with_rationale(G):
    L = engine.EloLearner(G=G); R = engine.Recommender(G, L); s = engine.LearnerState("t")
    out = R.recommend(s, k=3)
    assert 1 <= len(out["recommendations"]) <= 3
    for r in out["recommendations"]:
        assert r["rationale"] and r["indicator"] in G
        assert r["ghana_context"]
        assert all(L.mastery(s, p) >= engine.READY for p in G.predecessors(r["indicator"]))

def test_latency_under_5ms(G):
    L = engine.EloLearner(G=G); R = engine.Recommender(G, L); s = engine.LearnerState("t")
    assert R.recommend(s)["latency_ms"] < 5

def test_kwan_beats_baselines_on_gapped_learners(G):
    res = simulate.compare(G, n_learners=20, profile="gapped_p4")
    assert res["kwan"][0] > res["syllabus"][0] and res["kwan"][0] > res["random"][0]

def test_api_roundtrip():
    from fastapi.testclient import TestClient
    import api
    c = TestClient(api.app)
    assert c.get("/health").json()["ok"]
    r = c.post("/attempt", json={"learner_id": "t", "indicator": "B1.1.1.1.1", "correct": True})
    assert r.status_code == 200 and 0 <= r.json()["new_mastery"] <= 1
    rec = c.get("/recommend/t?k=2").json()
    assert len(rec["recommendations"]) == 2
    assert c.get("/recommend/t?goal=NOPE").status_code == 404
