"""REST API tests (PRD Section 5.4, Phase 7). Each test starts from an empty learner store."""

import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import networkx as nx
import pytest
from fastapi.testclient import TestClient

from kwan.api import app as api
from kwan.learner.elo import MASTERY, READY, LearnerState

ROOT_SKILL = "B1.1.1.1.1"  # a starting skill with no prerequisites
GOAL = "B4.1.1.3.3"  # the goal used in the PRD examples
EM_DASH = chr(0x2014)  # banned from user-facing text
REC_KEYS = {"indicator", "score", "mastery", "grade", "strand", "description", "rationale", "ghana_context"}


@pytest.fixture(scope="module")
def G():
    """The graph the v0 routes serve, seen with bare codes as the v0 routes send and return them."""
    return nx.relabel_nodes(api.G, {n: api._bare(n) for n in api.G}, copy=True)


@pytest.fixture()
def client():
    api.STATES.clear()
    with TestClient(api.app) as c:
        yield c
    api.STATES.clear()


def post(client, learner_id="a", indicator=ROOT_SKILL, correct=True, **extra):
    return client.post("/attempt", json={"learner_id": learner_id, "indicator": indicator, "correct": correct, **extra})


def recs(client, learner_id="a", **params):
    r = client.get(f"/recommend/{learner_id}", params=params)
    assert r.status_code == 200, r.text
    return r.json()["recommendations"]


def on_path(G, goal):
    return nx.ancestors(G, goal) | {goal}


# --------------------------------------------------------------------------- module surface


def test_module_exposes_app_graph_and_store():
    assert api.app.title == "Kwan Learning Path Engine"
    assert api.G.number_of_nodes() > 0
    assert isinstance(api.STATES, dict)


# --------------------------------------------------------------------------- /health


def test_health_counts_match_graph(client, G):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True, "nodes": G.number_of_nodes(), "edges": G.number_of_edges()}
    assert r.json() == {"ok": True, "nodes": api.G.number_of_nodes(), "edges": api.G.number_of_edges()}


# --------------------------------------------------------------------------- /attempt


def test_attempt_first_correct_gives_contract_numbers(client):
    r = post(client)
    assert r.status_code == 200
    assert r.json() == {"predicted_p_correct": 0.3775, "new_mastery": 0.4839}


def test_attempt_values_are_probabilities_rounded_to_4_decimals(client):
    for i in range(12):
        body = post(client, correct=i % 3 != 0).json()
        for key in ("predicted_p_correct", "new_mastery"):
            assert 0 <= body[key] <= 1
            assert body[key] == round(body[key], 4)


def test_correct_answer_raises_mastery_and_wrong_answer_lowers_it(client):
    before = client.get("/mastery/a").json()[ROOT_SKILL]
    up = post(client, correct=True).json()
    assert up["new_mastery"] > up["predicted_p_correct"]
    assert up["new_mastery"] > before
    down = post(client, correct=False).json()
    assert down["new_mastery"] < up["new_mastery"]
    assert down["new_mastery"] < down["predicted_p_correct"]


def test_wrong_first_answer_lowers_mastery_below_the_starting_estimate(client):
    before = client.get("/mastery/a").json()[ROOT_SKILL]
    body = post(client, correct=False).json()
    assert body["new_mastery"] < before
    assert client.get("/mastery/a").json()[ROOT_SKILL] < before


def test_item_id_is_optional_and_accepted(client):
    omitted = post(client, learner_id="no-item")
    null = post(client, learner_id="null-item", item_id=None)
    given = post(client, learner_id="with-item", item_id="item-42")
    assert omitted.status_code == null.status_code == given.status_code == 200
    # the API loads no item difficulties, so an item id does not change the estimate
    assert omitted.json() == null.json() == given.json() == {"predicted_p_correct": 0.3775, "new_mastery": 0.4839}


def test_unknown_indicator_is_404_with_the_id_in_the_message(client):
    r = post(client, indicator="B9.9.9.9.9")
    assert r.status_code == 404
    assert "B9.9.9.9.9" in r.json()["detail"]
    assert api.STATES == {}  # a rejected attempt stores nothing


@pytest.mark.parametrize(
    "body",
    [
        {"indicator": ROOT_SKILL, "correct": True},  # missing learner_id
        {"learner_id": "a", "correct": True},  # missing indicator
        {"learner_id": "a", "indicator": ROOT_SKILL},  # missing correct
        {"learner_id": "a", "indicator": ROOT_SKILL, "correct": "yes"},  # correct is a string
        {"learner_id": "a", "indicator": ROOT_SKILL, "correct": "true"},
        {"learner_id": "a", "indicator": ROOT_SKILL, "correct": 1},  # correct is a number
        {"learner_id": "a", "indicator": ROOT_SKILL, "correct": None},
        {"learner_id": "", "indicator": ROOT_SKILL, "correct": True},  # empty learner id
        {"learner_id": "   ", "indicator": ROOT_SKILL, "correct": True},  # blank learner id
        {"learner_id": "x" * 129, "indicator": ROOT_SKILL, "correct": True},  # learner id too long
        {"learner_id": 7, "indicator": ROOT_SKILL, "correct": True},  # learner id is a number
        {"learner_id": "a", "indicator": "", "correct": True},  # empty indicator
        {"learner_id": "a", "indicator": 5, "correct": True},
        {"learner_id": "a", "indicator": ROOT_SKILL, "correct": True, "item_id": 9},  # item id is a number
        [],  # not an object
    ],
)
def test_malformed_attempt_bodies_are_422(client, body):
    r = client.post("/attempt", json=body)
    assert r.status_code == 422, r.text
    assert api.STATES == {}


def test_attempt_body_that_is_not_json_is_422(client):
    r = client.post("/attempt", content=b"not json", headers={"content-type": "application/json"})
    assert r.status_code == 422
    assert client.post("/attempt").status_code == 422


def test_learner_id_at_the_length_limit_is_accepted(client):
    assert post(client, learner_id="x" * 128).status_code == 200


# --------------------------------------------------------------------------- /recommend


def test_recommend_default_k_is_3(client):
    r = client.get("/recommend/a")
    assert r.status_code == 200
    assert set(r.json()) == {"recommendations", "latency_ms"}
    assert len(r.json()["recommendations"]) == 3


@pytest.mark.parametrize("k", [1, 2, 5])
def test_recommend_honours_k(client, k):
    assert len(recs(client, k=k)) == k


def test_recommend_k_is_an_upper_bound_and_results_are_distinct_and_sorted(client, G):
    out = recs(client, k=50)
    ids = [r["indicator"] for r in out]
    assert 1 <= len(out) <= 50
    assert len(set(ids)) == len(ids) and all(i in G for i in ids)
    scores = [r["score"] for r in out]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.parametrize("k", ["0", "-1", "51", "1000", "abc", "1.5"])
def test_recommend_k_out_of_range_is_422(client, k):
    assert client.get(f"/recommend/a?k={k}").status_code == 422


def test_recommend_k_bounds_are_accepted(client):
    assert client.get("/recommend/a?k=1").status_code == 200
    assert client.get("/recommend/a?k=50").status_code == 200


def test_every_recommendation_has_the_prd_keys_and_a_reason(client, G):
    post(client)  # a learner with some history
    for r in recs(client, k=50) + recs(client, "fresh", k=50):
        assert set(r) == REC_KEYS
        assert r["indicator"] in G
        assert isinstance(r["score"], float)
        assert 0 <= r["mastery"] <= 1
        assert r["grade"] == G.nodes[r["indicator"]]["grade"]
        assert r["strand"] == G.nodes[r["indicator"]]["strand_name"]
        assert r["description"].strip()
        assert r["rationale"].strip()
        assert r["ghana_context"].strip()


def test_recommend_latency_is_reported_and_under_5ms(client):
    samples = [client.get("/recommend/a").json()["latency_ms"] for _ in range(21)]
    assert all(isinstance(s, float) and s >= 0 for s in samples)
    assert statistics.median(samples) < 5  # NFR1; the median keeps a busy CI box from flaking


def test_recommend_goal_is_honoured(client, G):
    path = on_path(G, GOAL)
    with_goal = recs(client, goal=GOAL)
    without = recs(client)
    assert with_goal, "a new pupil always has starting skills"
    assert with_goal[0]["indicator"] in path  # the best next step leads to the goal
    hits = lambda out: sum(r["indicator"] in path for r in out)
    assert hits(with_goal) >= hits(without)  # a goal never pushes on-path skills out


def test_recommend_goal_never_demotes_the_path_to_it(client, G):
    """PRD 5.3 scores goal distance as one signal among five, so check ranks, for every possible goal."""
    base = [r["indicator"] for r in recs(client, k=50)]
    checked = 0
    for goal in G:
        path = on_path(G, goal)
        if not any(i in path for i in base):
            continue  # nothing on the path is ready yet
        ranked = [r["indicator"] for r in recs(client, k=50, goal=goal)]
        assert set(ranked) <= set(base)  # a goal never surfaces a skill that is not ready
        best = lambda ids: min(i for i, n in enumerate(ids) if n in path)
        assert best(ranked) <= best(base)  # the path to the goal never ranks lower
        checked += 1
    assert checked == G.number_of_nodes()  # every skill is reachable from a starting skill


def test_recommend_unknown_goal_is_404_with_the_id_in_the_message(client):
    r = client.get("/recommend/a?goal=NOPE")
    assert r.status_code == 404
    assert "NOPE" in r.json()["detail"]


def test_recommend_empty_goal_means_no_goal(client):
    assert client.get("/recommend/a?goal=").status_code == 200
    assert recs(client, goal="") == recs(client)


@pytest.mark.parametrize("grade_max", [1, 2, 4, 6])
def test_recommend_grade_max_is_honoured(client, grade_max):
    out = recs(client, k=50, grade_max=grade_max)
    assert out
    assert all(r["grade"] <= grade_max for r in out)


def test_recommend_grade_max_actually_removes_higher_classes(client):
    uncapped = recs(client, k=50)
    assert any(r["grade"] > 1 for r in uncapped), "a new pupil has starting skills above B1"
    capped = recs(client, k=50, grade_max=1)
    assert {r["indicator"] for r in capped} == {r["indicator"] for r in uncapped if r["grade"] <= 1}


@pytest.mark.parametrize("grade_max", ["0", "7", "-1", "abc"])
def test_recommend_grade_max_out_of_range_is_422(client, grade_max):
    assert client.get(f"/recommend/a?grade_max={grade_max}").status_code == 422


def test_recommend_goal_and_grade_max_together(client, G):
    out = recs(client, k=50, goal=GOAL, grade_max=2)
    assert out and all(r["grade"] <= 2 for r in out)
    assert out[0]["indicator"] in on_path(G, GOAL)


def test_recommend_goal_mode_defaults_to_strict_and_soft_keeps_the_v1_ordering(client, G):
    for _ in range(3):
        post(client, indicator="B1.1.1.1.1")
        post(client, indicator="B1.1.1.1.2")
    ids = lambda **params: [r["indicator"] for r in recs(client, k=3, goal=GOAL, **params)]
    assert ids() == ids(goal_mode="strict")  # strict is the default; the path-first order is checked below
    with api.LOCK:  # soft: the v1 ordering, exactly what the engine gives with goal_mode="soft"
        direct = api.R.recommend(api.STATES["a"], k=3, goal=api.RESOLVE.resolve(GOAL), goal_mode="soft")
    assert ids(goal_mode="soft") == [api._bare(r["indicator"]) for r in direct["recommendations"]]
    assert ids(goal_mode="soft") != ids()  # soft lets a skill off the path in
    path = on_path(G, GOAL)
    full = [r["indicator"] in path for r in recs(client, k=50, goal=GOAL)]
    assert full == sorted(full, reverse=True) and True in full and False in full
    # without a goal the mode changes nothing
    assert recs(client, k=5, goal_mode="soft") == recs(client, k=5, goal_mode="strict") == recs(client, k=5)


@pytest.mark.parametrize("mode", ["STRICT", "hard", ""])
def test_recommend_unknown_goal_mode_is_422(client, mode):
    assert client.get("/recommend/a", params={"goal": GOAL, "goal_mode": mode}).status_code == 422


def test_recommend_goal_mode_is_documented_in_the_openapi_spec(client):
    params = client.get("/openapi.json").json()["paths"]["/recommend/{learner_id}"]["get"]["parameters"]
    mode = next(p for p in params if p["name"] == "goal_mode")
    assert mode["in"] == "query" and mode["required"] is False
    assert mode["schema"]["enum"] == ["strict", "soft"] and mode["schema"]["default"] == "strict"


@pytest.mark.parametrize("path", ["/recommend/%20", "/mastery/%20", "/recommend/" + "x" * 129, "/mastery/" + "x" * 129])
def test_blank_or_oversized_learner_id_in_the_path_is_422(client, path):
    assert client.get(path).status_code == 422


# --------------------------------------------------------------------------- /mastery


def test_mastery_returns_every_indicator_in_range(client, G):
    r = client.get("/mastery/a")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == set(G.nodes) and len(body) == G.number_of_nodes()
    assert all(isinstance(v, float) and 0 <= v <= 1 for v in body.values())


def test_mastery_reflects_attempts(client):
    before = client.get("/mastery/a").json()
    new = post(client).json()["new_mastery"]
    after = client.get("/mastery/a").json()
    assert after[ROOT_SKILL] == round(new, 3)
    assert after[ROOT_SKILL] > before[ROOT_SKILL]
    assert after != before


# --------------------------------------------------------------------------- learners never seen before


def test_reads_for_an_unseen_learner_give_the_new_pupil_view_and_store_nothing(client):
    first = client.get("/recommend/ghost?k=5")
    assert first.status_code == 200 and len(first.json()["recommendations"]) == 5
    m = client.get("/mastery/ghost")
    assert m.status_code == 200
    assert api.STATES == {}  # reads are safe: no state is created
    assert client.delete("/learner/ghost").status_code == 404
    # the view is the same for any unseen learner and is stable across calls
    assert client.get("/mastery/other").json() == m.json() == client.get("/mastery/ghost").json()
    assert [r["indicator"] for r in recs(client, "other", k=5)] == [
        r["indicator"] for r in first.json()["recommendations"]
    ]


def test_reads_do_not_change_a_stored_learner(client):
    post(client)
    snapshot = client.get("/mastery/a").json()
    for _ in range(3):
        recs(client, k=10, goal=GOAL, grade_max=3)
        client.get("/mastery/a")
    assert client.get("/mastery/a").json() == snapshot
    assert list(api.STATES) == ["a"]


def test_learners_are_independent(client):
    fresh = client.get("/mastery/b").json()
    for _ in range(3):
        post(client, learner_id="a")
    assert client.get("/mastery/b").json() == fresh
    assert client.get("/mastery/a").json() != fresh
    # b's first attempt is unaffected by a's history
    assert post(client, learner_id="b").json() == {"predicted_p_correct": 0.3775, "new_mastery": 0.4839}
    assert set(api.STATES) == {"a", "b"}


# --------------------------------------------------------------------------- DELETE /learner


def test_delete_resets_a_learner(client):
    fresh = client.get("/mastery/a").json()
    post(client)
    post(client)
    assert client.get("/mastery/a").json() != fresh
    r = client.delete("/learner/a")
    assert r.status_code == 204 and r.content == b""
    assert "a" not in api.STATES
    assert client.get("/mastery/a").json() == fresh
    assert post(client).json() == {"predicted_p_correct": 0.3775, "new_mastery": 0.4839}


def test_delete_unknown_learner_is_404(client):
    r = client.delete("/learner/nobody")
    assert r.status_code == 404 and "nobody" in r.json()["detail"]
    post(client)
    assert client.delete("/learner/a").status_code == 204
    assert client.delete("/learner/a").status_code == 404  # already gone


def test_delete_leaves_other_learners_alone(client):
    post(client, learner_id="a")
    post(client, learner_id="b")
    kept = client.get("/mastery/b").json()
    assert client.delete("/learner/a").status_code == 204
    assert client.get("/mastery/b").json() == kept
    assert list(api.STATES) == ["b"]


# --------------------------------------------------------------------------- OpenAPI


def test_openapi_lists_the_contract_paths_with_typed_responses(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    spec = r.json()
    paths = spec["paths"]
    assert "get" in paths["/health"]
    assert "post" in paths["/attempt"]
    assert "get" in paths["/recommend/{learner_id}"]
    assert "get" in paths["/mastery/{learner_id}"]
    assert "delete" in paths["/learner/{learner_id}"]
    assert {  # the v0 routes stay for one minor release; the v1 routes live next to them
        "/health",
        "/attempt",
        "/recommend/{learner_id}",
        "/mastery/{learner_id}",
        "/learner/{learner_id}",
    } <= set(paths)
    schemas = spec["components"]["schemas"]
    ok = lambda path, method: paths[path][method]["responses"]["200"]["content"]["application/json"]["schema"]
    assert ok("/health", "get")["$ref"].endswith("/HealthResponse")
    assert ok("/attempt", "post")["$ref"].endswith("/AttemptResponse")
    assert ok("/recommend/{learner_id}", "get")["$ref"].endswith("/RecommendResponse")
    assert ok("/mastery/{learner_id}", "get")["additionalProperties"]["type"] == "number"
    assert set(schemas["Recommendation"]["properties"]) == REC_KEYS
    assert set(schemas["Attempt"]["required"]) == {"learner_id", "indicator", "correct"}
    assert "404" in paths["/attempt"]["post"]["responses"]
    assert "404" in paths["/recommend/{learner_id}"]["get"]["responses"]
    params = {p["name"]: p["schema"] for p in paths["/recommend/{learner_id}"]["get"]["parameters"]}
    assert (params["k"]["minimum"], params["k"]["maximum"], params["k"]["default"]) == (1, 50, 3)


def test_docs_page_is_served_and_api_text_has_no_em_dash(client):
    assert client.get("/docs").status_code == 200
    assert EM_DASH not in client.get("/openapi.json").text
    assert EM_DASH not in post(client, indicator="NOPE").text
    assert EM_DASH not in client.get("/recommend/a?goal=NOPE").text


# --------------------------------------------------------------------------- concurrency


def test_concurrent_attempts_for_one_learner_are_all_counted(client):
    n_requests = 300
    with ThreadPoolExecutor(max_workers=24) as pool:
        codes = list(pool.map(lambda i: post(client, learner_id="busy").status_code, range(n_requests)))
    assert codes == [200] * n_requests
    st = api.STATES["busy"]
    root = api.RESOLVE.resolve(ROOT_SKILL)  # state is keyed by the node key (the canonical id on the v2 graph)
    assert st.n["_all"] == n_requests and st.n[root] == n_requests and st.step == n_requests
    # identical attempts commute, so the result must equal a plain sequential replay
    replay = LearnerState("replay")
    for _ in range(n_requests):
        api.L.update(replay, root, True)
    assert api.L.mastery(st, root) == api.L.mastery(replay, root)
    assert st.theta == replay.theta


def test_updates_are_serialised_even_when_the_engine_is_slow(client, monkeypatch):
    """Widen the race window on purpose: an unguarded read-modify-write would lose counts here."""
    real_update = api.L.update
    seen = {"calls": 0, "inside": 0, "overlap": 0}

    def slow_update(st, ind, correct, item=None):
        seen["inside"] += 1
        seen["overlap"] = max(seen["overlap"], seen["inside"])
        calls = seen["calls"]
        time.sleep(0.002)
        seen["calls"] = calls + 1
        try:
            return real_update(st, ind, correct, item)
        finally:
            seen["inside"] -= 1

    monkeypatch.setattr(api.L, "update", slow_update)
    n_requests, started = 60, threading.Barrier(12)

    def worker(i):
        if i < 12:
            started.wait(timeout=10)  # release the first wave together
        return post(client, learner_id="busy", correct=i % 2 == 0).status_code

    with ThreadPoolExecutor(max_workers=12) as pool:
        codes = list(pool.map(worker, range(n_requests)))
    assert codes == [200] * n_requests
    assert seen["overlap"] == 1  # never two updates at once
    assert seen["calls"] == n_requests  # no lost update
    assert api.STATES["busy"].n["_all"] == n_requests


def test_concurrent_reads_writes_and_deletes_do_not_error(client):
    def worker(i):
        lid = f"p{i % 5}"
        if i % 7 == 0:
            return client.delete(f"/learner/{lid}").status_code in (204, 404)
        if i % 3 == 0:
            return client.get(f"/recommend/{lid}?k=5").status_code == 200
        if i % 3 == 1:
            return client.get(f"/mastery/{lid}").status_code == 200
        return post(client, learner_id=lid, correct=i % 2 == 0).status_code == 200

    with ThreadPoolExecutor(max_workers=16) as pool:
        assert all(pool.map(worker, range(400)))


# --------------------------------------------------------------------------- the user stories, end to end


def test_user_story_end_to_end(client, G):
    pupil = "ama"

    # A new pupil needs no registration and gets starting skills (US2).
    start = recs(client, pupil)
    assert 1 <= len(start) <= 3
    for r in start:
        assert r["rationale"].strip() and r["ghana_context"].strip()
        assert all(client.get(f"/mastery/{pupil}").json()[p] >= READY for p in G.predecessors(r["indicator"]))
    skill = start[0]["indicator"]
    m0 = client.get(f"/mastery/{pupil}").json()[skill]

    # US1: after an answer, mastery updates immediately.
    first = post(client, learner_id=pupil, indicator=skill, correct=True).json()
    assert first["new_mastery"] > m0
    assert client.get(f"/mastery/{pupil}").json()[skill] == round(first["new_mastery"], 3)

    # A wrong answer pulls it back down.
    slip = post(client, learner_id=pupil, indicator=skill, correct=False).json()
    assert slip["new_mastery"] < first["new_mastery"]

    # Keep answering correctly until the skill is mastered; it then drops out of the recommendations.
    mastery = slip["new_mastery"]
    for _ in range(20):
        if mastery >= MASTERY:
            break
        mastery = post(client, learner_id=pupil, indicator=skill, correct=True).json()["new_mastery"]
    assert mastery >= MASTERY
    later = recs(client, pupil, k=50)
    assert skill not in [r["indicator"] for r in later]
    assert [r["indicator"] for r in recs(client, pupil)] != [r["indicator"] for r in start]

    # Mastering it makes the skills that build on it ready.
    unlocked = {s for s in G.successors(skill) if all(p == skill for p in G.predecessors(s))}
    assert unlocked <= {r["indicator"] for r in later}

    # US3: a teacher sets a goal and the best next step is on the path to it.
    toward = recs(client, pupil, goal=GOAL)
    assert toward and toward[0]["indicator"] in on_path(G, GOAL)
    assert client.get(f"/recommend/{pupil}?goal=B9.9.9.9.9").status_code == 404

    # US4: a class cap keeps everything at or below that class, with or without a goal.
    assert any(r["grade"] > 2 for r in later)
    for params in ({"grade_max": 2}, {"grade_max": 2, "goal": GOAL}):
        capped = recs(client, pupil, k=50, **params)
        assert capped and all(r["grade"] <= 2 for r in capped)

    # Another pupil is untouched, and starting over gives this one a clean slate.
    assert [r["indicator"] for r in recs(client, "kofi")] == [r["indicator"] for r in start]
    assert client.delete(f"/learner/{pupil}").status_code == 204
    assert [r["indicator"] for r in recs(client, pupil)] == [r["indicator"] for r in start]
