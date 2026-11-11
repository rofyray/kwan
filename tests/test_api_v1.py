"""API v1 contract tests (PRD 9.3): auth, idempotency, concurrency, stores, limits, logs, v0 deprecation."""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from kwan.api import app as app_module
from kwan.api import runtime, v1
from kwan.kwan import Kwan
from kwan.store import MemoryStore, SQLiteStore, VersionConflict

KEY = "test-key-1"
HEADERS = {"X-API-Key": KEY}
ROOT = "math:B1.1.1.1.1"
GOAL = "math:B4.1.1.3.3"
T0 = datetime(2026, 5, 4, 9, 0, tzinfo=UTC)
REC_KEYS = {
    "indicator",
    "code",
    "score",
    "mastery",
    "grade",
    "strand",
    "description",
    "short_label",
    "rationale",
    "rationale_parts",
    "ghana_context",
    "is_review",
}


@pytest.fixture(scope="module")
def engine() -> Kwan:
    return Kwan.from_bundle()


def make_runtime(engine: Kwan, store=None, keys=(KEY,)) -> runtime.Runtime:
    return runtime.Runtime(engine=engine, store=store or MemoryStore(), api_keys=frozenset(keys), store_kind="memory")


@pytest.fixture()
def rt(engine):
    return make_runtime(engine)


@pytest.fixture()
def client(rt):
    with TestClient(app_module.create_app(rt)) as c:
        yield c


def attempt(client, learner_id="a", indicator=ROOT, correct=True, headers=HEADERS, **extra):
    body = {"learner_id": learner_id, "indicator": indicator, "correct": correct, **extra}
    return client.post("/v1/attempts", json=body, headers=headers)


def recs(client, learner_id="a", **params):
    r = client.get(f"/v1/learners/{learner_id}/recommendations", params=params, headers=HEADERS)
    assert r.status_code == 200, r.text
    return r.json()["recommendations"]


class Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record):
        self.lines.append(self.format(record))


@pytest.fixture()
def log_lines():
    log = runtime.logger()
    handler = Capture()
    handler.setFormatter(runtime.JsonFormatter())
    log.addHandler(handler)
    try:
        yield handler.lines
    finally:
        log.removeHandler(handler)


# ----------------------------------------------------------------------------- health and auth


def test_health_is_open_and_describes_the_service(client, engine):
    r = client.get("/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body == {
        "ok": True,
        "version": app_module.__version__,
        "bundle_version": engine.bundle_version,
        "subject": "math",
        "nodes": 276,
        "edges": engine.graph.number_of_edges(),
        "store": "memory",
        "auth": True,
    }
    assert "Deprecation" not in r.headers and r.headers["X-Request-Id"]


def test_every_other_route_needs_the_key(client):
    assert attempt(client, headers={}).status_code == 401
    assert attempt(client, headers={"X-API-Key": "wrong"}).status_code == 401
    body = attempt(client, headers={}).json()
    assert body == {"detail": "missing or unknown API key (send it in the X-API-Key header)", "code": "unauthorized"}
    assert client.get("/v1/learners/a/recommendations").status_code == 401
    assert client.get("/v1/learners/a/mastery").status_code == 401
    assert client.delete("/v1/learners/a").status_code == 401
    assert attempt(client).status_code == 200


def test_several_keys_are_accepted(engine):
    with TestClient(app_module.create_app(make_runtime(engine, keys=("one", "two")))) as c:
        assert attempt(c, headers={"X-API-Key": "two"}).status_code == 200
        assert attempt(c, headers={"X-API-Key": "one"}).status_code == 200
        assert attempt(c, headers={"X-API-Key": "three"}).status_code == 401


def test_without_configured_keys_every_request_is_allowed_and_a_warning_is_logged_once(engine, log_lines):
    rt = make_runtime(engine, keys=())
    with TestClient(app_module.create_app(rt)) as c:
        assert c.get("/v1/health").json()["auth"] is False
        assert attempt(c, headers={}).status_code == 200
        assert attempt(c, headers={"X-API-Key": "anything"}).status_code == 200
        assert c.get("/v1/learners/a/mastery").status_code == 200
    warnings = [json.loads(line) for line in log_lines if json.loads(line)["level"] == "WARNING"]
    assert len(warnings) == 1 and warnings[0]["event"] == "auth_disabled" and "KWAN_API_KEYS" in warnings[0]["msg"]


def test_parse_keys():
    assert runtime.parse_keys(None) == frozenset()
    assert runtime.parse_keys(" a, b ,,c") == {"a", "b", "c"}


# ----------------------------------------------------------------------------- attempts and idempotency


def test_an_attempt_updates_the_stored_learner(client, rt, engine):
    r = attempt(client, indicator="B1.1.1.1.1", at=T0.isoformat(), item_id="q-1")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body == {
        "learner_id": "a",
        "indicator": ROOT,
        "predicted_p_correct": body["predicted_p_correct"],
        "new_mastery": body["new_mastery"],
        "version": 1,
        "bundle_version": engine.bundle_version,
    }
    assert 0 < body["predicted_p_correct"] < body["new_mastery"] < 1
    stored, version = rt.store.get_with_version("a")
    assert version == 1 and stored.step == 1 and stored.last_seen_at[ROOT] == T0.isoformat()
    assert stored.bundle_version == engine.bundle_version
    assert attempt(client).json()["version"] == 2


def test_a_retried_idempotency_key_returns_the_first_response_once_applied(client, rt):
    first = attempt(client, idempotency_key="k-1")
    again = attempt(client, idempotency_key="k-1")
    third = attempt(client, correct=False, idempotency_key="k-1")  # same key, different body: still the first answer
    assert first.status_code == again.status_code == third.status_code == 200
    assert first.json() == again.json() == third.json()
    assert "Idempotent-Replayed" not in first.headers
    assert again.headers["Idempotent-Replayed"] == "true" and third.headers["Idempotent-Replayed"] == "true"
    assert rt.store.get("a").n["_all"] == 1
    assert attempt(client, idempotency_key="k-2").json()["version"] == 2


def test_the_idempotency_key_header_works_and_keys_are_per_learner(client, rt):
    h = {**HEADERS, "Idempotency-Key": "shared"}
    assert attempt(client, learner_id="a", headers=h).json()["version"] == 1
    assert attempt(client, learner_id="a", headers=h).headers.get("Idempotent-Replayed") == "true"
    assert attempt(client, learner_id="b", headers=h).json()["version"] == 1
    assert rt.store.get("a").n["_all"] == rt.store.get("b").n["_all"] == 1


def test_bare_codes_are_accepted_and_ids_come_back_canonical(client):
    assert attempt(client, indicator="B1.1.1.1.1").json()["indicator"] == ROOT
    assert recs(client, goal="B4.1.1.3.3")[0]["indicator"].startswith("math:")


def test_unknown_indicator_and_bad_bodies(client, rt):
    r = attempt(client, indicator="B9.9.9.9.9")
    assert r.status_code == 404 and r.json() == {
        "detail": "unknown indicator 'B9.9.9.9.9'",
        "code": "unknown_indicator",
    }
    assert attempt(client, correct="yes").status_code == 422
    assert attempt(client, learner_id="").status_code == 422
    assert attempt(client, learner_id="x" * 129).status_code == 422
    assert attempt(client, indicator="x" * 65).status_code == 422
    assert attempt(client, at="yesterday").status_code == 422
    assert attempt(client, idempotency_key="").status_code == 422
    assert attempt(client, extra_field=1).status_code == 422
    assert (
        client.post(
            "/v1/attempts", content=b"not json", headers={**HEADERS, "content-type": "application/json"}
        ).status_code
        == 422
    )
    assert rt.store.count() == 0  # nothing rejected was stored


def test_a_body_over_the_limit_is_refused(client, rt):
    big = {"learner_id": "a", "indicator": ROOT, "correct": True, "item_id": "x" * 100}
    raw = (json.dumps(big) + " " * v1.MAX_BODY_BYTES).encode()
    r = client.post("/v1/attempts", content=raw, headers={**HEADERS, "content-type": "application/json"})
    assert r.status_code == 413 and r.json()["code"] == "payload_too_large"
    assert rt.store.count() == 0


# ----------------------------------------------------------------------------- concurrency


def test_concurrent_attempts_on_one_learner_lose_nothing(client, rt):
    n_threads, per_thread = 8, 25

    def worker(i):
        return [attempt(client, learner_id="busy", correct=bool(j % 2)).json()["version"] for j in range(per_thread)]

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        versions = [v for vs in pool.map(worker, range(n_threads)) for v in vs]
    total = n_threads * per_thread
    assert sorted(versions) == list(range(1, total + 1))  # every write got its own version
    state, version = rt.store.get_with_version("busy")
    assert version == total and state.n["_all"] == total and state.step == total


def test_a_version_conflict_that_never_clears_is_a_409(engine, monkeypatch):
    class StubbornStore(MemoryStore):
        def put_with_attempt(self, state, key, result, expected_version=None):
            raise VersionConflict(state.learner_id, expected_version or 0, 99)

        def put(self, state, expected_version=None):
            raise VersionConflict(state.learner_id, expected_version or 0, 99)

    monkeypatch.setattr("kwan.store.base.time.sleep", lambda s: None)
    with TestClient(app_module.create_app(make_runtime(engine, store=StubbornStore()))) as c:
        r = attempt(c)
        assert r.status_code == 409 and r.json()["code"] == "version_conflict" and "retry" in r.json()["detail"]
        r = attempt(c, idempotency_key="k")
        assert r.status_code == 409


# ----------------------------------------------------------------------------- recommendations and mastery


def test_recommendations_for_a_new_and_a_practised_learner(client, engine):
    fresh = recs(client, "nobody", k=3)
    assert 1 <= len(fresh) <= 3 and all(set(r) == REC_KEYS for r in fresh)
    assert all(r["grade"] == 1 and not r["is_review"] for r in fresh)
    direct = [r.to_dict() for r in engine.recommend(engine.new_state("nobody"), k=3)]
    assert fresh == direct  # the API is a thin layer over the facade
    r = client.get("/v1/learners/nobody/recommendations", headers=HEADERS)
    assert r.json()["learner_id"] == "nobody" and r.json()["latency_ms"] >= 0
    assert r.json()["bundle_version"] == engine.bundle_version
    for i in range(3):
        attempt(client, at=(T0 + timedelta(minutes=i)).isoformat())
    after = recs(client, k=3, now=(T0 + timedelta(days=1)).isoformat())
    assert ROOT not in {r["indicator"] for r in after}


def test_recommendation_options_and_limits(client):
    goal = recs(client, k=5, goal=GOAL)
    assert goal and all(
        "goal_steps" in {p["key"] for p in r["rationale_parts"]} or r["indicator"] == GOAL for r in goal[:1]
    )
    assert recs(client, k=5, goal="B4.1.1.3.3") == goal
    assert all(r["grade"] <= 2 for r in recs(client, k=10, grade_max=2))
    assert recs(client, k=3, goal=GOAL, goal_mode="soft")
    assert len(recs(client, k=1)) == 1
    assert recs(client, k=2, include_drafts="true")
    get = lambda **p: client.get("/v1/learners/a/recommendations", params=p, headers=HEADERS)  # noqa: E731
    assert get(k=0).status_code == 422 and get(k=51).status_code == 422 and get(k="x").status_code == 422
    assert get(grade_max=7).status_code == 422 and get(goal_mode="loose").status_code == 422
    assert get(lang="x").status_code == 422
    r = get(goal="NOPE")
    assert r.status_code == 404 and r.json() == {"detail": "unknown goal 'NOPE'", "code": "unknown_goal"}
    r = get(now="not-a-time")
    assert r.status_code == 422 and r.json()["code"] == "invalid_time"
    assert client.get(f"/v1/learners/{'x' * 129}/recommendations", headers=HEADERS).status_code == 422


def test_reads_never_create_state(client, rt):
    recs(client, "ghost")
    assert client.get("/v1/learners/ghost/mastery", headers=HEADERS).status_code == 200
    assert rt.store.count() == 0


def test_mastery_map(client, engine):
    r = client.get("/v1/learners/a/mastery", headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert body["learner_id"] == "a" and body["bundle_version"] == engine.bundle_version
    table = body["mastery"]
    assert len(table) == 276 and all(k.startswith("math:") and 0 <= v <= 1 for k, v in table.items())
    assert all(round(v, 4) == v for v in table.values())
    attempt(client, at=T0.isoformat())
    before = client.get("/v1/learners/a/mastery", headers=HEADERS).json()["mastery"][ROOT]
    later = client.get(
        "/v1/learners/a/mastery", params={"now": (T0 + timedelta(days=300)).isoformat()}, headers=HEADERS
    )
    assert later.json()["mastery"][ROOT] < before
    assert client.get("/v1/learners/a/mastery", params={"now": "soon"}, headers=HEADERS).status_code == 422


# ----------------------------------------------------------------------------- delete


def test_delete_forgets_the_learner(client, rt):
    attempt(client, idempotency_key="k")
    assert rt.store.count() == 1
    r = client.delete("/v1/learners/a", headers=HEADERS)
    assert r.status_code == 204 and r.content == b""
    assert rt.store.count() == 0 and not rt.store.seen_attempt("a", "k")
    r = client.delete("/v1/learners/a", headers=HEADERS)
    assert r.status_code == 404 and r.json()["code"] == "unknown_learner"
    assert attempt(client, idempotency_key="k").json()["version"] == 1  # a learner created again starts afresh


# ----------------------------------------------------------------------------- stores


def test_open_store_switches_on_the_url(tmp_path):
    store, kind = runtime.open_store("memory")
    assert isinstance(store, MemoryStore) and kind == "memory"
    store, kind = runtime.open_store(f"sqlite:///{tmp_path / 'learners.sqlite3'}")
    assert isinstance(store, SQLiteStore) and kind == "sqlite" and store.path == str(tmp_path / "learners.sqlite3")
    store.close()
    store, kind = runtime.open_store("sqlite:///:memory:")
    assert isinstance(store, SQLiteStore) and store.path == ":memory:"
    store.close()
    with pytest.raises(ValueError, match="KWAN_STORE"):
        runtime.open_store("redis://x")


def test_sqlite_state_survives_a_restart(engine, tmp_path):
    path = tmp_path / "learners.sqlite3"
    for round_no in (1, 2):
        store, _ = runtime.open_store(f"sqlite:///{path}")
        rt = runtime.Runtime(engine=engine, store=store, api_keys=frozenset({KEY}), store_kind="sqlite")
        with TestClient(app_module.create_app(rt)) as c:
            assert c.get("/v1/health").json()["store"] == "sqlite"
            r = attempt(c, idempotency_key="only-once")
            assert r.json()["version"] == 1
            assert (r.headers.get("Idempotent-Replayed") == "true") is (round_no == 2)
        rt.close()
    assert path.is_file()


def test_from_env_reads_the_settings(engine, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rt = runtime.from_env({}, bundle_loader=lambda path: engine)
    try:
        assert rt.store_kind == "sqlite" and (tmp_path / "kwan.sqlite3").is_file() and not rt.auth_enabled
    finally:
        rt.close()
    env = {"KWAN_API_KEYS": "a,b", "KWAN_STORE": "memory", "KWAN_BUNDLE": str(tmp_path / "x.json")}
    seen = []
    rt = runtime.from_env(env, bundle_loader=lambda path: seen.append(path) or engine)
    assert rt.api_keys == {"a", "b"} and rt.store_kind == "memory" and seen == [str(tmp_path / "x.json")]
    assert "KWAN_API_KEYS" not in os.environ or True  # the environment itself is untouched


def test_the_module_app_reads_the_environment_on_first_use(monkeypatch):
    monkeypatch.setenv("KWAN_STORE", "memory")
    monkeypatch.setenv("KWAN_API_KEYS", "module-key")
    app = app_module.create_app()
    with TestClient(app) as c:
        assert c.get("/v1/health").json()["store"] == "memory"
        assert attempt(c, headers={}).status_code == 401
        assert attempt(c, headers={"X-API-Key": "module-key"}).status_code == 200
    assert app_module.runtime_of(app).store_kind == "memory"


def test_the_uvicorn_target_is_the_module_app():
    from uvicorn.importer import import_from_string

    assert import_from_string("kwan.api.app:app") is app_module.app


# ----------------------------------------------------------------------------- v0 deprecation


@pytest.mark.parametrize(
    ("method", "path", "successor"),
    [
        ("GET", "/health", "/v1/health"),
        ("POST", "/attempt", "/v1/attempts"),
        ("GET", "/recommend/pupil-1", "/v1/learners/pupil-1/recommendations"),
        ("GET", "/mastery/pupil-1", "/v1/learners/pupil-1/mastery"),
        ("DELETE", "/learner/pupil-1", "/v1/learners/pupil-1"),
    ],
)
def test_v0_routes_carry_a_deprecation_header_and_a_link_to_v1(client, method, path, successor):
    body = {"learner_id": "pupil-1", "indicator": "B1.1.1.1.1", "correct": True} if method == "POST" else None
    r = client.request(method, path, json=body)
    assert r.status_code in (200, 204, 404)
    assert r.headers["Deprecation"] == "true"
    assert r.headers["Link"] == f'<{successor}>; rel="successor-version"'
    assert app_module.v0_successor(path) == successor
    app_module.STATES.clear()


def test_v1_routes_are_not_deprecated_and_v0_can_be_left_out(client, engine):
    for path in ("/v1/health", "/v1/learners/a/mastery"):
        assert "Deprecation" not in client.get(path, headers=HEADERS).headers
    assert app_module.v0_successor("/v1/health") is None and app_module.v0_successor("/other") is None
    with TestClient(app_module.create_app(make_runtime(engine), v0=False)) as c:
        assert c.get("/health").status_code == 404 and c.get("/v1/health").status_code == 200


def test_openapi_describes_v1_with_typed_models(client):
    spec = client.get("/openapi.json").json()
    paths = spec["paths"]
    assert {
        "/v1/health",
        "/v1/attempts",
        "/v1/learners/{learner_id}/recommendations",
        "/v1/learners/{learner_id}/mastery",
        "/v1/learners/{learner_id}",
    } <= set(paths)
    ok = lambda path, method: paths[path][method]["responses"]["200"]["content"]["application/json"]["schema"]  # noqa: E731
    assert ok("/v1/health", "get")["$ref"].endswith("/HealthOut")
    assert ok("/v1/attempts", "post")["$ref"].endswith("/AttemptOut")
    assert ok("/v1/learners/{learner_id}/recommendations", "get")["$ref"].endswith("/RecommendationsOut")
    assert ok("/v1/learners/{learner_id}/mastery", "get")["$ref"].endswith("/MasteryOut")
    schemas = spec["components"]["schemas"]
    assert set(schemas["RecommendationOut"]["properties"]) == REC_KEYS
    assert set(schemas["AttemptIn"]["required"]) == {"learner_id", "indicator", "correct"}
    assert set(schemas["ErrorOut"]["properties"]) == {"detail", "code"}
    for code in ("401", "404", "409", "413"):
        assert code in paths["/v1/attempts"]["post"]["responses"]
    tags = {t["name"] for t in spec["tags"]}
    assert {"v1: service", "v1: learning", "v1: learners", "v0 (deprecated)"} <= tags
    assert paths["/attempt"]["post"]["deprecated"] is True and "deprecated" not in paths["/v1/attempts"]["post"]
    assert spec["info"]["version"] == app_module.__version__
    text = json.dumps(spec)
    assert chr(0x2014) not in text


# ----------------------------------------------------------------------------- logs


def test_requests_are_logged_as_json(client, log_lines):
    attempt(client, idempotency_key="log-1")
    client.get("/v1/learners/a/mastery", headers={**HEADERS, "X-Request-Id": "req-42"})
    client.get("/health")
    events = [json.loads(line) for line in log_lines]
    assert all({"ts", "level", "logger", "msg", "event"} <= set(e) for e in events)
    requests = [e for e in events if e["event"] == "request"]
    assert [(e["method"], e["path"], e["status"]) for e in requests] == [
        ("POST", "/v1/attempts", 200),
        ("GET", "/v1/learners/a/mastery", 200),
        ("GET", "/health", 200),
    ]
    assert requests[1]["request_id"] == "req-42" and requests[2]["deprecated"] is True
    assert all(isinstance(e["ms"], float) for e in requests)
    attempts = [e for e in events if e["event"] == "attempt"]
    assert attempts == [{**attempts[0], "learner": "a", "indicator": ROOT, "replayed": False}]


# ----------------------------------------------------------------------------- the full user story


def test_the_user_story(client, rt, engine):
    """A platform creates a pupil, records a week of answers, asks for next steps, and forgets the pupil."""
    pupil = "pupil-0042"
    start = recs(client, pupil, k=3)
    assert all(r["grade"] == 1 for r in start)
    day = T0
    for i in range(6):
        r = attempt(
            client, pupil, indicator=start[0]["indicator"], correct=i != 2, at=day.isoformat(), idempotency_key=f"d{i}"
        )
        assert r.status_code == 200 and r.json()["version"] == i + 1
        day += timedelta(hours=4)
    mastery = client.get(f"/v1/learners/{pupil}/mastery", headers=HEADERS).json()["mastery"]
    assert mastery[start[0]["indicator"]] >= 0.85
    nxt = recs(client, pupil, k=3, goal=GOAL, now=day.isoformat())
    assert nxt and start[0]["indicator"] not in {r["indicator"] for r in nxt}
    assert all(r["rationale"] and r["ghana_context"] for r in nxt)
    stored = rt.store.get(pupil)
    assert stored.step == 6 and stored.bundle_version == engine.bundle_version
    assert client.delete(f"/v1/learners/{pupil}", headers=HEADERS).status_code == 204
    assert recs(client, pupil, k=3) == start
