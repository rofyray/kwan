"""Behaviour every learner-state store must have. The same tests run on `MemoryStore`, on an
in-memory `SQLiteStore` and on a `SQLiteStore` in a file. Set `KWAN_TEST_POSTGRES_DSN` to a throwaway
PostgreSQL database (and install `psycopg`) to run them on `PostgresStore` as well."""

import json
import os
import subprocess
import sys
import threading
from datetime import UTC, datetime, timedelta

import pytest

from kwan.learner.elo import LearnerState
from kwan.store import (
    MAX_ID_LENGTH,
    AttemptAlreadyRecorded,
    AttemptOutcome,
    LearnerStore,
    MemoryStore,
    PostgresStore,
    SQLiteStore,
    StoreClosed,
    VersionConflict,
    apply_attempt,
)

KINDS = ["memory", "sqlite-memory", "sqlite-file"]
POSTGRES_DSN = os.environ.get("KWAN_TEST_POSTGRES_DSN")  # opt-in: the suite also runs on a PostgreSQL you point it at
if POSTGRES_DSN:
    KINDS.append("postgres")


class FakeClock:
    """A clock the test moves by hand."""

    def __init__(self):
        self.now = datetime(2026, 11, 10, 8, 0, 0, tzinfo=UTC)

    def __call__(self):
        return self.now

    def tick(self, seconds):
        self.now += timedelta(seconds=seconds)


def make_store(kind, tmp_path, clock=None):
    if kind == "memory":
        return MemoryStore(clock=clock)
    if kind == "sqlite-memory":
        return SQLiteStore(":memory:", clock=clock)
    if kind == "postgres":
        store = PostgresStore(POSTGRES_DSN, clock=clock)
        with store._transaction() as conn:  # a test database: start every test empty
            conn.execute("TRUNCATE kwan_learners, kwan_attempts")
        return store
    return SQLiteStore(tmp_path / "learners.db", clock=clock)


@pytest.fixture(params=KINDS)
def kind(request):
    return request.param


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def store(kind, tmp_path, clock):
    s = make_store(kind, tmp_path, clock)
    yield s
    s.close()


def v2_state(learner_id="pupil-1", step=3):
    """A v2 state with every field filled, built only through `from_dict`."""
    return LearnerState.from_dict(
        {
            "learner_id": learner_id,
            "subject": "math",
            "bundle_version": "math-1.0.0",
            "theta": 0.25,
            "theta_k": {"math:B1.1.1.1.1": 0.5, "math:B1.1.1.1.2": -0.125},
            "n": {"math:B1.1.1.1.1": 2, "math:B1.1.1.1.2": 1, "_all": 3},
            "streak": {"math:B1.1.1.1.1": 2, "math:B1.1.1.1.2": 0},
            "last_seen": {"math:B1.1.1.1.1": 2, "math:B1.1.1.1.2": 1},
            "last_seen_at": {"math:B1.1.1.1.1": "2026-11-10T08:00:00+00:00"},
            "step": step,
            "schema_version": 2,
        }
    )


V1_STATE = {  # the v1 layout: bare codes, no subject, bundle_version, last_seen_at or schema_version
    "learner_id": "old-pupil",
    "theta": -0.5,
    "theta_k": {"B1.1.1.1.1": 0.75},
    "n": {"B1.1.1.1.1": 4, "_all": 4},
    "last_seen": {"B1.1.1.1.1": 3},
    "streak": {"B1.1.1.1.1": 1},
    "step": 4,
}


def bump(state):
    """An attempt for tests: one more step, and report the step reached."""
    state.step += 1
    return {"step": state.step}


# --------------------------------------------------------------------------- contract and unknown ids


def test_store_satisfies_the_protocol(store):
    assert isinstance(store, LearnerStore)


def test_unknown_learner_has_nothing(store):
    assert store.get("nobody") is None
    assert store.get_with_version("nobody") is None
    assert store.updated_at("nobody") is None
    assert store.delete("nobody") is False
    assert store.seen_attempt("nobody", "k1") is False
    assert store.get_attempt("nobody", "k1") is None
    assert store.list_ids() == []
    assert store.count() == 0


@pytest.mark.parametrize("bad", ["", "   ", "\t\n", None, 7, b"pupil", "x" * (MAX_ID_LENGTH + 1), "a\x00b"])
def test_bad_learner_ids_are_refused(store, bad):
    for call in (store.get, store.get_with_version, store.delete, store.updated_at):
        with pytest.raises(ValueError):
            call(bad)
    with pytest.raises(ValueError):
        store.seen_attempt(bad, "k1")
    with pytest.raises(ValueError):
        store.record_attempt(bad, "k1", {})
    with pytest.raises(ValueError):
        store.get_attempt(bad, "k1")
    assert store.count() == 0


@pytest.mark.parametrize("bad", ["", "  ", None, 3, "k" * (MAX_ID_LENGTH + 1)])
def test_bad_idempotency_keys_are_refused(store, bad):
    with pytest.raises(ValueError):
        store.seen_attempt("pupil-1", bad)
    with pytest.raises(ValueError):
        store.record_attempt("pupil-1", bad, {})
    with pytest.raises(ValueError):
        store.get_attempt("pupil-1", bad)
    with pytest.raises(ValueError):
        store.put_with_attempt(v2_state(), bad, {})
    assert store.get("pupil-1") is None


def test_state_with_a_bad_learner_id_is_not_stored(store):
    for bad in ("", "  ", "x" * (MAX_ID_LENGTH + 1)):
        with pytest.raises(ValueError):
            store.put(LearnerState.from_dict({"learner_id": bad}))
    assert store.count() == 0


def test_id_at_the_length_limit_is_accepted(store):
    long_id = "x" * MAX_ID_LENGTH
    assert store.put(LearnerState.from_dict({"learner_id": long_id})) == 1
    assert store.get(long_id).learner_id == long_id


def test_ids_are_exact_strings(store):
    """No trimming, no case folding, no number coercion: the id is opaque."""
    for learner_id in ("42", "Pupil-1", "pupil-1", " pupil-1", "pupil 1", "ɔkyerɛkyerɛni-7"):
        store.put(LearnerState.from_dict({"learner_id": learner_id, "step": len(learner_id)}))
    assert store.count() == 6
    assert store.get("42").learner_id == "42"
    assert store.get(" pupil-1").step == 8
    assert store.get("pupil-1").step == 7
    assert store.get("PUPIL-1") is None
    assert store.get("ɔkyerɛkyerɛni-7").learner_id == "ɔkyerɛkyerɛni-7"


# --------------------------------------------------------------------------- round trip and versions


def test_round_trip_keeps_every_field(store):
    state = v2_state()
    assert store.put(state) == 1
    back = store.get("pupil-1")
    assert isinstance(back, LearnerState)
    assert back.to_dict() == state.to_dict()
    assert back == state


def test_stored_state_is_a_copy(store):
    state = v2_state()
    store.put(state)
    state.step = 99  # changing the caller's object after the write changes nothing in the store
    state.theta_k["math:B1.1.1.1.1"] = 9.0
    first = store.get("pupil-1")
    assert first.step == 3 and first.theta_k["math:B1.1.1.1.1"] == 0.5
    first.step = 50  # nor does changing what the store handed out
    first.n["_all"] = 1000
    again = store.get("pupil-1")
    assert again.step == 3 and again.n["_all"] == 3
    assert again is not first


def test_v1_format_state_round_trip(store):
    old = LearnerState.from_dict(V1_STATE)
    assert store.put(old) == 1
    back = store.get("old-pupil")
    assert back.to_dict() == old.to_dict()
    for field, value in V1_STATE.items():  # nothing from the v1 dictionary is lost or renamed
        assert back.to_dict()[field] == value
    assert back.to_dict()["last_seen_at"] == {}


def test_state_that_json_cannot_hold_is_refused(store):
    bad = LearnerState.from_dict({"learner_id": "pupil-1", "theta": float("nan")})
    with pytest.raises(ValueError):
        store.put(bad)
    with pytest.raises(ValueError):
        store.put_with_attempt(bad, "k1", {})
    assert store.get("pupil-1") is None
    assert store.seen_attempt("pupil-1", "k1") is False


def test_version_goes_up_by_one_on_every_write(store):
    assert store.put(v2_state(step=1)) == 1
    assert store.put(v2_state(step=2)) == 2
    assert store.put(v2_state(step=3), expected_version=2) == 3
    state, version = store.get_with_version("pupil-1")
    assert (state.step, version) == (3, 3)
    assert store.put(v2_state("pupil-2")) == 1  # versions are per learner


def test_expected_version_zero_means_create(store):
    assert store.put(v2_state(step=1), expected_version=0) == 1
    with pytest.raises(VersionConflict) as err:
        store.put(v2_state(step=2), expected_version=0)
    assert (err.value.learner_id, err.value.expected, err.value.actual) == ("pupil-1", 0, 1)
    assert store.get_with_version("pupil-1")[0].step == 1


def test_stale_write_is_a_conflict_and_changes_nothing(store, clock):
    store.put(v2_state(step=1))
    stamp_before = store.updated_at("pupil-1")
    _, version = store.get_with_version("pupil-1")
    store.put(v2_state(step=2), expected_version=version)  # somebody else writes first
    clock.tick(5)
    with pytest.raises(VersionConflict) as err:
        store.put(v2_state(step=77), expected_version=version)
    assert (err.value.expected, err.value.actual) == (1, 2)
    assert "pupil-1" in str(err.value)
    state, now_version = store.get_with_version("pupil-1")
    assert (state.step, now_version) == (2, 2)
    assert store.updated_at("pupil-1") == stamp_before


def test_expected_version_for_a_learner_that_is_not_stored_is_a_conflict(store):
    with pytest.raises(VersionConflict) as err:
        store.put(v2_state(), expected_version=4)
    assert (err.value.expected, err.value.actual) == (4, 0)
    assert store.get("pupil-1") is None
    assert store.count() == 0


@pytest.mark.parametrize("bad", [-1, 1.0, "1", True])
def test_bad_expected_version_is_refused(store, bad):
    with pytest.raises(ValueError):
        store.put(v2_state(), expected_version=bad)
    with pytest.raises(ValueError):
        store.put_with_attempt(v2_state(), "k1", {}, expected_version=bad)
    assert store.get("pupil-1") is None


def test_updated_at_is_the_time_of_the_last_write(store, clock):
    store.put(v2_state(step=1))
    assert store.updated_at("pupil-1") == "2026-11-10T08:00:00.000000+00:00"
    clock.tick(90)
    store.put(v2_state(step=2))
    assert store.updated_at("pupil-1") == "2026-11-10T08:01:30.000000+00:00"
    clock.tick(30)
    store.get("pupil-1")  # reading is not writing
    assert store.updated_at("pupil-1") == "2026-11-10T08:01:30.000000+00:00"


# --------------------------------------------------------------------------- delete


def test_delete_removes_the_state(store):
    store.put(v2_state("pupil-1"))
    store.put(v2_state("pupil-2"))
    assert store.delete("pupil-1") is True
    assert store.get("pupil-1") is None
    assert store.get_with_version("pupil-1") is None
    assert store.updated_at("pupil-1") is None
    assert store.list_ids() == ["pupil-2"]
    assert store.count() == 1
    assert store.delete("pupil-1") is False  # already gone


def test_delete_removes_the_idempotency_records_too(store):
    store.put_with_attempt(v2_state("pupil-1"), "k1", {"a": 1})
    store.record_attempt("pupil-1", "k2", {"a": 2})
    store.put_with_attempt(v2_state("pupil-2"), "k1", {"b": 1})
    assert store.delete("pupil-1") is True
    assert store.seen_attempt("pupil-1", "k1") is False
    assert store.get_attempt("pupil-1", "k1") is None
    assert store.get_attempt("pupil-1", "k2") is None
    assert store.get_attempt("pupil-2", "k1") == {"b": 1}  # the other learner is untouched
    assert store.get("pupil-2") is not None


def test_delete_of_a_learner_with_only_idempotency_records(store):
    store.record_attempt("pupil-1", "k1", {"a": 1})
    assert store.count() == 0
    assert store.delete("pupil-1") is True
    assert store.seen_attempt("pupil-1", "k1") is False
    assert store.delete("pupil-1") is False


def test_learner_created_again_after_delete_starts_fresh(store):
    store.put(v2_state(step=1))
    store.put(v2_state(step=2))
    store.put_with_attempt(v2_state(step=3), "k1", {"step": 3})
    store.delete("pupil-1")
    with pytest.raises(VersionConflict):  # a writer holding the old version cannot bring the pupil back
        store.put(v2_state(step=4), expected_version=3)
    assert store.get("pupil-1") is None
    assert store.put(v2_state(step=1), expected_version=0) == 1
    outcome = apply_attempt(store, "pupil-1", "k1", bump)  # the old key is forgotten, so this is applied
    assert outcome.replayed is False and outcome.result == {"step": 2}


# --------------------------------------------------------------------------- list and count


def test_list_and_count(store):
    ids = [f"pupil-{i:02d}" for i in range(12)]
    for learner_id in reversed(ids):
        store.put(LearnerState.from_dict({"learner_id": learner_id}))
    store.put(LearnerState.from_dict({"learner_id": "pupil-03", "step": 1}))  # a second write adds no id
    assert store.count() == 12
    assert store.list_ids() == ids
    assert store.list_ids(limit=5) == ids[:5]
    assert store.list_ids(limit=5, offset=5) == ids[5:10]
    assert store.list_ids(limit=5, offset=10) == ids[10:]
    assert store.list_ids(limit=5, offset=50) == []
    assert store.list_ids(limit=0) == []
    assert store.list_ids(100, 0) == ids  # positional, as in the protocol
    pages = [store.list_ids(limit=5, offset=o) for o in range(0, 15, 5)]
    assert [i for page in pages for i in page] == ids


def test_default_page_is_100_ids(store):
    for i in range(103):
        store.put(LearnerState.from_dict({"learner_id": f"p{i:03d}"}))
    assert len(store.list_ids()) == 100
    assert store.list_ids(offset=100) == ["p100", "p101", "p102"]
    assert store.count() == 103


def test_idempotency_records_alone_do_not_list_a_learner(store):
    store.record_attempt("ghost", "k1", {})
    assert store.list_ids() == []
    assert store.count() == 0


@pytest.mark.parametrize(("limit", "offset"), [(-1, 0), (5, -1), (1.5, 0), ("5", 0), (5, None)])
def test_bad_paging_is_refused(store, limit, offset):
    with pytest.raises(ValueError):
        store.list_ids(limit, offset)


# --------------------------------------------------------------------------- idempotency records


def test_record_seen_and_get_attempt(store):
    result = {"predicted_p_correct": 0.3775, "new_mastery": 0.4839, "parts": [1, 2, {"k": None}], "ok": True}
    assert store.seen_attempt("pupil-1", "k1") is False
    store.record_attempt("pupil-1", "k1", result)
    assert store.seen_attempt("pupil-1", "k1") is True
    assert store.get_attempt("pupil-1", "k1") == result
    assert store.seen_attempt("pupil-1", "k2") is False
    assert store.get_attempt("pupil-1", "k2") is None


def test_first_record_wins(store):
    store.record_attempt("pupil-1", "k1", {"n": 1})
    store.record_attempt("pupil-1", "k1", {"n": 2})
    assert store.get_attempt("pupil-1", "k1") == {"n": 1}


def test_keys_belong_to_one_learner(store):
    store.record_attempt("pupil-1", "k1", {"who": 1})
    assert store.seen_attempt("pupil-2", "k1") is False
    store.record_attempt("pupil-2", "k1", {"who": 2})
    assert store.get_attempt("pupil-1", "k1") == {"who": 1}
    assert store.get_attempt("pupil-2", "k1") == {"who": 2}


def test_attempt_result_is_a_copy(store):
    result = {"items": [1, 2]}
    store.record_attempt("pupil-1", "k1", result)
    result["items"].append(3)
    first = store.get_attempt("pupil-1", "k1")
    first["items"].append(4)
    assert store.get_attempt("pupil-1", "k1") == {"items": [1, 2]}


@pytest.mark.parametrize("bad", [None, [1, 2], "text", {"when": datetime(2026, 1, 1)}, {"x": float("inf")}, {"s": {1}}])
def test_result_that_is_not_a_json_safe_dict_is_refused(store, bad):
    with pytest.raises(ValueError):
        store.record_attempt("pupil-1", "k1", bad)
    with pytest.raises(ValueError):
        store.put_with_attempt(v2_state(), "k1", bad)
    assert store.seen_attempt("pupil-1", "k1") is False
    assert store.get("pupil-1") is None


def test_put_with_attempt_writes_both(store):
    assert store.put_with_attempt(v2_state(step=1), "k1", {"step": 1}) == 1
    assert store.put_with_attempt(v2_state(step=2), "k2", {"step": 2}, expected_version=1) == 2
    assert store.get_with_version("pupil-1")[0].step == 2
    assert store.get_attempt("pupil-1", "k1") == {"step": 1}
    assert store.get_attempt("pupil-1", "k2") == {"step": 2}


def test_put_with_attempt_conflict_writes_neither(store):
    store.put(v2_state(step=1))
    with pytest.raises(VersionConflict):
        store.put_with_attempt(v2_state(step=2), "k1", {"step": 2}, expected_version=0)
    assert store.get_with_version("pupil-1") == (v2_state(step=1), 1)
    assert store.seen_attempt("pupil-1", "k1") is False


def test_put_with_attempt_on_a_recorded_key_writes_nothing(store):
    store.put_with_attempt(v2_state(step=1), "k1", {"step": 1})
    with pytest.raises(AttemptAlreadyRecorded) as err:
        store.put_with_attempt(v2_state(step=2), "k1", {"step": 2}, expected_version=1)
    assert err.value.result == {"step": 1}
    assert (err.value.learner_id, err.value.idempotency_key) == ("pupil-1", "k1")
    assert store.get_with_version("pupil-1") == (v2_state(step=1), 1)
    assert store.get_attempt("pupil-1", "k1") == {"step": 1}
    with pytest.raises(AttemptAlreadyRecorded):  # the key check comes before the version check
        store.put_with_attempt(v2_state(step=2), "k1", {"step": 2}, expected_version=9)


def test_prune_attempts_drops_only_old_records(store, clock):
    store.put_with_attempt(v2_state("pupil-1", step=1), "old-1", {"n": 1})
    store.record_attempt("pupil-2", "old-2", {"n": 2})
    clock.tick(3600)
    store.record_attempt("pupil-1", "new-1", {"n": 3})
    clock.tick(60)
    assert store.prune_attempts(max_age_seconds=7200) == 0
    assert store.prune_attempts(max_age_seconds=600) == 2
    assert store.seen_attempt("pupil-1", "old-1") is False
    assert store.seen_attempt("pupil-2", "old-2") is False
    assert store.get_attempt("pupil-1", "new-1") == {"n": 3}
    assert store.get("pupil-1").step == 1  # states are never pruned
    assert store.prune_attempts(max_age_seconds=600) == 0
    assert store.prune_attempts(max_age_seconds=0) == 1
    with pytest.raises(ValueError):
        store.prune_attempts(-1)


# --------------------------------------------------------------------------- apply_attempt


def test_apply_attempt_creates_then_updates(store):
    first = apply_attempt(store, "pupil-1", "k1", bump)
    assert isinstance(first, AttemptOutcome)
    assert (first.result, first.replayed, first.version) == ({"step": 1}, False, 1)
    assert first.state.learner_id == "pupil-1" and first.state.step == 1
    second = apply_attempt(store, "pupil-1", "k2", bump)
    assert (second.result, second.replayed, second.version) == ({"step": 2}, False, 2)
    assert store.get_with_version("pupil-1")[0].step == 2


def test_retried_attempt_returns_the_first_result_and_is_not_applied_twice(store):
    calls = []

    def fn(state):
        calls.append(state.step)
        return bump(state)

    first = apply_attempt(store, "pupil-1", "k1", fn)
    retry = apply_attempt(store, "pupil-1", "k1", fn)
    again = apply_attempt(store, "pupil-1", "k1", fn)
    assert first.result == retry.result == again.result == {"step": 1}
    assert (first.replayed, retry.replayed, again.replayed) == (False, True, True)
    assert (retry.state, retry.version) == (None, None)
    assert calls == [0]  # fn ran once
    assert store.get_with_version("pupil-1") == (first.state, 1)
    other = apply_attempt(store, "pupil-2", "k1", fn)  # the same key for another learner is a new attempt
    assert other.replayed is False and other.result == {"step": 1}


def test_apply_attempt_without_a_key_has_no_idempotency(store):
    assert apply_attempt(store, "pupil-1", None, bump).result == {"step": 1}
    assert apply_attempt(store, "pupil-1", None, bump).result == {"step": 2}
    assert store.get_with_version("pupil-1")[1] == 2


def test_apply_attempt_result_is_what_a_retry_will_get(store):
    """A tuple becomes a list in JSON; the first caller must already see the stored form."""
    first = apply_attempt(store, "pupil-1", "k1", lambda state: {"pair": (1, 2)})
    assert first.result == {"pair": [1, 2]}
    assert apply_attempt(store, "pupil-1", "k1", bump).result == first.result


def test_apply_attempt_fn_may_return_nothing(store):
    def quiet(state):
        state.step += 1

    assert apply_attempt(store, "pupil-1", "k1", quiet).result == {}
    assert apply_attempt(store, "pupil-1", "k1", quiet).replayed is True
    assert store.get("pupil-1").step == 1


def test_apply_attempt_uses_new_state_for_a_new_learner_only(store):
    made = []

    def new_state(learner_id):
        made.append(learner_id)
        return LearnerState.from_dict({"learner_id": learner_id, "bundle_version": "math-1.0.0", "step": 10})

    assert apply_attempt(store, "pupil-1", "k1", bump, new_state=new_state).result == {"step": 11}
    assert apply_attempt(store, "pupil-1", "k2", bump, new_state=new_state).result == {"step": 12}
    assert made == ["pupil-1"]
    assert store.get("pupil-1").bundle_version == "math-1.0.0"


def test_apply_attempt_writes_nothing_when_fn_fails(store):
    def broken(state):
        state.step += 1
        raise KeyError("unknown indicator 'B9.9.9.9.9'")

    with pytest.raises(KeyError):
        apply_attempt(store, "pupil-1", "k1", broken)
    assert store.get("pupil-1") is None
    assert store.seen_attempt("pupil-1", "k1") is False
    with pytest.raises(ValueError):  # a result JSON cannot hold
        apply_attempt(store, "pupil-1", "k1", lambda state: {"bad": {1, 2}})
    assert store.get("pupil-1") is None
    assert apply_attempt(store, "pupil-1", "k1", bump).result == {"step": 1}  # the key is still free


def test_apply_attempt_retries_after_a_conflict(store):
    store.put(v2_state(step=10))
    calls = []

    def fn(state):
        calls.append(state.step)
        if len(calls) <= 2:  # another writer gets in between, twice
            rival = store.get("pupil-1")
            rival.step += 100
            store.put(rival)
        return bump(state)

    outcome = apply_attempt(store, "pupil-1", "k1", fn)
    assert calls == [10, 110, 210]  # every retry starts from the fresh state
    assert outcome.result == {"step": 211}
    assert store.get_with_version("pupil-1") == (outcome.state, 4)


def test_apply_attempt_gives_up_after_max_retries(store):
    store.put(v2_state(step=0))
    calls = []

    def fn(state):
        calls.append(state.step)
        store.put(store.get("pupil-1"))  # a rival write every single time
        return bump(state)

    with pytest.raises(VersionConflict):
        apply_attempt(store, "pupil-1", "k1", fn, max_retries=3)
    assert len(calls) == 4
    assert store.get("pupil-1").step == 0
    assert store.seen_attempt("pupil-1", "k1") is False
    calls.clear()
    with pytest.raises(VersionConflict):
        apply_attempt(store, "pupil-1", None, fn, max_retries=0)
    assert len(calls) == 1
    with pytest.raises(ValueError):
        apply_attempt(store, "pupil-1", "k1", bump, max_retries=-1)


def test_apply_attempt_sees_a_rival_with_the_same_key(store):
    """Two requests with one key race: the loser must hand back the winner's result."""

    def fn(state):
        if not store.seen_attempt("pupil-1", "k1"):
            rival = LearnerState.from_dict({"learner_id": "pupil-1", "step": 1})
            store.put_with_attempt(rival, "k1", {"step": 1, "by": "rival"})
        return bump(state)

    outcome = apply_attempt(store, "pupil-1", "k1", fn)
    assert outcome.replayed is True
    assert outcome.result == {"step": 1, "by": "rival"}
    assert store.get_with_version("pupil-1")[1] == 1


def test_apply_attempt_refuses_bad_ids_and_a_changed_learner_id(store):
    with pytest.raises(ValueError):
        apply_attempt(store, "", "k1", bump)
    with pytest.raises(ValueError):
        apply_attempt(store, "pupil-1", "  ", bump)

    def swap(state):
        state.learner_id = "pupil-2"
        return {}

    with pytest.raises(ValueError):
        apply_attempt(store, "pupil-1", "k1", swap)
    assert store.count() == 0


# --------------------------------------------------------------------------- threads


def run_threads(n, target):
    """Start n threads at the same moment, wait for them, and re-raise the first error."""
    start = threading.Barrier(n)
    errors = []

    def body(i):
        try:
            start.wait(timeout=30)
            target(i)
        except BaseException as err:  # noqa: BLE001
            errors.append(err)

    threads = [threading.Thread(target=body, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert not any(t.is_alive() for t in threads)
    if errors:
        raise errors[0]


def test_many_threads_on_one_learner_lose_no_update(store):
    threads, each = 8, 25
    seen = [[] for _ in range(threads)]

    def work(i):
        for j in range(each):
            outcome = apply_attempt(store, "pupil-1", f"t{i}-a{j}", bump, max_retries=10_000)
            assert outcome.replayed is False
            seen[i].append(outcome.result["step"])

    run_threads(threads, work)
    state, version = store.get_with_version("pupil-1")
    assert state.step == threads * each
    assert version == threads * each
    steps = sorted(step for per_thread in seen for step in per_thread)
    assert steps == list(range(1, threads * each + 1))  # every attempt got its own step: no lost update
    for per_thread in seen:
        assert per_thread == sorted(per_thread)
    assert all(store.get_attempt("pupil-1", f"t{i}-a{j}") == {"step": seen[i][j]} for i in range(8) for j in range(25))


def test_many_threads_retrying_the_same_keys_apply_each_key_once(store):
    threads, keys = 8, 20
    results = [{} for _ in range(threads)]

    def work(i):
        for j in range(keys):
            results[i][j] = apply_attempt(store, "pupil-1", f"key-{j}", bump, max_retries=10_000)

    run_threads(threads, work)
    assert store.get("pupil-1").step == keys
    for j in range(keys):
        outcomes = [results[i][j] for i in range(threads)]
        assert sum(1 for o in outcomes if not o.replayed) == 1  # exactly one thread applied the key
        assert len({json.dumps(o.result, sort_keys=True) for o in outcomes}) == 1  # and all got its result


def test_threads_on_different_learners_do_not_conflict(store):
    threads, each = 8, 20

    def work(i):
        for j in range(each):
            apply_attempt(store, f"pupil-{i}", f"a{j}", bump, max_retries=0)  # no retry needed: nobody shares

    run_threads(threads, work)
    assert store.count() == threads
    assert all(store.get_with_version(f"pupil-{i}")[1] == each for i in range(threads))


def test_concurrent_creates_with_expected_version_zero_let_one_through(store):
    winners, losers = [], []

    def work(i):
        try:
            store.put(LearnerState.from_dict({"learner_id": "pupil-1", "step": i}), expected_version=0)
            winners.append(i)
        except VersionConflict:
            losers.append(i)

    run_threads(12, work)
    assert len(winners) == 1 and len(losers) == 11
    assert store.get_with_version("pupil-1")[0].step == winners[0]
    assert store.get_with_version("pupil-1")[1] == 1


def test_concurrent_delete_and_attempts_leave_a_consistent_store(store):
    """Deleting while attempts arrive: afterwards the version equals the number of attempts kept."""

    def work(i):
        for j in range(20):
            if i == 0 and j % 5 == 4:
                store.delete("pupil-1")
            else:
                apply_attempt(store, "pupil-1", f"t{i}-a{j}", bump, max_retries=10_000)

    run_threads(6, work)
    found = store.get_with_version("pupil-1")
    if found is not None:
        state, version = found
        assert state.step == version  # every kept write was one step
    store.delete("pupil-1")
    assert store.get("pupil-1") is None
    assert not any(store.seen_attempt("pupil-1", f"t{i}-a{j}") for i in range(6) for j in range(20))


# --------------------------------------------------------------------------- life cycle and imports


def test_closed_store_refuses_work(kind, tmp_path):
    store = make_store(kind, tmp_path)
    store.put(v2_state())
    store.close()
    store.close()  # twice is fine
    with pytest.raises(StoreClosed):
        store.get("pupil-1")
    with pytest.raises(StoreClosed):
        store.put(v2_state())
    with pytest.raises(StoreClosed):
        store.count()


def test_store_is_a_context_manager(kind, tmp_path):
    with make_store(kind, tmp_path) as store:
        store.put(v2_state())
        assert store.count() == 1
    with pytest.raises(StoreClosed):
        store.count()


def test_store_imports_with_the_core_only():
    """`kwan.store` must load with the standard library and networkx: no optional package."""
    code = (
        "import sys, kwan.store\n"
        "heavy = {'pandas', 'numpy', 'sklearn', 'yaml', 'pdfplumber', 'fastapi', 'pydantic', 'starlette',\n"
        "         'streamlit', 'pyvis', 'httpx', 'sentence_transformers', 'psycopg', 'psycopg2', 'sqlalchemy'}\n"
        "print(sorted(heavy & {name.split('.')[0] for name in sys.modules}))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=120)
    assert out.stdout.strip() == "[]"
