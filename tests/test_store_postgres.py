"""The optional PostgreSQL store. Nothing here needs a server or the `psycopg` driver, except the
tests at the end, which run only when `KWAN_TEST_POSTGRES_DSN` points at a throwaway database."""

import os
import subprocess
import sys
import threading

import pytest

from kwan.learner.elo import LearnerState
from kwan.store import LearnerStore, PostgresStore, StoreSchemaError, VersionConflict, apply_attempt
from kwan.store import postgres as postgres_module

DSN = os.environ.get("KWAN_TEST_POSTGRES_DSN")
needs_server = pytest.mark.skipif(not DSN, reason="set KWAN_TEST_POSTGRES_DSN to a throwaway PostgreSQL database")


def bump(state):
    state.step += 1
    return {"step": state.step}


def test_missing_driver_gives_a_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "psycopg", None)  # makes `import psycopg` fail, installed or not
    with pytest.raises(ImportError) as err:
        PostgresStore("dbname=kwan")
    message = str(err.value)
    assert "psycopg" in message and "pip install" in message
    assert "SQLiteStore" in message  # and it says what works without the driver
    assert "—" not in message


def test_importing_the_store_package_does_not_load_the_driver():
    code = "import sys, kwan.store, kwan.store.postgres; print('psycopg' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=120)
    assert out.stdout.strip() == "False"


def test_postgres_layout_matches_the_sqlite_layout():
    """Same tables and columns as the SQLite store (with a `kwan_` prefix): nothing extra is kept."""
    from kwan.store import sqlite as sqlite_module

    def columns(statement):
        body = statement[statement.index("(") + 1 :]
        names = [line.split()[0] for line in body.splitlines() if line.strip()]
        return [n for n in names if n.islower()]  # drops PRIMARY, ")" and other keywords

    assert postgres_module.SCHEMA_VERSION == sqlite_module.SCHEMA_VERSION == 1
    lite, post = sqlite_module._MIGRATIONS[0], postgres_module._MIGRATIONS[0]
    assert columns(post[0]) == columns(lite[0]) == ["learner_id", "state", "version", "updated_at"]
    assert columns(post[1]) == columns(lite[1]) == ["learner_id", "idempotency_key", "result", "created_at"]
    assert sum(s.lstrip().startswith("CREATE TABLE") for s in post) == 2


# --------------------------------------------------------------------------- opt-in: a real server


@pytest.fixture
def open_store():
    opened = []

    def opener(**kwargs):
        store = PostgresStore(DSN, **kwargs)
        opened.append(store)
        return store

    first = opener()
    with first._transaction() as conn:
        conn.execute("TRUNCATE kwan_learners, kwan_attempts")
    yield opener
    for store in opened:
        store.close()


@needs_server
def test_server_holds_only_the_expected_tables_and_columns(open_store):
    store = open_store()
    assert isinstance(store, LearnerStore)
    apply_attempt(store, "pupil-1", "k1", bump)
    with store._transaction() as conn:
        rows = conn.execute(
            "SELECT table_name, column_name FROM information_schema.columns "
            "WHERE table_name LIKE 'kwan\\_%' ORDER BY table_name, ordinal_position"
        ).fetchall()
    layout = {}
    for table, column in rows:
        layout.setdefault(table, []).append(column)
    assert layout == {
        "kwan_attempts": ["learner_id", "idempotency_key", "result", "created_at"],
        "kwan_learners": ["learner_id", "state", "version", "updated_at"],
        "kwan_schema_version": ["version"],
    }


@needs_server
def test_two_stores_on_one_server_see_each_other_and_detect_conflicts(open_store):
    one, two = open_store(), open_store()
    assert one.put(LearnerState.from_dict({"learner_id": "pupil-1", "step": 1})) == 1
    assert two.get_with_version("pupil-1")[0].step == 1
    assert two.put(LearnerState.from_dict({"learner_id": "pupil-1", "step": 2}), expected_version=1) == 2
    with pytest.raises(VersionConflict) as err:
        one.put(LearnerState.from_dict({"learner_id": "pupil-1", "step": 9}), expected_version=1)
    assert (err.value.expected, err.value.actual) == (1, 2)
    first = apply_attempt(one, "pupil-1", "k1", bump)
    retry = apply_attempt(two, "pupil-1", "k1", bump)
    assert (first.replayed, retry.replayed, retry.result) == (False, True, {"step": 3})
    assert two.delete("pupil-1") is True
    assert one.get("pupil-1") is None and one.seen_attempt("pupil-1", "k1") is False


@needs_server
def test_store_reconnects_after_the_connection_is_lost(open_store):
    store = open_store()
    apply_attempt(store, "pupil-1", "k1", bump)
    store._conn.close()  # as if the server had dropped the connection
    assert store.get("pupil-1").step == 1
    assert apply_attempt(store, "pupil-1", "k2", bump).version == 2


@needs_server
def test_database_from_a_newer_kwan_is_refused(open_store):
    store = open_store()
    with store._transaction() as conn:
        conn.execute("UPDATE kwan_schema_version SET version = %s", (postgres_module.SCHEMA_VERSION + 1,))
    try:
        with pytest.raises(StoreSchemaError):
            PostgresStore(DSN)
    finally:
        with store._transaction() as conn:
            conn.execute("UPDATE kwan_schema_version SET version = %s", (postgres_module.SCHEMA_VERSION,))


@needs_server
def test_threads_each_with_their_own_connection_lose_no_update(open_store):
    """Six connections write to one learner and all retry the same shared keys: the advisory lock
    and the version check together must let every attempt through exactly once."""
    threads, each = 6, 15
    stores = [open_store() for _ in range(threads)]
    start = threading.Barrier(threads)
    errors = []

    def work(i):
        try:
            start.wait(timeout=30)
            for j in range(each):
                apply_attempt(stores[i], "pupil-1", f"t{i}-a{j}", bump, max_retries=10_000)
                apply_attempt(stores[i], "pupil-1", f"shared-a{j}", bump, max_retries=10_000)
        except BaseException as err:  # noqa: BLE001
            errors.append(err)

    workers = [threading.Thread(target=work, args=(i,)) for i in range(threads)]
    for t in workers:
        t.start()
    for t in workers:
        t.join(timeout=300)
    assert not errors, errors[0]
    expected = threads * each + each
    state, version = stores[0].get_with_version("pupil-1")
    assert (state.step, version) == (expected, expected)


@needs_server
def test_stores_opened_at_the_same_moment_create_the_tables_once(open_store):
    first = open_store()
    with first._transaction() as conn:
        conn.execute("DROP TABLE kwan_learners, kwan_attempts, kwan_schema_version")
    start = threading.Barrier(5)
    errors = []

    def work():
        try:
            start.wait(timeout=30)
            open_store().count()
        except BaseException as err:  # noqa: BLE001
            errors.append(err)

    workers = [threading.Thread(target=work) for _ in range(5)]
    for t in workers:
        t.start()
    for t in workers:
        t.join(timeout=120)
    assert not errors, errors[0]
    with first._transaction() as conn:
        assert conn.execute("SELECT version FROM kwan_schema_version").fetchall() == [(postgres_module.SCHEMA_VERSION,)]
