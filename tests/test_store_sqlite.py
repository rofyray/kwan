"""What only the SQLite store has to prove: the file layout (privacy), persistence, several
instances and several processes on one file, the schema version, and what `delete` leaves behind."""

import json
import os
import sqlite3
import stat
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from kwan.learner.elo import LearnerState
from kwan.store import SQLiteStore, StoreSchemaError, VersionConflict, apply_attempt
from kwan.store.sqlite import SCHEMA_VERSION

SRC = Path(__file__).resolve().parent.parent / "src"


def state_of(learner_id="pupil-1", step=0):
    return LearnerState.from_dict({"learner_id": learner_id, "step": step, "bundle_version": "math-1.0.0"})


def bump(state):
    state.step += 1
    return {"step": state.step}


@pytest.fixture
def db(tmp_path):
    return tmp_path / "learners.db"


@pytest.fixture
def open_store(db):
    """Open any number of stores on the same file; all are closed at the end."""
    opened = []

    def opener(path=db, **kwargs):
        store = SQLiteStore(path, **kwargs)
        opened.append(store)
        return store

    yield opener
    for store in opened:
        store.close()


def raw_rows(path, sql):
    """Read the file with plain sqlite3, the way an auditor would."""
    conn = sqlite3.connect(path)
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def all_bytes(path):
    """Every byte SQLite keeps for this database: the main file and its side files."""
    return b"".join(p.read_bytes() for p in sorted(Path(path).parent.glob(Path(path).name + "*")))


# --------------------------------------------------------------------------- privacy: what is in the file


def test_file_holds_only_the_expected_tables_and_columns(db, open_store):
    store = open_store()
    apply_attempt(store, "pupil-1", "k1", bump)
    tables = {name for (name,) in raw_rows(db, "SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert tables == {"schema_version", "learners", "attempts"}
    columns = {t: [row[1] for row in raw_rows(db, f"PRAGMA table_info({t})")] for t in tables}
    assert columns == {
        "schema_version": ["version"],
        "learners": ["learner_id", "state", "version", "updated_at"],
        "attempts": ["learner_id", "idempotency_key", "result", "created_at"],
    }
    assert raw_rows(db, "SELECT name FROM sqlite_master WHERE type IN ('view', 'trigger')") == []


def test_learner_row_holds_the_id_the_state_json_the_version_and_a_timestamp(db, open_store):
    store = open_store()
    apply_attempt(store, "pupil-1", "k1", bump)
    apply_attempt(store, "pupil-1", "k2", bump)
    ((learner_id, state_text, version, updated_at),) = raw_rows(db, "SELECT * FROM learners")
    assert (learner_id, version) == ("pupil-1", 2)
    assert isinstance(state_text, str)
    assert json.loads(state_text) == store.get("pupil-1").to_dict()  # exactly the state, nothing added
    assert updated_at == store.updated_at("pupil-1")
    assert updated_at.endswith("+00:00") and "T" in updated_at
    attempts = raw_rows(db, "SELECT learner_id, idempotency_key, result FROM attempts ORDER BY idempotency_key")
    assert attempts == [("pupil-1", "k1", '{"step":1}'), ("pupil-1", "k2", '{"step":2}')]


def test_state_text_has_only_the_learner_state_keys(db, open_store):
    """The stored JSON is `LearnerState.to_dict()`: the store adds no field of its own."""
    store = open_store()
    state = state_of("pupil-1", step=4)
    store.put(state)
    ((text,),) = raw_rows(db, "SELECT state FROM learners")
    assert set(json.loads(text)) == set(state.to_dict())


@pytest.mark.skipif(os.name != "posix", reason="POSIX file permissions")
def test_new_database_file_is_private_to_its_owner(db, open_store):
    store = open_store()
    store.put(state_of())
    for file in db.parent.glob(db.name + "*"):  # the main file and the -wal and -shm side files
        assert stat.S_IMODE(file.stat().st_mode) & 0o077 == 0, file.name


def test_delete_leaves_no_trace_of_the_learner_in_the_files(db, open_store):
    store = open_store()
    secret_id, secret_key = "pupil-QZXJ-48151623", "request-WVKP-271828"
    store.put(state_of("someone-else", step=1))
    for step in range(1, 40):  # enough writes for the state to have passed through the write-ahead log
        store.put(state_of(secret_id, step=step))
    store.put_with_attempt(state_of(secret_id, step=40), secret_key, {"marker": "MRKR-314159"})
    assert secret_id.encode() in all_bytes(db)
    assert store.delete(secret_id) is True
    left = all_bytes(db)
    assert secret_id.encode() not in left
    assert secret_key.encode() not in left
    assert b"MRKR-314159" not in left
    assert b"someone-else" in left  # the other learner is still there
    assert store.get("someone-else").step == 1


# --------------------------------------------------------------------------- file, WAL, schema version


def test_file_database_uses_wal_and_memory_database_does_not(db, open_store):
    store = open_store()
    assert store.journal_mode == "wal"
    assert store.path == str(db)
    assert raw_rows(db, "PRAGMA journal_mode") == [("wal",)]
    memory = open_store(":memory:")
    assert memory.journal_mode == "memory"
    assert memory.path == ":memory:"


def test_two_memory_stores_are_separate(open_store):
    one, two = open_store(":memory:"), open_store(":memory:")
    one.put(state_of())
    assert two.get("pupil-1") is None
    assert not Path(":memory:").exists()


def test_missing_folders_are_created(tmp_path, open_store):
    path = tmp_path / "var" / "kwan" / "learners.db"
    store = open_store(path)
    store.put(state_of())
    assert path.is_file()


def test_path_may_be_a_string_and_must_not_be_empty(db, open_store):
    assert open_store(str(db)).put(state_of()) == 1
    with pytest.raises(ValueError):
        SQLiteStore("")


def test_schema_version_is_recorded_once(db, open_store):
    open_store().put(state_of())
    open_store()  # opening again must not add a row or run the migrations twice
    assert raw_rows(db, "SELECT version FROM schema_version") == [(SCHEMA_VERSION,)]
    assert SCHEMA_VERSION == 1


def test_database_from_a_newer_kwan_is_refused(db, open_store):
    store = open_store()
    store.put(state_of())
    store.close()
    conn = sqlite3.connect(db)
    conn.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION + 1,))
    conn.commit()
    conn.close()
    with pytest.raises(StoreSchemaError):
        SQLiteStore(db)
    assert raw_rows(db, "SELECT COUNT(*) FROM learners") == [(1,)]  # and it was left alone


def test_file_that_is_not_a_database_is_refused(tmp_path):
    path = tmp_path / "notes.db"
    path.write_bytes(b"this is not an SQLite file, it is only text " * 40)
    with pytest.raises(StoreSchemaError):
        SQLiteStore(path)
    assert path.read_bytes().startswith(b"this is not an SQLite file")


def test_older_layout_is_migrated_in_place(db, monkeypatch, open_store):
    """A later release appends a migration step; an existing file must move up and keep its rows."""
    from kwan.store import sqlite as sqlite_module

    store = open_store()
    apply_attempt(store, "pupil-1", "k1", bump)
    store.close()
    step_two = ("CREATE TABLE added_later (x INTEGER)",)
    monkeypatch.setattr(sqlite_module, "_MIGRATIONS", sqlite_module._MIGRATIONS + (step_two,))
    monkeypatch.setattr(sqlite_module, "SCHEMA_VERSION", 2)
    upgraded = open_store()
    assert raw_rows(db, "SELECT version FROM schema_version") == [(2,)]
    assert raw_rows(db, "SELECT name FROM sqlite_master WHERE name = 'added_later'") == [("added_later",)]
    assert upgraded.get_with_version("pupil-1")[1] == 1
    assert upgraded.get_attempt("pupil-1", "k1") == {"step": 1}
    fresh = open_store(db.with_name("fresh.db"))  # and a new file gets both steps at once
    assert fresh.count() == 0
    assert raw_rows(db.with_name("fresh.db"), "SELECT version FROM schema_version") == [(2,)]


def test_failed_migration_changes_nothing(db, monkeypatch, open_store):
    from kwan.store import sqlite as sqlite_module

    store = open_store()
    store.put(state_of())
    store.close()
    broken = ("CREATE TABLE half_done (x INTEGER)", "THIS IS NOT SQL")
    monkeypatch.setattr(sqlite_module, "_MIGRATIONS", sqlite_module._MIGRATIONS + (broken,))
    monkeypatch.setattr(sqlite_module, "SCHEMA_VERSION", 2)
    with pytest.raises(sqlite3.OperationalError):
        SQLiteStore(db)
    assert raw_rows(db, "SELECT version FROM schema_version") == [(1,)]
    assert raw_rows(db, "SELECT name FROM sqlite_master WHERE name = 'half_done'") == []
    monkeypatch.undo()
    assert open_store().get("pupil-1") is not None


# --------------------------------------------------------------------------- persistence


def test_state_versions_and_idempotency_records_survive_a_reopen(db):
    first = SQLiteStore(db)
    apply_attempt(first, "pupil-1", "k1", bump)
    apply_attempt(first, "pupil-1", "k2", bump)
    first.put(state_of("pupil-2", step=7))
    stamp = first.updated_at("pupil-1")
    before = first.get("pupil-1").to_dict()
    first.close()

    second = SQLiteStore(db)
    try:
        state, version = second.get_with_version("pupil-1")
        assert (state.to_dict(), version) == (before, 2)
        assert second.updated_at("pupil-1") == stamp
        assert second.list_ids() == ["pupil-1", "pupil-2"]
        assert second.count() == 2
        replay = apply_attempt(second, "pupil-1", "k1", bump)  # a retry after a restart is still a retry
        assert replay.replayed is True and replay.result == {"step": 1}
        assert apply_attempt(second, "pupil-1", "k3", bump).version == 3
        with pytest.raises(VersionConflict):
            second.put(state_of("pupil-2"), expected_version=0)
    finally:
        second.close()


def test_delete_survives_a_reopen(db):
    with SQLiteStore(db) as first:
        apply_attempt(first, "pupil-1", "k1", bump)
        first.delete("pupil-1")
    with SQLiteStore(db) as second:
        assert second.get("pupil-1") is None
        assert second.seen_attempt("pupil-1", "k1") is False
        assert second.count() == 0


def test_row_written_in_the_v1_layout_is_read(db, open_store):
    """A row whose JSON is a v1 state (bare codes, no v2 fields) reads as a v2 state."""
    store = open_store()
    v1 = {"learner_id": "old-pupil", "theta": 0.5, "theta_k": {"B1.1.1.1.1": 0.25}, "n": {"B1.1.1.1.1": 2, "_all": 2}}
    v1 |= {"last_seen": {"B1.1.1.1.1": 1}, "streak": {"B1.1.1.1.1": 2}, "step": 2}
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO learners (learner_id, state, version, updated_at) VALUES (?, ?, 5, ?)",
        ("old-pupil", json.dumps(v1), "2025-01-01T00:00:00.000000+00:00"),
    )
    conn.commit()
    conn.close()
    state, version = store.get_with_version("old-pupil")
    assert version == 5
    assert state.to_dict() == LearnerState.from_dict(v1).to_dict()
    assert apply_attempt(store, "old-pupil", "k1", bump).version == 6
    assert store.get("old-pupil").theta_k == {"B1.1.1.1.1": 0.25}


# --------------------------------------------------------------------------- several stores on one file


def test_two_stores_on_one_file_see_each_other(open_store):
    one, two = open_store(), open_store()
    assert one.put(state_of(step=1)) == 1
    assert two.get_with_version("pupil-1")[0].step == 1
    assert two.put(state_of(step=2), expected_version=1) == 2
    assert one.get_with_version("pupil-1") == (state_of(step=2), 2)
    one.record_attempt("pupil-1", "k1", {"from": "one"})
    assert two.get_attempt("pupil-1", "k1") == {"from": "one"}
    two.record_attempt("pupil-1", "k1", {"from": "two"})  # first record wins across stores too
    assert one.get_attempt("pupil-1", "k1") == {"from": "one"}
    assert (one.count(), two.count()) == (1, 1)
    assert two.delete("pupil-1") is True
    assert one.get("pupil-1") is None
    assert one.seen_attempt("pupil-1", "k1") is False


def test_two_stores_on_one_file_detect_conflicts(open_store):
    one, two = open_store(), open_store()
    one.put(state_of(step=1))
    _, seen_by_one = one.get_with_version("pupil-1")
    _, seen_by_two = two.get_with_version("pupil-1")
    assert two.put(state_of(step=20), expected_version=seen_by_two) == 2
    with pytest.raises(VersionConflict) as err:
        one.put(state_of(step=10), expected_version=seen_by_one)
    assert (err.value.expected, err.value.actual) == (1, 2)
    assert one.get("pupil-1").step == 20
    assert one.put(state_of("pupil-2"), expected_version=0) == 1  # both try to create the same learner
    with pytest.raises(VersionConflict):
        two.put(state_of("pupil-2"), expected_version=0)
    assert two.get_with_version("pupil-2")[1] == 1


def test_the_same_key_through_two_stores_is_applied_once(open_store):
    one, two = open_store(), open_store()
    first = apply_attempt(one, "pupil-1", "k1", bump)
    retry = apply_attempt(two, "pupil-1", "k1", bump)
    assert (first.replayed, retry.replayed) == (False, True)
    assert retry.result == first.result == {"step": 1}
    assert one.get("pupil-1").step == 1


def test_threads_each_with_their_own_store_lose_no_update(open_store):
    threads, each = 6, 20
    stores = [open_store() for _ in range(threads)]
    start = threading.Barrier(threads)
    errors = []

    def work(i):
        try:
            start.wait(timeout=30)
            for j in range(each):
                apply_attempt(stores[i], "pupil-1", f"t{i}-a{j}", bump, max_retries=10_000)
        except BaseException as err:  # noqa: BLE001
            errors.append(err)

    workers = [threading.Thread(target=work, args=(i,)) for i in range(threads)]
    for t in workers:
        t.start()
    for t in workers:
        t.join(timeout=120)
    assert not errors, errors[0]
    state, version = stores[0].get_with_version("pupil-1")
    assert (state.step, version) == (threads * each, threads * each)


WORKER = """
import sys
from kwan.store import SQLiteStore, apply_attempt

def bump(state):
    state.step += 1
    return {"step": state.step}

path, name, count = sys.argv[1], sys.argv[2], int(sys.argv[3])
with SQLiteStore(path) as store:
    for j in range(count):
        apply_attempt(store, "pupil-1", f"{name}-a{j}", bump, max_retries=10_000)
        apply_attempt(store, "pupil-1", f"shared-a{j}", bump, max_retries=10_000)
print("done")
"""


def test_several_processes_on_one_file_lose_no_update(db, open_store):
    """Three processes open the file at the same moment (so they also race to create the tables),
    each applies its own attempts and all of them retry the same shared keys."""
    processes, each = 3, 15
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(SRC), os.environ.get("PYTHONPATH", "")])}
    running = [
        subprocess.Popen(
            [sys.executable, "-c", WORKER, str(db), f"p{i}", str(each)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        for i in range(processes)
    ]
    for proc in running:
        out, err = proc.communicate(timeout=300)
        assert proc.returncode == 0, err
        assert out.strip() == "done"
    store = open_store()
    state, version = store.get_with_version("pupil-1")
    expected = processes * each + each  # every own attempt once, every shared key once
    assert (state.step, version) == (expected, expected)
    assert raw_rows(db, "SELECT COUNT(*) FROM attempts") == [(expected,)]
    assert raw_rows(db, "SELECT version FROM schema_version") == [(SCHEMA_VERSION,)]
    assert raw_rows(db, "PRAGMA integrity_check") == [("ok",)]


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
@pytest.mark.filterwarnings("ignore:.*fork.*:DeprecationWarning")
def test_store_carried_across_a_fork_works_in_parent_and_child(open_store):
    store = open_store()
    apply_attempt(store, "pupil-1", "before", bump)
    pid = os.fork()
    if pid == 0:  # the child: use the inherited store object, then leave without running pytest's teardown
        code = 1
        try:
            outcome = apply_attempt(store, "pupil-1", "child", bump, max_retries=1000)
            code = 0 if outcome.result == {"step": 2} else 2
        finally:
            os._exit(code)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    assert store.get("pupil-1").step == 2  # the parent's connection still works and sees the child's write
    assert apply_attempt(store, "pupil-1", "after", bump).result == {"step": 3}
    assert store.get_attempt("pupil-1", "child") == {"step": 2}


def test_writer_waits_for_another_writer_and_gives_up_after_the_timeout(db, open_store):
    patient, hasty = open_store(), open_store(timeout=0.2)
    blocker = sqlite3.connect(db, isolation_level=None, check_same_thread=False)
    try:
        blocker.execute("BEGIN IMMEDIATE")  # some other process is in the middle of a write
        assert patient.get("pupil-1") is None  # readers are not blocked in WAL mode
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            hasty.put(state_of())
        release = threading.Timer(0.3, lambda: blocker.execute("COMMIT"))
        release.start()
        assert patient.put(state_of()) == 1  # waits for the other writer instead of failing
        release.join()
    finally:
        blocker.close()
    assert hasty.put(state_of(step=1), expected_version=1) == 2  # and the hasty store is still usable
