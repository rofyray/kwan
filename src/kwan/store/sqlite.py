"""A learner-state store in one SQLite file (standard library `sqlite3` only).

What is in the file, and nothing more:

- `schema_version(version)`: one row, the layout version, so later releases can migrate.
- `learners(learner_id, state, version, updated_at)`: the opaque id, the state as JSON text, the
  version for optimistic concurrency and the time of the last write.
- `attempts(learner_id, idempotency_key, result, created_at)`: idempotency records.

Several threads may share one `SQLiteStore`, and several processes may each open their own
`SQLiteStore` on the same file: every write runs in a `BEGIN IMMEDIATE` transaction, file
databases use WAL mode, and a writer waits up to `timeout` seconds for another writer to finish.
A store carried across `os.fork()` opens a fresh connection in the child.

Privacy: a new database file is created readable by its owner only, `secure_delete` is on so
removed rows are overwritten with zeros, and after `delete` the write-ahead log is folded back
into the file (when no other connection is in the way) so the removed data does not linger there.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any

from kwan.learner.elo import LearnerState
from kwan.store.base import (
    AttemptAlreadyRecorded,
    Clock,
    StoreClosed,
    StoreSchemaError,
    VersionConflict,
    check_expected_version,
    check_id,
    check_page,
    decode_result,
    decode_state,
    encode_result,
    encode_state,
    stamp,
    utc_now,
)

__all__ = ["SCHEMA_VERSION", "SQLiteStore"]

# One tuple of statements per layout version; version N is reached by running the first N tuples.
# To change the layout later, append a tuple. Never edit one that has shipped.
_MIGRATIONS: tuple[tuple[str, ...], ...] = (
    (
        """CREATE TABLE learners (
               learner_id TEXT    NOT NULL PRIMARY KEY,
               state      TEXT    NOT NULL,
               version    INTEGER NOT NULL CHECK (version >= 1),
               updated_at TEXT    NOT NULL
           )""",
        """CREATE TABLE attempts (
               learner_id      TEXT NOT NULL,
               idempotency_key TEXT NOT NULL,
               result          TEXT NOT NULL,
               created_at      TEXT NOT NULL,
               PRIMARY KEY (learner_id, idempotency_key)
           ) WITHOUT ROWID""",
        "CREATE INDEX attempts_created_at ON attempts (created_at)",
    ),
)
SCHEMA_VERSION = len(_MIGRATIONS)

_MEMORY = ":memory:"


class SQLiteStore:
    """`LearnerStore` kept in one SQLite file, or in memory with `path=":memory:"`.

    `timeout` is how long a write waits for another process that is writing. `clock` returns the
    current time as a timezone-aware `datetime`; tests pass a fixed one. The tables are created
    on first use; a file written by a newer Kwan raises `StoreSchemaError`."""

    def __init__(
        self,
        path: str | os.PathLike[str] = _MEMORY,
        *,
        timeout: float = 30.0,
        clock: Clock | None = None,
    ) -> None:
        self._path = os.fspath(path)
        if not self._path:
            raise ValueError("path must be a file path or ':memory:'")
        self._in_memory = self._path == _MEMORY
        self._timeout = float(timeout)
        self._clock: Clock = clock or utc_now
        self._lock = threading.RLock()
        self._closed = False
        self._journal_mode = "memory"
        self._orphans: list[sqlite3.Connection] = []  # connections inherited across a fork: never touched again
        self._conn: sqlite3.Connection | None = self._open()
        _LIVE.add(self)

    @property
    def path(self) -> str:
        """The database file path, or `":memory:"`."""
        return self._path

    @property
    def journal_mode(self) -> str:
        """The SQLite journal mode in use: `"wal"` for a file database."""
        return self._journal_mode

    # ------------------------------------------------------------------ states

    def get(self, learner_id: str) -> LearnerState | None:
        """The stored state, or None when the learner has none."""
        found = self.get_with_version(learner_id)
        return None if found is None else found[0]

    def get_with_version(self, learner_id: str) -> tuple[LearnerState, int] | None:
        """The stored state and its version, or None when the learner has none."""
        check_id(learner_id)
        row = self._one("SELECT state, version FROM learners WHERE learner_id = ?", (learner_id,))
        return None if row is None else (decode_state(row[0]), int(row[1]))

    def put(self, state: LearnerState, expected_version: int | None = None) -> int:
        """Write the state and return its new version; see `LearnerStore.put`."""
        learner_id = check_id(state.learner_id)
        check_expected_version(expected_version)
        text = encode_state(state)
        with self._transaction() as conn:
            return self._write(conn, learner_id, text, expected_version)

    def put_with_attempt(
        self,
        state: LearnerState,
        idempotency_key: str,
        result: dict[str, Any],
        expected_version: int | None = None,
    ) -> int:
        """Write the state and record the attempt result in one transaction; see `LearnerStore`."""
        learner_id = check_id(state.learner_id)
        check_id(idempotency_key, "idempotency_key")
        check_expected_version(expected_version)
        text, result_text = encode_state(state), encode_result(result)
        with self._transaction() as conn:
            first = conn.execute(
                "SELECT result FROM attempts WHERE learner_id = ? AND idempotency_key = ?",
                (learner_id, idempotency_key),
            ).fetchone()
            if first is not None:
                raise AttemptAlreadyRecorded(learner_id, idempotency_key, decode_result(first[0]))
            version = self._write(conn, learner_id, text, expected_version)
            conn.execute(
                "INSERT INTO attempts (learner_id, idempotency_key, result, created_at) VALUES (?, ?, ?, ?)",
                (learner_id, idempotency_key, result_text, stamp(self._clock())),
            )
            return version

    def delete(self, learner_id: str) -> bool:
        """Remove the learner's state and idempotency records. True when something was removed."""
        check_id(learner_id)
        with self._lock:
            with self._transaction() as conn:
                attempts = conn.execute("DELETE FROM attempts WHERE learner_id = ?", (learner_id,)).rowcount
                states = conn.execute("DELETE FROM learners WHERE learner_id = ?", (learner_id,)).rowcount
            removed = attempts + states > 0
            if removed:
                self._fold_wal()
        return removed

    def list_ids(self, limit: int = 100, offset: int = 0) -> list[str]:
        """Learner ids that have a stored state, in ascending order, one page at a time."""
        check_page(limit, offset)
        with self._lock:
            rows = (
                self._connection()
                .execute("SELECT learner_id FROM learners ORDER BY learner_id LIMIT ? OFFSET ?", (limit, offset))
                .fetchall()
            )
        return [str(row[0]) for row in rows]

    def count(self) -> int:
        """Number of learners that have a stored state."""
        row = self._one("SELECT COUNT(*) FROM learners", ())
        return int(row[0]) if row is not None else 0

    def updated_at(self, learner_id: str) -> str | None:
        """Time of the learner's last state write (ISO 8601, UTC), or None when not stored."""
        check_id(learner_id)
        row = self._one("SELECT updated_at FROM learners WHERE learner_id = ?", (learner_id,))
        return None if row is None else str(row[0])

    # ------------------------------------------------------------------ idempotency records

    def seen_attempt(self, learner_id: str, idempotency_key: str) -> bool:
        """True when a result is recorded for this learner and key."""
        check_id(learner_id)
        check_id(idempotency_key, "idempotency_key")
        row = self._one(
            "SELECT 1 FROM attempts WHERE learner_id = ? AND idempotency_key = ?", (learner_id, idempotency_key)
        )
        return row is not None

    def record_attempt(self, learner_id: str, idempotency_key: str, result: dict[str, Any]) -> None:
        """Record the result for this learner and key; the first record wins."""
        check_id(learner_id)
        check_id(idempotency_key, "idempotency_key")
        result_text = encode_result(result)
        with self._transaction() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO attempts (learner_id, idempotency_key, result, created_at) VALUES (?, ?, ?, ?)",
                (learner_id, idempotency_key, result_text, stamp(self._clock())),
            )

    def get_attempt(self, learner_id: str, idempotency_key: str) -> dict[str, Any] | None:
        """The result recorded for this learner and key, or None."""
        check_id(learner_id)
        check_id(idempotency_key, "idempotency_key")
        row = self._one(
            "SELECT result FROM attempts WHERE learner_id = ? AND idempotency_key = ?", (learner_id, idempotency_key)
        )
        return None if row is None else decode_result(row[0])

    def prune_attempts(self, max_age_seconds: float) -> int:
        """Drop idempotency records older than `max_age_seconds`; returns how many were dropped."""
        if max_age_seconds < 0:
            raise ValueError("max_age_seconds must be 0 or more")
        cutoff = stamp(self._clock() - timedelta(seconds=max_age_seconds))
        with self._transaction() as conn:
            return int(conn.execute("DELETE FROM attempts WHERE created_at < ?", (cutoff,)).rowcount)

    # ------------------------------------------------------------------ life cycle

    def close(self) -> None:
        """Close the database connection. The store cannot be used afterwards."""
        with self._lock:
            self._closed = True
            conn, self._conn = self._conn, None
            if conn is not None:
                conn.close()

    def __enter__(self) -> SQLiteStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ internals

    def _open(self) -> sqlite3.Connection:
        """A connection with the pragmas set and the tables in place."""
        if not self._in_memory:
            _create_private_file(self._path)
        conn = sqlite3.connect(self._path, timeout=self._timeout, isolation_level=None, check_same_thread=False)
        try:
            conn.execute("PRAGMA secure_delete = ON")
            if not self._in_memory:
                self._journal_mode = _enable_wal(conn, self._timeout)
                conn.execute("PRAGMA synchronous = NORMAL")
            _migrate(conn)
        except sqlite3.DatabaseError as err:
            conn.close()
            if isinstance(err, sqlite3.OperationalError):
                raise
            raise StoreSchemaError(f"{self._path} is not a Kwan learner store: {err}") from err
        except BaseException:
            conn.close()
            raise
        return conn

    def _connection(self) -> sqlite3.Connection:
        """The live connection. Call with the lock held."""
        if self._closed:
            raise StoreClosed("the store is closed")
        if self._conn is None:  # first use in a forked child
            self._conn = self._open()
        return self._conn

    def _one(self, sql: str, params: tuple[Any, ...]) -> Any:
        """First row of a read, or None."""
        with self._lock:
            return self._connection().execute(sql, params).fetchone()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """A write transaction: commits on success, rolls back when the body raises."""
        with self._lock:
            conn = self._connection()
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
                conn.execute("COMMIT")
            except BaseException:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise

    def _write(self, conn: sqlite3.Connection, learner_id: str, text: str, expected_version: int | None) -> int:
        """Version check and write. Call inside a transaction."""
        row = conn.execute("SELECT version FROM learners WHERE learner_id = ?", (learner_id,)).fetchone()
        current = 0 if row is None else int(row[0])
        if expected_version is not None and expected_version != current:
            raise VersionConflict(learner_id, expected_version, current)
        conn.execute(
            """INSERT INTO learners (learner_id, state, version, updated_at) VALUES (?, ?, ?, ?)
               ON CONFLICT (learner_id) DO UPDATE
               SET state = excluded.state, version = excluded.version, updated_at = excluded.updated_at""",
            (learner_id, text, current + 1, stamp(self._clock())),
        )
        return current + 1

    def _fold_wal(self) -> None:
        """After a delete: move the write-ahead log into the main file and empty it, so the removed
        rows do not stay readable there. Best effort: skipped when another connection is busy."""
        if self._journal_mode != "wal":
            return
        conn = self._connection()
        try:
            conn.execute("PRAGMA busy_timeout = 250")
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()
        except sqlite3.Error:
            pass
        finally:
            conn.execute(f"PRAGMA busy_timeout = {int(self._timeout * 1000)}")

    def _reset_after_fork(self) -> None:
        """In a forked child: never touch the parent's connection or lock again."""
        self._lock = threading.RLock()
        if not self._in_memory and self._conn is not None:
            self._orphans.append(self._conn)  # keep it referenced so it is never closed from the child
            self._conn = None


# --------------------------------------------------------------------------- module helpers

_LIVE: weakref.WeakSet[SQLiteStore] = weakref.WeakSet()


def _reset_stores_after_fork() -> None:
    for store in list(_LIVE):
        store._reset_after_fork()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_stores_after_fork)


def _create_private_file(path: str) -> None:
    """Create the database file (and its folder) when missing, readable by its owner only.
    SQLite gives the `-wal` and `-shm` side files the same permissions as the main file."""
    file = Path(path)
    if file.exists():
        return
    file.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.close(os.open(file, os.O_RDWR | os.O_CREAT, 0o600))
    except OSError:
        pass  # sqlite3.connect reports the real problem


def _enable_wal(conn: sqlite3.Connection, timeout: float) -> str:
    """Switch the database to WAL mode and return the mode in use. Another connection that is
    starting up at the same moment can hold the file for an instant, so a locked file is retried."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            row = conn.execute("PRAGMA journal_mode = WAL").fetchone()
            return str(row[0]).lower()
        except sqlite3.OperationalError as err:
            if "locked" not in str(err) or time.monotonic() >= deadline:
                raise
            time.sleep(0.01)


def _stored_schema_version(conn: sqlite3.Connection) -> int:
    """The layout version recorded in the database; 0 for a new or foreign database."""
    try:
        row = conn.execute("SELECT version FROM schema_version").fetchone()
    except sqlite3.OperationalError:
        return 0
    return 0 if row is None else int(row[0])


def _migrate(conn: sqlite3.Connection) -> None:
    """Bring the database to `SCHEMA_VERSION`, one transaction for the whole move."""
    current = _stored_schema_version(conn)
    if current == SCHEMA_VERSION:
        return
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
        row = conn.execute("SELECT version FROM schema_version").fetchone()
        current = 0 if row is None else int(row[0])  # read again: another process may have migrated meanwhile
        if current > SCHEMA_VERSION:
            raise StoreSchemaError(
                f"the database has layout version {current} but this Kwan knows up to {SCHEMA_VERSION}; upgrade Kwan"
            )
        for statements in _MIGRATIONS[current:]:
            for statement in statements:
                conn.execute(statement)
        if row is None:
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
        else:
            conn.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION,))
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
