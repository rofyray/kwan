"""A learner-state store in PostgreSQL (optional).

Kwan does not depend on a PostgreSQL driver. This store needs `psycopg` (version 3), which is
imported only when a `PostgresStore` is created, so `import kwan.store` works without it:

    uv add 'kwan[postgres]'        (or: uv sync --extra postgres in a source checkout)

The tables hold the same things as the SQLite store, and nothing more:

- `kwan_schema_version(version)`: one row, the layout version.
- `kwan_learners(learner_id, state, version, updated_at)`: the opaque id, the state as JSON text,
  the version for optimistic concurrency and the time of the last write.
- `kwan_attempts(learner_id, idempotency_key, result, created_at)`: idempotency records.

A store holds one connection and is safe to share between threads. Several processes may each
open their own store on the same database: writes for one learner are put in line with a
transaction-level advisory lock, so version checks and idempotency checks never race. Open the
store after a fork, not before. `delete` removes the rows at once; PostgreSQL frees the space
they used at its next vacuum.
"""

from __future__ import annotations

import importlib
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import timedelta
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

__all__ = ["SCHEMA_VERSION", "PostgresStore"]

# One tuple of statements per layout version, as in the SQLite store: append, never edit.
# Ids and timestamps use the "C" collation so that order is plain byte order on every server.
_MIGRATIONS: tuple[tuple[str, ...], ...] = (
    (
        """CREATE TABLE kwan_learners (
               learner_id TEXT   COLLATE "C" PRIMARY KEY,
               state      TEXT   NOT NULL,
               version    BIGINT NOT NULL CHECK (version >= 1),
               updated_at TEXT   COLLATE "C" NOT NULL
           )""",
        """CREATE TABLE kwan_attempts (
               learner_id      TEXT COLLATE "C" NOT NULL,
               idempotency_key TEXT COLLATE "C" NOT NULL,
               result          TEXT NOT NULL,
               created_at      TEXT COLLATE "C" NOT NULL,
               PRIMARY KEY (learner_id, idempotency_key)
           )""",
        "CREATE INDEX kwan_attempts_created_at ON kwan_attempts (created_at)",
    ),
)
SCHEMA_VERSION = len(_MIGRATIONS)

_LOCK_SPACE = 0x4B57414E  # "KWAN": first key of every advisory lock this store takes
_LEARNER_LOCK = "SELECT pg_advisory_xact_lock(%s::int4, hashtext(%s))"
_SCHEMA_LOCK = "SELECT pg_advisory_xact_lock(%s::int4, 0)"


def _load_driver() -> Any:
    """The `psycopg` module, or a clear error when it is not installed."""
    try:
        psycopg = importlib.import_module("psycopg")
    except ImportError as err:
        raise ImportError(
            "PostgresStore needs the 'psycopg' package (version 3), which Kwan does not install. "
            'Install it with: uv add "kwan[postgres]" (or uv sync --extra postgres in a source checkout). '
            "MemoryStore and SQLiteStore work without it."
        ) from err
    return psycopg


class PostgresStore:
    """`LearnerStore` kept in a PostgreSQL database.

    `conninfo` is a connection string such as `"postgresql://user@host/dbname"` or
    `"dbname=kwan"`. `clock` returns the current time as a timezone-aware `datetime`. The tables
    are created on first use; a database written by a newer Kwan raises `StoreSchemaError`."""

    def __init__(self, conninfo: str, *, clock: Clock | None = None) -> None:
        self._driver = _load_driver()
        self._conninfo = conninfo
        self._clock: Clock = clock or utc_now
        self._lock = threading.RLock()
        self._closed = False
        self._conn: Any = self._open()

    # ------------------------------------------------------------------ states

    def get(self, learner_id: str) -> LearnerState | None:
        """The stored state, or None when the learner has none."""
        found = self.get_with_version(learner_id)
        return None if found is None else found[0]

    def get_with_version(self, learner_id: str) -> tuple[LearnerState, int] | None:
        """The stored state and its version, or None when the learner has none."""
        check_id(learner_id)
        row = self._one("SELECT state, version FROM kwan_learners WHERE learner_id = %s", (learner_id,))
        return None if row is None else (decode_state(row[0]), int(row[1]))

    def put(self, state: LearnerState, expected_version: int | None = None) -> int:
        """Write the state and return its new version; see `LearnerStore.put`."""
        learner_id = check_id(state.learner_id)
        check_expected_version(expected_version)
        text = encode_state(state)
        with self._transaction(learner_id) as conn:
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
        with self._transaction(learner_id) as conn:
            first = conn.execute(
                "SELECT result FROM kwan_attempts WHERE learner_id = %s AND idempotency_key = %s",
                (learner_id, idempotency_key),
            ).fetchone()
            if first is not None:
                raise AttemptAlreadyRecorded(learner_id, idempotency_key, decode_result(first[0]))
            version = self._write(conn, learner_id, text, expected_version)
            conn.execute(
                "INSERT INTO kwan_attempts (learner_id, idempotency_key, result, created_at) VALUES (%s, %s, %s, %s)",
                (learner_id, idempotency_key, result_text, stamp(self._clock())),
            )
            return version

    def delete(self, learner_id: str) -> bool:
        """Remove the learner's state and idempotency records. True when something was removed."""
        check_id(learner_id)
        with self._transaction(learner_id) as conn:
            attempts = conn.execute("DELETE FROM kwan_attempts WHERE learner_id = %s", (learner_id,)).rowcount
            states = conn.execute("DELETE FROM kwan_learners WHERE learner_id = %s", (learner_id,)).rowcount
        return bool(attempts + states > 0)

    def list_ids(self, limit: int = 100, offset: int = 0) -> list[str]:
        """Learner ids that have a stored state, in ascending order, one page at a time."""
        check_page(limit, offset)
        with self._lock:
            rows = (
                self._connection()
                .execute("SELECT learner_id FROM kwan_learners ORDER BY learner_id LIMIT %s OFFSET %s", (limit, offset))
                .fetchall()
            )
        return [str(row[0]) for row in rows]

    def count(self) -> int:
        """Number of learners that have a stored state."""
        row = self._one("SELECT COUNT(*) FROM kwan_learners", ())
        return int(row[0]) if row is not None else 0

    def updated_at(self, learner_id: str) -> str | None:
        """Time of the learner's last state write (ISO 8601, UTC), or None when not stored."""
        check_id(learner_id)
        row = self._one("SELECT updated_at FROM kwan_learners WHERE learner_id = %s", (learner_id,))
        return None if row is None else str(row[0])

    # ------------------------------------------------------------------ idempotency records

    def seen_attempt(self, learner_id: str, idempotency_key: str) -> bool:
        """True when a result is recorded for this learner and key."""
        check_id(learner_id)
        check_id(idempotency_key, "idempotency_key")
        row = self._one(
            "SELECT 1 FROM kwan_attempts WHERE learner_id = %s AND idempotency_key = %s", (learner_id, idempotency_key)
        )
        return row is not None

    def record_attempt(self, learner_id: str, idempotency_key: str, result: dict[str, Any]) -> None:
        """Record the result for this learner and key; the first record wins."""
        check_id(learner_id)
        check_id(idempotency_key, "idempotency_key")
        result_text = encode_result(result)
        with self._transaction(learner_id) as conn:
            conn.execute(
                """INSERT INTO kwan_attempts (learner_id, idempotency_key, result, created_at)
                   VALUES (%s, %s, %s, %s) ON CONFLICT (learner_id, idempotency_key) DO NOTHING""",
                (learner_id, idempotency_key, result_text, stamp(self._clock())),
            )

    def get_attempt(self, learner_id: str, idempotency_key: str) -> dict[str, Any] | None:
        """The result recorded for this learner and key, or None."""
        check_id(learner_id)
        check_id(idempotency_key, "idempotency_key")
        row = self._one(
            "SELECT result FROM kwan_attempts WHERE learner_id = %s AND idempotency_key = %s",
            (learner_id, idempotency_key),
        )
        return None if row is None else decode_result(row[0])

    def prune_attempts(self, max_age_seconds: float) -> int:
        """Drop idempotency records older than `max_age_seconds`; returns how many were dropped."""
        if max_age_seconds < 0:
            raise ValueError("max_age_seconds must be 0 or more")
        cutoff = stamp(self._clock() - timedelta(seconds=max_age_seconds))
        with self._transaction() as conn:
            return int(conn.execute("DELETE FROM kwan_attempts WHERE created_at < %s", (cutoff,)).rowcount)

    # ------------------------------------------------------------------ life cycle

    def close(self) -> None:
        """Close the database connection. The store cannot be used afterwards."""
        with self._lock:
            self._closed = True
            conn, self._conn = self._conn, None
            if conn is not None:
                conn.close()

    def __enter__(self) -> PostgresStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ internals

    def _open(self) -> Any:
        """A connection in autocommit mode with the tables in place."""
        conn = self._driver.connect(self._conninfo, autocommit=True)
        try:
            _migrate(conn)
        except BaseException:
            conn.close()
            raise
        return conn

    def _connection(self) -> Any:
        """The live connection, opened again when the server dropped it. Call with the lock held."""
        if self._closed:
            raise StoreClosed("the store is closed")
        if self._conn is None or self._conn.closed:
            self._conn = self._open()
        return self._conn

    def _one(self, sql: str, params: tuple[Any, ...]) -> Any:
        """First row of a read, or None."""
        with self._lock:
            return self._connection().execute(sql, params).fetchone()

    @contextmanager
    def _transaction(self, learner_id: str | None = None) -> Iterator[Any]:
        """A write transaction: commits on success, rolls back when the body raises. With a
        learner id, it first waits for any other write to the same learner to finish."""
        with self._lock:
            conn = self._connection()
            with conn.transaction():
                if learner_id is not None:
                    conn.execute(_LEARNER_LOCK, (_LOCK_SPACE, learner_id))
                yield conn

    def _write(self, conn: Any, learner_id: str, text: str, expected_version: int | None) -> int:
        """Version check and write. Call inside a transaction that holds the learner's lock."""
        row = conn.execute("SELECT version FROM kwan_learners WHERE learner_id = %s", (learner_id,)).fetchone()
        current = 0 if row is None else int(row[0])
        if expected_version is not None and expected_version != current:
            raise VersionConflict(learner_id, expected_version, current)
        conn.execute(
            """INSERT INTO kwan_learners (learner_id, state, version, updated_at) VALUES (%s, %s, %s, %s)
               ON CONFLICT (learner_id) DO UPDATE
               SET state = EXCLUDED.state, version = EXCLUDED.version, updated_at = EXCLUDED.updated_at""",
            (learner_id, text, current + 1, stamp(self._clock())),
        )
        return current + 1


def _migrate(conn: Any) -> None:
    """Bring the database to `SCHEMA_VERSION` in one transaction; one process at a time."""
    with conn.transaction():
        conn.execute(_SCHEMA_LOCK, (_LOCK_SPACE,))
        conn.execute("CREATE TABLE IF NOT EXISTS kwan_schema_version (version INTEGER NOT NULL)")
        row = conn.execute("SELECT version FROM kwan_schema_version").fetchone()
        current = 0 if row is None else int(row[0])
        if current == SCHEMA_VERSION:
            return
        if current > SCHEMA_VERSION:
            raise StoreSchemaError(
                f"the database has layout version {current} but this Kwan knows up to {SCHEMA_VERSION}; upgrade Kwan"
            )
        for statements in _MIGRATIONS[current:]:
            for statement in statements:
                conn.execute(statement)
        if row is None:
            conn.execute("INSERT INTO kwan_schema_version (version) VALUES (%s)", (SCHEMA_VERSION,))
        else:
            conn.execute("UPDATE kwan_schema_version SET version = %s", (SCHEMA_VERSION,))
