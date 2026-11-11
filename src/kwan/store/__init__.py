"""Learner-state stores.

`LearnerStore` is the contract; `MemoryStore` and `SQLiteStore` implement it, and so does the
optional `PostgresStore` (it needs the `psycopg` package, loaded only when you create one). A store keeps the
opaque learner id, the state as JSON, a version for optimistic concurrency, the time of the last
write and idempotency records for attempts. It keeps nothing else about a pupil.

    from kwan.store import SQLiteStore, apply_attempt

    store = SQLiteStore("kwan.db")
    outcome = apply_attempt(store, "pupil-17", "request-0001", lambda state: {"step": state.step})
"""

from kwan.store.base import (
    MAX_ID_LENGTH,
    AttemptAlreadyRecorded,
    AttemptOutcome,
    LearnerStore,
    StoreClosed,
    StoreError,
    StoreSchemaError,
    VersionConflict,
    apply_attempt,
)
from kwan.store.memory import MemoryStore
from kwan.store.postgres import PostgresStore
from kwan.store.sqlite import SQLiteStore

__all__ = [
    "MAX_ID_LENGTH",
    "AttemptAlreadyRecorded",
    "AttemptOutcome",
    "LearnerStore",
    "MemoryStore",
    "PostgresStore",
    "SQLiteStore",
    "StoreClosed",
    "StoreError",
    "StoreSchemaError",
    "VersionConflict",
    "apply_attempt",
]
