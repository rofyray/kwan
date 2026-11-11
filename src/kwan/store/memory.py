"""A learner-state store that lives in the memory of one process.

Good for tests, notebooks, the demo and a single-process service that may lose state on restart.
It keeps the same things as `SQLiteStore` in the same form (JSON text, version, time of the last
write, idempotency records), so both behave the same way.
"""

from __future__ import annotations

import threading
from datetime import timedelta
from typing import Any

from kwan.learner.elo import LearnerState
from kwan.store.base import (
    AttemptAlreadyRecorded,
    Clock,
    StoreClosed,
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

__all__ = ["MemoryStore"]


class MemoryStore:
    """Thread-safe in-memory `LearnerStore`.

    `clock` returns the current time as a timezone-aware `datetime`; tests pass a fixed one."""

    def __init__(self, clock: Clock | None = None) -> None:
        self._clock: Clock = clock or utc_now
        self._lock = threading.RLock()
        self._closed = False
        # learner id -> (state JSON, version, time of last write)
        self._states: dict[str, tuple[str, int, str]] = {}
        # learner id -> idempotency key -> (result JSON, time recorded)
        self._attempts: dict[str, dict[str, tuple[str, str]]] = {}

    # ------------------------------------------------------------------ states

    def get(self, learner_id: str) -> LearnerState | None:
        """The stored state, or None when the learner has none."""
        found = self.get_with_version(learner_id)
        return None if found is None else found[0]

    def get_with_version(self, learner_id: str) -> tuple[LearnerState, int] | None:
        """The stored state and its version, or None when the learner has none."""
        check_id(learner_id)
        with self._lock:
            self._check_open()
            row = self._states.get(learner_id)
        return None if row is None else (decode_state(row[0]), row[1])

    def put(self, state: LearnerState, expected_version: int | None = None) -> int:
        """Write the state and return its new version; see `LearnerStore.put`."""
        learner_id = check_id(state.learner_id)
        check_expected_version(expected_version)
        text = encode_state(state)
        with self._lock:
            self._check_open()
            return self._write(learner_id, text, expected_version)

    def put_with_attempt(
        self,
        state: LearnerState,
        idempotency_key: str,
        result: dict[str, Any],
        expected_version: int | None = None,
    ) -> int:
        """Write the state and record the attempt result as one step; see `LearnerStore`."""
        learner_id = check_id(state.learner_id)
        check_id(idempotency_key, "idempotency_key")
        check_expected_version(expected_version)
        text, result_text = encode_state(state), encode_result(result)
        with self._lock:
            self._check_open()
            first = self._attempts.get(learner_id, {}).get(idempotency_key)
            if first is not None:
                raise AttemptAlreadyRecorded(learner_id, idempotency_key, decode_result(first[0]))
            version = self._write(learner_id, text, expected_version)
            self._attempts.setdefault(learner_id, {})[idempotency_key] = (result_text, stamp(self._clock()))
            return version

    def delete(self, learner_id: str) -> bool:
        """Remove the learner's state and idempotency records. True when something was removed."""
        check_id(learner_id)
        with self._lock:
            self._check_open()
            had_state = self._states.pop(learner_id, None) is not None
            had_attempts = self._attempts.pop(learner_id, None) is not None
        return had_state or had_attempts

    def list_ids(self, limit: int = 100, offset: int = 0) -> list[str]:
        """Learner ids that have a stored state, in ascending order, one page at a time."""
        check_page(limit, offset)
        with self._lock:
            self._check_open()
            ids = sorted(self._states)
        return ids[offset : offset + limit]

    def count(self) -> int:
        """Number of learners that have a stored state."""
        with self._lock:
            self._check_open()
            return len(self._states)

    def updated_at(self, learner_id: str) -> str | None:
        """Time of the learner's last state write (ISO 8601, UTC), or None when not stored."""
        check_id(learner_id)
        with self._lock:
            self._check_open()
            row = self._states.get(learner_id)
        return None if row is None else row[2]

    # ------------------------------------------------------------------ idempotency records

    def seen_attempt(self, learner_id: str, idempotency_key: str) -> bool:
        """True when a result is recorded for this learner and key."""
        check_id(learner_id)
        check_id(idempotency_key, "idempotency_key")
        with self._lock:
            self._check_open()
            return idempotency_key in self._attempts.get(learner_id, {})

    def record_attempt(self, learner_id: str, idempotency_key: str, result: dict[str, Any]) -> None:
        """Record the result for this learner and key; the first record wins."""
        check_id(learner_id)
        check_id(idempotency_key, "idempotency_key")
        result_text = encode_result(result)
        with self._lock:
            self._check_open()
            records = self._attempts.setdefault(learner_id, {})
            if idempotency_key not in records:
                records[idempotency_key] = (result_text, stamp(self._clock()))

    def get_attempt(self, learner_id: str, idempotency_key: str) -> dict[str, Any] | None:
        """The result recorded for this learner and key, or None."""
        check_id(learner_id)
        check_id(idempotency_key, "idempotency_key")
        with self._lock:
            self._check_open()
            row = self._attempts.get(learner_id, {}).get(idempotency_key)
        return None if row is None else decode_result(row[0])

    def prune_attempts(self, max_age_seconds: float) -> int:
        """Drop idempotency records older than `max_age_seconds`; returns how many were dropped."""
        if max_age_seconds < 0:
            raise ValueError("max_age_seconds must be 0 or more")
        dropped = 0
        with self._lock:
            self._check_open()
            cutoff = stamp(self._clock() - timedelta(seconds=max_age_seconds))
            for learner_id in list(self._attempts):
                records = self._attempts[learner_id]
                old = [key for key, (_, recorded_at) in records.items() if recorded_at < cutoff]
                for key in old:
                    del records[key]
                dropped += len(old)
                if not records:
                    del self._attempts[learner_id]
        return dropped

    # ------------------------------------------------------------------ life cycle

    def close(self) -> None:
        """Forget everything. The store cannot be used afterwards."""
        with self._lock:
            self._closed = True
            self._states.clear()
            self._attempts.clear()

    def __enter__(self) -> MemoryStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ internals

    def _check_open(self) -> None:
        if self._closed:
            raise StoreClosed("the store is closed")

    def _write(self, learner_id: str, text: str, expected_version: int | None) -> int:
        """Version check and write. Call with the lock held."""
        row = self._states.get(learner_id)
        current = 0 if row is None else row[1]
        if expected_version is not None and expected_version != current:
            raise VersionConflict(learner_id, expected_version, current)
        self._states[learner_id] = (text, current + 1, stamp(self._clock()))
        return current + 1
