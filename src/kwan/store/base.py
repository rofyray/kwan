"""The learner-state store contract.

A store keeps one JSON document per learner (the output of `LearnerState.to_dict()`), a version
number for optimistic concurrency, the time of the last write, and idempotency records for
attempts. It holds nothing else: no names, no emails, no free text about a pupil. The learner id
is an opaque string chosen by whoever uses Kwan.

Versions: the first write of a learner gives version 1 and every later write adds 1. A learner
with no stored state has version 0, so `put(state, expected_version=0)` means "create, and fail
if somebody else created it first". After `delete` the learner is gone completely (nothing is
kept behind), so a learner created again starts at version 1.

Idempotency: an API gives every attempt an idempotency key. `put_with_attempt` writes the new
state and the response for that key in one step, so a retried request with the same key gets the
first response back and the answer is never applied twice. `apply_attempt` wraps the whole read,
change, write cycle.

Everything here needs only the standard library.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

from kwan.learner.elo import LearnerState

__all__ = [
    "MAX_ID_LENGTH",
    "AttemptAlreadyRecorded",
    "AttemptOutcome",
    "Clock",
    "LearnerStore",
    "StoreClosed",
    "StoreError",
    "StoreSchemaError",
    "VersionConflict",
    "apply_attempt",
]

MAX_ID_LENGTH = 256  # longest learner id or idempotency key a store accepts

Clock = Callable[[], datetime]


class StoreError(Exception):
    """Base class for store errors."""


class VersionConflict(StoreError):
    """The stored version is not the one the writer expected: somebody else wrote in between.

    `expected` is the version the caller passed, `actual` the version found in the store
    (0 when the learner has no stored state)."""

    def __init__(self, learner_id: str, expected: int, actual: int) -> None:
        super().__init__(f"version conflict for learner {learner_id!r}: expected {expected}, found {actual}")
        self.learner_id = learner_id
        self.expected = expected
        self.actual = actual


class AttemptAlreadyRecorded(StoreError):
    """`put_with_attempt` was called with an idempotency key that already has a result.

    Nothing was written. `result` is the result stored the first time."""

    def __init__(self, learner_id: str, idempotency_key: str, result: dict[str, Any]) -> None:
        super().__init__(f"attempt {idempotency_key!r} of learner {learner_id!r} is already recorded")
        self.learner_id = learner_id
        self.idempotency_key = idempotency_key
        self.result = result


class StoreSchemaError(StoreError):
    """The database was written by a newer Kwan, or is not a Kwan store."""


class StoreClosed(StoreError):
    """The store was used after `close()`."""


@runtime_checkable
class LearnerStore(Protocol):
    """What every learner-state store offers. `MemoryStore`, `SQLiteStore` and the optional
    `PostgresStore` implement it.

    All methods are safe to call from several threads. Ids and keys are non-blank strings of at
    most `MAX_ID_LENGTH` characters; anything else raises `ValueError`. States and results that
    come back are fresh copies: changing them never changes what is stored."""

    def get(self, learner_id: str) -> LearnerState | None:
        """The stored state, or None when the learner has none."""
        ...

    def get_with_version(self, learner_id: str) -> tuple[LearnerState, int] | None:
        """The stored state and its version, or None when the learner has none."""
        ...

    def put(self, state: LearnerState, expected_version: int | None = None) -> int:
        """Write the state and return its new version.

        With `expected_version=None` the write always goes through (last write wins). With a
        number, the write happens only when the stored version equals it (0 = not stored yet);
        otherwise `VersionConflict` is raised and nothing changes."""
        ...

    def put_with_attempt(
        self,
        state: LearnerState,
        idempotency_key: str,
        result: dict[str, Any],
        expected_version: int | None = None,
    ) -> int:
        """`put` and `record_attempt` as one step: both happen or neither does.

        Raises `AttemptAlreadyRecorded` (carrying the first result) when the key is already
        recorded for this learner, and `VersionConflict` as `put` does."""
        ...

    def delete(self, learner_id: str) -> bool:
        """Remove everything stored for the learner: the state and all idempotency records.

        Returns True when something was removed."""
        ...

    def list_ids(self, limit: int = 100, offset: int = 0) -> list[str]:
        """Learner ids that have a stored state, in ascending order, one page at a time."""
        ...

    def count(self) -> int:
        """Number of learners that have a stored state."""
        ...

    def updated_at(self, learner_id: str) -> str | None:
        """Time of the learner's last state write (ISO 8601, UTC), or None when not stored."""
        ...

    def seen_attempt(self, learner_id: str, idempotency_key: str) -> bool:
        """True when a result is recorded for this learner and key."""
        ...

    def record_attempt(self, learner_id: str, idempotency_key: str, result: dict[str, Any]) -> None:
        """Record the result for this learner and key. The first record wins: recording the same
        key again changes nothing."""
        ...

    def get_attempt(self, learner_id: str, idempotency_key: str) -> dict[str, Any] | None:
        """The result recorded for this learner and key, or None."""
        ...

    def prune_attempts(self, max_age_seconds: float) -> int:
        """Drop idempotency records older than `max_age_seconds`; returns how many were dropped.
        Learner states are not touched."""
        ...

    def close(self) -> None:
        """Release what the store holds. Calling it twice is fine."""
        ...


@dataclass(frozen=True)
class AttemptOutcome:
    """What `apply_attempt` did.

    `result` is the dictionary to send back to the caller. `replayed` is True when the key had
    been applied before, in which case `result` is the first result and nothing was written
    (`state` and `version` are then None). Otherwise `state` is the state that was written and
    `version` its new version."""

    result: dict[str, Any]
    replayed: bool
    state: LearnerState | None = None
    version: int | None = None


def apply_attempt(
    store: LearnerStore,
    learner_id: str,
    key: str | None,
    fn: Callable[[LearnerState], dict[str, Any] | None],
    *,
    new_state: Callable[[str], LearnerState] | None = None,
    max_retries: int = 20,
) -> AttemptOutcome:
    """Read the learner's state, let `fn` change it, write it back, exactly once per `key`.

    `fn(state)` changes the state in place (for example with `Kwan.update`) and returns the
    JSON-safe result dictionary to hand back (None means `{}`). When another writer got in
    between, the state is read again and `fn` is called again on the fresh copy, at most
    `max_retries` more times; after that the last `VersionConflict` is raised. So `fn` must do
    nothing but change the state it is given. If `fn` raises, nothing is written.

    With a `key`, a call whose key was applied before returns the first result with
    `replayed=True` and does not call `fn`. With `key=None` there is no idempotency, only the
    safe read, change, write. A learner with no stored state starts from `new_state(learner_id)`
    (default: a blank `LearnerState`)."""
    check_id(learner_id, "learner_id")
    if key is not None:
        check_id(key, "idempotency_key")
    if max_retries < 0:
        raise ValueError("max_retries must be 0 or more")
    conflict: VersionConflict | None = None
    for attempt in range(max_retries + 1):
        if attempt:
            time.sleep(_RNG.uniform(0.0, min(0.02, 0.0005 * 2**attempt)))
        if key is not None:
            first = store.get_attempt(learner_id, key)
            if first is not None:
                return AttemptOutcome(first, True)
        found = store.get_with_version(learner_id)
        if found is None:
            state = new_state(learner_id) if new_state is not None else blank_state(learner_id)
            version = 0
        else:
            state, version = found
        result = decode_result(encode_result(fn(state) or {}))
        if state.learner_id != learner_id:
            raise ValueError(f"fn changed the learner id from {learner_id!r} to {state.learner_id!r}")
        try:
            if key is None:
                new_version = store.put(state, expected_version=version)
            else:
                new_version = store.put_with_attempt(state, key, result, expected_version=version)
        except AttemptAlreadyRecorded as done:
            return AttemptOutcome(done.result, True)
        except VersionConflict as err:
            conflict = err
            continue
        return AttemptOutcome(result, False, state, new_version)
    assert conflict is not None
    raise conflict


# --------------------------------------------------------------------------- helpers for stores

_RNG = random.Random()  # jitter for retry pauses only; never touches engine output


def check_id(value: object, what: str = "learner_id") -> str:
    """Return `value` when it is a usable id or key, else raise `ValueError`."""
    if not isinstance(value, str):
        raise ValueError(f"{what} must be a string, got {type(value).__name__}")
    if not value.strip():
        raise ValueError(f"{what} must not be blank")
    if len(value) > MAX_ID_LENGTH:
        raise ValueError(f"{what} is longer than {MAX_ID_LENGTH} characters")
    if "\x00" in value:
        raise ValueError(f"{what} must not contain a NUL character")
    return value


def check_expected_version(expected_version: object) -> None:
    """Raise `ValueError` unless `expected_version` is None or a whole number of 0 or more."""
    if expected_version is None:
        return
    if isinstance(expected_version, bool) or not isinstance(expected_version, int) or expected_version < 0:
        raise ValueError("expected_version must be None or a whole number of 0 or more")


def check_page(limit: object, offset: object) -> None:
    """Raise `ValueError` unless `limit` and `offset` are whole numbers of 0 or more."""
    for name, value in (("limit", limit), ("offset", offset)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a whole number of 0 or more")


def blank_state(learner_id: str) -> LearnerState:
    """A new state for a learner Kwan has never seen."""
    return LearnerState.from_dict({"learner_id": learner_id})


def encode_state(state: LearnerState) -> str:
    """The state as JSON text. Raises `ValueError` for a state JSON cannot hold (NaN, infinity)."""
    return json.dumps(state.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)


def decode_state(text: str) -> LearnerState:
    """A state from JSON text written by `encode_state` (v1 layouts are read too)."""
    return LearnerState.from_dict(json.loads(text))


def encode_result(result: object) -> str:
    """An attempt result as JSON text. Raises `ValueError` when it is not a JSON-safe dict."""
    if not isinstance(result, dict):
        raise ValueError(f"an attempt result must be a dict, got {type(result).__name__}")
    try:
        return json.dumps(result, separators=(",", ":"), allow_nan=False)
    except TypeError as err:
        raise ValueError(f"an attempt result must be JSON-safe: {err}") from err


def decode_result(text: str) -> dict[str, Any]:
    """An attempt result from JSON text written by `encode_result`."""
    result: dict[str, Any] = json.loads(text)
    return result


def utc_now() -> datetime:
    """The current time in UTC (the default clock of every store)."""
    return datetime.now(UTC)


def stamp(moment: datetime) -> str:
    """ISO 8601 UTC text of fixed width, so that text order is time order."""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC).isoformat(timespec="microseconds")
