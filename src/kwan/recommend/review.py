"""The review slot (PRD 6.3): bring back one skill the learner mastered and has started to forget.

A skill is due for review when all of these hold:

- the learner answered it at a known time (`state.last_seen_at`), at least `min_days` before `now`;
- it was mastered: its mastery with forgetting ignored is 0.85 or more (three right in a row, or a
  high estimate);
- its effective mastery at `now`, with forgetting applied, is below 0.85.

Among the skills that are due, the one with the lowest effective mastery is chosen; equal values are
settled by the lowest indicator id. The rule needs timestamps: with no `now`, or a state without
`last_seen_at`, nothing is ever due.
"""

from __future__ import annotations

from collections.abc import Callable, Container, Mapping
from datetime import UTC, datetime

from kwan.learner.elo import MASTERY, LearnerState

MIN_DAYS = 7.0  # a skill answered more recently than this is not reviewed yet
_DAY = 86400.0


def as_utc(when: datetime | str) -> datetime:
    """A timezone-aware datetime from a datetime or an ISO 8601 string. Naive times are taken as UTC."""
    if isinstance(when, str):
        when = datetime.fromisoformat(when)
    elif not isinstance(when, datetime):
        raise TypeError(f"expected a datetime or an ISO 8601 string, got {when!r}")
    return when.replace(tzinfo=UTC) if when.tzinfo is None else when


def days_between(earlier: datetime | str, later: datetime | str) -> float:
    """Days from `earlier` to `later`, as a fraction (negative when `earlier` is after `later`)."""
    return (as_utc(later) - as_utc(earlier)).total_seconds() / _DAY


def review_candidate(
    state: LearnerState,
    nodes: Container[str],
    mastery_eff: Mapping[str, float],
    mastery_raw: Callable[[str], float],
    now: datetime | str | None,
    min_days: float = MIN_DAYS,
    allowed: Callable[[str], bool] | None = None,
) -> str | None:
    """The skill to review now, or None.

    nodes: the skills of the graph; ids in the state that are not in it are skipped.
    mastery_eff: mastery of each skill at `now`, with forgetting.
    mastery_raw: mastery of one skill with forgetting ignored (called only for skills that pass the
        other checks).
    allowed: an extra filter, for example the class cap.
    """
    if now is None or not state.last_seen_at:
        return None
    at = as_utc(now)
    best: tuple[float, str] | None = None
    for n, stamp in state.last_seen_at.items():
        if n not in nodes:
            continue
        m = mastery_eff[n]
        if m >= MASTERY or (best is not None and (m, n) >= best):
            continue
        if allowed is not None and not allowed(n):
            continue
        if (at - as_utc(stamp)).total_seconds() < min_days * _DAY:
            continue
        if mastery_raw(n) < MASTERY:
            continue
        best = (m, n)
    return best[1] if best is not None else None


__all__ = ["MIN_DAYS", "as_utc", "days_between", "review_candidate"]
