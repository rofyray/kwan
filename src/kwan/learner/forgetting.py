"""Forgetting (PRD 6.2): time helpers and the decay of a skill offset toward its prior.

    theta_eff = prior + (theta_k - prior) * exp(-days / tau)

`days` is the time since the skill was last practised. The functions here are pure; `EloLearner`
decides which prior and which timestamps go in (see the notes at the top of `kwan.learner.elo`).
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kwan.learner.elo import EloLearner, LearnerState

DAY = 86400.0  # seconds
TAU_DAYS = 45.0  # default time constant of forgetting
STREAK_EXPIRY_DAYS = 30.0  # default life of streak mastery

When = datetime | str  # a timezone-aware datetime or ISO 8601 text; naive values are read as UTC


@lru_cache(maxsize=1 << 16)
def _text_seconds(text: str) -> float:
    d = datetime.fromisoformat(text)
    return (d if d.tzinfo is not None else d.replace(tzinfo=UTC)).timestamp()


def to_seconds(when: When) -> float:
    """Seconds since 1970-01-01 UTC for a datetime or an ISO 8601 string. Naive values are UTC.

    Raises TypeError for any other type and ValueError for text that is not ISO 8601."""
    if isinstance(when, datetime):
        return (when if when.tzinfo is not None else when.replace(tzinfo=UTC)).timestamp()
    if isinstance(when, str):
        return _text_seconds(when)
    raise TypeError(f"a time must be a datetime or an ISO 8601 string, got {when!r}")


def to_iso(when: When) -> str:
    """ISO 8601 text for a timestamp. Text is checked and kept as given; a datetime is formatted."""
    if isinstance(when, datetime):
        return when.isoformat()
    to_seconds(when)
    return when


def days_between(earlier: When, later: When) -> float:
    """Days from `earlier` to `later`, never below zero (a time in the past of `earlier` counts as 0)."""
    return max(0.0, (to_seconds(later) - to_seconds(earlier)) / DAY)


def retention(days: float, tau_days: float | None = TAU_DAYS) -> float:
    """Share of the learned offset still there after `days`: exp(-days / tau), in 0 to 1.

    1.0 when `days` is zero or negative, or when forgetting is off (`tau_days` None or infinite)."""
    if tau_days is None or days <= 0.0:
        return 1.0
    return math.exp(-days / tau_days)


def decay_toward(theta_k: float, prior: float, days: float, tau_days: float | None = TAU_DAYS) -> float:
    """The effective offset: `theta_k` pulled toward `prior` by the forgetting curve.

    Zero days gives `theta_k` back exactly; a very long gap gives `prior`."""
    w = retention(days, tau_days)
    if w == 1.0:
        return theta_k
    return prior + (theta_k - prior) * w


def effective_mastery_map(learner: EloLearner, st: LearnerState, now: When | None = None) -> dict[str, float]:
    """Mastery of every skill of the learner model's graph at `now`, in one fast pass.

    Same numbers as `learner.mastery(st, n, now=now)` for each node; this is the call for anything that
    needs all of them (the recommender does on every request)."""
    return learner.effective_mastery_map(st, now)
