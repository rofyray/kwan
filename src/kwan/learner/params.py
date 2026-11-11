"""Tunable parameters of the Elo learner model, as one frozen record (the bundle's `params` block)."""

from __future__ import annotations

import math
import operator
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace
from typing import Any


@dataclass(frozen=True)
class EloParams:
    """Every tunable of `EloLearner`. The defaults are the v1 values plus the v2 forgetting defaults.

    k_s, k_k: step sizes of the global ability and of the per-skill offset.
    decay: how fast both step sizes shrink with the number of answers.
    grade_step, transfer: the prior of an unseen skill (lower for higher grades, plus a share of the
        offsets of the prerequisites already practised).
    streak_to_master: correct answers in a row that count as mastery.
    tau_days: time constant of forgetting in days; None switches forgetting off.
    streak_expiry_days: streak mastery stops counting after this many days unseen; None means never.
    """

    k_s: float = 0.3
    k_k: float = 0.4
    decay: float = 0.05
    grade_step: float = 0.4
    transfer: float = 0.5
    streak_to_master: int = 3
    tau_days: float | None = 45.0
    streak_expiry_days: float | None = 30.0

    def __post_init__(self) -> None:
        for name in ("k_s", "k_k", "decay", "grade_step", "transfer"):
            object.__setattr__(self, name, _number(name, getattr(self, name)))
        for name in ("k_s", "k_k", "decay"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be 0 or more, got {getattr(self, name)}")
        try:
            streak = operator.index(self.streak_to_master)
        except TypeError:
            raise TypeError(f"streak_to_master must be an integer, got {self.streak_to_master!r}") from None
        if streak < 1:
            raise ValueError(f"streak_to_master must be 1 or more, got {streak}")
        object.__setattr__(self, "streak_to_master", streak)
        for name in ("tau_days", "streak_expiry_days"):
            object.__setattr__(self, name, positive_days(name, getattr(self, name)))

    def to_dict(self) -> dict[str, Any]:
        """Plain dictionary that survives a JSON round trip (the bundle's `params` block)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> EloParams:
        """Read a `to_dict` dictionary. Missing keys take the defaults; unknown keys are ignored, so a
        bundle written by a newer version still loads."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in known})

    def replace(self, **changes: Any) -> EloParams:
        """A copy with some values changed (used by grid searches)."""
        return replace(self, **changes)


def _number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a number, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return float(value)


def positive_days(name: str, value: Any) -> float | None:
    """A number of days above zero, or None for "switched off" (infinity is read as None)."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a number of days or None, got {value!r}")
    if math.isnan(value) or value <= 0:
        raise ValueError(f"{name} must be more than 0 days, got {value!r}")
    return None if math.isinf(value) else float(value)
