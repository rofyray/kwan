"""Item-difficulty bank (PRD 6.2): beta per item, learned offline, with difficulty-label priors.

A bank answers `get(item_id) -> beta`. An item it does not know has beta 0.0, or the prior of its
difficulty label (`easy -0.5, medium 0, hard +0.5`) when a label is known. `fit_item_bank` learns the
betas from past answers with the same online rule as the `Elo2` evaluation model.

A bank carries a `provenance` record (`{"datasets": [...], "fitted_at": ...}`). The bundle builder
reads it and refuses banks fitted on non-commercial datasets.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from kwan.learner.forgetting import DAY, to_seconds

SCHEMA_VERSION = 1
LABEL_BETAS: dict[str, float] = {"easy": -0.5, "medium": 0.0, "hard": 0.5}

Interaction = Sequence[Any]  # (user, item, skill, correct) or (user, item, skill, correct, time)


def beta_for_label(label: str | None) -> float:
    """Prior difficulty of an item from its label: easy -0.5, medium 0, hard +0.5 (None gives 0.0).

    Labels are read without regard to case or surrounding spaces; any other label raises ValueError."""
    if label is None:
        return 0.0
    try:
        return LABEL_BETAS[label.strip().lower()]
    except (KeyError, AttributeError):
        raise ValueError(f"unknown difficulty label {label!r}: choose from {', '.join(LABEL_BETAS)}") from None


def item_key(item_id: Any) -> str:
    """The key an id is stored under. Ids are kept as text so a bank survives JSON unchanged
    (the item 42 and the item "42" are the same item)."""
    return item_id if isinstance(item_id, str) else str(item_id)


def stream_seconds(when: Any) -> float:
    """Time of an interaction in seconds: a number is seconds since 1970 (UTC), a datetime or
    ISO 8601 text is converted (naive values are UTC)."""
    if isinstance(when, int | float) and not isinstance(when, bool):
        return float(when)
    return to_seconds(when)


class ItemBank:
    """Difficulty (beta) per item, with optional difficulty labels and per-skill fallbacks."""

    def __init__(
        self,
        betas: Mapping[Any, float] | None = None,
        *,
        labels: Mapping[Any, str] | None = None,
        skill_betas: Mapping[Any, float] | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        self._betas: dict[str, float] = {}
        self._labels: dict[str, str] = {}
        self._skill_betas: dict[str, float] = {}
        for k, v in (betas or {}).items():
            self.set(k, v)
        for k, label in (labels or {}).items():
            self.set_label(k, label)
        for k, v in (skill_betas or {}).items():
            self.set_skill(k, v)
        self.provenance: dict[str, Any] = _provenance(provenance)

    # ------------------------------------------------------------------ reading

    def get(self, item_id: Any, label: str | None = None, skill: Any = None) -> float:
        """Difficulty of an item. The first that applies: the fitted beta of the item; the prior of
        `label` (or of the label stored for the item); the fitted difficulty of `skill`; 0.0."""
        if item_id is not None:
            key = item_key(item_id)
            beta = self._betas.get(key)
            if beta is not None:
                return beta
            if label is None:
                label = self._labels.get(key)
        if label is not None:
            return beta_for_label(label)
        if skill is not None and self._skill_betas:
            return self._skill_betas.get(item_key(skill), 0.0)
        return 0.0

    def __contains__(self, item_id: object) -> bool:
        return item_id is not None and item_key(item_id) in self._betas

    def __len__(self) -> int:
        """Number of items with a fitted beta."""
        return len(self._betas)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ItemBank):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    __hash__ = None  # type: ignore[assignment]  # a bank is mutable

    def __repr__(self) -> str:
        return f"ItemBank({len(self._betas)} items, {len(self._skill_betas)} skills, provenance={self.provenance})"

    @property
    def betas(self) -> dict[str, float]:
        """Copy of the fitted item betas, keyed by item id as text."""
        return dict(self._betas)

    @property
    def skill_betas(self) -> dict[str, float]:
        """Copy of the fitted skill difficulties (the fallback for items the bank has not seen)."""
        return dict(self._skill_betas)

    @property
    def labels(self) -> dict[str, str]:
        """Copy of the stored difficulty labels, keyed by item id as text."""
        return dict(self._labels)

    @property
    def has_skill_betas(self) -> bool:
        return bool(self._skill_betas)

    # ------------------------------------------------------------------ writing

    def set(self, item_id: Any, beta: float) -> None:
        """Store the beta of one item."""
        self._betas[_key(item_id, "item")] = _finite(beta, "beta")

    def set_label(self, item_id: Any, label: str) -> None:
        """Store the difficulty label of one item (used as its prior until a beta is fitted)."""
        beta_for_label(label)  # raises ValueError for an unknown label
        self._labels[_key(item_id, "item")] = label.strip().lower()

    def set_skill(self, skill: Any, beta: float) -> None:
        """Store the difficulty of a skill, the fallback for its items without a fitted beta."""
        self._skill_betas[_key(skill, "skill")] = _finite(beta, "beta")

    # ------------------------------------------------------------------ JSON

    def to_dict(self) -> dict[str, Any]:
        """Plain dictionary that survives a JSON round trip. Keys are sorted, so the same bank always
        gives the same text."""
        return {
            "schema_version": SCHEMA_VERSION,
            "provenance": {**self.provenance, "datasets": list(self.provenance["datasets"])},
            "label_betas": dict(LABEL_BETAS),
            "betas": {k: self._betas[k] for k in sorted(self._betas)},
            "skill_betas": {k: self._skill_betas[k] for k in sorted(self._skill_betas)},
            "labels": {k: self._labels[k] for k in sorted(self._labels)},
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> ItemBank:
        """Rebuild a bank written by `to_dict`. Missing parts are empty; unknown keys are ignored."""
        d = d or {}
        version = d.get("schema_version", SCHEMA_VERSION)
        if version != SCHEMA_VERSION:
            raise ValueError(f"item bank schema_version {version!r} is not supported (expected {SCHEMA_VERSION})")
        return cls(
            d.get("betas"), labels=d.get("labels"), skill_betas=d.get("skill_betas"), provenance=d.get("provenance")
        )


def _key(item_id: Any, what: str) -> str:
    if item_id is None:
        raise ValueError(f"an {what} id cannot be None")
    return item_key(item_id)


def _finite(x: Any, what: str) -> float:
    if isinstance(x, bool) or not isinstance(x, int | float):
        raise TypeError(f"{what} must be a number, got {x!r}")
    if not math.isfinite(x):
        raise ValueError(f"{what} must be finite, got {x!r}")
    return float(x)


def _provenance(p: Mapping[str, Any] | None) -> dict[str, Any]:
    """Provenance with the two fields the bundle builder relies on always present."""
    out: dict[str, Any] = dict(p or {})
    datasets = out.get("datasets") or []
    if isinstance(datasets, str):
        datasets = [datasets]
    out["datasets"] = [str(x) for x in datasets]
    out.setdefault("fitted_at", None)
    return out


def elo_pass(
    interactions: Iterable[Interaction],
    *,
    k_s: float,
    k_ss: float,
    decay: float,
    tau_days: float | None,
    b_item: dict[Any, float],
    b_skill: dict[Any, float],
    n_item: dict[Any, int] | None = None,
    label_beta: Mapping[Any, float] | None = None,
    learn: bool = False,
    k_b: float = 0.3,
    k_b_decay: float = 0.05,
    k_skill: float = 0.01,
    use_time: bool = True,
) -> tuple[list[float], list[int]]:
    """One pass of graph-free multi-level Elo over a stream: p = sigmoid(theta_u + theta_us - beta).

    Returns (predictions made before each answer, the answers as 0/1). Learner abilities start at zero
    and are updated online. `b_item` and `b_skill` are keyed by the ids exactly as they appear in the
    rows. With `learn`, they are updated in place with the rule of the `Elo2` evaluation model, and
    `n_item` counts the answers per item. When a row carries a time and `tau_days` is set, the
    learner-skill offset first decays toward zero by exp(-days / tau) since that learner last practised
    that skill, exactly as `EloLearner` does without a graph. `use_time=False` ignores the times.
    """
    th: dict[Any, float] = {}
    ths: dict[tuple[Any, Any], float] = {}
    n_u: dict[Any, int] = {}
    n_us: dict[tuple[Any, Any], int] = {}
    seen: dict[tuple[Any, Any], float] = {}
    preds: list[float] = []
    ys: list[int] = []
    counts: dict[Any, int] = n_item if n_item is not None else {}
    exp = math.exp
    for row in interactions:
        u, it, s, c = row[0], row[1], row[2], row[3]
        y = 1 if c else 0
        us = (u, s)
        off = ths.get(us, 0.0)
        when = row[4] if use_time and len(row) > 4 else None
        if when is None:
            if seen:
                seen.pop(us, None)  # an answer without a time: the offset stops decaying (as in EloLearner)
        else:
            t = when if type(when) is float else stream_seconds(when)
            last = seen.get(us)
            if last is None:
                seen[us] = t
            elif t > last:
                if tau_days is not None:
                    off *= exp(-((t - last) / DAY) / tau_days)  # the same arithmetic as EloLearner
                seen[us] = t
        b = b_item.get(it) if it is not None else None
        if b is None:
            b = label_beta.get(it) if label_beta else None
            if b is None:
                b = b_skill.get(s, 0.0)
        thu = th.get(u, 0.0)
        try:
            p = 1 / (1 + exp(-(thu + off - b)))
        except OverflowError:
            p = 0.0
        preds.append(p)
        ys.append(y)
        e = y - p
        nu, nus = n_u.get(u, 0), n_us.get(us, 0)
        th[u] = thu + k_s / (1 + decay * nu) * e
        ths[us] = off + k_ss / (1 + decay * nus) * e
        n_u[u], n_us[us] = nu + 1, nus + 1
        if learn:
            if it is not None:
                ni = counts.get(it, 0)
                b_item[it] = b - k_b / (1 + k_b_decay * ni) * e
                counts[it] = ni + 1
            b_skill[s] = b_skill.get(s, 0.0) - k_skill * e
    return preds, ys


def fit_item_bank(
    interactions: Iterable[Interaction],
    *,
    datasets: Sequence[str] = (),
    fitted_at: str | None = None,
    k_s: float = 0.3,
    k_ss: float = 0.4,
    k_b: float = 0.3,
    decay: float = 0.05,
    epochs: int = 2,
    tau_days: float | None = None,
    min_attempts: int = 1,
    labels: Mapping[Any, str] | None = None,
) -> ItemBank:
    """Learn item difficulties offline from past answers, without pandas.

    `interactions` is any iterable of `(user, item, skill, correct)` rows, in time order for each
    learner. A fifth element, the time of the answer (seconds since 1970, a datetime or ISO 8601 text),
    is used only when `tau_days` is given: the learner-skill offsets then forget between answers while
    the bank is fitted. An iterator is read into memory once, because the fit makes `epochs` passes.

    The rule is the training of the `Elo2` evaluation model (`kwan.eval.kt`): online multi-level Elo
    where each answer also moves the item's beta by `k_b / (1 + 0.05 * attempts on the item)` and the
    skill's difficulty by 0.01, run `epochs` times over the data (2, as the evaluation does) with the
    learner abilities reset between passes. A new item starts from its label prior if `labels` has
    one, otherwise from its skill's difficulty. With the default arguments the result equals
    `Elo2().b_item` and `.b_skill` after two `run(train, True)` calls.

    Ids are stored as text (`str(id)`), so use the same id type when fitting and when looking up.
    Items answered fewer than `min_attempts` times are left out. `datasets` and `fitted_at` go into
    the bank's `provenance`, with the fit settings; `fitted_at` defaults to now (UTC).
    """
    if epochs < 1:
        raise ValueError(f"epochs must be 1 or more, got {epochs}")
    if min_attempts < 1:
        raise ValueError(f"min_attempts must be 1 or more, got {min_attempts}")
    rows = interactions if isinstance(interactions, Sequence) else list(interactions)
    label_text = {item_key(k): v.strip().lower() for k, v in (labels or {}).items() if _check_label(v)}
    label_beta = {
        it: LABEL_BETAS[label_text[item_key(it)]] for it in {r[1] for r in rows} if item_key(it) in label_text
    }
    b_item: dict[Any, float] = {}
    b_skill: dict[Any, float] = {}
    n_item: dict[Any, int] = {}
    for _ in range(epochs):  # the counts carry over between passes, as in Elo2
        elo_pass(
            rows,
            k_s=k_s,
            k_ss=k_ss,
            decay=decay,
            tau_days=tau_days,
            b_item=b_item,
            b_skill=b_skill,
            n_item=n_item,
            label_beta=label_beta,
            learn=True,
            k_b=k_b,
        )
    keep = {k: b for k, b in b_item.items() if n_item[k] >= min_attempts * epochs}
    provenance = {
        "datasets": list(datasets),
        "fitted_at": fitted_at or datetime.now(UTC).isoformat(timespec="seconds"),
        "method": "elo-multi online",
        "settings": {
            "k_s": k_s,
            "k_ss": k_ss,
            "k_b": k_b,
            "decay": decay,
            "epochs": epochs,
            "tau_days": tau_days,
            "min_attempts": min_attempts,
        },
        "interactions": len(rows),
        "items": len(keep),
        "skills": len(b_skill),
    }
    return ItemBank(keep, labels=label_text, skill_betas=b_skill, provenance=provenance)


def _check_label(label: str) -> bool:
    beta_for_label(label)  # raises ValueError for an unknown label
    return True


def resolve_betas(bank: ItemBank | None, rows: Sequence[Interaction]) -> tuple[dict[Any, float], dict[Any, float]]:
    """(item betas, skill betas) of a bank, keyed by the ids exactly as they appear in `rows`.

    Item betas include the label priors of labelled items without a fitted beta."""
    b_item: dict[Any, float] = {}
    b_skill: dict[Any, float] = {}
    if bank is None:
        return b_item, b_skill
    betas, labels, skills = bank.betas, bank.labels, bank.skill_betas
    for it in {r[1] for r in rows}:
        if it is None:
            continue
        key = item_key(it)
        if key in betas:
            b_item[it] = betas[key]
        elif key in labels:
            b_item[it] = LABEL_BETAS[labels[key]]
    if skills:
        for s in {r[2] for r in rows}:
            key = item_key(s)
            if key in skills:
                b_skill[s] = skills[key]
    return b_item, b_skill
