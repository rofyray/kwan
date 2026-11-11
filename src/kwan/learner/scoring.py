"""Fitting primitives: score a learner-model configuration on a stream of past answers.

These are the building blocks of the fits in `kwan.eval.fit` (tau, Elo hyperparameters, item banks).
They take plain rows, `(user, item, skill, correct)` or `(user, item, skill, correct, time)`, in time
order for each learner, and need neither pandas nor numpy (numpy is used for the AUC when present).

    bank = fit_item_bank(train_rows, datasets=["assistments_2009"])
    score_stream(test_rows, EloParams(tau_days=30), item_bank=bank)   # {"auc": ..., "rmse": ..., ...}
    grid_search(test_rows, {"tau_days": [7, 14, 30, 45, 90, 180, None]}, train=train_rows)

Every answer is predicted before the model sees it (next-answer prediction), with each learner starting
from an empty state. `time` may be None, seconds since 1970 (a number), a datetime or ISO 8601 text.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from kwan.learner.elo import EloLearner, LearnerState
from kwan.learner.item_bank import Interaction, ItemBank, elo_pass, fit_item_bank, resolve_betas
from kwan.learner.params import EloParams

METRICS = ("auc", "rmse", "log_loss", "accuracy")
HIGHER_IS_BETTER = {"auc": True, "accuracy": True, "rmse": False, "log_loss": False}
ENGINES = ("auto", "fast", "learner")


def _rows(interactions: Iterable[Interaction]) -> Sequence[Interaction]:
    return interactions if isinstance(interactions, Sequence) else list(interactions)


def _params(params: EloParams | Mapping[str, Any] | None) -> EloParams:
    return params if isinstance(params, EloParams) else EloParams.from_dict(params)


def predict_stream(
    interactions: Iterable[Interaction],
    params: EloParams | Mapping[str, Any] | None = None,
    *,
    item_bank: ItemBank | None = None,
    G: Any = None,
    use_time: bool = True,
    engine: str = "auto",
) -> list[float]:
    """The model's probability of a correct answer for every row, made before seeing that answer.

    params: the configuration to run (an `EloParams` or its dictionary; None gives the defaults).
    item_bank: item difficulties to use; they are not changed.
    G: a prerequisite graph whose nodes are the skills of the rows (each with a `grade`), or None.
    use_time: False ignores the times in the rows, which gives the model without forgetting.
    engine: "learner" drives the real `EloLearner`, one `LearnerState` per user; "fast" is a tight loop
        with the same arithmetic for the graph-free case (tests pin that both give the same numbers);
        "auto" picks "fast" without a graph and "learner" with one.
    """
    if engine not in ENGINES:
        raise ValueError(f"unknown engine {engine!r}: choose from {', '.join(ENGINES)}")
    if engine == "fast" and G is not None:
        raise ValueError("the fast engine has no graph prior: use engine='learner' (or 'auto') with a graph")
    p = _params(params)
    rows = _rows(interactions)
    if engine == "learner" or G is not None:
        return _predict_with_learner(rows, p, item_bank, G, use_time)
    b_item, b_skill = resolve_betas(item_bank, rows)
    preds, _ = elo_pass(
        rows,
        k_s=p.k_s,
        k_ss=p.k_k,
        decay=p.decay,
        tau_days=p.tau_days,
        b_item=b_item,
        b_skill=b_skill,
        learn=False,
        use_time=use_time,
    )
    return preds


def _predict_with_learner(
    rows: Sequence[Interaction], p: EloParams, bank: ItemBank | None, G: Any, use_time: bool
) -> list[float]:
    learner = EloLearner.from_params(p, G=G, item_bank=bank)
    states: dict[Any, LearnerState] = {}
    preds: list[float] = []
    for row in rows:
        u = row[0]
        st = states.get(u)
        if st is None:
            st = states[u] = LearnerState(str(u))
        when: Any = row[4] if use_time and len(row) > 4 else None
        if isinstance(when, int | float) and not isinstance(when, bool):
            when = datetime.fromtimestamp(when, UTC)
        preds.append(learner.update(st, row[2], row[3], item=row[1], at=when))
    return preds


def auc(y: Sequence[int], p: Sequence[float]) -> float:
    """Area under the ROC curve (ties count half). NaN when only one class is present."""
    n = len(y)
    if n != len(p):
        raise ValueError(f"y and p must have the same length, got {n} and {len(p)}")
    n1 = sum(1 for v in y if v)
    n0 = n - n1
    if n1 == 0 or n0 == 0:
        return math.nan
    try:
        import numpy as np
    except ImportError:  # the core install has no numpy
        return _auc_python(y, p, n1, n0)
    pa = np.asarray(p, dtype=np.float64)
    ya = np.asarray([1.0 if v else 0.0 for v in y], dtype=np.float64)
    _, inverse, counts = np.unique(pa, return_inverse=True, return_counts=True)
    last = np.cumsum(counts)
    mean_rank = last - (counts - 1) / 2.0
    rank_sum = float((mean_rank[inverse] * ya).sum())
    return (rank_sum - n1 * (n1 + 1) / 2) / (n1 * n0)


def _auc_python(y: Sequence[int], p: Sequence[float], n1: int, n0: int) -> float:
    pairs = sorted(zip(p, (1 if v else 0 for v in y)))
    n = len(pairs)
    rank_sum = 0.0
    i = 0
    while i < n:
        j = i
        value = pairs[i][0]
        positives = 0
        while j < n and pairs[j][0] == value:
            positives += pairs[j][1]
            j += 1
        rank_sum += (i + 1 + j) / 2 * positives  # mean of the ranks i+1 .. j
        i = j
    return (rank_sum - n1 * (n1 + 1) / 2) / (n1 * n0)


def rmse(y: Sequence[int], p: Sequence[float]) -> float:
    """Root mean squared error of the predicted probabilities."""
    return math.sqrt(math.fsum(((1 if v else 0) - q) ** 2 for v, q in zip(y, p)) / len(y))


def log_loss(y: Sequence[int], p: Sequence[float], eps: float = 1e-12) -> float:
    """Mean negative log-likelihood, with probabilities kept inside [eps, 1 - eps]."""
    return -math.fsum(
        math.log(min(max(q, eps), 1 - eps) if v else 1 - min(max(q, eps), 1 - eps)) for v, q in zip(y, p)
    ) / len(y)


def accuracy(y: Sequence[int], p: Sequence[float]) -> float:
    """Share of answers on the right side of 0.5."""
    return sum(1 for v, q in zip(y, p) if (q >= 0.5) == bool(v)) / len(y)


def score_predictions(y: Sequence[Any], p: Sequence[float]) -> dict[str, float]:
    """{"n", "auc", "rmse", "log_loss", "accuracy", "base_rate", "mean_p"} for answers and predictions."""
    if len(y) != len(p):
        raise ValueError(f"y and p must have the same length, got {len(y)} and {len(p)}")
    if not y:
        raise ValueError("nothing to score: no rows")
    yy = [1 if v else 0 for v in y]
    return {
        "n": len(yy),
        "auc": auc(yy, p),
        "rmse": rmse(yy, p),
        "log_loss": log_loss(yy, p),
        "accuracy": accuracy(yy, p),
        "base_rate": sum(yy) / len(yy),
        "mean_p": math.fsum(p) / len(p),
    }


def score_stream(
    interactions: Iterable[Interaction],
    params: EloParams | Mapping[str, Any] | None = None,
    *,
    item_bank: ItemBank | None = None,
    G: Any = None,
    use_time: bool = True,
    engine: str = "auto",
    mask: Sequence[Any] | None = None,
) -> dict[str, float]:
    """Run one configuration over the rows and score its next-answer predictions.

    Arguments are those of `predict_stream`. `mask`, when given, has one truth value per row: every
    row updates the model, and only the rows with a true value are scored (for example the second
    half of each learner's history). Returns the `score_predictions` dictionary."""
    rows = _rows(interactions)
    preds = predict_stream(rows, params, item_bank=item_bank, G=G, use_time=use_time, engine=engine)
    ys = [1 if r[3] else 0 for r in rows]
    if mask is not None:
        if len(mask) != len(rows):
            raise ValueError(f"mask must have one value per row, got {len(mask)} for {len(rows)} rows")
        ys = [v for v, m in zip(ys, mask) if m]
        preds = [q for q, m in zip(preds, mask) if m]
    return score_predictions(ys, preds)


def grid_search(
    interactions: Iterable[Interaction],
    grid: Mapping[str, Sequence[Any]],
    *,
    base: EloParams | Mapping[str, Any] | None = None,
    train: Iterable[Interaction] | None = None,
    item_bank: ItemBank | None = None,
    G: Any = None,
    metric: str = "auc",
    use_time: bool = True,
    fit_bank_with_time: bool = False,
    engine: str = "auto",
    mask: Sequence[Any] | None = None,
) -> list[dict[str, Any]]:
    """Score every combination of the values in `grid` and return the results, best first.

    interactions: the rows to score (a validation split).
    grid: `{parameter name: values to try}` over `EloParams` fields, for example
        `{"tau_days": [7, 14, 30, 45, 90, 180, None], "k_k": [0.3, 0.4, 0.5]}`.
    base: the configuration the grid varies (None gives the defaults).
    train: rows to fit an item bank on for each configuration, with that configuration's `k_s`, `k_k`
        and `decay`. Banks are fitted once per distinct setting and reused. Without `train`, the fixed
        `item_bank` (or none) is used for every configuration.
    fit_bank_with_time: also let the bank fit forget between answers, with the configuration's
        `tau_days` (one fit per tau). The default fits banks without forgetting, as the evaluation does.
    metric: what to rank by: "auc" or "accuracy" (higher is better), "rmse" or "log_loss" (lower).

    Each result is `{"params": {the grid values}, "full_params": {...}, "n", "auc", "rmse", ...}`.
    Ties keep the order of the grid, so the result is deterministic."""
    if metric not in HIGHER_IS_BETTER:
        raise ValueError(f"unknown metric {metric!r}: choose from {', '.join(METRICS)}")
    base_params = _params(base)
    names = list(grid)
    unknown = [n for n in names if n not in base_params.to_dict()]
    if unknown:
        raise ValueError(f"not learner parameters: {', '.join(unknown)}")
    if any(len(grid[n]) == 0 for n in names):
        raise ValueError("every grid entry needs at least one value")
    rows = _rows(interactions)
    train_rows = _rows(train) if train is not None else None
    banks: dict[tuple[Any, ...], ItemBank] = {}
    results: list[dict[str, Any]] = []
    for values in itertools.product(*(grid[n] for n in names)):
        chosen = dict(zip(names, values))
        p = base_params.replace(**chosen)
        bank = item_bank
        if train_rows is not None:
            tau = p.tau_days if fit_bank_with_time and use_time else None
            bank_key = (p.k_s, p.k_k, p.decay, tau)
            if bank_key not in banks:
                banks[bank_key] = fit_item_bank(train_rows, k_s=p.k_s, k_ss=p.k_k, decay=p.decay, tau_days=tau)
            bank = banks[bank_key]
        scores = score_stream(rows, p, item_bank=bank, G=G, use_time=use_time, engine=engine, mask=mask)
        results.append({"params": chosen, "full_params": p.to_dict(), **scores})
    sign = -1.0 if HIGHER_IS_BETTER[metric] else 1.0

    def rank(r: dict[str, Any]) -> tuple[bool, float]:
        value = float(r[metric])
        return (math.isnan(value), sign * value)  # undefined scores go last

    results.sort(key=rank)  # stable: ties keep grid order
    return results
