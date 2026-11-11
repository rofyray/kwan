"""Metrics for next-answer prediction: AUC, RMSE, accuracy and expected calibration error (ECE).

`y` holds the marked answers (1 right, 0 wrong) and `p` the predicted chance of a right answer.
AUC and RMSE use the same scikit-learn calls as the v1 scripts, so v1 numbers do not move.
numpy and scikit-learn come with the `eval` extra and are imported inside the functions.
"""

from __future__ import annotations

import math
from typing import Any

METRICS = ("AUC", "RMSE", "ACC", "ECE")
ECE_BINS = 10


def auc(y: Any, p: Any) -> float:
    """Area under the ROC curve. Raises ValueError when `y` holds only one class (AUC is undefined)."""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    if len(np.unique(np.asarray(y))) < 2:
        raise ValueError("AUC needs both right and wrong answers in y")
    return float(roc_auc_score(y, p))


def rmse(y: Any, p: Any) -> float:
    """Root mean squared error between the predicted chance and the 0/1 answer."""
    from sklearn.metrics import mean_squared_error

    return math.sqrt(mean_squared_error(y, p))


def accuracy(y: Any, p: Any, threshold: float = 0.5) -> float:
    """Share of answers predicted on the right side of `threshold` (a chance equal to it counts as right)."""
    import numpy as np

    y, p = np.asarray(y), np.asarray(p, dtype=float)
    return float(np.mean((p >= threshold).astype(int) == y))


def ece(y: Any, p: Any, bins: int = ECE_BINS) -> float:
    """Expected calibration error over `bins` equal-width bins of the predicted chance.

    Each bin contributes |mean answer - mean prediction| weighted by its share of the rows.
    A prediction of exactly 1.0 falls in the last bin.
    """
    import numpy as np

    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if len(y) == 0:
        raise ValueError("ece needs at least one prediction")
    idx = np.minimum((p * bins).astype(int), bins - 1)
    idx = np.maximum(idx, 0)
    count = np.bincount(idx, minlength=bins)
    sum_y = np.bincount(idx, weights=y, minlength=bins)
    sum_p = np.bincount(idx, weights=p, minlength=bins)
    filled = count > 0
    gap = np.abs(sum_y[filled] / count[filled] - sum_p[filled] / count[filled])
    return float(np.sum(gap * count[filled]) / len(y))


def all_metrics(y: Any, p: Any) -> dict[str, float]:
    """The four metrics, unrounded, in results-file order: AUC, RMSE, ACC, ECE."""
    return {"AUC": auc(y, p), "RMSE": rmse(y, p), "ACC": accuracy(y, p), "ECE": ece(y, p)}
