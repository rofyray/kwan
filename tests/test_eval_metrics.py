"""AUC, RMSE, accuracy and expected calibration error, checked against hand computations."""

import math

import numpy as np
import pytest

from kwan.eval import metrics


def test_metric_names_and_order():
    assert metrics.METRICS == ("AUC", "RMSE", "ACC", "ECE")
    assert metrics.ECE_BINS == 10
    out = metrics.all_metrics([1, 0, 1, 0], [0.9, 0.2, 0.6, 0.4])
    assert list(out) == ["AUC", "RMSE", "ACC", "ECE"]
    assert all(type(v) is float for v in out.values())


def test_auc_by_hand():
    # positives 0.8 and 0.4, negatives 0.6 and 0.2: 3 of the 4 (positive, negative) pairs are ranked right
    assert metrics.auc([1, 0, 1, 0], [0.8, 0.6, 0.4, 0.2]) == pytest.approx(0.75)
    assert metrics.auc([1, 0], [0.9, 0.1]) == 1.0
    assert metrics.auc([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5]) == 0.5  # a constant prediction ranks nothing
    with pytest.raises(ValueError):
        metrics.auc([1, 1, 1], [0.2, 0.5, 0.9])  # one class only: AUC is not defined


def test_rmse_by_hand():
    assert metrics.rmse([1, 0], [0.5, 0.5]) == pytest.approx(0.5)
    assert metrics.rmse([1, 0, 1], [1.0, 0.0, 1.0]) == 0.0
    assert metrics.rmse([1, 0], [0.8, 0.4]) == pytest.approx(math.sqrt((0.04 + 0.16) / 2))


def test_accuracy_by_hand():
    assert metrics.accuracy([1, 0, 1, 0], [0.9, 0.2, 0.4, 0.6]) == 0.5
    assert metrics.accuracy([1, 0], [0.5, 0.5]) == 0.5  # exactly 0.5 counts as a predicted right answer
    assert metrics.accuracy([1, 1, 0], [0.51, 0.99, 0.49]) == 1.0
    assert metrics.accuracy([1, 0, 0, 0], [0.7, 0.7, 0.7, 0.2], threshold=0.8) == 0.75


def test_ece_by_hand():
    # bin [0.2, 0.3): predictions 0.2 and 0.2, answers 0 and 1  -> |0.5 - 0.2| = 0.3, weight 2/4
    # bin [0.8, 0.9): predictions 0.8 and 0.8, answers 1 and 1  -> |1.0 - 0.8| = 0.2, weight 2/4
    assert metrics.ece([0, 1, 1, 1], [0.2, 0.2, 0.8, 0.8]) == pytest.approx(0.25)
    # perfectly calibrated within every bin
    assert metrics.ece([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.0)
    assert metrics.ece([1, 1, 0, 0], [1.0, 1.0, 0.0, 0.0]) == 0.0  # 1.0 falls in the last bin, 0.0 in the first


def test_ece_bins_and_edges():
    y, p = [1, 0, 1, 1], [0.05, 0.15, 0.95, 1.0]
    # 10 bins: three bins in use. [0, .1): |1 - .05|/4; [.1, .2): |0 - .15|/4; [.9, 1]: |1 - .975| * 2/4
    assert metrics.ece(y, p) == pytest.approx((0.95 + 0.15 + 0.025 * 2) / 4)
    # one bin: |mean answer - mean prediction|
    assert metrics.ece(y, p, bins=1) == pytest.approx(abs(0.75 - np.mean(p)))
    with pytest.raises(ValueError, match="at least one"):
        metrics.ece([], [])


def test_ece_is_zero_for_a_calibrated_model_and_large_for_an_overconfident_one():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 200_000)
    y = (rng.random(len(p)) < p).astype(int)
    assert metrics.ece(y, p) < 0.01
    overconfident = np.where(p > 0.5, 0.99, 0.01)
    assert metrics.ece(y, overconfident) > 0.2
    assert 0.0 <= metrics.ece(y, overconfident) <= 1.0


def test_metrics_accept_lists_arrays_and_small_integer_types():
    y8 = np.array([1, 0, 1, 0], dtype=np.int8)
    p = np.array([0.9, 0.2, 0.6, 0.4])
    assert metrics.all_metrics(y8, p) == metrics.all_metrics(y8.tolist(), p.tolist())
    assert metrics.all_metrics(y8, p)["ACC"] == 1.0
