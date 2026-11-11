"""Fitting primitives (`kwan.learner.scoring`): scoring a configuration on a stream of answers and grid
search over tau and the Elo hyperparameters, without pandas."""

import datetime as dt
import math
import random

import pytest

from kwan.learner import scoring
from kwan.learner.elo import EloLearner, LearnerState
from kwan.learner.item_bank import fit_item_bank
from kwan.learner.params import EloParams
from kwan.learner.scoring import grid_search, predict_stream, score_predictions, score_stream


def timed_rows(n_users=40, steps=25, seed=0, tau_true=20.0):
    """Answers from a learner model that forgets with time constant `tau_true` days."""
    rng = random.Random(seed)
    rows = []
    for u in range(n_users):
        t = 1_700_000_000.0 + u
        ability = rng.gauss(0, 0.7)
        learned = {}
        for _ in range(steps):
            s = rng.choice("abc")
            gap = rng.choice([0.01, 0.5, 2, 10, 40])
            t += gap * 86400
            val, last = learned.get(s, (0.0, t))
            val *= math.exp(-(t - last) / 86400 / tau_true)
            y = int(rng.random() < 1 / (1 + math.exp(-(ability + val - 0.3))))
            learned[s] = (val + (0.6 if y else -0.3), t)
            rows.append((u, f"{s}{rng.randrange(5)}", s, y, t))
    return rows


# ----------------------------------------------------------------------------- metrics


def test_metrics_by_hand():
    y, p = [1, 0, 1, 0], [0.9, 0.2, 0.6, 0.6]
    assert scoring.auc(y, p) == 0.875  # pairs: (0.9>0.2), (0.9>0.6), (0.6>0.2), (0.6=0.6 counts half)
    assert scoring._auc_python(y, p, 2, 2) == 0.875
    assert scoring.rmse(y, p) == pytest.approx(math.sqrt((0.01 + 0.04 + 0.16 + 0.36) / 4))
    assert scoring.log_loss(y, p) == pytest.approx(-(math.log(0.9) + math.log(0.8) + math.log(0.6) + math.log(0.4)) / 4)
    assert scoring.accuracy(y, p) == 0.75
    assert math.isnan(scoring.auc([1, 1], [0.2, 0.3]))
    s = score_predictions(y, p)
    assert s["n"] == 4 and s["base_rate"] == 0.5 and s["mean_p"] == pytest.approx(0.575)
    with pytest.raises(ValueError):
        score_predictions([], [])
    with pytest.raises(ValueError):
        score_predictions([1], [0.5, 0.5])


def test_auc_matches_sklearn_when_available():
    sk = pytest.importorskip("sklearn.metrics")
    rng = random.Random(1)
    y = [rng.random() < 0.6 for _ in range(500)]
    p = [round(rng.random(), 2) for _ in y]  # ties on purpose
    assert scoring.auc(y, p) == pytest.approx(sk.roc_auc_score(y, p))
    assert scoring._auc_python(y, p, sum(y), len(y) - sum(y)) == pytest.approx(sk.roc_auc_score(y, p))


# ----------------------------------------------------------------------------- predict and score a stream


def test_fast_and_learner_engines_agree_with_and_without_time():
    rows = timed_rows(n_users=15)
    bank = fit_item_bank(rows)
    for use_time in (True, False):
        for p in (EloParams(), EloParams(k_s=0.2, k_k=0.5, decay=0.1, tau_days=10)):
            fast = predict_stream(rows, p, item_bank=bank, use_time=use_time, engine="fast")
            slow = predict_stream(rows, p, item_bank=bank, use_time=use_time, engine="learner")
            assert fast == pytest.approx(slow, abs=1e-12)
            assert predict_stream(rows, p, item_bank=bank, use_time=use_time) == fast  # auto picks fast
    with pytest.raises(ValueError):
        predict_stream(rows, engine="odd")


def test_the_learner_engine_is_the_real_model_one_state_per_user():
    rows = timed_rows(n_users=5, steps=10)
    L, states, expected = EloLearner(tau_days=10), {}, []
    for u, it, s, y, t in rows:
        st = states.setdefault(u, LearnerState(str(u)))
        expected.append(L.update(st, s, y, item=it, at=dt.datetime.fromtimestamp(t, dt.UTC)))
    assert predict_stream(rows, EloParams(tau_days=10), engine="learner") == pytest.approx(expected, abs=1e-12)


def test_times_can_be_numbers_datetimes_or_text():
    rows = timed_rows(n_users=5, steps=8)
    as_dt = [(u, it, s, y, dt.datetime.fromtimestamp(t, dt.UTC)) for u, it, s, y, t in rows]
    as_text = [(u, it, s, y, dt.datetime.fromtimestamp(t, dt.UTC).isoformat()) for u, it, s, y, t in rows]
    p = EloParams(tau_days=5)
    base = predict_stream(rows, p)
    assert predict_stream(as_dt, p) == pytest.approx(base, abs=1e-9)
    assert predict_stream(as_text, p) == pytest.approx(base, abs=1e-9)
    assert predict_stream(as_dt, p, engine="learner") == pytest.approx(base, abs=1e-9)


def test_score_stream_scores_only_the_masked_rows():
    rows = timed_rows(n_users=10)
    full = score_stream(rows, use_time=False)
    mask = [i % 2 == 1 for i in range(len(rows))]
    half = score_stream(rows, use_time=False, mask=mask)
    assert full["n"] == len(rows) and half["n"] == len(rows) // 2
    preds = predict_stream(rows, use_time=False)
    assert half == score_predictions([r[3] for r, m in zip(rows, mask) if m], [p for p, m in zip(preds, mask) if m])
    with pytest.raises(ValueError):
        score_stream(rows, mask=[True])


def test_a_graph_routes_to_the_learner_engine(G):
    nodes = list(G)[:6]
    rng = random.Random(2)
    rows = [(u, None, rng.choice(nodes), rng.random() < 0.6) for u in range(8) for _ in range(10)]
    with_graph = predict_stream(rows, G=G)
    assert with_graph == pytest.approx(predict_stream(rows, G=G, engine="learner"))
    assert with_graph != pytest.approx(predict_stream(rows))  # the graph prior changes the first guesses
    with pytest.raises(ValueError):
        predict_stream(rows, G=G, engine="fast")


def test_forgetting_helps_on_data_that_forgets():
    rows = timed_rows(n_users=120, steps=30, seed=4, tau_true=20)
    with_tau = score_stream(rows, EloParams(tau_days=20))
    without = score_stream(rows, EloParams(tau_days=None))
    assert with_tau["auc"] > without["auc"] and with_tau["log_loss"] < without["log_loss"]
    assert score_stream(rows, EloParams(tau_days=20), use_time=False) == without  # ignoring time = no forgetting


# ----------------------------------------------------------------------------- grid search


def test_grid_search_is_exhaustive_ordered_and_deterministic():
    rows = timed_rows(n_users=30, steps=15)
    grid = {"tau_days": [7, 45, None], "k_k": [0.3, 0.5]}
    out = grid_search(rows, grid, metric="auc")
    assert len(out) == 6 and sorted(r["params"]["tau_days"] is None for r in out) == [False] * 4 + [True] * 2
    aucs = [r["auc"] for r in out]
    assert aucs == sorted(aucs, reverse=True)
    assert all(set(r) >= {"params", "full_params", "n", "auc", "rmse", "log_loss", "accuracy"} for r in out)
    assert out[0]["full_params"] == EloParams(**out[0]["params"]).to_dict()
    assert grid_search(rows, grid, metric="auc") == out
    by_loss = grid_search(rows, grid, metric="log_loss")
    assert [r["log_loss"] for r in by_loss] == sorted(r["log_loss"] for r in by_loss)
    for r in out:
        assert score_stream(rows, EloParams(**r["params"])) == {k: r[k] for k in score_predictions([1, 0], [0.6, 0.4])}


def test_grid_search_finds_a_tau_near_the_truth():
    rows = timed_rows(n_users=200, steps=30, seed=9, tau_true=20)
    out = grid_search(rows, {"tau_days": [3, 7, 14, 20, 30, 45, 90, 180, None]}, metric="log_loss")
    # The generator is not the Elo model itself, so the best tau is near, not at, the truth: forgetting
    # on a time scale of weeks beats both a few days and never forgetting.
    assert out[0]["params"]["tau_days"] in (14, 20, 30, 45)
    assert out[-1]["params"]["tau_days"] in (3, None)


def test_grid_search_fits_one_bank_per_distinct_setting(monkeypatch):
    import kwan.learner.scoring as mod

    train, test = timed_rows(n_users=20, seed=1), timed_rows(n_users=10, seed=2)
    calls = []
    real = mod.fit_item_bank

    def counting(rows, **kw):
        calls.append(kw)
        return real(rows, **kw)

    monkeypatch.setattr(mod, "fit_item_bank", counting)
    out = grid_search(test, {"tau_days": [7, 30], "k_k": [0.3, 0.4]}, train=train)
    assert len(calls) == 2 and len(out) == 4  # k_k changes the bank, tau does not (banks fit without time)
    calls.clear()
    grid_search(test, {"tau_days": [7, 30]}, train=train, fit_bank_with_time=True)
    assert [c["tau_days"] for c in calls] == [7, 30]
    bank = real(train)
    expected = score_stream(test, EloParams(tau_days=7), item_bank=bank)
    assert grid_search(test, {"tau_days": [7]}, train=train)[0]["auc"] == expected["auc"]


def test_grid_search_rejects_bad_input():
    rows = timed_rows(n_users=5, steps=5)
    with pytest.raises(ValueError, match="not learner parameters"):
        grid_search(rows, {"tau": [7]})
    with pytest.raises(ValueError, match="at least one value"):
        grid_search(rows, {"tau_days": []})
    with pytest.raises(ValueError, match="unknown metric"):
        grid_search(rows, {"tau_days": [7]}, metric="f1")


def test_undefined_scores_sort_last():
    rows = [(0, None, "s", 1, 1.0), (0, None, "s", 1, 2.0)]  # one class only: AUC is undefined
    out = grid_search(rows, {"tau_days": [7, None]})
    assert all(math.isnan(r["auc"]) for r in out) and len(out) == 2
