"""EloParams (the bundle's `params` block) and EloLearner.from_params."""

import json
import math

import pytest

from kwan.learner.elo import EloLearner
from kwan.learner.params import EloParams, positive_days

V1_DEFAULTS = {"k_s": 0.3, "k_k": 0.4, "decay": 0.05, "grade_step": 0.4, "transfer": 0.5, "streak_to_master": 3}


def test_defaults_are_the_v1_values_plus_forgetting():
    p = EloParams()
    assert p.to_dict() == {**V1_DEFAULTS, "tau_days": 45.0, "streak_expiry_days": 30.0}


def test_params_are_frozen_and_hashable():
    p = EloParams()
    with pytest.raises(AttributeError):
        p.k_s = 1.0  # type: ignore[misc]
    assert hash(p) == hash(EloParams()) and p == EloParams()
    assert p.replace(k_s=0.2) == EloParams(k_s=0.2) and p.k_s == 0.3


def test_round_trip_through_json():
    p = EloParams(k_s=0.25, k_k=0.35, decay=0.1, grade_step=0.3, transfer=0.6, streak_to_master=4, tau_days=60)
    back = EloParams.from_dict(json.loads(json.dumps(p.to_dict())))
    assert back == p and back.tau_days == 60.0 and isinstance(back.tau_days, float)
    off = EloParams(tau_days=None, streak_expiry_days=None)
    assert json.loads(json.dumps(off.to_dict()))["tau_days"] is None
    assert EloParams.from_dict(json.loads(json.dumps(off.to_dict()))) == off


def test_from_dict_fills_missing_keys_and_ignores_unknown_ones():
    assert EloParams.from_dict({}) == EloParams() == EloParams.from_dict(None)
    p = EloParams.from_dict({"k_k": 0.5, "future_knob": 7, "streak_to_master": 2})
    assert p == EloParams(k_k=0.5, streak_to_master=2)
    assert "future_knob" not in p.to_dict()


def test_numbers_are_coerced_and_checked():
    assert EloParams(k_s=1).k_s == 1.0 and isinstance(EloParams(k_s=1).k_s, float)
    assert EloParams(tau_days=math.inf).tau_days is None  # infinity means "never forget"
    for bad in ({"k_s": -0.1}, {"decay": -1}, {"streak_to_master": 0}, {"tau_days": 0}, {"streak_expiry_days": -3}):
        with pytest.raises(ValueError):
            EloParams(**bad)
    for bad in ({"k_s": "0.3"}, {"k_k": None}, {"streak_to_master": 2.5}, {"tau_days": "45"}, {"transfer": True}):
        with pytest.raises(TypeError):
            EloParams(**bad)
    with pytest.raises(ValueError):
        EloParams(k_s=math.nan)


def test_positive_days_helper():
    assert positive_days("x", None) is None and positive_days("x", math.inf) is None
    assert positive_days("x", 7) == 7.0
    for bad, error in ((0, ValueError), (-1, ValueError), (math.nan, ValueError), ("7", TypeError), (True, TypeError)):
        with pytest.raises(error):
            positive_days("x", bad)


def test_from_params_builds_an_equivalent_learner(G):
    p = EloParams(k_s=0.2, k_k=0.3, decay=0.08, grade_step=0.35, transfer=0.7, streak_to_master=4, tau_days=20)
    L = EloLearner.from_params(p, G=G)
    assert L.G is G and L.params == p
    assert (L.k_s, L.k_k, L.decay, L.grade_step, L.transfer, L.streak_to_master) == (0.2, 0.3, 0.08, 0.35, 0.7, 4)
    assert (L.tau_days, L.streak_expiry_days) == (20.0, 30.0)
    assert EloLearner.from_params(p.to_dict(), G=G).params == p  # the bundle's dictionary works too
    assert EloLearner.from_params(None).params == EloParams() == EloLearner().params


def test_from_params_passes_the_item_tables_through(G):
    from kwan.learner.item_bank import ItemBank

    bank, table = ItemBank({"q1": 0.7}), {"q2": -0.2}
    L = EloLearner.from_params(EloParams(), G=G, item_bank=bank, item_difficulty=table)
    assert L.item_bank is bank and L.beta is table
