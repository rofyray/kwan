"""BKT baseline (PRD 5.5): forward probabilities by hand, the vectorised EM fit against known parameters
and against the v1 grid search, numerical stability and speed."""

import math
import random
import time

import numpy as np
import pytest

from kwan.learner import bkt
from kwan.learner.bkt import BKT, BKTParams, forward, log_likelihood, simulate

TRUE = {"add": BKTParams(0.3, 0.15, 0.08, 0.22), "frac": BKTParams(0.6, 0.05, 0.15, 0.3)}


def columns(rows):
    return [r[0] for r in rows], [r[2] for r in rows], [r[3] for r in rows]


# ----------------------------------------------------------------------------- forward probabilities by hand


def test_forward_matches_a_hand_computation():
    pi, pt, ps, pg = 0.4, 0.1, 0.1, 0.2
    p1 = pi * (1 - ps) + (1 - pi) * pg  # 0.36 + 0.12 = 0.48
    post = pi * (1 - ps) / p1  # right answer: 0.36 / 0.48 = 0.75
    k2 = post + (1 - post) * pt  # 0.775
    p2 = k2 * (1 - ps) + (1 - k2) * pg
    post2 = k2 * ps / (1 - p2)  # wrong answer
    k3 = post2 + (1 - post2) * pt
    p3 = k3 * (1 - ps) + (1 - k3) * pg
    assert forward([1, 0, 1], (pi, pt, ps, pg)) == pytest.approx([p1, p2, p3])
    assert p1 == pytest.approx(0.48) and k2 == pytest.approx(0.775)
    assert forward([], BKTParams()) == []
    assert forward([1, 1], BKTParams(1.0, 0.0, 0.0, 0.0)) == [1.0, 1.0]  # always known, never slips


def test_forward_agrees_with_the_v1_grid_search_helper():
    kt = pytest.importorskip("kwan.eval.kt")
    rng = random.Random(1)
    for _ in range(20):
        ys = [rng.random() < 0.6 for _ in range(rng.randrange(1, 12))]
        prm = rng.choice(kt.BKT.GRID)
        assert forward(ys, prm) == pytest.approx(kt.BKT.seq_preds(ys, *prm))


def test_log_likelihood_by_hand():
    ps = forward([1, 0], (0.4, 0.1, 0.1, 0.2))
    assert log_likelihood([1, 0], (0.4, 0.1, 0.1, 0.2)) == pytest.approx(math.log(ps[0]) + math.log(1 - ps[1]))


def test_params_are_checked_and_round_trip():
    p = BKTParams(0.3, 0.15, 0.08, 0.22)
    assert BKTParams.from_dict(p.to_dict()) == p and p.as_tuple() == (0.3, 0.15, 0.08, 0.22)
    with pytest.raises(ValueError):
        BKTParams(1.2, 0.1, 0.1, 0.1)


# ----------------------------------------------------------------------------- the vectorised model


def test_predict_equals_the_plain_forward_pass_for_every_sequence():
    rows = simulate(TRUE, n_users=40, n_steps=9, seed=2)
    rng = random.Random(4)
    shuffled = rows[:]
    # interleave learners and skills while keeping each learner's order on each skill
    shuffled.sort(key=lambda r: rng.random() * 0 + r[0] * 0)  # stable: order unchanged
    m = BKT(max_iter=0)
    m.params = dict(TRUE)
    p = m.predict(*columns(shuffled))
    by_seq = {}
    for i, r in enumerate(shuffled):
        by_seq.setdefault((r[0], r[2]), []).append(i)
    for (_u, s), idx in by_seq.items():
        ref = forward([shuffled[i][3] for i in idx], TRUE[s])
        assert [p[i] for i in idx] == pytest.approx(ref, abs=1e-12)
    users, skills, correct = columns(shuffled)
    total = sum(log_likelihood([shuffled[i][3] for i in idx], TRUE[s]) for (u, s), idx in by_seq.items())
    assert m.log_likelihood(users, skills, correct) == pytest.approx(total)


def test_rows_may_be_interleaved_across_learners_and_skills():
    a = [(1, None, "s", 1), (1, None, "s", 0), (1, None, "t", 1), (2, None, "s", 1), (1, None, "s", 1)]
    m = BKT(max_iter=0)
    p = m.predict(*columns(a))
    assert [p[0], p[1], p[4]] == pytest.approx(forward([1, 0, 1]))
    assert p[2] == pytest.approx(forward([1])[0]) and p[3] == pytest.approx(forward([1])[0])


def test_unknown_skills_use_the_default_and_empty_input_is_fine():
    m = BKT().fit([1, 1], ["a", "a"], [1, 1])
    p = m.predict([1, 2], ["a", "zzz"], [1, 0])
    assert p[1] == pytest.approx(forward([0], bkt.DEFAULT)[0])
    assert m.predict([], [], []).tolist() == [] and m.log_likelihood([], [], []) == 0.0
    assert BKT().fit([], [], []).params == {}


def test_numpy_and_python_inputs_give_the_same_result():
    rows = simulate(TRUE, n_users=30, n_steps=6, seed=8)
    users, skills, correct = columns(rows)
    a = BKT().fit(users, skills, correct)
    b = BKT().fit(np.array(users), np.array(skills), np.array(correct, dtype=bool))
    assert a.params == b.params
    c = BKT().fit(
        np.array(users, dtype=np.int32), np.array([0 if s == "add" else 1 for s in skills]), np.array(correct)
    )
    assert c.params[0] == a.params["add"] and c.params[1] == a.params["frac"]


# ----------------------------------------------------------------------------- EM


def test_em_recovers_known_parameters_approximately():
    rows = simulate(TRUE, n_users=3000, n_steps=12, seed=1)
    m = BKT().fit_interactions(rows)
    assert m.converged_ and 0 < m.n_iter_ <= 200
    for skill, true in TRUE.items():
        got = m.params[skill]
        assert got.p_init == pytest.approx(true.p_init, abs=0.06)
        assert got.p_learn == pytest.approx(true.p_learn, abs=0.04)
        assert got.p_slip == pytest.approx(true.p_slip, abs=0.04)
        assert got.p_guess == pytest.approx(true.p_guess, abs=0.05)


def test_em_never_lowers_the_likelihood_and_matches_the_plain_log_likelihood():
    rows = simulate(TRUE, n_users=200, n_steps=8, seed=3)
    users, skills, correct = columns(rows)
    lls = []
    for it in range(0, 25, 4):
        m = BKT(max_iter=it, tol=0.0).fit(users, skills, correct)
        lls.append(m.log_likelihood(users, skills, correct))
        assert sum(m.log_likelihood_.values()) == pytest.approx(lls[-1])
    assert all(b >= a - 1e-9 for a, b in zip(lls, lls[1:]))
    assert lls[-1] > lls[0]


def test_em_is_at_least_as_good_as_the_v1_grid_search():
    kt = pytest.importorskip("kwan.eval.kt")
    pd = pytest.importorskip("pandas")
    rows = simulate(TRUE, n_users=150, n_steps=10, seed=5)
    users, skills, correct = columns(rows)
    df = pd.DataFrame(rows, columns=["user_id", "problem_id", "skill_id", "correct"])
    grid = kt.BKT()
    grid.fit(df)
    em = BKT().fit(users, skills, correct)
    for skill in TRUE:
        seqs = [grp.correct.values for _, grp in df[df.skill_id == skill].groupby("user_id")]
        ll_grid = sum(log_likelihood(ys, grid.params[skill]) for ys in seqs)
        ll_em = sum(log_likelihood(ys, em.params[skill]) for ys in seqs)
        assert ll_em >= ll_grid - 1e-9
        assert em.log_likelihood_[skill] == pytest.approx(ll_em)
    # next-answer predictions from the two fits agree closely on the same rows
    assert np.abs(em.predict(users, skills, correct) - grid.predict(df)).mean() < 0.05  # the grid is coarse


def test_fit_is_deterministic():
    rows = simulate(TRUE, n_users=100, n_steps=6, seed=6)
    assert BKT().fit_interactions(rows).params == BKT().fit_interactions(rows).params


def test_starting_points_and_bounds_are_configurable():
    rows = simulate(TRUE, n_users=100, n_steps=6, seed=6)
    users, skills, correct = columns(rows)
    one = BKT(starts=[BKTParams(0.5, 0.1, 0.1, 0.1)], max_iter=0).fit(users, skills, correct)
    assert all(p == BKTParams(0.5, 0.1, 0.1, 0.1) for p in one.params.values())
    tight = BKT(max_slip=0.05, max_guess=0.05).fit(users, skills, correct)
    assert all(p.p_slip <= 0.05 and p.p_guess <= 0.05 for p in tight.params.values())
    for bad in (dict(max_iter=-1), dict(eps=0.0), dict(starts=[])):
        with pytest.raises(ValueError):
            BKT(**bad)


# ----------------------------------------------------------------------------- stability and speed


def test_degenerate_data_stays_finite():
    users = [1] * 30 + [2] * 30 + [3] * 5
    skills = ["all_right"] * 30 + ["all_wrong"] * 30 + ["one"] * 5
    correct = [1] * 30 + [0] * 30 + [1, 1, 1, 1, 1]
    m = BKT().fit(users, skills, correct)
    for p in m.params.values():
        assert all(math.isfinite(v) and 0 < v < 1 for v in p.as_tuple())
    p = m.predict(users, skills, correct)
    assert np.all(np.isfinite(p)) and np.all((p > 0) & (p < 1))
    assert math.isfinite(m.log_likelihood(users, skills, correct))
    assert m.params["all_right"].p_slip <= 1e-4 + 1e-12 and m.params["all_wrong"].p_guess <= 1e-4 + 1e-12


def test_long_sequences_do_not_underflow():
    users = [1] * 1500
    skills = ["s"] * 1500
    correct = [i % 3 != 0 for i in range(1500)]
    m = BKT(max_iter=5).fit(users, skills, correct)
    assert all(math.isfinite(v) for v in m.params["s"].as_tuple())
    assert math.isfinite(m.log_likelihood(users, skills, correct))


def test_fit_is_fast_enough_for_large_data():
    """200k rows, 20 EM rounds, in well under the time a few million rows would need per minute."""
    rng = np.random.default_rng(0)
    n = 200_000
    users, skills, correct = rng.integers(0, 4000, n), rng.integers(0, 50, n), rng.random(n) < 0.6
    t0 = time.process_time()
    m = BKT(max_iter=20, tol=0.0).fit(users, skills, correct)
    fit_s = time.process_time() - t0
    t0 = time.process_time()
    m.predict(users, skills, correct)
    predict_s = time.process_time() - t0
    assert m.n_iter_ == 20 and len(m.params) == 50
    assert fit_s < 20.0 and predict_s < 5.0  # measured about 0.7 s and 0.05 s of CPU
