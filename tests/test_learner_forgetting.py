"""Forgetting (PRD v2, 6.2 and task 5.1): decay toward the prior, streak expiry, and v1 equivalence.

The rules under test are written out at the top of `kwan/learner/elo.py`.
"""

import datetime as dt
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import pytest

from kwan.graph.build import load
from kwan.learner import forgetting
from kwan.learner.elo import EloLearner, LearnerState, sigmoid
from kwan.learner.params import EloParams

GRAPH_V1 = Path(__file__).resolve().parent / "fixtures" / "graph_v1.json"
UTC = dt.UTC
T0 = dt.datetime(2026, 1, 5, 8, 0, tzinfo=UTC)
DAY = dt.timedelta(days=1)

ROOT = "B1.1.1.1.1"  # grade 1, no prerequisites
NEXT = "B1.1.1.1.2"  # grade 1, only prerequisite is ROOT
THIRD = "B1.1.1.1.3"  # grade 1, only prerequisite is NEXT
OTHER_ROOT = "B1.1.2.1.1"  # grade 1, no prerequisites, unrelated to ROOT
TWO_PREREQS = "B2.1.2.2.1"  # grade 2, prerequisites B1.1.2.2.3 and B2.1.2.1.1


@pytest.fixture(scope="module")
def G():
    """The frozen v1 curriculum graph, with bare ids."""
    return load(GRAPH_V1)


@pytest.fixture
def L(G):
    return EloLearner(G=G)


# ----------------------------------------------------------------------------- the v1.1 learner model, frozen
# Copied from engine.py at commit a86195d (v1.0; the v1.1 changes did not touch the learner model).
# It is the "old code path" of the equivalence tests below. Do not edit it.


@dataclass
class V1State:
    learner_id: str
    theta: float = 0.0
    theta_k: dict = field(default_factory=dict)
    n: dict = field(default_factory=dict)
    last_seen: dict = field(default_factory=dict)
    streak: dict = field(default_factory=dict)
    step: int = 0


def v1_sigmoid(z):
    try:
        return 1 / (1 + math.exp(-z))
    except OverflowError:
        return 0.0


class V1Learner:
    def __init__(
        self,
        k_s=0.3,
        k_k=0.4,
        decay=0.05,
        item_difficulty=None,
        G=None,
        grade_step=0.4,
        transfer=0.5,
        streak_to_master=3,
    ):
        self.k_s, self.k_k, self.decay = k_s, k_k, decay
        self.beta = item_difficulty if item_difficulty is not None else {}
        self.G, self.grade_step, self.transfer, self.streak_to_master = G, grade_step, transfer, streak_to_master

    def _offset(self, st, ind):
        if ind in st.theta_k:
            return st.theta_k[ind]
        if self.G is None:
            return 0.0
        if ind not in self.G:
            raise KeyError(f"unknown indicator {ind!r}: not in the curriculum graph")
        g = self.G.nodes[ind]["grade"]
        preds = [st.theta_k[p] for p in self.G.predecessors(ind) if p in st.theta_k]
        return -self.grade_step * (g - 1) + (self.transfer * sum(preds) / len(preds) if preds else -0.5)

    def p_correct(self, st, ind, item=None):
        return v1_sigmoid(st.theta + self._offset(st, ind) - self.beta.get(item, 0.0))

    def mastery(self, st, ind):
        if st.streak.get(ind, 0) >= self.streak_to_master:
            return max(0.95, self.p_correct(st, ind))
        return self.p_correct(st, ind)

    def update(self, st, ind, correct, item=None):
        if self.G is not None and ind not in self.G:
            raise KeyError(f"unknown indicator {ind!r}: not in the curriculum graph")
        correct = bool(correct)
        p = self.p_correct(st, ind, item)
        e = int(correct) - p
        na, nk = st.n.get("_all", 0), st.n.get(ind, 0)
        st.theta += self.k_s / (1 + self.decay * na) * e
        st.theta_k[ind] = self._offset(st, ind) + self.k_k / (1 + self.decay * nk) * e
        st.streak[ind] = st.streak.get(ind, 0) + 1 if correct else 0
        st.n["_all"], st.n[ind] = na + 1, nk + 1
        st.step += 1
        st.last_seen[ind] = st.step
        return p


V1_FIELDS = ("learner_id", "theta", "theta_k", "n", "last_seen", "streak", "step")
ITEMS = {"easy-1": -0.8, "hard-1": 1.1, 7: 0.3}


def history(G, seed, steps=300):
    """A seeded answer history: (indicator, correct, item). Skills cluster so that prerequisites,
    repeats and streaks all occur."""
    rng = random.Random(seed)
    nodes = list(G)
    pool = rng.sample(nodes, 40) + nodes[: 30 + 10 * seed]
    return [
        (rng.choice(pool), rng.random() < 0.72, rng.choice([None, None, "easy-1", "hard-1", 7, "unknown"]))
        for _ in range(steps)
    ]


def same_state(new: LearnerState, old: V1State) -> bool:
    return all(getattr(new, f) == getattr(old, f) for f in V1_FIELDS)


# ----------------------------------------------------------------------------- v1 equivalence (the acceptance check)


@pytest.mark.parametrize("seed", range(6))
def test_without_timestamps_every_output_equals_v1_exactly(G, seed):
    """Replay a seeded history through the frozen v1 code and through the new model, with no `at` and
    no `now`: every prediction, every state and every mastery is bit-for-bit the same."""
    old, new = V1Learner(G=G, item_difficulty=dict(ITEMS)), EloLearner(G=G, item_difficulty=dict(ITEMS))
    s_old, s_new = V1State("ama"), LearnerState("ama")
    for i, (ind, y, item) in enumerate(history(G, seed)):
        assert new.p_correct(s_new, ind, item) == old.p_correct(s_old, ind, item)
        assert new.update(s_new, ind, y, item=item) == old.update(s_old, ind, y, item=item)
        if i % 25 == 0 or i == 299:
            assert same_state(s_new, s_old)
            expected = {n: old.mastery(s_old, n) for n in G}
            assert {n: new.mastery(s_new, n) for n in G} == expected
            assert new.effective_mastery_map(s_new) == expected
            assert new.effective_mastery_map(s_new, None) == expected
            assert forgetting.effective_mastery_map(new, s_new) == expected
            assert list(new.effective_mastery_map(s_new)) == list(G)  # graph order
    assert same_state(s_new, s_old)
    assert s_new.last_seen_at == {}
    json_state = {k: v for k, v in json.loads(json.dumps(s_new.to_dict())).items() if k in V1_FIELDS}
    assert json_state == {f: getattr(s_old, f) for f in V1_FIELDS}


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(k_s=0.2, k_k=0.55, decay=0.1),
        dict(grade_step=0.25, transfer=0.8),
        dict(streak_to_master=2),
        dict(k_s=0.5, k_k=0.1, decay=0.0, grade_step=0.6, transfer=0.0, streak_to_master=5),
    ],
)
def test_v1_equivalence_holds_for_other_parameter_values(G, kwargs):
    old, new = V1Learner(G=G, **kwargs), EloLearner.from_params(EloParams(**kwargs), G=G)
    s_old, s_new = V1State("kofi"), LearnerState("kofi")
    for ind, y, _ in history(G, 11, steps=200):
        assert new.update(s_new, ind, y) == old.update(s_old, ind, y)
    assert same_state(s_new, s_old)
    assert new.effective_mastery_map(s_new) == {n: old.mastery(s_old, n) for n in G}


def test_v1_equivalence_without_a_graph():
    old, new = V1Learner(item_difficulty=dict(ITEMS)), EloLearner(item_difficulty=dict(ITEMS))
    s_old, s_new = V1State("esi"), LearnerState("esi")
    rng = random.Random(3)
    for _ in range(200):
        ind, y, item = rng.choice("abcdef"), rng.random() < 0.6, rng.choice([None, "hard-1", 7])
        assert new.update(s_new, ind, y, item) == old.update(s_old, ind, y, item)
    assert same_state(s_new, s_old)
    assert new.effective_mastery_map(s_new) == {k: old.mastery(s_old, k) for k in s_old.theta_k}


def test_an_empty_item_bank_changes_nothing(G):
    from kwan.learner.item_bank import ItemBank

    old, new = V1Learner(G=G), EloLearner(G=G, item_bank=ItemBank())
    s_old, s_new = V1State("yaw"), LearnerState("yaw")
    for ind, y, item in history(G, 5, steps=150):
        assert new.update(s_new, ind, y, item=item) == old.update(s_old, ind, y, item=item)
    assert same_state(s_new, s_old)
    assert new.effective_mastery_map(s_new) == {n: old.mastery(s_old, n) for n in G}


def test_the_default_parameters_are_the_v1_values_plus_the_forgetting_defaults(G):
    L = EloLearner(G=G)
    assert (L.k_s, L.k_k, L.decay, L.grade_step, L.transfer, L.streak_to_master) == (0.3, 0.4, 0.05, 0.4, 0.5, 3)
    assert (L.tau_days, L.streak_expiry_days) == (45.0, 30.0)
    assert (forgetting.TAU_DAYS, forgetting.STREAK_EXPIRY_DAYS) == (45.0, 30.0)


# ----------------------------------------------------------------------------- the forgetting curve


def practise(L, st, ind, answers, at):
    for y in answers:
        L.update(st, ind, y, at=at)


def test_effective_offset_follows_the_formula(L):
    st = LearnerState("t")
    practise(L, st, ROOT, [True, True], T0)
    raw, prior = st.theta_k[ROOT], -0.5  # a B1 skill with no prerequisites
    for days in (1, 10, 45, 90):
        expected = prior + (raw - prior) * math.exp(-days / 45)
        assert L.effective_offset(st, ROOT, T0 + days * DAY) == pytest.approx(expected, abs=1e-15)
        assert L.p_correct(st, ROOT, now=T0 + days * DAY) == pytest.approx(sigmoid(st.theta + expected), abs=1e-15)
    assert L.effective_offset(st, ROOT, T0 + 45 * DAY) - prior == pytest.approx((raw - prior) / math.e)
    assert L.effective_offset(st, ROOT) == raw  # no `now`: the stored offset


def test_zero_days_is_the_identity(G, L):
    st = LearnerState("t")
    for ind, y, _ in history(G, 2, steps=120):
        L.update(st, ind, y, at=T0)  # everything answered at one instant
    plain = {n: L.mastery(st, n) for n in G}
    assert {n: L.mastery(st, n, now=T0) for n in G} == plain
    assert L.effective_mastery_map(st, T0) == plain
    assert L.effective_mastery_map(st, T0.isoformat()) == plain
    for n in st.theta_k:
        assert L.effective_offset(st, n, T0) == st.theta_k[n]
        assert L.p_correct(st, n, "x", T0) == L.p_correct(st, n, "x")


def test_a_now_before_the_last_answer_counts_as_zero_days(G, L):
    st = LearnerState("t")
    for ind, y, _ in history(G, 2, steps=120):
        L.update(st, ind, y, at=T0)
    assert L.effective_mastery_map(st, T0 - 400 * DAY) == {n: L.mastery(st, n) for n in G}
    assert L.mastery(st, ROOT, now=T0 - 400 * DAY) == L.mastery(st, ROOT)


def blank_prior(G, L, st):
    """The offsets of a learner whose practised skills are all back at their prior: the fixed point of
    prior(k) = -grade_step * (grade - 1) + transfer * mean(prior of practised prerequisites), or - 0.5."""
    out = {}
    for n in nx.topological_sort(G):
        seen = [out[p] for p in G.predecessors(n) if p in st.theta_k]
        out[n] = -L.grade_step * (G.nodes[n]["grade"] - 1) + (L.transfer * sum(seen) / len(seen) if seen else -0.5)
    return out


def test_ten_years_unseen_returns_to_the_prior(G, L):
    st = LearnerState("t")
    for i, (ind, y, _) in enumerate(history(G, 4, steps=250)):
        L.update(st, ind, y, at=T0 + dt.timedelta(hours=5 * i))
    later = T0 + 3650 * DAY
    prior = blank_prior(G, L, st)
    for n in G:
        assert L.effective_offset(st, n, later) == pytest.approx(prior[n], abs=1e-12)
        assert L.mastery(st, n, now=later) == pytest.approx(sigmoid(st.theta + prior[n]), abs=1e-12)  # streaks expired
    assert L.mastery(st, ROOT, now=later) == pytest.approx(sigmoid(st.theta - 0.5), abs=1e-12)
    assert st.theta != 0.0  # the global ability does not decay
    assert L.effective_mastery_map(st, later) == {n: L.mastery(st, n, now=later) for n in G}


def test_decay_is_monotonic_toward_the_prior(L):
    up, down = LearnerState("up"), LearnerState("down")
    practise(L, up, ROOT, [True] * 6, T0)  # offset above the prior
    practise(L, down, ROOT, [False] * 6, T0)  # offset below the prior
    assert up.theta_k[ROOT] > -0.5 > down.theta_k[ROOT]
    days = [0, 0.5, 1, 3, 7, 14, 30, 45, 90, 180, 365, 1000]
    for st in (up, down):
        gaps = [abs(L.effective_offset(st, ROOT, T0 + d * DAY) + 0.5) for d in days]
        assert all(a > b for a, b in zip(gaps, gaps[1:]))  # the learned part shrinks, never grows
    p_up = [L.p_correct(up, ROOT, now=T0 + d * DAY) for d in days]
    p_down = [L.p_correct(down, ROOT, now=T0 + d * DAY) for d in days]
    assert all(a > b for a, b in zip(p_up, p_up[1:]))  # what was learned fades
    assert all(a < b for a, b in zip(p_down, p_down[1:]))  # what looked hopeless drifts back up to the prior


def test_a_larger_tau_forgets_more_slowly(G):
    st = LearnerState("t")
    practise(EloLearner(G=G), st, ROOT, [True] * 4, T0)
    at_30_days = [EloLearner(G=G, tau_days=tau).p_correct(st, ROOT, now=T0 + 30 * DAY) for tau in (7, 20, 45, 90, 180)]
    assert all(a < b for a, b in zip(at_30_days, at_30_days[1:]))
    assert EloLearner(G=G, tau_days=None).p_correct(st, ROOT, now=T0 + 30 * DAY) == EloLearner(G=G).p_correct(st, ROOT)
    assert EloLearner(G=G, tau_days=math.inf).tau_days is None


def test_borrowed_evidence_fades_with_its_source(L):
    """NEXT was never practised: its prior borrows half of ROOT's offset, as that offset stands at `now`."""
    st = LearnerState("t")
    practise(L, st, ROOT, [True] * 5, T0)
    assert L.effective_offset(st, NEXT) == pytest.approx(0.5 * st.theta_k[ROOT])
    for days in (0, 20, 45, 400):
        root_now = L.effective_offset(st, ROOT, T0 + days * DAY)
        assert L.effective_offset(st, NEXT, T0 + days * DAY) == pytest.approx(0.5 * root_now, abs=1e-15)
    assert L.effective_offset(st, NEXT, T0 + 5000 * DAY) == pytest.approx(0.5 * -0.5)
    assert L.effective_offset(st, THIRD, T0 + 45 * DAY) == -0.5  # transfer still reaches one hop only


def test_a_practised_skill_decays_toward_a_prior_built_from_its_practised_prerequisites(L):
    st = LearnerState("t")
    practise(L, st, ROOT, [True] * 4, T0)
    practise(L, st, NEXT, [True, False, True], T0 + 10 * DAY)
    now = T0 + 55 * DAY
    root_now = -0.5 + (st.theta_k[ROOT] + 0.5) * math.exp(-55 / 45)
    next_prior = 0.5 * root_now
    expected = next_prior + (st.theta_k[NEXT] - next_prior) * math.exp(-45 / 45)
    assert L.effective_offset(st, ROOT, now) == pytest.approx(root_now, abs=1e-15)
    assert L.effective_offset(st, NEXT, now) == pytest.approx(expected, abs=1e-15)


def test_a_skill_without_a_timestamp_does_not_decay(L):
    st = LearnerState("t")
    for _ in range(4):
        L.update(st, ROOT, True)  # no time recorded
    practise(L, st, OTHER_ROOT, [True] * 4, T0)
    later = T0 + 400 * DAY
    assert L.effective_offset(st, ROOT, later) == st.theta_k[ROOT]
    assert L.mastery(st, ROOT, now=later) == L.mastery(st, ROOT)  # the streak cannot expire either
    assert L.effective_offset(st, OTHER_ROOT, later) < st.theta_k[OTHER_ROOT]
    assert L.effective_offset(st, NEXT, later) == 0.5 * st.theta_k[ROOT]  # borrowed from a skill that did not decay


def test_without_a_graph_the_offset_decays_toward_zero():
    L, st = EloLearner(), LearnerState("t")
    practise(L, st, "fractions", [True, True, True], T0)
    raw = st.theta_k["fractions"]
    assert L.effective_offset(st, "fractions", T0 + 45 * DAY) == pytest.approx(raw / math.e)
    assert L.effective_offset(st, "fractions", T0 + 3650 * DAY) == pytest.approx(0.0, abs=1e-12)
    assert L.effective_offset(st, "never seen", T0 + 45 * DAY) == 0.0
    assert L.effective_mastery_map(st, T0 + 45 * DAY) == {"fractions": L.mastery(st, "fractions", now=T0 + 45 * DAY)}


# ----------------------------------------------------------------------------- update with a timestamp


def test_update_starts_from_the_decayed_offset(L):
    st = LearnerState("t")
    practise(L, st, ROOT, [True, True], T0)
    theta, raw = st.theta, st.theta_k[ROOT]
    at = T0 + 45 * DAY
    decayed = -0.5 + (raw + 0.5) / math.e
    p = L.update(st, ROOT, True, at=at)
    assert p == pytest.approx(sigmoid(theta + decayed), abs=1e-15)  # the prediction uses the decayed offset
    assert st.theta == pytest.approx(theta + 0.3 / (1 + 0.05 * 2) * (1 - p), abs=1e-15)
    assert st.theta_k[ROOT] == pytest.approx(decayed + 0.4 / (1 + 0.05 * 2) * (1 - p), abs=1e-15)
    assert st.last_seen_at[ROOT] == at.isoformat()
    assert L.effective_offset(st, ROOT, at) == st.theta_k[ROOT]  # the stored offset is the offset at that time


def test_update_predicts_what_p_correct_predicted_for_that_time(G, L):
    st = LearnerState("t")
    for i, (ind, y, item) in enumerate(history(G, 8, steps=200)):
        at = T0 + dt.timedelta(hours=31 * i)
        expected = L.p_correct(st, ind, item, now=at)
        assert L.update(st, ind, y, item=item, at=at) == expected


def test_the_first_timed_answer_on_a_skill_starts_from_the_decayed_transfer_prior(L):
    st = LearnerState("t")
    practise(L, st, ROOT, [True] * 4, T0)
    theta, at = st.theta, T0 + 60 * DAY
    prior = 0.5 * L.effective_offset(st, ROOT, at)
    p = L.update(st, NEXT, True, at=at)
    assert p == pytest.approx(sigmoid(theta + prior), abs=1e-15)
    assert st.theta_k[NEXT] == pytest.approx(prior + 0.4 * (1 - p), abs=1e-15)


def test_answers_out_of_order_are_v1_steps_and_keep_the_later_time(L):
    timed, plain = LearnerState("t"), LearnerState("t")
    late, early = T0 + 20 * DAY, T0 + 3 * DAY
    for st, at in ((timed, late), (plain, None)):
        L.update(st, ROOT, True, at=at)
        L.update(st, ROOT, True, at=at)
    assert L.update(timed, ROOT, False, at=early) == L.update(plain, ROOT, False)  # no gap: nothing forgotten
    assert timed.theta_k == plain.theta_k and timed.theta == plain.theta and timed.streak == plain.streak
    assert timed.last_seen_at == {ROOT: late.isoformat()}  # the later time is kept
    assert L.update(timed, NEXT, True, at=early) == L.update(plain, NEXT, True)  # ROOT is "in the future": no decay
    assert timed.last_seen_at[NEXT] == early.isoformat()


def test_an_untimed_answer_is_a_v1_step_and_stops_the_decay(L):
    timed, plain = LearnerState("t"), LearnerState("t")
    for st, at in ((timed, T0), (plain, None)):
        L.update(st, ROOT, True, at=at)
        L.update(st, OTHER_ROOT, True, at=at)
    assert L.update(timed, ROOT, True) == L.update(plain, ROOT, True)
    assert timed.theta_k == plain.theta_k
    assert timed.last_seen_at == {OTHER_ROOT: T0.isoformat()}  # the time of ROOT's last answer is unknown now
    later = T0 + 200 * DAY
    assert L.effective_offset(timed, ROOT, later) == timed.theta_k[ROOT]
    assert L.effective_offset(timed, OTHER_ROOT, later) < timed.theta_k[OTHER_ROOT]


def test_reading_never_changes_the_state(G, L):
    st = LearnerState("t")
    for i, (ind, y, _) in enumerate(history(G, 6, steps=150)):
        L.update(st, ind, y, at=T0 + dt.timedelta(hours=9 * i))
    before = json.dumps(st.to_dict(), sort_keys=True)
    later = T0 + 300 * DAY
    L.effective_mastery_map(st, later)
    for n in list(G)[:50]:
        L.mastery(st, n, now=later)
        L.p_correct(st, n, "x", later)
        L.effective_offset(st, n, later)
    assert json.dumps(st.to_dict(), sort_keys=True) == before


# ----------------------------------------------------------------------------- timestamps


def test_naive_times_are_utc_and_offsets_are_respected(L):
    def run(at):
        st = LearnerState("t")
        practise(L, st, ROOT, [True, True], "2026-01-05T08:00:00+00:00")
        return L.update(st, ROOT, True, at=at), st.theta_k[ROOT]

    reference = run("2026-02-04T08:00:00+00:00")
    assert run("2026-02-04T08:00:00") == reference  # naive text is UTC
    assert run(dt.datetime(2026, 2, 4, 8, 0)) == reference  # so is a naive datetime
    assert run(dt.datetime(2026, 2, 4, 8, 0, tzinfo=UTC)) == reference
    assert run("2026-02-04T09:00:00+01:00") == reference  # the same instant in Lagos
    assert run("2026-02-04T08:00:00Z") == reference
    assert run("2026-02-04T08:00:00+01:00") != reference  # an hour earlier


def test_time_helpers():
    assert forgetting.to_seconds("1970-01-02T00:00:00") == 86400.0
    assert forgetting.to_seconds(dt.datetime(1970, 1, 2)) == 86400.0
    assert forgetting.to_seconds("1970-01-02T01:00:00+01:00") == 86400.0
    assert forgetting.to_iso("2026-11-10T09:30:00") == "2026-11-10T09:30:00"  # text is kept as given
    assert forgetting.to_iso(dt.datetime(2026, 11, 10, 9, 30, tzinfo=UTC)) == "2026-11-10T09:30:00+00:00"
    assert forgetting.days_between("2026-01-01", "2026-01-31T12:00:00") == 30.5
    assert forgetting.days_between("2026-01-31", "2026-01-01") == 0.0
    assert forgetting.retention(0) == forgetting.retention(-3) == forgetting.retention(100, None) == 1.0
    assert forgetting.retention(45) == pytest.approx(1 / math.e)
    assert forgetting.decay_toward(2.0, -0.5, 0) == 2.0
    assert forgetting.decay_toward(2.0, -0.5, 45) == pytest.approx(-0.5 + 2.5 / math.e)
    assert forgetting.decay_toward(2.0, -0.5, 1e6) == -0.5
    for bad, error in (("soon", ValueError), (12, TypeError), (None, TypeError)):
        with pytest.raises(error):
            forgetting.to_seconds(bad)


@pytest.mark.parametrize("bad", [0, -5, math.nan, "45", True])
def test_tau_and_expiry_must_be_positive_days(bad):
    for name in ("tau_days", "streak_expiry_days"):
        with pytest.raises((ValueError, TypeError)):
            EloLearner(**{name: bad})


# ----------------------------------------------------------------------------- streak expiry


def test_streak_mastery_expires_after_thirty_days(L):
    st = LearnerState("t")
    practise(L, st, ROOT, [True, True, True], T0)
    assert L.p_correct(st, ROOT) < 0.95  # only the streak lifts it to 0.95
    assert L.mastery(st, ROOT) == 0.95
    assert L.mastery(st, ROOT, now=T0) == 0.95
    assert L.mastery(st, ROOT, now=T0 + 29 * DAY) == 0.95
    assert L.mastery(st, ROOT, now=T0 + 30 * DAY) == 0.95  # exactly 30 days: still counts
    just_after = T0 + 30 * DAY + dt.timedelta(seconds=1)
    assert L.mastery(st, ROOT, now=just_after) == L.p_correct(st, ROOT, now=just_after) < 0.85
    assert L.effective_mastery_map(st, T0 + 30 * DAY)[ROOT] == 0.95
    assert L.effective_mastery_map(st, just_after)[ROOT] == L.p_correct(st, ROOT, now=just_after)
    assert st.streak[ROOT] == 3  # reading does not reset anything


def test_streak_expiry_is_configurable_and_can_be_switched_off(G):
    st = LearnerState("t")
    practise(EloLearner(G=G), st, ROOT, [True, True, True], T0)
    week = EloLearner(G=G, streak_expiry_days=7)
    assert week.mastery(st, ROOT, now=T0 + 7 * DAY) == 0.95
    assert week.mastery(st, ROOT, now=T0 + 8 * DAY) < 0.95
    never = EloLearner(G=G, streak_expiry_days=None)
    assert never.mastery(st, ROOT, now=T0 + 5000 * DAY) == 0.95
    assert never.effective_mastery_map(st, T0 + 5000 * DAY)[ROOT] == 0.95


def test_a_high_estimate_outlives_the_streak(L):
    st = LearnerState("t")
    practise(L, st, ROOT, [True] * 3, T0)
    st.theta_k[ROOT] = 9.0
    later = T0 + 31 * DAY
    assert L.mastery(st, ROOT, now=later) == L.p_correct(st, ROOT, now=later) > 0.95


def test_an_answer_after_the_streak_expired_starts_a_new_streak(L):
    fresh, stale = LearnerState("fresh"), LearnerState("stale")
    practise(L, fresh, ROOT, [True, True, True], T0)
    practise(L, stale, ROOT, [True, True, True], T0)
    L.update(fresh, ROOT, True, at=T0 + 30 * DAY)
    L.update(stale, ROOT, True, at=T0 + 31 * DAY)
    assert fresh.streak[ROOT] == 4
    assert stale.streak[ROOT] == 1
    assert L.mastery(stale, ROOT, now=T0 + 31 * DAY) < 0.95  # one right answer after a long break is not mastery
    L.update(stale, ROOT, False, at=T0 + 32 * DAY)
    assert stale.streak[ROOT] == 0


# ----------------------------------------------------------------------------- the all-skills pass


@pytest.mark.parametrize("seed", range(4))
def test_the_mastery_map_equals_one_call_per_skill_at_any_time(G, L, seed):
    st = LearnerState("t")
    rng = random.Random(seed)
    for i, (ind, y, item) in enumerate(history(G, seed, steps=250)):
        at = None if rng.random() < 0.1 else T0 + dt.timedelta(hours=rng.uniform(0, 24 * 120))  # some out of order
        L.update(st, ind, y, item=item, at=at)
        if i % 50 == 49:
            for days in (-10, 0, 0.01, 3, 30, 30.0001, 61, 400, 4000):
                now = T0 + days * DAY
                got = L.effective_mastery_map(st, now)
                assert got == {n: L.mastery(st, n, now=now) for n in G}
                assert list(got) == list(G)


def test_the_mastery_map_is_deterministic_and_fresh_each_call(G, L):
    st = LearnerState("t")
    for i, (ind, y, _) in enumerate(history(G, 9, steps=200)):
        L.update(st, ind, y, at=T0 + dt.timedelta(hours=7 * i))
    now = T0 + 90 * DAY
    a, b = L.effective_mastery_map(st, now), L.effective_mastery_map(st, now)
    assert a == b and a is not b
    a[ROOT] = -1.0
    assert L.effective_mastery_map(st, now) == b  # the caller owns the result
    other = EloLearner(G=G)
    assert (
        other.effective_mastery_map(LearnerState.from_dict(json.loads(json.dumps(st.to_dict()))), now.isoformat()) == b
    )


def test_same_history_gives_the_same_state_and_the_same_map(G):
    def run():
        L, st = EloLearner(G=G), LearnerState("t")
        for i, (ind, y, item) in enumerate(history(G, 12, steps=200)):
            L.update(st, ind, y, item=item, at=(T0 + dt.timedelta(hours=11 * i)).isoformat())
        return st.to_dict(), L.effective_mastery_map(st, "2026-09-01T00:00:00+00:00")

    assert run() == run()


def toy_graph(order):
    g = nx.DiGraph()
    for n in order:
        g.add_node(n, grade={"a": 1, "b": 1, "c": 2, "d": 2}[n])
    g.add_edges_from([("a", "b"), ("b", "c"), ("a", "c"), ("c", "d")])
    return g


def test_the_map_works_when_the_graph_lists_skills_before_their_prerequisites():
    g = toy_graph(["d", "c", "b", "a"])  # not in prerequisite order
    L, st = EloLearner(G=g), LearnerState("t")
    for i, (ind, y) in enumerate([("a", True), ("b", True), ("a", True), ("c", False), ("d", True), ("b", True)]):
        L.update(st, ind, y, at=T0 + i * DAY)
    for days in (0, 5, 40, 300):
        now = T0 + days * DAY
        got = L.effective_mastery_map(st, now)
        assert got == {n: L.mastery(st, n, now=now) for n in g}
        assert list(got) == ["d", "c", "b", "a"]  # still in graph order
    same = EloLearner(G=toy_graph(["a", "b", "c", "d"]))
    assert same.effective_mastery_map(st, T0 + 40 * DAY) == L.effective_mastery_map(st, T0 + 40 * DAY)


def test_a_graph_with_a_cycle_works_without_time_and_is_refused_with_it():
    g = toy_graph(["a", "b", "c", "d"])
    g.add_edge("d", "b")  # b -> c -> d -> b
    L, st = EloLearner(G=g), LearnerState("t")
    for ind in ("a", "b", "c", "d"):
        L.update(st, ind, True, at=T0)
    assert L.effective_mastery_map(st) == {n: L.mastery(st, n) for n in g}  # v1 never needed an order
    assert L.effective_mastery_map(st, T0) == {n: L.mastery(st, n) for n in g}  # nothing decayed yet
    with pytest.raises(ValueError, match="without cycles"):
        L.effective_mastery_map(st, T0 + 10 * DAY)
    with pytest.raises(ValueError, match="without cycles"):
        L.mastery(st, "c", now=T0 + 10 * DAY)


def test_the_map_follows_a_changed_grade_step_and_a_rebuilt_table(G):
    L, st = EloLearner(G=G), LearnerState("t")
    before = L.effective_mastery_map(st)
    L.grade_step = 0.1
    after = L.effective_mastery_map(st)
    assert after == {n: L.mastery(st, n) for n in G} and after != before
    L.forget_graph_tables()
    assert L.effective_mastery_map(st) == after


def best_mean_ms(f, batches=9, calls=40):
    """Mean CPU time of one call in the fastest of several batches. CPU time, not wall time, so the
    cost of the code is measured even when the machine is busy (`kwan bench` measures wall time)."""
    best = math.inf
    for _ in range(batches):
        t0 = time.process_time()
        for _ in range(calls):
            f()
        best = min(best, (time.process_time() - t0) / calls)
    return best * 1000


def busy_learner(G, L, timed):
    st, rng = LearnerState("busy"), random.Random(1)
    nodes = list(G)
    for i in range(400):
        L.update(st, rng.choice(nodes[:200]), rng.random() < 0.7, at=T0 + dt.timedelta(hours=7 * i) if timed else None)
    return st


def test_the_mastery_map_is_fast_on_the_real_graph(G, L):
    """The recommender calls this on every request and has 1 ms for the whole request (PRD 7)."""
    plain, timed = busy_learner(G, L, timed=False), busy_learner(G, L, timed=True)
    assert len(timed.last_seen_at) > 150
    now = T0 + 200 * DAY
    assert best_mean_ms(lambda: L.effective_mastery_map(plain)) < 0.35
    assert best_mean_ms(lambda: L.effective_mastery_map(timed, now)) < 0.6
    assert best_mean_ms(lambda: L.effective_mastery_map(LearnerState("new"), now)) < 0.35


def test_the_map_beats_one_call_per_skill(G, L):
    timed = busy_learner(G, L, timed=True)
    now = T0 + 200 * DAY
    one_pass = best_mean_ms(lambda: L.effective_mastery_map(timed, now), batches=5, calls=20)
    per_skill = best_mean_ms(lambda: {n: L.mastery(timed, n, now=now) for n in G}, batches=3, calls=3)
    assert one_pass * 3 < per_skill


def test_the_map_is_a_small_share_of_a_top3_recommend_with_forgetting(G, L):
    """The recommender's own tests hold its latency budget; this checks that the all-skills pass is
    not what fills it, with or without `now`."""
    from kwan.recommend.recommender import Recommender

    R, timed = Recommender(G, L), busy_learner(G, L, timed=True)
    now = T0 + 200 * DAY
    for kwargs in ({"now": now}, {}):
        whole = best_mean_ms(lambda: R.recommend(timed, k=3, **kwargs))
        part = best_mean_ms(lambda: L.effective_mastery_map(timed, kwargs.get("now")))
        assert whole < 5.0 and part < 0.6
        assert part < 0.6 * whole


# ----------------------------------------------------------------------------- state v2 round trips


def test_a_timed_state_survives_json_and_keeps_learning_the_same_way(G, L):
    st = LearnerState("ama", subject="math", bundle_version="math-1.0.0")
    for i, (ind, y, _) in enumerate(history(G, 13, steps=120)):
        L.update(st, ind, y, at=T0 + dt.timedelta(hours=13 * i))
    d = json.loads(json.dumps(st.to_dict()))
    assert d["schema_version"] == 2 and set(d["last_seen_at"]) == set(st.theta_k)
    back = LearnerState.from_dict(d)
    assert back == st
    now = T0 + 100 * DAY
    assert L.effective_mastery_map(back, now) == L.effective_mastery_map(st, now)
    assert L.update(back, ROOT, True, at=now) == L.update(st, ROOT, True, at=now)
    assert back == st


def test_a_v1_state_reads_the_same_at_any_time(G, L):
    st = LearnerState("kofi")
    for ind, y, _ in history(G, 14, steps=120):
        L.update(st, ind, y)
    v1 = {k: v for k, v in json.loads(json.dumps(st.to_dict())).items() if k in V1_FIELDS}
    back = LearnerState.from_dict(v1)
    assert back == st and back.last_seen_at == {} and back.schema_version == 2
    assert L.effective_mastery_map(back, T0 + 900 * DAY) == {n: L.mastery(st, n) for n in G}
    L.update(back, ROOT, True, at=T0)  # a v1 state starts recording times from its next timed answer
    assert back.last_seen_at == {ROOT: T0.isoformat()}
