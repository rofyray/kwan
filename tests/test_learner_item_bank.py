"""Item-difficulty bank (PRD 6.2, task 5.3): lookups, label priors, JSON, the offline fitter and its use
by EloLearner."""

import json
import math
import random

import pytest

from kwan.learner.elo import EloLearner, LearnerState, sigmoid
from kwan.learner.item_bank import LABEL_BETAS, ItemBank, beta_for_label, elo_pass, fit_item_bank, resolve_betas

ROOT = "B1.1.1.1.1"


@pytest.fixture(scope="module")
def G(G_v1):
    """The hand-computed numbers below use bare codes: they run on the frozen v1 graph."""
    return G_v1


def synthetic_rows(n_users=60, n_items=25, n_skills=4, steps=30, seed=0, with_time=False):
    """Seeded answers from a simple IRT-like model: (user, item, skill, correct[, time])."""
    rng = random.Random(seed)
    ability = [rng.gauss(0, 1) for _ in range(n_users)]
    difficulty = [rng.gauss(0, 1) for _ in range(n_items)]
    rows = []
    for u in range(n_users):
        t = 1_700_000_000.0 + u
        for _ in range(steps):
            it = rng.randrange(n_items)
            y = int(rng.random() < sigmoid(ability[u] - difficulty[it]))
            t += rng.uniform(60, 3 * 86400)
            rows.append((u, it, it % n_skills, y, t) if with_time else (u, it, it % n_skills, y))
    return rows, difficulty


# ----------------------------------------------------------------------------- lookups and labels


def test_label_priors():
    assert LABEL_BETAS == {"easy": -0.5, "medium": 0.0, "hard": 0.5}
    assert beta_for_label("easy") == -0.5 and beta_for_label("medium") == 0.0 and beta_for_label("hard") == 0.5
    assert beta_for_label(" Hard ") == 0.5 and beta_for_label(None) == 0.0
    with pytest.raises(ValueError, match="unknown difficulty label"):
        beta_for_label("tricky")


def test_get_falls_through_item_label_skill_then_zero():
    bank = ItemBank({"q1": 0.7, 42: -0.3}, labels={"q2": "hard", "q3": "easy"}, skill_betas={"frac": 0.2})
    assert bank.get("q1") == 0.7 and bank.get(42) == -0.3 and bank.get("42") == -0.3  # ids are text
    assert bank.get("q2") == 0.5 and bank.get("q3") == -0.5
    assert bank.get("q1", label="easy") == 0.7  # a fitted beta beats the label
    assert bank.get("new", label="easy") == -0.5
    assert bank.get("new", skill="frac") == 0.2 and bank.get("new", label="hard", skill="frac") == 0.5
    assert bank.get("new") == 0.0 and bank.get(None) == 0.0 and bank.get(None, skill="frac") == 0.2
    assert "q1" in bank and 42 in bank and "q2" not in bank and None not in bank
    assert len(bank) == 2 and bank.has_skill_betas


def test_empty_bank_is_all_zeros():
    bank = ItemBank()
    assert bank.get("anything") == 0.0 and len(bank) == 0 and not bank.has_skill_betas
    assert bank.provenance == {"datasets": [], "fitted_at": None}


def test_setters_check_their_input():
    bank = ItemBank()
    bank.set("q", 1.5)
    bank.set_label("q2", " EASY ")
    bank.set_skill("s", -0.1)
    assert bank.get("q") == 1.5 and bank.labels == {"q2": "easy"} and bank.skill_betas == {"s": -0.1}
    for call in (lambda: bank.set("q", "1"), lambda: bank.set("q", True)):
        with pytest.raises(TypeError):
            call()
    for call in (lambda: bank.set("q", math.nan), lambda: bank.set(None, 1.0), lambda: bank.set_label("q", "odd")):
        with pytest.raises(ValueError):
            call()


# ----------------------------------------------------------------------------- JSON and provenance


def test_round_trip_through_json_is_exact_and_sorted():
    bank = ItemBank(
        {"b": 0.25, "a": -1.0, 3: 0.5},
        labels={"z": "hard"},
        skill_betas={"s2": 0.1, "s1": -0.2},
        provenance={"datasets": ["assistments_2009"], "fitted_at": "2026-11-10T00:00:00+00:00", "note": "x"},
    )
    d = bank.to_dict()
    assert list(d["betas"]) == ["3", "a", "b"] and list(d["skill_betas"]) == ["s1", "s2"]
    assert d["provenance"] == {"datasets": ["assistments_2009"], "fitted_at": "2026-11-10T00:00:00+00:00", "note": "x"}
    assert d["schema_version"] == 1 and d["label_betas"] == LABEL_BETAS
    back = ItemBank.from_dict(json.loads(json.dumps(d)))
    assert back == bank and back.to_dict() == d
    assert json.dumps(back.to_dict(), sort_keys=True) == json.dumps(d, sort_keys=True)
    d["betas"]["a"] = 9.0
    assert bank.get("a") == -1.0  # to_dict copies


def test_provenance_always_has_datasets_and_fitted_at():
    assert ItemBank(provenance={"datasets": "junyi_2015"}).provenance == {"datasets": ["junyi_2015"], "fitted_at": None}
    assert ItemBank.from_dict({}).provenance == {"datasets": [], "fitted_at": None}
    assert ItemBank.from_dict(None) == ItemBank()
    with pytest.raises(ValueError, match="schema_version"):
        ItemBank.from_dict({"schema_version": 2})


# ----------------------------------------------------------------------------- the fitter


def test_fit_matches_the_elo2_training_of_the_evaluation():
    """`fit_item_bank` with the defaults gives exactly the item and skill betas that `Elo2` learns in
    `kwan.eval.kt.score_models` (two learning passes over the training frame)."""
    pd = pytest.importorskip("pandas")
    from kwan.eval.kt import Elo2

    rows, _ = synthetic_rows()
    df = pd.DataFrame(rows, columns=["user_id", "problem_id", "skill_id", "correct"])
    m = Elo2()
    m.run(df, True)
    m.run(df, True)
    bank = fit_item_bank(rows, datasets=["synthetic"], fitted_at="2026-11-10T00:00:00+00:00")
    assert bank.betas == {str(k): v for k, v in m.b_item.items()}
    assert bank.skill_betas == {str(k): v for k, v in m.b_skill.items()}
    assert len(bank) == 25 and set(bank.provenance["datasets"]) == {"synthetic"}
    assert bank.provenance["settings"]["epochs"] == 2 and bank.provenance["interactions"] == len(rows)
    for it in range(25):
        assert bank.get(it) == m.b_item[it]


def test_fit_recovers_the_ordering_of_true_difficulties():
    rows, difficulty = synthetic_rows(n_users=200, steps=40, seed=3)
    bank = fit_item_bank(rows)
    fitted = [bank.get(i) for i in range(25)]
    # Spearman rank correlation between the true and the fitted difficulties
    rank = lambda xs: {i: r for r, i in enumerate(sorted(range(len(xs)), key=xs.__getitem__))}
    rt, rf = rank(difficulty), rank(fitted)
    n = len(fitted)
    rho = 1 - 6 * sum((rt[i] - rf[i]) ** 2 for i in range(n)) / (n * (n * n - 1))
    assert rho > 0.8


def test_fit_accepts_iterators_and_is_deterministic():
    rows, _ = synthetic_rows(n_users=20)
    a = fit_item_bank(iter(rows), fitted_at="t")
    b = fit_item_bank(tuple(rows), fitted_at="t")
    assert a == b and a.to_dict() == b.to_dict()


def test_fit_options_min_attempts_epochs_and_labels():
    rows, _ = synthetic_rows(n_users=20, steps=10)
    counts = {}
    for r in rows:
        counts[r[1]] = counts.get(r[1], 0) + 1
    bank = fit_item_bank(rows, min_attempts=10)
    assert set(bank.betas) == {str(i) for i, c in counts.items() if c >= 10}
    assert len(fit_item_bank(rows, epochs=1)) == 25 and fit_item_bank(rows, epochs=1) != fit_item_bank(rows)
    labelled = fit_item_bank(rows, labels={0: "hard", 1: "easy"})
    assert labelled.labels == {"0": "hard", "1": "easy"}
    assert labelled.betas["0"] != fit_item_bank(rows).betas["0"]  # the label prior is the starting point
    for bad in (dict(epochs=0), dict(min_attempts=0), dict(labels={0: "odd"})):
        with pytest.raises(ValueError):
            fit_item_bank(rows, **bad)


def test_fit_with_tau_uses_the_times_and_without_it_ignores_them():
    timed, _ = synthetic_rows(n_users=30, with_time=True, seed=5)
    plain = [r[:4] for r in timed]
    assert fit_item_bank(timed, fitted_at="t") == fit_item_bank(plain, fitted_at="t")
    with_tau = fit_item_bank(timed, tau_days=7, fitted_at="t")
    assert with_tau != fit_item_bank(plain, fitted_at="t") and with_tau.provenance["settings"]["tau_days"] == 7


def test_elo_pass_with_time_matches_the_graph_free_learner_with_forgetting():
    """The fast stream pass and EloLearner (no graph) are the same arithmetic, forgetting included."""
    import datetime as dt

    rows, _ = synthetic_rows(n_users=15, with_time=True, seed=7)
    bank = fit_item_bank(rows)
    b_item, b_skill = resolve_betas(bank, rows)
    fast, _ = elo_pass(rows, k_s=0.3, k_ss=0.4, decay=0.05, tau_days=10, b_item=b_item, b_skill=b_skill)
    L, states = EloLearner(item_bank=bank, tau_days=10), {}
    slow = [
        L.update(states.setdefault(u, LearnerState(str(u))), s, y, item=it, at=dt.datetime.fromtimestamp(t, dt.UTC))
        for u, it, s, y, t in rows
    ]
    assert fast == pytest.approx(slow, abs=1e-12)
    without, _ = elo_pass(rows, k_s=0.3, k_ss=0.4, decay=0.05, tau_days=None, b_item=b_item, b_skill=b_skill)
    assert without != fast
    ignored, _ = elo_pass(
        rows, k_s=0.3, k_ss=0.4, decay=0.05, tau_days=10, b_item=b_item, b_skill=b_skill, use_time=False
    )
    assert ignored == without


def test_resolve_betas_keys_by_the_row_ids():
    bank = ItemBank({"1": 0.4}, labels={"2": "easy"}, skill_betas={"s": 0.1})
    rows = [(0, 1, "s", 1), (0, 2, "s", 0), (0, 3, "t", 1), (0, None, "s", 1)]
    assert resolve_betas(bank, rows) == ({1: 0.4, 2: -0.5}, {"s": 0.1})
    assert resolve_betas(None, rows) == ({}, {})


# ----------------------------------------------------------------------------- EloLearner with a bank


def test_learner_uses_the_bank_and_the_old_table_still_wins(G):
    bank = ItemBank({"q1": 1.0, "q2": -1.0}, labels={"q3": "hard"})
    L, st = EloLearner(G=G, item_bank=bank), LearnerState("t")
    base = L.p_correct(st, ROOT)
    assert L.p_correct(st, ROOT, "q1") == pytest.approx(sigmoid(-0.5 - 1.0))
    assert L.p_correct(st, ROOT, "q2") == pytest.approx(sigmoid(-0.5 + 1.0))
    assert L.p_correct(st, ROOT, "q1") < base < L.p_correct(st, ROOT, "q2")
    assert L.p_correct(st, ROOT, "q3") == pytest.approx(sigmoid(-0.5 - 0.5))
    assert L.p_correct(st, ROOT, "unknown") == base == L.p_correct(st, ROOT, None)
    assert L.p_correct(st, ROOT, "q9", difficulty="easy") == pytest.approx(sigmoid(-0.5 + 0.5))
    both = EloLearner(G=G, item_difficulty={"q1": 2.0}, item_bank=bank)
    assert both.p_correct(st, ROOT, "q1") == pytest.approx(sigmoid(-0.5 - 2.0))  # the explicit table first
    assert both.p_correct(st, ROOT, "q2") == pytest.approx(sigmoid(-0.5 + 1.0))  # then the bank
    assert L.mastery(st, ROOT) == base  # mastery is on a typical item
    p = L.update(st, ROOT, True, item="q1")
    assert p == pytest.approx(sigmoid(-0.5 - 1.0))


def test_a_difficulty_label_works_without_a_bank(G):
    L, st = EloLearner(G=G), LearnerState("t")
    assert L.p_correct(st, ROOT, "x", difficulty="hard") == pytest.approx(sigmoid(-1.0))
    assert L.update(st, ROOT, True, item="x", difficulty="easy") == pytest.approx(sigmoid(0.0))
    with pytest.raises(ValueError):
        L.p_correct(st, ROOT, difficulty="odd")


def test_a_bank_with_skill_betas_changes_mastery_consistently(G):
    """A skill-level difficulty is part of a typical item of that skill, so mastery and the all-skills
    pass agree with p_correct(item=None)."""
    bank = ItemBank(skill_betas={ROOT: 0.3})
    L, st = EloLearner(G=G, item_bank=bank), LearnerState("t")
    assert L.p_correct(st, ROOT) == pytest.approx(sigmoid(-0.5 - 0.3))
    assert L.mastery(st, ROOT) == L.p_correct(st, ROOT)
    assert L.effective_mastery_map(st) == {n: L.mastery(st, n) for n in G}
