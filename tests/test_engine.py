"""Learner model (PRD 5.2): multi-level Elo with graph transfer, checked against hand-computed numbers."""
import json
import math

import networkx as nx
import pytest

import engine

ROOT = "B1.1.1.1.1"          # grade 1, no prerequisites
NEXT = "B1.1.1.1.2"          # grade 1, only prerequisite is ROOT
THIRD = "B1.1.1.1.3"         # grade 1, only prerequisite is NEXT
OTHER_ROOT = "B1.1.2.1.1"    # grade 1, no prerequisites, unrelated to ROOT
TWO_PREREQS = "B2.1.2.2.1"   # grade 2, prerequisites B1.1.2.2.3 and B2.1.2.1.1


def sig(z):
    return 1 / (1 + math.exp(-z))


@pytest.fixture
def L(G):
    return engine.EloLearner(G=G)


@pytest.fixture
def st():
    return engine.LearnerState("t")


# ----------------------------------------------------------------------------- constants and helpers

def test_thresholds_match_the_prd():
    assert engine.MASTERY == 0.85
    assert engine.READY == 0.70


def test_sigmoid_values_and_extremes():
    assert engine.sigmoid(0) == 0.5
    assert engine.sigmoid(-0.5) == pytest.approx(0.3775406687981454, abs=1e-15)
    assert engine.sigmoid(2) + engine.sigmoid(-2) == pytest.approx(1.0)
    assert engine.sigmoid(1000) == 1.0
    assert engine.sigmoid(-1000) == 0.0          # no OverflowError on a hopeless learner


def test_graph_facts_these_tests_rely_on(G):
    assert list(G.predecessors(ROOT)) == [] and list(G.predecessors(OTHER_ROOT)) == []
    assert list(G.predecessors(NEXT)) == [ROOT]
    assert list(G.predecessors(THIRD)) == [NEXT]
    assert sorted(G.predecessors(TWO_PREREQS)) == ["B1.1.2.2.3", "B2.1.2.1.1"]
    assert G.nodes[ROOT]["grade"] == 1 and G.nodes[TWO_PREREQS]["grade"] == 2


# ----------------------------------------------------------------------------- Elo update

def test_first_update_matches_the_prd_example(L, st):
    """PRD 5.4: the first correct answer on B1.1.1.1.1 predicts 0.3775 and leaves mastery at 0.4839."""
    p = L.update(st, ROOT, True)
    assert p == pytest.approx(sig(-0.5), abs=1e-15)           # theta 0, prior -0.4*(1-1) - 0.5
    assert round(p, 4) == 0.3775
    assert st.theta == pytest.approx(0.3 * (1 - p), abs=1e-15)
    assert st.theta_k[ROOT] == pytest.approx(-0.5 + 0.4 * (1 - p), abs=1e-15)
    assert L.mastery(st, ROOT) == pytest.approx(sig(0.3 * (1 - p) - 0.5 + 0.4 * (1 - p)), abs=1e-15)
    assert round(L.mastery(st, ROOT), 4) == 0.4839


def test_first_wrong_answer_moves_both_abilities_down(L, st):
    p = L.update(st, ROOT, False)
    assert p == pytest.approx(sig(-0.5))
    assert st.theta == pytest.approx(-0.3 * p)                 # e = 0 - p
    assert st.theta_k[ROOT] == pytest.approx(-0.5 - 0.4 * p)
    assert L.mastery(st, ROOT) < p


def test_update_sequence_matches_a_hand_rolled_elo(L, st):
    """Replay correct, wrong, correct, correct on one skill with the formulas of PRD 5.2 written out here."""
    theta, tk = 0.0, -0.5
    for i, y in enumerate([1, 0, 1, 1]):                       # i is both n_u and n_{u,k} here
        p = sig(theta + tk)
        assert L.update(st, ROOT, bool(y)) == pytest.approx(p, abs=1e-12)
        theta += 0.3 / (1 + 0.05 * i) * (y - p)
        tk += 0.4 / (1 + 0.05 * i) * (y - p)
        assert st.theta == pytest.approx(theta, abs=1e-12)
        assert st.theta_k[ROOT] == pytest.approx(tk, abs=1e-12)
    assert L.p_correct(st, ROOT) == pytest.approx(sig(theta + tk), abs=1e-12)


def test_update_returns_the_prediction_made_before_the_answer(L, st):
    L.update(st, ROOT, True)
    before = L.p_correct(st, ROOT)
    assert L.update(st, ROOT, False) == before


def test_k_factors_decay_with_attempt_counts(L, st):
    """K_s shrinks with the learner's total attempts, K_k with the attempts on that skill."""
    st.theta, st.theta_k[ROOT] = 0.2, 0.1
    st.n = {"_all": 20, ROOT: 10}
    p = sig(0.2 + 0.1)
    assert L.update(st, ROOT, True) == pytest.approx(p)
    assert st.theta == pytest.approx(0.2 + 0.3 / (1 + 0.05 * 20) * (1 - p))        # K_s halved after 20 attempts
    assert st.theta_k[ROOT] == pytest.approx(0.1 + 0.4 / (1 + 0.05 * 10) * (1 - p))


def test_skill_k_is_fresh_for_a_new_skill_while_global_k_keeps_decaying(L, st):
    L.update(st, ROOT, True)
    theta1 = st.theta
    p2 = L.p_correct(st, OTHER_ROOT)
    L.update(st, OTHER_ROOT, True)
    assert st.theta - theta1 == pytest.approx(0.3 / 1.05 * (1 - p2))               # second attempt overall
    assert st.theta_k[OTHER_ROOT] == pytest.approx(-0.5 + 0.4 * (1 - p2))          # first attempt on this skill


def test_steps_get_smaller_as_evidence_accumulates(L, st):
    """With the error held equal, a later update moves the estimate less than an earlier one."""
    early = engine.LearnerState("early"); late = engine.LearnerState("late")
    late.n = {"_all": 40, ROOT: 40}
    L.update(early, ROOT, True); L.update(late, ROOT, True)
    assert 0 < late.theta < early.theta
    assert late.theta == pytest.approx(early.theta / 3)                             # 1 / (1 + 0.05*40)


def test_custom_k_factors_and_decay_are_used(G, st):
    L = engine.EloLearner(G=G, k_s=1.0, k_k=2.0, decay=1.0)
    st.n = {"_all": 1, ROOT: 3}
    p = L.update(st, ROOT, True)
    assert st.theta == pytest.approx(1.0 / 2 * (1 - p))
    assert st.theta_k[ROOT] == pytest.approx(-0.5 + 2.0 / 4 * (1 - p))


def test_correct_is_read_as_a_truth_value(L):
    a, b = engine.LearnerState("x"), engine.LearnerState("x")
    for y_bool, y_int in [(True, 1), (False, 0), (True, 1)]:
        L.update(a, ROOT, y_bool); L.update(b, ROOT, y_int)
    assert a == b


# ----------------------------------------------------------------------------- item difficulty

def test_item_difficulty_lowers_p_correct(G, st):
    L = engine.EloLearner(G=G, item_difficulty={"hard": 1.0, "easy": -1.0})
    assert L.p_correct(st, ROOT, "hard") == pytest.approx(sig(-0.5 - 1.0))
    assert L.p_correct(st, ROOT, "easy") == pytest.approx(sig(-0.5 + 1.0))
    assert L.p_correct(st, ROOT, "hard") < L.p_correct(st, ROOT) < L.p_correct(st, ROOT, "easy")


def test_unknown_or_missing_item_is_a_typical_item(G, st):
    L = engine.EloLearner(G=G, item_difficulty={"hard": 1.0})
    assert L.p_correct(st, ROOT, "never-seen") == L.p_correct(st, ROOT, None) == L.p_correct(st, ROOT)


def test_update_uses_item_difficulty_and_mastery_ignores_it(G, st):
    L = engine.EloLearner(G=G, item_difficulty={"hard": 1.0})
    p = L.update(st, ROOT, True, item="hard")
    assert p == pytest.approx(sig(-1.5))                        # a correct answer on a hard item is a bigger surprise
    assert st.theta == pytest.approx(0.3 * (1 - p))
    assert L.mastery(st, ROOT) == pytest.approx(sig(st.theta + st.theta_k[ROOT]))   # beta = 0


def test_item_difficulty_table_is_shared_not_copied(G, st):
    table = {}
    L = engine.EloLearner(G=G, item_difficulty=table)
    table["hard"] = 2.0
    assert L.p_correct(st, ROOT, "hard") == pytest.approx(sig(-0.5 - 2.0))


# ----------------------------------------------------------------------------- unseen-skill prior and graph transfer

def test_prior_of_every_unseen_skill_for_a_new_learner(G, L, st):
    """No prerequisite has been seen, so the prior is -0.4*(grade-1) - 0.5 for every indicator."""
    for n, d in G.nodes(data=True):
        assert L.p_correct(st, n) == pytest.approx(sig(-0.4 * (d["grade"] - 1) - 0.5), abs=1e-15), n
        assert L.mastery(st, n) == L.p_correct(st, n)


def test_prior_falls_with_grade_for_a_new_learner(G, L, st):
    by_grade = {d["grade"]: L.mastery(st, n) for n, d in G.nodes(data=True)}
    assert sorted(by_grade) == [1, 2, 3, 4, 5, 6]
    assert [round(by_grade[g], 4) for g in range(1, 7)] == [0.3775, 0.2891, 0.2142, 0.1545, 0.1091, 0.0759]
    assert all(by_grade[g] > by_grade[g + 1] for g in range(1, 6))


def test_prior_with_a_seen_prerequisite_borrows_half_its_offset(L, st):
    st.theta_k[ROOT] = 1.2
    assert L.p_correct(st, NEXT) == pytest.approx(sig(-0.4 * 0 + 0.5 * 1.2))
    st.theta_k[ROOT] = -2.0                                      # weak evidence transfers too
    assert L.p_correct(st, NEXT) == pytest.approx(sig(0.5 * -2.0))
    assert L.p_correct(st, NEXT) < sig(-0.5)                     # worse than knowing nothing about the learner


def test_prior_averages_only_the_prerequisites_that_were_seen(L, st):
    st.theta_k["B1.1.2.2.3"] = 1.0
    assert L.p_correct(st, TWO_PREREQS) == pytest.approx(sig(-0.4 * 1 + 0.5 * 1.0))           # one of two seen
    st.theta_k["B2.1.2.1.1"] = -0.4
    assert L.p_correct(st, TWO_PREREQS) == pytest.approx(sig(-0.4 * 1 + 0.5 * (1.0 - 0.4) / 2))


def test_prior_adds_the_global_ability(L, st):
    st.theta = 0.8
    assert L.p_correct(st, ROOT) == pytest.approx(sig(0.8 - 0.5))
    st.theta_k[ROOT] = 0.3
    assert L.p_correct(st, NEXT) == pytest.approx(sig(0.8 + 0.5 * 0.3))


def test_custom_grade_step_and_transfer(G, st):
    L = engine.EloLearner(G=G, grade_step=1.0, transfer=0.25)
    assert L.p_correct(st, TWO_PREREQS) == pytest.approx(sig(-1.0 * 1 - 0.5))
    st.theta_k["B1.1.2.2.3"] = 2.0
    assert L.p_correct(st, TWO_PREREQS) == pytest.approx(sig(-1.0 * 1 + 0.25 * 2.0))


def test_a_seen_skill_uses_its_own_offset_not_the_prior(L, st):
    st.theta_k[ROOT] = 3.0
    st.theta_k[NEXT] = -1.0
    assert L.p_correct(st, NEXT) == pytest.approx(sig(-1.0))


def test_b6_does_not_look_known_because_b1_is(G, L, st):
    """Master every B1 skill by the streak rule: no B6 skill may count as mastered, or even likely."""
    b1 = [n for n, d in G.nodes(data=True) if d["grade"] == 1]
    b6 = [n for n, d in G.nodes(data=True) if d["grade"] == 6]
    for n in b1:
        for _ in range(3):
            L.update(st, n, True)
    assert all(L.mastery(st, n) >= 0.95 for n in b1)
    assert all(L.mastery(st, n) < 0.5 for n in b6)
    assert max(L.mastery(st, n) for n in b6) < engine.READY < engine.MASTERY


def test_evidence_flows_to_the_successor(L, st):
    before = L.mastery(st, NEXT)
    L.update(st, ROOT, True)
    after = L.mastery(st, NEXT)
    assert after > before
    assert after == pytest.approx(sig(st.theta + 0.5 * st.theta_k[ROOT]))
    # an unrelated skill moves only through the global ability, so the successor gains more
    assert L.mastery(st, OTHER_ROOT) == pytest.approx(sig(st.theta - 0.5))
    assert after > L.mastery(st, OTHER_ROOT)


def test_poor_evidence_flows_to_the_successor_too(L, st):
    before = L.mastery(st, NEXT)
    for _ in range(4):
        L.update(st, ROOT, False)
    assert L.mastery(st, NEXT) < before
    assert L.mastery(st, NEXT) == pytest.approx(sig(st.theta + 0.5 * st.theta_k[ROOT]))


def test_transfer_reaches_one_hop_only(L, st):
    """THIRD's only prerequisite (NEXT) is unseen, so THIRD keeps the no-evidence prior."""
    L.update(st, ROOT, True)
    assert L.mastery(st, THIRD) == pytest.approx(sig(st.theta - 0.5))


def test_first_update_on_a_skill_starts_from_the_transferred_prior(L, st):
    L.update(st, ROOT, True)
    prior = 0.5 * st.theta_k[ROOT]
    p = L.update(st, NEXT, True)
    assert st.theta_k[NEXT] == pytest.approx(prior + 0.4 * (1 - p))
    # once NEXT has its own offset, later evidence on ROOT no longer changes it
    frozen = st.theta_k[NEXT]
    L.update(st, ROOT, False)
    assert st.theta_k[NEXT] == frozen


# ----------------------------------------------------------------------------- mastery and the streak rule

def test_three_correct_in_a_row_is_mastered(L, st):
    L.update(st, ROOT, True); L.update(st, ROOT, True)
    assert st.streak[ROOT] == 2
    assert L.mastery(st, ROOT) == L.p_correct(st, ROOT) < engine.MASTERY      # two in a row is not enough
    L.update(st, ROOT, True)
    assert st.streak[ROOT] == 3
    assert L.p_correct(st, ROOT) < 0.95                                       # the estimate alone is not there yet
    assert L.mastery(st, ROOT) == 0.95 >= engine.MASTERY


def test_a_wrong_answer_resets_the_streak(L, st):
    for _ in range(3):
        L.update(st, ROOT, True)
    assert L.mastery(st, ROOT) >= 0.95
    L.update(st, ROOT, False)
    assert st.streak[ROOT] == 0
    assert L.mastery(st, ROOT) == L.p_correct(st, ROOT) < 0.95


def test_streak_counts_consecutive_answers_only(L, st):
    for y in [True, True, False, True, True]:
        L.update(st, ROOT, y)
    assert st.streak[ROOT] == 2
    assert L.mastery(st, ROOT) == L.p_correct(st, ROOT)
    L.update(st, ROOT, True)
    assert st.streak[ROOT] == 3 and L.mastery(st, ROOT) >= 0.95


def test_streak_is_tracked_per_skill(L, st):
    for _ in range(3):
        L.update(st, ROOT, True)
    L.update(st, OTHER_ROOT, False)
    assert st.streak == {ROOT: 3, OTHER_ROOT: 0}
    assert L.mastery(st, ROOT) >= 0.95


def test_streak_mastery_never_lowers_a_higher_estimate(L, st):
    st.theta = 6.0
    st.streak[ROOT] = 3
    assert L.mastery(st, ROOT) == L.p_correct(st, ROOT) > 0.95


def test_streak_length_is_configurable(G, st):
    L = engine.EloLearner(G=G, streak_to_master=2)
    L.update(st, ROOT, True)
    assert L.mastery(st, ROOT) < 0.95
    L.update(st, ROOT, True)
    assert L.mastery(st, ROOT) == 0.95


def test_mastery_without_a_streak_is_p_correct_on_a_typical_item(G, st):
    L = engine.EloLearner(G=G, item_difficulty={"hard": 3.0})
    L.update(st, ROOT, True, item="hard")
    assert L.mastery(st, ROOT) == L.p_correct(st, ROOT, None)


# ----------------------------------------------------------------------------- state bookkeeping

def test_new_state_is_empty():
    st = engine.LearnerState("ama")
    assert (st.learner_id, st.theta, st.step) == ("ama", 0.0, 0)
    assert st.theta_k == st.n == st.last_seen == st.streak == {}


def test_states_do_not_share_dictionaries():
    a, b = engine.LearnerState("a"), engine.LearnerState("b")
    a.theta_k["x"] = 1.0; a.n["x"] = 1; a.last_seen["x"] = 1; a.streak["x"] = 1
    assert b.theta_k == b.n == b.last_seen == b.streak == {}


def test_bookkeeping_after_a_sequence_of_answers(L, st):
    for ind, y in [(ROOT, True), (ROOT, True), (OTHER_ROOT, False), (ROOT, False), (NEXT, True)]:
        L.update(st, ind, y)
    assert st.n == {"_all": 5, ROOT: 3, OTHER_ROOT: 1, NEXT: 1}
    assert st.step == 5
    assert st.last_seen == {ROOT: 4, OTHER_ROOT: 3, NEXT: 5}        # step index of the latest attempt, from 1
    assert st.streak == {ROOT: 0, OTHER_ROOT: 0, NEXT: 1}
    assert set(st.theta_k) == {ROOT, OTHER_ROOT, NEXT}
    assert st.learner_id == "t"


def test_one_learner_model_serves_many_learners(L):
    a, b = engine.LearnerState("a"), engine.LearnerState("b")
    L.update(a, ROOT, True)
    assert b == engine.LearnerState("b")
    L.update(b, ROOT, False)
    assert a.theta > 0 > b.theta


# ----------------------------------------------------------------------------- without a graph

def test_without_a_graph_the_offset_is_zero():
    L = engine.EloLearner()
    st = engine.LearnerState("t")
    assert L.G is None
    assert L.p_correct(st, "any-skill") == 0.5 == L.mastery(st, "any-skill")
    st.theta = 1.0
    assert L.p_correct(st, "B6.1.1.1.1") == pytest.approx(sig(1.0))             # no grade prior without a graph


def test_without_a_graph_any_skill_id_can_be_updated():
    L = engine.EloLearner(G=None)
    st = engine.LearnerState("t")
    assert L.update(st, "skill-42", True) == 0.5
    assert st.theta == pytest.approx(0.15) and st.theta_k["skill-42"] == pytest.approx(0.2)
    assert st.n == {"_all": 1, "skill-42": 1}


# ----------------------------------------------------------------------------- unknown indicators

def test_update_on_an_unknown_indicator_raises_and_leaves_the_state_alone(L, st):
    L.update(st, ROOT, True)
    before = st.to_dict()
    with pytest.raises(KeyError, match="unknown indicator 'B9.9.9.9.9'"):
        L.update(st, "B9.9.9.9.9", True)
    assert st.to_dict() == before


def test_reading_an_unknown_indicator_raises_a_clear_error(L, st):
    with pytest.raises(KeyError, match="unknown indicator 'NOPE'"):
        L.mastery(st, "NOPE")
    with pytest.raises(KeyError, match="unknown indicator 'NOPE'"):
        L.p_correct(st, "NOPE")


def test_works_on_any_graph_with_a_grade_attribute(st):
    g = nx.DiGraph()
    g.add_node("a", grade=1); g.add_node("b", grade=3); g.add_edge("a", "b")
    L = engine.EloLearner(G=g)
    assert L.p_correct(st, "b") == pytest.approx(sig(-0.4 * 2 - 0.5))
    L.update(st, "a", True)
    assert L.p_correct(st, "b") == pytest.approx(sig(st.theta - 0.8 + 0.5 * st.theta_k["a"]))


# ----------------------------------------------------------------------------- serialisation

def practised(L):
    st = engine.LearnerState("ama")
    for ind, y in [(ROOT, True), (ROOT, True), (ROOT, True), (NEXT, False), (OTHER_ROOT, True), (NEXT, True)]:
        L.update(st, ind, y)
    return st


def test_to_dict_has_exactly_the_prd_fields(L):
    d = practised(L).to_dict()
    assert set(d) == {"learner_id", "theta", "theta_k", "n", "last_seen", "streak", "step"}
    assert d["learner_id"] == "ama" and d["step"] == 6 and d["n"]["_all"] == 6


def test_state_survives_a_json_round_trip_exactly(L):
    st = practised(L)
    back = engine.LearnerState.from_dict(json.loads(json.dumps(st.to_dict())))
    assert back == st
    assert isinstance(back, engine.LearnerState)
    for n in (ROOT, NEXT, THIRD, OTHER_ROOT, TWO_PREREQS):
        assert L.mastery(back, n) == L.mastery(st, n)


def test_restored_state_keeps_learning_the_same_way(L):
    st = practised(L)
    back = engine.LearnerState.from_dict(json.loads(json.dumps(st.to_dict())))
    assert L.update(back, THIRD, True) == L.update(st, THIRD, True)
    assert back == st


def test_to_dict_and_from_dict_copy_the_dictionaries(L):
    st = practised(L)
    d = st.to_dict()
    d["theta_k"][ROOT] = 99.0; d["n"]["_all"] = 0
    assert st.theta_k[ROOT] != 99.0 and st.n["_all"] == 6
    src = st.to_dict()
    back = engine.LearnerState.from_dict(src)
    back.streak[ROOT] = 0; back.last_seen["x"] = 1
    assert src["streak"][ROOT] == 3 and "x" not in src["last_seen"]


def test_from_dict_needs_only_a_learner_id():
    assert engine.LearnerState.from_dict({"learner_id": "kofi"}) == engine.LearnerState("kofi")


def test_from_dict_without_a_learner_id_is_an_error():
    with pytest.raises(ValueError, match="learner_id"):
        engine.LearnerState.from_dict({"theta": 0.3})


def test_from_dict_ignores_unknown_keys_and_coerces_numbers():
    st = engine.LearnerState.from_dict({"learner_id": "esi", "theta": 1, "step": 2.0, "n": {"_all": 2.0},
                                        "schema": 7})
    assert st.theta == 1.0 and isinstance(st.theta, float)
    assert st.step == 2 and isinstance(st.step, int)
    assert st.n == {"_all": 2} and isinstance(st.n["_all"], int)
