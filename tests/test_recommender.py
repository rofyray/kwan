"""Recommender (PRD 5.3): candidate rules, score formula, goal term, ordering and rationale wording."""
import math
import random
import statistics
import time

import networkx as nx
import pytest

import context
import engine

EM_DASH = chr(0x2014)
GOAL = "B4.1.1.3.3"
ITEM_KEYS = {"indicator", "score", "mastery", "grade", "strand", "description", "rationale", "ghana_context"}
EVERYTHING = 10_000          # a k larger than the graph


# ----------------------------------------------------------------------------- fixtures and hand calculations

@pytest.fixture(scope="module")
def L(G):
    return engine.EloLearner(G=G)


@pytest.fixture(scope="module")
def R(G, L):
    return engine.Recommender(G, L)


@pytest.fixture(scope="module")
def desc(G):
    """Number of descendants per skill, computed here rather than read from the recommender."""
    return {n: len(nx.descendants(G, n)) for n in G}


@pytest.fixture
def fresh():
    return engine.LearnerState("fresh")


@pytest.fixture
def b1(L):
    """The PRD Appendix B.1 learner: 3 correct answers on each of B1.1.1.1.1 and B1.1.1.1.2."""
    st = engine.LearnerState("demo")
    for ind in ["B1.1.1.1.1", "B1.1.1.1.2"]:
        for _ in range(3):
            L.update(st, ind, True)
    return st


@pytest.fixture
def gapped(G, L):
    """The demo pupil of app.py: seeded answers on the first 60 skills in syllabus order."""
    st = engine.LearnerState("gapped"); rng = random.Random(7)
    for n in sorted(G, key=lambda x: (G.nodes[x]["grade"], x))[:60]:
        if rng.random() < 0.35:
            L.update(st, n, rng.random() < (0.85 if G.nodes[n]["grade"] <= 3 else 0.2))
    return st


def practised(G, L, seed, steps=150):
    """A learner who followed the engine's own top pick for a while, answering right 70% of the time."""
    R = engine.Recommender(G, L); st = engine.LearnerState(f"p{seed}"); rng = random.Random(seed)
    for _ in range(steps):
        recs = R.recommend(st, k=3)["recommendations"]
        L.update(st, rng.choice(recs)["indicator"], rng.random() < 0.7)
    return st


@pytest.fixture
def states(G, L, fresh, b1, gapped):
    return [fresh, b1, gapped, practised(G, L, 1), practised(G, L, 2)]


def hand_candidates(G, L, st, grade_max=None):
    """PRD 5.3: not mastered, within grade_max, every prerequisite at mastery >= 0.70."""
    out = set()
    for n, d in G.nodes(data=True):
        if L.mastery(st, n) >= 0.85: continue
        if grade_max is not None and d["grade"] > grade_max: continue
        if any(L.mastery(st, p) < 0.70 for p in G.predecessors(n)): continue
        out.add(n)
    return out


def hand_goal_term(G, n, goal):
    if goal is None: return 0.5
    if n == goal or n in nx.ancestors(G, goal):
        return 1 / (1 + nx.shortest_path_length(G, n, goal))
    return 0.0


def hand_score(G, L, st, n, desc, goal=None):
    """score = 0.30 readiness + 0.25 zpd + 0.25 goal + 0.15 unlock + 0.05 recency, as written in the PRD."""
    m = L.mastery(st, n)
    readiness = min((L.mastery(st, p) for p in G.predecessors(n)), default=1.0)
    zpd = 1 - abs(m - 0.6) / 0.6
    unlock = desc[n] / max(desc.values())
    since = st.step - st.last_seen.get(n, -99)            # a never-practised skill counts as last seen long ago
    recency = 1 - math.exp(-since / 5)
    return 0.30 * readiness + 0.25 * zpd + 0.25 * hand_goal_term(G, n, goal) + 0.15 * unlock + 0.05 * recency


def hand_rationale(G, L, st, n, desc, goal=None):
    """The rationale assembled from the PRD wording table."""
    parts = []
    preds = list(G.predecessors(n))
    if preds:
        best = max(preds, key=lambda p: L.mastery(st, p))
        parts.append(f"Builds on {best}, where you are at {L.mastery(st, best):.0%}.")
    else:
        parts.append("A starting skill with nothing before it.")
    m = L.mastery(st, n)
    if m < 0.5:
        parts.append(f"Still new to you ({m:.0%}), so expect a few tries.")
    elif m < 0.75:
        parts.append(f"You are at {m:.0%}, close enough to master with practice.")
    else:
        parts.append(f"Nearly there at {m:.0%}. A few correct answers will lock it in.")
    if goal is not None and n != goal and n in nx.ancestors(G, goal):
        parts.append(f"On the way to your goal, {nx.shortest_path_length(G, n, goal)} step(s) away.")
    if desc[n]:
        parts.append(f"Opens the way to {desc[n]} later skills.")
    return " ".join(parts)


def ids(res):
    return [r["indicator"] for r in res["recommendations"]]


def without_latency(res):
    return res["recommendations"]


class FixedLearner:
    """A learner model with masteries set by hand, to test the recommender on exact numbers."""
    def __init__(self, m): self.m = m
    def mastery(self, st, n): return self.m[n]


def tiny_graph():
    """a -> b -> c <- x, and d on its own. Descendants: a 2, b 1, x 1, c 0, d 0."""
    g = nx.DiGraph()
    for n, grade, strand in [("a", 1, 1), ("b", 1, 1), ("c", 2, 1), ("d", 1, 3), ("x", 2, 4)]:
        g.add_node(n, grade=grade, strand=strand, strand_name=f"Strand {strand}", description=f"Skill {n}")
    g.add_edges_from([("a", "b"), ("b", "c"), ("x", "c")])
    return g


NEVER_SEEN = 1 - math.exp(-99 / 5)          # recency of a skill a new learner has never practised (about 1)


# ----------------------------------------------------------------------------- response shape

def test_response_shape(G, R, fresh):
    res = R.recommend(fresh)
    assert set(res) == {"recommendations", "latency_ms"}
    assert len(res["recommendations"]) == 3                       # k defaults to 3
    for r in res["recommendations"]:
        assert set(r) == ITEM_KEYS
        assert isinstance(r["indicator"], str) and isinstance(r["score"], float)
        assert isinstance(r["mastery"], float) and isinstance(r["grade"], int)
        assert all(isinstance(r[key], str) and r[key] for key in ("strand", "description", "rationale", "ghana_context"))
    assert isinstance(res["latency_ms"], float) and res["latency_ms"] >= 0


def test_items_carry_the_graph_attributes_and_rounded_numbers(G, L, R, states):
    for st in states:
        for r in R.recommend(st, k=EVERYTHING)["recommendations"]:
            node = G.nodes[r["indicator"]]
            assert r["grade"] == node["grade"]
            assert r["strand"] == node["strand_name"]
            assert r["description"] == node["description"]
            assert r["mastery"] == round(L.mastery(st, r["indicator"]), 3)
            assert r["score"] == round(r["score"], 4)


def test_recommend_does_not_change_the_learner(R, gapped):
    before = gapped.to_dict()
    R.recommend(gapped, k=5, goal=GOAL, grade_max=4)
    assert gapped.to_dict() == before


def test_recommend_is_deterministic(R, gapped):
    first = without_latency(R.recommend(gapped, k=EVERYTHING, goal=GOAL))
    assert without_latency(R.recommend(gapped, k=EVERYTHING, goal=GOAL)) == first


# ----------------------------------------------------------------------------- candidates

def test_candidates_are_exactly_the_ready_unmastered_skills(G, L, R, states):
    for st in states:
        got = ids(R.recommend(st, k=EVERYTHING))
        assert len(got) == len(set(got))
        assert set(got) == hand_candidates(G, L, st)


def test_candidates_exclude_mastered_skills(G, L, R, states):
    for st in states:
        for n in ids(R.recommend(st, k=EVERYTHING)):
            assert L.mastery(st, n) < engine.MASTERY


def test_candidates_are_all_ready(G, L, R, states):
    for st in states:
        for n in ids(R.recommend(st, k=EVERYTHING)):
            assert all(L.mastery(st, p) >= engine.READY for p in G.predecessors(n))


def test_streak_mastered_skills_are_not_recommended(L, R, b1):
    """Both practised skills are mastered by the 3-in-a-row rule alone (their estimates are below 85%)."""
    for n in ("B1.1.1.1.1", "B1.1.1.1.2"):
        assert L.p_correct(b1, n) < engine.MASTERY <= L.mastery(b1, n)
        assert n not in ids(R.recommend(b1, k=EVERYTHING))


def test_a_skill_appears_once_its_prerequisite_is_ready(L, R, fresh, b1):
    assert "B1.1.1.1.2" not in ids(R.recommend(fresh, k=EVERYTHING))       # prerequisite at 38%
    assert "B1.1.1.1.3" not in ids(R.recommend(fresh, k=EVERYTHING))
    assert "B1.1.1.1.3" in ids(R.recommend(b1, k=EVERYTHING))              # prerequisite now mastered


def test_a_new_learner_is_offered_only_starting_skills(G, R, fresh):
    got = ids(R.recommend(fresh, k=EVERYTHING))
    assert set(got) == {n for n in G if G.in_degree(n) == 0}
    assert ids(R.recommend(fresh, k=3)) == ["B1.1.1.1.1", "B1.1.2.1.1", "B1.3.3.1.1"]


def test_candidate_thresholds_are_inclusive_at_the_boundary():
    g = tiny_graph(); st = engine.LearnerState("t")
    at = engine.Recommender(g, FixedLearner({"a": 0.70, "b": 0.6, "c": 0.3, "d": 0.85, "x": 0.8499}))
    assert set(ids(at.recommend(st, k=9))) == {"a", "b", "x"}               # b ready at exactly 0.70, d mastered at 0.85
    below = engine.Recommender(g, FixedLearner({"a": 0.6999, "b": 0.6, "c": 0.3, "d": 0.8499, "x": 0.85}))
    assert set(ids(below.recommend(st, k=9))) == {"a", "d"}                 # b no longer ready, x now mastered


def test_every_prerequisite_must_be_ready_not_just_one():
    g = tiny_graph(); st = engine.LearnerState("t")
    r = engine.Recommender(g, FixedLearner({"a": 0.9, "b": 0.9, "c": 0.3, "d": 0.9, "x": 0.5}))
    assert "c" not in ids(r.recommend(st, k=9))                             # b is ready, x is not
    r = engine.Recommender(g, FixedLearner({"a": 0.9, "b": 0.9, "c": 0.3, "d": 0.9, "x": 0.75}))
    assert "c" in ids(r.recommend(st, k=9))


# ----------------------------------------------------------------------------- grade cap

@pytest.mark.parametrize("cap", [1, 2, 3, 4, 5, 6])
def test_grade_max_caps_the_class_level(G, L, R, gapped, cap):
    full = R.recommend(gapped, k=EVERYTHING)["recommendations"]
    capped = R.recommend(gapped, k=EVERYTHING, grade_max=cap)["recommendations"]
    assert all(r["grade"] <= cap for r in capped)
    assert capped == [r for r in full if r["grade"] <= cap]                 # same scores, same order, fewer skills
    assert set(ids({"recommendations": capped})) == hand_candidates(G, L, gapped, grade_max=cap)


def test_grade_max_changes_what_a_new_learner_sees(R, fresh):
    grades = {r["grade"] for r in R.recommend(fresh, k=EVERYTHING)["recommendations"]}
    assert grades == {1, 4, 5, 6}                                           # starting skills exist in these classes
    assert {r["grade"] for r in R.recommend(fresh, k=EVERYTHING, grade_max=4)["recommendations"]} == {1, 4}


def test_no_cap_and_a_cap_of_six_are_the_same(R, gapped):
    assert without_latency(R.recommend(gapped, k=EVERYTHING, grade_max=None)) == \
           without_latency(R.recommend(gapped, k=EVERYTHING, grade_max=6))


def test_a_cap_below_the_first_class_gives_nothing(R, gapped):
    assert R.recommend(gapped, k=3, grade_max=0)["recommendations"] == []


# ----------------------------------------------------------------------------- score

def test_weights_match_the_prd(R):
    assert engine.Recommender.W == dict(readiness=0.30, zpd=0.25, goal=0.25, unlock=0.15, recency=0.05)
    assert sum(R.W.values()) == pytest.approx(1.0)


def test_score_components_by_hand_for_one_candidate(G, L, R, b1, desc):
    """B1.1.1.1.3 for the Appendix B.1 learner with goal B4.1.1.3.3, every component written out."""
    n = "B1.1.1.1.3"
    m = L.mastery(b1, n)
    assert round(m, 3) == 0.713
    readiness = L.mastery(b1, "B1.1.1.1.2")
    assert readiness == 0.95                                                # its one prerequisite, streak-mastered
    zpd = 1 - abs(m - 0.6) / 0.6
    assert nx.shortest_path_length(G, n, GOAL) == 19
    goal = 1 / (1 + 19)
    assert desc[n] == 56 and max(desc.values()) == 58
    unlock = 56 / 58
    recency = 1 - math.exp(-(6 + 99) / 5)                                   # 6 steps taken, skill never practised
    score = 0.30 * readiness + 0.25 * zpd + 0.25 * goal + 0.15 * unlock + 0.05 * recency
    top = R.recommend(b1, k=1, goal=GOAL)["recommendations"][0]
    assert top["indicator"] == n
    assert top["score"] == round(score, 4) == 0.6953


def test_every_returned_score_matches_the_formula(G, L, R, states, desc):
    for st in states:
        for goal in (None, GOAL, "B2.1.1.1.1"):
            for r in R.recommend(st, k=EVERYTHING, goal=goal)["recommendations"]:
                assert r["score"] == pytest.approx(hand_score(G, L, st, r["indicator"], desc, goal), abs=5.1e-5), \
                    (st.learner_id, goal, r["indicator"])


def test_scores_on_a_tiny_graph_with_fixed_masteries():
    g = tiny_graph(); st = engine.LearnerState("t")
    r = engine.Recommender(g, FixedLearner({"a": 0.9, "b": 0.6, "c": 0.3, "d": 0.5, "x": 0.7}))
    assert r.desc_count == {"a": 2, "b": 1, "c": 0, "d": 0, "x": 1} and r.max_desc == 2
    recs = r.recommend(st, k=9)["recommendations"]
    assert [x["indicator"] for x in recs] == ["b", "x", "d"]                # a mastered, c not ready (b at 60%)
    by_id = {x["indicator"]: x for x in recs}
    #                           readiness     zpd                         goal          unlock        recency
    assert by_id["b"]["score"] == round(0.30 * 0.9 + 0.25 * 1.0 + 0.25 * 0.5 + 0.15 * 1 / 2 + 0.05 * NEVER_SEEN, 4) == 0.77
    assert by_id["x"]["score"] == round(0.30 * 1.0 + 0.25 * (1 - 0.1 / 0.6) + 0.25 * 0.5 + 0.15 * 1 / 2 + 0.05 * NEVER_SEEN, 4) == 0.7583
    assert by_id["d"]["score"] == round(0.30 * 1.0 + 0.25 * (1 - 0.1 / 0.6) + 0.25 * 0.5 + 0.15 * 0 / 2 + 0.05 * NEVER_SEEN, 4) == 0.6833
    assert [x["mastery"] for x in recs] == [0.6, 0.7, 0.5]


def test_zpd_peaks_at_sixty_percent():
    g = nx.DiGraph()
    for n in "pqr":
        g.add_node(n, grade=1, strand=1, strand_name="Number", description=n)
    r = engine.Recommender(g, FixedLearner({"p": 0.6, "q": 0.4, "r": 0.8}))
    recs = {x["indicator"]: x["score"] for x in r.recommend(engine.LearnerState("t"), k=3)["recommendations"]}
    assert recs["p"] > recs["q"] == recs["r"]                               # equally far from 60% on either side
    assert recs["p"] - recs["q"] == pytest.approx(0.25 * (0.2 / 0.6), abs=1.1e-4)


def test_readiness_is_the_weakest_prerequisite():
    g = tiny_graph(); st = engine.LearnerState("t")
    r = engine.Recommender(g, FixedLearner({"a": 0.9, "b": 0.8, "c": 0.6, "d": 0.9, "x": 0.72}))
    c = next(x for x in r.recommend(st, k=9)["recommendations"] if x["indicator"] == "c")
    assert c["score"] == round(0.30 * 0.72 + 0.25 * 1.0 + 0.25 * 0.5 + 0.15 * 0 + 0.05 * NEVER_SEEN, 4)


def test_a_skill_just_practised_has_no_recency_bonus(G, L, R, desc):
    st = engine.LearnerState("t")
    L.update(st, "B1.1.1.1.1", False)                                       # step 1, last seen at step 1
    r = next(x for x in R.recommend(st, k=EVERYTHING)["recommendations"] if x["indicator"] == "B1.1.1.1.1")
    m = L.mastery(st, "B1.1.1.1.1")
    no_recency = 0.30 * 1.0 + 0.25 * (1 - abs(m - 0.6) / 0.6) + 0.25 * 0.5 + 0.15 * 58 / 58
    assert r["score"] == pytest.approx(no_recency, abs=5.1e-5)


def test_recency_grows_back_as_other_skills_are_practised():
    g = tiny_graph(); st = engine.LearnerState("t")
    r = engine.Recommender(g, FixedLearner({"a": 0.9, "b": 0.6, "c": 0.3, "d": 0.5, "x": 0.7}))
    st.step, st.last_seen["d"] = 10, 10
    scores = []
    for later in (0, 5, 10, 50):
        st.step = 10 + later
        d = next(x for x in r.recommend(st, k=9)["recommendations"] if x["indicator"] == "d")
        assert d["score"] == pytest.approx(0.30 + 0.25 * (1 - 0.1 / 0.6) + 0.125 + 0.05 * (1 - math.exp(-later / 5)), abs=5.1e-5)
        scores.append(d["score"])
    assert scores == sorted(scores) and scores[0] < scores[-1]


# ----------------------------------------------------------------------------- goal term

def test_goal_term_is_one_over_one_plus_hops_for_ancestors_and_zero_otherwise(G, R, gapped):
    """The candidate set does not depend on the goal, so the score shift isolates the goal term."""
    no_goal = {r["indicator"]: r["score"] for r in R.recommend(gapped, k=EVERYTHING)["recommendations"]}
    with_goal = {r["indicator"]: r["score"] for r in R.recommend(gapped, k=EVERYTHING, goal=GOAL)["recommendations"]}
    assert set(no_goal) == set(with_goal)
    ancestors = nx.ancestors(G, GOAL)
    seen_ancestor = seen_other = False
    for n in no_goal:
        if n in ancestors:
            term = 1 / (1 + nx.shortest_path_length(G, n, GOAL)); seen_ancestor = True
        else:
            term = 0.0; seen_other = True
        assert with_goal[n] - no_goal[n] == pytest.approx(0.25 * (term - 0.5), abs=1.1e-4), n
    assert seen_ancestor and seen_other


def test_goal_term_on_a_tiny_graph():
    g = tiny_graph(); st = engine.LearnerState("t")
    r = engine.Recommender(g, FixedLearner({"a": 0.9, "b": 0.6, "c": 0.3, "d": 0.5, "x": 0.7}))
    recs = {x["indicator"]: x["score"] for x in r.recommend(st, k=9, goal="c")["recommendations"]}
    assert recs["b"] == 0.77                       # one hop from c: 1/(1+1) = 0.5, same as with no goal
    assert recs["x"] == 0.7583
    assert recs["d"] == round(0.6833 - 0.25 * 0.5, 4)        # not an ancestor of c: the goal term drops to 0


def test_the_goal_itself_gets_the_full_goal_term(R, fresh):
    plain = R.recommend(fresh, k=1)["recommendations"][0]
    aimed = R.recommend(fresh, k=1, goal="B1.1.1.1.1")["recommendations"][0]
    assert plain["indicator"] == aimed["indicator"] == "B1.1.1.1.1"
    assert aimed["score"] - plain["score"] == pytest.approx(0.25 * (1.0 - 0.5), abs=1.1e-4)
    assert "goal" not in aimed["rationale"]                                 # zero hops away: nothing to say


def test_a_goal_moves_its_ancestors_up_the_ranking(G, R, fresh):
    goal = "B1.4.1.1.2"
    assert list(G.predecessors(goal)) == ["B1.4.1.1.1"]
    plain = ids(R.recommend(fresh, k=EVERYTHING))
    aimed = ids(R.recommend(fresh, k=EVERYTHING, goal=goal))
    assert plain[0] == "B1.1.1.1.1" and plain.index("B1.4.1.1.1") == 5
    assert aimed[0] == "B1.4.1.1.1"
    others = [n for n in plain if n != "B1.4.1.1.1"]
    assert [n for n in aimed if n != "B1.4.1.1.1"] == others               # skills off the path keep their order


def test_the_goal_reorders_the_appendix_b1_learner(R, b1):
    assert ids(R.recommend(b1, k=2)) == ["B1.1.2.1.1", "B1.1.1.1.3"]
    assert ids(R.recommend(b1, k=2, goal=GOAL)) == ["B1.1.1.1.3", "B1.1.2.1.1"]


def test_an_empty_goal_means_no_goal(R, gapped):
    assert without_latency(R.recommend(gapped, k=5, goal="")) == without_latency(R.recommend(gapped, k=5, goal=None))


def test_an_unknown_goal_raises_a_clear_error(R, fresh):
    with pytest.raises(KeyError, match="unknown goal 'NOPE'"):
        R.recommend(fresh, goal="NOPE")


# ----------------------------------------------------------------------------- ordering and k

def test_results_are_sorted_by_score_descending(R, states):
    for st in states:
        for goal in (None, GOAL):
            scores = [r["score"] for r in R.recommend(st, k=EVERYTHING, goal=goal)["recommendations"]]
            assert scores == sorted(scores, reverse=True)


def test_equal_scores_are_ordered_by_indicator_id_highest_first(G, L, R, fresh, desc):
    """Two B6 starting skills score exactly the same for a new learner; the order between them is fixed."""
    assert hand_score(G, L, fresh, "B6.4.2.2.1", desc) == hand_score(G, L, fresh, "B6.3.3.5.1", desc)
    for _ in range(3):
        tail = R.recommend(fresh, k=EVERYTHING)["recommendations"][-2:]
        assert [r["indicator"] for r in tail] == ["B6.4.2.2.1", "B6.3.3.5.1"]
        assert tail[0]["score"] == tail[1]["score"]


def test_ties_do_not_depend_on_node_insertion_order():
    def build(order):
        g = nx.DiGraph()
        for n in order:
            g.add_node(n, grade=1, strand=1, strand_name="Number", description=n)
        return engine.Recommender(g, FixedLearner({n: 0.5 for n in order}))
    st = engine.LearnerState("t")
    assert ids(build(["s1", "s2", "s3"]).recommend(st, k=3)) == ["s3", "s2", "s1"]
    assert ids(build(["s2", "s3", "s1"]).recommend(st, k=3)) == ["s3", "s2", "s1"]


def test_top_k_is_a_prefix_of_the_full_ranking(R, gapped):
    full = ids(R.recommend(gapped, k=EVERYTHING, goal=GOAL))
    for k in (1, 2, 3, 10):
        assert ids(R.recommend(gapped, k=k, goal=GOAL)) == full[:k]


def test_k_larger_than_the_candidate_count_returns_every_candidate(G, L, R, fresh, gapped):
    assert len(R.recommend(fresh, k=EVERYTHING)["recommendations"]) == len(hand_candidates(G, L, fresh)) == 16
    assert len(R.recommend(gapped, k=EVERYTHING)["recommendations"]) == len(hand_candidates(G, L, gapped))
    assert len(R.recommend(fresh, k=17)["recommendations"]) == 16


def test_k_zero_returns_nothing(R, fresh):
    res = R.recommend(fresh, k=0)
    assert res["recommendations"] == [] and res["latency_ms"] >= 0


def test_negative_k_is_rejected(R, fresh):
    with pytest.raises(ValueError, match="k must be 0 or more"):
        R.recommend(fresh, k=-1)


@pytest.mark.parametrize("bad", ["3", 2.0, None])
def test_non_integer_k_is_rejected(R, fresh, bad):
    with pytest.raises(TypeError, match="k must be an integer"):
        R.recommend(fresh, k=bad)


def test_integer_like_k_is_accepted(R, fresh):
    np = pytest.importorskip("numpy")
    assert len(R.recommend(fresh, k=np.int64(2))["recommendations"]) == 2


# ----------------------------------------------------------------------------- rationale

def rationale_for(masteries, n, goal=None):
    g = tiny_graph()
    r = engine.Recommender(g, FixedLearner(masteries))
    return next(x for x in r.recommend(engine.LearnerState("t"), k=9, goal=goal)["recommendations"]
                if x["indicator"] == n)["rationale"]


BASE = {"a": 0.9, "b": 0.6, "c": 0.3, "d": 0.5, "x": 0.7}


def test_rationale_with_prerequisites_names_the_skill_it_builds_on():
    assert rationale_for(BASE, "b") == ("Builds on a, where you are at 90%. "
                                        "You are at 60%, close enough to master with practice. "
                                        "Opens the way to 1 later skills.")


def test_rationale_names_the_strongest_prerequisite():
    m = {"a": 0.9, "b": 0.74, "c": 0.3, "d": 0.5, "x": 0.81}
    assert rationale_for(m, "c") == "Builds on x, where you are at 81%. Still new to you (30%), so expect a few tries."
    m = {"a": 0.9, "b": 0.84, "c": 0.3, "d": 0.5, "x": 0.81}
    assert rationale_for(m, "c").startswith("Builds on b, where you are at 84%.")


def test_rationale_without_prerequisites():
    assert rationale_for(BASE, "d") == ("A starting skill with nothing before it. "
                                        "You are at 50%, close enough to master with practice.")


@pytest.mark.parametrize("mastery, sentence", [
    (0.07, "Still new to you (7%), so expect a few tries."),
    (0.30, "Still new to you (30%), so expect a few tries."),
    (0.50, "You are at 50%, close enough to master with practice."),          # 50% opens the middle band
    (0.62, "You are at 62%, close enough to master with practice."),
    (0.75, "Nearly there at 75%. A few correct answers will lock it in."),    # 75% opens the top band
    (0.84, "Nearly there at 84%. A few correct answers will lock it in."),
])
def test_rationale_mastery_bands(mastery, sentence):
    assert rationale_for({**BASE, "d": mastery}, "d") == f"A starting skill with nothing before it. {sentence}"


def test_rationale_on_the_way_to_the_goal_gives_the_hop_count():
    m = {"a": 0.6, "b": 0.6, "c": 0.3, "d": 0.5, "x": 0.7}
    assert rationale_for(m, "a", goal="c") == ("A starting skill with nothing before it. "
                                               "You are at 60%, close enough to master with practice. "
                                               "On the way to your goal, 2 step(s) away. "
                                               "Opens the way to 2 later skills.")
    assert "On the way to your goal, 1 step(s) away." in rationale_for(BASE, "x", goal="c")
    assert "goal" not in rationale_for(m, "d", goal="c")                    # d is not on any path to c
    assert "goal" not in rationale_for(m, "a")                              # no goal set


def test_rationale_opens_the_way_counts_all_later_skills(G, R, fresh, desc):
    recs = {r["indicator"]: r["rationale"] for r in R.recommend(fresh, k=EVERYTHING)["recommendations"]}
    assert recs["B1.1.1.1.1"] == ("A starting skill with nothing before it. "
                                  "Still new to you (38%), so expect a few tries. "
                                  "Opens the way to 58 later skills.")
    assert desc["B1.1.1.1.1"] == 58
    assert desc["B5.3.4.2.2"] == 0                                          # nothing after it: no such sentence
    assert recs["B5.3.4.2.2"] == "A starting skill with nothing before it. Still new to you (11%), so expect a few tries."


def test_every_rationale_matches_the_prd_templates(G, L, R, states, desc):
    seen_goal_sentence = False
    for st in states:
        for goal in (None, GOAL, "B2.1.1.1.1"):
            for r in R.recommend(st, k=EVERYTHING, goal=goal)["recommendations"]:
                assert r["rationale"] == hand_rationale(G, L, st, r["indicator"], desc, goal)
                seen_goal_sentence |= "On the way to your goal" in r["rationale"]
    assert seen_goal_sentence


def test_rationale_covers_every_branch_on_the_real_graph(R, states):
    text = " ".join(r["rationale"] for st in states
                    for r in R.recommend(st, k=EVERYTHING, goal=GOAL)["recommendations"])
    for fragment in ("Builds on B", "A starting skill with nothing before it.", "Still new to you (",
                     "close enough to master with practice.", "A few correct answers will lock it in.",
                     "On the way to your goal, ", " step(s) away.", "Opens the way to "):
        assert fragment in text, fragment


# ----------------------------------------------------------------------------- Ghana context

def test_ghana_context_comes_from_the_context_bank(G, R, states):
    bank = {s for options in context.CONTEXTS.values() for s in options}
    for st in states:
        for r in R.recommend(st, k=EVERYTHING)["recommendations"]:
            node = G.nodes[r["indicator"]]
            assert r["ghana_context"] == context.practice_setting(r["indicator"], node["strand"], node["grade"])
            assert r["ghana_context"] in bank


def test_ghana_context_is_empty_for_a_strand_without_settings():
    g = nx.DiGraph()
    g.add_node("z", grade=1, strand=9, strand_name="Elsewhere", description="z")
    rec = engine.Recommender(g, FixedLearner({"z": 0.5})).recommend(engine.LearnerState("t"))["recommendations"]
    assert [r["ghana_context"] for r in rec] == [""]


# ----------------------------------------------------------------------------- PRD Appendix B.1

def test_appendix_b1_sample_reproduces_exactly(G, R, b1):
    res = R.recommend(b1, k=3, goal=GOAL)
    expected = [
        ("B1.1.1.1.3", 0.6953, 0.713, 1, "Number",
         "Builds on B1.1.1.1.2, where you are at 95%. You are at 71%, close enough to master with practice. "
         "On the way to your goal, 19 step(s) away. Opens the way to 56 later skills.",
         "Use 1 cedi and 2 cedi coins to make 5 cedis in as many ways as you can."),
        ("B1.1.2.1.1", 0.6951, 0.549, 1, "Number",
         "A starting skill with nothing before it. You are at 55%, close enough to master with practice. "
         "Opens the way to 45 later skills.",
         "Share 12 groundnuts equally between 3 friends after school. How many does each friend get?"),
        ("B1.3.3.1.1", 0.6538, 0.549, 1, "Geometry and Measurement",
         "A starting skill with nothing before it. You are at 55%, close enough to master with practice. "
         "Opens the way to 29 later skills.",
         "Find circles, squares and triangles in Adinkra symbols printed on cloth at home."),
    ]
    assert len(res["recommendations"]) == 3
    for r, (ind, score, mastery, grade, strand, rationale, setting) in zip(res["recommendations"], expected):
        assert r["indicator"] == ind
        assert r["score"] == score
        assert r["mastery"] == mastery
        assert r["grade"] == grade
        assert r["strand"] == strand
        assert r["rationale"] == rationale
        assert r["ghana_context"] == setting
        assert r["description"] == G.nodes[ind]["description"]
    assert res["latency_ms"] < 5


# ----------------------------------------------------------------------------- edge cases

def test_a_learner_who_has_mastered_everything_gets_an_empty_list(G, L, R):
    st = engine.LearnerState("done")
    for n in G:
        for _ in range(3):
            L.update(st, n, True)
    assert all(L.mastery(st, n) >= engine.MASTERY for n in G)
    for kwargs in ({}, {"k": EVERYTHING}, {"goal": GOAL}, {"grade_max": 2}):
        res = R.recommend(st, **kwargs)
        assert res["recommendations"] == [] and res["latency_ms"] >= 0


def test_a_very_able_learner_who_never_practised_also_gets_an_empty_list(R):
    st = engine.LearnerState("able"); st.theta = 10.0
    assert R.recommend(st, k=3)["recommendations"] == []


def test_an_empty_graph_recommends_nothing():
    r = engine.Recommender(nx.DiGraph(), engine.EloLearner())
    assert r.recommend(engine.LearnerState("t"))["recommendations"] == []


def test_a_restored_state_gets_the_same_recommendations(R, gapped):
    back = engine.LearnerState.from_dict(gapped.to_dict())
    assert without_latency(R.recommend(back, k=EVERYTHING, goal=GOAL)) == \
           without_latency(R.recommend(gapped, k=EVERYTHING, goal=GOAL))


def test_no_em_dash_in_any_rationale_or_context(R, states):
    n_strings = 0
    for st in states:
        for goal in (None, GOAL, "B6.1.1.1.1"):
            for r in R.recommend(st, k=EVERYTHING, goal=goal)["recommendations"]:
                for key in ("rationale", "ghana_context"):
                    assert EM_DASH not in r[key], (r["indicator"], key)
                    n_strings += 1
    assert n_strings > 500


# ----------------------------------------------------------------------------- latency (PRD NFR1)

def test_top3_latency_is_under_5_ms(R, fresh, b1, gapped):
    for st, goal in ((fresh, None), (gapped, None), (b1, GOAL)):
        R.recommend(st, k=3, goal=goal)                                     # warm up
        reported, wall = [], []
        for _ in range(30):
            t0 = time.perf_counter()
            res = R.recommend(st, k=3, goal=goal)
            wall.append((time.perf_counter() - t0) * 1000)
            reported.append(res["latency_ms"])
        assert statistics.median(reported) < 5
        assert statistics.median(wall) < 5
        assert statistics.median(reported) <= statistics.median(wall) + 0.01     # the engine times its own work
