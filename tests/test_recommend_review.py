"""The review slot (PRD 6.3): one place for a skill the learner mastered and is starting to forget."""

from datetime import UTC, datetime, timedelta, timezone

import networkx as nx
import pytest

from kwan.learner.elo import MASTERY, EloLearner, LearnerState
from kwan.recommend import review as rv
from kwan.recommend.recommender import Recommender

T0 = datetime(2026, 9, 14, 8, 0, tzinfo=UTC)  # a Monday morning in the first term
NOW = T0 + timedelta(days=21)


def ids(res):
    return [r["indicator"] for r in res["recommendations"]]


class Fading:
    """A learner model set by hand: `raw` is mastery with forgetting ignored, `eff` is mastery at `now`."""

    def __init__(self, raw, eff=None):
        self.raw, self.eff = raw, {**raw, **(eff or {})}

    def mastery(self, st, n, now=None):
        return self.raw[n] if now is None else self.eff[n]


def graph(n_new=4, grades=None):
    """Old skills o1..o3 (grade 1) with nothing before them, and new skills n1.. in separate standards."""
    g = nx.DiGraph()
    for n in ("o1", "o2", "o3"):
        g.add_node(
            n,
            grade=(grades or {}).get(n, 1),
            strand=1,
            strand_name="Number",
            description=f"Old skill {n}",
            content_standard=f"cs-{n}",
        )
    for i in range(1, n_new + 1):
        n = f"n{i}"
        g.add_node(n, grade=2, strand=1, strand_name="Number", description=f"New skill {n}", content_standard=f"cs-{n}")
    return g


def seen(st, when, *skills):
    for n in skills:
        st.last_seen_at[n] = when.isoformat()
        st.last_seen[n] = 1
        st.n[n] = 3
    return st


RAW = {"o1": 0.95, "o2": 0.95, "o3": 0.95, "n1": 0.60, "n2": 0.55, "n3": 0.50, "n4": 0.45}


def setup(eff, seen_at=T0, skills=("o1", "o2", "o3"), **graph_kw):
    g = graph(**graph_kw)
    st = seen(LearnerState("ama"), seen_at, *skills)
    return Recommender(g, Fading(RAW, eff)), st


# ----------------------------------------------------------------------------- the review pick in a list


def test_the_review_pick_takes_the_last_of_three_places():
    r, st = setup({"o2": 0.70})
    recs = r.recommend(st, k=3, now=NOW)["recommendations"]
    assert [x["indicator"] for x in recs] == ["n1", "n2", "o2"]
    assert [x["is_review"] for x in recs] == [False, False, True]
    assert recs[-1]["rationale"] == "You learned this before. A quick check keeps it fresh."
    assert recs[-1]["rationale_parts"] == [{"key": "review", "params": {"pct": 70, "days": 21}}]
    assert recs[-1]["mastery"] == 0.7  # the effective mastery, with forgetting


def test_without_the_review_slot_the_faded_skill_is_an_ordinary_candidate():
    r, st = setup({"o2": 0.70})
    off = r.recommend(st, k=3, now=NOW, review=False)["recommendations"]
    assert not any(x["is_review"] for x in off)
    full = r.recommend(st, k=9, now=NOW, review=False)["recommendations"]
    o2 = next(x for x in full if x["indicator"] == "o2")
    assert o2["rationale"].startswith("A starting skill with nothing before it.")


def test_the_lowest_effective_mastery_wins():
    r, st = setup({"o1": 0.80, "o2": 0.62, "o3": 0.74})
    assert ids(r.recommend(st, k=3, now=NOW))[-1] == "o2"
    r, st = setup({"o1": 0.60, "o2": 0.62, "o3": 0.74})
    assert ids(r.recommend(st, k=3, now=NOW))[-1] == "o1"


def test_equal_mastery_is_settled_by_the_lowest_id():
    r, st = setup({"o1": 0.70, "o2": 0.70, "o3": 0.70})
    assert ids(r.recommend(st, k=3, now=NOW))[-1] == "o1"
    r, st = setup({"o1": 0.70, "o2": 0.70, "o3": 0.70}, skills=("o3", "o2"))
    assert ids(r.recommend(st, k=3, now=NOW))[-1] == "o2"


def test_only_one_review_pick_and_the_skill_is_not_listed_twice():
    r, st = setup({"o1": 0.70, "o2": 0.60, "o3": 0.65})
    recs = r.recommend(st, k=9, now=NOW)["recommendations"]
    got = [x["indicator"] for x in recs]
    assert len(got) == len(set(got)) == 7
    assert got[-1] == "o2" and sum(x["is_review"] for x in recs) == 1
    # the other faded skills are ordinary candidates, ranked by score like any other
    assert {"o1", "o3"} <= set(got[:-1])
    assert all(x["rationale_parts"][0]["key"] != "review" for x in recs[:-1])


def test_the_review_slot_needs_k_of_three_or_more():
    r, st = setup({"o2": 0.70})
    for k in (1, 2):
        recs = r.recommend(st, k=k, now=NOW)["recommendations"]
        assert len(recs) == k and not any(x["is_review"] for x in recs)
    for k in (3, 4, 9):
        recs = r.recommend(st, k=k, now=NOW)["recommendations"]
        assert [x["is_review"] for x in recs] == [False] * (len(recs) - 1) + [True]
        assert len(recs) == min(k, 5)  # four new skills plus the review pick


def test_the_other_places_are_the_usual_ranking():
    """Taking the review skill out and asking for k - 1 gives the same first places."""
    r, st = setup({"o2": 0.70})
    with_review = r.recommend(st, k=4, now=NOW)["recommendations"]
    usual = [x for x in r.recommend(st, k=9, now=NOW, review=False)["recommendations"] if x["indicator"] != "o2"]
    assert with_review[:3] == usual[:3]


def test_review_can_be_switched_off():
    r, st = setup({"o2": 0.70})
    assert not any(x["is_review"] for x in r.recommend(st, k=3, now=NOW, review=False)["recommendations"])


# ----------------------------------------------------------------------------- when nothing is due


def test_no_now_means_no_review_slot():
    r, st = setup({"o2": 0.70})
    recs = r.recommend(st, k=3)["recommendations"]
    assert ids({"recommendations": recs}) == ["n1", "n2", "n3"] and not any(x["is_review"] for x in recs)


def test_no_timestamps_in_the_state_means_no_review_slot():
    r, st = setup({"o2": 0.70})
    st.last_seen_at.clear()
    assert not any(x["is_review"] for x in r.recommend(st, k=3, now=NOW)["recommendations"])


@pytest.mark.parametrize("days, due", [(6, False), (6.99, False), (7, True), (7.01, True), (60, True)])
def test_a_skill_must_be_unseen_for_seven_days(days, due):
    r, st = setup({"o2": 0.70}, seen_at=NOW - timedelta(days=days))
    recs = r.recommend(st, k=3, now=NOW)["recommendations"]
    assert any(x["is_review"] for x in recs) is due


def test_a_skill_seen_after_now_is_not_due():
    r, st = setup({"o2": 0.70}, seen_at=NOW + timedelta(days=30))
    assert not any(x["is_review"] for x in r.recommend(st, k=3, now=NOW)["recommendations"])


def test_a_skill_that_was_never_mastered_is_not_reviewed():
    raw = {**RAW, "o2": 0.84}
    st = seen(LearnerState("ama"), T0, "o1", "o2", "o3")
    r = Recommender(graph(), Fading(raw, {"o2": 0.60}))
    recs = r.recommend(st, k=3, now=NOW)["recommendations"]
    assert not any(x["is_review"] for x in recs)


def test_a_skill_still_at_mastery_is_not_reviewed():
    r, st = setup({"o2": 0.85})
    assert not any(x["is_review"] for x in r.recommend(st, k=3, now=NOW)["recommendations"])
    r, st = setup({"o2": 0.8499})
    assert ids(r.recommend(st, k=3, now=NOW))[-1] == "o2"


def test_the_class_cap_applies_to_the_review_pick():
    r, st = setup({"o1": 0.80, "o2": 0.60}, grades={"o2": 3})
    assert ids(r.recommend(st, k=3, now=NOW))[-1] == "o2"
    capped = r.recommend(st, k=3, now=NOW, grade_max=2)["recommendations"]
    assert capped[-1]["indicator"] == "o1" and capped[-1]["is_review"]
    assert all(x["grade"] <= 2 for x in capped)


def test_ids_in_the_state_that_are_not_in_the_graph_are_skipped():
    r, st = setup({"o2": 0.70})
    st.last_seen_at["gone.1"] = T0.isoformat()
    assert ids(r.recommend(st, k=3, now=NOW))[-1] == "o2"


# ----------------------------------------------------------------------------- how it combines with the goal


def chain_graph():
    """p1 -> p2 -> goal, plus the old skills and three unrelated new ones."""
    g = graph(n_new=3)
    for n in ("p1", "p2", "goal"):
        g.add_node(n, grade=2, strand=1, strand_name="Number", description=n, content_standard=f"cs-{n}")
    g.add_edges_from([("p1", "p2"), ("p2", "goal")])
    return g


def test_the_review_slot_never_displaces_the_only_skill_on_the_path():
    raw = {**RAW, "p1": 0.30, "p2": 0.20, "goal": 0.10}
    st = seen(LearnerState("ama"), T0, "o1", "o2", "o3")
    r = Recommender(chain_graph(), Fading(raw, {"o2": 0.70}))
    recs = r.recommend(st, k=3, goal="goal", now=NOW)["recommendations"]
    assert recs[0]["indicator"] == "p1"  # the one candidate on the path keeps the first place
    assert recs[1]["indicator"] == "n1"  # the best skill off the path
    assert [x["is_review"] for x in recs] == [False, False, True] and recs[2]["indicator"] == "o2"
    assert "On the way to your goal, 2 steps away." in recs[0]["rationale"]


def test_a_review_skill_on_the_path_is_still_the_review_pick():
    raw = {**RAW, "p1": 0.95, "p2": 0.50, "goal": 0.10}
    st = seen(LearnerState("ama"), T0, "p1")
    r = Recommender(chain_graph(), Fading(raw, {"p1": 0.75}))
    recs = r.recommend(st, k=3, goal="goal", now=NOW)["recommendations"]
    assert recs[-1]["indicator"] == "p1" and recs[-1]["is_review"]
    assert recs[0]["indicator"] == "p2"  # p1 at 75% still counts as ready, so p2 is on offer
    assert [x["indicator"] for x in recs].count("p1") == 1


def test_review_and_exploration_and_variety_together_are_deterministic():
    r, st = setup({"o1": 0.70, "o2": 0.60})
    first = r.recommend(st, k=4, now=NOW, eps=0.3, variety=0.05)["recommendations"]
    assert r.recommend(st, k=4, now=NOW, eps=0.3, variety=0.05)["recommendations"] == first
    assert first[-1]["indicator"] == "o2" and first[-1]["is_review"]  # exploration never moves the review pick
    later = LearnerState.from_dict({**st.to_dict(), "step": st.step + 5})
    assert r.recommend(later, k=4, now=NOW, eps=0.3)["recommendations"][-1]["indicator"] == "o2"


# ----------------------------------------------------------------------------- time handling


def test_now_can_be_a_datetime_a_string_or_naive():
    r, st = setup({"o2": 0.70})
    want = r.recommend(st, k=3, now=NOW)["recommendations"]
    assert r.recommend(st, k=3, now=NOW.isoformat())["recommendations"] == want
    assert r.recommend(st, k=3, now=NOW.replace(tzinfo=None))["recommendations"] == want  # naive is UTC
    accra_plus_two = NOW.astimezone(timezone(timedelta(hours=2)))
    assert r.recommend(st, k=3, now=accra_plus_two)["recommendations"] == want
    assert r.recommend(st, k=3, now="2026-10-05T08:00:00Z")["recommendations"] == want


def test_time_helpers():
    assert rv.as_utc("2026-11-10T09:30:00") == datetime(2026, 11, 10, 9, 30, tzinfo=UTC)
    assert rv.as_utc(datetime(2026, 11, 10, 9, 30)) == datetime(2026, 11, 10, 9, 30, tzinfo=UTC)
    assert rv.as_utc("2026-11-10T09:30:00+02:00") == datetime(2026, 11, 10, 7, 30, tzinfo=UTC)
    assert rv.days_between("2026-11-01T00:00:00+00:00", "2026-11-08T12:00:00+00:00") == 7.5
    assert rv.days_between(NOW, T0) == -21
    with pytest.raises(TypeError):
        rv.as_utc(20261110)
    with pytest.raises(ValueError):
        rv.as_utc("next Tuesday")


def test_a_bad_now_is_an_error_not_a_silent_skip():
    r, st = setup({"o2": 0.70})
    with pytest.raises(ValueError):
        r.recommend(st, k=3, now="soon")


# ----------------------------------------------------------------------------- review_candidate()


def test_review_candidate_unit():
    st = seen(LearnerState("t"), T0, "a", "b", "c")
    eff = {"a": 0.80, "b": 0.70, "c": 0.90, "d": 0.10}
    raw = {"a": 0.95, "b": 0.95, "c": 0.95, "d": 0.95}
    nodes = set(eff)
    assert rv.review_candidate(st, nodes, eff, raw.__getitem__, NOW) == "b"
    assert rv.review_candidate(st, nodes, eff, raw.__getitem__, None) is None
    assert rv.review_candidate(st, nodes, eff, raw.__getitem__, T0 + timedelta(days=3)) is None
    assert rv.review_candidate(st, nodes, eff, raw.__getitem__, T0 + timedelta(days=3), min_days=2) == "b"
    assert rv.review_candidate(st, nodes, eff, raw.__getitem__, NOW, allowed=lambda n: n != "b") == "a"
    assert rv.review_candidate(st, nodes, eff, {**raw, "b": 0.5, "a": 0.5}.__getitem__, NOW) is None
    assert rv.review_candidate(LearnerState("new"), nodes, eff, raw.__getitem__, NOW) is None
    assert rv.MIN_DAYS == 7


def test_raw_mastery_is_asked_only_for_skills_that_pass_the_other_checks():
    st = seen(LearnerState("t"), T0, "a", "b", "c")
    asked = []

    def raw(n):
        asked.append(n)
        return 0.95

    rv.review_candidate(st, {"a", "b", "c"}, {"a": 0.9, "b": 0.7, "c": 0.8}, raw, NOW)
    assert "a" not in asked and "b" in asked  # a is still mastered, so nobody asks about it


# ----------------------------------------------------------------------------- with the real learner model


def test_review_with_the_real_learner_model(G_v1):
    """Three right in a row at T0. Six weeks later the streak no longer counts and the skill has faded."""
    G = G_v1  # bare codes, as the v1 contract
    L = EloLearner(G=G)
    R = Recommender(G, L)
    st = LearnerState("kojo")
    skill = "B1.1.1.1.1"
    for i in range(3):
        L.update(st, skill, True, at=T0 + timedelta(minutes=i))
    later = T0 + timedelta(days=42)
    assert L.mastery(st, skill) >= MASTERY > L.mastery(st, skill, now=later)
    recs = R.recommend(st, k=3, now=later)["recommendations"]
    assert recs[-1]["indicator"] == skill and recs[-1]["is_review"]
    assert recs[-1]["rationale"] == "You learned this before. A quick check keeps it fresh."
    assert recs[-1]["rationale_parts"][0]["params"]["days"] == 41  # whole days since the last answer
    assert [x["is_review"] for x in recs] == [False, False, True]
    assert skill not in ids({"recommendations": recs[:2]})
    # a week on, still mastered (the streak holds for 30 days): nothing to review
    soon = R.recommend(st, k=3, now=T0 + timedelta(days=8))["recommendations"]
    assert not any(x["is_review"] for x in soon)
    # and with no `now` the engine behaves as in v1: no forgetting, no review
    assert not any(x["is_review"] for x in R.recommend(st, k=3)["recommendations"])


def test_the_fast_mastery_map_of_the_learner_equals_mastery_skill_by_skill(G):
    """The recommender may use `effective_mastery_map`; it must give the numbers `mastery` gives."""
    L = EloLearner(G=G)
    if not hasattr(L, "effective_mastery_map"):
        pytest.skip("this learner model has no one-pass mastery map")
    import random

    rng = random.Random(4)
    st = LearnerState("map")
    nodes = sorted(G)
    for i in range(80):
        L.update(st, rng.choice(nodes[:60]), rng.random() < 0.7, at=T0 + timedelta(hours=7 * i))
    for now in (None, T0 + timedelta(days=30), T0 + timedelta(days=200)):
        fast = L.effective_mastery_map(st, now)
        assert list(fast) == list(G)
        for n in G:
            slow = L.mastery(st, n) if now is None else L.mastery(st, n, now=now)
            assert fast[n] == pytest.approx(slow, abs=1e-12)
    R = Recommender(G, L)
    assert R._mastery_map(st, None) == L.effective_mastery_map(st, None)
