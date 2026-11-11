"""v1 equivalence mode: with `goal_mode="soft"`, `eps=0`, `variety=0`, `review=False` and no timestamps,
every output equals v1.1 (the v1.0 engine plus half-up rounding and correct plurals).

The v1.1 recommender is written out below, on its own, and compared with the package on the frozen v1
graph for many learners, goals, class caps and list sizes.
"""

import math
import random
from datetime import UTC, datetime, timedelta

import networkx as nx
import pytest

from kwan.context.bank import practice_setting
from kwan.learner.elo import EloLearner, LearnerState
from kwan.recommend.recommender import Recommender

V1 = dict(goal_mode="soft", eps=0.0, variety=0.0, review=False)
V1_KEYS = ("indicator", "score", "mastery", "grade", "strand", "description", "rationale", "ghana_context")
W = dict(readiness=0.30, zpd=0.25, goal=0.25, unlock=0.15, recency=0.05)
GOALS = (None, "B4.1.1.3.3", "B2.1.1.1.1", "B6.1.1.1.1")


def half_up(x, d):
    return math.floor(x * 10**d + 0.5) / 10**d


def pct(x):
    return f"{math.floor(x * 100 + 0.5)}%"


def count(n, word):
    return f"{n} {word}{'' if n == 1 else 's'}"


def v11_recommend(G, L, st, k=3, goal=None, grade_max=None, goal_mode="soft"):
    """The v1.1 recommender (PRD v2 Appendix A.1 applied to the v1.0 engine), with the strict goal option."""
    desc_count = {n: len(nx.descendants(G, n)) for n in G}
    max_desc = max(desc_count.values(), default=0) or 1
    m = {n: L.mastery(st, n) for n in G}
    gd = nx.single_source_shortest_path_length(G.reverse(copy=False), goal) if goal else {}
    cands = []
    for n, d in G.nodes(data=True):
        if m[n] >= 0.85 or (grade_max is not None and d["grade"] > grade_max):
            continue
        preds = list(G.predecessors(n))
        ready = min((m[p] for p in preds), default=1.0)
        if ready < 0.70:
            continue
        zpd = 1 - abs(m[n] - 0.6) / 0.6
        goal_term = 1 / (1 + gd[n]) if n in gd else (0.0 if gd else 0.5)
        unlock = desc_count[n] / max_desc
        rec = 1 - math.exp(-(st.step - st.last_seen.get(n, -99)) / 5)
        s = W["readiness"] * ready + W["zpd"] * zpd + W["goal"] * goal_term + W["unlock"] * unlock + W["recency"] * rec
        cands.append((s, n, preds, m[n]))
    if gd and goal_mode == "strict":
        cands.sort(key=lambda c: (c[1] in gd, c[0], c[1]), reverse=True)
    else:
        cands.sort(key=lambda c: (c[0], c[1]), reverse=True)
    out = []
    for s, n, preds, mn in cands[:k]:
        parts = []
        if preds:
            best = max(preds, key=lambda p: m[p])
            parts.append(f"Builds on {best}, where you are at {pct(m[best])}.")
        else:
            parts.append("A starting skill with nothing before it.")
        if mn < 0.5:
            parts.append(f"Still new to you ({pct(mn)}), so expect a few tries.")
        elif mn < 0.75:
            parts.append(f"You are at {pct(mn)}, close enough to master with practice.")
        else:
            parts.append(f"Nearly there at {pct(mn)}. A few correct answers will lock it in.")
        if n in gd and gd[n] > 0:
            parts.append(f"On the way to your goal, {count(gd[n], 'step')} away.")
        if desc_count[n]:
            parts.append(f"Opens the way to {count(desc_count[n], 'later skill')}.")
        out.append(
            {
                "indicator": n,
                "score": half_up(s, 4),
                "mastery": half_up(mn, 3),
                "grade": G.nodes[n]["grade"],
                "strand": G.nodes[n]["strand_name"],
                "description": G.nodes[n]["description"],
                "rationale": " ".join(parts),
                "ghana_context": practice_setting(n, G.nodes[n]["strand"], G.nodes[n]["grade"]),
            }
        )
    return out


def v1_part(recs):
    return [{key: r[key] for key in V1_KEYS} for r in recs]


@pytest.fixture(scope="module")
def G(G_v1):
    """v1 equivalence is pinned on the frozen v1 graph (bare ids)."""
    return G_v1


@pytest.fixture(scope="module")
def L(G):
    return EloLearner(G=G)


@pytest.fixture(scope="module")
def R(G, L):
    return Recommender(G, L)


def learner(G, L, seed, timed=False):
    """A seeded learner. With `timed`, answers carry timestamps (the state then has `last_seen_at`)."""
    rng = random.Random(seed)
    st = LearnerState(f"v1-{seed}")
    syllabus = sorted(G, key=lambda x: (G.nodes[x]["grade"], x))
    reach = rng.choice([10, 40, 90, 200])
    start = datetime(2026, 1, 5, 8, 0, tzinfo=UTC)
    for i in range(rng.randint(0, 150)):
        n = rng.choice(syllabus[:reach])
        at = start + timedelta(hours=3 * i) if timed else None
        L.update(st, n, rng.random() < rng.choice([0.4, 0.7, 0.9]), at=at)
    return st


@pytest.mark.parametrize("seed", range(12))
def test_v1_mode_equals_the_v11_engine(G, L, R, seed):
    st = learner(G, L, seed)
    assert not st.last_seen_at
    for goal in GOALS:
        for grade_max in (None, 3):
            for k in (1, 3, 5, 10_000):
                got = R.recommend(st, k=k, goal=goal, grade_max=grade_max, **V1)["recommendations"]
                assert v1_part(got) == v11_recommend(G, L, st, k, goal, grade_max), (goal, grade_max, k)
                assert not any(r["is_review"] for r in got)


@pytest.mark.parametrize("seed", range(4))
def test_strict_mode_without_the_v2_features_equals_v11_strict(G, L, R, seed):
    st = learner(G, L, seed)
    for goal in GOALS[1:]:
        got = R.recommend(st, k=10_000, goal=goal, eps=0.0, variety=0.0, review=False)["recommendations"]
        assert v1_part(got) == v11_recommend(G, L, st, 10_000, goal, goal_mode="strict")


@pytest.mark.parametrize("seed", range(4))
def test_a_state_with_timestamps_is_read_the_v1_way_without_now(G, L, R, seed):
    """Timestamps on the answers alone do not switch forgetting or review on: that needs a `now`."""
    timed = learner(G, L, seed, timed=True)
    assert timed.last_seen_at
    for goal in GOALS[:2]:
        got = R.recommend(timed, k=10_000, goal=goal, **V1)["recommendations"]
        assert v1_part(got) == v11_recommend(G, L, timed, 10_000, goal)
        default = R.recommend(timed, k=10_000, goal=goal)["recommendations"]
        assert not any(r["is_review"] for r in default)


def test_the_new_fields_are_only_added_never_replacing_a_v1_field(G, L, R):
    st = learner(G, L, 1)
    for r in R.recommend(st, k=5, **V1)["recommendations"]:
        assert list(r)[: len(V1_KEYS) - 2] == list(V1_KEYS[:-2])
        assert set(r) - set(V1_KEYS) == {"short_label", "rationale_parts", "is_review"}


def test_v1_mode_on_a_learner_who_knows_nothing_and_one_who_knows_everything(G, L, R):
    fresh = LearnerState("fresh")
    assert v1_part(R.recommend(fresh, k=10_000, **V1)["recommendations"]) == v11_recommend(G, L, fresh, 10_000)
    done = LearnerState("done")
    for n in G:
        for _ in range(3):
            L.update(done, n, True)
    assert R.recommend(done, k=3, **V1)["recommendations"] == v11_recommend(G, L, done, 3) == []
