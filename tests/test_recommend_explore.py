"""Seeded exploration and strand variety (PRD 6.3, decisions D6 and D8, Appendix B.2)."""

import json
import math
import random
import statistics
import zlib
from pathlib import Path

import networkx as nx
import pytest

from kwan.learner.elo import EloLearner, LearnerState
from kwan.recommend import explore as ex
from kwan.recommend.explore import Candidate, Mulberry32, pick_with_variety
from kwan.recommend.recommender import Recommender

VECTORS = json.loads((Path(__file__).parent / "fixtures" / "mulberry32_vectors.json").read_text(encoding="utf-8"))
GOAL = "B4.1.1.3.3"
EVERYTHING = 10_000


def ids(res):
    return [r["indicator"] for r in res["recommendations"]]


@pytest.fixture(scope="module")
def G(G_v1):
    """The seeded pupils and goals below use bare codes (the v1 contract): the frozen v1 graph."""
    return G_v1


@pytest.fixture(scope="module")
def L(G):
    return EloLearner(G=G)


@pytest.fixture(scope="module")
def R(G, L):
    return Recommender(G, L)


def pupil(G, L, name="amina", seed=7, answers=60):
    """A seeded pupil with answers on the first skills in syllabus order."""
    st = LearnerState(name)
    rng = random.Random(seed)
    for n in sorted(G, key=lambda x: (G.nodes[x]["grade"], x))[:answers]:
        if rng.random() < 0.5:
            L.update(st, n, rng.random() < 0.8)
    return st


# ----------------------------------------------------------------------------- the generator


def test_fixture_has_twenty_outputs_for_three_seeds():
    assert VECTORS["generator"] == "mulberry32"
    assert len(VECTORS["vectors"]) == 3
    for v in VECTORS["vectors"]:
        assert len(v["u32"]) == len(v["uniform"]) == len(v["normal"]) == 20
    assert len({v["seed"] for v in VECTORS["vectors"]}) == 3


@pytest.mark.parametrize("v", VECTORS["vectors"], ids=lambda v: v["name"])
def test_mulberry32_matches_the_committed_vectors(v):
    rng = Mulberry32(v["seed"])
    assert [rng.next_u32() for _ in range(20)] == v["u32"]  # exact
    rng = Mulberry32(v["seed"])
    assert [rng.next() for _ in range(20)] == v["uniform"]  # exact: one division by 2**32
    assert v["uniform"] == [u / 4294967296 for u in v["u32"]]
    rng = Mulberry32(v["seed"])
    assert [rng.normal() for _ in range(20)] == pytest.approx(v["normal"], abs=1e-12)


def test_mulberry32_against_an_independent_implementation():
    """The reference JavaScript algorithm, written out with explicit 32-bit wrap-around."""

    def imul(a, b):
        return (a * b) & 0xFFFFFFFF

    def reference(seed, count):
        a, out = seed & 0xFFFFFFFF, []
        for _ in range(count):
            a = (a + 0x6D2B79F5) & 0xFFFFFFFF
            t = imul(a ^ (a >> 15), a | 1)
            t = ((t + imul(t ^ (t >> 7), t | 61)) & 0xFFFFFFFF) ^ t
            out.append((t ^ (t >> 14)) & 0xFFFFFFFF)
        return out

    for seed in (0, 1, 123456789, 2**32 - 1, 2**32 + 5, -1):
        rng = Mulberry32(seed)
        assert [rng.next_u32() for _ in range(50)] == reference(seed, 50)


def test_the_seed_is_reduced_to_32_bits():
    assert Mulberry32(2**32 + 5).next_u32() == Mulberry32(5).next_u32()
    assert Mulberry32(-1).next_u32() == Mulberry32(2**32 - 1).next_u32()


def test_uniform_values_are_in_range_and_spread_out():
    rng = Mulberry32(2026)
    xs = [rng.next() for _ in range(20_000)]
    assert all(0.0 <= x < 1.0 for x in xs)
    assert statistics.mean(xs) == pytest.approx(0.5, abs=0.01)
    assert [sum(lo <= x < lo + 0.25 for x in xs) for lo in (0, 0.25, 0.5, 0.75)] == pytest.approx([5000] * 4, rel=0.05)


def test_normal_values_have_mean_zero_and_variance_one():
    rng = Mulberry32(11)
    xs = [rng.normal() for _ in range(20_000)]
    assert statistics.mean(xs) == pytest.approx(0.0, abs=0.03)
    assert statistics.pstdev(xs) == pytest.approx(1.0, abs=0.03)
    assert sum(abs(x) < 1.96 for x in xs) / len(xs) == pytest.approx(0.95, abs=0.01)


def test_normal_uses_exactly_two_uniforms_and_never_fails_on_zero():
    a, b = Mulberry32(3), Mulberry32(3)
    a.normal()
    b.next(), b.next()
    assert a.a == b.a

    class Zero(Mulberry32):
        def next(self):
            return 0.0

    assert Zero(0).normal() == 0.0  # 1 - 0 = 1, log(1) = 0: no clamp needed, no error


def test_the_seed_rule_is_crc32_of_learner_and_step():
    v = next(v for v in VECTORS["vectors"] if "seed_from" in v)
    src = v["seed_from"]
    assert ex.seed_for(src["learner_id"], src["step"]) == v["seed"]
    assert ex.seed_for("pupil-1", 7) == zlib.crc32(b"pupil-1:7")
    assert ex.seed_for("Ama Owusu ɔ", 0) == zlib.crc32("Ama Owusu ɔ:0".encode())  # UTF-8
    assert ex._rng("pupil-1", 7).next_u32() == Mulberry32(zlib.crc32(b"pupil-1:7")).next_u32()
    assert ex.seed_for("pupil-1", 7) != ex.seed_for("pupil-1", 8) != ex.seed_for("pupil-2", 8)


# ----------------------------------------------------------------------------- explore()


def test_explore_adds_scaled_noise():
    z = Mulberry32(5).normal()
    assert ex.explore(0.5, 0, Mulberry32(5), 0.05) == 0.5 + 0.05 * z
    assert ex.explore(0.5, 3, Mulberry32(5), 0.05) == 0.5 + 0.05 * z / 2  # sqrt(1 + 3) = 2
    assert ex.explore(0.5, 99, Mulberry32(5), 0.2) == pytest.approx(0.5 + 0.2 * z / 10)


def test_explore_with_zero_eps_changes_nothing_and_draws_nothing():
    rng = Mulberry32(5)
    assert ex.explore(0.4321, 2, rng, 0) == 0.4321
    assert ex.explore(0.4321, 2, rng, 0.0) == 0.4321
    assert rng.a == 5


def test_the_nudge_shrinks_with_the_number_of_answers():
    def spread(n_k):
        rng = Mulberry32(1)
        return statistics.pstdev(ex.explore(0.0, n_k, rng, 1.0) for _ in range(4000))

    assert spread(0) == pytest.approx(1.0, abs=0.05)
    assert spread(3) == pytest.approx(0.5, abs=0.03)
    assert spread(24) == pytest.approx(0.2, abs=0.02)


# ----------------------------------------------------------------------------- pick_with_variety()


def sketch_pick(cands, k, penalty):
    """The greedy pick written the slow, obvious way (PRD Appendix B.2): sort the whole pool every time."""
    chosen, seen, pool = [], {}, list(cands)
    while pool and len(chosen) < k:
        pool.sort(key=lambda c: (c.tier, c.rank - penalty * seen.get(c.group, 0), c.id), reverse=True)
        best = pool.pop(0)
        chosen.append(best)
        seen[best.group] = seen.get(best.group, 0) + 1
    return chosen


def test_variety_spreads_picks_over_content_standards():
    cands = [
        Candidate(0, 0.80, "s1", "A"),
        Candidate(0, 0.78, "s2", "A"),
        Candidate(0, 0.76, "s3", "A"),
        Candidate(0, 0.75, "t1", "B"),
        Candidate(0, 0.70, "u1", "C"),
    ]
    assert [c.id for c in pick_with_variety(cands, 3, 0.0)] == ["s1", "s2", "s3"]
    assert [c.id for c in pick_with_variety(cands, 3, 0.05)] == ["s1", "t1", "s2"]  # s2: 0.73, t1: 0.75
    assert [c.id for c in pick_with_variety(cands, 5, 0.05)] == ["s1", "t1", "s2", "u1", "s3"]  # s3: 0.76 - 0.10


def test_the_penalty_adds_up_per_pick_from_the_same_group():
    cands = [Candidate(0, 1.0, f"a{i}", "A") for i in range(3)] + [Candidate(0, 0.91, "b0", "B")]
    # a2 first (highest id on ties), then a1 at 0.95, then a0 at 0.90 loses to b0 at 0.91
    assert [c.id for c in pick_with_variety(cands, 4, 0.05)] == ["a2", "a1", "b0", "a0"]


def test_variety_never_lifts_a_lower_tier_over_a_higher_one():
    cands = [Candidate(1, 0.30, f"on{i}", "A") for i in range(4)] + [Candidate(0, 0.99, "off", "B")]
    for penalty in (0.0, 0.05, 0.5):
        got = [c.id for c in pick_with_variety(cands, 5, penalty)]
        assert got[-1] == "off" and set(got[:4]) == {"on0", "on1", "on2", "on3"}


def test_ties_go_to_the_higher_id_whatever_the_input_order():
    cands = [Candidate(0, 0.5, n, n) for n in ("b", "a", "c")]
    for order in (cands, cands[::-1], [cands[1], cands[2], cands[0]]):
        assert [c.id for c in pick_with_variety(order, 3, 0.05)] == ["c", "b", "a"]


def test_k_and_penalty_edges():
    cands = [Candidate(0, 0.5, "a", "A"), Candidate(0, 0.4, "b", "A")]
    assert pick_with_variety(cands, 0, 0.05) == []
    assert pick_with_variety([], 3, 0.05) == []
    assert [c.id for c in pick_with_variety(cands, 9, 0.05)] == ["a", "b"]
    assert [c.id for c in pick_with_variety(iter(cands), 1, 0.05)] == ["a"]
    with pytest.raises(ValueError, match="penalty must be 0 or more"):
        pick_with_variety(cands, 2, -0.1)


@pytest.mark.parametrize("seed", range(25))
def test_the_fast_pick_equals_the_sketch_on_random_pools(seed):
    rng = random.Random(seed)
    n = rng.randint(1, 40)
    cands = [
        Candidate(rng.randint(0, 1), round(rng.random(), rng.choice([1, 2, 6])), f"n{i:02d}", rng.choice("ABCDE"))
        for i in range(n)
    ]
    for penalty in (0.0, 0.05, 0.3):
        for k in (1, 3, n, n + 5):
            assert pick_with_variety(cands, k, penalty) == sketch_pick(cands, k, penalty)


def test_content_standard_of_a_code():
    assert ex.content_standard_of("B4.1.1.1.2") == "B4.1.1.1"
    assert ex.content_standard_of("B4") == "B4"
    assert ex.content_standard_of("") == ""


# ----------------------------------------------------------------------------- variety in the recommender


def graph_of(nodes, edges=()):
    g = nx.DiGraph()
    for n, cs in nodes:
        g.add_node(n, grade=1, strand=1, strand_name="Number", description=n, content_standard=cs)
    g.add_edges_from(edges)
    return g


class FixedLearner:
    def __init__(self, m):
        self.m = m

    def mastery(self, st, n):
        return self.m[n]


def test_recommender_variety_uses_the_content_standard_of_the_node():
    g = graph_of([("s1", "A"), ("s2", "A"), ("s3", "A"), ("t1", "B")])
    r = Recommender(g, FixedLearner({"s1": 0.60, "s2": 0.59, "s3": 0.58, "t1": 0.55}))
    st = LearnerState("t")
    assert ids(r.recommend(st, k=3, variety=0.0)) == ["s1", "s2", "s3"]
    assert ids(r.recommend(st, k=3)) == ["s1", "t1", "s2"]  # the default penalty of 0.05
    assert ids(r.recommend(st, k=3, variety=0.05)) == ids(r.recommend(st, k=3))
    by_id = {x["indicator"]: x["score"] for x in r.recommend(st, k=4)["recommendations"]}
    assert by_id == {x["indicator"]: x["score"] for x in r.recommend(st, k=4, variety=0.0)["recommendations"]}


def test_without_a_content_standard_the_group_comes_from_the_code():
    g = nx.DiGraph()
    for n in ("B1.1.1.1.1", "B1.1.1.1.2", "B1.1.2.1.1"):
        g.add_node(n, grade=1, strand=1, strand_name="Number", description=n)
    r = Recommender(g, FixedLearner({"B1.1.1.1.1": 0.58, "B1.1.1.1.2": 0.60, "B1.1.2.1.1": 0.57}))
    assert r._group == {"B1.1.1.1.1": "B1.1.1.1", "B1.1.1.1.2": "B1.1.1.1", "B1.1.2.1.1": "B1.1.2.1"}
    st = LearnerState("t")
    assert ids(r.recommend(st, k=3, variety=0.0)) == ["B1.1.1.1.2", "B1.1.1.1.1", "B1.1.2.1.1"]
    assert ids(r.recommend(st, k=3)) == ["B1.1.1.1.2", "B1.1.2.1.1", "B1.1.1.1.1"]


def test_variety_on_the_real_graph_is_the_greedy_rule(G, L, R):
    """The default list is what the slow, obvious greedy rule gives on the exact formula scores."""
    seen_change = False
    for seed in range(6):
        st = pupil(G, L, f"p{seed}", seed)
        m = {n: L.mastery(st, n) for n in G}
        for goal in (None, GOAL):
            plain = ids(R.recommend(st, k=EVERYTHING, goal=goal, variety=0.0))
            gd = R._goal_dist(goal)
            cands = [Candidate(int(n in gd), R._score(st, n, m, gd), n, G.nodes[n]["content_standard"]) for n in plain]
            got = ids(R.recommend(st, k=6, goal=goal))
            assert got == [c.id for c in sketch_pick(cands, 6, 0.05)]
            assert got[0] == plain[0]
            seen_change |= got != plain[:6]
    assert seen_change


def test_variety_gives_more_content_standards_in_the_top_five(G, L, R):
    with_variety = without = 0
    for seed in range(8):
        st = pupil(G, L, f"v{seed}", seed)
        for variety, bucket in ((0.05, "with"), (0.0, "without")):
            groups = {G.nodes[n]["content_standard"] for n in ids(R.recommend(st, k=5, variety=variety))}
            if bucket == "with":
                with_variety += len(groups)
            else:
                without += len(groups)
    assert with_variety >= without


def test_top_k_is_a_prefix_of_the_full_list_with_variety(G, L, R):
    st = pupil(G, L)
    full = ids(R.recommend(st, k=EVERYTHING, goal=GOAL))
    for k in (1, 2, 3, 7):
        assert ids(R.recommend(st, k=k, goal=GOAL)) == full[:k]
    assert sorted(full) == sorted(ids(R.recommend(st, k=EVERYTHING, goal=GOAL, variety=0.0)))  # same skills


# ----------------------------------------------------------------------------- exploration in the recommender


def test_zero_eps_is_the_default_and_draws_nothing(G, L, R):
    st = pupil(G, L)
    assert R.recommend(st, k=5)["recommendations"] == R.recommend(st, k=5, eps=0.0)["recommendations"]


def test_exploration_is_reproducible_for_the_same_learner_and_step(G, L, R):
    st = pupil(G, L)
    first = R.recommend(st, k=EVERYTHING, eps=0.05)["recommendations"]
    assert R.recommend(st, k=EVERYTHING, eps=0.05)["recommendations"] == first
    again = Recommender(G, EloLearner(G=G)).recommend(LearnerState.from_dict(st.to_dict()), k=EVERYTHING, eps=0.05)
    assert again["recommendations"] == first


def test_exploration_changes_with_the_step_and_with_the_learner(G, L, R):
    st = pupil(G, L)
    base = ids(R.recommend(st, k=EVERYTHING, eps=0.3))
    other = LearnerState.from_dict({**st.to_dict(), "learner_id": "kofi"})
    assert ids(R.recommend(other, k=EVERYTHING, eps=0.3)) != base
    later = LearnerState.from_dict({**st.to_dict(), "step": st.step + 1})
    assert ids(R.recommend(later, k=EVERYTHING, eps=0.3)) != base
    assert sorted(ids(R.recommend(other, k=EVERYTHING, eps=0.3))) == sorted(base)  # same skills, new order


def test_exploration_reorders_but_never_changes_scores_or_candidates(G, L, R):
    st = pupil(G, L)
    plain = {r["indicator"]: r for r in R.recommend(st, k=EVERYTHING)["recommendations"]}
    noisy = R.recommend(st, k=EVERYTHING, eps=0.5)["recommendations"]
    assert {r["indicator"]: r for r in noisy} == plain  # same items, same reported scores and reasons
    assert [r["indicator"] for r in noisy] != list(plain)


def test_exploration_is_the_documented_formula(G, L, R):
    """Noise is drawn for the candidates in ascending id order from the learner-and-step generator."""
    st = pupil(G, L)
    eps = 0.2
    plain = {r["indicator"]: r["indicator"] for r in R.recommend(st, k=EVERYTHING, variety=0.0)["recommendations"]}
    m = {n: L.mastery(st, n) for n in G}
    exact = {n: R._score(st, n, m, {}) for n in plain}
    rng = Mulberry32(zlib.crc32(f"{st.learner_id}:{st.step}".encode()))
    ranks = {}
    for n in sorted(exact):
        u1, u2 = 1.0 - rng.next(), rng.next()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        ranks[n] = exact[n] + eps * z / math.sqrt(1 + st.n.get(n, 0))
    want = sorted(ranks, key=lambda n: (ranks[n], n), reverse=True)
    assert ids(R.recommend(st, k=EVERYTHING, eps=eps, variety=0.0)) == want


def test_exploration_stays_inside_the_goal_tiers(G, L, R):
    path = set(nx.ancestors(G, GOAL)) | {GOAL}
    for seed in range(5):
        st = pupil(G, L, f"e{seed}", seed)
        got = ids(R.recommend(st, k=EVERYTHING, goal=GOAL, eps=1.0))
        flags = [n in path for n in got]
        assert flags == sorted(flags, reverse=True) and any(flags) and not all(flags)
        soft = ids(R.recommend(st, k=EVERYTHING, goal=GOAL, eps=1.0, goal_mode="soft"))
        assert sorted(soft) == sorted(got)


def test_exploration_does_not_depend_on_node_insertion_order():
    nodes = [(f"s{i}", f"g{i % 3}") for i in range(9)]
    m = {n: 0.3 + 0.05 * i for i, (n, _) in enumerate(nodes)}
    st = LearnerState("t")
    a = Recommender(graph_of(nodes), FixedLearner(m)).recommend(st, k=9, eps=0.3)
    b = Recommender(graph_of(nodes[::-1]), FixedLearner(m)).recommend(st, k=9, eps=0.3)
    assert ids(a) == ids(b)
    assert ids(a) != ids(Recommender(graph_of(nodes), FixedLearner(m)).recommend(st, k=9))


def test_small_eps_mostly_keeps_the_top_pick(G, L, R):
    """With the demo value 0.05 the best skill usually stays first: exploration is a nudge, not a shuffle."""
    st = pupil(G, L)
    top = ids(R.recommend(st, k=1))[0]
    same = 0
    for step in range(200):
        probe = LearnerState.from_dict({**st.to_dict(), "learner_id": f"probe{step}"})
        same += ids(R.recommend(probe, k=1, eps=0.05))[0] == top
    assert 0 < same < 200  # it does explore, and it does not always explore
