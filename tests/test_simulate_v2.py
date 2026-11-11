"""Simulator v2 (PRD Phase 6.3 and 6.4): profiles, policies, forgetting over simulated days, the report
and the blocks written to data/eval_results.json. The v1 simulator is tested in `tests/test_simulate.py`."""

import json
import random
import statistics
from datetime import datetime
from pathlib import Path

import networkx as nx
import pytest

from kwan.eval import simulate
from kwan.learner.elo import EloLearner, LearnerState
from kwan.recommend.recommender import Recommender

ROOT = Path(__file__).resolve().parent.parent
GRAPH_PATH = ROOT / "data" / "graph" / "math.graph.json"  # the committed v2 graph
GRAPH_V1 = Path(__file__).resolve().parent / "fixtures" / "graph_v1.json"  # the frozen v1 graph
RESULTS_PATH = ROOT / "data" / "eval_results.json"
EM_DASH = chr(0x2014)
FIVE = ["kwan-strict", "kwan-explore", "syllabus", "random", "remedial-first"]
PROFILE_NAMES = ["beginner_p1", "gapped_p4", "gapped_p6", "uneven_strands"]


@pytest.fixture(scope="module")
def world(G):
    return simulate.World(G)


@pytest.fixture(scope="module")
def small(G):
    """A small real report: 2 learners, 40 steps, goal runs of at most 60 steps."""
    return simulate.report(G, n_learners=2, steps=40, goal_max_steps=60, retention_days=30)


# ----------------------------------------------------------------------------- profiles and policies


def test_the_four_profiles():
    assert list(simulate.PROFILES_V2) == PROFILE_NAMES
    known = {name: p.known for name, p in simulate.PROFILES_V2.items()}
    for strand in (1, 2, 3, 4):
        assert [known["beginner_p1"](g, strand) for g in range(1, 7)] == [0.15, 0.15, 0.03, 0.03, 0.03, 0.03]
        assert [known["gapped_p4"](g, strand) for g in range(1, 7)] == [0.75, 0.75, 0.75, 0.05, 0.05, 0.05]
        assert [known["gapped_p6"](g, strand) for g in range(1, 7)] == [0.80, 0.80, 0.80, 0.50, 0.50, 0.05]
    uneven = known["uneven_strands"]
    assert [uneven(g, simulate.NUMBER) for g in range(1, 7)] == [0.85, 0.85, 0.85, 0.85, 0.05, 0.05]
    assert [uneven(g, simulate.GEOMETRY_AND_MEASUREMENT) for g in range(1, 7)] == [0.15, 0.15, 0.15, 0.15, 0.05, 0.05]
    assert uneven(2, simulate.ALGEBRA) == uneven(4, simulate.DATA) == 0.50
    assert uneven(3, simulate.NUMBER) > uneven(3, simulate.ALGEBRA) > uneven(3, simulate.GEOMETRY_AND_MEASUREMENT)


def test_the_v1_profiles_are_the_first_two_v2_profiles():
    for name, v1 in simulate.PROFILES.items():
        assert [simulate.PROFILES_V2[name].known(g, 1) for g in range(1, 7)] == [v1(g) for g in range(1, 7)]


def test_profiles_say_plainly_that_they_are_assumptions():
    for name, prof in simulate.PROFILES_V2.items():
        assert prof.about.startswith("ASSUMPTION, not data."), name
        assert EM_DASH not in prof.about
        assert 1 <= prof.class_level <= 6 and 1 <= prof.goal_grade <= 6
    assert "assumption" in simulate.__doc__.lower()
    assert "not measurements" in simulate.ASSUMPTIONS["text"]
    assert simulate.PROFILES_V2["uneven_strands"].goal_strand == simulate.GEOMETRY_AND_MEASUREMENT


def test_the_strand_numbers_match_the_graph(G):
    names = {d["strand"]: d["strand_name"] for _, d in G.nodes(data=True)}
    assert names[simulate.NUMBER] == "Number"
    assert names[simulate.GEOMETRY_AND_MEASUREMENT] == "Geometry and Measurement"
    assert names[simulate.ALGEBRA] == "Algebra" and names[simulate.DATA] == "Data"


def test_the_policies():
    assert list(simulate.POLICIES_V2) == [
        "kwan-strict",
        "kwan-explore",
        "kwan-top1",
        "syllabus",
        "random",
        "remedial-first",
    ]
    assert set(FIVE) <= set(simulate.POLICIES_V2)  # the five of the PRD, plus one sensitivity check
    for text in simulate.POLICIES_V2.values():
        assert text.endswith(".") and EM_DASH not in text
    assert simulate.POLICIES == ("kwan", "syllabus", "random")  # v1 names are untouched
    assert (simulate.N_LEARNERS_V2, simulate.STEPS, simulate.RETENTION_DAYS) == (30, 150, 30)


# ----------------------------------------------------------------------------- the start state


def test_the_start_state_depends_only_on_profile_and_seed(world):
    known, goal = simulate.start_state(world, "gapped_p4", 3)
    assert simulate.start_state(world, "gapped_p4", 3) == (known, goal)
    assert simulate.start_state(world, "gapped_p4", 4)[0] != known
    assert simulate.start_state(world, "gapped_p6", 3)[0] != known
    assert set(known) == set(world.G)


def test_start_knowledge_follows_the_profile(G, world):
    share = {}
    for name in PROFILE_NAMES:
        hits, total = {}, {}
        for seed in range(40):
            known, _ = simulate.start_state(world, name, seed)
            for n, k in known.items():
                key = (G.nodes[n]["grade"] <= 3, G.nodes[n]["strand"])
                hits[key] = hits.get(key, 0) + k
                total[key] = total.get(key, 0) + 1
        share[name] = {key: hits[key] / total[key] for key in total}
    assert share["gapped_p4"][(True, 1)] == pytest.approx(0.75, abs=0.03)
    assert share["gapped_p4"][(False, 1)] == pytest.approx(0.05, abs=0.02)
    assert share["uneven_strands"][(True, 1)] == pytest.approx(0.85, abs=0.03)
    assert share["uneven_strands"][(True, 3)] == pytest.approx(0.15, abs=0.04)
    assert share["beginner_p1"][(False, 1)] == pytest.approx(0.03, abs=0.02)


def test_the_goal_is_an_unknown_skill_of_the_goal_grade_with_a_prerequisite(G, world):
    for name, prof in simulate.PROFILES_V2.items():
        goals = set()
        for seed in range(12):
            known, goal = simulate.start_state(world, name, seed)
            assert goal is not None and not known[goal]
            assert G.nodes[goal]["grade"] == prof.goal_grade
            assert G.in_degree(goal) > 0
            if prof.goal_strand is not None:
                assert G.nodes[goal]["strand"] == prof.goal_strand
            goals.add(goal)
        assert len(goals) > 1


# ----------------------------------------------------------------------------- one learner


@pytest.mark.parametrize("policy", list(simulate.POLICIES_V2))
def test_a_run_is_deterministic_and_well_formed(G, world, policy):
    for profile in ("gapped_p4", "uneven_strands"):
        a = simulate.simulate_learner(G, policy, profile, seed=2, steps=40, world=world)
        assert simulate.simulate_learner(G, policy, profile, seed=2, steps=40, world=world) == a
        assert simulate.simulate_learner(G, policy, profile, seed=2, steps=40) == a  # with a world of its own
        assert set(a) == {"new", "retained", "start", "goal", "reached", "steps_to_goal", "steps", "review_steps"}
        assert 0 <= a["retained"] <= a["new"] <= 40
        assert a["steps"] == 40 and a["goal"] is None and a["reached"] is False and a["steps_to_goal"] is None
        assert a["start"] == sum(simulate.start_state(world, profile, 2)[0].values())
        assert 0 <= a["review_steps"] <= 40
        if not policy.startswith("kwan") or policy == "kwan-top1":
            assert a["review_steps"] == 0


def test_different_seeds_give_different_learners(G, world):
    for policy in ("kwan-strict", "random"):
        runs = {
            simulate.simulate_learner(G, policy, "beginner_p1", seed=s, steps=60, world=world)["new"] for s in range(8)
        }
        assert len(runs) > 1


def test_zero_steps_and_impossible_learning(G, world):
    r = simulate.simulate_learner(G, "kwan-strict", "gapped_p4", steps=0, world=world)
    assert (r["new"], r["retained"], r["steps"]) == (0, 0, 0)
    for policy in simulate.POLICIES_V2:
        r = simulate.simulate_learner(
            G, policy, "gapped_p4", seed=1, steps=30, world=world, p_learn=0.0, p_learn_blocked=0.0
        )
        assert r["new"] == r["retained"] == 0


def test_prerequisites_gate_learning(G, world):
    gated = simulate.simulate_learner(G, "random", "beginner_p1", seed=5, world=world, p_learn=1.0, p_learn_blocked=0.0)
    free = simulate.simulate_learner(G, "random", "beginner_p1", seed=5, world=world, p_learn=1.0, p_learn_blocked=1.0)
    assert 0 < gated["new"] < free["new"]


def test_unknown_policy_profile_and_bad_sizes_are_rejected(G):
    with pytest.raises(ValueError, match="unknown policy 'kwan'"):
        simulate.simulate_learner(G, "kwan", "gapped_p4")
    with pytest.raises(ValueError, match="unknown profile 'p9'"):
        simulate.simulate_learner(G, "random", "p9")
    with pytest.raises(ValueError, match="1 or more"):
        simulate.simulate_learner(G, "random", "gapped_p4", steps_per_day=0)
    with pytest.raises(ValueError, match="n_learners"):
        simulate.report(G, n_learners=0)


def test_a_run_leaves_the_graph_untouched(G, world):
    first = world.syllabus[0]
    before = (G.number_of_nodes(), G.number_of_edges(), dict(G.nodes[first]))
    simulate.simulate_learner(G, "remedial-first", "gapped_p6", steps=20, world=world)
    assert (G.number_of_nodes(), G.number_of_edges(), dict(G.nodes[first])) == before


# ----------------------------------------------------------------------------- forgetting


def test_nothing_is_lost_without_a_break_and_more_is_lost_over_a_longer_one(G, world):
    def total(days, key="retained"):
        return sum(
            simulate.simulate_learner(
                G, "kwan-strict", "beginner_p1", seed=s, steps=100, world=world, retention_days=days
            )[key]
            for s in range(6)
        )

    new = total(0, "new")
    assert total(0) == new > 0  # no break: every new skill is still known
    assert total(0, "new") == total(30, "new") == total(365, "new")  # the break comes after the count
    assert new > total(30) > total(365)


def test_forgetting_also_happens_during_the_run(G, world):
    """With one step a day the same 100 steps take 100 days, so fewer new skills survive to the end."""

    def new(per_day):
        return sum(
            simulate.simulate_learner(
                G, "syllabus", "beginner_p1", seed=s, steps=100, world=world, steps_per_day=per_day
            )["new"]
            for s in range(6)
        )

    assert new(100) > new(1)


def test_memory_strength_constants():
    assert simulate.STABILITY_NEW == 45.0 and simulate.STABILITY_MAX == 365.0 and simulate.STABILITY_GROWTH == 2.0
    assert (simulate.P_LEARN, simulate.P_LEARN_BLOCKED, simulate.SLIP, simulate.GUESS) == (0.35, 0.03, 0.1, 0.2)
    a = simulate.ASSUMPTIONS
    assert (a["p_learn"], a["stability_new_days"], a["steps_per_day"]) == (0.35, 45.0, 5)
    assert EM_DASH not in a["text"]


# ----------------------------------------------------------------------------- what the policies ask of the engine


def spy_on_recommend(world, monkeypatch):
    calls = []
    real = world.R.recommend

    def recommend(st, **kw):
        calls.append(kw)
        return real(st, **kw)

    monkeypatch.setattr(world.R, "recommend", recommend)
    return calls


def test_kwan_strict_uses_the_library_defaults_with_lists_of_three(G, monkeypatch):
    world = simulate.World(G)
    calls = spy_on_recommend(world, monkeypatch)
    simulate.simulate_learner(G, "kwan-strict", "gapped_p4", steps=30, world=world)
    assert len(calls) == 10  # one list of 3 for every 3 steps
    for kw in calls:
        assert set(kw) == {"k", "goal", "now", "eps"}  # everything else is the library default
        assert kw["k"] == 3 and kw["goal"] is None and kw["eps"] == 0.0
        assert isinstance(kw["now"], datetime) and kw["now"].tzinfo is not None
    days = sorted({(kw["now"] - simulate.START).days for kw in calls})
    assert days == [0, 1, 2, 3, 4, 5]  # 30 steps at 5 a day


def test_kwan_explore_and_top1_differ_only_as_documented(G, monkeypatch):
    world = simulate.World(G)
    calls = spy_on_recommend(world, monkeypatch)
    simulate.simulate_learner(G, "kwan-explore", "gapped_p4", steps=9, world=world)
    assert [(kw["k"], kw["eps"]) for kw in calls] == [(3, 0.05)] * 3
    calls.clear()
    simulate.simulate_learner(G, "kwan-top1", "gapped_p4", steps=9, world=world)
    assert [(kw["k"], kw["eps"]) for kw in calls] == [(1, 0.0)] * 9


def test_goal_runs_tell_kwan_the_goal_and_stop_when_it_is_known(G, monkeypatch):
    world = simulate.World(G)
    calls = spy_on_recommend(world, monkeypatch)
    _, goal = simulate.start_state(world, "beginner_p1", 0)
    r = simulate.simulate_learner(G, "kwan-strict", "beginner_p1", seed=0, steps=300, with_goal=True, world=world)
    assert r["goal"] == goal and {kw["goal"] for kw in calls} == {goal}
    assert r["reached"] and r["steps_to_goal"] == r["steps"] <= 300
    short = simulate.simulate_learner(G, "random", "gapped_p6", seed=0, steps=5, with_goal=True, world=world)
    assert not short["reached"] and short["steps_to_goal"] is None and short["steps"] == 5


def test_the_engine_gets_a_timestamp_with_every_answer(G, monkeypatch):
    world = simulate.World(G)
    stamps = []
    real = world.L.update

    def update(st, ind, correct, item=None, at=None, **kw):
        stamps.append(at)
        return real(st, ind, correct, item, at=at, **kw)

    monkeypatch.setattr(world.L, "update", update)
    simulate.simulate_learner(G, "syllabus", "gapped_p4", steps=12, world=world)
    assert len(stamps) == 12 and all(isinstance(t, datetime) and t.tzinfo is not None for t in stamps)
    assert stamps == sorted(stamps) and len(set(stamps)) == 12
    assert [(t - simulate.START).days for t in stamps] == [0] * 5 + [1] * 5 + [2] * 2


def chain_world():
    """a -> b -> c -> d in class 4, all in Number, plus e in class 1 on its own."""
    g = nx.DiGraph()
    for n, grade in [("a", 4), ("b", 4), ("c", 4), ("d", 4), ("e", 1)]:
        g.add_node(n, grade=grade, strand=1, strand_name="Number", description=n)
    g.add_edges_from([("a", "b"), ("b", "c"), ("c", "d")])
    return g


def test_remedial_first_goes_back_to_the_earliest_weak_prerequisite(monkeypatch):
    """Certain learning and no slips: the pupil's class syllabus is walked through its prerequisites."""
    g = chain_world()
    world = simulate.World(g)
    practised = []
    real = world.L.update

    def update(st, ind, correct, item=None, at=None, **kw):
        practised.append(ind)
        return real(st, ind, correct, item, at=at, **kw)

    monkeypatch.setattr(world.L, "update", update)
    monkeypatch.setattr(simulate, "start_state", lambda w, profile, seed: ({n: False for n in w.nodes}, "d"))
    simulate.simulate_learner(g, "remedial-first", "gapped_p4", steps=12, world=world, p_learn=1.0, slip=0.0, guess=0.0)
    firsts = list(dict.fromkeys(practised))
    assert firsts[:4] == ["a", "b", "c", "d"]  # prerequisites first, in order, then the skill itself
    assert practised[:3] == ["a", "a", "a"]  # stays on a skill until the learner model counts it as ready
    practised.clear()
    simulate.simulate_learner(g, "syllabus", "gapped_p4", steps=3, world=world, p_learn=1.0, slip=0.0, guess=0.0)
    assert practised[0] == "e"  # plain syllabus order starts in class 1


def test_world_orderings(G, world):
    from kwan import ids

    resolve = ids.Resolver(G).resolve  # bare codes on the v1 graph, canonical ids on the v2 graph
    assert world.syllabus[0] == resolve("B1.1.1.1.1") and len(world.syllabus) == len(G)
    grades = [G.nodes[n]["grade"] for n in world.syllabus]
    assert grades == sorted(grades)
    targets = world.targets(4)
    assert G.nodes[targets[0]]["grade"] == 4 and sorted(targets) == sorted(G)
    assert [G.nodes[n]["grade"] for n in targets[-3:]] == [3, 3, 3]
    goal = resolve("B4.1.1.3.3")
    anc = world.ancestors(goal)
    assert set(anc) == nx.ancestors(G, goal) and anc == sorted(anc, key=world.syllabus.index)


# ----------------------------------------------------------------------------- any graph


def test_a_namespaced_graph_gives_the_same_simulation(G_v1):
    G, world = G_v1, simulate.World(G_v1)  # the bare-id graph and the same graph with namespaced ids
    H = nx.relabel_nodes(G, {n: f"math:{n}" for n in G}, copy=True)
    for n, d in H.nodes(data=True):
        d["code"] = n.split(":", 1)[1]
    hw = simulate.World(H)
    for policy in simulate.POLICIES_V2:
        a = simulate.simulate_learner(G, policy, "gapped_p4", seed=1, steps=45, world=world)
        b = simulate.simulate_learner(H, policy, "gapped_p4", seed=1, steps=45, world=hw)
        assert a == b, policy
    a = simulate.simulate_learner(G, "kwan-strict", "gapped_p4", seed=1, steps=80, with_goal=True, world=world)
    b = simulate.simulate_learner(H, "kwan-strict", "gapped_p4", seed=1, steps=80, with_goal=True, world=hw)
    assert b["goal"] == f"math:{a['goal']}" and {**b, "goal": a["goal"]} == a
    assert simulate.run(H, "kwan", steps=40, seed=2) == simulate.run(G, "kwan", steps=40, seed=2)  # v1 too


def test_load_graph_reads_the_v1_file_and_a_v2_file(G_v1, tmp_path):
    from kwan.graph import schema

    G = G_v1
    v1 = simulate.load_graph(GRAPH_V1)
    assert sorted(v1) == sorted(G) and v1.number_of_edges() == G.number_of_edges()
    assert simulate.run(v1, "kwan", steps=30, seed=1) == simulate.run(G, "kwan", steps=30, seed=1)
    path = tmp_path / "math.graph.json"
    schema.save_graph(schema.strip_text(v1) if hasattr(schema, "strip_text") else v1, path)
    v2 = simulate.load_graph(path)
    assert all(str(n).startswith("math:") for n in v2) and len(v2) == len(G)
    r = simulate.simulate_learner(v2, "kwan-strict", "gapped_p4", seed=1, steps=30)
    assert r == simulate.simulate_learner(G, "kwan-strict", "gapped_p4", seed=1, steps=30)


# ----------------------------------------------------------------------------- the report


def test_report_structure(small, G):
    assert list(small) == [
        "setup",
        "graph",
        "n_learners",
        "steps",
        "retention_days",
        "goal_max_steps",
        "assumptions",
        "profiles",
        "policies",
        "new_skills_mastered",
        "skills_retained",
        "steps_to_goal",
        "review_steps",
        "acceptance",
    ]
    assert small["setup"] == (
        "2 simulated learners per profile and policy (seeds 0 to 1), 40 practice steps at 5 a day, "
        "then 30 days without practice; goal runs stop at 60 steps"
    )
    assert small["graph"] == {"nodes": len(G), "edges": G.number_of_edges(), "ids": "namespaced"}  # the v2 graph
    assert (small["n_learners"], small["steps"], small["retention_days"], small["goal_max_steps"]) == (2, 40, 30, 60)
    assert small["profiles"] == {name: p.about for name, p in simulate.PROFILES_V2.items()}
    assert small["policies"] == simulate.POLICIES_V2
    for key in ("new_skills_mastered", "skills_retained", "steps_to_goal"):
        assert list(small[key]) == PROFILE_NAMES
        for by_pol in small[key].values():
            assert list(by_pol) == list(simulate.POLICIES_V2)
    for by_pol in small["review_steps"].values():
        assert list(by_pol) == ["kwan-strict", "kwan-explore", "kwan-top1"]
    json.dumps(small)  # plain data


def test_report_numbers_are_the_mean_and_sd_over_the_seeded_learners(small, G, world):
    for profile, policy in (
        ("gapped_p4", "kwan-strict"),
        ("uneven_strands", "remedial-first"),
        ("gapped_p6", "random"),
    ):
        runs = [simulate.simulate_learner(G, policy, profile, seed=s, steps=40, world=world) for s in range(2)]
        for key, field in (("new_skills_mastered", "new"), ("skills_retained", "retained")):
            vals = [r[field] for r in runs]
            assert small[key][profile][policy] == {
                "mean": round(statistics.mean(vals), 1),
                "sd": round(statistics.stdev(vals), 1),
            }
        goal_runs = [
            simulate.simulate_learner(G, policy, profile, seed=s, steps=60, with_goal=True, world=world)
            for s in range(2)
        ]
        taken = [r["steps_to_goal"] if r["reached"] else 60 for r in goal_runs]
        cell = small["steps_to_goal"][profile][policy]
        assert cell["mean"] == round(statistics.mean(taken), 1) and cell["sd"] == round(statistics.stdev(taken), 1)
        assert cell["reached"] == round(sum(r["reached"] for r in goal_runs) / 2, 2)


def test_report_values_are_in_range(small):
    for profile in PROFILE_NAMES:
        for policy in simulate.POLICIES_V2:
            new, kept = small["new_skills_mastered"][profile][policy], small["skills_retained"][profile][policy]
            assert 0 <= kept["mean"] <= new["mean"] <= 40
            goal = small["steps_to_goal"][profile][policy]
            assert 1 <= goal["mean"] <= 60 and 0 <= goal["reached"] <= 1
            assert all(isinstance(v, float) for v in (*new.values(), *kept.values(), *goal.values()))


def test_one_learner_gives_zero_spread(G):
    rep = simulate.report(G, n_learners=1, steps=10, goal_max_steps=10, retention_days=5)
    assert all(cell["sd"] == 0.0 for by_pol in rep["new_skills_mastered"].values() for cell in by_pol.values())


def test_format_report(small):
    text = simulate.format_report(small)
    lines = text.splitlines()
    assert lines[0] == small["setup"]
    assert len(lines) >= 1 + 4 * len(simulate.POLICIES_V2) + 1
    assert lines[1].startswith("beginner_p1     kwan-strict     mastered ")
    assert "retained" in lines[1] and "steps to goal" in lines[1] and "(reached " in lines[1]
    assert any(line.startswith("acceptance (PRD Phase 6): ") for line in lines)
    assert EM_DASH not in text


# ----------------------------------------------------------------------------- acceptance check


def table(**rows):
    """{profile: {policy: mean}} into the report shape."""
    return {prof: {pol: {"mean": mu, "sd": 1.0} for pol, mu in by_pol.items()} for prof, by_pol in rows.items()}


GOOD = dict(
    beginner_p1={"kwan-strict": 20.0, "syllabus": 20.9, "random": 8.0},
    gapped_p4={"kwan-strict": 12.0, "syllabus": 8.0, "random": 9.0},
    gapped_p6={"kwan-strict": 9.0, "syllabus": 6.0, "random": 8.9},
    uneven_strands={"kwan-strict": 9.0, "syllabus": 10.0, "random": 12.0},
)


def test_acceptance_met():
    acc = simulate.acceptance(table(**GOOD), table(**GOOD))
    assert acc == {
        "kwan_strict_beats_syllabus_and_random_on_gapped_profiles": True,
        "kwan_strict_within_1_skill_of_syllabus_on_every_profile": True,
        "met": True,
        "failed": [],
    }


def test_acceptance_fails_when_a_baseline_wins_or_ties_on_a_gapped_profile():
    bad = {**GOOD, "gapped_p6": {"kwan-strict": 9.0, "syllabus": 6.0, "random": 9.0}}  # a tie is not a win
    acc = simulate.acceptance(table(**GOOD), table(**bad))
    assert acc["kwan_strict_beats_syllabus_and_random_on_gapped_profiles"] is False
    assert acc["kwan_strict_within_1_skill_of_syllabus_on_every_profile"] is True
    assert acc["met"] is False
    assert acc["failed"] == ["skills_retained, gapped_p6: kwan-strict 9.0 is not above random 9.0"]


def test_acceptance_fails_when_kwan_is_more_than_one_skill_below_syllabus():
    bad = {**GOOD, "uneven_strands": {"kwan-strict": 9.0, "syllabus": 10.1, "random": 12.0}}
    acc = simulate.acceptance(table(**bad), table(**GOOD))
    assert acc["kwan_strict_beats_syllabus_and_random_on_gapped_profiles"] is True
    assert acc["kwan_strict_within_1_skill_of_syllabus_on_every_profile"] is False
    assert acc["failed"] == ["new_skills_mastered, uneven_strands: kwan-strict 9.0 is more than 1 below syllabus 10.1"]
    assert simulate.acceptance(table(**GOOD), table(**GOOD))["met"]  # exactly 1 below (uneven_strands) is allowed


def test_the_report_carries_its_own_acceptance_check(small):
    assert small["acceptance"] == simulate.acceptance(small["new_skills_mastered"], small["skills_retained"])


# ----------------------------------------------------------------------------- blocks and the command line


def test_simulation_block_keeps_the_v1_fields_at_the_top():
    v1 = {"gapped_p4": {"kwan": (14.0, 3.4), "syllabus": (8.84, 2.7)}}
    block = simulate.simulation_block(v1, {"anything": 1})
    assert block == {
        "setup": "30 simulated learners x 150 practice steps, NaCCA B1-B6 graph",
        "new_skills_mastered": {"gapped_p4": {"kwan": 14.0, "syllabus": 8.8}},
        "v2": {"anything": 1},
    }
    assert simulate.simulation_block(v1) == simulate.results_block(v1)


def reference_on_path_share(G, pupils=50, k=3):
    """The protocol of tests/test_recommender.py::test_strict_goal_on_path_share, with the default recommender."""
    L = EloLearner(G=G)
    R = Recommender(G, L)
    lower = sorted(n for n in G if G.nodes[n]["grade"] <= 3)
    upper = sorted(n for n in G if G.nodes[n]["grade"] >= 4)
    picks = {"strict": [0, 0], "soft": [0, 0]}
    for seed in range(pupils):
        rng = random.Random(seed)
        st = LearnerState(f"pupil{seed}")
        for _ in range(rng.randint(0, 60)):
            L.update(st, rng.choice(lower), rng.random() < 0.85)
        goal = rng.choice(upper)
        path = nx.ancestors(G, goal) | {goal}
        for mode, count in picks.items():
            recs = [r["indicator"] for r in R.recommend(st, k=k, goal=goal, goal_mode=mode)["recommendations"]]
            count[0] += sum(n in path for n in recs)
            count[1] += len(recs)
    return {mode: on / total for mode, (on, total) in picks.items()}


def test_on_path_share_follows_the_test_protocol(G):
    share = simulate.on_path_share(G)
    assert share == reference_on_path_share(G)
    assert share["strict"] >= 0.70 and share["strict"] > share["soft"]
    block = simulate.goal_path_block(G)
    assert list(block) == ["setup", "pupils", "k", "on_path_share"]
    assert (block["pupils"], block["k"]) == (50, 3)
    assert block["on_path_share"] == {mode: round(v, 3) for mode, v in share.items()}
    assert EM_DASH not in block["setup"]


def test_out_writes_the_v2_report_and_the_goal_path_block(G, tmp_path, monkeypatch, capsys):
    """The real v2 simulator through the command line, one learner per cell; the v1 table is stubbed."""
    out = tmp_path / "eval_results.json"
    out.write_text(json.dumps({"latency": {"calls": 3}}), encoding="utf-8")
    monkeypatch.setattr(simulate, "compare_all", lambda G, *a, **kw: {"gapped_p4": {"kwan": (14.0, 3.4)}})
    simulate.main([str(GRAPH_PATH), "--out", str(out), "--learners", "1"])
    written = json.loads(out.read_text(encoding="utf-8"))
    assert list(written) == ["latency", "simulation", "goal_path"] and written["latency"] == {"calls": 3}
    sim = written["simulation"]
    assert list(sim) == ["setup", "new_skills_mastered", "v2"]
    assert sim["new_skills_mastered"] == {"gapped_p4": {"kwan": 14.0}}
    assert sim["v2"] == json.loads(json.dumps(simulate.report(G, n_learners=1)))
    assert written["goal_path"] == simulate.goal_path_block(G)
    printed = capsys.readouterr().out
    assert simulate.format_report(simulate.report(G, n_learners=1)) in printed
    assert "on-path share of the top 3: strict " in printed and EM_DASH not in printed


# ----------------------------------------------------------------------------- the committed results


@pytest.fixture(scope="module")
def committed():
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


def test_committed_simulation_block_has_the_v1_fields_and_the_v2_report(committed):
    sim = committed["simulation"]
    assert list(sim) == ["setup", "new_skills_mastered", "v2"]
    assert sim["setup"] == "30 simulated learners x 150 practice steps, NaCCA B1-B6 graph"
    assert list(sim["new_skills_mastered"]) == ["beginner_p1", "gapped_p4"]
    v2 = sim["v2"]
    assert (v2["n_learners"], v2["steps"], v2["retention_days"], v2["goal_max_steps"]) == (30, 150, 30, 300)
    assert v2["assumptions"] == simulate.ASSUMPTIONS
    assert v2["profiles"] == {name: p.about for name, p in simulate.PROFILES_V2.items()}
    assert v2["policies"] == simulate.POLICIES_V2
    for key in ("new_skills_mastered", "skills_retained", "steps_to_goal"):
        assert list(v2[key]) == PROFILE_NAMES
        assert all(list(by_pol) == list(simulate.POLICIES_V2) for by_pol in v2[key].values())
    assert v2["acceptance"] == simulate.acceptance(v2["new_skills_mastered"], v2["skills_retained"])
    assert EM_DASH not in json.dumps(sim, ensure_ascii=False)


def test_committed_goal_path_block(committed):
    block = committed["goal_path"]
    assert list(block) == ["setup", "pupils", "k", "on_path_share"]
    share = block["on_path_share"]
    assert share["strict"] >= 0.70 and share["strict"] > share["soft"]
