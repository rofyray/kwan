"""Policy simulation, v1 simulator (PRD v1 Phase 6 and Section 8): determinism, the published numbers,
and the v1 fields of the --out block. The v2 simulator is tested in `tests/test_simulate_v2.py`."""

import contextlib
import io
import json
import statistics
from pathlib import Path

import networkx as nx
import pytest

from kwan.eval import simulate
from kwan.graph import build as graph
from kwan.graph import schema

GRAPH_PATH = schema.default_graph_path()  # the committed v2 graph
GRAPH_V1 = Path(__file__).resolve().parent / "fixtures" / "graph_v1.json"  # the frozen v1 graph

# The published run on the committed v2 graph (`kwan sim`): 30 learners, 150 steps, seeds 0..29.
# (mean, sd) to 1 decimal, as printed. Regenerate with `kwan sim` after any change to the graph.
REFERENCE = {
    "beginner_p1": {"kwan": (25.2, 2.7), "syllabus": (23.7, 2.8), "random": (9.7, 3.3)},
    "gapped_p4": {"kwan": (13.5, 3.6), "syllabus": (8.8, 2.6), "random": (10.7, 3.5)},
}
REFERENCE_OUTPUT = """\
beginner_p1  kwan      new skills mastered in 150 steps: 25.2 ± 2.7
beginner_p1  syllabus  new skills mastered in 150 steps: 23.7 ± 2.8
beginner_p1  random    new skills mastered in 150 steps: 9.7 ± 3.3
gapped_p4    kwan      new skills mastered in 150 steps: 13.5 ± 3.6
gapped_p4    syllabus  new skills mastered in 150 steps: 8.8 ± 2.6
gapped_p4    random    new skills mastered in 150 steps: 10.7 ± 3.5
"""
# The v1.0 run (PRD v1 Phase 6.1): the same simulator on the frozen v1 graph (tests/fixtures/graph_v1.json).
# These six numbers must never move: they prove the v1 engine path is unchanged.
REFERENCE_V1 = {
    "beginner_p1": {"kwan": (25.6, 3.4), "syllabus": (24.0, 2.7), "random": (10.4, 3.3)},
    "gapped_p4": {"kwan": (14.0, 3.4), "syllabus": (8.8, 2.7), "random": (10.9, 3.2)},
}
REFERENCE_OUTPUT_V1 = """\
beginner_p1  kwan      new skills mastered in 150 steps: 25.6 ± 3.4
beginner_p1  syllabus  new skills mastered in 150 steps: 24.0 ± 2.7
beginner_p1  random    new skills mastered in 150 steps: 10.4 ± 3.3
gapped_p4    kwan      new skills mastered in 150 steps: 14.0 ± 3.4
gapped_p4    syllabus  new skills mastered in 150 steps: 8.8 ± 2.7
gapped_p4    random    new skills mastered in 150 steps: 10.9 ± 3.2
"""
SETUP = "30 simulated learners x 150 practice steps, NaCCA B1-B6 graph"
OTHER_BLOCK = {"recommend_top3_mean_ms": 0.0, "hardware": "placeholder written by the test"}


# Every single learner of the published run: new skills mastered for seeds 0..29.
REFERENCE_RUNS = {
    ("beginner_p1", "kwan"): [
        28,
        21,
        27,
        27,
        27,
        26,
        22,
        25,
        22,
        23,
        28,
        24,
        21,
        26,
        28,
        22,
        29,
        29,
        26,
        21,
        29,
        26,
        28,
        28,
        27,
        24,
        25,
        21,
        23,
        24,
    ],
    ("beginner_p1", "syllabus"): [
        25,
        25,
        23,
        26,
        21,
        23,
        23,
        22,
        23,
        24,
        26,
        23,
        20,
        22,
        24,
        23,
        28,
        18,
        26,
        23,
        29,
        27,
        27,
        20,
        22,
        24,
        27,
        21,
        19,
        28,
    ],
    ("beginner_p1", "random"): [
        9,
        9,
        14,
        7,
        8,
        8,
        12,
        12,
        9,
        9,
        6,
        15,
        6,
        14,
        13,
        10,
        11,
        10,
        8,
        5,
        14,
        12,
        6,
        4,
        9,
        7,
        7,
        7,
        13,
        17,
    ],
    ("gapped_p4", "kwan"): [
        15,
        15,
        9,
        14,
        9,
        19,
        7,
        16,
        13,
        12,
        18,
        15,
        14,
        11,
        13,
        12,
        11,
        12,
        10,
        9,
        17,
        16,
        21,
        11,
        15,
        15,
        16,
        9,
        21,
        11,
    ],
    ("gapped_p4", "syllabus"): [
        14,
        9,
        7,
        11,
        9,
        13,
        10,
        5,
        6,
        7,
        10,
        6,
        7,
        10,
        5,
        12,
        8,
        11,
        7,
        6,
        10,
        9,
        14,
        6,
        10,
        8,
        12,
        9,
        6,
        6,
    ],
    ("gapped_p4", "random"): [
        7,
        9,
        15,
        9,
        15,
        18,
        10,
        10,
        9,
        13,
        10,
        18,
        9,
        12,
        11,
        6,
        8,
        10,
        5,
        6,
        11,
        13,
        9,
        7,
        11,
        9,
        9,
        10,
        17,
        16,
    ],
}
# Every single learner of the v1.0 run on the frozen v1 graph.
REFERENCE_RUNS_V1 = {
    ("beginner_p1", "kwan"): [
        29,
        28,
        27,
        28,
        24,
        29,
        25,
        30,
        24,
        20,
        26,
        24,
        21,
        27,
        21,
        22,
        30,
        25,
        30,
        22,
        29,
        25,
        27,
        27,
        28,
        25,
        27,
        16,
        28,
        25,
    ],
    ("beginner_p1", "syllabus"): [
        26,
        24,
        23,
        25,
        24,
        25,
        21,
        24,
        23,
        25,
        26,
        24,
        21,
        22,
        22,
        24,
        28,
        18,
        25,
        22,
        29,
        28,
        28,
        22,
        20,
        23,
        26,
        25,
        19,
        27,
    ],
    ("beginner_p1", "random"): [
        9,
        11,
        15,
        7,
        7,
        10,
        9,
        11,
        8,
        13,
        9,
        9,
        8,
        14,
        16,
        10,
        14,
        15,
        12,
        8,
        15,
        11,
        5,
        5,
        8,
        9,
        5,
        9,
        14,
        15,
    ],
    ("gapped_p4", "kwan"): [
        14,
        10,
        17,
        22,
        9,
        17,
        15,
        18,
        14,
        16,
        15,
        15,
        10,
        10,
        14,
        15,
        13,
        15,
        10,
        9,
        19,
        16,
        12,
        10,
        14,
        8,
        18,
        14,
        16,
        15,
    ],
    ("gapped_p4", "syllabus"): [
        13,
        9,
        7,
        6,
        11,
        13,
        10,
        5,
        7,
        6,
        10,
        7,
        7,
        11,
        5,
        12,
        8,
        12,
        7,
        5,
        10,
        9,
        14,
        6,
        11,
        8,
        12,
        9,
        6,
        7,
    ],
    ("gapped_p4", "random"): [
        10,
        14,
        14,
        14,
        5,
        15,
        7,
        10,
        11,
        15,
        9,
        11,
        6,
        13,
        11,
        9,
        6,
        19,
        8,
        6,
        12,
        11,
        12,
        7,
        9,
        12,
        12,
        12,
        13,
        13,
    ],
}


@pytest.fixture(scope="module")
def full_run(tmp_path_factory):
    """One full published v1 run through the command line entry point, shared by the tests below.
    Each single-learner result is recorded on the way through. `--out` also runs the v2 simulator; it is
    kept to one learner per profile and policy here, which does not touch the v1 table."""
    out = tmp_path_factory.mktemp("sim") / "eval_results.json"
    out.write_text(json.dumps({"latency": OTHER_BLOCK}), encoding="utf-8")
    runs, real_run = {}, simulate.run

    def recording_run(G, policy, steps=simulate.STEPS, seed=0, **kw):
        new = real_run(G, policy, steps, seed=seed, **kw)
        runs.setdefault((kw.get("profile", "beginner_p1"), policy), []).append(new)
        return new

    stdout = io.StringIO()
    with pytest.MonkeyPatch.context() as mp, contextlib.redirect_stdout(stdout):
        mp.setattr(simulate, "run", recording_run)
        res = simulate.main([str(GRAPH_PATH), "--out", str(out), "--learners", "1"])
    return {"res": res, "stdout": stdout.getvalue(), "out": out, "runs": runs}


@pytest.fixture(scope="module")
def v1_run(G_v1):
    """The published v1.0 table recomputed on the frozen v1 graph, every single learner recorded."""
    runs, real_run = {}, simulate.run

    def recording_run(G, policy, steps=simulate.STEPS, seed=0, **kw):
        new = real_run(G, policy, steps, seed=seed, **kw)
        runs.setdefault((kw.get("profile", "beginner_p1"), policy), []).append(new)
        return new

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(simulate, "run", recording_run)
        res = simulate.compare_all(G_v1)
    return {"res": res, "runs": runs}


# ----------------------------------------------------------------------------- run


@pytest.mark.parametrize("policy", ["kwan", "syllabus", "random"])
def test_run_is_deterministic_per_seed(G, policy):
    for profile in simulate.PROFILES:
        a = simulate.run(G, policy, steps=60, seed=3, profile=profile)
        assert simulate.run(G, policy, steps=60, seed=3, profile=profile) == a


@pytest.mark.parametrize("policy", ["kwan", "syllabus", "random"])
def test_different_seeds_give_different_learners(G, policy):
    results = [simulate.run(G, policy, steps=80, seed=s) for s in range(8)]
    assert len(set(results)) > 1


@pytest.mark.parametrize("policy", ["kwan", "syllabus", "random"])
@pytest.mark.parametrize("profile", ["beginner_p1", "gapped_p4"])
def test_each_policy_runs_and_returns_a_count_of_new_skills(G, policy, profile):
    steps = 50
    new = simulate.run(G, policy, steps=steps, seed=1, profile=profile)
    assert isinstance(new, int)
    assert 0 <= new <= steps  # at most one skill can be learned per practice step


def run_reference(G, policy, seed, profile, steps=150):
    """The simulator of PRD Appendix A.4 written out independently of simulate.run."""
    import random

    from kwan.learner.elo import EloLearner, LearnerState
    from kwan.recommend.recommender import Recommender

    rng = random.Random(seed)
    order = list(nx.topological_sort(G))
    syllabus = sorted(G, key=lambda n: (G.nodes[n]["grade"], n))
    p_known = {"beginner_p1": lambda g: 0.15 if g <= 2 else 0.03, "gapped_p4": lambda g: 0.75 if g <= 3 else 0.05}[
        profile
    ]
    true = {n: rng.random() < p_known(G.nodes[n]["grade"]) for n in G}
    start = sum(true.values())
    L = EloLearner(G=G)
    R = Recommender(G, L)
    st = LearnerState("ref")
    for _ in range(steps):
        if policy == "kwan":
            recs = R.recommend(st, k=1, goal_mode="soft", eps=0.0, variety=0.0, review=False)["recommendations"]
            n = recs[0]["indicator"] if recs else rng.choice(order)
        elif policy == "syllabus":
            n = next((x for x in syllabus if L.mastery(st, x) < 0.85), syllabus[-1])
        else:
            n = rng.choice(order)
        prereq_ok = all(true[p] for p in G.predecessors(n))
        if not true[n] and rng.random() < (0.35 if prereq_ok else 0.03):
            true[n] = True
        L.update(st, n, rng.random() < (0.9 if true[n] else 0.2))
    return sum(true.values()) - start


@pytest.mark.parametrize("policy", ["kwan", "syllabus", "random"])
def test_run_matches_the_reference_simulator(G, policy):
    for profile in simulate.PROFILES:
        for seed in (0, 11):
            assert simulate.run(G, policy, steps=90, seed=seed, profile=profile) == run_reference(
                G, policy, seed, profile, steps=90
            )


def test_zero_steps_learn_nothing(G):
    assert simulate.run(G, "kwan", steps=0) == 0


def test_nothing_is_learned_when_practice_never_works(G):
    for policy in simulate.POLICIES:
        assert simulate.run(G, policy, steps=40, seed=2, p_learn=0.0, p_learn_blocked=0.0) == 0


def test_prerequisites_gate_learning(G):
    """With learning impossible unless prerequisites are truly known, random practice still learns something,
    and never more than when prerequisites do not matter."""
    gated = simulate.run(G, "random", steps=150, seed=5, p_learn=1.0, p_learn_blocked=0.0)
    free = simulate.run(G, "random", steps=150, seed=5, p_learn=1.0, p_learn_blocked=1.0)
    assert 0 < gated < free


def test_unknown_policy_is_rejected(G):
    with pytest.raises(ValueError, match="unknown policy 'greedy'"):
        simulate.run(G, "greedy")


def test_unknown_profile_is_rejected(G):
    with pytest.raises(ValueError, match="unknown profile 'p9'"):
        simulate.run(G, "kwan", profile="p9")


def test_run_leaves_the_graph_untouched(G):
    first = next(iter(G))
    before = (G.number_of_nodes(), G.number_of_edges(), dict(G.nodes[first]))
    simulate.run(G, "kwan", steps=20)
    assert (G.number_of_nodes(), G.number_of_edges(), dict(G.nodes[first])) == before


# ----------------------------------------------------------------------------- profiles and policies


def test_profiles_match_the_documented_starting_knowledge():
    assert list(simulate.PROFILES) == ["beginner_p1", "gapped_p4"]
    assert [simulate.PROFILES["beginner_p1"](g) for g in range(1, 7)] == [0.15, 0.15, 0.03, 0.03, 0.03, 0.03]
    assert [simulate.PROFILES["gapped_p4"](g) for g in range(1, 7)] == [0.75, 0.75, 0.75, 0.05, 0.05, 0.05]


def test_policies_and_published_run_size():
    assert simulate.POLICIES == ("kwan", "syllabus", "random")
    assert (simulate.N_LEARNERS, simulate.STEPS) == (30, 150)


# ----------------------------------------------------------------------------- compare


def test_compare_returns_mean_and_sd_per_policy(G):
    res = simulate.compare(G, n_learners=4, steps=30, profile="gapped_p4")
    assert list(res) == ["kwan", "syllabus", "random"]
    for policy, (mu, sd) in res.items():
        vals = [simulate.run(G, policy, 30, seed=s, profile="gapped_p4") for s in range(4)]
        assert mu == statistics.mean(vals)
        assert sd == statistics.stdev(vals)  # sample sd over seeds 0..n-1


def test_compare_with_one_learner_has_zero_spread(G):
    res = simulate.compare(G, n_learners=1, steps=20)
    assert all(sd == 0.0 for _, sd in res.values())
    assert res["kwan"][0] == simulate.run(G, "kwan", 20, seed=0)


def test_compare_needs_at_least_one_learner(G):
    with pytest.raises(ValueError, match="n_learners"):
        simulate.compare(G, n_learners=0)


def test_compare_all_covers_every_profile(G):
    res = simulate.compare_all(G, n_learners=2, steps=10)
    assert list(res) == list(simulate.PROFILES)
    assert res["gapped_p4"] == simulate.compare(G, n_learners=2, steps=10, profile="gapped_p4")


# ----------------------------------------------------------------------------- the published numbers


def test_the_six_reference_numbers(full_run):
    """Any change to the engine, the simulator or the graph structure that shifts the published table fails here."""
    got = {
        prof: {pol: (round(mu, 1), round(sd, 1)) for pol, (mu, sd) in by_pol.items()}
        for prof, by_pol in full_run["res"].items()
    }
    assert got == REFERENCE


def test_the_printed_table_is_byte_identical(full_run):
    assert full_run["stdout"].startswith(REFERENCE_OUTPUT + "\n")  # the six v1 lines first, then the v2 report
    assert full_run["stdout"].splitlines()[:6] == REFERENCE_OUTPUT.splitlines()


def test_without_flags_the_command_prints_exactly_the_six_v1_lines(G, capsys, monkeypatch):
    """`kwan sim data/graph/math.graph.json`: the v1.0 table and nothing else. The numbers are pinned above;
    here the v1 comparison is replaced by its published result so the test costs nothing."""
    monkeypatch.setattr(simulate, "load_graph", lambda path=None: G)
    monkeypatch.setattr(simulate, "compare_all", lambda G, *a, **kw: REFERENCE)
    monkeypatch.setattr(simulate, "report", lambda *a, **kw: pytest.fail("the v2 simulator must not run"))
    simulate.main([str(GRAPH_PATH)])
    assert capsys.readouterr().out == REFERENCE_OUTPUT


def test_kwan_beats_both_baselines_for_gapped_p4_learners(full_run):
    res = full_run["res"]["gapped_p4"]
    assert res["kwan"][0] > res["syllabus"][0]
    assert res["kwan"][0] > res["random"][0]


def test_kwan_beats_both_baselines_for_beginners(full_run):
    res = full_run["res"]["beginner_p1"]
    assert res["kwan"][0] > res["syllabus"][0] > res["random"][0]


def test_every_single_learner_of_the_published_run(full_run):
    """180 simulated learners pinned one by one, so a change is caught even if it happens to keep the averages."""
    assert full_run["runs"] == REFERENCE_RUNS


@pytest.mark.parametrize("runs, table", [(REFERENCE_RUNS, REFERENCE), (REFERENCE_RUNS_V1, REFERENCE_V1)])
def test_reference_runs_agree_with_the_reference_table(runs, table):
    for (profile, policy), vals in runs.items():
        assert len(vals) == 30
        assert (round(statistics.mean(vals), 1), round(statistics.stdev(vals), 1)) == table[profile][policy]


# ----------------------------------------------------------------------------- the v1.0 numbers on the frozen v1 graph


def test_the_v1_engine_path_is_unchanged_on_the_frozen_v1_graph(v1_run):
    """The six v1.0 numbers (PRD v1 Phase 6.1, README of 1.0) on the frozen v1 graph: the v1 graph had 266
    skills and 287 links, so they differ from the committed table, and they must never move."""
    got = {
        prof: {pol: (round(mu, 1), round(sd, 1)) for pol, (mu, sd) in by_pol.items()}
        for prof, by_pol in v1_run["res"].items()
    }
    assert got == REFERENCE_V1
    assert v1_run["runs"] == REFERENCE_RUNS_V1
    res = v1_run["res"]["gapped_p4"]
    assert res["kwan"][0] / res["syllabus"][0] - 1 == pytest.approx(0.59, abs=0.01)  # "+59% vs syllabus"
    assert res["kwan"][0] / res["random"][0] - 1 == pytest.approx(0.28, abs=0.01)  # "+28% vs random"


def test_the_v1_table_prints_as_published(G_v1, capsys, monkeypatch):
    monkeypatch.setattr(simulate, "load_graph", lambda path=None: G_v1)
    monkeypatch.setattr(simulate, "compare_all", lambda G, *a, **kw: REFERENCE_V1)
    simulate.main([str(GRAPH_V1)])
    assert capsys.readouterr().out == REFERENCE_OUTPUT_V1


# ----------------------------------------------------------------------------- the --out block


def test_out_writes_the_simulation_block_of_prd_section_8(full_run):
    written = json.loads(full_run["out"].read_text(encoding="utf-8"))
    v1_fields = {key: written["simulation"][key] for key in ("setup", "new_skills_mastered")}
    assert v1_fields == {
        "setup": SETUP,
        "new_skills_mastered": {
            prof: {pol: mu for pol, (mu, _) in by_pol.items()} for prof, by_pol in REFERENCE.items()
        },
    }


def test_out_block_key_order_and_number_format(full_run):
    block = json.loads(full_run["out"].read_text(encoding="utf-8"))["simulation"]
    assert list(block) == ["setup", "new_skills_mastered", "v2"]  # the v1 fields stay where the demo reads them
    assert list(block["new_skills_mastered"]) == ["beginner_p1", "gapped_p4"]
    for by_pol in block["new_skills_mastered"].values():
        assert list(by_pol) == ["kwan", "syllabus", "random"]
        assert all(isinstance(v, float) and v == round(v, 1) for v in by_pol.values())
    mu = REFERENCE["gapped_p4"]["kwan"][0]
    assert f'"kwan": {mu:.1f}' in full_run["out"].read_text(encoding="utf-8")  # one decimal, even for a whole mean


def test_out_keeps_the_other_blocks_of_the_results_file(full_run):
    written = json.loads(full_run["out"].read_text(encoding="utf-8"))
    assert written["latency"] == OTHER_BLOCK
    assert set(written) == {"latency", "simulation", "goal_path"}


def test_results_block_structure():
    res = {"p": {"kwan": (14, 1.0), "syllabus": (8.84, 2.0)}, "q": {"kwan": (0.25, 0.0)}}
    block = simulate.results_block(res, n_learners=5, steps=10)
    assert block == {
        "setup": "5 simulated learners x 10 practice steps, NaCCA B1-B6 graph",
        "new_skills_mastered": {"p": {"kwan": 14.0, "syllabus": 8.8}, "q": {"kwan": 0.2}},
    }
    assert isinstance(block["new_skills_mastered"]["p"]["kwan"], float)
    assert simulate.results_block({})["setup"] == SETUP


# ----------------------------------------------------------------------------- command line plumbing (simulation stubbed)


@pytest.fixture
def stubbed(monkeypatch):
    """Replace the slow part with a stub and record which graph path was loaded."""
    loaded = []
    monkeypatch.setattr(simulate, "load_graph", lambda path=None: loaded.append(Path(path)) or "GRAPH")
    monkeypatch.setattr(
        simulate, "compare_all", lambda G, *a, **kw: {"gapped_p4": {"kwan": (14.0, 3.4), "syllabus": (8.8, 2.7)}}
    )
    monkeypatch.setattr(simulate, "report", lambda G, *a, **kw: {"stub": "v2 report"})
    monkeypatch.setattr(simulate, "format_report", lambda rep: "V2 TABLE")
    monkeypatch.setattr(
        simulate, "goal_path_block", lambda G, *a, **kw: {"k": 3, "on_path_share": {"strict": 0.9, "soft": 0.5}}
    )
    return loaded


def test_default_graph_is_the_committed_one(stubbed, capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # not the repo root: the old default "graph.json" broke here
    simulate.main([])
    assert stubbed == [Path(graph.DEFAULT_PATH)] == [schema.default_graph_path()]
    assert stubbed[0] == GRAPH_PATH and stubbed[0].is_absolute() and stubbed[0].is_file()
    assert capsys.readouterr().out == (
        "gapped_p4    kwan      new skills mastered in 150 steps: 14.0 ± 3.4\n"
        "gapped_p4    syllabus  new skills mastered in 150 steps: 8.8 ± 2.7\n"
    )


def test_positional_graph_argument_still_works(stubbed, capsys):
    simulate.main(["some/other/graph.json"])
    assert stubbed == [Path("some/other/graph.json")]


def test_without_out_nothing_is_written(stubbed, capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    simulate.main([])
    assert list(tmp_path.iterdir()) == []


def test_out_creates_the_file_with_both_blocks(stubbed, capsys, tmp_path):
    out = tmp_path / "nested" / "results.json"
    simulate.main(["data/graph/math.graph.json", "--out", str(out)])
    assert json.loads(out.read_text(encoding="utf-8")) == {
        "simulation": {
            "setup": SETUP,
            "new_skills_mastered": {"gapped_p4": {"kwan": 14.0, "syllabus": 8.8}},
            "v2": {"stub": "v2 report"},
        },
        "goal_path": {"k": 3, "on_path_share": {"strict": 0.9, "soft": 0.5}},
    }
    assert capsys.readouterr().out.splitlines() == [
        "gapped_p4    kwan      new skills mastered in 150 steps: 14.0 ± 3.4",
        "gapped_p4    syllabus  new skills mastered in 150 steps: 8.8 ± 2.7",
        "",
        "V2 TABLE",
        "on-path share of the top 3: strict 0.900, soft 0.500",
    ]


def test_v2_flag_prints_the_report_and_writes_nothing(stubbed, capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    simulate.main(["--v2"])
    assert capsys.readouterr().out.splitlines()[2:] == [
        "",
        "V2 TABLE",
        "on-path share of the top 3: strict 0.900, soft 0.500",
    ]
    assert list(tmp_path.iterdir()) == []


def test_learners_must_be_positive(stubbed, capsys):
    with pytest.raises(SystemExit) as exc:
        simulate.main(["--v2", "--learners", "0"])
    assert exc.value.code == 2
    assert "--learners must be 1 or more" in capsys.readouterr().err


def test_the_default_graph_is_the_committed_v2_graph_and_the_bundle_outside_a_checkout(monkeypatch, tmp_path):
    assert simulate.default_graph_path() == graph.DEFAULT_PATH == schema.default_graph_path()
    assert simulate.default_graph_path().name == "math.graph.json" and simulate.default_graph_path().is_file()
    monkeypatch.setenv("KWAN_DATA_DIR", str(tmp_path))  # an install without the repository data
    assert simulate.default_graph_path() is None  # the loader then picks the graph of the shipped bundle
    assert simulate.load_graph(None).number_of_nodes() > 0


def test_help_text_has_no_em_dash(capsys):
    with pytest.raises(SystemExit) as exc:
        simulate.main(["--help"])
    assert exc.value.code == 0
    text = capsys.readouterr().out
    assert "--out" in text and "--v2" in text and "--learners" in text and chr(0x2014) not in text
