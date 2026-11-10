"""Policy simulation (PRD Phase 6 and Section 8): determinism, the published numbers, and the --out block."""
import contextlib
import io
import json
import statistics
from pathlib import Path

import networkx as nx
import pytest

import graph
import simulate

GRAPH_PATH = Path(__file__).resolve().parent.parent / "data" / "graph.json"

# The published run (PRD Phase 6.1 and README): 30 learners, 150 steps, seeds 0..29. (mean, sd) to 1 decimal.
REFERENCE = {
    "beginner_p1": {"kwan": (25.6, 3.4), "syllabus": (24.0, 2.7), "random": (10.4, 3.3)},
    "gapped_p4":   {"kwan": (14.0, 3.4), "syllabus": (8.8, 2.7), "random": (10.9, 3.2)},
}
REFERENCE_OUTPUT = """\
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
    ("beginner_p1", "kwan"): [29, 28, 27, 28, 24, 29, 25, 30, 24, 20, 26, 24, 21, 27, 21, 22, 30, 25, 30, 22, 29, 25, 27, 27, 28, 25, 27, 16, 28, 25],
    ("beginner_p1", "syllabus"): [26, 24, 23, 25, 24, 25, 21, 24, 23, 25, 26, 24, 21, 22, 22, 24, 28, 18, 25, 22, 29, 28, 28, 22, 20, 23, 26, 25, 19, 27],
    ("beginner_p1", "random"): [9, 11, 15, 7, 7, 10, 9, 11, 8, 13, 9, 9, 8, 14, 16, 10, 14, 15, 12, 8, 15, 11, 5, 5, 8, 9, 5, 9, 14, 15],
    ("gapped_p4", "kwan"): [14, 10, 17, 22, 9, 17, 15, 18, 14, 16, 15, 15, 10, 10, 14, 15, 13, 15, 10, 9, 19, 16, 12, 10, 14, 8, 18, 14, 16, 15],
    ("gapped_p4", "syllabus"): [13, 9, 7, 6, 11, 13, 10, 5, 7, 6, 10, 7, 7, 11, 5, 12, 8, 12, 7, 5, 10, 9, 14, 6, 11, 8, 12, 9, 6, 7],
    ("gapped_p4", "random"): [10, 14, 14, 14, 5, 15, 7, 10, 11, 15, 9, 11, 6, 13, 11, 9, 6, 19, 8, 6, 12, 11, 12, 7, 9, 12, 12, 12, 13, 13],
}


@pytest.fixture(scope="module")
def full_run(tmp_path_factory):
    """One full published run through the command line entry point, shared by the tests below (about 2 s).
    Each single-learner result is recorded on the way through."""
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
        res = simulate.main([str(GRAPH_PATH), "--out", str(out)])
    return {"res": res, "stdout": stdout.getvalue(), "out": out, "runs": runs}


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
    assert 0 <= new <= steps                                  # at most one skill can be learned per practice step


def run_reference(G, policy, seed, profile, steps=150):
    """The simulator of PRD Appendix A.4 written out independently of simulate.run."""
    import random
    import engine
    rng = random.Random(seed)
    order = list(nx.topological_sort(G)); syllabus = sorted(G, key=lambda n: (G.nodes[n]["grade"], n))
    p_known = {"beginner_p1": lambda g: 0.15 if g <= 2 else 0.03, "gapped_p4": lambda g: 0.75 if g <= 3 else 0.05}[profile]
    true = {n: rng.random() < p_known(G.nodes[n]["grade"]) for n in G}
    start = sum(true.values())
    L = engine.EloLearner(G=G); R = engine.Recommender(G, L); st = engine.LearnerState("ref")
    for _ in range(steps):
        if policy == "kwan":
            recs = R.recommend(st, k=1)["recommendations"]; n = recs[0]["indicator"] if recs else rng.choice(order)
        elif policy == "syllabus":
            n = next((x for x in syllabus if L.mastery(st, x) < 0.85), syllabus[-1])
        else:
            n = rng.choice(order)
        prereq_ok = all(true[p] for p in G.predecessors(n))
        if not true[n] and rng.random() < (0.35 if prereq_ok else 0.03): true[n] = True
        L.update(st, n, rng.random() < (0.9 if true[n] else 0.2))
    return sum(true.values()) - start


@pytest.mark.parametrize("policy", ["kwan", "syllabus", "random"])
def test_run_matches_the_reference_simulator(G, policy):
    for profile in simulate.PROFILES:
        for seed in (0, 11):
            assert simulate.run(G, policy, steps=90, seed=seed, profile=profile) == \
                run_reference(G, policy, seed, profile, steps=90)


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
    before = (G.number_of_nodes(), G.number_of_edges(), dict(G.nodes["B1.1.1.1.1"]))
    simulate.run(G, "kwan", steps=20)
    assert (G.number_of_nodes(), G.number_of_edges(), dict(G.nodes["B1.1.1.1.1"])) == before


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
        assert sd == statistics.stdev(vals)                   # sample sd over seeds 0..n-1


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
    got = {prof: {pol: (round(mu, 1), round(sd, 1)) for pol, (mu, sd) in by_pol.items()}
           for prof, by_pol in full_run["res"].items()}
    assert got == REFERENCE


def test_the_printed_table_is_byte_identical(full_run):
    assert full_run["stdout"] == REFERENCE_OUTPUT


def test_kwan_beats_both_baselines_for_gapped_p4_learners(full_run):
    res = full_run["res"]["gapped_p4"]
    assert res["kwan"][0] > res["syllabus"][0]
    assert res["kwan"][0] > res["random"][0]
    assert res["kwan"][0] / res["syllabus"][0] - 1 == pytest.approx(0.59, abs=0.01)     # "+59% vs syllabus"
    assert res["kwan"][0] / res["random"][0] - 1 == pytest.approx(0.28, abs=0.01)       # "+28% vs random"


def test_kwan_beats_both_baselines_for_beginners(full_run):
    res = full_run["res"]["beginner_p1"]
    assert res["kwan"][0] > res["syllabus"][0] > res["random"][0]


def test_every_single_learner_of_the_published_run(full_run):
    """180 simulated learners pinned one by one, so a change is caught even if it happens to keep the averages."""
    assert full_run["runs"] == REFERENCE_RUNS


def test_reference_runs_agree_with_the_reference_table():
    for (profile, policy), vals in REFERENCE_RUNS.items():
        assert len(vals) == 30
        assert (round(statistics.mean(vals), 1), round(statistics.stdev(vals), 1)) == REFERENCE[profile][policy]


# ----------------------------------------------------------------------------- the --out block

def test_out_writes_the_simulation_block_of_prd_section_8(full_run):
    written = json.loads(full_run["out"].read_text(encoding="utf-8"))
    assert written["simulation"] == {
        "setup": SETUP,
        "new_skills_mastered": {
            "beginner_p1": {"kwan": 25.6, "syllabus": 24.0, "random": 10.4},
            "gapped_p4": {"kwan": 14.0, "syllabus": 8.8, "random": 10.9},
        },
    }


def test_out_block_key_order_and_number_format(full_run):
    block = json.loads(full_run["out"].read_text(encoding="utf-8"))["simulation"]
    assert list(block) == ["setup", "new_skills_mastered"]
    assert list(block["new_skills_mastered"]) == ["beginner_p1", "gapped_p4"]
    for by_pol in block["new_skills_mastered"].values():
        assert list(by_pol) == ["kwan", "syllabus", "random"]
        assert all(isinstance(v, float) and v == round(v, 1) for v in by_pol.values())
    assert '"kwan": 14.0' in full_run["out"].read_text(encoding="utf-8")       # a whole mean is still written as 14.0


def test_out_keeps_the_other_blocks_of_the_results_file(full_run):
    written = json.loads(full_run["out"].read_text(encoding="utf-8"))
    assert written["latency"] == OTHER_BLOCK
    assert set(written) == {"latency", "simulation"}


def test_results_block_structure():
    res = {"p": {"kwan": (14, 1.0), "syllabus": (8.84, 2.0)}, "q": {"kwan": (0.25, 0.0)}}
    block = simulate.results_block(res, n_learners=5, steps=10)
    assert block == {"setup": "5 simulated learners x 10 practice steps, NaCCA B1-B6 graph",
                     "new_skills_mastered": {"p": {"kwan": 14.0, "syllabus": 8.8}, "q": {"kwan": 0.2}}}
    assert isinstance(block["new_skills_mastered"]["p"]["kwan"], float)
    assert simulate.results_block({})["setup"] == SETUP


# ----------------------------------------------------------------------------- command line plumbing (simulation stubbed)

@pytest.fixture
def stubbed(monkeypatch):
    """Replace the slow part with a stub and record which graph path was loaded."""
    loaded = []
    monkeypatch.setattr(graph, "load", lambda path=graph.DEFAULT_PATH: loaded.append(Path(path)) or "GRAPH")
    monkeypatch.setattr(simulate, "compare_all",
                        lambda G, *a, **kw: {"gapped_p4": {"kwan": (14.0, 3.4), "syllabus": (8.8, 2.7)}})
    return loaded


def test_default_graph_is_the_committed_one(stubbed, capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)                               # not the repo root: the old default "graph.json" broke here
    simulate.main([])
    assert stubbed == [Path(graph.DEFAULT_PATH)]
    assert stubbed[0] == GRAPH_PATH and stubbed[0].is_absolute()
    assert capsys.readouterr().out == ("gapped_p4    kwan      new skills mastered in 150 steps: 14.0 ± 3.4\n"
                                       "gapped_p4    syllabus  new skills mastered in 150 steps: 8.8 ± 2.7\n")


def test_positional_graph_argument_still_works(stubbed, capsys):
    simulate.main(["some/other/graph.json"])
    assert stubbed == [Path("some/other/graph.json")]


def test_without_out_nothing_is_written(stubbed, capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    simulate.main([])
    assert list(tmp_path.iterdir()) == []


def test_out_creates_the_file_and_prints_nothing_extra(stubbed, capsys, tmp_path):
    out = tmp_path / "nested" / "results.json"
    simulate.main(["data/graph.json", "--out", str(out)])
    assert json.loads(out.read_text(encoding="utf-8")) == {"simulation": {
        "setup": SETUP, "new_skills_mastered": {"gapped_p4": {"kwan": 14.0, "syllabus": 8.8}}}}
    assert len(capsys.readouterr().out.splitlines()) == 2


def test_help_text_has_no_em_dash(capsys):
    with pytest.raises(SystemExit) as exc:
        simulate.main(["--help"])
    assert exc.value.code == 0
    text = capsys.readouterr().out
    assert "--out" in text and chr(0x2014) not in text
