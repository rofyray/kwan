"""Latency benchmark (PRD NFR1 and Section 8): honest timing of top-3 recommendations and the `latency` block."""
import json

import pytest

import bench_latency
import engine

BLOCK_KEYS = ["recommend_top3_mean_ms", "recommend_top3_p95_ms", "calls", "hardware"]


# ----------------------------------------------------------------------------- percentile

def test_percentile_nearest_rank():
    assert bench_latency.percentile(list(range(1, 101)), 95) == 95
    assert bench_latency.percentile(list(range(1, 101)), 50) == 50
    assert bench_latency.percentile(list(range(1, 101)), 100) == 100
    assert bench_latency.percentile([1, 2, 3, 4], 50) == 2
    assert bench_latency.percentile([5.0, 1.0, 3.0], 95) == 5.0            # unsorted input, small sample: the maximum
    assert bench_latency.percentile(list(range(1, 21)), 95) == 19
    assert bench_latency.percentile([7.5], 95) == 7.5


def test_percentile_returns_an_observed_value():
    values = [0.21, 0.19, 0.2, 4.0, 0.18]
    assert bench_latency.percentile(values, 95) in values


def test_percentile_of_nothing_is_an_error():
    with pytest.raises(ValueError):
        bench_latency.percentile([], 95)


# ----------------------------------------------------------------------------- learner states

def test_three_scenarios_fresh_gapped_and_goal():
    assert list(bench_latency.SCENARIOS) == ["fresh learner", "gapped P4 learner", "learner with a goal"]
    goals = [goal for _, goal in bench_latency.SCENARIOS.values()]
    assert goals == [None, None, "B4.1.1.3.3"]


def test_goal_used_by_the_benchmark_is_in_the_graph(G):
    assert bench_latency.GOAL in G


def test_fresh_learner_has_no_history(G):
    st = bench_latency.fresh_learner(G, engine.EloLearner(G=G))
    assert st.step == 0 and st.theta_k == {}


def test_gapped_p4_learner_is_seeded_and_deterministic(G):
    L = engine.EloLearner(G=G)
    a, b = bench_latency.gapped_p4_learner(G, L), bench_latency.gapped_p4_learner(G, L)
    assert a == b
    assert a.n["_all"] == a.step == 26                                      # the demo pupil: 26 seeded answers
    first60 = set(sorted(G, key=lambda x: (G.nodes[x]["grade"], x))[:60])
    assert set(a.theta_k) <= first60


def test_goal_learner_is_the_appendix_b1_learner(G):
    L = engine.EloLearner(G=G)
    st = bench_latency.goal_learner(G, L)
    assert st.n == {"_all": 6, "B1.1.1.1.1": 3, "B1.1.1.1.2": 3}
    top = engine.Recommender(G, L).recommend(st, k=3, goal=bench_latency.GOAL)["recommendations"]
    assert [r["indicator"] for r in top] == ["B1.1.1.1.3", "B1.1.2.1.1", "B1.3.3.1.1"]


# ----------------------------------------------------------------------------- timing

def test_time_calls_returns_one_positive_time_per_call(G):
    L = engine.EloLearner(G=G); R = engine.Recommender(G, L)
    ms = bench_latency.time_calls(R, engine.LearnerState("t"), calls=25, warmup=2)
    assert len(ms) == 25
    assert all(isinstance(t, float) and t > 0 for t in ms)


def test_time_calls_asks_for_the_top_3_with_the_goal_and_skips_the_warm_up():
    class Spy:
        def __init__(self): self.calls = []
        def recommend(self, st, k=None, goal=None): self.calls.append((k, goal))
    spy = Spy()
    ms = bench_latency.time_calls(spy, object(), calls=4, goal="B1.1.1.1.2", warmup=3)
    assert spy.calls == [(3, "B1.1.1.1.2")] * 7                             # 3 warm-up calls + 4 timed calls
    assert len(ms) == 4


def test_time_calls_measures_wall_time_around_each_call():
    """The reported time is the clock difference across the whole call, in milliseconds."""
    now = [0]
    class Slow:
        def recommend(self, st, k=None, goal=None): now[0] += 2_500_000    # each call takes 2.5 ms of fake time
    ms = bench_latency.time_calls(Slow(), object(), calls=3, warmup=2, clock_ns=lambda: now[0])
    assert ms == [2.5, 2.5, 2.5]


def test_measure_times_every_scenario(G):
    samples = bench_latency.measure(G, calls=10, warmup=1)
    assert list(samples) == list(bench_latency.SCENARIOS)
    assert all(len(ms) == 10 for ms in samples.values())


def test_measure_needs_at_least_one_call(G):
    with pytest.raises(ValueError, match="calls"):
        bench_latency.measure(G, calls=0)


# ----------------------------------------------------------------------------- the latency block

def test_results_block_pools_all_scenarios_and_rounds_to_2_decimals():
    block = bench_latency.results_block({"a": [0.1111, 0.2222], "b": [0.3333, 0.7777]})
    assert block == {"recommend_top3_mean_ms": 0.36,                        # (0.1111+0.2222+0.3333+0.7777)/4 = 0.361075
                     "recommend_top3_p95_ms": 0.78,                         # nearest rank of 4 values: the largest
                     "calls": 4,
                     "hardware": "single CPU core, pure Python"}
    assert list(block) == BLOCK_KEYS


def test_results_block_reports_what_was_measured_even_when_it_is_slow():
    block = bench_latency.results_block({"a": [1.0] * 99 + [250.0]})
    assert block["recommend_top3_mean_ms"] == 3.49                          # the outlier is not filtered out
    assert block["recommend_top3_p95_ms"] == 1.0
    assert block["calls"] == 100


def test_results_block_hardware_can_be_described():
    assert bench_latency.results_block({"a": [1.0]}, hardware="one core of a laptop")["hardware"] == "one core of a laptop"


# ----------------------------------------------------------------------------- command line

def test_main_writes_the_latency_block_and_keeps_other_blocks(tmp_path, capsys):
    out = tmp_path / "eval_results.json"
    out.write_text(json.dumps({"simulation": {"setup": "kept"}}), encoding="utf-8")
    block = bench_latency.main(["--calls", "200", "--out", str(out)])
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["simulation"] == {"setup": "kept"}
    assert written["latency"] == block
    assert list(written["latency"]) == BLOCK_KEYS
    assert block["calls"] == 3 * 200
    assert block["hardware"] == "single CPU core, pure Python"
    for key in ("recommend_top3_mean_ms", "recommend_top3_p95_ms"):
        assert isinstance(block[key], float) and block[key] > 0
        assert block[key] == round(block[key], 2)
    assert block["recommend_top3_p95_ms"] < 5                               # PRD NFR1


def test_main_prints_one_line_per_scenario_and_a_total(tmp_path, capsys):
    block = bench_latency.main(["--calls", "20"])
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 4
    for line, name in zip(lines, [*bench_latency.SCENARIOS, "all"]):
        assert line.startswith(name) and "mean" in line and "p95" in line and " ms" in line
    assert f"{block['calls']} calls" in lines[-1] and "60 calls" in lines[-1]
    assert f"mean {block['recommend_top3_mean_ms']:.2f} ms" in lines[-1]
    assert f"p95 {block['recommend_top3_p95_ms']:.2f} ms" in lines[-1]
    assert chr(0x2014) not in "".join(lines)


def test_main_without_out_writes_nothing(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)                                             # also checks the default graph path
    bench_latency.main(["--calls", "5"])
    assert list(tmp_path.iterdir()) == []


def test_main_records_a_custom_hardware_description(tmp_path, capsys):
    out = tmp_path / "r.json"
    bench_latency.main(["--calls", "5", "--hardware", "test rig", "--out", str(out)])
    assert json.loads(out.read_text(encoding="utf-8"))["latency"]["hardware"] == "test rig"


def test_main_rejects_zero_calls(capsys):
    with pytest.raises(SystemExit) as exc:
        bench_latency.main(["--calls", "0"])
    assert exc.value.code == 2


def test_help_text_has_no_em_dash(capsys):
    with pytest.raises(SystemExit) as exc:
        bench_latency.main(["--help"])
    assert exc.value.code == 0
    text = capsys.readouterr().out
    assert "--out" in text and "--calls" in text and chr(0x2014) not in text
