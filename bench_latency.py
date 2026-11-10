"""Latency benchmark for the recommender (PRD NFR1: top-3 under 5 ms p95 on one CPU core).

Times `Recommender.recommend(state, k=3)` in-process, wall clock, one call at a time, on three seeded
learner states: a fresh learner, the gapped P4 demo learner and a learner working towards a goal.
The numbers are whatever this machine measures: nothing is filtered out except a short warm-up.

Run:  python bench_latency.py [GRAPH] [--calls N] [--out data/eval_results.json]
"""
import math, random, statistics, time
import engine

HARDWARE = "single CPU core, pure Python"
CALLS = 1000          # timed calls per learner state
WARMUP = 50           # untimed calls per learner state before timing starts
GOAL = "B4.1.1.3.3"

def fresh_learner(G, L):
    return engine.LearnerState("fresh")

def gapped_p4_learner(G, L):
    """The demo pupil loaded by the button in app.py: seeded answers on the first 60 skills in syllabus order."""
    st = engine.LearnerState("gapped_p4"); rng = random.Random(7)
    for n in sorted(G, key=lambda x: (G.nodes[x]["grade"], x))[:60]:
        if rng.random() < 0.35:
            L.update(st, n, rng.random() < (0.85 if G.nodes[n]["grade"] <= 3 else 0.2))
    return st

def goal_learner(G, L):
    """The PRD Appendix B.1 learner: three correct answers on each of the first two B1 skills."""
    st = engine.LearnerState("goal")
    for ind in ("B1.1.1.1.1", "B1.1.1.1.2"):
        for _ in range(3): L.update(st, ind, True)
    return st

SCENARIOS = {                                   # label -> (state builder, goal passed to recommend)
    "fresh learner": (fresh_learner, None),
    "gapped P4 learner": (gapped_p4_learner, None),
    "learner with a goal": (goal_learner, GOAL),
}

def percentile(values, q):
    """Nearest-rank percentile: the smallest value with at least q percent of the values at or below it."""
    if not values: raise ValueError("percentile of an empty list")
    s = sorted(values)
    return s[max(0, math.ceil(q / 100 * len(s)) - 1)]

def time_calls(R, st, calls=CALLS, goal=None, warmup=WARMUP, clock_ns=time.perf_counter_ns):
    """Wall time in ms of each of `calls` top-3 recommendations for one learner state."""
    for _ in range(warmup): R.recommend(st, k=3, goal=goal)
    out = []
    for _ in range(calls):
        t0 = clock_ns()
        R.recommend(st, k=3, goal=goal)
        out.append((clock_ns() - t0) / 1e6)
    return out

def measure(G, calls=CALLS, warmup=WARMUP):
    """{scenario label: [ms per call]} for every scenario."""
    if calls < 1: raise ValueError(f"calls must be 1 or more, got {calls}")
    L = engine.EloLearner(G=G); R = engine.Recommender(G, L)
    return {name: time_calls(R, build(G, L), calls, goal, warmup) for name, (build, goal) in SCENARIOS.items()}

def results_block(samples, hardware=HARDWARE):
    """The `latency` block of data/eval_results.json from a `measure` result, pooled over all scenarios."""
    ms = [t for ts in samples.values() for t in ts]
    return {
        "recommend_top3_mean_ms": round(statistics.mean(ms), 2),
        "recommend_top3_p95_ms": round(percentile(ms, 95), 2),
        "calls": len(ms),
        "hardware": hardware,
    }

def main(argv=None):
    import argparse, graph
    ap = argparse.ArgumentParser(description="Time Recommender.recommend(state, k=3) on three seeded learner states.")
    ap.add_argument("graph", nargs="?", default=str(graph.DEFAULT_PATH),
                    help="curriculum graph JSON (default: the committed data/graph.json)")
    ap.add_argument("--calls", type=int, default=CALLS, metavar="N",
                    help=f"timed calls per learner state (default: {CALLS})")
    ap.add_argument("--hardware", default=HARDWARE, metavar="TEXT",
                    help=f"what to record as the hardware (default: '{HARDWARE}')")
    ap.add_argument("--out", metavar="PATH",
                    help="also write the 'latency' block to this results file, e.g. data/eval_results.json")
    args = ap.parse_args(argv)
    if args.calls < 1: ap.error("--calls must be 1 or more")
    samples = measure(graph.load(args.graph), calls=args.calls)
    for name, ms in samples.items():
        print(f"{name:20s} {len(ms):6d} calls   mean {statistics.mean(ms):.2f} ms   p95 {percentile(ms, 95):.2f} ms")
    block = results_block(samples, hardware=args.hardware)
    print(f"{'all':20s} {block['calls']:6d} calls   mean {block['recommend_top3_mean_ms']:.2f} ms   "
          f"p95 {block['recommend_top3_p95_ms']:.2f} ms   ({block['hardware']})")
    if args.out:
        import results
        results.update("latency", block, path=args.out)
    return block

if __name__ == "__main__":
    main()
