"""Simulated-learner evaluation: Kwan policy vs syllabus-order vs random.
Ground truth: each simulated learner has hidden mastery per node; practising a node whose prerequisites are
truly mastered has a high chance to master it; practising without prerequisites rarely works.

Run:  python simulate.py [GRAPH] [--out data/eval_results.json]
"""
import random, statistics, networkx as nx
from engine import EloLearner, Recommender, LearnerState, MASTERY

PROFILES = {
    # probability a skill is already truly mastered at start, by grade band
    "beginner_p1": lambda g: 0.15 if g <= 2 else 0.03,
    "gapped_p4":   lambda g: 0.75 if g <= 3 else 0.05,   # typical upper-primary pupil with foundational gaps
}
POLICIES = ("kwan", "syllabus", "random")
N_LEARNERS, STEPS = 30, 150                               # the published run: seeds 0..29, 150 practice steps each

def run(G, policy, steps=STEPS, seed=0, p_learn=0.35, p_learn_blocked=0.03, slip=0.1, guess=0.2, profile="beginner_p1"):
    if policy not in POLICIES: raise ValueError(f"unknown policy {policy!r}: choose from {', '.join(POLICIES)}")
    if profile not in PROFILES: raise ValueError(f"unknown profile {profile!r}: choose from {', '.join(PROFILES)}")
    rng = random.Random(seed)
    order = list(nx.topological_sort(G)); syllabus = sorted(G, key=lambda n: (G.nodes[n]["grade"], n))
    true = {n: rng.random() < PROFILES[profile](G.nodes[n]["grade"]) for n in G}
    start = sum(true.values())
    L = EloLearner(G=G); R = Recommender(G, L); st = LearnerState("sim")
    for t in range(steps):
        if policy == "kwan":
            recs = R.recommend(st, k=1)["recommendations"]; n = recs[0]["indicator"] if recs else rng.choice(order)
        elif policy == "syllabus":
            n = next((x for x in syllabus if L.mastery(st, x) < MASTERY), syllabus[-1])
        else:
            n = rng.choice(order)
        prereq_ok = all(true[p] for p in G.predecessors(n))
        if not true[n] and rng.random() < (p_learn if prereq_ok else p_learn_blocked): true[n] = True
        correct = rng.random() < ((1 - slip) if true[n] else guess)
        L.update(st, n, correct)
    return sum(true.values()) - start            # skills newly mastered

def compare(G, n_learners=N_LEARNERS, steps=STEPS, profile="beginner_p1"):
    """{policy: (mean, sd)} of new skills mastered over learners with seeds 0..n_learners-1."""
    if n_learners < 1: raise ValueError(f"n_learners must be 1 or more, got {n_learners}")
    res = {}
    for pol in POLICIES:
        vals = [run(G, pol, steps, seed=s, profile=profile) for s in range(n_learners)]
        res[pol] = (statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0)
    return res

def compare_all(G, n_learners=N_LEARNERS, steps=STEPS):
    """{profile: {policy: (mean, sd)}} for every profile."""
    return {prof: compare(G, n_learners, steps, profile=prof) for prof in PROFILES}

def results_block(res, n_learners=N_LEARNERS, steps=STEPS):
    """The `simulation` block of data/eval_results.json (PRD Section 8) from a `compare_all` result."""
    return {
        "setup": f"{n_learners} simulated learners x {steps} practice steps, NaCCA B1-B6 graph",
        "new_skills_mastered": {prof: {pol: round(float(mu), 1) for pol, (mu, sd) in by_pol.items()}
                                for prof, by_pol in res.items()},
    }

def main(argv=None):
    import argparse, graph
    ap = argparse.ArgumentParser(description="Compare the Kwan policy with syllabus order and random practice "
                                             "on simulated learners.")
    ap.add_argument("graph", nargs="?", default=str(graph.DEFAULT_PATH),
                    help="curriculum graph JSON (default: the committed data/graph.json)")
    ap.add_argument("--out", metavar="PATH",
                    help="also write the 'simulation' block to this results file, e.g. data/eval_results.json")
    args = ap.parse_args(argv)
    res = compare_all(graph.load(args.graph))
    for prof, by_pol in res.items():
        for pol, (mu, sd) in by_pol.items():
            print(f"{prof:12s} {pol:9s} new skills mastered in {STEPS} steps: {mu:.1f} ± {sd:.1f}")
    if args.out:
        import results
        results.update("simulation", results_block(res), path=args.out)
    return res

if __name__ == "__main__":
    main()
