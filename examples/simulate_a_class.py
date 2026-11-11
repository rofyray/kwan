"""Put a class of 30 simulated pupils through Kwan's public interface.

Run:       python examples/simulate_a_class.py [--pupils 30] [--days 12]

Each pupil starts from one of the simulator's profiles (`kwan.eval.simulate.PROFILES_V2`: a Basic 1
beginner, a Basic 4 pupil with gaps, a Basic 6 pupil with gaps, and a Basic 4 pupil who is strong in
Number but weak in Geometry). Every day the pupil asks Kwan for three skills, practises each once,
and Kwan records the answer with its time. A hidden "true" state says whether the pupil really
knows a skill; practising a skill whose prerequisites are truly known teaches it with probability
0.35, otherwise 0.03 (the simulator's assumptions, not measurements of Ghanaian pupils).

The numbers printed come from this run only; they are not evidence. The evidence lives in
`data/eval_results.json`, written by `kwan sim`.
"""

from __future__ import annotations

import argparse
import random
import statistics
from datetime import UTC, datetime, timedelta

from kwan import Kwan
from kwan.eval.simulate import GUESS, P_LEARN, P_LEARN_BLOCKED, PROFILES_V2, SLIP

START = datetime(2026, 1, 5, 8, 0, tzinfo=UTC)
SKILLS_PER_DAY = 3


def simulate_pupil(engine: Kwan, profile: str, seed: int, days: int) -> dict[str, float]:
    """One pupil for `days` days. Returns what they knew at the start, what they learned, and Kwan's view."""
    rng = random.Random(seed)
    known = PROFILES_V2[profile].known
    G = engine.graph
    truth = {n: rng.random() < known(d["grade"], d["strand"]) for n, d in G.nodes(data=True)}
    start = sum(truth.values())
    state = engine.new_state(f"{profile}-{seed}")
    when = START
    for _ in range(days):
        recs = engine.recommend(state, k=SKILLS_PER_DAY, now=when)
        for rec in recs:
            n = rec.indicator
            if not truth[n]:
                ready = all(truth[p] for p in G.predecessors(n))
                if rng.random() < (P_LEARN if ready else P_LEARN_BLOCKED):
                    truth[n] = True
            correct = rng.random() < ((1 - SLIP) if truth[n] else GUESS)
            engine.update(state, n, correct, at=when)
            when += timedelta(minutes=5)
        when += timedelta(days=1, minutes=-5 * len(recs))
    mastery = engine.mastery_map(state, now=when)
    return {
        "start": start,
        "learned": sum(truth.values()) - start,
        "kwan_mastered": sum(1 for v in mastery.values() if v >= 0.85),
        "answers": state.step,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Run a class of simulated pupils through Kwan.")
    ap.add_argument("--pupils", type=int, default=30, help="pupils in the class (default: 30)")
    ap.add_argument("--days", type=int, default=12, help="school days to simulate (default: 12)")
    args = ap.parse_args(argv)

    engine = Kwan.from_bundle()
    profiles = list(PROFILES_V2)
    rows: dict[str, list[dict[str, float]]] = {p: [] for p in profiles}
    for i in range(args.pupils):
        profile = profiles[i % len(profiles)]
        rows[profile].append(simulate_pupil(engine, profile, seed=i, days=args.days))

    print(f"{args.pupils} pupils, {args.days} days, {SKILLS_PER_DAY} skills a day, bundle {engine.bundle_version}\n")
    print(f"{'profile':16s} {'pupils':>6s} {'known at start':>15s} {'learned':>8s} {'Kwan counts mastered':>21s}")
    for profile, results in rows.items():
        if not results:
            continue
        start, learned, mastered = (
            statistics.mean(r[key] for r in results) for key in ("start", "learned", "kwan_mastered")
        )
        print(f"{profile:16s} {len(results):6d} {start:15.1f} {learned:8.1f} {mastered:21.1f}")
    print("\n'learned' counts skills the hidden model says the pupil now truly knows.")
    print("'Kwan counts mastered' is how many skills Kwan estimates at 85% or more after the last day.")


if __name__ == "__main__":
    main()
