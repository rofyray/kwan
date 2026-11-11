"""Kwan in one minute: load the bundle, record a few answers, ask what to learn next.

Install:   pip install kwan
Run:       python examples/quickstart.py

The learner id is any opaque string you choose. Never use a pupil's name or phone number.
"""

from datetime import UTC, datetime, timedelta

from kwan import Kwan


def main() -> None:
    engine = Kwan.from_bundle()  # the bundle shipped with the package
    print(f"Loaded {engine!r}")

    state = engine.new_state("pupil-0042")

    # A Basic 1 pupil answers four questions over two days. Bare codes and canonical ids both work.
    day_one = datetime(2026, 11, 9, 9, 0, tzinfo=UTC)
    answers = [
        ("B1.1.1.1.1", True, day_one),
        ("B1.1.1.1.1", True, day_one + timedelta(minutes=2)),
        ("math:B1.1.1.1.1", True, day_one + timedelta(minutes=4)),
        ("B1.1.1.1.2", False, day_one + timedelta(days=1)),
    ]
    for indicator, correct, at in answers:
        p = engine.update(state, indicator, correct, at=at)
        word = "right" if correct else "wrong"
        print(f"  {indicator:16s} {word:5s}  (the model expected a right answer with p = {p:.2f})")

    # What should the pupil do next? Give the time so forgetting and the review slot can work.
    now = day_one + timedelta(days=2)
    print("\nNext skills:")
    for rec in engine.recommend(state, k=3, grade_max=2, now=now):
        tag = " (review)" if rec.is_review else ""
        print(f"  {rec.code}: {rec.short_label}{tag}")
        print(f"      mastery {rec.mastery:.0%}, score {rec.score:.3f}")
        print(f"      why: {rec.rationale}")
        print(f"      try: {rec.ghana_context}")
    print(f"\n(recommend took {engine.last_latency_ms:.2f} ms)")

    # The whole picture: estimated mastery of every skill, keyed by canonical id.
    mastery = engine.mastery_map(state, now=now)
    best = sorted(mastery.items(), key=lambda kv: -kv[1])[:3]
    print("\nStrongest skills right now:")
    for indicator, value in best:
        print(f"  {indicator:18s} {value:.0%}")

    # Learner state is plain data: keep it wherever you like and rebuild it with LearnerState.from_dict.
    saved = state.to_dict()
    print(f"\nState has {saved['step']} answers recorded under bundle {saved['bundle_version']}.")


if __name__ == "__main__":
    main()
