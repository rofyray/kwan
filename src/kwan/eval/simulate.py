"""Simulated-learner evaluation of the recommendation policy.

Two simulators live here.

v1 (kept unchanged, used by the demo and the published v1.0 table): `PROFILES`, `POLICIES`, `run`,
`compare`, `compare_all`, `results_block`. Each simulated learner has a hidden true state per skill;
practising a skill whose prerequisites are truly known has a high chance to teach it; practising
without them rarely works. Nothing is ever forgotten.

v2 (`PROFILES_V2`, `POLICIES_V2`, `simulate_learner`, `report`): the same idea over simulated days,
with forgetting in the hidden true state, timestamps given to the engine, four pupil profiles and five
policies (plus `kwan-top1`, a sensitivity check on how the pupil uses Kwan's list). It reports new
skills mastered, skills still known after a break, and steps to reach a goal.

Everything in the simulator is an assumption, written down in the docstrings below. No profile or
parameter is fitted to Ghanaian pupils (PRD Section 13 adds survey calibration later). The numbers say
how the policies compare inside this model, not how much real pupils will learn.

Run:  kwan sim [GRAPH] [--v2] [--out data/eval_results.json]

The graph can be any Kwan graph file: schema v2 with namespaced ids or the v1 file with bare ids.
"""

from __future__ import annotations

import math
import random
import statistics
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import networkx as nx

from kwan.learner.elo import MASTERY, READY, EloLearner, LearnerState
from kwan.recommend.recommender import Recommender

# ----------------------------------------------------------------------------- v1 simulator (unchanged)

PROFILES = {
    # probability a skill is already truly mastered at start, by grade band
    "beginner_p1": lambda g: 0.15 if g <= 2 else 0.03,
    "gapped_p4": lambda g: 0.75 if g <= 3 else 0.05,  # typical upper-primary pupil with foundational gaps
}
POLICIES = ("kwan", "syllabus", "random")
N_LEARNERS, STEPS = 30, 150  # the published run: seeds 0..29, 150 practice steps each
V1_MODE = dict(goal_mode="soft", eps=0.0, variety=0.0, review=False)  # the v1 engine, exactly


def run(G, policy, steps=STEPS, seed=0, p_learn=0.35, p_learn_blocked=0.03, slip=0.1, guess=0.2, profile="beginner_p1"):
    """v1 simulator: new skills truly mastered by one seeded learner after `steps` practice steps."""
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}: choose from {', '.join(POLICIES)}")
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}: choose from {', '.join(PROFILES)}")
    rng = random.Random(seed)
    order = list(nx.topological_sort(G))
    syllabus = sorted(G, key=lambda n: (G.nodes[n]["grade"], n))
    true = {n: rng.random() < PROFILES[profile](G.nodes[n]["grade"]) for n in G}
    start = sum(true.values())
    L = EloLearner(G=G)
    R = Recommender(G, L)
    st = LearnerState("sim")
    for _ in range(steps):
        if policy == "kwan":
            recs = R.recommend(st, k=1, **V1_MODE)["recommendations"]
            n = recs[0]["indicator"] if recs else rng.choice(order)
        elif policy == "syllabus":
            n = next((x for x in syllabus if L.mastery(st, x) < MASTERY), syllabus[-1])
        else:
            n = rng.choice(order)
        prereq_ok = all(true[p] for p in G.predecessors(n))
        if not true[n] and rng.random() < (p_learn if prereq_ok else p_learn_blocked):
            true[n] = True
        correct = rng.random() < ((1 - slip) if true[n] else guess)
        L.update(st, n, correct)
    return sum(true.values()) - start  # skills newly mastered


def compare(G, n_learners=N_LEARNERS, steps=STEPS, profile="beginner_p1"):
    """{policy: (mean, sd)} of new skills mastered over learners with seeds 0..n_learners-1 (v1 simulator)."""
    if n_learners < 1:
        raise ValueError(f"n_learners must be 1 or more, got {n_learners}")
    res = {}
    for pol in POLICIES:
        vals = [run(G, pol, steps, seed=s, profile=profile) for s in range(n_learners)]
        res[pol] = (statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0)
    return res


def compare_all(G, n_learners=N_LEARNERS, steps=STEPS):
    """{profile: {policy: (mean, sd)}} for every v1 profile (v1 simulator)."""
    return {prof: compare(G, n_learners, steps, profile=prof) for prof in PROFILES}


def results_block(res, n_learners=N_LEARNERS, steps=STEPS):
    """The v1 fields of the `simulation` block of data/eval_results.json from a `compare_all` result."""
    return {
        "setup": f"{n_learners} simulated learners x {steps} practice steps, NaCCA B1-B6 graph",
        "new_skills_mastered": {
            prof: {pol: round(float(mu), 1) for pol, (mu, sd) in by_pol.items()} for prof, by_pol in res.items()
        },
    }


# ----------------------------------------------------------------------------- v2: profiles (assumptions)

NUMBER, ALGEBRA, GEOMETRY_AND_MEASUREMENT, DATA = 1, 2, 3, 4  # NaCCA strand numbers


def _beginner_p1(grade: int, strand: int) -> float:
    """ASSUMPTION, not data. A pupil starting Basic 1: knows 15% of the B1 and B2 skills (from home and
    kindergarten) and 3% of everything later. Same as the v1.0 profile."""
    return 0.15 if grade <= 2 else 0.03


def _gapped_p4(grade: int, strand: int) -> float:
    """ASSUMPTION, not data. A Basic 4 pupil with foundational gaps: knows 75% of the B1 to B3 skills,
    so one lower-primary skill in four is missing, and 5% of B4 to B6. Same as the v1.0 profile."""
    return 0.75 if grade <= 3 else 0.05


def _gapped_p6(grade: int, strand: int) -> float:
    """ASSUMPTION, not data. A Basic 6 pupil with gaps that grew over the years: knows 80% of B1 to B3,
    50% of B4 and B5, and 5% of B6."""
    if grade <= 3:
        return 0.80
    return 0.50 if grade <= 5 else 0.05


def _uneven_strands(grade: int, strand: int) -> float:
    """ASSUMPTION, not data. A Basic 4 pupil who is strong in Number and weak in Geometry and
    Measurement: of the B1 to B4 skills, knows 85% in Number, 15% in Geometry and Measurement and 50% in
    Algebra and Data; knows 5% of B5 and B6 in every strand."""
    if grade > 4:
        return 0.05
    if strand == NUMBER:
        return 0.85
    return 0.15 if strand == GEOMETRY_AND_MEASUREMENT else 0.50


@dataclass(frozen=True)
class Profile:
    """A kind of simulated pupil. Every field is an assumption made for the simulation.

    known: probability that a skill is truly known at the start, from its grade and strand number.
    class_level: the class the pupil sits in. The `remedial-first` policy teaches this class's syllabus.
    goal_grade, goal_strand: goal runs draw the goal from the skills of this grade (and strand, when
        set) that the pupil does not know at the start.
    """

    known: Callable[[int, int], float]
    class_level: int
    goal_grade: int
    goal_strand: int | None = None

    @property
    def about(self) -> str:
        """The assumption in words (the docstring of `known`)."""
        return " ".join((self.known.__doc__ or "").split())


PROFILES_V2: dict[str, Profile] = {
    "beginner_p1": Profile(_beginner_p1, class_level=1, goal_grade=2),
    "gapped_p4": Profile(_gapped_p4, class_level=4, goal_grade=4),
    "gapped_p6": Profile(_gapped_p6, class_level=6, goal_grade=6),
    "uneven_strands": Profile(_uneven_strands, class_level=4, goal_grade=4, goal_strand=GEOMETRY_AND_MEASUREMENT),
}

# ----------------------------------------------------------------------------- v2: policies

POLICIES_V2: dict[str, str] = {
    "kwan-strict": (
        "Kwan with the library defaults (strict goal mode, no exploration, variety 0.05, review slot on). "
        "The pupil gets a list of 3 skills, practises each once in the order given, then gets a new list."
    ),
    "kwan-explore": "The same as kwan-strict with exploration switched on (eps 0.05, as in the demo).",
    "kwan-top1": (
        "Kwan used as in the v1.0 simulator, added as a sensitivity check: the pupil practises only the top "
        "pick and asks again after every answer. With a list of 1 there is no review slot."
    ),
    "syllabus": (
        "Syllabus order from Basic 1: always the first skill, by class and then by code, that the learner "
        "model does not count as mastered. It does not use the prerequisite map."
    ),
    "random": "A skill drawn at random from the whole curriculum at every step.",
    "remedial-first": (
        "The pupil's class syllabus with remediation first: take the first skill of the pupil's class (then "
        "later classes, then earlier ones) that is not mastered; if any skill that leads to it is below 70%, "
        "practise the earliest such skill, otherwise practise the skill itself. It uses the prerequisite map "
        "but none of Kwan's scoring. With a goal, the goal is the skill it works towards."
    ),
}
EXPLORE_EPS = 0.05  # what the demo uses
SESSION = 3  # skills per Kwan list
KWAN_POLICIES = ("kwan-strict", "kwan-explore", "kwan-top1")

# ----------------------------------------------------------------------------- v2: the hidden learner (assumptions)

N_LEARNERS_V2 = 30  # seeds 0..29 per profile and policy
STEPS_PER_DAY = 5  # one short practice session a day
RETENTION_DAYS = 30  # the break after the last practice step, with no practice at all
GOAL_MAX_STEPS = 300  # a goal run stops here when the goal was not reached
P_LEARN, P_LEARN_BLOCKED, SLIP, GUESS = 0.35, 0.03, 0.1, 0.2  # the v1.0 values
STABILITY_NEW = 45.0  # days: memory strength of a skill that was just learned
STABILITY_MAX = 365.0  # days: the strongest a memory gets; also the strength of what was known at the start
STABILITY_GROWTH = 2.0  # practising a known skill again on a later day multiplies its strength by this
START = datetime(2026, 1, 5, 8, 0, tzinfo=UTC)  # day 0 of every simulated learner (any fixed date works)

ASSUMPTIONS: dict[str, Any] = {
    "p_learn": P_LEARN,
    "p_learn_blocked": P_LEARN_BLOCKED,
    "slip": SLIP,
    "guess": GUESS,
    "steps_per_day": STEPS_PER_DAY,
    "stability_new_days": STABILITY_NEW,
    "stability_max_days": STABILITY_MAX,
    "stability_growth": STABILITY_GROWTH,
    "text": (
        "Each simulated pupil has a hidden true state for every skill. One practice step on a skill the pupil "
        "does not know teaches it with probability 0.35 when every prerequisite is truly known and 0.03 "
        "otherwise. A known skill is answered wrongly 10% of the time and an unknown one is guessed right 20% "
        "of the time. The pupil does 5 steps a day. Each night every known skill can be forgotten: a skill "
        "with memory strength S days survives one day with probability exp(-1/S). S is 45 days for a skill "
        "just learned and 365 days for what the pupil knew at the start; practising a known skill again on a "
        "later day doubles S, up to 365 days. Practising a skill does not refresh its prerequisites. These "
        "are modelling assumptions, not measurements of Ghanaian pupils."
    ),
}


def _syllabus_key(G: nx.DiGraph) -> Callable[[str], tuple[Any, ...]]:
    """Sort key for syllabus order: class first, then the code in natural order (B4.1.2 before B4.1.10)."""
    from kwan.graph.schema import sort_key

    return lambda n: (G.nodes[n]["grade"], sort_key(n))


class World:
    """What every simulated learner on one graph shares: the engine and the orderings. Build it once."""

    def __init__(self, G: nx.DiGraph) -> None:
        self.G = G
        self.L = EloLearner(G=G)
        self.R = Recommender(G, self.L)
        key = _syllabus_key(G)
        self.syllabus: list[str] = sorted(G, key=key)
        self.rank: dict[str, int] = {n: i for i, n in enumerate(self.syllabus)}
        self.nodes: list[str] = sorted(G)
        self.preds: dict[str, list[str]] = {n: list(G.predecessors(n)) for n in G}
        self.grade: dict[str, int] = {n: G.nodes[n]["grade"] for n in G}
        self.strand: dict[str, int] = {n: G.nodes[n]["strand"] for n in G}
        self._ancestors: dict[str, list[str]] = {}
        self._targets: dict[int, list[str]] = {}

    def ancestors(self, n: str) -> list[str]:
        """Every skill that leads to `n`, in syllabus order."""
        if n not in self._ancestors:
            self._ancestors[n] = sorted(nx.ancestors(self.G, n), key=self.rank.__getitem__)
        return self._ancestors[n]

    def targets(self, class_level: int) -> list[str]:
        """Teaching order of `remedial-first`: the pupil's class and later classes, then earlier ones."""
        if class_level not in self._targets:
            later = [n for n in self.syllabus if self.grade[n] >= class_level]
            self._targets[class_level] = later + [n for n in self.syllabus if self.grade[n] < class_level]
        return self._targets[class_level]


def _stream(*parts: object) -> random.Random:
    """A seeded random stream named by its parts, the same on every run and every machine."""
    return random.Random(zlib.crc32(":".join(str(p) for p in parts).encode()))


def start_state(world: World, profile: str, seed: int) -> tuple[dict[str, bool], str | None]:
    """The hidden knowledge of one learner at the start, and the goal used in goal runs.

    Both depend only on the profile and the seed, so every policy meets the same pupil with the same goal.
    The goal is drawn from the skills of the profile's goal grade (and strand) that the pupil does not
    know and that have at least one prerequisite; None when there is no such skill."""
    prof = PROFILES_V2[profile]
    rng = _stream(profile, seed, "start")
    known = {n: rng.random() < prof.known(world.grade[n], world.strand[n]) for n in world.nodes}
    pool = [
        n
        for n in world.nodes
        if world.grade[n] == prof.goal_grade
        and (prof.goal_strand is None or world.strand[n] == prof.goal_strand)
        and not known[n]
        and world.preds[n]
    ]
    return known, (rng.choice(pool) if pool else None)


def simulate_learner(
    G: nx.DiGraph,
    policy: str,
    profile: str,
    seed: int = 0,
    steps: int = STEPS,
    with_goal: bool = False,
    world: World | None = None,
    p_learn: float = P_LEARN,
    p_learn_blocked: float = P_LEARN_BLOCKED,
    slip: float = SLIP,
    guess: float = GUESS,
    steps_per_day: int = STEPS_PER_DAY,
    retention_days: int = RETENTION_DAYS,
    session: int = SESSION,
) -> dict[str, Any]:
    """One seeded learner under one policy (v2 simulator, see `ASSUMPTIONS`).

    Without a goal the learner practises for `steps` steps, then rests for `retention_days` days.
    With `with_goal=True` the Kwan and `remedial-first` policies are told the learner's goal and the run
    stops as soon as the goal is truly known, or after `steps` steps.

    Returns `new` (skills not known at the start and known after the last step), `retained` (those of
    them still known after the rest), `start` (skills known at the start), `goal`, `reached`,
    `steps_to_goal` (the number of steps taken when the goal became known, else None), `steps` (steps
    taken) and `review_steps` (steps spent on a review-slot pick).
    """
    if policy not in POLICIES_V2:
        raise ValueError(f"unknown policy {policy!r}: choose from {', '.join(POLICIES_V2)}")
    if profile not in PROFILES_V2:
        raise ValueError(f"unknown profile {profile!r}: choose from {', '.join(PROFILES_V2)}")
    if steps_per_day < 1 or session < 1:
        raise ValueError("steps_per_day and session must be 1 or more")
    w = world if world is not None and world.G is G else World(G)
    L, R = w.L, w.R
    known, goal = start_state(w, profile, seed)
    if not with_goal:
        goal = None
    at_start = dict(known)
    rng = _stream(profile, seed, policy, "run")
    stability = {n: STABILITY_MAX for n in w.nodes if known[n]}
    last_day: dict[str, int] = {}
    st = LearnerState(f"{profile}:{seed}")
    class_level = PROFILES_V2[profile].class_level
    eps = EXPLORE_EPS if policy == "kwan-explore" else 0.0
    list_size = 1 if policy == "kwan-top1" else session
    queue: list[tuple[str, bool]] = []
    review_steps = 0

    def forget_overnight() -> None:
        for n in w.nodes:
            if known[n] and rng.random() >= math.exp(-1.0 / stability[n]):
                known[n] = False
                del stability[n]

    def choose(now: datetime) -> str:
        nonlocal review_steps
        if policy in KWAN_POLICIES:
            if not queue:
                recs = R.recommend(st, k=list_size, goal=goal, now=now, eps=eps)["recommendations"]
                queue.extend((r["indicator"], r["is_review"]) for r in recs)
            if not queue:
                return rng.choice(w.nodes)
            pick, is_review = queue.pop(0)
            review_steps += is_review
            return pick
        if policy == "random":
            return rng.choice(w.nodes)
        m = L.effective_mastery_map(st, now) if hasattr(L, "effective_mastery_map") else None

        def mastery(n: str) -> float:
            return m[n] if m is not None else L.mastery(st, n, now=now)

        if policy == "syllabus":
            return next((n for n in w.syllabus if mastery(n) < MASTERY), w.syllabus[-1])
        order = w.targets(class_level)
        target = goal if goal is not None and mastery(goal) < MASTERY else None
        if target is None:
            target = next((n for n in order if mastery(n) < MASTERY), order[-1])
        return next((a for a in w.ancestors(target) if mastery(a) < READY), target)

    day, taken, steps_to_goal = 0, 0, None
    for i in range(steps):
        today = i // steps_per_day
        while day < today:  # a night passes
            forget_overnight()
            day += 1
        now = START + timedelta(days=day, minutes=5 * (i % steps_per_day))
        n = choose(now)
        if known[n]:
            if last_day.get(n, day) < day:
                stability[n] = min(STABILITY_MAX, stability[n] * STABILITY_GROWTH)
        else:
            ready = all(known[p] for p in w.preds[n])
            if rng.random() < (p_learn if ready else p_learn_blocked):
                known[n] = True
                stability[n] = STABILITY_NEW
        last_day[n] = day
        correct = rng.random() < ((1 - slip) if known[n] else guess)
        L.update(st, n, correct, at=now)
        taken = i + 1
        if goal is not None and known[goal]:
            steps_to_goal = taken
            break
    new = [n for n in w.nodes if known[n] and not at_start[n]]
    if goal is None:
        for _ in range(retention_days):
            forget_overnight()
    return {
        "new": len(new),
        "retained": sum(known[n] for n in new),
        "start": sum(at_start.values()),
        "goal": goal,
        "reached": steps_to_goal is not None,
        "steps_to_goal": steps_to_goal,
        "steps": taken,
        "review_steps": review_steps,
    }


def _mean_sd(vals: list[float]) -> dict[str, float]:
    return {
        "mean": round(float(statistics.mean(vals)), 1),
        "sd": round(float(statistics.stdev(vals)) if len(vals) > 1 else 0.0, 1),
    }


def acceptance(new: dict[str, dict[str, Any]], kept: dict[str, dict[str, Any]], margin: float = 1.0) -> dict[str, Any]:
    """The PRD Phase 6 acceptance check, worked out from the reported means.

    1. For every gapped profile, kwan-strict is above syllabus and above random on new skills mastered
       and on skills retained.
    2. On every profile, kwan-strict is at most `margin` skills below syllabus on both measures.
    Returns both verdicts and the list of comparisons that failed, in words."""
    failed: list[str] = []
    beats = within = True
    for name, table in (("new_skills_mastered", new), ("skills_retained", kept)):
        for prof, by_pol in table.items():
            kwan = by_pol["kwan-strict"]["mean"]
            if prof.startswith("gapped"):
                for other in ("syllabus", "random"):
                    if not kwan > by_pol[other]["mean"]:
                        beats = False
                        failed.append(
                            f"{name}, {prof}: kwan-strict {kwan} is not above {other} {by_pol[other]['mean']}"
                        )
            if kwan < by_pol["syllabus"]["mean"] - margin:
                within = False
                failed.append(
                    f"{name}, {prof}: kwan-strict {kwan} is more than {margin:g} below syllabus "
                    f"{by_pol['syllabus']['mean']}"
                )
    return {
        "kwan_strict_beats_syllabus_and_random_on_gapped_profiles": beats,
        "kwan_strict_within_1_skill_of_syllabus_on_every_profile": within,
        "met": beats and within,
        "failed": failed,
    }


def report(
    G: nx.DiGraph,
    n_learners: int = N_LEARNERS_V2,
    steps: int = STEPS,
    goal_max_steps: int = GOAL_MAX_STEPS,
    retention_days: int = RETENTION_DAYS,
) -> dict[str, Any]:
    """The v2 report: per profile and policy, mean and sd over learners with seeds 0..n_learners-1 of

    - `new_skills_mastered` after `steps` practice steps,
    - `skills_retained`: how many of those are still known `retention_days` days later, with no practice,
    - `steps_to_goal`: steps until the learner's goal is truly known. A run that does not get there
      within `goal_max_steps` counts as `goal_max_steps`; `reached` is the share of runs that got there.
    - `review_steps` (Kwan policies only): how many of the `steps` went to a review-slot pick.
    """
    if n_learners < 1:
        raise ValueError(f"n_learners must be 1 or more, got {n_learners}")
    world = World(G)
    new: dict[str, dict[str, Any]] = {}
    kept: dict[str, dict[str, Any]] = {}
    goal: dict[str, dict[str, Any]] = {}
    reviews: dict[str, dict[str, Any]] = {}
    for prof in PROFILES_V2:
        new[prof], kept[prof], goal[prof], reviews[prof] = {}, {}, {}, {}
        for pol in POLICIES_V2:
            runs = [
                simulate_learner(G, pol, prof, seed=s, steps=steps, world=world, retention_days=retention_days)
                for s in range(n_learners)
            ]
            new[prof][pol] = _mean_sd([r["new"] for r in runs])
            kept[prof][pol] = _mean_sd([r["retained"] for r in runs])
            if pol in KWAN_POLICIES:
                reviews[prof][pol] = _mean_sd([r["review_steps"] for r in runs])
            goal_runs = [
                simulate_learner(G, pol, prof, seed=s, steps=goal_max_steps, with_goal=True, world=world)
                for s in range(n_learners)
            ]
            goal_runs = [r for r in goal_runs if r["goal"] is not None]
            if goal_runs:
                taken = [float(r["steps_to_goal"] if r["reached"] else goal_max_steps) for r in goal_runs]
                goal[prof][pol] = {
                    **_mean_sd(taken),
                    "reached": round(sum(r["reached"] for r in goal_runs) / len(goal_runs), 2),
                }
    ids = "namespaced" if any(":" in str(n) for n in G) else "bare"
    return {
        "setup": (
            f"{n_learners} simulated learners per profile and policy (seeds 0 to {n_learners - 1}), {steps} practice "
            f"steps at {STEPS_PER_DAY} a day, then {retention_days} days without practice; goal runs stop at "
            f"{goal_max_steps} steps"
        ),
        "graph": {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(), "ids": ids},
        "n_learners": n_learners,
        "steps": steps,
        "retention_days": retention_days,
        "goal_max_steps": goal_max_steps,
        "assumptions": dict(ASSUMPTIONS),
        "profiles": {name: prof.about for name, prof in PROFILES_V2.items()},
        "policies": dict(POLICIES_V2),
        "new_skills_mastered": new,
        "skills_retained": kept,
        "steps_to_goal": goal,
        "review_steps": reviews,
        "acceptance": acceptance(new, kept),
    }


def format_report(rep: dict[str, Any]) -> str:
    """The v2 report as a plain text table, one line per profile and policy."""
    lines = [rep["setup"]]
    for prof, by_pol in rep["new_skills_mastered"].items():
        for pol, new in by_pol.items():
            kept = rep["skills_retained"][prof][pol]
            line = (
                f"{prof:15s} {pol:15s} mastered {new['mean']:5.1f} ± {new['sd']:4.1f}   "
                f"retained {kept['mean']:5.1f} ± {kept['sd']:4.1f}"
            )
            g = rep["steps_to_goal"][prof].get(pol)
            if g is not None:
                line += f"   steps to goal {g['mean']:5.1f} ± {g['sd']:5.1f} (reached {g['reached']:.0%})"
            lines.append(line)
    acc = rep["acceptance"]
    lines.append("acceptance (PRD Phase 6): " + ("met" if acc["met"] else "NOT met"))
    lines.extend(f"  {text}" for text in acc["failed"])
    return "\n".join(lines)


def simulation_block(v1_res, v2_report=None, n_learners=N_LEARNERS, steps=STEPS):
    """The `simulation` block of data/eval_results.json.

    The v1 fields (`setup`, `new_skills_mastered`) stay at the top level, where the demo and the README
    read them; the v2 report goes under `v2`."""
    block = results_block(v1_res, n_learners, steps)
    if v2_report is not None:
        block["v2"] = v2_report
    return block


# ----------------------------------------------------------------------------- goal path (strict vs soft)

GOAL_PATH_PUPILS, GOAL_PATH_K = 50, 3


def on_path_share(G: nx.DiGraph, pupils: int = GOAL_PATH_PUPILS, k: int = GOAL_PATH_K) -> dict[str, float]:
    """Share of the top-k picks that sit on the path to the goal, per goal mode, over seeded pupils.

    The protocol of `tests/test_recommender.py::test_strict_goal_on_path_share`: each pupil has up to 60
    answers on B1 to B3 skills (85% right) and a goal drawn from B4 to B6. Both modes are asked about the
    same pupil and the same goal, with every other option at its default."""
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


def goal_path_block(G: nx.DiGraph, pupils: int = GOAL_PATH_PUPILS, k: int = GOAL_PATH_K) -> dict[str, Any]:
    """The `goal_path` block of data/eval_results.json."""
    share = on_path_share(G, pupils, k)
    return {
        "setup": (
            f"{pupils} seeded pupils, each with up to 60 answers on B1 to B3 skills (85% right) and a goal "
            f"drawn from B4 to B6; share of the top {k} recommendations that are on the path to the goal"
        ),
        "pupils": pupils,
        "k": k,
        "on_path_share": {mode: round(v, 3) for mode, v in share.items()},
    }


# ----------------------------------------------------------------------------- command line


def default_graph_path():
    """The graph `kwan sim` uses when none is named: the committed v2 graph `data/graph/math.graph.json`.

    Outside a source checkout (an installed wheel with no `data/` folder) the file is not there and this
    returns None, which makes `load_graph` pick the graph inside the bundle shipped with the package."""
    from kwan.graph import schema

    path = schema.default_graph_path()
    return path if path.is_file() else None


def load_graph(path=None):
    """Load any Kwan graph file: schema v2 (namespaced ids) or v1 (bare ids)."""
    from kwan.graph import schema

    return schema.load_graph(path)


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(
        description=(
            "Compare the Kwan policy with syllabus order and random practice on simulated learners. "
            "Prints the v1.0 table. With --v2 or --out it also runs the v2 simulator (forgetting over "
            "simulated days, four profiles, five policies)."
        )
    )
    ap.add_argument(
        "graph",
        nargs="?",
        help="curriculum graph JSON, schema v2 or v1 (default: the committed graph)",
    )
    ap.add_argument("--v2", action="store_true", help="also run and print the v2 report")
    ap.add_argument(
        "--learners",
        type=int,
        default=N_LEARNERS_V2,
        metavar="N",
        help=f"simulated learners per profile and policy in the v2 report (default: {N_LEARNERS_V2})",
    )
    ap.add_argument(
        "--out",
        metavar="PATH",
        help=(
            "write the 'simulation' block (v1.0 fields, and the v2 report under 'v2') and the 'goal_path' "
            "block to this results file, e.g. data/eval_results.json"
        ),
    )
    args = ap.parse_args(argv)
    if args.learners < 1:
        ap.error("--learners must be 1 or more")
    G = load_graph(args.graph if args.graph is not None else default_graph_path())
    res = compare_all(G)
    for prof, by_pol in res.items():
        for pol, (mu, sd) in by_pol.items():
            print(f"{prof:12s} {pol:9s} new skills mastered in {STEPS} steps: {mu:.1f} ± {sd:.1f}")
    if args.v2 or args.out:
        rep = report(G, n_learners=args.learners)
        goal_path = goal_path_block(G)
        print()
        print(format_report(rep))
        share = goal_path["on_path_share"]
        print(f"on-path share of the top {goal_path['k']}: strict {share['strict']:.3f}, soft {share['soft']:.3f}")
        if args.out:
            from kwan.eval import results

            results.update("simulation", simulation_block(res, rep), path=args.out)
            results.update("goal_path", goal_path, path=args.out)
    return res


if __name__ == "__main__":
    main()
