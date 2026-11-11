"""ZPD frontier recommender: next skills for a learner, each with a reason and a Ghana-context setting.

How the parts of a recommendation fit together (the order never depends on chance or on the order of
the graph's nodes):

1. Candidates. A skill is a candidate when it is not mastered, is within the class cap, and every
   prerequisite is ready. Each candidate gets the v1.0 score (readiness, zone of proximal
   development, goal, unlock, recency). This `score` is what the response reports, always.
2. Review slot. With `review=True`, `k >= 3` and a `now`, the skill that is due for review (see
   `kwan.recommend.review`) takes the last of the k places. It is taken out of the candidates, so it
   never shows twice. The other `k - 1` places are filled as usual, so the review slot can never push
   out the best pick or the only skill on the path to the goal.
3. Exploration. With `eps > 0` each remaining candidate, taken in ascending id order, gets
   `eps * N(0, 1) / sqrt(1 + answers on that skill)` added to its ranking score. The generator is
   seeded by `crc32("<learner_id>:<step>")`, so the same learner at the same step gets the same picks.
4. Strict goal. With a goal and `goal_mode="strict"`, every candidate on the path to the goal ranks
   above every candidate off it. Exploration and variety only reorder skills inside each of the two
   groups.
5. Variety. Picks are made one at a time, best first. After each pick, the other candidates from the
   same content standard lose `variety` (it adds up). Equal ranking scores go to the higher id.
6. The review pick, when there is one, is added last.

v1 equivalence: with `goal_mode="soft"`, `eps=0`, `variety=0`, `review=False` and no `now`, the picks,
scores and wording are exactly those of v1.1.
"""

from __future__ import annotations

import math
import operator
import time
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Protocol

import networkx as nx

from kwan.context.bank import practice_setting
from kwan.learner.elo import MASTERY, READY, LearnerState
from kwan.recommend import explain
from kwan.recommend.explore import Candidate, _rng, content_standard_of, explore, pick_with_variety
from kwan.recommend.review import days_between, review_candidate

GOAL_MODES = ("strict", "soft")
EPS, VARIETY, REVIEW = 0.0, 0.05, True  # library defaults (the demo turns exploration on at 0.05)
REVIEW_MIN_K = 3  # the review slot exists only when at least this many skills are asked for


class Learner(Protocol):
    """What the recommender needs from a learner model: the mastery of one skill, 0 to 1.

    A model that knows about forgetting also takes `now=` as a keyword; it is passed only when the
    caller gives a `now`."""

    def mastery(self, st: LearnerState, ind: str, /) -> float: ...


class ContextSource(Protocol):
    """Where practice settings come from (`kwan.context.settings.ContextStore` is one)."""

    def pick(
        self, indicator_id: str, strand: int, grade: int, lang: str = "en", include_drafts: bool = False
    ) -> str: ...


def _pct_int(x: float) -> int:
    """Whole-number percent, rounding half up. Portable across languages (D8)."""
    return math.floor(x * 100 + 0.5)


def _pct(x: float) -> str:
    """Whole-number percent as text, rounding half up. Portable across languages (D8)."""
    return f"{_pct_int(x)}%"


def _round(x: float, d: int) -> float:
    """Round half up to d decimals. Portable across languages (D8)."""
    return float(math.floor(x * 10**d + 0.5) / 10**d)


def _bare(node_id: str, attrs: Mapping[str, Any]) -> str:
    """The bare curriculum code of a node: its `code` attribute, else the id without `<subject>:`."""
    code = attrs.get("code")
    return str(code) if code else node_id.rpartition(":")[2]


class Recommender:
    W: dict[str, float] = dict(readiness=0.30, zpd=0.25, goal=0.25, unlock=0.15, recency=0.05)

    def __init__(
        self,
        G: nx.DiGraph,
        learner: Learner,
        i18n: Mapping[str, Any] | None = None,
        contexts: ContextSource | None = None,
    ) -> None:
        """G: the curriculum graph (bare or namespaced ids). It must not change afterwards.
        learner: the learner model.
        i18n: rationale templates, either `{lang: {key: template}}` (a bundle's `i18n` block) or one
            flat `{key: template}` table taken as English. Default: the built-in English templates.
        contexts: a source of practice settings. Default: the v1 strand bank.
        """
        self.G, self.L = G, learner
        self.contexts = contexts
        self.templates: dict[str, dict[str, str]] = explain.resolve_templates(i18n)
        self.desc_count: dict[str, int] = {n: len(nx.descendants(G, n)) for n in G}
        self.max_desc: int = max(self.desc_count.values(), default=0) or 1
        self._preds: dict[str, list[str]] = {n: list(G.predecessors(n)) for n in G}
        self._unlock: dict[str, float] = {n: self.desc_count[n] / self.max_desc for n in G}
        self._code: dict[str, str] = {n: _bare(n, d) for n, d in G.nodes(data=True)}
        self._group: dict[str, str] = {
            n: str(d.get("content_standard") or content_standard_of(self._code[n])) for n, d in G.nodes(data=True)
        }

    def _goal_dist(self, goal: str | None) -> dict[str, int]:
        if not goal:
            return {}
        if goal not in self.G:
            raise KeyError(f"unknown goal {goal!r}: not in the curriculum graph")
        rev = self.G.reverse(copy=False)
        dist: dict[str, int] = nx.single_source_shortest_path_length(rev, goal)  # ancestors of goal -> hops
        return dist

    def recommend(
        self,
        st: LearnerState,
        k: int = 3,
        goal: str | None = None,
        grade_max: int | None = None,
        goal_mode: str = "strict",
        now: datetime | str | None = None,
        eps: float = EPS,
        variety: float = VARIETY,
        review: bool = REVIEW,
        lang: str = "en",
        include_drafts: bool = False,
    ) -> dict[str, Any]:
        """Top-k next skills for a learner, best first.

        k: how many to return at most (0 gives an empty list, a negative k raises ValueError).
        goal: an indicator id to steer towards, or None/"" for no goal (an unknown id raises KeyError).
        grade_max: class cap, nothing above this grade is returned (None for no cap).
        goal_mode: "strict" (default) ranks every skill on the path to the goal above every skill off it;
            "soft" keeps the v1.0 ordering, where the goal is one term of the score. Anything else raises
            ValueError. Without a goal both modes give the same result.
        now: the time of the request (a timezone-aware datetime or an ISO 8601 string; a naive time is
            taken as UTC). When given, it is passed to the learner model's `mastery`, so forgetting
            counts, and the review slot can be used.
        eps: exploration strength, 0 for none (the default).
        variety: what a candidate loses for each skill already picked from its content standard.
        review: allow the review slot (needs `k >= 3`, a `now`, and answers with timestamps).
        lang: language of `rationale` and of the practice setting; unknown languages fall back to English.
        include_drafts: let the practice-setting source use settings no teacher has reviewed yet.

        Each recommendation has `indicator`, `score` (the formula score, before exploration and variety),
        `mastery`, `grade`, `strand`, `description`, `short_label`, `rationale`, `rationale_parts`,
        `ghana_context` and `is_review`. The module docstring says how the options combine. The same
        state and the same arguments always give the same result.
        """
        t0 = time.perf_counter()
        try:
            k = operator.index(k)
        except TypeError:
            raise TypeError(f"k must be an integer, got {k!r}") from None
        if k < 0:
            raise ValueError(f"k must be 0 or more, got {k}")
        if goal_mode not in GOAL_MODES:
            raise ValueError(f"unknown goal_mode {goal_mode!r}: choose from {', '.join(GOAL_MODES)}")
        if not eps >= 0:
            raise ValueError(f"eps must be 0 or more, got {eps!r}")
        if not variety >= 0:
            raise ValueError(f"variety must be 0 or more, got {variety!r}")
        G = self.G
        m = self._mastery_map(st, now)
        gd = self._goal_dist(goal)
        strict = bool(gd) and goal_mode == "strict"
        grade_of: Any = G.nodes

        review_id: str | None = None
        if review and k >= REVIEW_MIN_K and now is not None and st.last_seen_at:
            review_id = review_candidate(
                st,
                G,
                m,
                lambda n: float(self.L.mastery(st, n)),  # forgetting ignored: was it mastered?
                now,
                allowed=None if grade_max is None else (lambda n: bool(grade_of[n]["grade"] <= grade_max)),
            )

        scores: dict[str, float] = {}
        for n, d in G.nodes(data=True):
            if n == review_id or m[n] >= MASTERY or (grade_max is not None and d["grade"] > grade_max):
                continue
            preds = self._preds[n]
            if preds and min(m[p] for p in preds) < READY:
                continue
            scores[n] = self._score(st, n, m, gd)

        if eps:
            rng = _rng(st.learner_id, st.step)
            ranks = {n: explore(scores[n], st.n.get(n, 0), rng, eps) for n in sorted(scores)}
        else:
            ranks = scores
        pool = [Candidate(int(strict and n in gd), ranks[n], n, self._group[n]) for n in scores]
        picks = pick_with_variety(pool, k - 1 if review_id is not None else k, variety)

        templates = self.templates.get(lang) or self.templates[explain.DEFAULT_LANG]
        out = [
            self._item(c.id, scores[c.id], m, self._why_parts(c.id, m, gd), False, templates, lang, include_drafts)
            for c in picks
        ]
        if review_id is not None:
            days = int(days_between(st.last_seen_at[review_id], now)) if now is not None else 0
            parts = [explain.part("review", pct=_pct_int(m[review_id]), days=days)]
            score = self._score(st, review_id, m, gd)
            out.append(self._item(review_id, score, m, parts, True, templates, lang, include_drafts))
        return {"recommendations": out, "latency_ms": round((time.perf_counter() - t0) * 1000, 3)}

    def _score(self, st: LearnerState, n: str, m: Mapping[str, float], gd: Mapping[str, int]) -> float:
        """The v1.0 score of one skill: readiness, zone of proximal development, goal, unlock, recency."""
        preds = self._preds[n]
        ready = min(m[p] for p in preds) if preds else 1.0
        zpd = 1 - abs(m[n] - 0.6) / 0.6  # peak when ~60% likely correct
        goal_term = 1 / (1 + gd[n]) if n in gd else (0.0 if gd else 0.5)
        rec = 1 - math.exp(-(st.step - st.last_seen.get(n, -99)) / 5)
        w = self.W
        return (
            w["readiness"] * ready
            + w["zpd"] * zpd
            + w["goal"] * goal_term
            + w["unlock"] * self._unlock[n]
            + w["recency"] * rec
        )

    def _item(
        self,
        n: str,
        score: float,
        m: Mapping[str, float],
        parts: list[dict[str, Any]],
        is_review: bool,
        templates: Mapping[str, str],
        lang: str,
        include_drafts: bool,
    ) -> dict[str, Any]:
        """One recommendation as it appears in the response."""
        d = self.G.nodes[n]
        if self.contexts is None:
            setting = practice_setting(self._code[n], d["strand"], d["grade"])
        else:
            setting = self.contexts.pick(n, d["strand"], d["grade"], lang=lang, include_drafts=include_drafts)
        return {
            "indicator": n,
            "score": _round(score, 4),
            "mastery": _round(m[n], 3),
            "grade": d["grade"],
            "strand": d["strand_name"],
            "description": d["description"],
            "short_label": d.get("short_label") or d["description"],
            "rationale": explain.render(parts, templates),
            "rationale_parts": parts,
            "ghana_context": setting,
            "is_review": is_review,
        }

    def _mastery_map(self, st: LearnerState, now: datetime | str | None) -> dict[str, float]:
        """Mastery of every skill. `now` reaches the learner model only when it is given, so a learner
        model with the v1 signature `mastery(st, ind)` keeps working.

        A learner model built on this same graph may offer `effective_mastery_map(st, now)`, one pass
        that returns the same numbers as `mastery` for every skill. It is used when present."""
        fast: Any = getattr(self.L, "effective_mastery_map", None)
        if fast is not None and getattr(self.L, "G", None) is self.G:
            return dict(fast(st, now))
        if now is None:
            return {n: self.L.mastery(st, n) for n in self.G}
        mastery_at: Any = self.L.mastery
        return {n: float(mastery_at(st, n, now=now)) for n in self.G}

    def _why_parts(self, n: str, m: Mapping[str, float], gd: Mapping[str, int]) -> list[dict[str, Any]]:
        """The reason as parts, built only from the signals that produced the score."""
        parts: list[dict[str, Any]] = []
        preds = self._preds[n]
        if preds:
            best = max(preds, key=lambda p: m[p])
            parts.append(
                explain.part("builds_on", prerequisite=self._code[best], prerequisite_id=best, pct=_pct_int(m[best]))
            )
        else:
            parts.append(explain.part("starting_skill"))
        mn = m[n]
        band = "mastery_new" if mn < 0.5 else ("mastery_close" if mn < 0.75 else "mastery_nearly")
        parts.append(explain.part(band, pct=_pct_int(mn)))
        if n in gd and gd[n] > 0:
            parts.append(explain.part("goal_steps", count=gd[n]))
        if self.desc_count[n]:
            parts.append(explain.part("unlocks", count=self.desc_count[n]))
        return parts
