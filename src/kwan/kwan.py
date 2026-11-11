"""The public interface of Kwan: `Kwan`, `LearnerState` and `Recommendation` (PRD 9.1).

    from kwan import Kwan

    engine = Kwan.from_bundle()                       # the bundle shipped with the package
    state = engine.new_state("pupil-0042")            # any opaque id, never a name
    engine.update(state, "B1.1.1.1.1", correct=True)  # bare codes and canonical ids both work
    for rec in engine.recommend(state, k=3):
        print(rec.indicator, rec.rationale, rec.ghana_context)

A `Kwan` is read-only after it is built, so one can serve many learners from many threads. The
learner's progress lives in the `LearnerState` the caller keeps (plain data: `to_dict` / `from_dict`).
Every output uses canonical ids (`math:B1.1.1.1.1`). The same bundle, the same answers and the same
`now` always give the same output (PRD 7).

This module needs only the standard library and networkx.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import networkx as nx

from kwan.bundle import Bundle, load_bundle
from kwan.ids import Resolver
from kwan.learner.elo import EloLearner, LearnerState
from kwan.learner.item_bank import ItemBank
from kwan.learner.params import EloParams
from kwan.recommend.recommender import EPS, REVIEW, VARIETY, ContextSource, Recommender

__all__ = ["Kwan", "LearnerState", "Recommendation"]

When = datetime | str  # a timezone-aware datetime or ISO 8601 text; naive values are read as UTC


@dataclass(frozen=True)
class Recommendation:
    """One recommended skill.

    indicator: canonical id, for example `math:B4.1.1.1.2`.
    code: the bare curriculum code, `B4.1.1.1.2`.
    score: the ranking score (higher is better; comparable only within one call).
    mastery: the learner's estimated mastery of the skill, 0 to 1.
    grade: class level 1 to 6 (Basic 1 to Basic 6).
    strand: the curriculum strand name.
    description: the skill text (the NaCCA description in a full-text bundle, else the short label).
    short_label: Kwan's short label for the skill.
    rationale: the reason in the language asked for.
    rationale_parts: the same reason as `[{"key", "params"}]` parts, for other renderings.
    ghana_context: a practice setting from everyday Ghanaian life.
    is_review: True when this is the review slot (a skill learned before and starting to fade).
    """

    indicator: str
    code: str
    score: float
    mastery: float
    grade: int
    strand: str
    description: str
    short_label: str
    rationale: str
    rationale_parts: list[dict[str, Any]]
    ghana_context: str
    is_review: bool

    def to_dict(self) -> dict[str, Any]:
        """A plain, JSON-safe dictionary of the recommendation."""
        return asdict(self)


class Kwan:
    """The engine for one subject: a curriculum graph, a learner model and a recommender.

    Build one with `Kwan.from_bundle()` (the normal way) or `Kwan.from_graph(G)` (tests and
    experiments). Learner ids are opaque strings; indicator ids may be canonical (`math:B1.1.1.1.1`)
    or bare codes (`B1.1.1.1.1`) everywhere they are accepted."""

    def __init__(
        self,
        G: nx.DiGraph,
        params: EloParams | Mapping[str, Any] | None = None,
        contexts: ContextSource | None = None,
        i18n: Mapping[str, Any] | None = None,
        item_bank: ItemBank | None = None,
        bundle_version: str = "",
        subject: str | None = None,
        text_mode: str | None = None,
    ) -> None:
        """`G` must not change after this. See `from_bundle` and `from_graph` for the usual entry points."""
        self._G = G
        self._subject: str = str(subject or G.graph.get("subject") or "math")
        self._text_mode: str = str(text_mode or G.graph.get("text_mode") or "labels")
        self._bundle_version = bundle_version
        self._resolver = Resolver(G, self._subject)
        self._learner = EloLearner.from_params(params, G=G, item_bank=item_bank)
        self._recommender = Recommender(G, self._learner, i18n=i18n, contexts=contexts)
        self._bundle: Bundle | None = None
        self.last_latency_ms: float = 0.0
        """Wall time in milliseconds of the last `recommend` call (engine time only)."""

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_bundle(cls, path: str | os.PathLike[str] | None = None) -> Kwan:
        """Load a bundle file (by default the one shipped with the package) and build the engine from it.

        Raises `FileNotFoundError` when the file is missing and `kwan.bundle.BundleError` when the
        file is not a bundle this version reads."""
        from kwan.context.settings import ContextStore

        bundle = load_bundle(path)
        engine = cls(
            bundle.load_graph(),
            params=bundle.elo_params(),
            contexts=ContextStore.from_bundle(bundle.to_dict()),
            i18n=bundle.i18n,
            item_bank=bundle.load_item_bank(),
            bundle_version=bundle.bundle_version,
            subject=bundle.subject,
            text_mode=bundle.text_mode,
        )
        engine._bundle = bundle
        return engine

    @classmethod
    def from_graph(
        cls,
        G: nx.DiGraph,
        params: EloParams | Mapping[str, Any] | None = None,
        contexts: ContextSource | None = None,
        i18n: Mapping[str, Any] | None = None,
        bundle_version: str = "",
    ) -> Kwan:
        """Build the engine straight from a graph (any Kwan graph: v2 canonical ids or v1 bare codes).

        With `contexts=None` the practice settings come from the built-in strand bank."""
        return cls(G, params=params, contexts=contexts, i18n=i18n, bundle_version=bundle_version)

    # ------------------------------------------------------------------ what it holds

    @property
    def graph(self) -> nx.DiGraph:
        """The curriculum graph. Read it; do not change it."""
        return self._G

    @property
    def subject(self) -> str:
        """The subject, for example `math`."""
        return self._subject

    @property
    def bundle_version(self) -> str:
        """The version of the bundle the engine was built from (empty when built from a bare graph)."""
        return self._bundle_version

    @property
    def text_mode(self) -> str:
        """`labels` (short labels only) or `full` (NaCCA descriptions)."""
        return self._text_mode

    @property
    def bundle(self) -> Bundle | None:
        """The loaded bundle, or None when the engine was built from a graph."""
        return self._bundle

    @property
    def params(self) -> EloParams:
        """The learner-model parameters in use."""
        return self._learner.params

    @property
    def learner(self) -> EloLearner:
        """The learner model (the low-level interface; `update` and `mastery_map` cover most needs)."""
        return self._learner

    @property
    def recommender(self) -> Recommender:
        """The recommender (the low-level interface; `recommend` covers most needs)."""
        return self._recommender

    @property
    def indicators(self) -> list[dict[str, Any]]:
        """Every indicator as a record (`id`, `code`, `grade`, `strand`, `strand_name`, `short_label`,
        `description`, ...) in the graph's order."""
        return [{"id": n, **d} for n, d in self._G.nodes(data=True)]

    def __len__(self) -> int:
        return int(self._G.number_of_nodes())

    def __repr__(self) -> str:
        version = self._bundle_version or "no bundle"
        return f"Kwan({self._subject}, {version}, {len(self)} skills, {self._G.number_of_edges()} links)"

    # ------------------------------------------------------------------ ids

    def resolve(self, indicator: str) -> str:
        """The canonical id for any accepted spelling (`B4.1.1.1.2`, `math:B4.1.1.1.2`, an alias).

        Raises `KeyError("unknown indicator '...'")` when the skill is not in this subject."""
        return self._resolver.resolve(indicator)

    def __contains__(self, indicator: object) -> bool:
        return indicator in self._resolver

    # ------------------------------------------------------------------ learners

    def new_state(self, learner_id: str) -> LearnerState:
        """A blank state for a learner Kwan has never seen. The id is any opaque string you choose."""
        if not isinstance(learner_id, str) or not learner_id.strip():
            raise ValueError("learner_id must be a non-blank string")
        return LearnerState(learner_id, subject=self._subject, bundle_version=self._bundle_version)

    def update(
        self,
        state: LearnerState,
        indicator: str,
        correct: bool,
        item: str | None = None,
        at: When | None = None,
    ) -> float:
        """Record one answer and return the probability the model gave it before seeing it.

        `indicator` is the skill practised (bare code or canonical id). `item` is the question's id,
        used when the bundle has its difficulty. `at` is when the answer was given; pass it so that
        forgetting and the review slot can work. The state is changed in place."""
        node = self.resolve(indicator)
        if not state.bundle_version:
            state.bundle_version = self._bundle_version
        if not state.subject:
            state.subject = self._subject
        return self._learner.update(state, node, bool(correct), item, at)

    def mastery(self, state: LearnerState, indicator: str, now: When | None = None) -> float:
        """The learner's estimated mastery of one skill, 0 to 1 (with forgetting when `now` is given)."""
        return self._learner.mastery(state, self.resolve(indicator), now=now)

    def mastery_map(self, state: LearnerState, now: When | None = None) -> dict[str, float]:
        """Estimated mastery of every skill, `{canonical id: 0 to 1}`, in the graph's order.

        With `now`, skills practised some time ago have faded toward their prior."""
        return self._learner.effective_mastery_map(state, now)

    def recommend(
        self,
        state: LearnerState,
        k: int = 3,
        goal: str | None = None,
        grade_max: int | None = None,
        goal_mode: str = "strict",
        now: When | None = None,
        eps: float = EPS,
        lang: str = "en",
        include_drafts: bool = False,
        *,
        variety: float = VARIETY,
        review: bool = REVIEW,
    ) -> list[Recommendation]:
        """The next skills for a learner, best first.

        k: how many at most (fewer come back when fewer skills are ready).
        goal: a skill to work towards (bare code or canonical id), or None.
        grade_max: class cap, nothing above this grade (1 to 6), or None.
        goal_mode: `"strict"` (skills on the path to the goal come first) or `"soft"` (the goal is one
            signal among several, the v1.0 behaviour).
        now: the time of the request; needed for forgetting and the review slot.
        eps: exploration strength, 0 for none. Seeded by learner id and step, so it is reproducible.
        lang: language of the reason and the practice setting (falls back to English).
        include_drafts: allow practice settings no teacher has reviewed yet.
        variety, review: the recommender's own knobs (see `kwan.recommend.recommender`).

        Raises `KeyError` for an unknown goal and `ValueError` for bad arguments."""
        goal_id = self.resolve(goal) if goal else None
        t0 = time.perf_counter()
        out = self._recommender.recommend(
            state,
            k=k,
            goal=goal_id,
            grade_max=grade_max,
            goal_mode=goal_mode,
            now=now,
            eps=eps,
            variety=variety,
            review=review,
            lang=lang,
            include_drafts=include_drafts,
        )
        self.last_latency_ms = round((time.perf_counter() - t0) * 1000, 3)
        return [self._recommendation(r) for r in out["recommendations"]]

    def _recommendation(self, r: Mapping[str, Any]) -> Recommendation:
        node = str(r["indicator"])
        code = str(self._G.nodes[node].get("code") or node.rpartition(":")[2])
        return Recommendation(
            indicator=node,
            code=code,
            score=float(r["score"]),
            mastery=float(r["mastery"]),
            grade=int(r["grade"]),
            strand=str(r["strand"]),
            description=str(r["description"]),
            short_label=str(r["short_label"]),
            rationale=str(r["rationale"]),
            rationale_parts=[dict(p) for p in r["rationale_parts"]],
            ghana_context=str(r["ghana_context"]),
            is_review=bool(r["is_review"]),
        )
