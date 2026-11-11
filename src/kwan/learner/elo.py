"""Elo-style learner model: p(correct | learner, indicator, item) = sigmoid(theta + theta_k - beta_item).

Forgetting (PRD 6.2), the exact rules
-------------------------------------
Nothing below applies unless a time is given: without `at` and `now`, every result equals v1.

1. What is stored. `st.theta_k[k]` is the offset of skill `k` as it stood at `st.last_seen_at[k]`, the
   time of its last timed answer. Reading a state never changes it.
2. Effective offset at a time `now`:
       theta_eff(k) = prior(k) + (theta_k[k] - prior(k)) * exp(-days / tau_days)
   with `days` = time from `last_seen_at[k]` to `now`, never below zero. `prior(k)` is the prior the
   model uses for an unseen skill, `-grade_step * (grade - 1) + transfer * mean(prerequisites seen)`
   (or `- 0.5` when no prerequisite was practised), where each practised prerequisite counts with its
   own effective offset at `now`. So borrowed evidence fades together with its source. A skill never
   practised has `theta_eff = prior`. Without a graph the prior is 0.
3. Unknown times. A skill practised without a time (no entry in `last_seen_at`) does not decay. A
   `now` earlier than `last_seen_at[k]` counts as zero days. Zero days gives the stored offset exactly.
4. `update(..., at=t)` first works out the effective offset at `t`, predicts with it, and starts the
   Elo step from it: `theta_k[k] = theta_eff(k at t) + step`. Then `last_seen_at[k] = t`. If `t` is
   earlier than the stored time (answers arriving out of order) the gap is zero, the step is the v1
   step and the later time is kept. A repeated time is a zero gap, so it is the v1 step too.
5. `update(...)` without `at` is the v1 step from the stored offset, and it removes the skill's entry
   in `last_seen_at`: the time of that answer is unknown, so the skill stops decaying until its next
   timed answer.
6. Streak mastery (`streak_to_master` correct in a row gives mastery of at least 0.95) counts only
   while the skill was seen at most `streak_expiry_days` ago. A timed answer that arrives after the
   streak expired starts a new streak.
7. The global ability `theta` and the answer counts do not decay.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from kwan.learner.forgetting import DAY, STREAK_EXPIRY_DAYS, TAU_DAYS, _text_seconds, to_iso, to_seconds
from kwan.learner.item_bank import ItemBank, beta_for_label
from kwan.learner.params import EloParams, positive_days

MASTERY = 0.85  # p(correct) at/above which an indicator counts as mastered
READY = 0.70  # min prerequisite mastery for a node to be "ready"
STREAK_MASTERY = 0.95  # mastery is at least this after `streak_to_master` correct answers in a row
SCHEMA_VERSION = 2  # learner state layout written by `LearnerState.to_dict`


def sigmoid(z: float) -> float:
    try:
        return 1 / (1 + math.exp(-z))
    except OverflowError:  # z below about -709: the probability is 0 to machine precision
        return 0.0


def _iso(at: datetime | str) -> str:
    """ISO 8601 text for a timestamp given as a datetime or as an ISO 8601 string."""
    return to_iso(at)


@dataclass
class LearnerState:
    learner_id: str
    theta: float = 0.0  # global ability
    theta_k: dict[str, float] = field(default_factory=dict)  # per-indicator ability offset
    n: dict[str, int] = field(default_factory=dict)  # attempts per indicator (+ "_all")
    last_seen: dict[str, int] = field(default_factory=dict)  # indicator -> step index
    streak: dict[str, int] = field(default_factory=dict)  # consecutive correct answers per indicator
    step: int = 0
    subject: str = "math"
    bundle_version: str = ""  # version of the bundle this state was built with ("" when not recorded)
    last_seen_at: dict[str, str] = field(default_factory=dict)  # indicator -> ISO 8601 time of the last answer
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict copy of the state. It survives a JSON round trip unchanged, so a platform
        such as Syto can keep it in any store and rebuild it with `from_dict`."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> LearnerState:
        """Rebuild a state written by `to_dict`. Only `learner_id` is required; keys this version
        does not know are ignored. The dictionaries are copied, never shared with `d`.

        Both layouts are read: v2 and v1 (v1 has no `subject`, `bundle_version`, `last_seen_at` or
        `schema_version`). The result is always a v2 state."""
        if "learner_id" not in d:
            raise ValueError("learner state needs a 'learner_id'")
        return cls(
            learner_id=d["learner_id"],
            theta=float(d.get("theta", 0.0)),
            theta_k={k: float(v) for k, v in d.get("theta_k", {}).items()},
            n={k: int(v) for k, v in d.get("n", {}).items()},
            last_seen={k: int(v) for k, v in d.get("last_seen", {}).items()},
            streak={k: int(v) for k, v in d.get("streak", {}).items()},
            step=int(d.get("step", 0)),
            subject=str(d.get("subject", "math")),
            bundle_version=str(d.get("bundle_version", "")),
            last_seen_at={k: str(v) for k, v in (d.get("last_seen_at") or {}).items()},
        )


@dataclass(frozen=True)
class _Layout:
    """Per-graph tables for the all-skills pass, built once per learner model and graph."""

    key: tuple[int, int, float]
    rows: list[tuple[str, float | None, tuple[str, ...]]]  # (skill, grade part of the prior, prerequisites)
    topological: list[tuple[str, float | None, tuple[str, ...]]] | None  # same rows, prerequisites first
    reorder: bool  # the topological rows are in another order than the graph


class EloLearner:
    """p(correct | learner, indicator, item) = sigmoid(theta + theta_k - beta_item).

    Validated on ASSISTments 2009 (AUC 0.75). The module notes give the forgetting rules."""

    def __init__(
        self,
        k_s: float = 0.3,
        k_k: float = 0.4,
        decay: float = 0.05,
        item_difficulty: dict[Any, float] | None = None,
        G: Any = None,
        grade_step: float = 0.4,
        transfer: float = 0.5,
        streak_to_master: int = 3,
        tau_days: float | None = TAU_DAYS,
        streak_expiry_days: float | None = STREAK_EXPIRY_DAYS,
        item_bank: ItemBank | None = None,
    ) -> None:
        self.k_s, self.k_k, self.decay = k_s, k_k, decay
        self.beta: dict[Any, float] = item_difficulty if item_difficulty is not None else {}
        self.item_bank = item_bank
        self.G = G
        self.grade_step, self.transfer, self.streak_to_master = grade_step, transfer, streak_to_master
        self.tau_days = positive_days("tau_days", tau_days)
        self.streak_expiry_days = positive_days("streak_expiry_days", streak_expiry_days)
        self._layout_cache: _Layout | None = None

    # ------------------------------------------------------------------ parameters

    @classmethod
    def from_params(
        cls,
        params: EloParams | Mapping[str, Any] | None = None,
        G: Any = None,
        item_bank: ItemBank | None = None,
        item_difficulty: dict[Any, float] | None = None,
    ) -> EloLearner:
        """A learner model with the given tunables (an `EloParams`, or its dictionary as stored in a
        bundle's `params` block; None gives the defaults)."""
        p = params if isinstance(params, EloParams) else EloParams.from_dict(params)
        return cls(
            k_s=p.k_s,
            k_k=p.k_k,
            decay=p.decay,
            item_difficulty=item_difficulty,
            G=G,
            grade_step=p.grade_step,
            transfer=p.transfer,
            streak_to_master=p.streak_to_master,
            tau_days=p.tau_days,
            streak_expiry_days=p.streak_expiry_days,
            item_bank=item_bank,
        )

    @property
    def params(self) -> EloParams:
        """The tunables this learner model runs with."""
        return EloParams(
            k_s=self.k_s,
            k_k=self.k_k,
            decay=self.decay,
            grade_step=self.grade_step,
            transfer=self.transfer,
            streak_to_master=self.streak_to_master,
            tau_days=self.tau_days,
            streak_expiry_days=self.streak_expiry_days,
        )

    # ------------------------------------------------------------------ offsets

    def _offset(self, st: LearnerState, ind: str) -> float:
        """Per-indicator offset; for never-seen indicators, borrow evidence from prerequisites (graph transfer)
        and apply a grade-difficulty prior so B6 skills are not assumed known because B1 skills are."""
        if ind in st.theta_k:
            return st.theta_k[ind]
        if self.G is None:
            return 0.0
        if ind not in self.G:
            raise KeyError(f"unknown indicator {ind!r}: not in the curriculum graph")
        g: int = self.G.nodes[ind]["grade"]
        preds = [st.theta_k[p] for p in self.G.predecessors(ind) if p in st.theta_k]
        return -self.grade_step * (g - 1) + (self.transfer * sum(preds) / len(preds) if preds else -0.5)

    def _retention(self, stamp: str | None, ts: float) -> float:
        """Share of a learned offset left at time `ts` (seconds) for a skill last seen at `stamp`."""
        if stamp is None or self.tau_days is None:
            return 1.0
        days = (ts - to_seconds(stamp)) / DAY
        if days <= 0.0:
            return 1.0
        return math.exp(-days / self.tau_days)

    def _offset_at(self, st: LearnerState, ind: str, ts: float) -> float:
        """Effective offset of `ind` at time `ts` (seconds since 1970): rule 2 of the module notes."""
        tk, stamps, G = st.theta_k, st.last_seen_at, self.G
        if G is None or ind not in G:
            raw = tk.get(ind)
            if raw is None:
                if G is None:
                    return 0.0
                raise KeyError(f"unknown indicator {ind!r}: not in the curriculum graph")
            w = self._retention(stamps.get(ind), ts) if G is None else 1.0
            return raw if w == 1.0 else raw * w  # no graph: the prior is 0
        memo: dict[str, float] = {}
        stack = [ind]
        waiting: set[str] = set()
        while stack:
            k = stack[-1]
            if k in memo:
                stack.pop()
                continue
            raw = tk.get(k)
            w = 1.0
            if raw is not None:
                w = self._retention(stamps.get(k), ts)
                if w == 1.0:
                    memo[k] = raw
                    stack.pop()
                    continue
            seen = [p for p in G.predecessors(k) if p in tk]
            todo = [p for p in seen if p not in memo]
            if todo:
                if k in waiting:
                    raise ValueError("forgetting needs a curriculum graph without cycles")
                waiting.add(k)
                stack.extend(todo)
                continue
            waiting.discard(k)
            stack.pop()
            vals = [memo[p] for p in seen]
            prior = -self.grade_step * (G.nodes[k]["grade"] - 1) + (
                self.transfer * sum(vals) / len(vals) if vals else -0.5
            )
            memo[k] = prior if raw is None else prior + (raw - prior) * w
        return memo[ind]

    def effective_offset(self, st: LearnerState, ind: str, now: datetime | str | None = None) -> float:
        """The skill offset the model uses for `ind` at `now` (the stored or prior offset when `now` is None)."""
        if now is None:
            return self._offset(st, ind)
        ts = to_seconds(now)
        return self._offset_at(st, ind, ts) if st.last_seen_at else self._offset(st, ind)

    def _beta(self, item: Any, ind: str, difficulty: str | None) -> float:
        """Difficulty of an item: the `item_difficulty` table first, then the item bank, then 0."""
        bank = self.item_bank
        if bank is None and difficulty is None:
            return self.beta.get(item, 0.0)
        if item in self.beta:
            return self.beta[item]
        if bank is None:
            return beta_for_label(difficulty)
        return bank.get(item, difficulty, ind)

    def _streak_expired(self, st: LearnerState, ind: str, ts: float) -> bool:
        """True when `ind` was last seen more than `streak_expiry_days` before time `ts`."""
        if self.streak_expiry_days is None:
            return False
        stamp = st.last_seen_at.get(ind)
        return stamp is not None and (ts - to_seconds(stamp)) / DAY > self.streak_expiry_days

    # ------------------------------------------------------------------ reading

    def p_correct(
        self,
        st: LearnerState,
        ind: str,
        item: Any = None,
        now: datetime | str | None = None,
        *,
        difficulty: str | None = None,
    ) -> float:
        """Probability of a correct answer on `item` of skill `ind` (a typical item when `item` is None).

        `now` is the time of the question (a datetime or ISO 8601 text, naive values are UTC). When it is
        given, skills practised some time ago have decayed toward their prior. `difficulty` is an optional
        label (`easy`, `medium`, `hard`) used when no fitted difficulty is known for the item."""
        if now is None:
            off = self._offset(st, ind)
        else:
            ts = to_seconds(now)
            off = self._offset_at(st, ind, ts) if st.last_seen_at else self._offset(st, ind)
        return sigmoid(st.theta + off - self._beta(item, ind, difficulty))

    def mastery(self, st: LearnerState, ind: str, now: datetime | str | None = None) -> float:
        """Mastery of `ind`: the probability of a correct answer on a typical item, and at least 0.95
        after `streak_to_master` correct answers in a row.

        With `now`, forgetting applies and a streak older than `streak_expiry_days` no longer counts."""
        p = self.p_correct(st, ind, now=now)
        if st.streak.get(ind, 0) >= self.streak_to_master:
            if now is None or not self._streak_expired(st, ind, to_seconds(now)):
                return max(STREAK_MASTERY, p)
        return p

    def effective_mastery_map(self, st: LearnerState, now: datetime | str | None = None) -> dict[str, float]:
        """Mastery of every skill in the graph at `now`, in graph order, in one pass.

        Each value equals `mastery(st, skill, now=now)`. Use this when all skills are needed: it is
        several times faster than one call per skill. The tables it uses are built on the first call
        and the graph is assumed not to change afterwards (call `forget_graph_tables()` if it does).
        Without a graph, the result covers the skills the learner has practised."""
        G = self.G
        if G is None:
            return {k: self.mastery(st, k, now) for k in st.theta_k}
        ts = to_seconds(now) if now is not None else None
        layout = self._layout()
        tk, theta, transfer, exp = st.theta_k, st.theta, self.transfer, math.exp
        beta_free = self.item_bank is None or not self.item_bank.has_skill_betas
        beta_free = beta_free and None not in self.beta
        out: dict[str, float] = {}
        # Share of each learned offset that is left at `now` (only the skills that have lost some).
        left: dict[str, float] = {}
        tau = self.tau_days
        if ts is not None and tau is not None and tk:
            for k, stamp in st.last_seen_at.items():
                days = (ts - _text_seconds(stamp)) / DAY
                if days > 0.0:
                    w = exp(-days / tau)
                    if w != 1.0:
                        left[k] = w
        if not left:  # nothing has decayed: the v1 arithmetic
            for n, base, preds in layout.rows:
                off = tk.get(n)
                if off is None:
                    if base is None:
                        raise KeyError("grade")
                    vals = [tk[p] for p in preds if p in tk] if tk else None
                    off = base + (transfer * sum(vals) / len(vals) if vals else -0.5)
                z = theta + off if beta_free else theta + off - self._beta(None, n, None)
                try:
                    out[n] = 1 / (1 + exp(-z))
                except OverflowError:
                    out[n] = 0.0
        else:
            if layout.topological is None:
                raise ValueError("forgetting needs a curriculum graph without cycles")
            eff: dict[str, float] = {}  # effective offset of every practised skill met so far
            for n, base, preds in layout.topological:
                raw = tk.get(n)
                if raw is not None and n not in left:
                    off = eff[n] = raw
                else:
                    if base is None:
                        raise KeyError("grade")
                    vals = [eff[p] for p in preds if p in eff]
                    off = base + (transfer * sum(vals) / len(vals) if vals else -0.5)
                    if raw is not None:
                        off = eff[n] = off + (raw - off) * left[n]
                z = theta + off if beta_free else theta + off - self._beta(None, n, None)
                try:
                    out[n] = 1 / (1 + exp(-z))
                except OverflowError:
                    out[n] = 0.0
            if layout.reorder:
                out = {n: out[n] for n, _, _ in layout.rows}
        need = self.streak_to_master
        for n, run in st.streak.items():
            if run >= need:
                p = out.get(n)
                if p is not None and p < STREAK_MASTERY and (ts is None or not self._streak_expired(st, n, ts)):
                    out[n] = STREAK_MASTERY
        return out

    def forget_graph_tables(self) -> None:
        """Drop the per-graph tables of `effective_mastery_map` (needed only after editing the graph)."""
        self._layout_cache = None

    def _layout(self) -> _Layout:
        G = self.G
        key = (id(G), len(G), self.grade_step)
        cached = self._layout_cache
        if cached is not None and cached.key == key:
            return cached
        rows: list[tuple[str, float | None, tuple[str, ...]]] = []
        for n, d in G.nodes(data=True):
            g = d.get("grade")
            rows.append((n, None if g is None else -self.grade_step * (g - 1), tuple(G.predecessors(n))))
        index = {n: i for i, (n, _, _) in enumerate(rows)}
        topological: list[tuple[str, float | None, tuple[str, ...]]] | None = rows
        reorder = False
        if any(index[p] >= i for i, (_, _, preds) in enumerate(rows) for p in preds):
            reorder = True
            waiting = {n: len(preds) for n, _, preds in rows}
            ready = [n for n, _, preds in rows if not preds]
            order: list[str] = []
            while ready:
                n = ready.pop()
                order.append(n)
                for s in G.successors(n):
                    waiting[s] -= 1
                    if waiting[s] == 0:
                        ready.append(s)
            topological = [rows[index[n]] for n in order] if len(order) == len(rows) else None  # None: a cycle
        layout = _Layout(key, rows, topological, reorder)
        self._layout_cache = layout
        return layout

    # ------------------------------------------------------------------ learning

    def update(
        self,
        st: LearnerState,
        ind: str,
        correct: object,
        item: Any = None,
        at: datetime | str | None = None,
        *,
        difficulty: str | None = None,
    ) -> float:
        """Apply one answer and return the probability predicted before seeing it.

        With a graph, an indicator that is not in it raises KeyError and leaves the state untouched.
        `at` is when the answer was given (a datetime or ISO 8601 text, naive values are UTC). With it,
        the prediction and the step start from the offset as it has decayed by that time, and the time
        is recorded in `st.last_seen_at[ind]` (rules 4 and 6 of the module notes). Without it the
        update is exactly the v1 update (rule 5). `difficulty` is an optional label (`easy`, `medium`,
        `hard`) used when no fitted difficulty is known for the item."""
        if self.G is not None and ind not in self.G:
            raise KeyError(f"unknown indicator {ind!r}: not in the curriculum graph")
        correct = bool(correct)
        stamp, ts, keep_streak = None, 0.0, True
        if at is None:
            off = self._offset(st, ind)
        else:
            stamp, ts = to_iso(at), to_seconds(at)  # checked before the state is touched
            off = self._offset_at(st, ind, ts) if st.last_seen_at else self._offset(st, ind)
            keep_streak = not self._streak_expired(st, ind, ts)
        p = sigmoid(st.theta + off - self._beta(item, ind, difficulty))
        e = int(correct) - p
        na, nk = st.n.get("_all", 0), st.n.get(ind, 0)
        st.theta += self.k_s / (1 + self.decay * na) * e
        st.theta_k[ind] = off + self.k_k / (1 + self.decay * nk) * e
        st.streak[ind] = (st.streak.get(ind, 0) if keep_streak else 0) + 1 if correct else 0
        st.n["_all"], st.n[ind] = na + 1, nk + 1
        st.step += 1
        st.last_seen[ind] = st.step
        if stamp is None:
            if st.last_seen_at:
                st.last_seen_at.pop(ind, None)
        else:
            last = st.last_seen_at.get(ind)
            if last is None or ts >= to_seconds(last):
                st.last_seen_at[ind] = stamp
        return p
