"""Bayesian Knowledge Tracing baseline (PRD 5.5): four parameters per skill, fitted by EM.

The model, for one learner on one skill: the skill is known at the start with probability `p_init`;
after each answer an unknown skill becomes known with probability `p_learn` (nothing is forgotten); a
learner who knows the skill still slips with probability `p_slip`, and one who does not guesses right
with probability `p_guess`.

The fit is the standard EM (Baum-Welch) for this two-state model, vectorised with numpy across all
learners and all skills at once: sequences are laid out longest first, so step t of every sequence
that is long enough is one slice. One EM round costs a few numpy calls per time step, which keeps
millions of rows practical. This module is for evaluation only and is the one learner module that
imports numpy.

    model = BKT().fit(users, skills, correct)        # three aligned columns, rows in time order
    p = model.predict(users, skills, correct)        # P(correct) before each answer, in row order
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import astuple, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class BKTParams:
    """The four BKT probabilities of one skill."""

    p_init: float = 0.4
    p_learn: float = 0.1
    p_slip: float = 0.1
    p_guess: float = 0.2

    def __post_init__(self) -> None:
        for name, value in zip(("p_init", "p_learn", "p_slip", "p_guess"), astuple(self)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a probability, got {value!r}")

    def as_tuple(self) -> tuple[float, float, float, float]:
        """(p_init, p_learn, p_slip, p_guess)."""
        return (self.p_init, self.p_learn, self.p_slip, self.p_guess)

    def to_dict(self) -> dict[str, float]:
        return {"p_init": self.p_init, "p_learn": self.p_learn, "p_slip": self.p_slip, "p_guess": self.p_guess}

    @classmethod
    def from_dict(cls, d: Mapping[str, float]) -> BKTParams:
        return cls(float(d["p_init"]), float(d["p_learn"]), float(d["p_slip"]), float(d["p_guess"]))


DEFAULT = BKTParams()  # used for a skill that was not in the fitted data (the v1 evaluation fallback)

# Starting points tried for every skill before EM: the corners of the v1 grid, plus the default.
STARTS: tuple[BKTParams, ...] = (
    DEFAULT,
    *(BKTParams(pi, pt, ps, pg) for pi in (0.2, 0.6) for pt in (0.05, 0.3) for ps in (0.05, 0.2) for pg in (0.1, 0.3)),
)


def _as_params(p: BKTParams | Sequence[float]) -> BKTParams:
    return p if isinstance(p, BKTParams) else BKTParams(*(float(x) for x in p))


def forward(ys: Iterable[Any], params: BKTParams | Sequence[float] = DEFAULT) -> list[float]:
    """P(correct) before each answer of one learner on one skill (plain Python, the reference).

    With p_known the probability that the skill is known before an answer:
        p_correct = p_known * (1 - p_slip) + (1 - p_known) * p_guess
        posterior = P(known | answer), by Bayes' rule
        p_known'  = posterior + (1 - posterior) * p_learn
    """
    pi, pt, ps, pg = _as_params(params).as_tuple()
    known, out = pi, []
    for y in ys:
        pc = known * (1 - ps) + (1 - known) * pg
        out.append(pc)
        if y:
            post = known * (1 - ps) / pc if pc > 0 else known
        else:
            post = known * ps / (1 - pc) if pc < 1 else known
        known = post + (1 - post) * pt
    return out


def log_likelihood(ys: Sequence[Any], params: BKTParams | Sequence[float] = DEFAULT) -> float:
    """Log-probability of one answer sequence under the model (plain Python)."""
    total = 0.0
    for y, pc in zip(ys, forward(ys, params)):
        q = pc if y else 1 - pc
        total += math.log(q) if q > 0 else -math.inf
    return total


def simulate(
    params: Mapping[Any, BKTParams | Sequence[float]], n_users: int, n_steps: int, seed: int = 0
) -> list[tuple[int, None, Any, int]]:
    """Synthetic answers drawn from the model: `(user, None, skill, correct)` rows, `n_steps` answers
    per user on every skill, grouped by user and skill. For tests and for checking a fit."""
    rng = random.Random(seed)
    rows: list[tuple[int, None, Any, int]] = []
    for u in range(n_users):
        for skill, p in params.items():
            pi, pt, ps, pg = _as_params(p).as_tuple()
            known = rng.random() < pi
            for _ in range(n_steps):
                rows.append((u, None, skill, int(rng.random() < ((1 - ps) if known else pg))))
                known = known or rng.random() < pt
    return rows


def _codes(values: Any) -> tuple[IntArray, list[Any]]:
    """Integer code per value and the list of values by code (code -> value).

    Whole-number arrays over a compact range are coded by offset, without sorting; the list may then
    hold values that never occur."""
    if isinstance(values, np.ndarray) and values.dtype.kind in "iu" and values.size:
        lo, hi = int(values.min()), int(values.max())
        if hi - lo < 4 * values.size + 1024:
            return (values.astype(np.int64) - lo).reshape(-1), list(range(lo, hi + 1))
    if isinstance(values, np.ndarray) and values.dtype.kind in "iuUSb":
        uniq, inverse = np.unique(values, return_inverse=True)
        return inverse.astype(np.int64).reshape(-1), list(uniq.tolist())
    items = values.tolist() if isinstance(values, np.ndarray) else values
    index: dict[Any, int] = {}
    codes = np.fromiter((index.setdefault(v, len(index)) for v in items), dtype=np.int64)
    return codes, list(index)


def _answers(correct: Any) -> BoolArray:
    arr = np.asarray(correct)
    out: BoolArray = arr.astype(bool) if arr.dtype.kind in "biuf" else np.array([bool(v) for v in correct])
    return out.reshape(-1)


@dataclass(frozen=True)
class _Layout:
    """Rows rearranged step by step: all first answers, then all second answers, and so on.

    Sequences (one per learner and skill) are ranked longest first, so the sequences still running at
    step t are the ranks 0 .. counts[t]-1, and slot j of step t belongs to rank j."""

    n_rows: int
    n_seqs: int
    n_skills: int
    counts: list[int]  # sequences still running at each step
    row: IntArray  # slot -> index of the row in the input
    seq: IntArray  # slot -> rank of its sequence
    skill: IntArray  # slot -> skill code
    seq_skill: IntArray  # rank -> skill code
    has_next: BoolArray  # slot -> the sequence has another answer after this one
    y: BoolArray  # slot -> the answer


def _stable_argsort(keys: IntArray) -> IntArray:
    """Stable argsort of whole numbers that are 0 or more, 16 bits at a time (numpy sorts 16-bit keys
    by counting, which is linear and much faster than its general stable sort on long arrays)."""
    order = np.arange(len(keys), dtype=np.int64)
    top = int(np.max(keys)) if len(keys) else 0
    shift = 0
    while True:
        digit = ((keys[order] >> shift) & 0xFFFF).astype(np.uint16)
        order = order[np.argsort(digit, kind="stable")]
        shift += 16
        if top >> shift == 0:
            return order


def _layout(user_codes: IntArray, skill_codes: IntArray, n_skills: int, y: BoolArray) -> _Layout:
    n_rows = len(y)
    if not (len(user_codes) == len(skill_codes) == n_rows):
        raise ValueError("users, skills and correct must have the same length")
    width = np.int64(max(n_skills, 1))
    key = user_codes * width + skill_codes
    by_seq = _stable_argsort(key)  # rows grouped by sequence, input order kept inside
    grouped = key[by_seq]
    first = np.flatnonzero(np.append(True, grouped[1:] != grouped[:-1])) if n_rows else np.zeros(0, dtype=np.int64)
    lengths = np.diff(np.append(first, n_rows))
    n_seqs = len(first)
    pos = np.arange(n_rows, dtype=np.int64) - np.repeat(first, lengths)  # step of each grouped row
    longest_first = (
        _stable_argsort((int(np.max(lengths)) - lengths).astype(np.int64)) if n_seqs else np.zeros(0, dtype=np.int64)
    )
    rank_of_seq = np.empty(n_seqs, dtype=np.int64)
    rank_of_seq[longest_first] = np.arange(n_seqs, dtype=np.int64)
    rank = np.repeat(rank_of_seq, lengths)
    counts = np.bincount(pos).astype(np.int64) if n_rows else np.zeros(0, dtype=np.int64)
    slot = (np.cumsum(counts) - counts)[pos] + rank  # step t starts at the sum of the earlier counts
    row = np.empty(n_rows, dtype=np.int64)
    row[slot] = by_seq
    seq = np.empty(n_rows, dtype=np.int64)
    seq[slot] = rank
    seq_skill = (grouped[first] % width)[longest_first].astype(np.int64)
    next_count = np.append(counts[1:], 0)
    return _Layout(
        n_rows=n_rows,
        n_seqs=n_seqs,
        n_skills=n_skills,
        counts=[int(c) for c in counts],
        row=row,
        seq=seq,
        skill=seq_skill[seq],
        seq_skill=seq_skill,
        has_next=seq < np.repeat(next_count, counts),
        y=y[row],
    )


def _emissions(lay: _Layout, slip: FloatArray, guess: FloatArray) -> tuple[FloatArray, FloatArray]:
    """P(answer | known) and P(answer | not known) for every slot."""
    s, g = slip[lay.skill], guess[lay.skill]
    return np.where(lay.y, 1.0 - s, s), np.where(lay.y, g, 1.0 - g)


def _filter(
    lay: _Layout, init: FloatArray, learn: FloatArray, e1: FloatArray, e0: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Forward pass. Per slot: P(known before the answer | earlier answers), P(known | answers so far)
    and P(this answer | earlier answers)."""
    prior = np.empty(lay.n_rows)
    post = np.empty(lay.n_rows)
    lik = np.empty(lay.n_rows)
    known = init[lay.seq_skill].copy()
    learn_seq = learn[lay.seq_skill]
    off = 0
    for n in lay.counts:
        end = off + n
        k = known[:n]
        prior[off:end] = k
        num = k * e1[off:end]
        den = num + (1.0 - k) * e0[off:end]
        f = np.divide(num, den, out=k.copy(), where=den > 0.0)  # an impossible answer changes nothing
        post[off:end] = f
        lik[off:end] = den
        known[:n] = f + (1.0 - f) * learn_seq[:n]
        off = end
    return prior, post, lik


def _smooth(lay: _Layout, learn: FloatArray, prior: FloatArray, post: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Backward pass. Per slot: P(known | all answers of the sequence) and
    P(not known here and known at the next answer | all answers)."""
    gamma = post.copy()  # correct as it stands for the last answer of every sequence
    xi = np.zeros(lay.n_rows)
    learn_seq = learn[lay.seq_skill]
    offsets = np.cumsum([0, *lay.counts])
    for t in range(len(lay.counts) - 2, -1, -1):
        n1 = lay.counts[t + 1]
        a, b = int(offsets[t]), int(offsets[t + 1])
        ratio = gamma[b : b + n1] / prior[b : b + n1]
        f = post[a : a + n1]
        gamma[a : a + n1] = f * ratio
        xi[a : a + n1] = (1.0 - f) * learn_seq[:n1] * ratio
    np.clip(gamma, 0.0, 1.0, out=gamma)
    return gamma, xi


def _skill_sum(lay: _Layout, values: FloatArray) -> FloatArray:
    return np.bincount(lay.skill, weights=values, minlength=lay.n_skills)


def _ratio(num: FloatArray, den: FloatArray, old: FloatArray) -> FloatArray:
    """num / den where there is evidence, the old value elsewhere."""
    out = old.copy()
    ok = den > 1e-12
    out[ok] = num[ok] / den[ok]
    return out


class BKT:
    """Standard four-parameter BKT per skill with a vectorised EM fit.

    max_iter, tol: EM stops after `max_iter` rounds, or when no skill gains more than `tol` in mean
        log-likelihood per answer in a round.
    starts: starting points tried for every skill; EM starts from the best one per skill. EM never
        lowers the likelihood, so the fit is at least as good as every starting point.
    max_slip, max_guess: upper bounds that keep the model meaningful (a known skill is answered right
        more often than an unknown one). eps: every probability stays in [eps, 1 - eps].
    default: parameters used for a skill that was not in the fitted data.

    After `fit`: `params` ({skill: BKTParams}), `log_likelihood_` ({skill: training log-likelihood}),
    `n_iter_` and `converged_`."""

    def __init__(
        self,
        max_iter: int = 200,
        tol: float = 1e-6,
        starts: Sequence[BKTParams | Sequence[float]] | None = None,
        max_slip: float = 0.5,
        max_guess: float = 0.5,
        eps: float = 1e-4,
        default: BKTParams = DEFAULT,
    ) -> None:
        if max_iter < 0:
            raise ValueError(f"max_iter must be 0 or more, got {max_iter}")
        if not 0.0 < eps < 0.5:
            raise ValueError(f"eps must be between 0 and 0.5, got {eps}")
        self.max_iter, self.tol, self.eps = max_iter, tol, eps
        self.starts: tuple[BKTParams, ...] = tuple(_as_params(s) for s in starts) if starts is not None else STARTS
        if not self.starts:
            raise ValueError("starts needs at least one starting point")
        self.max_slip, self.max_guess = max_slip, max_guess
        self.default = default
        self.params: dict[Any, BKTParams] = {}
        self.log_likelihood_: dict[Any, float] = {}
        self.n_iter_ = 0
        self.converged_ = False

    # ------------------------------------------------------------------ fitting

    def _clip(self, p: FloatArray, upper: float | None = None) -> FloatArray:
        return np.clip(p, self.eps, min(1.0 - self.eps, upper) if upper is not None else 1.0 - self.eps)

    def fit(self, users: Any, skills: Any, correct: Any) -> BKT:
        """Fit every skill from three aligned columns. Rows must be in time order for each learner
        (any interleaving of learners and skills is fine). Returns self."""
        y = _answers(correct)
        user_codes, _ = _codes(users)
        skill_codes, skill_values = _codes(skills)
        n_skills = len(skill_values)
        self.params, self.log_likelihood_, self.n_iter_, self.converged_ = {}, {}, 0, True
        if len(y) == 0:
            return self
        lay = _layout(user_codes, skill_codes, n_skills, y)
        n_k = np.bincount(lay.skill, minlength=n_skills).astype(np.float64)
        seqs_k = np.bincount(lay.seq_skill, minlength=n_skills).astype(np.float64)
        yf = lay.y.astype(np.float64)
        steps = lay.has_next.astype(np.float64)

        # Starting point: the best of `starts` for each skill.
        init, learn = np.empty(n_skills), np.empty(n_skills)
        slip, guess = np.empty(n_skills), np.empty(n_skills)
        best = np.full(n_skills, -np.inf)
        for start in self.starts:
            c = [np.full(n_skills, v) for v in start.as_tuple()]
            ci, cl, cs, cg = self._clip(c[0]), self._clip(c[1]), self._clip(c[2]), self._clip(c[3])
            _, _, lik = _filter(lay, ci, cl, *_emissions(lay, cs, cg))
            ll = _skill_sum(lay, np.log(lik))
            better = ll > best
            best = np.where(better, ll, best)
            init, learn = np.where(better, ci, init), np.where(better, cl, learn)
            slip, guess = np.where(better, cs, slip), np.where(better, cg, guess)

        ll = previous = best
        self.converged_ = self.max_iter == 0
        for it in range(1, self.max_iter + 1):
            prior, post, lik = _filter(lay, init, learn, *_emissions(lay, slip, guess))
            ll = _skill_sum(lay, np.log(lik))
            if it > 1 and bool(np.all(ll - previous <= self.tol * n_k)):
                self.converged_ = True
                break
            previous = ll
            gamma, xi = _smooth(lay, learn, prior, post)
            known = _skill_sum(lay, gamma)
            init = self._clip(_ratio(np.bincount(lay.seq_skill, gamma[: lay.n_seqs], n_skills), seqs_k, init))
            learn = self._clip(_ratio(_skill_sum(lay, xi), _skill_sum(lay, (1.0 - gamma) * steps), learn))
            slip = self._clip(_ratio(_skill_sum(lay, gamma * (1.0 - yf)), known, slip), self.max_slip)
            guess = self._clip(_ratio(_skill_sum(lay, (1.0 - gamma) * yf), n_k - known, guess), self.max_guess)
            self.n_iter_ = it
        else:
            if self.max_iter:
                _, _, lik = _filter(lay, init, learn, *_emissions(lay, slip, guess))
                ll = _skill_sum(lay, np.log(lik))
        for code, skill in enumerate(skill_values):
            if n_k[code] == 0:  # a code without rows (see `_codes`)
                continue
            self.params[skill] = BKTParams(float(init[code]), float(learn[code]), float(slip[code]), float(guess[code]))
            self.log_likelihood_[skill] = float(ll[code])
        return self

    def fit_interactions(self, interactions: Iterable[Sequence[Any]]) -> BKT:
        """Fit from `(user, item, skill, correct, ...)` rows, the format of the other fitting tools."""
        rows = interactions if isinstance(interactions, Sequence) else list(interactions)
        return self.fit([r[0] for r in rows], [r[2] for r in rows], [bool(r[3]) for r in rows])

    # ------------------------------------------------------------------ predicting

    def _table(self, skill_values: list[Any]) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        rows = [self.params.get(s, self.default).as_tuple() for s in skill_values]
        table = np.array(rows, dtype=np.float64).reshape(-1, 4)
        return table[:, 0].copy(), table[:, 1].copy(), table[:, 2].copy(), table[:, 3].copy()

    def _run(self, users: Any, skills: Any, correct: Any) -> tuple[_Layout, FloatArray, FloatArray]:
        y = _answers(correct)
        user_codes, _ = _codes(users)
        skill_codes, skill_values = _codes(skills)
        lay = _layout(user_codes, skill_codes, len(skill_values), y)
        init, learn, slip, guess = self._table(skill_values)
        prior, _, lik = _filter(lay, init, learn, *_emissions(lay, slip, guess))
        s, g = slip[lay.skill], guess[lay.skill]
        return lay, prior * (1.0 - s) + (1.0 - prior) * g, lik

    def predict(self, users: Any, skills: Any, correct: Any) -> FloatArray:
        """P(correct) before each answer, in the order of the rows. Each learner starts fresh on each
        skill, and earlier answers in the rows update the estimate (next-answer prediction). Skills
        that were not fitted use `default`."""
        if len(_answers(correct)) == 0:
            return np.empty(0)
        lay, p_correct, _ = self._run(users, skills, correct)
        out = np.empty(lay.n_rows)
        out[lay.row] = p_correct
        return out

    def predict_interactions(self, interactions: Iterable[Sequence[Any]]) -> FloatArray:
        """`predict` for `(user, item, skill, correct, ...)` rows."""
        rows = interactions if isinstance(interactions, Sequence) else list(interactions)
        return self.predict([r[0] for r in rows], [r[2] for r in rows], [bool(r[3]) for r in rows])

    def log_likelihood(self, users: Any, skills: Any, correct: Any) -> float:
        """Total log-likelihood of the answers under the fitted parameters."""
        if len(_answers(correct)) == 0:
            return 0.0
        _, _, lik = self._run(users, skills, correct)
        return float(np.log(lik).sum())
