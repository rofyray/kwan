"""Seeded exploration and strand variety for the recommender (PRD 6.3, decisions D6 and D8).

Everything here is small and portable on purpose: a port in another language can reproduce the same
picks from the same learner id and step. The test vectors in `tests/fixtures/mulberry32_vectors.json`
pin the generator.

What a port must match:

- `Mulberry32.next_u32` and `Mulberry32.next` are exact: 32-bit integer arithmetic only, then one
  division by 2**32.
- `Mulberry32.normal` uses `log`, `sqrt` and `cos`. `sqrt` is exact everywhere, but `log` and `cos`
  may differ in the last bit between maths libraries, so a port matches `normal` to about 1e-12,
  not bit for bit.
- The seed is `crc32` (the zlib one, also used by PNG and gzip) of the UTF-8 text
  `"<learner_id>:<step>"`.
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Iterable, Sequence
from typing import NamedTuple

_MASK = 0xFFFFFFFF
_TWO_32 = 4294967296.0


class Mulberry32:
    """Mulberry32, a tiny 32-bit pseudo-random generator with one word of state."""

    def __init__(self, seed: int) -> None:
        self.a: int = int(seed) & _MASK

    def next_u32(self) -> int:
        """The next 32-bit unsigned integer."""
        self.a = (self.a + 0x6D2B79F5) & _MASK
        t = self.a
        t = ((t ^ (t >> 15)) * (t | 1)) & _MASK
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & _MASK)) & _MASK
        return (t ^ (t >> 14)) & _MASK

    def next(self) -> float:
        """The next number in [0, 1): `next_u32() / 2**32`."""
        return self.next_u32() / _TWO_32

    def normal(self) -> float:
        """One standard normal value by Box-Muller. Uses exactly two calls to `next`.

        The first uniform is turned into `1 - u`, which lies in (0, 1], so its logarithm always exists."""
        u1 = 1.0 - self.next()
        u2 = self.next()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def seed_for(learner_id: str, step: int) -> int:
    """The exploration seed for one learner at one step: crc32 of `"<learner_id>:<step>"` in UTF-8."""
    return zlib.crc32(f"{learner_id}:{step}".encode())


def _rng(learner_id: str, step: int) -> Mulberry32:
    """The generator for one learner at one step (D6), so any recommendation can be reproduced."""
    return Mulberry32(seed_for(learner_id, step))


def explore(score: float, n_k: int, rng: Mulberry32, eps: float) -> float:
    """`score + eps * N(0, 1) / sqrt(1 + n_k)`: skills with few answers get the larger nudge.

    With `eps == 0` the score comes back unchanged and the generator is not advanced."""
    if not eps:
        return score
    return score + eps * rng.normal() / math.sqrt(1 + n_k)


class Candidate(NamedTuple):
    """One skill in the greedy pick.

    tier: skills in a higher tier always come first (strict goal mode: 1 on the path, 0 off it).
    rank: the score used for ordering inside a tier (the formula score, plus exploration when on).
    id: the indicator id. It breaks ties, highest first, so the order never depends on input order.
    group: the content standard, used by the variety penalty.
    """

    tier: int
    rank: float
    id: str
    group: str


def pick_with_variety(cands: Iterable[Candidate], k: int, penalty: float = 0.05) -> list[Candidate]:
    """Greedy top-k, best first. After each pick, the others from the same `group` lose `penalty`.

    The penalty adds up: a skill whose content standard was already picked twice loses twice the
    penalty. It only reorders skills inside a tier. With `penalty == 0` this is a plain sort by
    `(tier, rank, id)`, highest first.
    """
    if penalty < 0:
        raise ValueError(f"penalty must be 0 or more, got {penalty}")
    pool: list[Candidate] = sorted(cands, reverse=True)
    if k <= 0:
        return []
    if not penalty:
        return pool[:k]
    chosen: list[Candidate] = []
    seen: dict[str, int] = {}
    while pool and len(chosen) < k:
        best_i = 0
        best_key: tuple[int, float, str] | None = None
        for i, c in enumerate(pool):
            # `pool` is sorted without penalties and a penalty only lowers a rank, so once an
            # unpenalised entry is below the best penalised one, nothing after it can win.
            if best_key is not None and (c.tier, c.rank, c.id) < best_key:
                break
            key = (c.tier, c.rank - penalty * seen.get(c.group, 0), c.id)
            if best_key is None or key > best_key:
                best_i, best_key = i, key
        best = pool.pop(best_i)
        chosen.append(best)
        seen[best.group] = seen.get(best.group, 0) + 1
    return chosen


def content_standard_of(code: str) -> str:
    """The content standard of a bare indicator code: `B4.1.1.1.2` gives `B4.1.1.1`.

    A code without a dot is its own group, so it never shares a penalty."""
    head, dot, _ = code.rpartition(".")
    return head if dot else code


__all__: Sequence[str] = (
    "Candidate",
    "Mulberry32",
    "content_standard_of",
    "explore",
    "pick_with_variety",
    "seed_for",
)
