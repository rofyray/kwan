"""A decision model with fixed outputs, for tests and for `kwan graph augment --dry-run`.

It never looks at meaning. With `outputs` it returns the number given for a pair; otherwise it
returns a number derived from a checksum of the two ids, which is stable from run to run.
"""

from __future__ import annotations

import zlib
from collections.abc import Mapping

from kwan.graph.augment import PairQuestion

__all__ = ["FakeDecisionModel"]


class FakeDecisionModel:
    """Fixed answers. `outputs` maps `(a_id, b_id)` to a probability; other pairs get `default`.

    With `default=None` a pair that is not listed gets a checksum-based number between 0 and 1, the
    same on every run. `asked` records the pairs in the order they were asked.
    """

    name = "fake"
    model = "fixed-outputs"
    licence = "not a model"
    last_cost_usd = 0.0

    def __init__(self, outputs: Mapping[tuple[str, str], float] | None = None, default: float | None = None) -> None:
        self.outputs = dict(outputs or {})
        self.default = default
        self.asked: list[tuple[str, str]] = []

    def prerequisite_probability(self, q: PairQuestion) -> float:
        pair = (q.a_id, q.b_id)
        self.asked.append(pair)
        if pair in self.outputs:
            return float(self.outputs[pair])
        if self.default is not None:
            return float(self.default)
        return (zlib.crc32(f"{q.a_id}>{q.b_id}".encode()) % 10_000) / 10_000
