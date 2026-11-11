"""Candidate prerequisite pairs for the decision model (PRD Phase 7, task 7.1).

A pair A -> B is a candidate when

* grade(A) <= grade(B) <= grade(A) + 2,
* A and B are not already joined by an edge (in either direction),
* A and B are in the same strand, or (strand of A, strand of B) is listed in
  `data/graph/cross_strand_rules.yaml`,

and, for each target B, only the 20 sources whose text is most similar to B's are kept. Similarity
is the cosine of sentence embeddings (`all-MiniLM-L6-v2`, run on this machine).

The embedder is injected. `MiniLMEmbedder` needs the `graph-ai` extra and is imported only when it
is used; `HashingEmbedder` is a small bag-of-words stand-in with no dependencies, used by the tests
and by `kwan graph augment --dry-run`.
"""

from __future__ import annotations

import math
import re
import zlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from kwan._paths import data_dir

TOP_K = 20
GRADE_WINDOW = 2
MINILM = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_LICENCE = "Apache-2.0"

__all__ = [
    "GRADE_WINDOW",
    "MINILM",
    "MINILM_LICENCE",
    "TOP_K",
    "Candidate",
    "CandidateIndex",
    "CrossStrandRules",
    "Embedder",
    "HashingEmbedder",
    "MiniLMEmbedder",
    "candidate_pairs",
    "default_rules_path",
    "load_cross_strand_rules",
    "similarity_table",
]


class Embedder(Protocol):
    """Anything that turns texts into vectors. Vectors need not be normalised."""

    name: str

    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


_TOKEN = re.compile(r"[a-z]+|\d+(?:\.\d+)?")


class HashingEmbedder:
    """Deterministic bag-of-words vectors (words and word pairs hashed into `dim` buckets).

    It knows nothing about meaning. It exists so that the pipeline can run in tests and in a dry run
    without downloading a model.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self.name = f"hashing-bow-{dim}"

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            words = _TOKEN.findall(text.lower())
            for gram in [*words, *(f"{a} {b}" for a, b in zip(words, words[1:]))]:
                h = zlib.crc32(gram.encode("utf-8"))
                vec[h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
            out.append(vec)
        return out


class MiniLMEmbedder:
    """`all-MiniLM-L6-v2` sentence embeddings, run locally (Apache-2.0). Needs `uv add 'kwan[graph-ai]'`."""

    licence = MINILM_LICENCE

    def __init__(self, model: str = MINILM, batch_size: int = 64) -> None:
        self.name = model
        self.batch_size = batch_size
        self._model: Any = None

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # optional dependency: graph-ai extra

            self._model = SentenceTransformer(self.name)
        vecs = self._model.encode(list(texts), batch_size=self.batch_size, show_progress_bar=False)
        return [[float(x) for x in v] for v in vecs]


# ---------------------------------------------------------------- cross-strand rules


@dataclass(frozen=True)
class CrossStrandRules:
    """The (source strand, target strand) pairs that may be proposed across strands."""

    pairs: frozenset[tuple[str, str]]

    def allows(self, a_strand: Any, b_strand: Any) -> bool:
        """True for the same strand, or for a listed cross-strand pair."""
        a, b = str(a_strand), str(b_strand)
        return a == b or (a, b) in self.pairs


def default_rules_path() -> Path:
    """`data/graph/cross_strand_rules.yaml` in the repository data directory."""
    return data_dir() / "graph" / "cross_strand_rules.yaml"


def load_cross_strand_rules(path: str | Path | None = None, section: str = "math") -> CrossStrandRules:
    """Read one section of the rules file. Each rule is `{from: <strand>, to: <strand>, why: "..."}`."""
    import yaml  # optional dependency: graph-ai or dev extra

    with open(path or default_rules_path(), encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    rules = (doc.get(section) or {}).get("rules") or []
    pairs = set()
    for rule in rules:
        if "from" not in rule or "to" not in rule:
            raise ValueError(f"cross-strand rule without 'from' and 'to': {rule!r}")
        pairs.add((str(rule["from"]), str(rule["to"])))
    return CrossStrandRules(frozenset(pairs))


# ---------------------------------------------------------------- similarity


def _unit(vec: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm > 0 else [0.0] * len(vec)


def similarity_table(vectors: Sequence[Sequence[float]]) -> list[list[float]]:
    """Cosine similarity of every vector with every other. Uses numpy when it is installed."""
    try:
        import numpy as np
    except ModuleNotFoundError:
        units = [_unit(v) for v in vectors]
        return [[sum(x * y for x, y in zip(u, w)) for w in units] for u in units]
    m = np.asarray(vectors, dtype=float)
    if m.size == 0:
        return []
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    m = m / norms
    return [[float(x) for x in row] for row in (m @ m.T)]


def _mean_sd(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mu = sum(values) / len(values)
    return mu, math.sqrt(sum((v - mu) ** 2 for v in values) / len(values))


@dataclass(frozen=True)
class Candidate:
    """One candidate pair `source -> target` with what the candidate stage knows about it.

    `signals` are numbers a decision model may use: `sim` (cosine), `rank` (1 = most similar source
    for this target), `z` (how far `sim` stands above this target's other eligible sources, in
    standard deviations), `margin` (best eligible similarity for this target minus `sim`), `z_rev`
    (the same pair seen from the source: how far `sim` stands above the source's similarity to every
    other node) and `central` (mean similarity of the source to all nodes minus that of the target).
    """

    source: str
    target: str
    similarity: float
    rank: int
    signals: tuple[tuple[str, float], ...] = ()


class CandidateIndex:
    """Similarities and eligibility for one set of nodes; gives the candidates and the signals of any pair.

    `nodes` maps an id to a record with `grade` and `strand`; `texts` maps an id to the text that is
    embedded (the indicator description when it is available, otherwise the short label);
    `connected` lists the pairs that already have an edge.
    """

    def __init__(
        self,
        nodes: Mapping[str, Mapping[str, Any]],
        texts: Mapping[str, str],
        embedder: Embedder,
        rules: CrossStrandRules,
        connected: Iterable[tuple[str, str]] = (),
        grade_window: int = GRADE_WINDOW,
    ) -> None:
        self.nodes = nodes
        self.rules = rules
        self.grade_window = grade_window
        self.order = sorted(nodes)
        self.index = {n: k for k, n in enumerate(self.order)}
        self.sims = similarity_table(embedder.encode([texts[n] for n in self.order])) if self.order else []
        self.joined: set[tuple[str, str]] = set()
        for a, b in connected:
            self.joined.add((a, b))
            self.joined.add((b, a))
        self._others = [_mean_sd([x for k, x in enumerate(row) if k != i]) for i, row in enumerate(self.sims)]
        self._eligible: dict[str, list[tuple[str, float]]] = {}

    def similarity(self, a: str, b: str) -> float:
        """Cosine similarity of the texts of `a` and `b`."""
        return self.sims[self.index[a]][self.index[b]]

    def is_eligible(self, a: str, b: str) -> bool:
        """True when `a -> b` passes the grade window, the strand rule and the not-connected rule."""
        if a == b or (a, b) in self.joined:
            return False
        ga, gb = int(self.nodes[a]["grade"]), int(self.nodes[b]["grade"])
        if not (ga <= gb <= ga + self.grade_window):
            return False
        return self.rules.allows(self.nodes[a]["strand"], self.nodes[b]["strand"])

    def eligible(self, b: str) -> list[tuple[str, float]]:
        """The eligible sources of target `b` with their similarity, most similar first (ties by id)."""
        if b not in self._eligible:
            j = self.index[b]
            pairs = [(a, self.sims[self.index[a]][j]) for a in self.order if self.is_eligible(a, b)]
            self._eligible[b] = sorted(pairs, key=lambda t: (-t[1], t[0]))
        return self._eligible[b]

    def signals(self, a: str, b: str, rank: int | None = None) -> tuple[tuple[str, float], ...]:
        """The signals of the pair `a -> b` (see `Candidate`). Works for any pair, eligible or not."""
        s = self.similarity(a, b)
        ranked = self.eligible(b)
        sims = [x for _, x in ranked]
        if rank is None:
            rank = 1 + sum(1 for x in sims if x > s)
        mu, sd = _mean_sd(sims)
        mu_a, sd_a = self._others[self.index[a]]
        mu_b, _ = self._others[self.index[b]]
        best = max(sims) if sims else s
        return (
            ("sim", round(s, 6)),
            ("rank", float(rank)),
            ("z", round((s - mu) / sd, 6) if sd > 0 else 0.0),
            ("margin", round(max(best - s, 0.0), 6)),
            ("z_rev", round((s - mu_a) / sd_a, 6) if sd_a > 0 else 0.0),
            ("central", round(mu_a - mu_b, 6)),
        )

    def candidates(self, top_k: int = TOP_K) -> list[Candidate]:
        """At most `top_k` candidates per target, most similar first."""
        out: list[Candidate] = []
        for b in self.order:
            for rank, (a, s) in enumerate(self.eligible(b)[:top_k], start=1):
                out.append(Candidate(a, b, s, rank, self.signals(a, b, rank)))
        return out


def candidate_pairs(
    nodes: Mapping[str, Mapping[str, Any]],
    texts: Mapping[str, str],
    embedder: Embedder,
    rules: CrossStrandRules,
    connected: Iterable[tuple[str, str]] = (),
    top_k: int = TOP_K,
    grade_window: int = GRADE_WINDOW,
) -> list[Candidate]:
    """The candidate pairs of task 7.1, at most `top_k` per target, most similar first.

    See `CandidateIndex` for the arguments. The result is deterministic: ties in similarity are
    broken by id.
    """
    return CandidateIndex(nodes, texts, embedder, rules, connected, grade_window).candidates(top_k)
