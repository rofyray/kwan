"""AI-augmented prerequisite edges (PRD Phase 7).

The flow is: candidate pairs (`kwan.graph.candidates`) -> one question per pair to a decision model
-> keep the pairs at or above a threshold -> add them to the graph from the most likely down, never
closing a cycle and never repeating a short path that is already there -> write them to
`data/graph/math.augmented.json` as `pending`. A teacher then accepts or rejects each one
(`kwan.graph.review`); only accepted edges enter the shipped graph.

This module holds the parts that need nothing beyond the standard library and networkx: the
question, the provider protocol, the cache with its spend log, cycle-safe insertion, the proposed
edge file and the simulator comparison. Providers live in `kwan.graph.providers`.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import networkx as nx

from kwan import ids
from kwan._paths import data_dir
from kwan.graph.candidates import Candidate

QUESTION = "Must a pupil be able to do A before they can learn B?"
AUGMENTED_SCHEMA_VERSION = 1
MAX_DUPLICATE_PATH = 2
PRECISION_BAR = 0.80
STATUSES = ("pending", "accepted", "rejected")

__all__ = [
    "PRECISION_BAR",
    "QUESTION",
    "AugmentRun",
    "CacheMiss",
    "CachedDecisionModel",
    "DecisionModel",
    "InsertReport",
    "PairQuestion",
    "Proposal",
    "accepted_edges",
    "build_dir",
    "build_questions",
    "cycle_safe_insert",
    "default_augmented_path",
    "default_cache_path",
    "default_full_text_path",
    "impact",
    "load_augmented",
    "load_texts",
    "merge_accepted",
    "propose_edges",
    "render_prompt",
    "review_counts",
    "rule_graph",
    "save_augmented",
    "today",
]


# ---------------------------------------------------------------- paths


def build_dir() -> Path:
    """The git-ignored `build/` folder next to the data directory."""
    return data_dir().parent / "build"


def default_cache_path() -> Path:
    """`build/augment/cache.jsonl`: one line per decision-model call."""
    return build_dir() / "augment" / "cache.jsonl"


def default_augmented_path(subject: str = ids.DEFAULT_SUBJECT) -> Path:
    """`data/graph/<subject>.augmented.json`: the proposed edges and their review status."""
    return data_dir() / "graph" / f"{subject}.augmented.json"


def default_full_text_path(subject: str = ids.DEFAULT_SUBJECT) -> Path:
    """`build/curriculum/<subject>.full.json`: indicator records with the curriculum text (never committed)."""
    return build_dir() / "curriculum" / f"{subject}.full.json"


def today() -> str:
    """Today's date (UTC) as `YYYY-MM-DD`."""
    return dt.datetime.now(dt.UTC).date().isoformat()


# ---------------------------------------------------------------- the question and the provider protocol


@dataclass(frozen=True)
class PairQuestion:
    """One yes/no question about a pair of skills (PRD Appendix B.3).

    `a_text` and `b_text` are the descriptions (the short label when the curriculum text is not on
    this machine). The optional fields carry the rest of the input the PRD lists: short labels, one
    exemplar each, and the numbers the candidate stage worked out (`signals`, see
    `kwan.graph.candidates.Candidate`). A provider may ignore any optional field.
    """

    a_id: str
    a_text: str
    a_grade: int
    b_id: str
    b_text: str
    b_grade: int
    question: str = QUESTION
    a_label: str = ""
    b_label: str = ""
    a_exemplar: str = ""
    b_exemplar: str = ""
    signals: tuple[tuple[str, float], ...] = ()

    def signal(self, name: str, default: float = 0.0) -> float:
        """One candidate-stage number by name."""
        return dict(self.signals).get(name, default)


class DecisionModel(Protocol):
    """A provider: it answers a `PairQuestion` with a probability between 0 and 1.

    Optional attributes that `CachedDecisionModel` uses when they are there: `model` (the model id,
    part of the cache key), `licence`, `last_cost_usd` (cost of the latest call) and `prompt(q)` (the
    exact input, when it is not the default `render_prompt(q)`).
    """

    name: str

    def prerequisite_probability(self, q: PairQuestion) -> float: ...


def _skill_block(letter: str, id_: str, grade: int, label: str, text: str, exemplar: str) -> str:
    lines = [f"Skill {letter}: {ids.code_of(id_)} (Basic {grade})"]
    if label:
        lines.append(f"Short label: {label}")
    lines.append(f"Description: {text}")
    if exemplar:
        lines.append(f"Example activity: {exemplar}")
    return "\n".join(lines)


def render_prompt(q: PairQuestion) -> str:
    """The text sent to a chat model: both skills (code, grade, label, description, one exemplar) and the question."""
    return (
        "You are checking the order of skills in a primary school mathematics curriculum.\n\n"
        f"{_skill_block('A', q.a_id, q.a_grade, q.a_label, q.a_text, q.a_exemplar)}\n\n"
        f"{_skill_block('B', q.b_id, q.b_grade, q.b_label, q.b_text, q.b_exemplar)}\n\n"
        f"Question: {q.question}\n"
        'Answer with JSON only, in this form: {"probability": <a number from 0 to 1>}'
    )


# ---------------------------------------------------------------- cache and spend log


class CacheMiss(LookupError):
    """Raised in cache-only mode when a question has no stored answer."""


class CachedDecisionModel:
    """Wraps any provider and stores every answer in `build/augment/cache.jsonl`.

    The key is the sha256 of (provider, model, prompt), so a rerun with the same inputs calls nothing
    and costs nothing. Each line holds the provider, the model, the input hash and the output; the
    prompt itself is not stored. With `cache_only=True` a miss raises `CacheMiss` and the provider is
    never called, which is what the dry run in CI uses. `write_spend()` appends one summary line per
    run to `spend.jsonl` next to the cache.
    """

    def __init__(self, inner: DecisionModel, path: str | os.PathLike[str] | None = None, cache_only: bool = False):
        self.inner = inner
        self.name = inner.name
        self.model = str(getattr(inner, "model", "") or "")
        self.path = Path(path) if path is not None else default_cache_path()
        self.cache_only = cache_only
        self.hits = 0
        self.calls = 0
        self.cost_usd = 0.0
        self.seconds = 0.0
        self._store: dict[str, float] = {}
        if self.path.is_file():
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                        self._store[row["key"]] = float(row["probability"])
                    except (ValueError, KeyError, TypeError):
                        continue  # a half-written line from an interrupted run

    def prompt(self, q: PairQuestion) -> str:
        """The exact input for `q`: the provider's own prompt when it defines one."""
        own = getattr(self.inner, "prompt", None)
        return own(q) if callable(own) else render_prompt(q)

    def key(self, q: PairQuestion) -> str:
        """sha256 over provider, model and prompt."""
        payload = json.dumps([self.name, self.model, self.prompt(q)], ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def cached(self, q: PairQuestion) -> bool:
        """True when the answer to `q` is already stored."""
        return self.key(q) in self._store

    def prerequisite_probability(self, q: PairQuestion) -> float:
        key = self.key(q)
        if key in self._store:
            self.hits += 1
            return self._store[key]
        if self.cache_only:
            raise CacheMiss(f"no cached answer for {q.a_id} -> {q.b_id} ({self.name}/{self.model})")
        start = time.perf_counter()
        p = float(self.inner.prerequisite_probability(q))
        elapsed = time.perf_counter() - start
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"provider {self.name!r} returned {p!r}, not a probability")
        cost = float(getattr(self.inner, "last_cost_usd", 0.0) or 0.0)
        self.calls += 1
        self.cost_usd += cost
        self.seconds += elapsed
        self._store[key] = p
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "key": key,
            "provider": self.name,
            "model": self.model,
            "input_hash": hashlib.sha256(self.prompt(q).encode("utf-8")).hexdigest(),
            "probability": p,
            "cost_usd": cost,
            "seconds": round(elapsed, 6),
            "at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return p

    def spend(self) -> dict[str, Any]:
        """What this run cost so far: new calls, cache hits, dollars and compute seconds."""
        return {
            "provider": self.name,
            "model": self.model,
            "calls": self.calls,
            "cache_hits": self.hits,
            "cost_usd": round(self.cost_usd, 6),
            "compute_seconds": round(self.seconds, 3),
        }

    def write_spend(self, label: str = "", path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
        """Append this run's spend to `spend.jsonl` (next to the cache unless `path` is given)."""
        row = {"at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), "run": label, **self.spend()}
        out = Path(path) if path is not None else self.path.with_name("spend.jsonl")
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row


# ---------------------------------------------------------------- cycle-safe insertion (task 7.5)


@dataclass(frozen=True)
class Proposal:
    """One proposed edge `source -> target` with the model probability as its weight."""

    source: str
    target: str
    weight: float
    provenance: Mapping[str, Any] | None = None


@dataclass
class InsertReport:
    """What `cycle_safe_insert` did: the edges it added, and the ones it skipped with the reason."""

    inserted: list[Proposal] = field(default_factory=list)
    skipped: list[tuple[Proposal, str]] = field(default_factory=list)

    def reasons(self) -> dict[str, int]:
        """How many edges were skipped for each reason."""
        return dict(sorted(Counter(reason for _, reason in self.skipped).items()))


def _short_path(G: nx.DiGraph, a: Any, b: Any, max_len: int) -> bool:
    """True when a directed path from `a` to `b` of at most `max_len` edges exists."""
    frontier = {a}
    for _ in range(max_len):
        frontier = {y for x in frontier for y in G.successors(x)}
        if b in frontier:
            return True
        if not frontier:
            return False
    return False


def cycle_safe_insert(
    G: nx.DiGraph, proposals: Iterable[Proposal], max_duplicate_path: int = MAX_DUPLICATE_PATH
) -> InsertReport:
    """Add proposed edges to `G` in place, from the highest probability down.

    An edge is skipped when one of its ends is not in the graph (`unknown_node`), when it links a node
    to itself (`self_loop`), when a path from source to target of at most `max_duplicate_path` edges
    is already there (`duplicate_path`), or when a path from target back to source exists, so the edge
    would close a cycle (`cycle`). Ties in probability are broken by id, so the result does not
    depend on the order of the input. Added edges get `type="augmented"`, `weight` and `provenance`.
    """
    report = InsertReport()
    ordered = sorted(proposals, key=lambda p: (-p.weight, str(p.source), str(p.target)))
    for p in ordered:
        a, b = p.source, p.target
        if a not in G or b not in G:
            report.skipped.append((p, "unknown_node"))
        elif a == b:
            report.skipped.append((p, "self_loop"))
        elif _short_path(G, a, b, max_duplicate_path):
            report.skipped.append((p, "duplicate_path"))
        elif nx.has_path(G, b, a):
            report.skipped.append((p, "cycle"))
        else:
            G.add_edge(a, b, type="augmented", weight=float(p.weight), provenance=dict(p.provenance or {}))
            report.inserted.append(p)
    return report


# ---------------------------------------------------------------- the proposed edge file


def _status(edge: Mapping[str, Any]) -> str:
    prov = edge.get("provenance")
    review = prov.get("review") if isinstance(prov, Mapping) else None
    status = review.get("status") if isinstance(review, Mapping) else None
    return status if status in STATUSES else "pending"


def load_augmented(path: str | os.PathLike[str] | None = None, subject: str = ids.DEFAULT_SUBJECT) -> dict[str, Any]:
    """Read the proposed edge file. A missing file gives an empty document."""
    p = Path(path) if path is not None else default_augmented_path(subject)
    if not p.is_file():
        return {"schema_version": AUGMENTED_SCHEMA_VERSION, "subject": subject, "run": None, "edges": []}
    with open(p, encoding="utf-8") as f:
        doc = json.load(f)
    if isinstance(doc, list):  # a bare list of edges is accepted
        doc = {"schema_version": AUGMENTED_SCHEMA_VERSION, "subject": subject, "run": None, "edges": doc}
    if not isinstance(doc.get("edges"), list):
        raise ValueError(f"{p}: no 'edges' list")
    return doc


def save_augmented(doc: Mapping[str, Any], path: str | os.PathLike[str] | None = None) -> Path:
    """Write the proposed edge file: one edge per line, highest probability first, replaced in one step.

    The file holds ids, weights and provenance only. Any other key on an edge (for example a
    description) is dropped, so curriculum text cannot reach this committed file.
    """
    subject = doc.get("subject") or ids.DEFAULT_SUBJECT
    p = Path(path) if path is not None else default_augmented_path(subject)
    edges = []
    for e in doc.get("edges") or []:
        prov = e.get("provenance") or {}
        review = prov.get("review") or {}
        edges.append(
            {
                "source": e["source"],
                "target": e["target"],
                "type": "augmented",
                "weight": round(float(e["weight"]), 4),
                "provenance": {
                    "by": prov.get("by", "model:unknown"),
                    "at": prov.get("at"),
                    "review": {"status": _status(e), "by": review.get("by"), "at": review.get("at")},
                },
            }
        )
    edges.sort(key=lambda e: (-e["weight"], e["source"], e["target"]))
    head = {"schema_version": AUGMENTED_SCHEMA_VERSION, "subject": subject, "run": doc.get("run")}
    lines = ["{"] + [f' "{k}": {json.dumps(v, ensure_ascii=False)},' for k, v in head.items()]
    rows = ",\n".join("  " + json.dumps(e, ensure_ascii=False) for e in edges)
    lines.append(f' "edges": [\n{rows}\n ]' if edges else ' "edges": []')
    lines.append("}")
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(tmp, 0o644)
    os.replace(tmp, p)
    return p


def review_counts(doc: Mapping[str, Any]) -> dict[str, int]:
    """`{"pending": n, "accepted": n, "rejected": n}` for a proposed edge document."""
    counts = Counter(_status(e) for e in doc.get("edges") or [])
    return {s: counts.get(s, 0) for s in STATUSES}


def accepted_edges(doc: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The edges a reviewer has accepted."""
    return [e for e in doc.get("edges") or [] if _status(e) == "accepted"]


def rule_graph(G: nx.DiGraph) -> nx.DiGraph:
    """A copy of `G` without its augmented edges (the graph the rules alone give)."""
    H = G.copy()
    H.remove_edges_from([(a, b) for a, b, d in G.edges(data=True) if d.get("type") == "augmented"])
    return H


def merge_accepted(
    G: nx.DiGraph, augmented: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None = None
) -> tuple[nx.DiGraph, InsertReport]:
    """A copy of `G` plus the accepted edges of the proposed edge file, added by `cycle_safe_insert`.

    `augmented` is the document, a list of edge records, or None for the committed file. Ids are
    resolved (canonical id, bare code or alias). Pending and rejected edges are never added. This is
    the function to call when building the shipped graph.
    """
    if augmented is None:
        augmented = load_augmented(subject=G.graph.get("subject") or ids.DEFAULT_SUBJECT)
    edges = augmented.get("edges") or [] if isinstance(augmented, Mapping) else list(augmented)
    H = G.copy()
    resolver = ids.Resolver(H)
    proposals = []
    for e in edges:
        if _status(e) != "accepted":
            continue
        a = resolver.get(str(e.get("source"))) or str(e.get("source"))
        b = resolver.get(str(e.get("target"))) or str(e.get("target"))
        proposals.append(Proposal(a, b, float(e.get("weight", 0.0)), e.get("provenance")))
    return H, cycle_safe_insert(H, proposals)


# ---------------------------------------------------------------- texts and questions


def load_texts(path: str | os.PathLike[str] | None = None, subject: str = ids.DEFAULT_SUBJECT) -> dict[str, dict]:
    """`{bare code: {"description", "exemplar"}}` from the full-text file, or `{}` when it is not on this machine."""
    p = Path(path) if path is not None else default_full_text_path(subject)
    if not p.is_file():
        return {}
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, Mapping):
        data = next((data[k] for k in ("indicators", "records", "nodes") if isinstance(data.get(k), list)), [])
    out: dict[str, dict] = {}
    for rec in data:
        raw = rec.get("id") or rec.get("code")
        if not raw or not rec.get("description"):
            continue
        exemplars = rec.get("exemplars") or []
        first = exemplars[0] if exemplars else ""
        if isinstance(first, Mapping):
            first = first.get("text", "")
        entry = {"description": str(rec["description"]).strip(), "exemplar": " ".join(str(first).split())[:400]}
        for spelling in {ids.code_of(str(raw)), *(ids.code_of(a) for a in rec.get("aliases") or [] if a)}:
            out.setdefault(spelling, entry)
    return out


def node_texts(G: nx.DiGraph, texts: Mapping[str, Mapping[str, str]] | None = None) -> dict[str, dict[str, str]]:
    """Per node: `label`, `text` (description, or the label when there is no curriculum text) and `exemplar`."""
    texts = texts or {}
    out = {}
    for n, data in G.nodes(data=True):
        code = ids.code_of(str(n))
        label = str(data.get("short_label") or "")
        spellings = [code, *(ids.code_of(a) for a in data.get("aliases") or [] if isinstance(a, str) and a)]
        entry = next((texts[s] for s in spellings if s in texts), None)
        description = entry["description"] if entry else str(data.get("description") or label or f"Skill {code}")
        exemplar = entry.get("exemplar", "") if entry else ""
        if not exemplar and isinstance(data.get("exemplars"), list) and data["exemplars"]:
            first = data["exemplars"][0]
            exemplar = " ".join(str(first.get("text", "") if isinstance(first, Mapping) else first).split())[:400]
        out[n] = {"label": "" if label == description else label, "text": description, "exemplar": exemplar}
    return out


def build_questions(
    G: nx.DiGraph, candidates: Sequence[Candidate], texts: Mapping[str, Mapping[str, str]]
) -> list[PairQuestion]:
    """One `PairQuestion` per candidate. `texts` comes from `node_texts`."""
    out = []
    for c in candidates:
        a, b = texts[c.source], texts[c.target]
        out.append(
            PairQuestion(
                a_id=str(c.source),
                a_text=a["text"],
                a_grade=int(G.nodes[c.source].get("grade", 0)),
                b_id=str(c.target),
                b_text=b["text"],
                b_grade=int(G.nodes[c.target].get("grade", 0)),
                a_label=a.get("label", ""),
                b_label=b.get("label", ""),
                a_exemplar=a.get("exemplar", ""),
                b_exemplar=b.get("exemplar", ""),
                signals=c.signals,
            )
        )
    return out


# ---------------------------------------------------------------- the run (task 7.4)


@dataclass
class AugmentRun:
    """The outcome of one proposal run."""

    questions: int
    scored: list[tuple[PairQuestion, float]]
    above_threshold: int
    report: InsertReport
    edges: list[dict[str, Any]]
    threshold: float
    spend: dict[str, Any]


def propose_edges(
    G: nx.DiGraph,
    questions: Sequence[PairQuestion],
    model: DecisionModel,
    threshold: float,
    at: str | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> AugmentRun:
    """Ask the model every question, keep answers at or above `threshold`, and insert them cycle-safely.

    `G` is the graph the proposals must fit into (the rule edges, plus any edge already accepted); it
    is not changed. Every edge that survives comes back as a record with `type: "augmented"`,
    `weight` = probability and a `pending` review; nothing here can mark an edge accepted.
    """
    at = at or today()
    by = f"model:{model.name}/{getattr(model, 'model', '') or 'default'}"
    scored: list[tuple[PairQuestion, float]] = []
    for k, q in enumerate(questions, start=1):
        scored.append((q, float(model.prerequisite_probability(q))))
        if progress is not None:
            progress(k, len(questions))
    proposals = [
        Proposal(q.a_id, q.b_id, p, {"by": by, "at": at, "review": {"status": "pending", "by": None, "at": None}})
        for q, p in scored
        if p >= threshold
    ]
    H = G.copy()
    report = cycle_safe_insert(H, proposals)
    if not nx.is_directed_acyclic_graph(H):
        raise RuntimeError("the graph with the proposed edges is not a DAG")  # cannot happen; kept as a guard
    canon = {n: n if ids.is_namespaced(str(n)) else ids.make(G.graph.get("subject") or "math", str(n)) for n in G}
    edges = [
        {
            "source": canon[p.source],
            "target": canon[p.target],
            "type": "augmented",
            "weight": round(p.weight, 4),
            "provenance": dict(p.provenance or {}),
        }
        for p in report.inserted
    ]
    spend = model.spend() if isinstance(model, CachedDecisionModel) else {"provider": model.name}
    return AugmentRun(len(questions), scored, len(proposals), report, edges, threshold, spend)


def keep_reviews(new_edges: Sequence[Mapping[str, Any]], old_doc: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Carry review decisions over a rerun.

    An edge that a reviewer already accepted or rejected keeps its old record (weight, provenance and
    decision), whether or not the new run proposes it again. New proposals are pending.
    """
    decided = {(e["source"], e["target"]): dict(e) for e in old_doc.get("edges") or [] if _status(e) != "pending"}
    out = [dict(e) for e in new_edges if (e["source"], e["target"]) not in decided]
    return [*decided.values(), *out]


# ---------------------------------------------------------------- impact in the simulator (task 7.7)


def _means(result: Any) -> Any:
    """Turn a simulator result into plain numbers: `(mean, sd)` pairs become the mean."""
    if isinstance(result, Mapping):
        return {str(k): _means(v) for k, v in result.items()}
    if isinstance(result, tuple | list) and result and all(isinstance(x, int | float) for x in result):
        return round(float(result[0]), 3)
    if isinstance(result, int | float):
        return round(float(result), 3)
    return result


def _delta(a: Any, b: Any) -> Any:
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return {k: _delta(a[k], b[k]) for k in a if k in b}
    if isinstance(a, int | float) and isinstance(b, int | float):
        return round(b - a, 3)
    return None


def impact(
    G: nx.DiGraph,
    edges: Iterable[Mapping[str, Any]] | None = None,
    simulate: Callable[[nx.DiGraph], Any] | None = None,
    **sim_kwargs: Any,
) -> dict[str, Any]:
    """Run the simulator on the rule graph and on the rule graph plus accepted edges, and compare.

    `edges` defaults to the accepted edges of the committed proposed edge file. With no accepted edge
    there is nothing to compare, and the result says so (`status: "no accepted edges yet"`) without
    running anything. `simulate` defaults to `kwan.eval.simulate.compare_all`; extra keyword arguments
    go to it. Numbers are means of new skills mastered, by profile and policy.
    """
    base = rule_graph(G)
    if edges is None:
        edges = accepted_edges(load_augmented(subject=G.graph.get("subject") or ids.DEFAULT_SUBJECT))
    accepted = [{**e, "provenance": {**(e.get("provenance") or {}), "review": {"status": "accepted"}}} for e in edges]
    if not accepted:
        return {"status": "no accepted edges yet", "accepted_edges": 0, "merged_edges": 0}
    with_edges, report = merge_accepted(base, accepted)
    if not report.inserted:
        return {
            "status": "no accepted edge could be added",
            "accepted_edges": len(accepted),
            "merged_edges": 0,
            "skipped": report.reasons(),
        }
    if simulate is None:
        from kwan.eval import simulate as sim_module

        def simulate(graph: nx.DiGraph) -> Any:
            return sim_module.compare_all(graph, **sim_kwargs)

    without = _means(simulate(base))
    with_ = _means(simulate(with_edges))
    return {
        "status": "measured",
        "accepted_edges": len(accepted),
        "merged_edges": len(report.inserted),
        "skipped": report.reasons(),
        "without_accepted_edges": without,
        "with_accepted_edges": with_,
        "difference": _delta(without, with_),
    }
