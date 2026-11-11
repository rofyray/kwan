"""Validate the decision-model method on Junyi Academy (PRD Phase 7, task 7.3).

Junyi (Chang, Hsu and Chen, EDM 2015; non-commercial licence) gives two kinds of ground truth:

* the exercise table: each exercise names its prerequisite exercises (the parents on Junyi's
  knowledge map, set by its content experts);
* relationship annotations: for a sample of exercise pairs, mean opinion scores from 1 to 9. A row
  (Exercise_A, Exercise_B) scores how far B is a prerequisite of A, so the edge is B -> A.

The same pipeline that runs on the NaCCA graph runs here:

1. Rule edges first. On NaCCA the rules join the indicators of one content standard in order; the
   nearest thing on Junyi is a numbered series (`addition_1`, `addition_2`, ...), so consecutive
   members of a series get a rule edge. Pairs joined by a rule edge are not candidates, exactly as
   on NaCCA, and expert links that the rules already give are not counted for or against the model.
2. Candidates: for each target the 20 most similar eligible sources. Junyi has no grades, so every
   exercise has grade 0 and the grade window never excludes anything; Junyi's `area` plays the part
   of the strand, with the cross-area pairs listed under `junyi_2015` in the rules file.
3. The decision model answers each candidate, and a threshold picks the proposals.

Exercises are split by target into a training part and a held-out part. A provider that is fitted
(`local-encoder`) sees only training targets; its threshold is chosen on out-of-fold scores of the
training part; every reported number comes from the held-out targets. Everything fitted here is
written under `build/augment/` and is never committed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import networkx as nx

from kwan.graph.augment import PRECISION_BAR, DecisionModel, PairQuestion, build_dir
from kwan.graph.candidates import TOP_K, CandidateIndex, CrossStrandRules, Embedder

SUBJECT = "junyi"
RULES_SECTION = "junyi_2015"
TEST_SHARE = 0.30
MIN_SUPPORT = 20  # a threshold must leave at least this many proposals to count as an operating point
CITATION = "Chang, Hsu and Chen, Modeling Exercise Relationships in E-Learning: A Unified Approach, EDM 2015"
LICENCE = "non-commercial use only"
ANNOTATION_POSITIVE = 5.0  # mean opinion score on the 1 to 9 scale at or above the midpoint

__all__ = [
    "benchmark",
    "choose_threshold",
    "expert_links",
    "junyi_graph",
    "load_annotations",
    "load_exercises",
    "pr_table",
    "split_targets",
]

_SERIES = re.compile(r"^(.*?)[_ ]+(\d+(?:\.\d+)?)$")


def node_id(name: str) -> str:
    """The graph id of a Junyi exercise."""
    return f"{SUBJECT}:{name}"


def humanise(name: str) -> str:
    """`similar_triangles_1` -> `similar triangles 1`."""
    return " ".join(name.replace("_", " ").replace("-", " ").split())


def load_exercises(root: str | Path) -> list[dict[str, Any]]:
    """Rows of `junyi_Exercise_table.csv`: name, prerequisite names, topic and area."""
    out = []
    with open(Path(root) / "junyi_Exercise_table.csv", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            name = (row.get("name") or "").strip()
            if not name:
                continue
            pre = [p.strip() for p in (row.get("prerequisites") or "").split(",") if p.strip()]
            out.append(
                {
                    "name": name,
                    "prerequisites": pre,
                    "topic": (row.get("topic") or "").strip(),
                    "area": (row.get("area") or "").strip() or "N/A",
                }
            )
    return out


def load_annotations(root: str | Path) -> list[dict[str, Any]]:
    """Rows of both relationship annotation files as `{source, target, prerequisite_avg, raters, file}`.

    The source is `Exercise_B` and the target is `Exercise_A` (see the module docstring).
    """
    out = []
    for part in ("training", "testing"):
        path = Path(root) / f"relationship_annotation_{part}.csv"
        if not path.is_file():
            continue
        with open(path, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    avg = float(row["Prerequisite_avg"])
                except (KeyError, ValueError):
                    continue
                out.append(
                    {
                        "source": row["Exercise_B"].strip(),
                        "target": row["Exercise_A"].strip(),
                        "prerequisite_avg": avg,
                        "raters": len((row.get("Prerequisite_raw") or "").split("_")),
                        "file": part,
                    }
                )
    return out


def series_edges(names: Iterable[str]) -> list[tuple[str, str]]:
    """Rule edges of numbered series: `x_1 -> x_2 -> x_3` (and `x -> x_1` when the bare name exists)."""
    series: dict[str, list[tuple[float, str]]] = {}
    plain = set(names)
    for name in plain:
        m = _SERIES.match(name)
        if m:
            series.setdefault(m.group(1), []).append((float(m.group(2)), name))
    out = []
    for stem, members in series.items():
        ordered = [n for _, n in sorted(members)]
        if stem in plain:
            ordered.insert(0, stem)
        out.extend(zip(ordered, ordered[1:]))
    return sorted(out)


def junyi_graph(exercises: Sequence[Mapping[str, Any]], with_topic: bool = True) -> nx.DiGraph:
    """Junyi as a Kwan graph: one node per exercise (grade 0, strand = area) and the series rule edges."""
    G = nx.DiGraph(subject=SUBJECT, schema_version=2, text_mode="full")
    for ex in exercises:
        label = humanise(ex["name"])
        topic = humanise(ex["topic"])
        text = f"{label} ({topic})" if with_topic and topic else label
        G.add_node(
            node_id(ex["name"]),
            subject=SUBJECT,
            code=ex["name"],
            grade=0,
            strand=ex["area"],
            short_label=label,
            description=text,
            aliases=[ex["name"]],
        )
    for a, b in series_edges(ex["name"] for ex in exercises):
        G.add_edge(node_id(a), node_id(b), type="sequence", weight=1.0)
    return G


def expert_links(exercises: Sequence[Mapping[str, Any]]) -> set[tuple[str, str]]:
    """The expert prerequisite links `(source id, target id)` of the exercise table."""
    names = {ex["name"] for ex in exercises}
    return {(node_id(p), node_id(ex["name"])) for ex in exercises for p in ex["prerequisites"] if p in names}


def split_targets(names: Iterable[str], test_share: float = TEST_SHARE, seed: str = "kwan-junyi-v1") -> set[str]:
    """The held-out targets: a fixed pseudo-random share of the exercises, chosen by a hash of the name."""
    cut = int(test_share * 10_000)
    return {n for n in names if int(hashlib.sha256(f"{seed}|{n}".encode()).hexdigest(), 16) % 10_000 < cut}


# ---------------------------------------------------------------- scoring


def pr_table(
    scores: Sequence[float],
    labels: Sequence[int],
    total_positive: int,
    thresholds: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    """Precision and recall when everything at or above each threshold is proposed.

    `total_positive` is the number of true links recall is measured against (it may be larger than
    the number of positives among the scored pairs, because candidate selection can miss links).
    """
    if thresholds is None:
        thresholds = [round(0.05 * k, 2) for k in range(1, 20)]
    rows = []
    for t in thresholds:
        chosen = [lab for s, lab in zip(scores, labels) if s >= t]
        tp = sum(chosen)
        rows.append(
            {
                "threshold": t,
                "proposed": len(chosen),
                "correct": tp,
                "precision": round(tp / len(chosen), 4) if chosen else None,
                "recall": round(tp / total_positive, 4) if total_positive else None,
            }
        )
    return rows


def choose_threshold(
    scores: Sequence[float], labels: Sequence[int], bar: float = PRECISION_BAR, min_support: int = MIN_SUPPORT
) -> dict[str, Any]:
    """The lowest threshold whose precision is at or above `bar` with at least `min_support` proposals.

    Every distinct score is tried as a threshold. When no threshold reaches the bar, the result has
    `meets_bar: False` and gives the threshold with the best precision among those that still leave
    `min_support` proposals (ties go to the lower threshold, which keeps more recall).
    """
    ranked = sorted(zip(scores, labels), key=lambda t: -t[0])
    best: dict[str, Any] | None = None
    meets: dict[str, Any] | None = None
    tp = 0
    for k, (s, lab) in enumerate(ranked, start=1):
        tp += lab
        if k < len(ranked) and ranked[k][0] == s:
            continue  # same score as the next pair: the threshold cannot fall between them
        if k < min_support:
            continue
        precision = tp / k
        point = {"threshold": float(s), "proposed": k, "correct": tp, "precision": round(precision, 4)}
        if precision >= bar:
            meets = point  # later (lower) thresholds overwrite: we want the lowest one that still meets the bar
        if best is None or precision >= best["precision"] - 1e-12:
            best = point  # thresholds fall as we go, so a tie keeps the lower one
    if meets is not None:
        return {**meets, "meets_bar": True, "bar": bar, "min_support": min_support}
    if best is None:
        return {
            "threshold": 1.0,
            "proposed": 0,
            "correct": 0,
            "precision": None,
            "meets_bar": False,
            "bar": bar,
            "min_support": min_support,
        }
    return {**best, "meets_bar": False, "bar": bar, "min_support": min_support}


def _average_precision(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    ranked = sorted(zip(scores, labels), key=lambda t: -t[0])
    total = sum(labels)
    if not total:
        return None
    tp, acc = 0, 0.0
    for k, (_, lab) in enumerate(ranked, start=1):
        if lab:
            tp += 1
            acc += tp / k
    return round(acc / total, 4)


def _spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    def ranks(v: Sequence[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        out = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                out[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return out

    if len(x) < 3:
        return None
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx, vy = sum((a - mx) ** 2 for a in rx), sum((b - my) ** 2 for b in ry)
    return round(cov / (vx * vy) ** 0.5, 4) if vx > 0 and vy > 0 else None


def _reliability(scores: Sequence[float], labels: Sequence[int], bins: int = 5) -> list[dict[str, Any]]:
    """Mean predicted probability against the observed share of true links, in equal-count bins."""
    ranked = sorted(zip(scores, labels))
    out = []
    for k in range(bins):
        part = ranked[k * len(ranked) // bins : (k + 1) * len(ranked) // bins]
        if part:
            out.append(
                {
                    "pairs": len(part),
                    "mean_probability": round(sum(s for s, _ in part) / len(part), 4),
                    "observed": round(sum(lab for _, lab in part) / len(part), 4),
                }
            )
    return out


def _question(G: nx.DiGraph, a: str, b: str, signals: tuple[tuple[str, float], ...]) -> PairQuestion:
    da, db = G.nodes[a], G.nodes[b]
    return PairQuestion(
        a_id=a,
        a_text=da["description"],
        a_grade=0,
        b_id=b,
        b_text=db["description"],
        b_grade=0,
        a_label=da["short_label"],
        b_label=db["short_label"],
        signals=signals,
    )


def benchmark(
    root: str | Path,
    embedder: Embedder,
    rules: CrossStrandRules,
    provider: str = "local-encoder",
    model: DecisionModel | None = None,
    top_k: int = TOP_K,
    out_dir: str | Path | None = None,
    with_topic: bool = True,
    features: Sequence[str] | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Run the pipeline on Junyi and score it against the expert links and the annotations.

    With `provider="local-encoder"` and no `model`, the classifier is fitted on the training targets
    and saved (with the chosen threshold) to `<out_dir>/local_encoder.json`. With a ready `model`
    (for example a hosted provider) nothing is fitted: the model scores the training targets, where
    the threshold is chosen, and the held-out targets, where the numbers are reported.

    Returns the report, also written to `<out_dir>/junyi_report.json`. `out_dir` defaults to
    `build/augment/`.
    """
    start = time.perf_counter()
    out = Path(out_dir) if out_dir is not None else build_dir() / "augment"
    exercises = load_exercises(root)
    G = junyi_graph(exercises, with_topic=with_topic)
    links = expert_links(exercises)
    rule = set(G.edges())
    truth = links - rule  # links the rules do not already give; these are what the model is scored on
    nodes = {n: {"grade": 0, "strand": d["strand"]} for n, d in G.nodes(data=True)}
    texts = {n: d["description"] for n, d in G.nodes(data=True)}
    index = CandidateIndex(nodes, texts, embedder, rules, connected=rule)
    candidates = index.candidates(top_k)
    test_targets = {node_id(n) for n in split_targets(ex["name"] for ex in exercises)}
    train = [c for c in candidates if c.target not in test_targets]
    test = [c for c in candidates if c.target in test_targets]
    q_train = [_question(G, c.source, c.target, c.signals) for c in train]
    q_test = [_question(G, c.source, c.target, c.signals) for c in test]
    y_train = [int((c.source, c.target) in truth) for c in train]
    y_test = [int((c.source, c.target) in truth) for c in test]

    fitted = None
    if model is None:
        if provider != "local-encoder":
            raise ValueError(f"provider {provider!r} needs a ready model")
        from kwan.graph.providers import local_encoder as le

        X = [le.pair_features(q) for q in q_train]
        fitted, p_train = le.fit_artifact(X, y_train, [c.target for c in train], features=features or le.FEATURES)
        model = le.LocalEncoderDecisionModel(fitted, embedder=embedder)
    else:
        p_train = [float(model.prerequisite_probability(q)) for q in q_train]
    p_test = [float(model.prerequisite_probability(q)) for q in q_test]

    choice = choose_threshold(p_train, y_train)
    t = choice["threshold"]
    truth_test = {(a, b) for a, b in truth if b in test_targets}
    truth_train = truth - truth_test
    chosen = [(c, lab) for c, p, lab in zip(test, p_test, y_test) if p >= t]
    tp = sum(lab for _, lab in chosen)
    either = sum(1 for c, _ in chosen if (c.source, c.target) in links or (c.target, c.source) in links)
    expert_map = nx.DiGraph(sorted(links))
    ancestors = {n: nx.ancestors(expert_map, n) for n in expert_map}
    transitive = sum(1 for c, _ in chosen if c.source in ancestors.get(c.target, ()))
    held_out = {
        "threshold": round(t, 4),
        "proposed": len(chosen),
        "correct": tp,
        "precision": round(tp / len(chosen), 4) if chosen else None,
        "recall": round(tp / len(truth_test), 4) if truth_test else None,
        "precision_transitive": round(transitive / len(chosen), 4) if chosen else None,
        "precision_ignoring_direction": round(either / len(chosen), 4) if chosen else None,
        "meets_precision_bar": bool(chosen) and tp / len(chosen) >= PRECISION_BAR,
        "note": "precision counts direct expert links only; precision_transitive also counts a source that is an "
        "earlier prerequisite of the target on the expert map (a true prerequisite, not the immediate one)",
    }
    thresholds = sorted({round(0.05 * k, 2) for k in range(1, 20)} | {round(t, 4)})
    best_test = choose_threshold(p_test, y_test)

    # the relationship annotations for held-out targets, scored with the same model
    ann_rows = []
    for row in load_annotations(root):
        a, b = node_id(row["source"]), node_id(row["target"])
        if a not in G or b not in G or a == b or b not in test_targets or (a, b) in rule or (b, a) in rule:
            continue
        p = float(model.prerequisite_probability(_question(G, a, b, index.signals(a, b))))
        ann_rows.append((p, row["prerequisite_avg"]))
    ann_chosen = [avg for p, avg in ann_rows if p >= t]
    annotations = {
        "pairs_scored": len(ann_rows),
        "direction": "Exercise_B -> Exercise_A",
        "spearman_probability_vs_mean_opinion": _spearman([p for p, _ in ann_rows], [avg for _, avg in ann_rows]),
        "mean_opinion_all_pairs": round(sum(avg for _, avg in ann_rows) / len(ann_rows), 3) if ann_rows else None,
        "at_threshold": {
            "proposed": len(ann_chosen),
            "mean_opinion": round(sum(ann_chosen) / len(ann_chosen), 3) if ann_chosen else None,
            f"share_at_or_above_{ANNOTATION_POSITIVE:g}": round(
                sum(avg >= ANNOTATION_POSITIVE for avg in ann_chosen) / len(ann_chosen), 4
            )
            if ann_chosen
            else None,
        },
        f"base_share_at_or_above_{ANNOTATION_POSITIVE:g}": round(
            sum(avg >= ANNOTATION_POSITIVE for _, avg in ann_rows) / len(ann_rows), 4
        )
        if ann_rows
        else None,
    }

    seconds = time.perf_counter() - start
    report: dict[str, Any] = {
        "dataset": "Junyi Academy 2015 (exercise table and relationship annotations)",
        "licence": LICENCE,
        "cite": CITATION,
        "provider": model.name,
        "model": getattr(model, "model", ""),
        "model_licence": getattr(model, "licence", ""),
        "embedder": embedder.name,
        "text": "exercise name and topic" if with_topic else "exercise name",
        "exercises": len(exercises),
        "expert_links": len(links),
        "rule_edges_series": len(rule),
        "expert_links_already_given_by_rules": len(links & rule),
        "expert_links_scored": len(truth),
        "split": f"by target exercise, {round(100 * TEST_SHARE)}% held out (hash of the exercise name)",
        "targets": {"train": G.number_of_nodes() - len(test_targets), "held_out": len(test_targets)},
        "candidates_per_target": top_k,
        "candidate_pairs": {"train": len(train), "held_out": len(test)},
        "candidate_positives": {"train": sum(y_train), "held_out": sum(y_test)},
        "expert_links_scored_by_split": {"train": len(truth_train), "held_out": len(truth_test)},
        "candidate_recall_ceiling_held_out": round(sum(y_test) / len(truth_test), 4) if truth_test else None,
        "threshold_choice": {
            "where": "out-of-fold scores of the training targets"
            if fitted is not None
            else "scores of the training targets",
            "rule": f"lowest threshold with precision >= {PRECISION_BAR} and at least {MIN_SUPPORT} proposals; "
            "if none, the threshold with the best precision that leaves that many proposals",
            **{k: v for k, v in choice.items() if k not in ("bar", "min_support")},
        },
        "held_out": held_out,
        "held_out_average_precision": _average_precision(p_test, y_test),
        "held_out_curve": pr_table(p_test, y_test, len(truth_test), thresholds),
        "held_out_best_precision_point": {k: v for k, v in best_test.items() if k not in ("bar", "min_support")},
        "held_out_calibration": _reliability(p_test, y_test),
        "held_out_top": [
            {"top": k, "precision": round(sum(lab for _, lab in ranked_k) / len(ranked_k), 4)}
            for k in (10, 20, 50, 100, 200)
            for ranked_k in [sorted(zip(p_test, y_test), key=lambda z: -z[0])[:k]]
            if len(p_test) >= k
        ],
        "annotations_held_out": annotations,
        "precision_bar": PRECISION_BAR,
        "cost_usd": 0.0 if fitted is not None else None,
        "compute_seconds": round(seconds, 1),
    }
    if fitted is not None:
        fitted = {
            **fitted,
            "threshold": round(t, 6),
            "meets_precision_bar": held_out["meets_precision_bar"],
            "fitted_on": "Junyi 2015 training targets (non-commercial; do not commit or ship this file)",
        }
        report["classifier"] = {
            "features": fitted["features"],
            "coef_standardised": dict(zip(fitted["features"], fitted["coef"])),
            "platt": fitted["platt"],
        }
    if write:
        out.mkdir(parents=True, exist_ok=True)
        if fitted is not None:
            (out / "local_encoder.json").write_text(json.dumps(fitted, indent=1) + "\n", encoding="utf-8")
        (out / "junyi_report.json").write_text(
            json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return report
