"""`kwan graph augment`, `kwan graph review-export` and `kwan graph review-import` (PRD Phase 7).

`kwan.graph._cli` calls `register_into(graph_subparsers)`. Nothing heavy is imported at module level:
each handler imports what it needs, so `--dry-run` works with the standard install.

    kwan graph augment --dry-run            walk the whole pipeline with no model and no writes (CI)
    kwan graph augment --junyi raw/junyi    check the method on Junyi; fits the local model, picks the threshold
    kwan graph augment                      propose edges for the curriculum graph (all stay pending)
    kwan graph augment --impact             simulator with and without the accepted edges
    kwan graph review-export                write the sheet for teachers (under build/)
    kwan graph review-import SHEET.csv      record the teachers' decisions
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

EXTRA = "graph-ai"

__all__ = ["main", "register_into", "results_block"]


# ---------------------------------------------------------------- helpers


def _load_graph(path: str | None) -> Any:
    from kwan.graph import schema

    return schema.load_graph(path)


def _run_file() -> Path:
    from kwan.graph.augment import build_dir

    return build_dir() / "augment" / "nacca_run.json"


def _junyi_file() -> Path:
    from kwan.graph.augment import build_dir

    return build_dir() / "augment" / "junyi_report.json"


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _embedder(kind: str) -> Any:
    from kwan.graph import candidates

    return candidates.MiniLMEmbedder() if kind == "minilm" else candidates.HashingEmbedder()


def _candidates(G: Any, texts: dict[str, dict[str, str]], embedder: Any, rules_path: str | None, top_k: int) -> list:
    from kwan.graph import candidates

    rules = candidates.load_cross_strand_rules(rules_path, section=G.graph.get("subject") or "math")
    nodes = {n: {"grade": d["grade"], "strand": d["strand"]} for n, d in G.nodes(data=True)}
    connected = [(a, b) for a, b, d in G.edges(data=True) if d.get("type") != "augmented"]
    return candidates.candidate_pairs(
        nodes, {n: t["text"] for n, t in texts.items()}, embedder, rules, connected, top_k
    )


def check_augmented(G: Any, doc: dict[str, Any]) -> list[str]:
    """Problems in a proposed edge document: unknown ids, bad weights or statuses, text keys, cycles.

    The cycle check covers the accepted edges, because those are the ones that enter the shipped
    graph. A pending edge may conflict with an accepted one; `review-import` refuses it if a
    reviewer later tries to accept it.
    """
    import networkx as nx

    from kwan import ids
    from kwan.graph.augment import STATUSES, rule_graph

    problems = []
    resolver = ids.Resolver(G)
    seen: set[tuple[str, str]] = set()
    accepted: list[tuple[float, str, str, str]] = []
    for k, e in enumerate(doc.get("edges") or []):
        where = f"edge {k + 1} ({e.get('source')} -> {e.get('target')})"
        extra = sorted(set(e) - {"source", "target", "type", "weight", "provenance"})
        if extra:
            problems.append(f"{where}: unexpected keys {extra}; the file holds ids, weights and provenance only")
        a, b = resolver.get(str(e.get("source"))), resolver.get(str(e.get("target")))
        if a is None or b is None:
            problems.append(f"{where}: unknown indicator")
            continue
        if (a, b) in seen:
            problems.append(f"{where}: listed twice")
        seen.add((a, b))
        if e.get("type") != "augmented":
            problems.append(f"{where}: type is {e.get('type')!r}, not 'augmented'")
        w = e.get("weight")
        if isinstance(w, bool) or not isinstance(w, int | float) or not 0.0 <= w <= 1.0:
            problems.append(f"{where}: weight {w!r} is not a probability")
            w = 0.0
        prov = e.get("provenance") or {}
        review = prov.get("review") or {}
        if not str(prov.get("by", "")).startswith("model:"):
            problems.append(f"{where}: provenance.by must start with 'model:'")
        if review.get("status") not in STATUSES:
            problems.append(f"{where}: review status {review.get('status')!r}")
        elif review["status"] != "pending" and not review.get("by"):
            problems.append(f"{where}: {review['status']} without a reviewer id")
        if review.get("status") == "accepted":
            accepted.append((-float(w), a, b, where))
    H = rule_graph(G)
    for _, a, b, where in sorted(accepted):
        if a == b or nx.has_path(H, b, a):
            problems.append(f"{where}: accepted, but it closes a cycle")
        else:
            H.add_edge(a, b)
    return problems


def _strand_pairs(G: Any, edges: list[dict[str, Any]]) -> dict[str, int]:
    from kwan import ids

    resolver = ids.Resolver(G)
    counts: Counter[str] = Counter()
    for e in edges:
        a, b = resolver.get(e["source"]), resolver.get(e["target"])
        if a is None or b is None:
            continue
        na, nb = G.nodes[a], G.nodes[b]
        counts[f"{na.get('strand_name') or na.get('strand')} -> {nb.get('strand_name') or nb.get('strand')}"] += 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def results_block(G: Any, doc: dict[str, Any], impact: dict[str, Any] | None = None) -> dict[str, Any]:
    """The `graph_augment` block of `data/eval_results.json`.

    It is put together from the Junyi report and the run summary under `build/augment/` and from the
    committed proposed edge file. Teacher acceptance is reported from the review records only.
    """
    from kwan.graph import augment, review

    junyi = _read_json(_junyi_file())
    run = doc.get("run") or {}
    local = _read_json(_run_file()) or {}
    edges = doc.get("edges") or []
    block: dict[str, Any] = {
        "question": augment.QUESTION,
        "provider": run.get("provider"),
        "model": run.get("model"),
        "model_licence": run.get("licence"),
        "threshold": run.get("threshold"),
        "precision_bar": augment.PRECISION_BAR,
        "meets_precision_bar": bool(junyi and junyi.get("held_out", {}).get("meets_precision_bar")),
        "junyi_benchmark": None,
        "nacca": {
            "run_at": run.get("at"),
            "text": run.get("text"),
            "graph_nodes": run.get("graph_nodes"),
            "candidate_pairs": run.get("candidate_pairs"),
            "at_or_above_threshold": run.get("at_or_above_threshold"),
            "skipped_at_insertion": run.get("skipped_at_insertion"),
            "proposed_edges": len(edges),
            "proposed_edges_by_strand_pair": _strand_pairs(G, edges),
            "below_acceptance_bar": run.get("below_acceptance_bar"),
            "acyclic": not any("cycle" in p for p in check_augmented(G, doc)),
        },
        "review": {**augment.review_counts(doc), **review.acceptance(doc)},
        "impact_in_simulator": impact or augment.impact(G, augment.accepted_edges(doc)),
        "spend": local.get("spend") or run.get("spend"),
    }
    if junyi:
        keep = (
            "dataset",
            "licence",
            "cite",
            "provider",
            "model",
            "embedder",
            "text",
            "exercises",
            "expert_links",
            "rule_edges_series",
            "expert_links_already_given_by_rules",
            "expert_links_scored",
            "split",
            "targets",
            "candidates_per_target",
            "candidate_pairs",
            "expert_links_scored_by_split",
            "candidate_recall_ceiling_held_out",
            "threshold_choice",
            "held_out",
            "held_out_average_precision",
            "held_out_curve",
            "held_out_best_precision_point",
            "held_out_top",
            "annotations_held_out",
            "cost_usd",
            "compute_seconds",
        )
        block["junyi_benchmark"] = {k: junyi[k] for k in keep if k in junyi}
        same = junyi.get("provider") == run.get("provider") and junyi.get("model") == run.get("model")
        block["junyi_benchmark"]["same_model_as_nacca_run"] = same
    return block


# ---------------------------------------------------------------- augment


def _dry_run(args: argparse.Namespace) -> int:
    from kwan.graph import augment
    from kwan.graph.providers import get_provider

    G = _load_graph(args.graph)
    doc = augment.load_augmented(args.out, subject=G.graph.get("subject") or "math")
    texts = augment.node_texts(G, augment.load_texts(args.full_text))
    embedder = _embedder(args.embedder or "hashing")
    candidates = _candidates(G, texts, embedder, args.rules, args.top_k)
    questions = augment.build_questions(G, candidates, texts)
    print("Dry run: nothing is written and no model is called.")
    print(f"graph: {G.number_of_nodes()} indicators, text mode {G.graph.get('text_mode')}")
    print(f"candidate pairs: {len(questions)} (top {args.top_k} per target, embedder {embedder.name})")
    provider = args.provider or "fake"
    if provider != "fake":
        try:
            real = augment.CachedDecisionModel(get_provider(provider), args.cache, cache_only=True)
            hits = sum(real.cached(q) for q in questions)
            print(
                f"provider {provider}: {hits} answers in the cache, {len(questions) - hits} calls a real run would make"
            )
        except (RuntimeError, FileNotFoundError, ModuleNotFoundError) as exc:
            print(f"provider {provider} is not set up on this machine ({exc})")
        print("walking the rest of the pipeline with the fake provider:")
    threshold = args.threshold if args.threshold is not None else 0.9
    run = augment.propose_edges(augment.rule_graph(G), questions, get_provider("fake"), threshold)
    print(
        f"fake provider at threshold {threshold:g}: {run.above_threshold} pairs at or above it, "
        f"{len(run.edges)} would be proposed, skipped {run.report.reasons() or 'none'}"
    )
    counts = augment.review_counts(doc)
    print(
        f"committed proposed edges: {len(doc.get('edges') or [])} "
        f"(pending {counts['pending']}, accepted {counts['accepted']}, rejected {counts['rejected']})"
    )
    problems = check_augmented(G, doc)
    for p in problems[:20]:
        print(f"  problem: {p}", file=sys.stderr)
    if problems:
        print(f"{len(problems)} problems in the proposed edge file", file=sys.stderr)
        return 1
    print("proposed edge file: ids resolve, weights and review records are valid, no cycles")
    return 0


def _junyi(args: argparse.Namespace) -> int:
    from kwan.graph import augment, candidates, junyi
    from kwan.graph.providers import get_provider

    provider = args.provider or "local-encoder"
    embedder = _embedder(args.embedder or "minilm")
    rules = candidates.load_cross_strand_rules(args.rules, section=junyi.RULES_SECTION)
    model = None
    if provider != "local-encoder":
        model = augment.CachedDecisionModel(get_provider(provider), args.cache)
    report = junyi.benchmark(args.junyi, embedder, rules, provider=provider, model=model, top_k=args.top_k)
    if isinstance(model, augment.CachedDecisionModel):
        report["spend"] = model.write_spend("junyi")
        report["cost_usd"] = report["spend"]["cost_usd"]
        _junyi_file().write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    held = report["held_out"]
    print(f"Junyi: {report['exercises']} exercises, {report['expert_links_scored']} expert links scored")
    print(f"provider {report['provider']} / {report['model']}")
    print(
        f"held-out targets: {report['targets']['held_out']}, candidate pairs: {report['candidate_pairs']['held_out']}"
    )
    print("threshold  proposed  correct  precision  recall")
    for row in report["held_out_curve"]:
        print(
            f"{row['threshold']:9.2f}  {row['proposed']:8d}  {row['correct']:7d}  "
            f"{row['precision'] if row['precision'] is not None else 'n/a':>9}  {row['recall']}"
        )
    print(
        f"chosen threshold {held['threshold']}: precision {held['precision']}, recall {held['recall']}, "
        f"{held['proposed']} proposals on held-out targets"
    )
    if not held["meets_precision_bar"]:
        print(f"BELOW THE ACCEPTANCE BAR: held-out precision is under {augment.PRECISION_BAR}.")
    print(f"cost: {report.get('cost_usd')} USD, {report['compute_seconds']} seconds; report: {_junyi_file()}")
    return 0


def _impact(args: argparse.Namespace) -> int:
    from kwan.eval import results
    from kwan.graph import augment

    G = _load_graph(args.graph)
    doc = augment.load_augmented(args.out, subject=G.graph.get("subject") or "math")
    outcome = augment.impact(G, augment.accepted_edges(doc))
    print(json.dumps(outcome, indent=1))
    if args.results:
        results.update("graph_augment", results_block(G, doc, impact=outcome), path=args.results)
    return 0


def _augment(args: argparse.Namespace) -> int:
    if args.dry_run:
        return _dry_run(args)
    if args.junyi:
        return _junyi(args)
    if args.impact:
        return _impact(args)
    from kwan.graph import augment
    from kwan.graph.providers import get_provider

    G = _load_graph(args.graph)
    subject = G.graph.get("subject") or "math"
    provider = args.provider or "local-encoder"
    if provider == "fake":
        print("The fake provider is for tests and --dry-run; it cannot propose real edges.", file=sys.stderr)
        return 2
    inner = get_provider(provider)
    model = augment.CachedDecisionModel(inner, args.cache)
    threshold = args.threshold if args.threshold is not None else getattr(inner, "threshold", None)
    if threshold is None:
        print("No threshold: run the Junyi check first (--junyi DIR) or pass --threshold.", file=sys.stderr)
        return 2
    junyi = _read_json(_junyi_file()) or {}
    benchmarked = junyi.get("provider") == model.name and junyi.get("model") == model.model
    meets = benchmarked and bool(junyi.get("held_out", {}).get("meets_precision_bar"))
    full = augment.load_texts(args.full_text, subject=subject)
    texts = augment.node_texts(G, full)
    has_text = bool(full) or G.graph.get("text_mode") == "full"
    if not has_text:
        print("The curriculum descriptions are not on this machine; the model will see short labels only.")
    embedder = _embedder(args.embedder or "minilm")
    candidates = _candidates(G, texts, embedder, args.rules, args.top_k)
    questions = augment.build_questions(G, candidates, texts)
    old = augment.load_augmented(args.out, subject=subject)
    base, _ = augment.merge_accepted(augment.rule_graph(G), old)  # new proposals must fit the edges already accepted
    run = augment.propose_edges(base, questions, model, float(threshold))
    spend = model.write_spend(f"{subject} proposals")
    doc = {
        "schema_version": augment.AUGMENTED_SCHEMA_VERSION,
        "subject": subject,
        "run": {
            "provider": model.name,
            "model": model.model,
            "licence": getattr(inner, "licence", None),
            "embedder": embedder.name,
            "at": augment.today(),
            "question": augment.QUESTION,
            "threshold": round(float(threshold), 4),
            "text": "descriptions and one exemplar" if has_text else "short labels only",
            "graph_nodes": G.number_of_nodes(),
            "candidate_pairs": len(questions),
            "at_or_above_threshold": run.above_threshold,
            "skipped_at_insertion": run.report.reasons(),
            "benchmarked_on_junyi": benchmarked,
            "below_acceptance_bar": not meets,
            "review_note": "Every edge is a proposal. Only a teacher's review can mark an edge accepted.",
        },
        "edges": augment.keep_reviews(run.edges, old),
    }
    path = augment.save_augmented(doc, args.out)
    doc = augment.load_augmented(path, subject=subject)
    _run_file().parent.mkdir(parents=True, exist_ok=True)
    scores = sorted((round(p, 4) for _, p in run.scored), reverse=True)
    _run_file().write_text(
        json.dumps({"run": doc["run"], "spend": spend, "top_scores": scores[:50]}, indent=1) + "\n", encoding="utf-8"
    )
    counts = augment.review_counts(doc)
    print(f"candidate pairs: {len(questions)}; at or above threshold {threshold:g}: {run.above_threshold}")
    print(f"skipped at insertion: {run.report.reasons() or 'none'}")
    print(f"proposed edges written to {path}: {len(doc['edges'])} (pending {counts['pending']})")
    print(f"spend: {spend['cost_usd']} USD, {spend['calls']} calls, {spend['cache_hits']} cache hits")
    if not meets:
        why = "its Junyi precision is under the bar" if benchmarked else "this model has no Junyi check on this machine"
        print(f"BELOW THE ACCEPTANCE BAR: {why}. Treat every edge as a suggestion for teachers to check.")
    if args.results:
        from kwan.eval import results

        results.update("graph_augment", results_block(G, doc), path=args.results)
        print(f"graph_augment block written to {args.results}")
    return 0


# ---------------------------------------------------------------- review


def _review_export(args: argparse.Namespace) -> int:
    from kwan.graph import augment, review

    G = _load_graph(args.graph)
    doc = augment.load_augmented(args.augmented, subject=G.graph.get("subject") or "math")
    texts = augment.load_texts(args.full_text, subject=G.graph.get("subject") or "math")
    path, rows, has_text = review.export_review(
        G, doc, out=args.out, texts=texts, status=args.status, sample=args.sample, seed=args.seed
    )
    print(f"{rows} edges written to {path}")
    if has_text:
        print("The sheet holds curriculum descriptions. Share it with reviewers, do not commit it.")
    else:
        print("The curriculum descriptions are not on this machine, so the sheet shows short labels only.")
    print(f"For each row: {review.QUESTION_FOR_REVIEWER}")
    return 0


def _review_import(args: argparse.Namespace) -> int:
    from kwan.graph import augment, review

    G = _load_graph(args.graph)
    doc = augment.load_augmented(args.augmented, subject=G.graph.get("subject") or "math")
    try:
        new_doc, report = review.import_review(G, doc, args.sheet, reviewer=args.reviewer, at=args.date)
    except review.ReviewError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    path = augment.save_augmented(new_doc, args.augmented)
    print(
        f"recorded: {report.accepted} accepted, {report.rejected} rejected; "
        f"{report.blank} rows left blank, {report.unchanged} unchanged"
    )
    print(f"{path}: {report.counts}")
    for row in report.refused:
        print(f"refused: {row['source']} -> {row['target']}: {row['reason']}", file=sys.stderr)
    return 1 if report.refused else 0


# ---------------------------------------------------------------- registration


def register_into(graph_subparsers: Any) -> None:
    """Add `augment`, `review-export` and `review-import` under `kwan graph`."""
    p = graph_subparsers.add_parser(
        "augment", help="propose extra prerequisite edges with a decision model (all stay pending until reviewed)"
    )
    p.add_argument("--dry-run", action="store_true", help="walk the pipeline with no model call and no write")
    p.add_argument("--junyi", metavar="DIR", help="check the method on the Junyi files in DIR instead (e.g. raw/junyi)")
    p.add_argument("--impact", action="store_true", help="run the simulator with and without the accepted edges")
    p.add_argument("--provider", choices=("fake", "local-encoder", "llm-json", "typesafe-jev"), help="decision model")
    p.add_argument("--threshold", type=float, metavar="P", help="propose pairs at or above this probability")
    p.add_argument("--embedder", choices=("minilm", "hashing"), help="similarity model for the candidate pairs")
    p.add_argument("--top-k", type=int, default=20, metavar="N", help="candidate sources kept per target")
    p.add_argument("--graph", metavar="PATH", help="curriculum graph (default: data/graph/math.graph.json)")
    p.add_argument("--full-text", metavar="PATH", help="full-text records (default: build/curriculum/math.full.json)")
    p.add_argument("--rules", metavar="PATH", help="cross-strand rules (default: data/graph/cross_strand_rules.yaml)")
    p.add_argument("--cache", metavar="PATH", help="answer cache (default: build/augment/cache.jsonl)")
    p.add_argument("--out", metavar="PATH", help="proposed edge file (default: data/graph/math.augmented.json)")
    p.add_argument("--results", metavar="PATH", help="write the graph_augment block to this results file")
    p.set_defaults(func=_augment, extra=EXTRA)

    p = graph_subparsers.add_parser("review-export", help="write the sheet of proposed edges for teachers to review")
    p.add_argument("--out", metavar="PATH", help="sheet to write (default: build/review/math.augmented.review.csv)")
    p.add_argument("--status", default="pending", choices=("pending", "accepted", "rejected", "all"))
    p.add_argument("--sample", type=int, metavar="N", help="a random sample of N edges, e.g. 100")
    p.add_argument("--seed", type=int, default=0, help="seed of the random sample")
    p.add_argument("--graph", metavar="PATH", help="curriculum graph (default: data/graph/math.graph.json)")
    p.add_argument("--augmented", metavar="PATH", help="proposed edge file (default: data/graph/math.augmented.json)")
    p.add_argument("--full-text", metavar="PATH", help="full-text records (default: build/curriculum/math.full.json)")
    p.set_defaults(func=_review_export, extra=None)

    p = graph_subparsers.add_parser("review-import", help="record the decisions of a filled review sheet")
    p.add_argument("sheet", help="the filled CSV sheet")
    p.add_argument("--reviewer", metavar="ID", help="reviewer id for rows whose reviewer cell is empty")
    p.add_argument("--date", metavar="YYYY-MM-DD", help="review date (default: today)")
    p.add_argument("--graph", metavar="PATH", help="curriculum graph (default: data/graph/math.graph.json)")
    p.add_argument("--augmented", metavar="PATH", help="proposed edge file (default: data/graph/math.augmented.json)")
    p.set_defaults(func=_review_import, extra=None)


def main(argv: list[str] | None = None) -> int:
    """Run the three commands on their own: `python -m kwan.graph._cli_augment augment --dry-run`."""
    parser = argparse.ArgumentParser(prog="kwan graph")
    register_into(parser.add_subparsers(dest="graph_command", metavar="COMMAND", required=True))
    args = parser.parse_args(argv)
    code = args.func(args)
    return code if isinstance(code, int) else 0


if __name__ == "__main__":
    sys.exit(main())
