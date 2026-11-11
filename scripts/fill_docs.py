#!/usr/bin/env python3
"""Fill the RESULTS blocks of the README and the docs from data/eval_results.json.

Every number that readers see in README.md, docs/evidence.md, docs/model-card.md and docs/method.md
sits between two markers:

    <!-- RESULTS:START id=<block id> source=data/eval_results.json -->
    ...rendered from the results file...
    <!-- RESULTS:END -->

This script renders each block from the results file and writes it back. Nothing between the markers
is typed by hand. A block whose results are not in the file yet says "not run yet".

    python scripts/fill_docs.py            rewrite every block
    python scripts/fill_docs.py --check    exit 1 if any block is out of date (what the tests run)

Rounding: AUC and the other prediction metrics to 3 decimals in the README and 4 in the docs,
latency to 2 decimals, simulation means to 1 decimal. Rounding is half-up (PRD decision D8).
Standard library only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = ROOT / "data" / "eval_results.json"
SOURCE = "data/eval_results.json"
FILES = ("README.md", "docs/evidence.md", "docs/model-card.md", "docs/method.md")

START = re.compile(r"<!-- RESULTS:START id=([\w-]+) source=([\w./-]+) -->")
END = "<!-- RESULTS:END -->"
BLOCK = re.compile(r"(<!-- RESULTS:START id=([\w-]+) source=([\w./-]+) -->)(.*?)(<!-- RESULTS:END -->)", re.S)

NOT_RUN = "not run yet"

# ------------------------------------------------------------------------------------------- formatting


def _round(x: float, dp: int) -> str:
    """Half-up rounding to `dp` decimals, as text."""
    q = Decimal(1).scaleb(-dp) if dp > 0 else Decimal(1)
    return str(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))


def num(x: Any, dp: int) -> str:
    """A number for a table cell, or a dash-free blank when it is missing."""
    if x is None or isinstance(x, bool):
        return "not run" if x is None else str(x)
    return _round(x, dp)


def count(x: Any) -> str:
    """An integer with thousands separators."""
    if x is None:
        return "not run"
    return f"{int(x):,}"


def pct(x: Any, dp: int = 1) -> str:
    """A share from 0 to 1 as a percentage."""
    if x is None:
        return "not run"
    return _round(float(x) * 100, dp) + "%"


def table(header: Sequence[str], rows: Sequence[Sequence[str]], align_right: bool = True) -> str:
    """A Markdown table. Numeric columns are right-aligned."""
    sep = ["---"] + (["---:"] * (len(header) - 1) if align_right else ["---"] * (len(header) - 1))
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def yes_no(x: Any) -> str:
    return "yes" if x else "no"


def _get(d: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, Mapping) or k not in cur:
            return default
        cur = cur[k]
    return cur


class NotRun(Exception):
    """Raised by a renderer when the block's results are not in the file."""


# ------------------------------------------------------------------------------------------- names

DATASET_ROWS: list[tuple[str, str | None, str]] = [
    # (results key, sub-block, display name)
    ("ammore_2024", "all", "AMMORE, all pupils (Ghana, Nigeria, Kenya, South Africa)"),
    ("ammore_2024", "ghana", "AMMORE, Ghanaian pupils only"),
    ("assistments_2009", None, "ASSISTments 2009"),
    ("eedi_2020_task34", None, "Eedi 2020, task 3/4"),
    ("eedi_2020_task12", None, "Eedi 2020, task 1/2"),
    ("assistments_2012", None, "ASSISTments 2012-13"),
    ("assistments_2015", None, "ASSISTments 2015"),
    ("assistments_2017", None, "ASSISTments 2017"),
    ("xes3g5m", None, "XES3G5M"),
    ("junyi_2015", None, "Junyi Academy 2015 (evaluation only)"),
]
MODELS: list[tuple[str, str]] = [
    ("skill-mean", "Skill average"),
    ("bkt", "BKT"),
    ("elo", "Classic Elo"),
    ("elo-multi (Kwan)", "**Kwan**"),
]
KWAN = "elo-multi (Kwan)"
DEEP_MODELS: list[tuple[str, str]] = [
    (KWAN, "**Kwan (multi-level Elo)**"),
    ("dkt", "DKT"),
    ("sakt", "SAKT"),
    ("simplekt", "simpleKT"),
    ("akt", "AKT"),
]
DEEP_DATASETS: list[tuple[str, str]] = [
    ("assistments_2009", "ASSISTments 2009"),
    ("eedi_34", "Eedi task 3/4"),
    ("ammore", "AMMORE"),
    ("xes3g5m", "XES3G5M"),
]
FIT_DATASETS: list[tuple[str, str]] = [
    ("assistments_2009", "ASSISTments 2009"),
    ("eedi_2020_task34", "Eedi 2020, task 3/4"),
    ("eedi_2020_task12", "Eedi 2020, task 1/2"),
    ("ammore_2024", "AMMORE (evaluation only)"),
    ("assistments_2015", "ASSISTments 2015"),
    ("assistments_2017", "ASSISTments 2017"),
    ("xes3g5m", "XES3G5M"),
]
PROFILES: list[tuple[str, str]] = [
    ("beginner_p1", "Beginner in Basic 1"),
    ("gapped_p4", "Basic 4 pupil with gaps in B1 to B3"),
    ("gapped_p6", "Basic 6 pupil with gaps that grew"),
    ("uneven_strands", "Basic 4 pupil, strong in Number, weak in Geometry"),
]
POLICIES: list[tuple[str, str]] = [
    ("kwan-strict", "**Kwan**"),
    ("kwan-explore", "Kwan + exploration"),
    ("kwan-top1", "Kwan, top pick only"),
    ("syllabus", "Syllabus order"),
    ("random", "Random"),
    ("remedial-first", "Remedial first"),
]
V1_POLICIES: list[tuple[str, str]] = [
    ("kwan", "**Following Kwan**"),
    ("syllabus", "Syllabus order"),
    ("random", "Random practice"),
]
V1_PROFILES: list[tuple[str, str]] = [
    ("gapped_p4", "P4 pupil with gaps in B1 to B3"),
    ("beginner_p1", "Beginner in P1"),
]
PARAM_NAMES: list[tuple[str, str]] = [
    ("k_s", "`k_s`, step size for the pupil's overall ability"),
    ("k_k", "`k_k`, step size for a skill offset"),
    ("decay", "`decay`, how fast the steps shrink"),
    ("grade_step", "`grade_step`, prior drop per class level"),
    ("transfer", "`transfer`, share borrowed from prerequisites"),
    ("streak_to_master", "`streak_to_master`, right answers in a row that count as mastery"),
    ("tau_days", "`tau_days`, the forgetting time constant (days)"),
    ("streak_expiry_days", "`streak_expiry_days`, life of a streak mastery (days)"),
]
LANGS: list[tuple[str, str]] = [("tw", "Twi"), ("ee", "Ewe"), ("gaa", "Ga")]


# ------------------------------------------------------------------------------------------- renderers


def _dataset_blocks(res: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    out = []
    for key, sub, name in DATASET_ROWS:
        block = res.get(key)
        if sub is not None and isinstance(block, Mapping):
            block = block.get(sub)
        if isinstance(block, Mapping) and "metrics" in block:
            out.append((name, block))
    if not out:
        raise NotRun("no dataset block")
    return out


def kt_table(res: Mapping[str, Any], dp: int, full: bool) -> str:
    """One row per dataset: size and the AUC of each model. In full mode also Kwan's other metrics."""
    rows = []
    for name, block in _dataset_blocks(res):
        metrics = block["metrics"]
        cells = [name, count(block.get("rows")), count(block.get("students"))]
        for model, _ in MODELS:
            auc = _get(metrics, model, "AUC")
            cells.append("not run" if auc is None else (f"**{num(auc, dp)}**" if model == KWAN else num(auc, dp)))
        rows.append(cells)
    out = [table(["Dataset", "Answers", "Students"] + [label for _, label in MODELS], rows)]
    if full:
        rows = []
        for name, block in _dataset_blocks(res):
            m = _get(block, "metrics", KWAN, default={})
            rows.append([name, num(m.get("RMSE"), dp), num(m.get("ACC"), dp), num(m.get("ECE"), dp)])
        out.append("Kwan's model, the same runs, other measures:")
        out.append(table(["Dataset", "RMSE", "Accuracy", "ECE"], rows))
    return "\n\n".join(out)


def ghana(res: Mapping[str, Any], dp: int) -> str:
    """AMMORE for all pupils next to the Ghanaian pupils only."""
    block = res.get("ammore_2024")
    if not isinstance(block, Mapping) or "ghana" not in block or "all" not in block:
        raise NotRun("ammore_2024")
    rows = []
    for sub, name in (("all", "All pupils"), ("ghana", "**Ghanaian pupils only**")):
        b = block[sub]
        cells = [name, count(b.get("students")), count(b.get("rows")), count(b.get("skills"))]
        for model, _ in MODELS:
            if model == "bkt":
                continue
            auc = _get(b, "metrics", model, "AUC")
            cells.append("not run" if auc is None else (f"**{num(auc, dp)}**" if model == KWAN else num(auc, dp)))
        rows.append(cells)
    head = ["AMMORE", "Pupils", "Answers", "Skills"] + [label for m, label in MODELS if m != "bkt"]
    ghana_auc = _get(block, "ghana", "metrics", KWAN, "AUC")
    all_auc = _get(block, "all", "metrics", KWAN, "AUC")
    line = (
        f"Kwan's AUC is {num(all_auc, dp)} on all AMMORE pupils and {num(ghana_auc, dp)} on the "
        f"{count(_get(block, 'ghana', 'students'))} Ghanaian pupils ({count(_get(block, 'ghana', 'rows'))} answers)."
    )
    return line + "\n\n" + table(head, rows)


def learner_fit(res: Mapping[str, Any], dp: int, full: bool) -> str:
    """Forgetting and the tuned step sizes, per dataset, and what ships."""
    fit = res.get("learner_fit")
    if not isinstance(fit, Mapping) or "datasets" not in fit:
        raise NotRun("learner_fit")
    rows = []
    for key, name in FIT_DATASETS:
        d = fit["datasets"].get(key)
        if not isinstance(d, Mapping):
            continue
        has_time = d.get("timestamps")
        tau = d.get("tau_chosen")
        rows.append(
            [
                name,
                yes_no(has_time),
                "none" if tau is None else f"{num(tau, 0)} days",
                num(d.get("auc_without_forgetting"), dp),
                num(d.get("auc_with_forgetting"), dp),
                num(d.get("auc_change_from_forgetting"), dp),
                num(d.get("auc_chosen"), dp),
            ]
        )
    out = [
        table(
            [
                "Dataset",
                "Has times",
                "Best τ",
                "AUC, v1 steps, no forgetting",
                "AUC, with forgetting",
                "Change",
                "AUC, tuned steps",
            ],
            rows,
        )
    ]
    shipped = fit.get("shipped", {})
    params = shipped.get("params", {})
    held = shipped.get("held_out", {})
    if params:
        chosen_on = ", ".join(shipped.get("chosen_on", []))
        tau_on = ", ".join(shipped.get("tau_chosen_on", []))
        out.append(
            f"Shipped in the maths bundle: `k_s` {num(params.get('k_s'), 2)}, `k_k` {num(params.get('k_k'), 2)}, "
            f"`decay` {num(params.get('decay'), 2)}, τ {num(params.get('tau_days'), 0)} days "
            f"(step sizes chosen on {chosen_on}; τ chosen on {tau_on})."
        )
    if held:
        rows = [
            [name, num(h.get("auc_v1_defaults"), dp), num(h.get("auc_shipped"), dp), num(h.get("gain"), dp)]
            for key, name in FIT_DATASETS
            if isinstance((h := held.get(key)), Mapping)
        ]
        out.append("Held-out students, v1 defaults against the shipped parameters:")
        out.append(table(["Dataset", "AUC, v1 defaults", "AUC, shipped", "Gain"], rows))
    if full:
        acc = fit.get("acceptance", {})
        if acc:
            out.append(
                f"Check: {acc.get('rule')}. Largest drop seen: {num(acc.get('largest_auc_drop_from_forgetting'), dp)}. "
                f"Passed: {yes_no(acc.get('passed'))}."
            )
        proto = fit.get("protocol")
        if proto:
            out.append(f"Protocol: {proto}.")
    return "\n\n".join(out)


def params_block(res: Mapping[str, Any], dp: int) -> str:
    """The parameters shipped in the bundle and where they were chosen."""
    shipped = _get(res, "learner_fit", "shipped")
    if not isinstance(shipped, Mapping) or "params" not in shipped:
        raise NotRun("learner_fit.shipped")
    params = shipped["params"]
    rows = []
    for key, label in PARAM_NAMES:
        if key in params:
            v = params[key]
            if v is None:
                cell = "none"
            elif isinstance(v, int) or key.endswith("_days"):
                cell = count(v) if isinstance(v, int) else num(v, 0)
            else:
                cell = num(v, 2)
            rows.append([label, cell])
    out = [table(["Parameter", "Value"], rows)]
    out.append(
        f"Chosen on: {', '.join(shipped.get('chosen_on', []))}. "
        f"τ chosen on: {', '.join(shipped.get('tau_chosen_on', []))}."
    )
    if shipped.get("rule"):
        out.append(f"Rule: {shipped['rule']}.")
    held = shipped.get("held_out", {})
    if held:
        rows = [
            [name, num(h.get("auc_v1_defaults"), dp), num(h.get("auc_shipped"), dp), num(h.get("gain"), dp)]
            for key, name in FIT_DATASETS
            if isinstance((h := held.get(key)), Mapping)
        ]
        out.append(table(["Held-out students", "AUC, v1 defaults", "AUC, shipped", "Gain"], rows))
    notes = list(_get(res, "learner_fit", "notes", default=[]) or []) + list(shipped.get("notes", []) or [])
    if notes:
        out.append("\n".join(f"- {n}" for n in notes))
    return "\n\n".join(out)


def deep_kt(res: Mapping[str, Any], dp: int, full: bool) -> str:
    """pyKT models against Kwan's Elo on the same folds, and the D7 decision."""
    kt = res.get("kt_deep")
    if not isinstance(kt, Mapping) or "datasets" not in kt:
        raise NotRun("kt_deep")
    datasets = [(k, n) for k, n in DEEP_DATASETS if k in kt["datasets"]]
    per_model = _get(kt, "decision", "per_model", default={}) or {}
    rows = []
    for model, label in DEEP_MODELS:
        cells = [label]
        ms = None
        for key, _ in datasets:
            m = _get(kt, "datasets", key, "models", model)
            if not isinstance(m, Mapping):
                cells.append("not run")
                continue
            cells.append(f"{num(m.get('AUC_mean'), dp)} ± {num(m.get('AUC_sd'), dp)}")
            if ms is None:
                ms = m.get("ms_per_update_cpu")
        if model in per_model and per_model[model].get("ms_per_update_cpu") is not None:
            ms = per_model[model]["ms_per_update_cpu"]
        cells.append("not run" if ms is None else ("under 0.01" if float(ms) < 0.005 else num(ms, 2)))
        rows.append(cells)
    out = [table(["Model"] + [f"{n} AUC" for _, n in datasets] + ["ms per update (CPU)"], rows)]
    dec = kt.get("decision", {})
    if dec:
        verdict = f"adopt {dec.get('model')}" if dec.get("adopt") else "Elo stays the default learner model"
        out.append(f"**Decision: {verdict}.** {dec.get('reason', '')}".rstrip())
        if dec.get("rule"):
            out.append(f"Rule: {dec['rule']}")
    if full:
        proto = kt.get("protocol", {})
        if proto.get("split"):
            out.append(f"Protocol: {proto.get('split')}. {proto.get('validation', '')}".rstrip())
        note = _get(kt, "budget", "note")
        if note:
            out.append(f"Training budget: {note}")
        skipped = kt.get("skipped", {})
        if skipped:
            by_text: dict[str, list[str]] = {}
            for key, text in skipped.items():
                by_text.setdefault(str(text), []).append(f"`{key}`")
            out.append("Not run:\n\n" + "\n".join(f"- {', '.join(keys)}: {text}" for text, keys in by_text.items()))
    return "\n\n".join(out)


def simulation_v1(res: Mapping[str, Any], dp: int) -> str:
    """The Round 1 table: new skills mastered under three policies."""
    sim = res.get("simulation")
    if not isinstance(sim, Mapping) or "new_skills_mastered" not in sim:
        raise NotRun("simulation")
    data = sim["new_skills_mastered"]
    rows = []
    for key, name in V1_PROFILES:
        if key not in data:
            continue
        cells = [name]
        for pol, _ in V1_POLICIES:
            v = data[key].get(pol)
            cells.append("not run" if v is None else (f"**{num(v, dp)}**" if pol == "kwan" else num(v, dp)))
        rows.append(cells)
    return f"{sim.get('setup', '')}\n\n" + table(["Pupil"] + [label for _, label in V1_POLICIES], rows)


def _policy_table(data: Mapping[str, Any], dp: int, goal: bool = False) -> str:
    rows = []
    for prof, pname in PROFILES:
        if prof not in data:
            continue
        cells = [pname]
        for pol, _ in POLICIES:
            v = data[prof].get(pol)
            if v is None:
                cells.append("not run")
            elif goal:
                cells.append(f"{num(v.get('mean'), dp)} ({pct(v.get('reached'), 0)})")
            else:
                cells.append(f"**{num(v.get('mean'), dp)}**" if pol == "kwan-strict" else num(v.get("mean"), dp))
        rows.append(cells)
    return table(["Pupil profile"] + [label for _, label in POLICIES], rows)


def simulation_v2(res: Mapping[str, Any], dp: int, full: bool) -> str:
    """The v2 simulator: four profiles, six policies, forgetting in the hidden state."""
    v2 = _get(res, "simulation", "v2")
    if not isinstance(v2, Mapping) or "new_skills_mastered" not in v2:
        raise NotRun("simulation.v2")
    steps, days = v2.get("steps"), v2.get("retention_days")
    out = [v2.get("setup", "")]
    out.append(
        f"New skills mastered after {count(steps)} practice steps "
        f"(mean over {count(v2.get('n_learners'))} simulated pupils):"
    )
    out.append(_policy_table(v2["new_skills_mastered"], dp))
    out.append(f"Of those, skills still known after {count(days)} days without practice:")
    out.append(_policy_table(v2["skills_retained"], dp))
    if full:
        out.append(
            "Steps to reach a goal skill, with the share of runs that reached it within "
            f"{count(v2.get('goal_max_steps'))} steps:"
        )
        out.append(_policy_table(v2.get("steps_to_goal", {}), dp, goal=True))
    acc = v2.get("acceptance", {})
    if acc:
        checks = [k for k in acc if k not in ("met", "failed")]
        out.append(
            "Checks set before the run: "
            + "; ".join(f"{k.replace('_', ' ')}: {yes_no(acc[k])}" for k in checks)
            + f". All met: {yes_no(acc.get('met'))}."
        )
    if full:
        profs = v2.get("profiles", {})
        pols = v2.get("policies", {})
        if profs:
            out.append("Profiles:\n\n" + "\n".join(f"- `{k}`: {v}" for k, v in profs.items()))
        if pols:
            out.append("Policies:\n\n" + "\n".join(f"- `{k}`: {v}" for k, v in pols.items()))
    return "\n\n".join(out)


def simulation_assumptions(res: Mapping[str, Any]) -> str:
    """The simulator's assumptions, in the simulator's own words."""
    text = _get(res, "simulation", "v2", "assumptions", "text")
    if not text:
        raise NotRun("simulation.v2.assumptions")
    profs = _get(res, "simulation", "v2", "profiles", default={}) or {}
    out = [text]
    if profs:
        out.append("\n".join(f"- `{k}`: {v}" for k, v in profs.items()))
    return "\n\n".join(out)


def goal_path(res: Mapping[str, Any]) -> str:
    g = res.get("goal_path")
    if not isinstance(g, Mapping) or "on_path_share" not in g:
        raise NotRun("goal_path")
    share = g["on_path_share"]
    return (
        f"With a goal set, {pct(share.get('strict'))} of the top-{count(g.get('k'))} picks sit on the path to the goal "
        "in "
        f"strict mode (the default) and {pct(share.get('soft'))} in soft mode (the Round 1 behaviour). "
        f"Setup: {g.get('setup', '')}."
    )


def latency(res: Mapping[str, Any]) -> str:
    lat = res.get("latency")
    if not isinstance(lat, Mapping) or "recommend_top3_mean_ms" not in lat:
        raise NotRun("latency")
    return (
        f"A top-3 recommendation takes {num(lat.get('recommend_top3_mean_ms'), 2)} ms on average and "
        f"{num(lat.get('recommend_top3_p95_ms'), 2)} ms at the 95th percentile "
        f"({lat.get('hardware', 'one CPU core')}; {count(lat.get('calls'))} timed calls per learner state, "
        "in-process, no network)."
    )


def curriculum(res: Mapping[str, Any], full: bool) -> str:
    cur = res.get("curriculum")
    if not isinstance(cur, Mapping) or "indicators" not in cur:
        raise NotRun("curriculum")
    out = [
        f"Indicators parsed: {count(cur.get('indicators'))} of {count(cur.get('expected'))} expected from the hand "
        f"count (counts match: {yes_no(cur.get('counts_match'))}). Descriptions that look cut short: "
        f"{count(cur.get('truncated'))} ({pct(cur.get('truncation_rate'))}). Indicators with at least one exemplar: "
        f"{count(cur.get('with_exemplar'))} "
        f"({pct(cur.get('share_with_exemplar'), 0)}), {count(cur.get('exemplars'))} exemplars in all, "
        f"{count(cur.get('exemplars_ghana_context'))} of them set in Ghana. Misprinted codes corrected: "
        f"{count(cur.get('corrections_applied'))}."
    ]
    labels = cur.get("labels", {})
    if labels:
        out.append(
            f"Short labels: {count(labels.get('draft'))} drafted, {count(labels.get('reviewed'))} reviewed by a "
            "teacher, "
            f"{count(labels.get('missing'))} missing, {count(labels.get('over_60_characters'))} over 60 characters."
        )
    if full:
        pgs = cur.get("per_grade_strand", {})
        per_grade = cur.get("per_grade", {})
        if pgs:
            strands = ["1", "2", "3", "4"]
            rows = []
            for g in sorted(pgs, key=int):
                rows.append([f"B{g}"] + [count(pgs[g].get(s, 0)) for s in strands] + [count(per_grade.get(g))])
            out.append(table(["Class", "Number", "Algebra", "Geometry and Measurement", "Data", "All"], rows))
        if cur.get("truncated_codes"):
            out.append("Cut short: " + ", ".join(f"`{c}`" for c in cur["truncated_codes"]) + ".")
        out.append(f"NaCCA text committed to the repository: {yes_no(cur.get('nacca_text_committed'))}.")
    return "\n\n".join(out)


def graph(res: Mapping[str, Any], full: bool) -> str:
    g = res.get("graph")
    if not isinstance(g, Mapping) or "nodes" not in g:
        raise NotRun("graph")
    by_type = g.get("edges_by_type", {})
    types = ", ".join(f"{count(v)} {k}" for k, v in sorted(by_type.items()))
    out = [
        f"Skill map: {count(g.get('nodes'))} skills and {count(g.get('edges'))} prerequisite links ({types}). "
        f"No cycles: {yes_no(g.get('is_dag'))}. "
        f"Every skill reachable: {yes_no((g.get('unreachable_from_a_root') or 0) == 0)}. "
        f"Longest chain: {count(g.get('longest_path_edges'))} links."
    ]
    if full:
        bys = g.get("nodes_by_strand", {})
        names = {"1": "Number", "2": "Algebra", "3": "Geometry and Measurement", "4": "Data"}
        if bys:
            out.append(table(["Strand", "Skills"], [[names.get(k, k), count(v)] for k, v in sorted(bys.items())]))
    return "\n\n".join(out)


def graph_augment(res: Mapping[str, Any], full: bool) -> str:
    ga = res.get("graph_augment")
    if not isinstance(ga, Mapping) or "review" not in ga:
        raise NotRun("graph_augment")
    jb = ga.get("junyi_benchmark", {})
    held = jb.get("held_out", {})
    nacca = ga.get("nacca", {})
    rev = ga.get("review", {})
    out = [
        f"Decision model: {ga.get('provider')} ({ga.get('model')}). Precision bar: {pct(ga.get('precision_bar'), 0)}. "
        f"On the Junyi benchmark (held-out exercises) it proposed {count(held.get('proposed'))} links, "
        f"{count(held.get('correct'))} of them expert links: precision {pct(held.get('precision'))}, "
        f"recall {pct(held.get('recall'))}. Meets the bar: {yes_no(ga.get('meets_precision_bar'))}."
    ]
    out.append(
        f"On NaCCA Mathematics it scored {count(nacca.get('candidate_pairs'))} candidate pairs and proposed "
        f"{count(nacca.get('proposed_edges'))} links. Teacher review: {count(rev.get('pending'))} pending, "
        f"{count(rev.get('accepted'))} accepted, {count(rev.get('rejected'))} rejected. "
        f"Accepted links shipped in the graph: {count(_get(ga, 'impact_in_simulator', 'accepted_edges'))}."
    )
    ta = rev.get("teacher_acceptance")
    out.append(f"Teacher acceptance: {rev.get('status') if ta is None else pct(ta)}.")
    if full:
        if held.get("precision_transitive") is not None:
            out.append(
                f"Counting a source that is an earlier prerequisite of the target as correct, precision is "
                f"{pct(held.get('precision_transitive'))}; ignoring direction, "
                f"{pct(held.get('precision_ignoring_direction'))}."
            )
        by_pair = nacca.get("proposed_edges_by_strand_pair", {})
        if by_pair:
            out.append(
                "Proposed links by strand pair: " + "; ".join(f"{k}: {count(v)}" for k, v in by_pair.items()) + "."
            )
        out.append(f"Cost of the NaCCA run: {num(_get(ga, 'spend', 'cost_usd'), 2)} USD.")
    return "\n\n".join(out)


def contexts(res: Mapping[str, Any], full: bool) -> str:
    c = res.get("contexts")
    if not isinstance(c, Mapping) or "indicators" not in c:
        raise NotRun("contexts")
    out = [
        f"Practice settings: {count(c.get('settings'))} drafted for {count(c.get('indicators_with_a_setting'))} of "
        f"{count(c.get('indicators'))} indicators; {count(c.get('settings_passing_automatic_checks'))} pass the "
        "automatic checks. "
        f"Reviewed by a teacher: {count(c.get('reviewed_settings'))} settings, covering "
        f"{count(c.get('indicators_with_reviewed_setting'))} indicators "
        f"({num(c.get('reviewed_coverage_pct'), 1)}% coverage). "
        f"Reviewers so far: {count(c.get('reviewers'))}."
    ]
    tr = c.get("translations", {})
    if tr:
        rows = []
        for key, name in LANGS:
            t = tr.get(key)
            if isinstance(t, Mapping):
                rows.append(
                    [
                        name,
                        count(t.get("indicators_b1_b3")),
                        count(t.get("unreviewed_drafts")),
                        count(t.get("reviewed_translations")),
                        count(t.get("indicators_with_shipping_translation")),
                    ]
                )
        out.append("Translations for Basic 1 to Basic 3:")
        out.append(table(["Language", "B1 to B3 indicators", "Drafts", "Reviewed", "Indicators shipping"], rows))
    if full:
        out.append(
            f"Short labels: {count(c.get('short_labels'))}, {count(c.get('reviewed_labels'))} reviewed. "
            f"{c.get('note', '')}".rstrip()
        )
    return "\n\n".join(out)


def forgetting_tau(res: Mapping[str, Any]) -> str:
    shipped = _get(res, "learner_fit", "shipped")
    if not isinstance(shipped, Mapping) or "params" not in shipped:
        raise NotRun("learner_fit.shipped")
    p = shipped["params"]
    return (
        f"The shipped value of τ is {num(p.get('tau_days'), 0)} days, "
        f"chosen on {', '.join(shipped.get('tau_chosen_on', []))}. "
        f"A streak mastery expires after {num(p.get('streak_expiry_days'), 0)} days without practice."
    )


def counts_sentence(res: Mapping[str, Any]) -> str:
    """One line for the README's 'What it does': skills and links, from the graph block."""
    g = res.get("graph")
    if not isinstance(g, Mapping) or "nodes" not in g:
        raise NotRun("graph")
    return (
        f"{count(g.get('nodes'))} NaCCA learning indicators linked by {count(g.get('edges'))} prerequisite connections."
    )


# ------------------------------------------------------------------------------------------- block registry

Renderer = Callable[[Mapping[str, Any]], str]

RENDERERS: dict[str, Renderer] = {
    # README (AUC to 3 decimals)
    "readme-counts": counts_sentence,
    "readme-kt-table": lambda r: kt_table(r, 3, full=False),
    "readme-ghana": lambda r: ghana(r, 3),
    "readme-learner-fit": lambda r: learner_fit(r, 3, full=False),
    "readme-deep-kt": lambda r: deep_kt(r, 3, full=False),
    "readme-simulation-v2": lambda r: simulation_v2(r, 1, full=False),
    "readme-simulation-v1": lambda r: simulation_v1(r, 1),
    "readme-simulation-assumptions": simulation_assumptions,
    "readme-goal-path": goal_path,
    "readme-latency": latency,
    "readme-curriculum": lambda r: curriculum(r, full=False),
    "readme-graph": lambda r: graph(r, full=False),
    "readme-graph-augment": lambda r: graph_augment(r, full=False),
    "readme-contexts": lambda r: contexts(r, full=False),
    # docs/evidence.md (4 decimals)
    "evidence-kt-table": lambda r: kt_table(r, 4, full=True),
    "evidence-ghana": lambda r: ghana(r, 4),
    "evidence-forgetting": lambda r: learner_fit(r, 4, full=True),
    "evidence-deep-kt": lambda r: deep_kt(r, 4, full=True),
    "evidence-simulation": lambda r: simulation_v2(r, 1, full=True) + "\n\n" + simulation_assumptions(r),
    "evidence-goal-path": goal_path,
    "evidence-curriculum": lambda r: curriculum(r, full=True),
    "evidence-graph": lambda r: graph(r, full=True) + "\n\n" + graph_augment(r, full=True),
    "evidence-contexts": lambda r: contexts(r, full=True),
    "evidence-latency": latency,
    # docs/model-card.md (4 decimals)
    "model-card-kt-table": lambda r: kt_table(r, 4, full=True),
    "model-card-ghana": lambda r: ghana(r, 4),
    "model-card-forgetting": lambda r: learner_fit(r, 4, full=False),
    "model-card-deep-kt": lambda r: deep_kt(r, 4, full=True),
    "model-card-params": lambda r: params_block(r, 4),
    "model-card-simulation": lambda r: simulation_v2(r, 1, full=True),
    "model-card-latency": latency,
    # docs/method.md
    "method-forgetting-tau": forgetting_tau,
}


def render(block_id: str, res: Mapping[str, Any]) -> str:
    """The text of one block, or a 'not run yet' notice when its results are missing."""
    renderer = RENDERERS.get(block_id)
    if renderer is None:
        raise KeyError(f"no renderer for RESULTS block {block_id!r}")
    try:
        body = renderer(res)
    except NotRun as e:
        body = f"*Results {NOT_RUN}: `{e}` is not in `{SOURCE}`.*"
    except (KeyError, TypeError, AttributeError) as e:  # the block exists but not in the shape expected
        body = f"*Results {NOT_RUN}: `{SOURCE}` does not hold what this block needs ({e!r}).*"
    return "\n" + body.strip() + "\n"


def fill_text(text: str, res: Mapping[str, Any]) -> tuple[str, list[str]]:
    """Rewrite every RESULTS block in `text`. Returns the new text and the ids of blocks that changed."""
    changed: list[str] = []

    def sub(m: re.Match[str]) -> str:
        start, block_id, source, old, end = m.groups()
        if source != SOURCE:
            raise ValueError(f"RESULTS block {block_id!r} names source {source!r}; only {SOURCE!r} is supported")
        new = render(block_id, res)
        if new != old:
            changed.append(block_id)
        return f"{start}{new}{end}"

    return BLOCK.sub(sub, text), changed


def block_ids(text: str) -> list[str]:
    return [m.group(1) for m in START.finditer(text)]


def load_results(path: Path = RESULTS_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="report blocks that are out of date and exit 1; write nothing")
    ap.add_argument("--results", default=str(RESULTS_PATH), help=f"results file (default {SOURCE})")
    ap.add_argument("--list", action="store_true", help="list the RESULTS blocks of every file and exit")
    args = ap.parse_args(argv)
    res = load_results(Path(args.results))
    stale: list[str] = []
    for name in FILES:
        path = ROOT / name
        text = path.read_text(encoding="utf-8")
        if args.list:
            for bid in block_ids(text):
                print(f"{name}: {bid}")
            continue
        new, changed = fill_text(text, res)
        unknown = [b for b in block_ids(text) if b not in RENDERERS]
        if unknown:
            print(f"{name}: no renderer for {unknown}", file=sys.stderr)
            return 2
        if changed:
            stale += [f"{name}: {b}" for b in changed]
            if not args.check:
                path.write_text(new, encoding="utf-8")
                print(f"{name}: rewrote {len(changed)} block(s): {', '.join(changed)}")
        elif not args.check:
            print(f"{name}: up to date")
    if args.list:
        return 0
    if args.check:
        if stale:
            print("out of date (run `python scripts/fill_docs.py`):\n  " + "\n  ".join(stale))
            return 1
        print("every RESULTS block is up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
