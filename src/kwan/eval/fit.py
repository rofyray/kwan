"""Fits of the learner model on the public datasets (PRD Phase 5.2, 5.3 and 5.4).

Three things are fitted here: the forgetting time constant tau, the Elo step sizes (k_s, k_k, decay)
and the item-difficulty banks. The results go to the key `learner_fit` of data/eval_results.json,
the shipped defaults to data/params/math.params.json and the banks to build/fit/ (git-ignored).

Protocol, per dataset
    outer split   80/20 by student, seed 42 (`kwan.eval.kt.split`); the test students are held out
    inner split   the training students are split again 80/20 by student, seed 43: the model is
                  fitted on the first part and every configuration is chosen on the second
                  (the validation students); the held-out numbers are reported afterwards
    grid          every (k_s, k_k, decay) triple of GRID, each with every tau of TAU_GRID on data
                  with timestamps (one bank per triple, fitted on the fit students without
                  forgetting, as the evaluation does); `tau` stage = the slice at the v1 step sizes
    item bank     fitted on all the training students at the v1 step sizes; held-out AUC with and
                  without it; saved only when the dataset is commercial-OK (`registry.commercial_ok`)
    engine        `engine="fast"` of `kwan.learner.scoring` (graph-free, the same arithmetic as
                  `EloLearner` without a graph); large datasets are subsampled by student

`grade_step` and `transfer` act only through a prerequisite graph. The public datasets have no such
graph, so no grid can move them: they keep the v1 values (0.4 and 0.5).

Shipped defaults (`choose_shipped`) come from the commercial-OK L1 to L3 datasets only: tau is the
value with the best mean validation AUC on the Eedi sets (the commercial-OK timestamped data); the
step sizes are the triple with the best mean validation AUC across the L1 to L3 sets at that tau;
then `guard` checks the result on the held-out students of ASSISTments 2009 and Eedi 3/4 against the
v1 defaults and puts a parameter back to its v1 value when the tuned set is worse by more than
0.002 AUC. AMMORE (CC BY-NC) and Junyi inform the report but never a shipped number.

    kwan fit dataset eedi-34            one dataset -> build/fit/eedi_2020_task34.json (+ .itembank.json)
    kwan fit all                        every dataset whose raw files are present
    kwan fit report --out data/eval_results.json --params data/params/math.params.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kwan._paths import data_dir
from kwan.data import registry
from kwan.learner.item_bank import ItemBank, fit_item_bank
from kwan.learner.params import EloParams
from kwan.learner.scoring import score_stream

KEY = "learner_fit"  # the top-level key this module owns in data/eval_results.json
SCHEMA = 1
BUILD_DIR = Path("build") / "fit"
PARAMS_PATH = data_dir() / "params" / "math.params.json"
SUBJECT = "math"

OUTER_SEED, INNER_SEED = 42, 43
SPLIT = "outer 80/20 by student (seed 42); chosen on an inner 80/20 by student split (seed 43) of the training students"
DEFAULT_MAX_ROWS = 1_500_000  # datasets above this are subsampled by student (seed 42) for the grid

V1_PARAMS = EloParams(tau_days=None, streak_expiry_days=None)  # v1 had no forgetting
TAU_GRID: tuple[float | None, ...] = (7.0, 14.0, 30.0, 45.0, 60.0, 90.0, 120.0, 180.0, None)
GRID: dict[str, list[float]] = {"k_s": [0.2, 0.3, 0.5], "k_k": [0.3, 0.4, 0.6], "decay": [0.02, 0.05, 0.1]}
STEP_FIELDS = ("k_s", "k_k", "decay")
ENGINE_NOTE = (
    "kwan.learner.scoring engine='fast' (graph-free multi-level Elo, the arithmetic of EloLearner without a graph)"
)
GUARD_FIELDS = (*STEP_FIELDS, "tau_days")
TOLERANCE = 0.002  # the shipped set may never be worse than v1 by more than this (AUC) on a guarded set
GUARDED = ("assistments_2009", "eedi_2020_task34")
SHIP_FROM = ("assistments_2009", "eedi_2020_task34", "eedi_2020_task12")  # L1 to L3
TAU_FROM = ("eedi_2020_task34", "eedi_2020_task12")  # commercial-OK data with timestamps
TAU_INFORM = ("ammore_2024",)  # timestamped but not commercial-OK: reported, never shipped

# results key -> (name after `kwan fit`, loader name, has item ids distinct from skills)
DATASETS: dict[str, tuple[str, str, bool]] = {
    "assistments_2009": ("assistments-2009", "assistments_2009", True),
    "eedi_2020_task34": ("eedi-34", "eedi_2020_task34", True),
    "eedi_2020_task12": ("eedi-12", "eedi_2020_task12", True),
    "ammore_2024": ("ammore", "ammore_2024:all", True),
    "assistments_2012": ("assistments-2012", "assistments_2012", True),
    "assistments_2015": ("assistments-2015", "assistments_2015", False),
    "assistments_2017": ("assistments-2017", "assistments_2017", True),
    "xes3g5m": ("xes3g5m", "xes3g5m", True),
    "junyi_2015": ("junyi", "junyi_2015", True),
}
BY_CLI = {cli: key for key, (cli, _, _) in DATASETS.items()}

Rows = list[tuple[Any, ...]]


# --------------------------------------------------------------------------------------------- rows


def has_timestamps(df: Any) -> bool:
    """True when the frame's `t` column holds real times (not an order id)."""
    from pandas.api import types as pt

    return bool(pt.is_datetime64_any_dtype(df["t"]))


def frame_rows(df: Any, timestamps: bool | None = None) -> Rows:
    """The rows of a schema frame as `(user, item, skill, correct[, seconds])` tuples for the scorers.

    The time is added only when the frame has real timestamps (or `timestamps` says so); an order id
    is never passed as a time, so forgetting cannot act on it."""
    from kwan.eval.kt import item_column

    timestamps = has_timestamps(df) if timestamps is None else timestamps
    cols = ["user_id", item_column(df), "skill_id", "correct"]
    arrays = [df[c].to_numpy().tolist() for c in cols]
    if timestamps:
        arrays.append((df["t"].to_numpy().astype("datetime64[ms]").astype("int64") / 1000.0).tolist())
    return list(zip(*arrays))


def subsample(df: Any, max_rows: int | None, seed: int = OUTER_SEED) -> tuple[Any, dict[str, Any] | None]:
    """Keep whole students, in a seeded shuffle, until `max_rows` rows are reached.

    Returns the frame and a record of what was done (None when nothing was)."""
    if max_rows is None or len(df) <= max_rows:
        return df, None
    import numpy as np

    users = np.array(df.user_id.unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(users)
    per_user = df.groupby("user_id", sort=False).size()
    total = per_user.reindex(users).to_numpy().cumsum()
    keep = users[: int(np.searchsorted(total, max_rows, side="left")) + 1]
    out = df[df.user_id.isin(set(keep.tolist()))]
    return out, {
        "max_rows": int(max_rows),
        "rows_before": int(len(df)),
        "students_before": int(len(users)),
        "seed": seed,
    }


def split_rows(df: Any, timestamps: bool) -> dict[str, Rows]:
    """{"fit", "val", "train", "test"}: the outer split and the inner split of its training students."""
    from kwan.eval.kt import split

    train, test = split(df, OUTER_SEED)
    fit, val = split(train, INNER_SEED)
    return {
        name: frame_rows(part, timestamps)
        for name, part in (("fit", fit), ("val", val), ("train", train), ("test", test))
    }


# --------------------------------------------------------------------------------------------- grids


def _metrics(scores: Mapping[str, float]) -> dict[str, float]:
    return {k: round(float(scores[k]), 5) for k in ("auc", "rmse", "log_loss", "accuracy")}


def _tau_json(tau: float | None) -> float | None:
    return None if tau is None else float(tau)


def _chosen(params: EloParams, timestamps: bool) -> dict[str, Any]:
    out: dict[str, Any] = {f: getattr(params, f) for f in STEP_FIELDS}
    out["tau_days"] = _tau_json(params.tau_days) if timestamps else None
    return out


def run_grid(
    parts: Mapping[str, Rows],
    timestamps: bool,
    grid: Mapping[str, Sequence[float]] = GRID,
    tau_grid: Sequence[float | None] = TAU_GRID,
    base: EloParams = V1_PARAMS,
    log: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Score every (k_s, k_k, decay) triple of `grid`, times every tau when `timestamps`, on the
    validation and on the held-out rows. One bank per triple, fitted on the fit rows.

    Each entry is `{"params": {k_s, k_k, decay, tau_days}, "val": {auc, ...}, "test": {auc, ...}}`.
    Best validation AUC first; ties keep the grid order."""
    names = list(grid)
    taus: Sequence[float | None] = tuple(tau_grid) if timestamps else (None,)
    results: list[dict[str, Any]] = []
    started = time.time()
    for values in itertools.product(*(grid[n] for n in names)):
        steps = dict(zip(names, values))
        p0 = base.replace(**steps)
        bank = fit_item_bank(parts["fit"], k_s=p0.k_s, k_ss=p0.k_k, decay=p0.decay)
        for tau in taus:
            p = p0.replace(tau_days=tau)
            val = score_stream(parts["val"], p, item_bank=bank, use_time=timestamps, engine="fast")
            test = score_stream(parts["test"], p, item_bank=bank, use_time=timestamps, engine="fast")
            results.append({"params": _chosen(p, timestamps), "val": _metrics(val), "test": _metrics(test)})
        if log:
            log(f"  {steps} done, {len(results)} configurations, {time.time() - started:.0f} s")
    results.sort(key=lambda r: (math.isnan(r["val"]["auc"]), -r["val"]["auc"]))
    return results


def lookup(
    results: Iterable[Mapping[str, Any]], params: EloParams | Mapping[str, Any], timestamps: bool
) -> dict[str, Any] | None:
    """The grid entry whose step sizes (and tau, on timestamped data) equal `params`, or None."""
    want = _chosen(params if isinstance(params, EloParams) else EloParams.from_dict(params), timestamps)
    for r in results:
        if r["params"] == want:
            return dict(r)
    return None


def tau_curve(results: Iterable[Mapping[str, Any]], base: EloParams = V1_PARAMS) -> list[dict[str, Any]]:
    """The entries at the step sizes of `base`, in TAU_GRID order (the tau stage of the report)."""
    at_base = [r for r in results if all(r["params"][f] == getattr(base, f) for f in STEP_FIELDS)]
    order = {_tau_json(t): i for i, t in enumerate(TAU_GRID)}
    return sorted(at_base, key=lambda r: order.get(r["params"]["tau_days"], len(order)))


def choose_tau(curve: Sequence[Mapping[str, Any]]) -> float | None:
    """The tau with the best validation AUC on a tau curve (ties: the first in TAU_GRID order)."""
    if not curve:
        raise ValueError("an empty tau curve has no best tau")
    return max(curve, key=lambda r: r["val"]["auc"])["params"]["tau_days"]


# --------------------------------------------------------------------------------------------- datasets


def _bank_stage(
    key: str, parts: Mapping[str, Rows], timestamps: bool, tau: float | None, subsampled: dict[str, Any] | None
) -> tuple[dict[str, Any], ItemBank]:
    """Fit the bank on all the training students at the v1 step sizes; held-out AUC with and without."""
    p = V1_PARAMS.replace(tau_days=tau)
    bank = fit_item_bank(parts["train"], datasets=[key], k_s=p.k_s, k_ss=p.k_k, decay=p.decay)
    bank.provenance["split"] = "training students of the 80/20 split by student, seed 42"
    bank.provenance["subsample"] = subsampled
    with_bank = score_stream(parts["test"], p, item_bank=bank, use_time=timestamps, engine="fast")
    without = score_stream(parts["test"], p, item_bank=None, use_time=timestamps, engine="fast")
    skills_only = ItemBank(skill_betas=bank.skill_betas)
    with_skills = score_stream(parts["test"], p, item_bank=skills_only, use_time=timestamps, engine="fast")
    seen = sum(1 for r in parts["test"] if r[1] in bank)
    block = {
        "items": len(bank),
        "skills": len(bank.skill_betas),
        "test_rows_with_known_item": seen,
        "test_rows": len(parts["test"]),
        "params": {**_chosen(p, timestamps)},
        "auc_without_bank": round(float(without["auc"]), 5),
        "auc_skill_betas_only": round(float(with_skills["auc"]), 5),
        "auc_with_bank": round(float(with_bank["auc"]), 5),
        "rmse_without_bank": round(float(without["rmse"]), 5),
        "rmse_with_bank": round(float(with_bank["rmse"]), 5),
    }
    return block, bank


def fit_rows(
    key: str,
    parts: Mapping[str, Rows],
    timestamps: bool,
    *,
    counts: Mapping[str, int],
    subsampled: dict[str, Any] | None = None,
    grid: Mapping[str, Sequence[float]] = GRID,
    tau_grid: Sequence[float | None] = TAU_GRID,
    log: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], ItemBank]:
    """Every fit of one dataset on prepared rows. Returns (the dataset's fit record, its item bank)."""
    started = time.time()
    results = run_grid(parts, timestamps, grid, tau_grid, log=log)
    grid_seconds = time.time() - started
    curve = tau_curve(results)
    tau = choose_tau(curve)
    v1 = lookup(results, V1_PARAMS, timestamps)
    with_tau = lookup(results, V1_PARAMS.replace(tau_days=tau), timestamps)
    best = results[0]
    if v1 is None or with_tau is None:
        raise ValueError("the grid must contain the v1 step sizes")
    forgetting: dict[str, Any]
    if timestamps:
        forgetting = {
            "tau_chosen": tau,
            "auc_without_forgetting": v1["test"]["auc"],
            "auc_with_forgetting": with_tau["test"]["auc"],
            "val_auc_without_forgetting": v1["val"]["auc"],
            "val_auc_with_forgetting": with_tau["val"]["auc"],
        }
    else:
        # no times reach the model, so a tau of 45 days must give exactly the numbers of "off"
        p45 = V1_PARAMS.replace(tau_days=45.0)
        bank = fit_item_bank(parts["fit"], k_s=p45.k_s, k_ss=p45.k_k, decay=p45.decay)
        on = score_stream(parts["test"], p45, item_bank=bank, use_time=True, engine="fast")
        forgetting = {
            "tau_chosen": None,
            "auc_without_forgetting": v1["test"]["auc"],
            "auc_with_forgetting": round(float(on["auc"]), 5),
            "forgetting_check": "tau_days=45 with use_time=True on rows without times",
        }
    forgetting["auc_change_from_forgetting"] = round(
        forgetting["auc_with_forgetting"] - forgetting["auc_without_forgetting"], 5
    )
    started = time.time()
    bank_block, bank = _bank_stage(key, parts, timestamps, tau, subsampled)
    record = {
        "dataset": key,
        "licence": registry.licence(key),
        "commercial_params_ok": registry.commercial_ok(key),
        "rows": int(counts["rows"]),
        "students": int(counts["students"]),
        "items": int(counts.get("items", 0)),
        "skills": int(counts["skills"]),
        "timestamps": timestamps,
        "subsample": subsampled,
        "split": SPLIT,
        "split_rows": {name: len(rows) for name, rows in parts.items()},
        "tau_grid": [_tau_json(t) for t in tau_grid] if timestamps else [],
        "tau_curve": [
            {"tau_days": r["params"]["tau_days"], "val_auc": r["val"]["auc"], "test_auc": r["test"]["auc"]}
            for r in curve
        ],
        **forgetting,
        "grid": {k: list(v) for k, v in grid.items()},
        "grid_chosen_params": best["params"],
        "auc_v1_defaults": v1["test"]["auc"],
        "auc_chosen": best["test"]["auc"],
        "rmse_v1_defaults": v1["test"]["rmse"],
        "rmse_chosen": best["test"]["rmse"],
        "val_auc_v1_defaults": v1["val"]["auc"],
        "val_auc_chosen": best["val"]["auc"],
        "grid_results": results,
        "item_bank": bank_block,
        "seconds": {"grid": round(grid_seconds, 1), "item_bank": round(time.time() - started, 1)},
    }
    return record, bank


def default_path(key: str, raw_dir: str | Path = "raw") -> Path:
    """Where the raw files of a dataset live after `make data`."""
    from kwan.data import loaders

    return loaders.raw_path(key, raw_dir)


def run_dataset(
    name: str,
    path: str | Path | None = None,
    *,
    max_rows: int | None = DEFAULT_MAX_ROWS,
    build_dir: str | Path = BUILD_DIR,
    raw_dir: str | Path = "raw",
    grid: Mapping[str, Sequence[float]] = GRID,
    tau_grid: Sequence[float | None] = TAU_GRID,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Load one dataset, run every fit and write `build/fit/<key>.json` (plus the bank when allowed).

    `name` is a results key (`eedi_2020_task34`) or the name after `kwan fit` (`eedi-34`)."""
    from kwan.data import loaders
    from kwan.data.schema import counts

    key = BY_CLI.get(name, name)
    if key not in DATASETS:
        raise KeyError(f"unknown dataset '{name}' (known: {', '.join(BY_CLI)})")
    _, loader_name, _ = DATASETS[key]
    log = log or (lambda s: print(s, file=sys.stderr))
    started = time.time()
    df = loaders.load(loader_name, path if path is not None else default_path(key, raw_dir))
    n = counts(df)
    df, subsampled = subsample(df, max_rows)
    timestamps = has_timestamps(df)
    log(
        f"{key}: {n['rows']} rows, {n['students']} students, timestamps={timestamps}"
        + (" (subsampled)" if subsampled else "")
    )
    parts = split_rows(df, timestamps)
    n = counts(df)
    del df
    log(f"{key}: fit={len(parts['fit'])} val={len(parts['val'])} train={len(parts['train'])} test={len(parts['test'])}")
    record, bank = fit_rows(
        key, parts, timestamps, counts=n, subsampled=subsampled, grid=grid, tau_grid=tau_grid, log=log
    )
    record["seconds"]["total"] = round(time.time() - started, 1)
    record["fitted_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    build = Path(build_dir)
    build.mkdir(parents=True, exist_ok=True)
    if record["commercial_params_ok"]:
        bank_path = build / f"{key}.itembank.json"
        bank_path.write_text(json.dumps(bank.to_dict(), indent=1) + "\n", encoding="utf-8")
        record["item_bank"]["path"] = str(bank_path)
        record["item_bank"]["saved"] = True
    else:
        record["item_bank"]["path"] = None
        record["item_bank"]["saved"] = False
        record["item_bank"]["note"] = "not saved: the dataset is evaluation only (not commercial-OK)"
    (build / f"{key}.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    log(f"{key}: done in {record['seconds']['total']} s, wrote {build / f'{key}.json'}")
    return record


# --------------------------------------------------------------------------------------------- shipped


def guard(
    candidate: EloParams,
    auc_of: Callable[[EloParams, str], float | None],
    datasets: Sequence[str] = GUARDED,
    v1: EloParams = V1_PARAMS,
    tolerance: float = TOLERANCE,
    fields: Sequence[str] = GUARD_FIELDS,
) -> tuple[EloParams, list[str]]:
    """Never ship a set that is worse than v1 by more than `tolerance` AUC on a guarded dataset.

    `auc_of(params, dataset)` gives the held-out AUC (None when the dataset was not run). While the
    candidate regresses on any guarded dataset, the one tuned parameter whose return to its v1 value
    helps most is put back, and the check runs again. Returns the kept set and a note per change."""
    notes: list[str] = []

    def worst(p: EloParams) -> float:
        gaps = []
        for d in datasets:
            a, b = auc_of(p, d), auc_of(v1, d)
            if a is not None and b is not None:
                gaps.append(a - b)
        return min(gaps) if gaps else 0.0

    current = candidate
    while worst(current) < -tolerance:
        changed = [f for f in fields if getattr(current, f) != getattr(v1, f)]
        if not changed:
            break
        options = [(current.replace(**{f: getattr(v1, f)}), f) for f in changed]
        best, field = max(options, key=lambda o: worst(o[0]))
        notes.append(
            f"{field} back to the v1 value {getattr(v1, field)!r}: the tuned set was worse than v1 by "
            f"{-worst(current):.4f} AUC on a guarded dataset (tolerance {tolerance})"
        )
        current = best
    return current, notes


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def choose_shipped(fits: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The shipped defaults and how they were chosen, from the fit records (`build/fit/<key>.json`).

    Only commercial-OK datasets decide anything; the rule is in the module notes."""
    for key, rec in fits.items():
        if key not in SHIP_FROM + TAU_FROM:
            continue
        own = rec.get("dataset", key)
        if own != key or not registry.commercial_ok(own) or not rec.get("commercial_params_ok", False):
            raise ValueError(
                f"{key}: the fit record is of {own!r}, not commercial-OK, so it cannot choose shipped parameters"
            )
    tau_sets = [k for k in TAU_FROM if k in fits and fits[k]["timestamps"] and registry.commercial_ok(k)]
    ship_sets = [k for k in SHIP_FROM if k in fits and registry.commercial_ok(k)]
    if not ship_sets:
        raise ValueError(f"no fit record for any of {', '.join(SHIP_FROM)}: nothing to choose from")
    notes: list[str] = []
    # tau: best mean validation AUC at the v1 step sizes over the commercial-OK timestamped sets
    if tau_sets:
        curves = {
            k: {r["params"]["tau_days"]: r["val"]["auc"] for r in tau_curve(fits[k]["grid_results"])} for k in tau_sets
        }
        taus = [t for t in curves[tau_sets[0]] if all(t in c for c in curves.values())]
        tau_scores = {t: _mean([curves[k][t] for k in tau_sets]) for t in taus}
        tau = max(taus, key=lambda t: tau_scores[t])
    else:
        tau, tau_scores = EloParams().tau_days, {}
        notes.append("no commercial-OK timestamped dataset was run: tau keeps the PRD default of 45 days")
    # step sizes: best mean validation AUC over the L1 to L3 sets, at the shipped tau where times exist
    triples: dict[tuple[float, ...], list[float]] = {}
    for k in ship_sets:
        ts = fits[k]["timestamps"]
        for r in fits[k]["grid_results"]:
            if ts and r["params"]["tau_days"] != tau:
                continue
            triples.setdefault(tuple(r["params"][f] for f in STEP_FIELDS), []).append(r["val"]["auc"])
    complete = {t: v for t, v in triples.items() if len(v) == len(ship_sets)}
    if not complete:
        raise ValueError("the fit records share no grid configuration")
    best = max(complete, key=lambda t: _mean(complete[t]))
    candidate = EloParams(**dict(zip(STEP_FIELDS, best)), tau_days=tau)

    def auc_of(p: EloParams, d: str) -> float | None:
        fit = fits.get(d)
        if fit is None:
            return None
        found = lookup(fit["grid_results"], p, fit["timestamps"])
        return None if found is None else float(found["test"]["auc"])

    shipped, guard_notes = guard(candidate, auc_of)
    notes.extend(guard_notes)
    missing = [k for k in SHIP_FROM if k not in fits]
    if missing:
        notes.append(f"not run, so not part of the choice: {', '.join(missing)}")
    held_out = {}
    for k in sorted(set(ship_sets) | set(GUARDED) & set(fits)):
        a, b = auc_of(shipped, k), auc_of(V1_PARAMS, k)
        if a is not None and b is not None:
            held_out[k] = {"auc_v1_defaults": round(b, 5), "auc_shipped": round(a, 5), "gain": round(a - b, 5)}
    informing = {}
    for k in TAU_INFORM:
        if k in fits and fits[k]["timestamps"]:
            informing[k] = {"would_choose_tau_days": choose_tau(tau_curve(fits[k]["grid_results"]))}
    rule = (
        "tau: the value with the best mean validation AUC at the v1 step sizes on "
        + (", ".join(tau_sets) if tau_sets else "no timestamped set")
        + "; k_s, k_k, decay: the triple with the best mean validation AUC across "
        + ", ".join(ship_sets)
        + f" at that tau; then never worse than the v1 defaults by more than {TOLERANCE} AUC on the "
        f"held-out students of {', '.join(GUARDED)} (a parameter that breaks this goes back to its v1 value); "
        "grade_step and transfer keep the v1 values because the public datasets have no prerequisite graph; "
        "streak_to_master and streak_expiry_days are not fitted (PRD defaults)"
    )
    return {
        "params": shipped.to_dict(),
        "candidate_before_guard": candidate.to_dict(),
        "rule": rule,
        "chosen_on": ship_sets,
        "tau_chosen_on": tau_sets,
        "tau_mean_val_auc": {str(t): round(v, 5) for t, v in tau_scores.items()},
        "step_mean_val_auc": {str(list(t)): round(_mean(v), 5) for t, v in sorted(complete.items())},
        "held_out": held_out,
        "informing_only": informing,
        "notes": notes,
    }


# --------------------------------------------------------------------------------------------- params file


def write_params(params: EloParams, provenance: Mapping[str, Any], path: str | Path = PARAMS_PATH) -> Path:
    """Write the shipped defaults: `EloParams.to_dict()` plus a `provenance` block.

    Refuses a provenance that names a dataset which is not commercial-OK (PRD 5.3), or none at all."""
    datasets = list(provenance.get("datasets") or [])
    if not datasets:
        raise ValueError("the provenance must list the datasets the parameters were chosen on")
    bad = [d for d in datasets if not registry.commercial_ok(d)]
    if bad:
        raise ValueError(f"parameters fitted on non-commercial data cannot be shipped: {', '.join(bad)}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {**params.to_dict(), "provenance": {**provenance, "datasets": datasets}}
    path.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    return path


def load_params(path: str | Path = PARAMS_PATH) -> EloParams:
    """Read the shipped defaults file (`EloParams.from_dict` ignores the `provenance` block)."""
    return EloParams.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# --------------------------------------------------------------------------------------------- report


def load_fits(build_dir: str | Path = BUILD_DIR) -> dict[str, dict[str, Any]]:
    """Every `build/fit/<key>.json` record, keyed by results key."""
    build = Path(build_dir)
    fits = {}
    for key in DATASETS:
        p = build / f"{key}.json"
        if p.is_file():
            fits[key] = json.loads(p.read_text(encoding="utf-8"))
    return fits


_DATASET_FIELDS = (
    "licence",
    "commercial_params_ok",
    "rows",
    "students",
    "items",
    "skills",
    "timestamps",
    "subsample",
    "split",
    "split_rows",
    "tau_grid",
    "tau_curve",
    "tau_chosen",
    "auc_without_forgetting",
    "auc_with_forgetting",
    "auc_change_from_forgetting",
    "forgetting_check",
    "grid",
    "grid_chosen_params",
    "auc_v1_defaults",
    "auc_chosen",
    "rmse_v1_defaults",
    "rmse_chosen",
    "val_auc_v1_defaults",
    "val_auc_chosen",
    "seconds",
    "fitted_at",
)


def dataset_block(fit: Mapping[str, Any], top: int = 5) -> dict[str, Any]:
    """The results-file view of one fit record: everything but the full grid (the top `top` stay)."""
    block = {k: fit[k] for k in _DATASET_FIELDS if k in fit}
    block["grid_top"] = [dict(r) for r in fit["grid_results"][:top]]
    block["item_bank"] = {k: v for k, v in fit["item_bank"].items() if k != "path"}
    return block


def build_block(fits: Mapping[str, Mapping[str, Any]], shipped: Mapping[str, Any]) -> dict[str, Any]:
    """The `learner_fit` block."""
    no_time = [k for k, f in fits.items() if not f["timestamps"]]
    worst_drop = min((fits[k]["auc_change_from_forgetting"] for k in no_time), default=0.0)
    return {
        "schema": SCHEMA,
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "protocol": SPLIT,
        "engine": ENGINE_NOTE,
        "tau_grid": [_tau_json(t) for t in TAU_GRID],
        "grid": {k: list(v) for k, v in GRID.items()},
        "v1_defaults": V1_PARAMS.to_dict(),
        "datasets": {k: dataset_block(f) for k, f in fits.items()},
        "shipped": dict(shipped),
        "acceptance": {
            "rule": "forgetting never lowers AUC by more than 0.002 on datasets without timestamps",
            "datasets_without_timestamps": no_time,
            "largest_auc_drop_from_forgetting": round(worst_drop, 5),
            "passed": worst_drop >= -0.002,
        },
        "item_banks": {
            k: {
                "saved": f["item_bank"].get("saved", False),
                "path": f["item_bank"].get("path"),
                "items": f["item_bank"]["items"],
                "auc_without_bank": f["item_bank"]["auc_without_bank"],
                "auc_with_bank": f["item_bank"]["auc_with_bank"],
            }
            for k, f in fits.items()
        },
        "notes": [
            "grade_step and transfer act only through a prerequisite graph; "
            "the public datasets have none, so they keep the v1 values",
            "the grid's banks are fitted on the inner fit students; the saved banks on all the training students",
            "the maths bundle ships without an item bank: Eedi and ASSISTments item ids do not map to Kwan items",
        ],
    }


def report(
    build_dir: str | Path = BUILD_DIR,
    out: str | Path | None = None,
    params_path: str | Path | None = PARAMS_PATH,
) -> dict[str, Any]:
    """Assemble the fit records, choose the shipped defaults, write the params file and the block."""
    from kwan.eval import results

    fits = load_fits(build_dir)
    if not fits:
        raise FileNotFoundError(f"no fit records under {build_dir} (run `kwan fit all` first)")
    shipped = choose_shipped(fits)
    if params_path:
        provenance = {
            "subject": SUBJECT,
            "datasets": list(shipped["chosen_on"]),
            "tau_datasets": list(shipped["tau_chosen_on"]),
            "rule": shipped["rule"],
            "held_out_auc": shipped["held_out"],
            "notes": list(shipped["notes"]),
            "fitted_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "results_key": KEY,
        }
        shipped["params_file"] = str(write_params(EloParams.from_dict(shipped["params"]), provenance, params_path))
    block = build_block(fits, shipped)
    if out:
        results.update(KEY, block, path=out)
    return block


# --------------------------------------------------------------------------------------------- CLI


def _log(s: str) -> None:
    print(s, file=sys.stderr, flush=True)


def _cmd_dataset(args: argparse.Namespace) -> int:
    try:
        run_dataset(args.dataset, args.path, max_rows=args.max_rows, build_dir=args.build, raw_dir=args.raw, log=_log)
    except FileNotFoundError as exc:
        print(f"kwan fit: {exc}", file=sys.stderr)
        return 1
    return 0


def _cmd_all(args: argparse.Namespace) -> int:
    only = (
        [BY_CLI.get(n.strip(), n.strip()) for n in args.only.split(",") if n.strip()] if args.only else list(DATASETS)
    )
    for key in only:
        if key not in DATASETS:
            print(f"kwan fit all: unknown dataset '{key}'", file=sys.stderr)
            return 2
        try:
            run_dataset(key, max_rows=args.max_rows, build_dir=args.build, raw_dir=args.raw, log=_log)
        except FileNotFoundError as exc:
            _log(f"skipping {key}: {exc}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    block = report(args.build, args.out, args.params)
    print(json.dumps(block["shipped"]["params"], indent=1))
    return 0


def register(subparsers: Any) -> None:
    """Add `fit` to the `kwan` command."""
    fit = subparsers.add_parser("fit", help="fit tau, the Elo step sizes and item banks on the public datasets")
    _add_commands(fit.add_subparsers(dest="fit_command", metavar="WHAT", required=True))


def _add_commands(sub: Any) -> None:
    """The `dataset`, `all` and `report` commands."""
    for name, func, help_ in (
        ("dataset", _cmd_dataset, "fit one dataset and write build/fit/<key>.json"),
        ("all", _cmd_all, "fit every dataset whose raw files are present"),
    ):
        p = sub.add_parser(name, help=help_)
        if name == "dataset":
            p.add_argument("dataset", choices=sorted(BY_CLI), help="which dataset")
            p.add_argument("path", nargs="?", help="its raw file or directory (default: under --raw)")
        else:
            p.add_argument("--only", metavar="NAMES", help="comma-separated names, e.g. eedi-34,ammore")
        p.add_argument("--raw", default="raw", metavar="DIR", help="the raw-data directory (default: raw)")
        p.add_argument(
            "--build", default=str(BUILD_DIR), metavar="DIR", help="where fit records go (default: build/fit)"
        )
        p.add_argument(
            "--max-rows", type=int, default=DEFAULT_MAX_ROWS, metavar="N", help="subsample larger datasets by student"
        )
        p.set_defaults(func=func, extra="eval")
    p = sub.add_parser("report", help="choose the shipped defaults, write the params file and the results block")
    p.add_argument(
        "--build", default=str(BUILD_DIR), metavar="DIR", help="where the fit records are (default: build/fit)"
    )
    p.add_argument("--out", metavar="PATH", help="merge the learner_fit block into this results file")
    p.add_argument("--params", default=str(PARAMS_PATH), metavar="PATH", help="the params file to write")
    p.set_defaults(func=_cmd_report, extra=None)


def main(argv: Sequence[str] | None = None) -> int:
    """`python -m kwan.eval.fit <what> ...`, the same commands as `kwan fit`."""
    parser = argparse.ArgumentParser(prog="kwan fit")
    _add_commands(parser.add_subparsers(dest="fit_command", metavar="WHAT", required=True))
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
