"""Deep knowledge tracing benchmark: pyKT models against Kwan's elo-multi on the same folds (PRD 5.6, 5.7).

What it does, per dataset:

1. Converts Kwan's interaction form (`user_id, problem_id, skill_id, correct`, ordered per student) to
   pyKT's sequence format: one row per sequence of at most `maxlen` answers, padded with -1, tails
   shorter than `min_seq_len` dropped (the same rule as `pykt.preprocess.split_datasets.generate_sequences`).
2. Makes 5 folds by student (seed 42). Each fold is held out once.
3. Trains DKT, SAKT, simpleKT and AKT (pyKT model classes, pyKT default hyperparameters) on the training
   students of the fold and scores next-answer AUC, RMSE and accuracy on the held-out students.
4. Scores elo-multi (`kwan.eval.kt.Elo2`) on exactly the same held-out answers.
5. Times one prediction step on CPU for every model.
6. Applies adoption rule D7 mechanically and writes the `kt_deep` block of `data/eval_results.json`.

Only the standard library is imported at module level. numpy, pandas and scikit-learn (extra `eval`) are
imported when a function needs them; torch and pyKT (extra `kt`) only when a deep model is trained or timed.

Licence note: AMMORE is CC BY-NC 4.0. Model weights fitted on it stay under the work directory
(`build/kt/`, git-ignored) and are deleted after scoring unless `--keep-checkpoints` is given.
Only aggregate metrics leave that directory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

KEY = "kt_deep"  # top-level key this module owns in data/eval_results.json
ELO = "elo-multi (Kwan)"  # same results-file name as kwan.eval.kt.KWAN
MODELS = ("dkt", "sakt", "simplekt", "akt")
SEED = 42
FOLDS = 5
MAXLEN = 200  # pyKT's default sequence length (configs/kt_config.json: train_config.seq_len)
MIN_SEQ_LEN = 3  # pyKT's default minimum sequence length
PAD = -1  # pyKT's pad value
VAL_FRACTION = 0.1  # share of each fold's training students used only to pick the best epoch
DEFAULT_WORK = "build/kt"

# pyKT defaults: argparse defaults of examples/wandb_<model>_train.py at the installed pyKT commit, and the
# batch sizes examples/wandb_train.py applies (256, or 64 for the attention models). Optimiser: Adam.
PYKT_DEFAULTS: dict[str, dict[str, Any]] = {
    "dkt": {"learning_rate": 1e-3, "batch_size": 256, "model_config": {"dropout": 0.2, "emb_size": 200}},
    "sakt": {
        "learning_rate": 1e-3,
        "batch_size": 64,
        "model_config": {"dropout": 0.2, "emb_size": 256, "num_attn_heads": 8, "num_en": 1},
        "needs_seq_len": True,
    },
    "simplekt": {
        "learning_rate": 1e-4,
        "batch_size": 64,
        "model_config": {
            "dropout": 0.1, "final_fc_dim": 256, "final_fc_dim2": 256, "num_layers": 2, "nheads": 4,
            "loss1": 0.5, "loss2": 0.5, "loss3": 0.5, "start": 50, "d_model": 256, "d_ff": 256,
            "num_attn_heads": 4, "n_blocks": 2,
        },
        "needs_seq_len": True,
    },
    "akt": {
        "learning_rate": 1e-4,
        "batch_size": 64,
        "model_config": {"dropout": 0.2, "d_model": 256, "d_ff": 512, "num_attn_heads": 8, "n_blocks": 4},
    },
}  # fmt: skip

# name -> how to load it. `path` is relative to the raw data directory. `d7` marks the three datasets of rule D7.
DATASETS: dict[str, dict[str, Any]] = {
    "assistments_2009": {
        "title": "ASSISTments 2009 skill-builder (corrected)",
        "licence": "Research use",
        "path": "skill_builder_data_corrected.csv",
        "d7": True,
    },
    "eedi_34": {
        "title": "Eedi NeurIPS 2020 Education Challenge, task 3/4 subset",
        "licence": "Research use, cite",
        "path": "eedi/data",
        "d7": True,
    },
    "ammore": {
        "title": "AMMORE (Rori WhatsApp maths tutor, Rising Academies), Jan to Apr 2024",
        "licence": "CC BY-NC 4.0 (evaluation only)",
        "path": "ammore",
        "d7": True,
    },
    "xes3g5m": {
        "title": "XES3G5M (Grade 3 maths, China)",
        "licence": "MIT",
        "path": "xes3g5m",
        "d7": False,
    },
}
D7_DATASETS = tuple(name for name, spec in DATASETS.items() if spec["d7"])
D7_MIN_MARGIN = 0.03
D7_MIN_DATASETS = 2
D7_MAX_MS = 20.0
D7_RULE = (
    "D7: a deep model is adopted only if it beats elo-multi by at least 0.03 AUC on at least two of "
    "AMMORE, Eedi and ASSISTments 2009, and stays under 20 ms per update on CPU."
)


# --------------------------------------------------------------------------------------------------
# Folds and sequence conversion (no torch, no pyKT)
# --------------------------------------------------------------------------------------------------


def _plain(value: Any) -> Any:
    """A numpy scalar as a plain Python value, anything else unchanged."""
    return value.item() if hasattr(value, "item") else value


def make_folds(user_ids: Any, k: int = FOLDS, seed: int = SEED) -> dict[Any, int]:
    """Assign every student to one of `k` folds. Returns {user_id: fold}.

    The result depends only on the set of ids and the seed, never on the order of the rows:
    ids are sorted, shuffled with `random.Random(seed)` and dealt out in turn, so fold sizes
    differ by at most one student.
    """
    if k < 2:
        raise ValueError("need at least 2 folds")
    users = sorted({_plain(u) for u in user_ids})
    if len(users) < k:
        raise ValueError(f"cannot make {k} folds from {len(users)} students")
    random.Random(seed).shuffle(users)
    return {u: i % k for i, u in enumerate(users)}


def validation_users(train_users: Any, fraction: float = VAL_FRACTION, seed: int = SEED) -> set[Any]:
    """A seeded share of the training students, held back to pick the best epoch (never the test fold)."""
    users = sorted({_plain(u) for u in train_users})
    random.Random(seed).shuffle(users)
    n = max(1, round(fraction * len(users))) if len(users) > 1 else 0
    return set(users[:n])


def to_pykt_sequences(df: Any, maxlen: int = MAXLEN, min_seq_len: int = MIN_SEQ_LEN, pad: int = PAD) -> dict[str, Any]:
    """Convert Kwan's interaction form to pyKT's sequence format.

    `df` has columns `user_id, problem_id, skill_id, correct`; the rows of one student are already in
    answer order (they do not have to be next to each other). Item and skill ids are re-indexed from 0,
    as pyKT does. A student's answers are cut into consecutive pieces of `maxlen`; the last piece is
    padded with `pad`, or dropped when it is shorter than `min_seq_len`.

    Returns a dict of arrays, one row per sequence:
    `uid` (n,), `questions`, `concepts`, `responses`, `selectmasks`, `rows` (all n x maxlen; `rows` holds
    the position of each answer in `df`, `selectmasks` is 1 for an answer and `pad` for padding), plus
    `num_q`, `num_c`, `maxlen`, `pad` and `dropped` (answers lost to the `min_seq_len` rule).
    """
    import numpy as np
    import pandas as pd

    ucodes, uniq = pd.factorize(df["user_id"].to_numpy())  # students in order of first appearance
    q = pd.factorize(df["problem_id"].to_numpy(), sort=True)[0].astype(np.int64)
    c = pd.factorize(df["skill_id"].to_numpy(), sort=True)[0].astype(np.int64)
    r = df["correct"].to_numpy().astype(np.int64)
    if len(r) and not np.isin(r, (0, 1)).all():
        raise ValueError("`correct` must be 0 or 1")
    order = np.argsort(ucodes, kind="stable")  # groups students, keeps each student's answer order
    counts = np.bincount(ucodes, minlength=len(uniq))
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)

    uid, rows, dropped = [], [], 0
    for u in range(len(uniq)):
        idx = order[starts[u] : starts[u] + counts[u]]
        j = 0
        while len(idx) >= j + maxlen:
            uid.append(uniq[u])
            rows.append(idx[j : j + maxlen])
            j += maxlen
        rest = len(idx) - j
        if rest < min_seq_len:
            dropped += rest
            continue
        uid.append(uniq[u])
        rows.append(np.concatenate([idx[j:], np.full(maxlen - rest, pad, dtype=np.int64)]))

    rows_arr = np.stack(rows).astype(np.int64) if rows else np.empty((0, maxlen), dtype=np.int64)
    real = rows_arr != pad
    safe = np.where(real, rows_arr, 0)

    def take(values: Any) -> Any:
        return np.where(real, values[safe], pad) if len(values) else np.full(rows_arr.shape, pad, dtype=np.int64)

    return {
        "uid": np.asarray(uid, dtype=uniq.dtype if len(uid) else object),
        "questions": take(q),
        "concepts": take(c),
        "responses": take(r),
        "selectmasks": np.where(real, 1, pad).astype(np.int64),
        "rows": rows_arr,
        "num_q": int(q.max()) + 1 if len(q) else 0,
        "num_c": int(c.max()) + 1 if len(c) else 0,
        "maxlen": maxlen,
        "pad": pad,
        "dropped": int(dropped),
    }


def to_pykt_frame(seqs: dict[str, Any], folds: dict[Any, int] | None = None) -> Any:
    """The sequences as a pyKT sequence file: one row per sequence, comma-joined id strings.

    Columns `fold, uid, questions, concepts, responses, selectmasks`, the layout pyKT's `KTDataset` reads
    from `train_valid_sequences.csv`. Kwan trains from memory and never writes this file; it is here so
    the format can be checked and so the same data can be handed to pyKT's own scripts.
    """
    import pandas as pd

    def joined(key: str) -> list[str]:
        return [",".join(str(int(v)) for v in row) for row in seqs[key]]

    uid = [_plain(u) for u in seqs["uid"]]
    return pd.DataFrame(
        {
            "fold": [folds[u] if folds is not None else 0 for u in uid],
            "uid": uid,
            "questions": joined("questions"),
            "concepts": joined("concepts"),
            "responses": joined("responses"),
            "selectmasks": joined("selectmasks"),
        }
    )


def scored_rows(seqs: dict[str, Any], seq_idx: Any) -> Any:
    """Positions in the source frame of the answers pyKT scores for these sequences.

    pyKT predicts answers 2..n of each sequence from the answers before them, so the first answer of
    every sequence is input only. The order matches a row-major walk over the sequences, which is the
    order `torch.masked_select` returns predictions in.
    """
    rows = seqs["rows"][seq_idx][:, 1:]
    return rows[seqs["selectmasks"][seq_idx][:, 1:] != seqs["pad"]]


# --------------------------------------------------------------------------------------------------
# Metrics, elo-multi, summaries, decision D7, results block (no torch, no pyKT)
# --------------------------------------------------------------------------------------------------


def score(y: Any, p: Any) -> dict[str, float]:
    """Next-answer AUC, RMSE and accuracy (threshold 0.5, as pyKT). Unrounded."""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return {
        "AUC": float(roc_auc_score(y, p)),
        "RMSE": float(math.sqrt(np.mean((y - p) ** 2))),
        "ACC": float(np.mean((p >= 0.5) == (y >= 0.5))),
    }


def elo_multi_fold(df: Any, test_mask: Any) -> tuple[Any, float]:
    """elo-multi on one fold, exactly as `kwan.eval.kt.score_models` runs it.

    Item difficulty is fitted on the training students (two passes); the held-out students are then
    predicted online with difficulty frozen. Returns (predictions for the held-out rows in frame order,
    milliseconds per predict-and-update step measured on the held-out pass).
    """
    from kwan.eval.kt import Elo2

    train, test = df[~test_mask], df[test_mask]
    model = Elo2()
    model.run(train, True)
    model.run(train, True)
    t0 = time.perf_counter()
    preds = model.run(test, False)
    ms = 1000.0 * (time.perf_counter() - t0) / max(1, len(test))
    return preds, ms


def _mean_sd(values: list[float]) -> tuple[float, float]:
    """Mean and sample standard deviation (0.0 when there is one value)."""
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def summarise(fold_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean and sample standard deviation across folds, rounded to 4 decimals."""
    out: dict[str, Any] = {}
    for metric in ("AUC", "RMSE", "ACC"):
        mean, sd = _mean_sd([float(rec[metric]) for rec in fold_records])
        out[f"{metric}_mean"] = round(mean, 4)
        out[f"{metric}_sd"] = round(sd, 4)
    out["folds"] = len(fold_records)
    return out


def d7_decision(
    datasets: dict[str, dict[str, Any]],
    min_margin: float = D7_MIN_MARGIN,
    min_datasets: int = D7_MIN_DATASETS,
    max_ms: float = D7_MAX_MS,
    rule_datasets: tuple[str, ...] = D7_DATASETS,
    baseline: str = ELO,
) -> dict[str, Any]:
    """Apply adoption rule D7 to the per-dataset model summaries of the results block.

    `datasets` is `{dataset: {"models": {model: {"AUC_mean": float, "ms_per_update_cpu": float}}}}`.
    A deep model meets the rule when (a) its AUC margin over the baseline is at least `min_margin` on at
    least `min_datasets` of `rule_datasets`, and (b) its slowest measured CPU step on those datasets is
    under `max_ms`. A missing number never counts in a model's favour: a model that was not run on a
    dataset has not won it. `conclusive` is false for a model whose outcome the datasets it was not run
    on could still change. When several models meet the rule, the one with the largest mean margin is named.
    """
    deep = sorted({m for name in rule_datasets for m in datasets.get(name, {}).get("models", {}) if m != baseline})
    per_model: dict[str, Any] = {}
    for model in deep:
        margins: dict[str, float] = {}
        times: list[float] = []
        for name in rule_datasets:
            models = datasets.get(name, {}).get("models", {})
            if model not in models or baseline not in models:
                continue
            margins[name] = round(float(models[model]["AUC_mean"]) - float(models[baseline]["AUC_mean"]), 4)
            ms = models[model].get("ms_per_update_cpu")
            if ms is not None:
                times.append(float(ms))
        won = [name for name, margin in margins.items() if margin >= min_margin]
        not_run = [name for name in rule_datasets if name not in margins]
        slowest = max(times) if times and len(times) == len(margins) else None
        accuracy_ok = len(won) >= min_datasets
        latency_ok = slowest is not None and slowest < max_ms
        # Could the datasets that were not run still change the outcome for this model?
        too_slow = slowest is not None and slowest >= max_ms
        conclusive = (accuracy_ok and latency_ok) or too_slow or len(won) + len(not_run) < min_datasets
        per_model[model] = {
            "margins": margins,
            "datasets_won": won,
            "datasets_not_run": not_run,
            "accuracy_ok": accuracy_ok,
            "ms_per_update_cpu": slowest,
            "latency_ok": latency_ok,
            "meets_rule": accuracy_ok and latency_ok,
            "conclusive": conclusive,
        }
    winners = [m for m in deep if per_model[m]["meets_rule"]]
    best = max(winners, key=lambda m: statistics.fmean(per_model[m]["margins"].values())) if winners else None
    evaluated = [name for name in rule_datasets if baseline in datasets.get(name, {}).get("models", {})]
    if best is not None:
        reason = f"{best} meets rule D7; it may ship as an optional backend and Elo stays the default."
    elif not deep:
        reason = "No deep model was evaluated on the D7 datasets, so none is adopted."
    else:
        top = max(
            deep, key=lambda m: (len(per_model[m]["datasets_won"]), max(per_model[m]["margins"].values(), default=-1.0))
        )
        margins = per_model[top]["margins"]
        best_margin = (
            f"{max(margins.values()):+.4f} on {max(margins, key=lambda name: margins[name])}" if margins else "none"
        )
        reason = (
            f"No deep model meets rule D7. The closest is {top}: at least {min_margin} AUC ahead of {baseline} on "
            f"{len(per_model[top]['datasets_won'])} of the {len(margins)} D7 datasets it was run on "
            f"(needs {min_datasets}); its best margin is {best_margin}. Elo stays the only learner model."
        )
    open_models = [m for m in deep if not per_model[m]["conclusive"]]
    if best is None and open_models:
        reason += f" Not conclusive for {', '.join(open_models)}: not run on every D7 dataset."
    return {
        "rule": D7_RULE,
        "thresholds": {"min_auc_margin": min_margin, "min_datasets": min_datasets, "max_ms_per_update_cpu": max_ms},
        "baseline": baseline,
        "datasets_in_rule": list(rule_datasets),
        "datasets_evaluated": evaluated,
        "adopt": best is not None,
        "model": best,
        "conclusive": best is not None or not open_models,
        "per_model": per_model,
        "reason": reason,
    }


def _same(values: list[Any]) -> Any:
    """The single value when all are equal, else the list (so a mixed budget is visible, never hidden)."""
    return values[0] if all(v == values[0] for v in values) else values


def model_summary(fold_records: list[dict[str, Any]], latency: dict[str, Any] | None = None) -> dict[str, Any]:
    """One model on one dataset: fold means, the per-fold AUCs and, for deep models, the budget used."""
    recs = sorted(fold_records, key=lambda rec: rec["fold"])
    out = summarise(recs)
    out["AUC_folds"] = [round(float(rec["AUC"]), 4) for rec in recs]
    out["scored_answers"] = int(sum(rec["n_scored"] for rec in recs))
    if recs[0].get("kind") == "elo":
        # timed on its own when the block is built; the fold runs only have a rougher wall-time figure
        ms = latency["ms_median"] if latency else statistics.fmean(float(rec["ms_per_update_cpu"]) for rec in recs)
        out["ms_per_update_cpu"] = round(float(ms), 4)
        if latency and "ms_wall_median" in latency:
            out["ms_per_update_wall"] = round(float(latency["ms_wall_median"]), 4)
        out["AUC_mean_all_heldout_answers"] = round(statistics.fmean(float(rec["AUC_all"]) for rec in recs), 4)
        out["all_heldout_answers"] = int(sum(rec["n_all"] for rec in recs))
    else:
        out["ms_per_update_cpu"] = round(float(latency["ms_median"]), 3) if latency else None
        if latency:
            out["ms_per_update_cpu_p95"] = round(float(latency["ms_p95"]), 3)
            if "ms_min" in latency:
                out["ms_per_update_cpu_min"] = round(float(latency["ms_min"]), 3)
            if "ms_wall_median" in latency:
                out["ms_per_update_wall"] = round(float(latency["ms_wall_median"]), 3)
        out["epochs_max"] = _same([rec["config"]["epochs"] for rec in recs])
        out["epochs_run"] = [int(rec["epochs_run"]) for rec in recs]
        out["best_epoch"] = [int(rec["best_epoch"]) for rec in recs]
        # folds that trained right up to the cap: more epochs could still have helped there
        out["folds_stopped_by_epoch_cap"] = sum(int(rec["epochs_run"]) >= int(rec["config"]["epochs"]) for rec in recs)
        out["train_minutes"] = round(sum(float(rec["train_seconds"]) for rec in recs) / 60.0, 1)
        out["device"] = _same([rec["config"]["device"] for rec in recs])
        out["trainer"] = _same([rec["config"]["trainer"] for rec in recs])
    return out


def make_block(
    datasets: dict[str, dict[str, Any]],
    protocol: dict[str, Any],
    versions: dict[str, Any],
    latency_protocol: dict[str, Any] | None = None,
    skipped: dict[str, str] | None = None,
    implementation: str | None = None,
) -> dict[str, Any]:
    """The `kt_deep` block of data/eval_results.json.

    `datasets` is `{name: {"meta": {...}, "folds": {model: [fold record, ...]}, "latency": {model: {...}}}}`
    where a fold record carries `fold, AUC, RMSE, ACC, n_scored` (and the training facts for deep models).
    """
    out_datasets: dict[str, Any] = {}
    for name, entry in datasets.items():
        models: dict[str, Any] = {}
        for model in (ELO, *MODELS):
            if entry["folds"].get(model):
                models[model] = model_summary(entry["folds"][model], entry.get("latency", {}).get(model))
        out_datasets[name] = {**entry["meta"], "models": models}
    deep_summaries = [s for d in out_datasets.values() for m, s in d["models"].items() if m != ELO]
    budget = {
        "epochs_max_per_fit": sorted(
            {
                e
                for s in deep_summaries
                for e in (s["epochs_max"] if isinstance(s["epochs_max"], list) else [s["epochs_max"]])
            }
        ),
        "device": sorted(
            {str(d) for s in deep_summaries for d in (s["device"] if isinstance(s["device"], list) else [s["device"]])}
        ),
        "train_minutes_total": round(sum(s["train_minutes"] for s in deep_summaries), 1),
        "pykt_reference_budget": "200 epochs with early stopping (patience 10) on a GPU",
        "note": (
            "Reduced budget: every fit was capped at far fewer epochs than pyKT's reference runs. Where a fit was "
            "stopped by the cap, the model's AUC is likely below what full training would reach. The cap, the epochs "
            "run and the count of folds stopped by the cap are under datasets.<name>.models.<model>. "
            "train_minutes_total adds up fits that ran side by side on one shared GPU."
        ),
    }
    decision = d7_decision(out_datasets)
    skipped = dict(skipped or {})
    if not decision["adopt"]:
        skipped.setdefault("onnx_backend", "not built: no deep model met rule D7, so Elo stays the only learner model")
    return {
        "task": "PRD 5.6 and 5.7: deep knowledge tracing (pyKT) against elo-multi on the same folds",
        "implementation": implementation or "pyKT model classes (not re-implementations)",
        "protocol": protocol,
        "budget": budget,
        "hyperparameters": {m: {k: v for k, v in PYKT_DEFAULTS[m].items() if k != "needs_seq_len"} for m in MODELS},
        "latency_protocol": latency_protocol or {},
        "versions": versions,
        "datasets": out_datasets,
        "skipped": skipped,
        "decision": decision,
    }


# --------------------------------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------------------------------


def load_dataset(name: str, raw: str | os.PathLike[str] = "raw") -> Any:
    """Load one dataset in Kwan's interaction form, rows of each student in answer order."""
    path = Path(raw) / DATASETS[name]["path"]
    if not path.exists():
        raise FileNotFoundError(f"{name}: {path} is missing; see scripts/download_data.sh")
    if name == "assistments_2009":
        from kwan.eval.kt import load

        return load(str(path))
    if name == "eedi_34":
        from kwan.eval.eedi import load_eedi

        return load_eedi(str(path))
    if name == "ammore":
        from kwan.eval.ammore import load_ammore

        return load_ammore(str(path))
    if name == "xes3g5m":
        return load_xes3g5m(path)
    raise KeyError(name)


def load_xes3g5m(root: str | os.PathLike[str]) -> Any:
    """XES3G5M in Kwan's interaction form, from the question-level sequence files of the official release.

    Reads `question_level/train_valid_sequences_quelevel.csv` (pieces of at most 200 answers, padded
    with -1) and `question_level/test_quelevel.csv` (one full sequence per student). The pieces of one
    student are joined and ordered by timestamp, file order breaking ties. A question tagged with several
    concepts (`166_170`) keeps the first one as its skill. The release's own folds are ignored: Kwan
    makes its own folds by student over all students.
    """
    import numpy as np
    import pandas as pd

    base = next((p for p in (Path(root), *sorted(Path(root).glob("*"))) if (p / "question_level").is_dir()), None)
    if base is None:
        raise FileNotFoundError(f"no question_level/ directory under {root}")
    cols = ["uid", "questions", "concepts", "responses", "timestamps"]
    users, items, skills, correct, stamps = [], [], [], [], []
    for fname in ("train_valid_sequences_quelevel.csv", "test_quelevel.csv"):
        frame = pd.read_csv(base / "question_level" / fname, usecols=cols)
        for uid, qs, cs, rs, ts in zip(frame.uid, frame.questions, frame.concepts, frame.responses, frame.timestamps):
            r = np.array(rs.split(","), dtype=np.int64)
            keep = r != -1
            users.append(np.full(int(keep.sum()), int(uid), dtype=np.int64))
            items.append(np.array(qs.split(","), dtype=np.int64)[keep])
            skills.append(np.array([int(part.split("_")[0]) for part in cs.split(",")], dtype=np.int64)[keep])
            correct.append(r[keep])
            stamps.append(np.array(ts.split(","), dtype=np.int64)[keep])
    df = pd.DataFrame(
        {
            "user_id": np.concatenate(users),
            "problem_id": np.concatenate(items),
            "skill_id": np.concatenate(skills),
            "correct": np.concatenate(correct),
            "t": np.concatenate(stamps),
        }
    )
    return df.sort_values(["user_id", "t"], kind="stable").reset_index(drop=True)


def dataset_meta(name: str, df: Any, seqs: dict[str, Any]) -> dict[str, Any]:
    """Counts reported with every dataset block."""
    return {
        "dataset": DATASETS[name]["title"],
        "licence": DATASETS[name]["licence"],
        "rows": int(len(df)),
        "students": int(df["user_id"].nunique()),
        "items": int(df["problem_id"].nunique()),
        "skills": int(df["skill_id"].nunique()),
        "sequences": int(len(seqs["uid"])),
        "answers_dropped_by_min_seq_len": int(seqs["dropped"]),
    }


# --------------------------------------------------------------------------------------------------
# pyKT training and timing (torch and pyKT are imported here, not above)
# --------------------------------------------------------------------------------------------------


def _pykt(device: str) -> Any:
    """Import torch and pyKT and point pyKT at `device`.

    pyKT picks "cuda or cpu" once, in module globals of its model, training and evaluation files.
    Setting those globals is the only way to run its models on Apple MPS, and it is how the CPU is
    selected again for the timing runs.
    """
    import pykt.models  # noqa: F401  (imports every model module, so every `device` global exists)
    import torch

    dev = torch.device(device)
    for mod_name, mod in list(sys.modules.items()):
        if mod_name.startswith("pykt") and hasattr(mod, "device"):
            mod.device = dev
    return torch


def library_versions() -> dict[str, Any]:
    """Versions of everything that affects the numbers."""
    import platform
    from importlib import metadata

    out: dict[str, Any] = {"python": platform.python_version(), "platform": f"{platform.system()} {platform.machine()}"}
    for pkg in ("pykt-toolkit", "torch", "numpy", "pandas", "scikit-learn"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    try:
        direct = metadata.distribution("pykt-toolkit").read_text("direct_url.json")
        if direct:
            info = json.loads(direct)
            out["pykt-toolkit_source"] = f"{info.get('url')}@{info.get('vcs_info', {}).get('commit_id')}"
    except (metadata.PackageNotFoundError, ValueError):
        pass
    return out


def _dataset(seqs: dict[str, Any], seq_idx: Any, torch: Any, device: str) -> Any:
    """A pyKT `KTDataset` over these sequences, built in memory instead of from a sequence file.

    The tensors are the ones `KTDataset.__load_data__` builds (`qseqs, cseqs, rseqs, smasks, masks`), and
    pyKT's own `__getitem__` serves them. They live on the training device, as they do in pyKT on CUDA.
    """
    from pykt.datasets.data_loader import KTDataset
    from torch.utils.data import Dataset

    pad = seqs["pad"]
    c = torch.as_tensor(seqs["concepts"][seq_idx], dtype=torch.long, device=device)
    empty = torch.empty(0, dtype=torch.long, device=device)
    ds = KTDataset.__new__(KTDataset)
    Dataset.__init__(ds)
    ds.input_type, ds.qtest = ["questions", "concepts"], False
    ds.dori = {
        "qseqs": torch.as_tensor(seqs["questions"][seq_idx], dtype=torch.long, device=device),
        "cseqs": c,
        "rseqs": torch.as_tensor(seqs["responses"][seq_idx], dtype=torch.float32, device=device),
        "tseqs": empty,
        "utseqs": empty,
        "smasks": torch.as_tensor(seqs["selectmasks"][seq_idx][:, 1:] != pad, device=device),
        "masks": (c[:, :-1] != pad) * (c[:, 1:] != pad),
    }
    return ds


def _forward(model: Any, name: str, dcur: dict[str, Any], torch: Any, train: bool = False) -> tuple[Any, Any]:
    """Next-answer probabilities for one batch, shape (batch, maxlen - 1), and any extra loss term.

    The calls are the ones `pykt.models.evaluate_model.evaluate` and `train_model.model_forward` make.
    """
    q, c, r = dcur["qseqs"].long(), dcur["cseqs"].long(), dcur["rseqs"].long()
    qs, cs, rs = dcur["shft_qseqs"].long(), dcur["shft_cseqs"].long(), dcur["shft_rseqs"].long()
    if name == "dkt":
        y = model(c, r)
        return (y * torch.nn.functional.one_hot(cs, model.num_c)).sum(-1), 0.0
    if name == "sakt":
        return model(c, r, cs), 0.0
    if name == "akt":
        y, reg = model(torch.cat((c[:, :1], cs), 1), torch.cat((r[:, :1], rs), 1), torch.cat((q[:, :1], qs), 1))
        return y[:, 1:], reg
    if name == "simplekt":
        y = model(dcur, train=True)[0] if train else model(dcur)
        return y[:, 1:], 0.0
    raise KeyError(name)


def _to(dcur: dict[str, Any], device: str) -> dict[str, Any]:
    """One batch on the model's device (a no-op when it is already there)."""
    return {key: value.to(device) for key, value in dcur.items()}


def _predict(model: Any, name: str, loader: Any, torch: Any, device: str = "cpu") -> tuple[Any, Any]:
    """Held-out predictions and targets, in the order of `scored_rows`."""
    import numpy as np

    model.eval()
    ys, ts = [], []
    with torch.no_grad():
        for dcur in loader:
            dcur = _to(dcur, device)
            y, _ = _forward(model, name, dcur, torch)
            sm = dcur["smasks"]
            ys.append(torch.masked_select(y, sm).float().cpu().numpy())
            ts.append(torch.masked_select(dcur["shft_rseqs"], sm).cpu().numpy())
    return np.concatenate(ys), np.concatenate(ts)


def _train_minimal(
    model: Any,
    name: str,
    train_loader: Any,
    valid_loader: Any,
    epochs: int,
    opt: Any,
    ckpt: str,
    torch: Any,
    device: str,
) -> tuple[float, int, int]:
    """Kwan's minimal copy of pyKT's `train_model` loop, in float32, for devices without float64 (Apple MPS).

    Same loss (binary cross-entropy on the selected answers, plus AKT's Rasch regulariser), same model
    selection (best validation AUC, improvement above 0.001) and same early stopping (10 epochs).
    """
    best_auc, best_epoch, epoch = 0.0, -1, 0
    bce = torch.nn.functional.binary_cross_entropy
    for epoch in range(1, epochs + 1):
        model.train()
        for dcur in train_loader:
            dcur = _to(dcur, device)
            y, extra = _forward(model, name, dcur, torch, train=True)
            sm = dcur["smasks"]
            loss = bce(torch.masked_select(y, sm).float(), torch.masked_select(dcur["shft_rseqs"], sm).float()) + extra
            opt.zero_grad()
            loss.backward()
            opt.step()
        p, t = _predict(model, name, valid_loader, torch, device)
        auc = score(t, p)["AUC"]
        if auc > best_auc + 1e-3:
            torch.save(model.state_dict(), ckpt)
            best_auc, best_epoch = auc, epoch
        print(f"    epoch {epoch}: valid AUC {auc:.4f} (best {best_auc:.4f} at epoch {best_epoch})", flush=True)
        if epoch - best_epoch >= 10:
            break
    return best_auc, best_epoch, epoch


def trainer_for(device: str) -> str:
    """pyKT's own `train_model` wherever it can run; the float32 loop only where float64 does not exist."""
    return (
        "kwan minimal loop over pyKT model classes (float32)"
        if device == "mps"
        else "pykt.models.train_model.train_model"
    )


def fit_and_predict(
    name: str,
    seqs: dict[str, Any],
    train_idx: Any,
    valid_idx: Any,
    test_idx: Any,
    epochs: int,
    device: str = "cpu",
    ckpt_dir: str | os.PathLike[str] = DEFAULT_WORK,
    seed: int = SEED,
    keep_checkpoint: bool = False,
    dataset_name: str = "kwan",
) -> tuple[Any, Any, dict[str, Any]]:
    """Train one pyKT model on the training sequences and predict the held-out ones.

    Returns (predictions, targets, info) with predictions in the order of `scored_rows(seqs, test_idx)`.
    The best-epoch weights are written under `ckpt_dir` during training and removed afterwards.
    """
    torch = _pykt(device)
    from pykt.models.init_model import init_model
    from pykt.utils import set_seed
    from torch.utils.data import DataLoader

    spec = PYKT_DEFAULTS[name]
    model_config = dict(spec["model_config"])
    if spec.get("needs_seq_len"):
        model_config["seq_len"] = seqs["maxlen"]
    set_seed(seed)
    data_config = {"num_q": seqs["num_q"], "num_c": seqs["num_c"], "emb_path": "", "dataset_name": dataset_name}
    model = init_model(name, model_config, data_config, "qid")
    opt = torch.optim.Adam(model.parameters(), spec["learning_rate"])
    batch = spec["batch_size"]
    data_device = "cpu" if device == "mps" else device  # the float32 loop moves each batch itself
    train_loader = DataLoader(_dataset(seqs, train_idx, torch, data_device), batch_size=batch)  # no shuffle, as pyKT
    valid_loader = DataLoader(_dataset(seqs, valid_idx, torch, data_device), batch_size=batch)
    test_loader = DataLoader(_dataset(seqs, test_idx, torch, data_device), batch_size=batch, shuffle=False)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = os.path.join(ckpt_dir, "qid_model.ckpt")  # the name pyKT's train_model writes
    if os.path.exists(ckpt):
        os.remove(ckpt)

    t0 = time.perf_counter()
    if device == "mps":
        valid_auc, best_epoch, epochs_run = _train_minimal(
            model, name, train_loader, valid_loader, epochs, opt, ckpt, torch, device
        )
    else:
        from pykt.models.train_model import train_model

        *_, valid_auc, _valid_acc, best_epoch = train_model(
            model, train_loader, valid_loader, epochs, opt, str(ckpt_dir), None, None, True
        )
        epochs_run = min(epochs, best_epoch + 10)
    train_seconds = time.perf_counter() - t0
    model.load_state_dict(torch.load(ckpt, map_location=device))
    preds, targets = _predict(model, name, test_loader, torch, device)
    if not keep_checkpoint:
        os.remove(ckpt)
    info = {
        "valid_AUC": float(valid_auc),
        "best_epoch": int(best_epoch),
        "epochs_run": int(epochs_run),
        "train_seconds": float(train_seconds),
        "parameters": int(sum(p.numel() for p in model.parameters())),
    }
    return preds, targets, info


def measure_latency(
    name: str,
    num_q: int,
    num_c: int,
    history: int = MAXLEN,
    reps: int = 200,
    warmup: int = 20,
    threads: int = 1,
    seed: int = SEED,
) -> dict[str, Any]:
    """CPU time of one prediction step of a pyKT model for a single learner.

    One step is one forward pass over a full window: `history - 1` past answers plus the item being
    predicted, batch size 1, evaluation mode, no gradients, `threads` CPU threads. That is what serving
    costs after every new answer when nothing is cached. Timing does not depend on the weights, so a
    freshly initialised model of the right size is used. Times are process CPU time, which other work on the machine
    barely moves, unlike wall time. Returns the median, 95th percentile and fastest pass in ms, and
    the wall-time median for reference.
    """
    import numpy as np

    torch = _pykt("cpu")
    from pykt.models.init_model import init_model

    spec = PYKT_DEFAULTS[name]
    model_config = dict(spec["model_config"])
    if spec.get("needs_seq_len"):
        model_config["seq_len"] = MAXLEN
    torch.manual_seed(seed)
    model = init_model(name, model_config, {"num_q": num_q, "num_c": num_c, "emb_path": ""}, "qid")
    model.eval()
    rng = np.random.default_rng(seed)
    seqs = {
        "questions": rng.integers(0, max(1, num_q), size=(1, history)),
        "concepts": rng.integers(0, max(1, num_c), size=(1, history)),
        "responses": rng.integers(0, 2, size=(1, history)),
        "selectmasks": np.ones((1, history), dtype=np.int64),
        "pad": PAD,
    }
    dcur = next(iter(torch.utils.data.DataLoader(_dataset(seqs, np.array([0]), torch, "cpu"), batch_size=1)))
    before = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        times, wall = [], []
        with torch.no_grad():
            for i in range(warmup + reps):
                c0, t0 = time.process_time(), time.perf_counter()
                _forward(model, name, dcur, torch)
                if i >= warmup:
                    times.append(1000.0 * (time.process_time() - c0))
                    wall.append(1000.0 * (time.perf_counter() - t0))
    finally:
        torch.set_num_threads(before)
    return {
        "ms_median": float(np.median(times)),
        "ms_p95": float(np.percentile(times, 95)),
        "ms_min": float(np.min(times)),
        "ms_wall_median": float(np.median(wall)),
        "history": history,
        "reps": reps,
        "threads": threads,
    }


def measure_elo_latency(df: Any, test_mask: Any, max_rows: int = 50_000, repeats: int = 5) -> dict[str, Any]:
    """CPU time of one elo-multi step (one prediction and one update), in ms.

    Item difficulty is fitted on up to `max_rows` training answers so the timed pass meets known and
    unknown items, then up to `max_rows` held-out answers are replayed `repeats` times.
    """
    from kwan.eval.kt import Elo2

    train, test = df[~test_mask].head(max_rows), df[test_mask].head(max_rows)
    model = Elo2()
    model.run(train, True)
    times, wall = [], []
    for _ in range(repeats):
        c0, t0 = time.process_time(), time.perf_counter()
        model.run(test, False)
        times.append(1000.0 * (time.process_time() - c0) / max(1, len(test)))
        wall.append(1000.0 * (time.perf_counter() - t0) / max(1, len(test)))
    return {
        "ms_median": statistics.median(times),
        "ms_p95": max(times),
        "ms_min": min(times),
        "ms_wall_median": statistics.median(wall),
        "steps": int(len(test)),
        "repeats": repeats,
    }


# --------------------------------------------------------------------------------------------------
# Runner, cache of fold results, CLI
# --------------------------------------------------------------------------------------------------


def _record_path(work: Path, dataset: str, model: str, fold: int) -> Path:
    return work / "runs" / dataset / f"{'elo' if model == ELO else model}.fold{fold}.json"


def _load_record(path: Path, config: dict[str, Any] | None) -> dict[str, Any] | None:
    """A cached fold result, or None when it is missing or was made with a different configuration."""
    if not path.exists():
        return None
    rec = json.loads(path.read_text(encoding="utf-8"))
    return rec if config is None or rec.get("config") == config else None


def _save_record(path: Path, rec: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")  # several processes may share the work directory
    tmp.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def parse_epochs(spec: str, models: tuple[str, ...] = MODELS) -> dict[str, int]:
    """`"8"` gives every model 8 epochs; `"dkt=30,akt=4"` sets them one by one (all models must be named)."""
    if "=" not in spec:
        return {m: int(spec) for m in models}
    out = {k.strip(): int(v) for k, v in (part.split("=") for part in spec.split(","))}
    missing = [m for m in models if m not in out]
    if missing:
        raise ValueError(f"--epochs gives no value for: {', '.join(missing)}")
    return out


def prepare(name: str, raw: str, folds: int, seed: int, maxlen: int, max_students: int | None = None) -> dict[str, Any]:
    """Load one dataset, convert it and assign folds. Everything the fold runs and the block need."""
    import numpy as np

    df = load_dataset(name, raw)[["user_id", "problem_id", "skill_id", "correct"]].reset_index(drop=True)
    subsample = None
    if max_students is not None and df["user_id"].nunique() > max_students:
        keep = sorted({_plain(u) for u in df["user_id"].unique()})
        random.Random(seed).shuffle(keep)
        df = df[df["user_id"].isin(set(keep[:max_students]))].reset_index(drop=True)
        subsample = f"{max_students} students drawn with seed {seed}"
    seqs = to_pykt_sequences(df, maxlen=maxlen)
    fold_of = make_folds(df["user_id"].unique(), folds, seed)
    meta = dataset_meta(name, df, seqs)
    if subsample:
        meta["subsample"] = subsample
    return {
        "name": name,
        "df": df,
        "seqs": seqs,
        "fold_of": fold_of,
        "row_fold": df["user_id"].map(fold_of).to_numpy(),
        "seq_fold": np.array([fold_of[_plain(u)] for u in seqs["uid"]]),
        "meta": meta,
    }


def run_dataset(
    prep: dict[str, Any],
    models: tuple[str, ...],
    epochs: dict[str, int],
    folds: int = FOLDS,
    seed: int = SEED,
    device: str = "cpu",
    work: str | os.PathLike[str] = DEFAULT_WORK,
    resume: bool = True,
    keep_checkpoints: bool = False,
    only_folds: tuple[int, ...] | None = None,
) -> None:
    """Run every missing (model, fold) of one dataset and cache each result under `work/runs/`."""
    import numpy as np

    name, df, seqs = prep["name"], prep["df"], prep["seqs"]
    work = Path(work)
    y_all = df["correct"].to_numpy()
    base = {
        "dataset": name,
        "rows": int(len(df)),
        "folds": folds,
        "seed": seed,
        "maxlen": seqs["maxlen"],
        "min_seq_len": MIN_SEQ_LEN,
    }
    fold_list = tuple(only_folds) if only_folds is not None else tuple(range(folds))
    uid = np.array([_plain(u) for u in seqs["uid"]], dtype=object)

    def split(k: int) -> tuple[Any, Any, Any, Any, Any]:
        """Sequence indices (train, validation, held out) of fold k, the scored rows and their answers."""
        test_idx = np.where(prep["seq_fold"] == k)[0]
        valid = validation_users([u for u, f in prep["fold_of"].items() if f != k], VAL_FRACTION, seed + k)
        is_valid = np.array([u in valid for u in uid], dtype=bool)
        train_idx = np.where((prep["seq_fold"] != k) & ~is_valid)[0]
        valid_idx = np.where((prep["seq_fold"] != k) & is_valid)[0]
        rows = scored_rows(seqs, test_idx)
        return train_idx, valid_idx, test_idx, rows, y_all[rows]

    for k in fold_list:  # elo-multi first: it is quick and every comparison needs it
        elo_config = {**base, "model": ELO, "fold": k}
        path = _record_path(work, name, ELO, k)
        if resume and _load_record(path, elo_config):
            continue
        *_, rows, y = split(k)
        test_mask = prep["row_fold"] == k
        preds, ms = elo_multi_fold(df, test_mask)
        full = np.full(len(df), np.nan)
        full[np.where(test_mask)[0]] = preds
        rec = {
            "kind": "elo",
            "fold": k,
            "config": elo_config,
            **score(y, full[rows]),
            "n_scored": int(len(rows)),
            "ms_per_update_cpu": ms,
        }
        everything = score(y_all[test_mask], preds)
        rec.update({"AUC_all": everything["AUC"], "RMSE_all": everything["RMSE"], "n_all": int(test_mask.sum())})
        _save_record(path, rec)
        print(
            f"[{name}] fold {k} {ELO}: AUC {rec['AUC']:.4f} on {len(rows)} scored answers "
            f"({rec['AUC_all']:.4f} on all {rec['n_all']})",
            flush=True,
        )

    for model in models:  # one model at a time over all folds, so a run cut short leaves whole models
        for k in fold_list:
            spec = PYKT_DEFAULTS[model]
            config = {
                **base,
                "model": model,
                "fold": k,
                "epochs": int(epochs[model]),
                "device": device,
                "trainer": trainer_for(device),
                "learning_rate": spec["learning_rate"],
                "batch_size": spec["batch_size"],
                "model_config": spec["model_config"],
                "validation_fraction": VAL_FRACTION,
            }
            path = _record_path(work, name, model, k)
            if resume and _load_record(path, config):
                print(f"[{name}] fold {k} {model}: cached", flush=True)
                continue
            train_idx, valid_idx, test_idx, rows, y = split(k)
            print(
                f"[{name}] fold {k} {model}: training on {len(train_idx)} sequences, "
                f"up to {epochs[model]} epochs, {device}",
                flush=True,
            )
            preds, targets, info = fit_and_predict(
                model, seqs, train_idx, valid_idx, test_idx, int(epochs[model]), device,
                work / "ckpt" / name / model / f"fold{k}", seed, keep_checkpoints, name,
            )  # fmt: skip
            if len(preds) != len(rows) or not np.array_equal(targets.astype(int), y.astype(int)):
                raise RuntimeError(f"{name} {model} fold {k}: predictions do not line up with the scored answers")
            rec = {"kind": "deep", "fold": k, "config": config, **score(y, preds), "n_scored": int(len(rows)), **info}
            _save_record(path, rec)
            print(
                f"[{name}] fold {k} {model}: AUC {rec['AUC']:.4f}, best epoch {info['best_epoch']}, "
                f"{info['train_seconds'] / 60:.1f} min",
                flush=True,
            )


def collect(
    name: str, models: tuple[str, ...], folds: int, work: str | os.PathLike[str] = DEFAULT_WORK
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    """Cached fold results of one dataset. A model with a missing fold, or folds made with different
    configurations, is left out and named in the second return value; it is never filled in."""
    out: dict[str, list[dict[str, Any]]] = {}
    skipped: dict[str, str] = {}
    for model in (ELO, *models):
        loaded = [_load_record(_record_path(Path(work), name, model, k), None) for k in range(folds)]
        recs = [rec for rec in loaded if rec is not None]
        missing = [k for k, rec in enumerate(loaded) if rec is None]
        if missing:
            skipped[f"{name}/{model}"] = f"no result for fold(s) {missing}; not run or not finished in the time budget"
            continue
        configs = [{key: value for key, value in rec["config"].items() if key != "fold"} for rec in recs]
        if any(config != configs[0] for config in configs):
            skipped[f"{name}/{model}"] = "folds were run with different configurations"
            continue
        out[model] = recs
    return out, skipped


def protocol_block(folds: int, seed: int, maxlen: int) -> dict[str, Any]:
    return {
        "folds": folds,
        "split": f"{folds} folds by student, seed {seed}; each fold is held out once and no student is in two folds",
        "validation": (
            f"{int(VAL_FRACTION * 100)}% of each fold's training students (seed {seed} + fold) are used only "
            "to pick the best epoch of the deep models; the held-out fold is never used for model selection"
        ),
        "max_seq_len": maxlen,
        "min_seq_len": MIN_SEQ_LEN,
        "sequence_format": (
            f"pyKT: each student's answers are cut into pieces of at most {maxlen}, padded with -1; "
            f"a last piece shorter than {MIN_SEQ_LEN} answers is dropped; a deep model sees one piece at a time"
        ),
        "scored_answers": (
            "every answer of a held-out sequence except its first (pyKT's next-answer scoring); "
            "elo-multi is scored on exactly these answers, after reading each held-out student's full history in order"
        ),
        "elo_multi": (
            "kwan.eval.kt.Elo2: item difficulty fitted on the training students (two passes), "
            "held-out students predicted online"
        ),
        "metrics": "AUC, RMSE, accuracy at 0.5; mean and sample standard deviation (n - 1) across folds",
        "seed": seed,
    }


LATENCY_PROTOCOL = {
    "clock": (
        "ms_per_update_cpu is process CPU time, so that other work on a shared machine does not inflate it; "
        "ms_per_update_wall is the wall-time median of the same passes"
    ),
    "deep_models": (
        f"one forward pass for one learner over a full window ({MAXLEN - 1} past answers plus the item to predict), "
        "batch size 1, CPU, 1 thread, no gradients; median of {reps} timed passes after 20 warm-up passes"
    ),
    "elo_multi": (
        "one prediction and one update per answer, pure Python, 1 thread: CPU time of a pass over up to 50,000 "
        "held-out answers of fold 0 divided by their number; median of 5 passes"
    ),
    "note": (
        "The deep figure is the most favourable serving case: one next-answer prediction. Kwan needs mastery for "
        "every candidate skill after each answer, which costs the attention models one pass per candidate."
    ),
}


def build_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    p = parser or argparse.ArgumentParser(
        prog="kwan eval kt-deep",
        description="Deep knowledge tracing benchmark (pyKT) against elo-multi on the same folds.",
    )
    p.add_argument("--models", default=",".join(MODELS), help="comma-separated: dkt,sakt,simplekt,akt")
    p.add_argument("--datasets", default=",".join(D7_DATASETS), help=f"comma-separated, from: {', '.join(DATASETS)}")
    p.add_argument(
        "--epochs",
        default="8",
        help='most epochs per fit: one number, or per model, e.g. "dkt=30,sakt=20,simplekt=8,akt=4"',
    )
    p.add_argument("--folds", type=int, default=FOLDS, help="folds by student (default 5)")
    p.add_argument(
        "--device", choices=("cpu", "mps", "cuda"), default="cpu", help="training device; timing always runs on CPU"
    )
    p.add_argument(
        "--out", metavar="PATH", help=f"merge the '{KEY}' block into this results file, e.g. data/eval_results.json"
    )
    p.add_argument("--raw", default="raw", help="directory holding the downloaded datasets (default raw)")
    p.add_argument(
        "--work",
        default=DEFAULT_WORK,
        help=f"scratch directory for fold results and checkpoints (default {DEFAULT_WORK})",
    )
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--maxlen", type=int, default=MAXLEN, help="longest sequence, as in pyKT (default 200)")
    p.add_argument("--threads", type=int, default=4, help="torch CPU threads while training (default 4, as pyKT)")
    p.add_argument(
        "--max-students",
        type=int,
        metavar="N",
        help="train and score on a seeded sample of N students (recorded in the block)",
    )
    p.add_argument(
        "--only-folds", metavar="LIST", help="run only these folds, e.g. 0,1 (for splitting work across processes)"
    )
    p.add_argument(
        "--latency-reps", type=int, default=200, metavar="N", help="timed passes per deep model (default 200)"
    )
    p.add_argument("--no-resume", action="store_true", help="recompute fold results even when a cached one matches")
    p.add_argument(
        "--assemble", action="store_true", help="train nothing: build the block from the cached fold results"
    )
    p.add_argument(
        "--keep-checkpoints", action="store_true", help="keep the best-epoch weights under the work directory"
    )
    p.add_argument(
        "--skip-note",
        action="append",
        default=[],
        metavar="NAME=WHY",
        help="record that something was skipped and why (repeatable)",
    )
    return p


def run(args: argparse.Namespace) -> dict[str, Any]:
    models = tuple(m.strip() for m in args.models.split(",") if m.strip())
    names = tuple(d.strip() for d in args.datasets.split(",") if d.strip())
    for m in models:
        if m not in MODELS:
            raise SystemExit(f"unknown model {m!r}; choose from {', '.join(MODELS)}")
    for d in names:
        if d not in DATASETS:
            raise SystemExit(f"unknown dataset {d!r}; choose from {', '.join(DATASETS)}")
    epochs = parse_epochs(args.epochs, models)
    only = tuple(int(k) for k in args.only_folds.split(",")) if args.only_folds else None
    skipped = dict(note.split("=", 1) for note in args.skip_note)
    block_datasets: dict[str, Any] = {}
    if not args.assemble:
        import torch

        torch.set_num_threads(args.threads)
    for name in names:
        try:
            prep = prepare(name, args.raw, args.folds, args.seed, args.maxlen, args.max_students)
        except FileNotFoundError as exc:
            skipped[name] = f"data not available: {exc}"
            print(f"[{name}] skipped: {exc}", flush=True)
            continue
        meta = prep["meta"]
        counts = " ".join(f"{key}={meta[key]}" for key in ("rows", "students", "items", "skills", "sequences"))
        print(f"[{name}] {counts}", flush=True)
        if not args.assemble:
            run_dataset(
                prep,
                models,
                epochs,
                args.folds,
                args.seed,
                args.device,
                args.work,
                not args.no_resume,
                args.keep_checkpoints,
                only,
            )
        fold_records, missing = collect(name, models, args.folds, args.work)
        for what, why in missing.items():
            skipped.setdefault(what, why)  # a reason given with --skip-note wins over the generic one
        latency = {}
        if args.out:
            if ELO in fold_records:
                latency[ELO] = measure_elo_latency(prep["df"], prep["row_fold"] == 0)
                print(f"[{name}] {ELO}: {latency[ELO]['ms_median']:.4f} ms per step on CPU", flush=True)
            for m in models:
                if m in fold_records:
                    latency[m] = measure_latency(
                        m, prep["seqs"]["num_q"], prep["seqs"]["num_c"], reps=args.latency_reps
                    )
                    print(f"[{name}] {m}: {latency[m]['ms_median']:.2f} ms per step on CPU", flush=True)
        block_datasets[name] = {"meta": prep["meta"], "folds": fold_records, "latency": latency}
    versions = library_versions() if args.out else {}
    implementation = (
        "Real pyKT model classes (DKT, SAKT, simpleKT, AKT), built by pykt.models.init_model with pyKT default "
        "hyperparameters and fed by pyKT's KTDataset; not re-implementations. On CPU or CUDA they are trained by "
        "pyKT's own train_model. On Apple MPS, which has no float64, they are trained by Kwan's float32 copy of "
        "that loop (same loss, model selection and early stopping). The trainer used is recorded per model under "
        "datasets.<name>.models.<model>.trainer."
    )
    latency_protocol: dict[str, Any] = dict(LATENCY_PROTOCOL)
    latency_protocol["deep_models"] = latency_protocol["deep_models"].format(reps=args.latency_reps)
    if args.out:
        latency_protocol["machine_load_average_1min_when_timed"] = round(os.getloadavg()[0], 1)
    block = make_block(
        block_datasets,
        protocol_block(args.folds, args.seed, args.maxlen),
        versions,
        latency_protocol,
        skipped,
        implementation,
    )
    for name, entry in block["datasets"].items():
        for model, s in entry["models"].items():
            print(
                f"{name:18s} {model:18s} AUC {s['AUC_mean']:.4f} (sd {s['AUC_sd']:.4f})  RMSE {s['RMSE_mean']:.4f}  "
                f"ACC {s['ACC_mean']:.4f}  {s['ms_per_update_cpu']} ms",
                flush=True,
            )
    print(
        f"D7: adopt={block['decision']['adopt']} model={block['decision']['model']}. {block['decision']['reason']}",
        flush=True,
    )
    if args.out:
        from kwan.eval import results

        results.update(KEY, block, path=args.out)
        print(f"wrote {KEY} to {args.out}", file=sys.stderr)
    return block


def _handler(args: argparse.Namespace) -> int:
    run(args)
    return 0


def register(subparsers: Any) -> None:
    """Add `kt-deep` under `kwan eval`. Call with the sub-parsers of the `eval` command."""
    p = subparsers.add_parser("kt-deep", help="deep knowledge tracing (pyKT) against elo-multi on 5 folds by student")
    build_parser(p)
    p.set_defaults(func=_handler, extra="kt")


def main(argv: list[str] | None = None) -> dict[str, Any]:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
