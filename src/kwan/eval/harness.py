"""Evaluation harness v2 (PRD Phase 4.4 to 4.6): one protocol for every learner dataset.

Protocol
    split    80/20 by student, seed 42 (`split`, the v1 split), no student on both sides;
             `kfold_by_student` gives 5 folds by student for comparisons with pyKT
    models   global-mean, skill-mean, elo, elo-multi (the Kwan learner model) and, where asked, bkt
    metrics  AUC, RMSE, accuracy and ECE (10 bins) on the next-answer predictions of the test students
    counts   every block reports rows, students, items and skills

`run_dataset("eedi_2020_task12", out="data/eval_results.json")` loads a dataset through
`kwan.data.loaders`, runs the protocol and merges one block into the results file through
`kwan.eval.results.update`. `run_all` does it for every dataset whose raw files are present.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kwan.data import loaders, registry
from kwan.eval import results
from kwan.eval.metrics import METRICS

SEED = 42
SPLIT = "80/20 by student, seed 42"
FOLDS = 5


@dataclass(frozen=True)
class Spec:
    """How one results-file block is produced."""

    key: str  # top-level key in data/eval_results.json
    cli: str  # name after `kwan eval`
    title: str  # the block's `dataset` text
    files: tuple[str, ...]  # files the loader needs, relative to the dataset path ("" = the path itself)
    subsets: tuple[str, ...] = ()  # sub-blocks (AMMORE: all, ghana); empty = one flat block
    legacy: tuple[str, ...] = ()  # v1 fields kept beside `metrics` (see results.dataset_block)
    bkt: bool = False  # run the BKT grid baseline in `kwan eval all`
    note: str = ""  # anything a reader of the block must know


SPECS: tuple[Spec, ...] = (
    Spec(
        "assistments_2009",
        "assistments-2009",
        "ASSISTments 2009 skill-builder (corrected)",
        ("",),
        legacy=("AUC", "RMSE"),
        bkt=True,
    ),
    Spec(
        "eedi_2020_task34",
        "eedi-34",
        "Eedi NeurIPS 2020 Education Challenge, task 3/4 subset",
        (
            "train_data/train_task_3_4.csv",
            "metadata/answer_metadata_task_3_4.csv",
            "metadata/question_metadata_task_3_4.csv",
            "metadata/subject_metadata.csv",
        ),
        legacy=("questions", "AUC", "RMSE"),
    ),
    Spec(
        "eedi_2020_task12",
        "eedi-12",
        "Eedi NeurIPS 2020 Education Challenge, task 1/2 (full training set)",
        (
            "train_data/train_task_1_2.csv",
            "metadata/answer_metadata_task_1_2.csv",
            "metadata/question_metadata_task_1_2.csv",
            "metadata/subject_metadata.csv",
        ),
    ),
    Spec(
        "ammore_2024",
        "ammore",
        "AMMORE (Rori WhatsApp maths tutor, Rising Academies), Jan to Apr 2024",
        ("AMMORE_student_answers.csv", "AMMORE_demographic_data.csv"),
        subsets=("all", "ghana"),
        legacy=("questions", "AUC", "RMSE"),
    ),
    Spec(
        "assistments_2012",
        "assistments-2012",
        "ASSISTments 2012-2013 (rows with a skill)",
        ("2012-2013-data-with-predictions-4-final.csv",),
    ),
    Spec(
        "assistments_2015",
        "assistments-2015",
        "ASSISTments 2015 (100 skill builders, main problems)",
        ("2015_100_skill_builders_main_problems.csv",),
        note="This release has no question ids: the skill builder is both the skill and the item.",
    ),
    Spec(
        "assistments_2017",
        "assistments-2017",
        "ASSISTments 2017 (Longitudinal Data Mining Competition)",
        ("anonymized_full_release_competition_dataset.csv",),
    ),
    Spec(
        "xes3g5m",
        "xes3g5m",
        "XES3G5M (Grade 3 maths, China), question level",
        (),
    ),
    Spec(
        "junyi_2015",
        "junyi",
        "Junyi Academy Math Practicing Log (2015), problem log",
        ("junyi_ProblemLog_original.csv",),
        note="Skill = exercise; item = exercise and problem template. Evaluation only (non-commercial data).",
    ),
)
BY_KEY = {s.key: s for s in SPECS}
BY_CLI = {s.cli: s for s in SPECS}


def spec(name: str) -> Spec:
    """The spec for a results-file key (`eedi_2020_task12`) or a command name (`eedi-12`)."""
    found = BY_KEY.get(name) or BY_CLI.get(name)
    if found is None:
        raise KeyError(f"unknown dataset '{name}' (known: {', '.join(BY_CLI)})")
    return found


# ------------------------------------------------------------------------------------------ splits


def split(df: Any, seed: int = SEED) -> tuple[Any, Any]:
    """80/20 split by student (the v1 split, unchanged): returns (train, test)."""
    from kwan.eval import kt

    return kt.split(df, seed)


def fold_students(df: Any, k: int = FOLDS, seed: int = SEED) -> list[Any]:
    """Shuffle the students with `seed` and cut them into `k` folds of near-equal size."""
    import numpy as np

    users = np.array(df.user_id.unique())
    if k < 2 or k > len(users):
        raise ValueError(f"k must be between 2 and the number of students ({len(users)}), got {k}")
    rng = np.random.default_rng(seed)
    rng.shuffle(users)
    return np.array_split(users, k)


def kfold_by_student(df: Any, k: int = FOLDS, seed: int = SEED) -> Iterator[tuple[Any, Any]]:
    """Yield (train, test) `k` times. Every student is in exactly one test fold; rows keep their order."""
    for fold in fold_students(df, k, seed):
        in_test = df.user_id.isin(set(fold.tolist()))
        yield df[~in_test], df[in_test]


# ------------------------------------------------------------------------------------------ scoring


def header(df: Any, tr: Any, te: Any) -> str:
    """The console line printed before the model lines."""
    from kwan.data.schema import counts

    c = counts(df)
    return (
        f"rows={c['rows']} students={c['students']} items={c.get('items', 0)} skills={c['skills']} "
        f"train={len(tr)} test={len(te)}"
    )


def evaluate(df: Any, bkt: bool = False, seed: int = SEED) -> dict[str, dict[str, float]]:
    """Split by student, fit every model on the train students, score the test students.

    Prints the header and one line per model. Returns {"AUC": {model: value}, "RMSE": ..., "ACC": ...,
    "ECE": ...}, unrounded.
    """
    from kwan.eval import kt

    tr, te = split(df, seed)
    print(header(df, tr, te))
    return kt.score_models(tr, te, bkt)


def cross_validate(df: Any, k: int = FOLDS, seed: int = SEED, bkt: bool = False) -> dict[str, Any]:
    """`k`-fold cross-validation by student. Returns the `cv` part of a block: mean and standard
    deviation (population, over folds) of every metric per model, rounded to 4 decimals."""
    import numpy as np

    from kwan.eval import kt

    per_fold = []
    for i, (tr, te) in enumerate(kfold_by_student(df, k, seed)):
        print(f"[fold {i + 1}/{k}] train={len(tr)} test={len(te)}")
        per_fold.append(kt.score_models(tr, te, bkt))
    models = [m for m in kt.MODEL_ORDER if m in per_fold[0]["AUC"]]
    summary = {
        m: {
            metric: {
                "mean": round(float(np.mean([f[metric][m] for f in per_fold])), 4),
                "std": round(float(np.std([f[metric][m] for f in per_fold])), 4),
            }
            for metric in METRICS
        }
        for m in models
    }
    return {"split": f"{k}-fold by student, seed {seed}", "folds": k, "metrics": summary}


def make_block(spec_: Spec, df: Any, scores: dict[str, dict[str, float]]) -> dict[str, Any]:
    """One flat schema v2 block (`results.dataset_block`) for a dataset, with its licence from the registry."""
    from kwan.data.schema import counts
    from kwan.eval import kt

    block = results.dataset_block(
        spec_.title,
        registry.licence(spec_.key),
        counts(df),
        SPLIT,
        scores,
        model_order=kt.MODEL_ORDER,
        legacy=spec_.legacy,
    )
    if spec_.note:
        block["note"] = spec_.note
    return block


# ------------------------------------------------------------------------------------------ running


def default_path(name: str, raw_dir: str | Path = "raw") -> Path:
    """Where `make data` puts a dataset under the raw-data directory."""
    return loaders.raw_path(spec(name).key, raw_dir)


def missing_files(name: str, path: str | Path) -> list[Path]:
    """The files of a dataset that are not on disk (empty list = ready to evaluate)."""
    s, path = spec(name), Path(path)
    if s.key == "xes3g5m":
        from kwan.data.loaders import xes3g5m

        try:
            base = xes3g5m.find_base(path)
        except FileNotFoundError:
            return [path / "XES3G5M" / f for f in xes3g5m.FILES]
        return [base / f for f in xes3g5m.FILES if not (base / f).is_file()]
    if path.is_file():  # the CSV itself was given
        return []
    return [path / f if f else path for f in s.files if not (path / f if f else path).is_file()]


def run_dataset(
    name: str,
    path: str | Path | None = None,
    out: str | Path | None = None,
    bkt: bool = False,
    folds: int = 0,
    raw_dir: str | Path = "raw",
) -> dict[str, Any]:
    """Evaluate one dataset and return its block; with `out`, merge the block into that results file.

    `folds` > 1 adds a `cv` entry with k-fold results by student. Raises FileNotFoundError when the
    raw files are missing.
    """
    from kwan.eval import kt

    s = spec(name)
    path = Path(path) if path is not None else default_path(s.key, raw_dir)
    gone = missing_files(s.key, path)
    if gone:
        raise FileNotFoundError(f"{s.cli}: missing {', '.join(str(p) for p in gone)} (run `make data`)")
    started = time.time()
    if s.subsets:
        block: dict[str, Any] = {"dataset": s.title, "licence": registry.licence(s.key)}
        if s.note:
            block["note"] = s.note
        for subset in s.subsets:
            df = loaders.load(f"{s.key}:{subset}", path)
            tr, te = split(df)
            print(f"[{subset}] {header(df, tr, te)}")
            sub = make_block(s, df, kt.score_models(tr, te, bkt))
            del sub["dataset"], sub["licence"]
            sub.pop("note", None)
            if folds > 1:
                sub["cv"] = cross_validate(df, folds, bkt=bkt)
            block[subset] = sub
    else:
        df = loaders.load(s.key, path)
        block = make_block(s, df, evaluate(df, bkt))
        if folds > 1:
            block["cv"] = cross_validate(df, folds, bkt=bkt)
    print(f"{s.cli}: done in {time.time() - started:.0f} s", file=sys.stderr)
    if out:
        results.update(s.key, block, path=out)
        print(f"wrote {s.key} to {out}", file=sys.stderr)
    return block


def skip(name: str, reason: str, out: str | Path | None = None) -> dict[str, Any]:
    """Report a dataset that cannot be evaluated. With `out`, a `skipped` block is written, but only
    when the file has no numbers for that dataset yet (a missing download never erases old results)."""
    s = spec(name)
    block = results.skipped_block(s.title, registry.licence(s.key), reason)
    print(f"skipping {s.cli}: {reason}", file=sys.stderr)
    if out:
        have = results.load(out).get(s.key)
        if not have or "skipped" in have:
            results.update(s.key, block, path=out)
    return block


def run_all(
    raw_dir: str | Path = "raw",
    out: str | Path | None = None,
    only: tuple[str, ...] | list[str] | None = None,
    bkt: bool | None = None,
    folds: int = 0,
) -> dict[str, dict[str, Any]]:
    """Evaluate every dataset (or those named in `only`) whose raw files are under `raw_dir`.

    Datasets with missing files are skipped with a clear message, never an error. `bkt=None` runs
    the BKT grid only where the spec says so (ASSISTments 2009). Returns {results key: block}.
    """
    chosen = [spec(n) for n in only] if only else list(SPECS)
    blocks = {}
    for s in chosen:
        path = default_path(s.key, raw_dir)
        gone = missing_files(s.key, path)
        if gone:
            blocks[s.key] = skip(s.key, f"raw data is missing ({gone[0]}); run `make data`", out)
            continue
        print(f"== {s.cli}: {s.title}")
        blocks[s.key] = run_dataset(s.key, path, out, bkt=s.bkt if bkt is None else bkt, folds=folds)
    return blocks
