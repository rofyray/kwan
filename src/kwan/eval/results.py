"""Single writer for data/eval_results.json.

Every number quoted in the README, the demo or a deck must come from that file, and every block in it
must be written by a script in this repo. Each script owns one top-level key and calls `update()`,
which merges that key into the file and leaves the other blocks untouched.
"""

import contextlib
import json
import os
import tempfile
from pathlib import Path

from kwan._paths import data_dir

DATA_DIR = data_dir()
ROOT = DATA_DIR.parent
DEFAULT_PATH = DATA_DIR / "eval_results.json"


def load(path=DEFAULT_PATH) -> dict:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@contextlib.contextmanager
def _locked(directory: Path):
    """Hold an exclusive lock on the results directory, so two scripts that finish at the same moment
    do not lose each other's block. No lock file is created. Where `fcntl` is missing, there is no lock."""
    try:
        import fcntl
    except ImportError:  # not a POSIX system
        yield
        return
    fd = os.open(directory, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)  # closing the descriptor releases the lock


def update(key: str, block: dict, path=DEFAULT_PATH) -> dict:
    """Replace one top-level block and write the file atomically. Returns the full results dict."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _locked(path.parent):
        res = load(path)
        res[key] = block
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, path)
    return res


BLOCK_SCHEMA = 2
METRICS = ("AUC", "RMSE", "ACC", "ECE")


def metrics_by_model(scores: dict, model_order=None) -> dict:
    """Turn {"AUC": {model: value}, ...} into {model: {"AUC", "RMSE", "ACC", "ECE"}}, rounded to 4 decimals.

    Models follow `model_order` (models it does not list keep their order and go last). A metric that
    was not computed is left out.
    """
    models = list(next(iter(scores.values()), {}))
    if model_order is not None:
        models = [m for m in model_order if m in models] + [m for m in models if m not in model_order]
    return {m: {k: round(float(scores[k][m]), 4) for k in METRICS if k in scores and m in scores[k]} for m in models}


def dataset_block(
    dataset: str,
    licence: str,
    counts: dict,
    split: str,
    scores: dict,
    model_order=None,
    legacy: tuple = (),
) -> dict:
    """One dataset block of eval_results.json, schema v2 (PRD Phase 4.6).

        {"dataset": name, "licence": text, "rows": n, "students": n, "items": n, "skills": n,
         "split": text, "metrics": {model: {"AUC": x, "RMSE": x, "ACC": x, "ECE": x}}}

    `counts` is {"rows", "students", "items", "skills"} and `scores` is {"AUC": {model: value}, ...}
    as returned by `kwan.eval.kt.score_models`.

    `legacy` keeps v1 fields beside `metrics`, for now: "AUC" and "RMSE" ({model: value}, the v1 shape)
    and "questions" (the same number as `items`). The demo and several tests still read those fields
    of `assistments_2009` and `eedi_2020_task34`; they go away once every reader uses `metrics`.
    """
    by_model = metrics_by_model(scores, model_order)
    block = {"dataset": dataset, "licence": licence}
    for name in ("rows", "students", "items"):
        if name in counts:
            block[name] = int(counts[name])
    if "questions" in legacy and "items" in counts:
        block["questions"] = int(counts["items"])
    block["skills"] = int(counts["skills"])
    block["split"] = split
    block["metrics"] = by_model
    for name in ("AUC", "RMSE"):
        if name in legacy:
            block[name] = {m: v[name] for m, v in by_model.items() if name in v}
    return block


def skipped_block(dataset: str, licence: str, reason: str) -> dict:
    """The block written when a dataset could not be evaluated (for example a download that failed)."""
    return {"dataset": dataset, "licence": licence, "skipped": reason}
