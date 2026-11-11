"""ASSISTments (US, middle-school maths): the 2009, 2012-13, 2015 and 2017 releases.

Licence: research use. Raw rows are never committed. Filters follow the pyKT benchmark set-up so
that results can be compared with published numbers: rows without a skill are dropped and only
answers marked 0 or 1 are kept (some releases carry partial credit).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kwan.data import schema
from kwan.data.loaders._common import dataset_attrs, need

FILE_2012 = "2012-2013-data-with-predictions-4-final.csv"
FILE_2015 = "2015_100_skill_builders_main_problems.csv"
FILE_2017 = "anonymized_full_release_competition_dataset.csv"


def _file(path: str | Path, name: str) -> Path:
    """`path` itself when it is the CSV, else the CSV of that name inside the directory `path`."""
    path = Path(path)
    return need(path if path.is_file() else path / name)


def load_2009(path: str | Path) -> Any:
    """ASSISTments 2009 skill-builder (corrected). Wraps `kwan.eval.kt.load`: same rows, same order.

    Main problems only, one row per attempt. `t` is the order id (it grows with time).
    """
    from kwan.eval.kt import load

    df = load(need(Path(path)))
    df["t"] = df["order_id"]
    return schema.conform(df[["user_id", "problem_id", "skill_id", "t", "correct"]], dataset_attrs("assistments_2009"))


def load_2012(path: str | Path, chunk_rows: int = 500_000) -> Any:
    """ASSISTments 2012-13. The 3 GB file is read `chunk_rows` rows at a time, six columns only.

    Rows without a skill id are dropped (more than half of the log). Ordered by student, start time,
    then problem log id.
    """
    import pandas as pd

    cols = ["problem_log_id", "problem_id", "user_id", "start_time", "correct", "skill_id"]
    parts = []
    reader = pd.read_csv(_file(path, FILE_2012), usecols=cols, chunksize=chunk_rows, encoding="latin-1")
    for chunk in reader:
        chunk = chunk[chunk.skill_id.notna() & chunk.correct.isin([0, 1])]
        parts.append(
            pd.DataFrame(
                {
                    "user_id": chunk.user_id.astype("int32"),
                    "item_id": chunk.problem_id.astype("int32"),
                    "skill_id": chunk.skill_id.astype("int32"),
                    "t": pd.to_datetime(chunk.start_time, format="ISO8601"),  # some rows carry fractions of a second
                    "correct": chunk.correct.astype("int8"),
                    "meta_problem_log_id": chunk.problem_log_id.astype("int64"),
                }
            )
        )
    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values(["user_id", "t", "meta_problem_log_id"], kind="stable")
    return schema.conform(df, dataset_attrs("assistments_2012"), sort=False)


def load_2015(path: str | Path) -> Any:
    """ASSISTments 2015 (100 skill builders, main problems).

    This release has no question ids: each row names only its skill builder (`sequence_id`). The skill
    builder is therefore both the skill and the item. `t` is the log id (it grows with time).
    """
    import pandas as pd

    df = pd.read_csv(_file(path, FILE_2015), usecols=["user_id", "log_id", "sequence_id", "correct"])
    df = df[df.correct.isin([0, 1])]
    out = pd.DataFrame(
        {
            "user_id": df.user_id.astype("int32"),
            "item_id": df.sequence_id.astype("int32"),
            "skill_id": df.sequence_id.astype("int32"),
            "t": df.log_id.astype("int64"),
            "correct": df.correct.astype("int8"),
        }
    )
    return schema.conform(out, dataset_attrs("assistments_2015", items_are_skills=True))


def load_2017(path: str | Path) -> Any:
    """ASSISTments 2017 (Longitudinal Data Mining Competition), one row per logged action.

    The skill is a name in this release; it is given a number in alphabetical order and the names are
    kept in `attrs["skill_names"]`. Ordered by student, start time, then file order.
    """
    import numpy as np
    import pandas as pd

    cols = ["studentId", "skill", "problemId", "startTime", "correct"]
    df = pd.read_csv(_file(path, FILE_2017), usecols=cols, low_memory=False)
    df["row"] = np.arange(len(df))
    df = df.dropna(subset=cols)
    df = df[df.correct.isin([0, 1])]
    skill = df.skill.astype("category")
    out = pd.DataFrame(
        {
            "user_id": df.studentId.astype("int32"),
            "item_id": df.problemId.astype("int32"),
            "skill_id": skill.cat.codes.astype("int32"),
            "t": pd.to_datetime(df.startTime.astype("int64"), unit="s"),
            "correct": df.correct.astype("int8"),
            "meta_row": df.row.astype("int32"),
        }
    )
    out = out.sort_values(["user_id", "t", "meta_row"], kind="stable")
    out = schema.conform(out, dataset_attrs("assistments_2017"), sort=False)
    out.attrs["skill_names"] = dict(enumerate(skill.cat.categories))
    return out
