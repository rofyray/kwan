"""Eedi NeurIPS 2020 Education Challenge: task 3/4 (1.4M answers) and task 1/2 (15.9M answers).

Skill = the most specific subject (deepest Level) attached to the question, for both subsets.
Licence: research use, cite Wang et al. (2020). Raw rows are never committed.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from kwan.data import schema
from kwan.data.loaders._common import dataset_attrs, need, read_csv_chunks

KEY = "eedi_2020"
TASK12_FILES = (
    "train_data/train_task_1_2.csv",
    "metadata/answer_metadata_task_1_2.csv",
    "metadata/question_metadata_task_1_2.csv",
    "metadata/subject_metadata.csv",
)
TASK34_FILES = (
    "train_data/train_task_3_4.csv",
    "metadata/answer_metadata_task_3_4.csv",
    "metadata/question_metadata_task_3_4.csv",
    "metadata/subject_metadata.csv",
)


def deepest_subject(root: str | Path, question_file: str) -> dict[int, int]:
    """{QuestionId: the subject with the deepest Level among the question's subjects}."""
    import pandas as pd

    root = Path(root)
    qm = pd.read_csv(need(root / "metadata" / question_file))
    sm = pd.read_csv(need(root / "metadata" / "subject_metadata.csv"), encoding="utf-8-sig")
    level = dict(zip(sm.SubjectId, sm.Level))
    skill = qm.SubjectId.map(lambda s: max(ast.literal_eval(s), key=lambda x: level.get(x, -1)))
    return dict(zip(qm.QuestionId.astype(int), skill.astype(int)))


def load_task34(root: str | Path) -> Any:
    """Task 3/4 in the interaction schema. Wraps `kwan.eval.eedi.load_eedi`: same rows, same order."""
    from kwan.eval.eedi import load_eedi

    df = load_eedi(str(root)).rename(columns={"DateAnswered": "t", "AnswerId": "meta_answer_id"})
    return schema.conform(df, dataset_attrs(KEY, "task34"))


def load_task12(root: str | Path, block_bytes: int = 32 << 20) -> Any:
    """Task 1/2 (the full public training set) in the interaction schema.

    The files are read in chunks of `block_bytes` with pyarrow, ids are held as 32-bit integers and
    the answer as 8 bits, so the 15.9M rows fit in well under 1 GB. Answers without a row in the
    answer metadata (no time) are dropped, as in task 3/4. Rows are ordered by student, time answered,
    then AnswerId.
    """
    import numpy as np
    import pyarrow as pa

    root = Path(root)
    for name in TASK12_FILES:
        need(root / name)
    answers = read_csv_chunks(
        root / "train_data" / "train_task_1_2.csv",
        {"QuestionId": pa.int32(), "UserId": pa.int32(), "AnswerId": pa.int32(), "IsCorrect": pa.int8()},
        block_bytes,
    ).to_pandas()
    # AnswerId is written as a decimal ("11808339.0") in this file
    meta = read_csv_chunks(
        root / "metadata" / "answer_metadata_task_1_2.csv",
        {"AnswerId": pa.float64(), "DateAnswered": pa.timestamp("ms")},
        block_bytes,
    ).to_pandas()
    meta = meta[meta["AnswerId"].notna() & meta["DateAnswered"].notna()]  # a few rows carry no id or no time
    meta["AnswerId"] = meta["AnswerId"].astype(np.int32)
    meta = meta.drop_duplicates("AnswerId")
    df = answers.merge(meta, on="AnswerId")
    del answers, meta
    skill = deepest_subject(root, "question_metadata_task_1_2.csv")
    df["skill_id"] = df["QuestionId"].map(skill)
    df = df[df["skill_id"].notna()]
    df["skill_id"] = df["skill_id"].astype(np.int32)
    df = df.rename(
        columns={
            "UserId": "user_id",
            "QuestionId": "item_id",
            "IsCorrect": "correct",
            "DateAnswered": "t",
            "AnswerId": "meta_answer_id",
        }
    )
    df = df.sort_values(["user_id", "t", "meta_answer_id"], kind="stable")
    return schema.conform(df, dataset_attrs(KEY, "task12"), sort=False)
