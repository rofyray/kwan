"""XES3G5M (China, Grade 3 maths): 5M+ answers from 18k pupils. Licence: MIT.

The official release stores each pupil's answers as comma-separated sequences. This loader turns the
question-level files into one row per answer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kwan.data import schema
from kwan.data.loaders._common import dataset_attrs

KEY = "xes3g5m"
FILES = ("question_level/train_valid_sequences_quelevel.csv", "question_level/test_quelevel.csv")
PAD = -1


def find_base(root: str | Path) -> Path:
    """The directory that holds `question_level/` (the root itself or a folder one level down)."""
    root = Path(root)
    for cand in (root, *(sorted(p for p in root.glob("*") if p.is_dir()) if root.is_dir() else ())):
        if (cand / "question_level").is_dir():
            return cand
    raise FileNotFoundError(f"no question_level/ directory under {root} (run `make data`)")


def load(root: str | Path) -> Any:
    """XES3G5M in the interaction schema.

    Reads `question_level/train_valid_sequences_quelevel.csv` (pieces of at most 200 answers, padded
    with -1) and `question_level/test_quelevel.csv` (one sequence per pupil). The pieces of one pupil
    are joined and ordered by timestamp, file order breaking ties. A question tagged with several
    concepts (`166_170`) keeps the first one as its skill. `meta.fold` is the fold of the release
    (-1 for its test set); Kwan makes its own split by pupil over everyone.
    """
    import numpy as np
    import pandas as pd

    base = find_base(root)
    cols = ["fold", "uid", "questions", "concepts", "responses", "timestamps"]
    users, items, skills, correct, stamps, folds = [], [], [], [], [], []
    for name in FILES:
        frame = pd.read_csv(base / name, usecols=cols)
        for fold, uid, qs, cs, rs, ts in zip(
            frame.fold, frame.uid, frame.questions, frame.concepts, frame.responses, frame.timestamps
        ):
            r = np.array(rs.split(","), dtype=np.int64)
            keep = r != PAD
            n = int(keep.sum())
            users.append(np.full(n, int(uid), dtype=np.int32))
            folds.append(np.full(n, int(fold), dtype=np.int8))
            items.append(np.array(qs.split(","), dtype=np.int32)[keep])
            skills.append(np.array([part.split("_", 1)[0] for part in cs.split(",")], dtype=np.int32)[keep])
            correct.append(r[keep].astype(np.int8))
            stamps.append(np.array(ts.split(","), dtype=np.int64)[keep])
    df = pd.DataFrame(
        {
            "user_id": np.concatenate(users),
            "item_id": np.concatenate(items),
            "skill_id": np.concatenate(skills),
            "t": pd.to_datetime(np.concatenate(stamps), unit="ms"),
            "correct": np.concatenate(correct),
            "meta_fold": np.concatenate(folds),
        }
    )
    return schema.conform(df, dataset_attrs(KEY))
