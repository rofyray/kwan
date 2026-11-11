"""Junyi Academy 2015 (Taiwan, K-12 maths): the problem log, for knowledge tracing.

Licence: non-commercial (cite Chang, Hsu & Chen, EDM 2015). Evaluation only: raw rows are never
committed and parameters fitted on this dataset never leave `build/` (PRD 5.3). The prerequisite
and relationship annotations of the same download are read by `kwan.graph.junyi`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kwan.data import schema
from kwan.data.loaders._common import dataset_attrs, need

KEY = "junyi_2015"
LOG_FILE = "junyi_ProblemLog_original.csv"


def _encode(values: Any, table: dict[Any, int]) -> Any:
    """Number the values of one chunk with a table shared by all chunks (first seen, first numbered)."""
    import numpy as np
    import pandas as pd

    codes, uniques = pd.factorize(values)
    lut = np.array([table.setdefault(u, len(table)) for u in uniques.tolist()], dtype=np.int64)
    return lut[codes] if len(lut) else np.empty(0, dtype=np.int64)


def load(root: str | Path, block_bytes: int = 32 << 20) -> Any:
    """The Junyi problem log in the interaction schema (about 26M rows).

    The 2.7 GB file is read in chunks of `block_bytes` with pyarrow, five columns only. Skill = the
    exercise; item = the exercise together with the problem template (`problem_type`). `correct` is
    the first attempt without a hint, as the dataset defines it. Exercise names are kept in
    `attrs["skill_names"]`. Ordered by student, then time done, file order breaking ties.
    """
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    from pyarrow import csv

    root = Path(root)
    path = need(root if root.is_file() else root / LOG_FILE, "run `make data`; the .rar needs bsdtar, unar or unrar")
    types = {
        "user_id": pa.int32(),
        "exercise": pa.string(),
        "problem_type": pa.string(),
        "time_done": pa.int64(),
        "correct": pa.bool_(),
    }
    reader = csv.open_csv(
        str(path),
        read_options=csv.ReadOptions(block_size=block_bytes),
        convert_options=csv.ConvertOptions(include_columns=list(types), column_types=types, strings_can_be_null=False),
    )
    exercises: dict[str, int] = {}
    items: dict[int, int] = {}
    templates: dict[str, int] = {}
    parts = []
    for batch in reader:
        chunk = batch.to_pandas()
        skill = _encode(chunk["exercise"].values, exercises)
        template = _encode(chunk["problem_type"].values, templates)
        item = _encode(skill * 1_000_000 + template, items)
        parts.append(
            pd.DataFrame(
                {
                    "user_id": chunk["user_id"].values,
                    "item_id": item.astype(np.int32),
                    "skill_id": skill.astype(np.int32),
                    "t": chunk["time_done"].values,
                    "correct": chunk["correct"].values.astype(np.int8),
                }
            )
        )
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=list(schema.CORE))
    df["t"] = pd.to_datetime(df["t"], unit="us")
    out = schema.conform(df, dataset_attrs(KEY))
    out.attrs["skill_names"] = {code: name for name, code in exercises.items()}
    return out
