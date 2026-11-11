"""One interaction schema for every learner dataset (PRD Phase 4.2).

An interaction is one marked answer:

    user_id   who answered (an opaque id from the dataset, never a name)
    item_id   which question
    skill_id  the skill the question practises
    t         when: a datetime, or an ordinal that grows with time (for example an order id)
    correct   1 right, 0 wrong
    meta      a dictionary of anything else the dataset gives for that answer

Loaders return a pandas DataFrame with the five core columns, in that order, sorted by student and
then by time. `meta` is stored column-wise so that 16 million rows stay small: every extra column is
named `meta_<field>`, and `records()` folds those columns back into one `meta` dictionary per row.
Facts about the whole dataset (its registry key, licence, subset) live in `df.attrs["dataset"]`.

pandas is imported inside the functions; importing this module needs only the standard library.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

CORE = ("user_id", "item_id", "skill_id", "t", "correct")
COLUMNS = (*CORE, "meta")
META_PREFIX = "meta_"
ALIASES = {"problem_id": "item_id"}  # v1 name -> schema name


class SchemaError(ValueError):
    """An interaction frame does not follow the schema."""


def meta_columns(df: Any) -> list[str]:
    """The `meta_<field>` columns of a frame, in frame order."""
    return [c for c in df.columns if c.startswith(META_PREFIX)]


def conform(df: Any, dataset: dict[str, Any] | None = None, sort: bool = True) -> Any:
    """Bring a frame to the schema: rename v1 aliases, order the columns, sort by student then time.

    Columns that are neither core nor `meta_*` are dropped. The sort is stable, so rows that share a
    student and a time keep the order the loader gave them. `dataset` is stored in `attrs["dataset"]`.
    """
    df = df.rename(columns={old: new for old, new in ALIASES.items() if old in df.columns and new not in df.columns})
    missing = [c for c in CORE if c not in df.columns]
    if missing:
        raise SchemaError(f"missing column(s): {', '.join(missing)}")
    df = df[[*CORE, *meta_columns(df)]]
    if sort:
        df = df.sort_values(["user_id", "t"], kind="stable")
    df = df.reset_index(drop=True)
    if dataset is not None:
        df.attrs["dataset"] = dict(dataset)
    return validate(df)


def validate(df: Any) -> Any:
    """Check a frame against the schema and return it unchanged. Raises SchemaError with the reason."""
    import pandas as pd
    from pandas.api import types as pt

    missing = [c for c in CORE if c not in df.columns]
    if missing:
        raise SchemaError(f"missing column(s): {', '.join(missing)}")
    extra = [c for c in df.columns if c not in CORE and not c.startswith(META_PREFIX)]
    if extra:
        raise SchemaError(f"column(s) outside the schema (name them meta_<field>): {', '.join(extra)}")
    for col in CORE:
        if df[col].isna().any():
            raise SchemaError(f"column '{col}' has missing values")
    if not (pt.is_integer_dtype(df["correct"]) or pt.is_bool_dtype(df["correct"])):
        raise SchemaError("column 'correct' must hold whole numbers 0 or 1")
    if not df["correct"].isin([0, 1]).all():
        raise SchemaError("column 'correct' must hold only 0 or 1")
    if not (pt.is_datetime64_any_dtype(df["t"]) or pt.is_numeric_dtype(df["t"])):
        raise SchemaError("column 't' must be a datetime or a number that grows with time")
    if len(df):
        same_user = df["user_id"].values[1:] == df["user_id"].values[:-1]
        if pd.Index(df["user_id"]).nunique() != int((~same_user).sum()) + 1:
            raise SchemaError("rows of one student must sit together (sort by user_id, then t)")
        t = df["t"].values
        if (same_user & (t[1:] < t[:-1])).any():
            raise SchemaError("rows of one student must be in time order (sort by user_id, then t)")
    return df


def is_valid(df: Any) -> bool:
    """True when `validate` accepts the frame."""
    try:
        validate(df)
    except SchemaError:
        return False
    return True


def records(df: Any) -> Iterator[dict[str, Any]]:
    """Yield each row as `{"user_id", "item_id", "skill_id", "t", "correct", "meta": {...}}`."""
    metas = meta_columns(df)
    names = [c[len(META_PREFIX) :] for c in metas]
    for row in df[[*CORE, *metas]].itertuples(index=False, name=None):
        rec: dict[str, Any] = dict(zip(CORE, row[: len(CORE)]))
        rec["meta"] = dict(zip(names, row[len(CORE) :]))
        yield rec


def with_problem_id(df: Any) -> Any:
    """A copy of the frame that also carries the v1 column name `problem_id` (same values as `item_id`)."""
    out = df.copy()
    out["problem_id"] = out["item_id"]
    return out


def counts(df: Any) -> dict[str, int]:
    """rows, students, items and skills of a frame (v1 frames with `problem_id` are accepted)."""
    item = "item_id" if "item_id" in df.columns else "problem_id"
    out = {"rows": int(len(df)), "students": int(df["user_id"].nunique())}
    if item in df.columns:
        out["items"] = int(df[item].nunique())
    out["skills"] = int(df["skill_id"].nunique())
    return out
