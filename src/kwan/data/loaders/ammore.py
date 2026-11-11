"""AMMORE (Rori WhatsApp maths tutor, Rising Academies): African pupils, grades 6 to 9.

Licence: CC BY-NC 4.0. Evaluation only: raw rows are never committed and parameters fitted on this
dataset never leave `build/` (PRD 5.3).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kwan.data import schema
from kwan.data.loaders._common import dataset_attrs, need

KEY = "ammore_2024"
FILES = ("AMMORE_student_answers.csv", "AMMORE_demographic_data.csv")
SUBSETS = {"all": None, "ghana": "Ghana"}  # subset name -> country filter


def load(root: str | Path, subset: str = "all") -> Any:
    """AMMORE in the interaction schema. Wraps `kwan.eval.ammore.load_ammore`: same rows, same order.

    `subset` is "all" or "ghana" (pupils whose country is Ghana). Skill = GPF micro-lesson, item =
    question text; both are numbered within the subset. Answers marked "other" (not an answer attempt)
    are dropped. `meta` holds the pupil's country and the micro-lesson code.
    """
    from kwan.eval.ammore import load_ammore

    if subset not in SUBSETS:
        raise KeyError(f"unknown AMMORE subset '{subset}' (use one of: {', '.join(SUBSETS)})")
    root = Path(root)
    for name in FILES:
        need(root / name)
    df = load_ammore(str(root), SUBSETS[subset])
    df = df.rename(columns={"country": "meta_country", "question_micro_lesson": "meta_micro_lesson"})
    return schema.conform(df, dataset_attrs(KEY, subset))


def load_ghana(root: str | Path) -> Any:
    """The Ghana-only subset (`load(root, "ghana")`)."""
    return load(root, "ghana")
