"""Learner-model validation on AMMORE (Rori, Rising Academies): 53k answers from African pupils, grades 6 to 9.
Skill = GPF micro-lesson (e.g. G6.N2.2.3.10). Item = question text. "other" answers are dropped.
Reports the full dataset and the Ghana-only subset. Data licence: CC BY-NC 4.0 (evaluation only, never committed)."""

import argparse

import numpy as np
import pandas as pd

from kwan.eval.kt import MODEL_ORDER, SPLIT, score_models, split, write_block

KEY = "ammore_2024"
DATASET = "AMMORE (Rori WhatsApp maths tutor, Rising Academies), Jan to Apr 2024"
LICENCE = "CC BY-NC 4.0"
SUBSETS = (("all", None), ("ghana", "Ghana"))  # sub-block name, country filter


def load_ammore(root, country=None):
    d = pd.read_csv(f"{root}/AMMORE_student_answers.csv")
    dm = pd.read_csv(f"{root}/AMMORE_demographic_data.csv").rename(columns={"user ID": "user_id"})
    d = d[d.human_label.isin(["correct_answer", "wrong_answer"])].copy()
    d["correct"] = (d.human_label == "correct_answer").astype(int)
    d["t"] = pd.to_datetime(d.message_inserted_at, format="%m/%d/%y %H:%M")
    d["row"] = np.arange(len(d))  # file order breaks same-minute ties
    d = d.merge(dm[["user_id", "country"]], on="user_id", how="left")
    if country:
        d = d[d.country == country]
    d["problem_id"] = d.question.astype("category").cat.codes
    d["skill_id"] = d.question_micro_lesson.astype("category").cat.codes
    return d.sort_values(["user_id", "t", "row"]).reset_index(drop=True)


def block_for(df, scores):
    """Counts and metrics of one subset (`all` or `ghana`): the schema v2 fields of `results.dataset_block`
    without `dataset` and `licence` (they sit once at the top of the block), plus the v1 fields
    `questions`, `AUC` and `RMSE`."""
    from kwan.data.schema import counts
    from kwan.eval import results

    block = results.dataset_block(
        DATASET, LICENCE, counts(df), SPLIT, scores, model_order=MODEL_ORDER, legacy=("questions", "AUC", "RMSE")
    )
    del block["dataset"], block["licence"]
    return block


def main(argv=None):
    ap = argparse.ArgumentParser(description="Validate Kwan's learner model on AMMORE (all countries and Ghana only).")
    ap.add_argument("root", help="directory holding AMMORE_student_answers.csv, e.g. raw/ammore")
    ap.add_argument(
        "--out", metavar="PATH", help=f"merge the '{KEY}' block into this results JSON, e.g. data/eval_results.json"
    )
    a = ap.parse_args(argv)
    block = {"dataset": DATASET, "licence": LICENCE}
    for name, country in SUBSETS:
        df = load_ammore(a.root, country)
        tr, te = split(df)
        print(f"[{name}] rows={len(df)} students={df.user_id.nunique()} skills={df.skill_id.nunique()}")
        block[name] = block_for(df, score_models(tr, te))
    if a.out:
        write_block(KEY, block, a.out)
    return block


if __name__ == "__main__":
    main()
