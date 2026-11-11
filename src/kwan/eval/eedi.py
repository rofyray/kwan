"""Learner-model validation on Eedi NeurIPS 2020 (task 3/4 subset: ~1.5M answers, ~6k students, 948 questions).
Skill = most specific subject (deepest Level) attached to the question.
Same 80/20 student split + models as kwan.eval.kt."""

import argparse
import ast

import pandas as pd

from kwan.eval.kt import MODEL_ORDER, SPLIT, score_models, split, write_block

KEY = "eedi_2020_task34"  # top-level key this script owns in data/eval_results.json
DATASET = "Eedi NeurIPS 2020 Education Challenge, task 3/4 subset"
LICENCE = "research use; cite Wang et al. (2020) arXiv:2007.12061"


def load_eedi(root):
    tr = pd.read_csv(f"{root}/train_data/train_task_3_4.csv", usecols=["QuestionId", "UserId", "AnswerId", "IsCorrect"])
    am = pd.read_csv(f"{root}/metadata/answer_metadata_task_3_4.csv", usecols=["AnswerId", "DateAnswered"])
    qm = pd.read_csv(f"{root}/metadata/question_metadata_task_3_4.csv")
    sm = pd.read_csv(f"{root}/metadata/subject_metadata.csv", encoding="utf-8-sig")
    level = dict(zip(sm.SubjectId, sm.Level))
    qm["skill_id"] = qm.SubjectId.map(lambda s: max(ast.literal_eval(s), key=lambda x: level.get(x, -1)))
    df = tr.merge(am, on="AnswerId").merge(qm[["QuestionId", "skill_id"]], on="QuestionId")
    df = df.rename(columns={"UserId": "user_id", "QuestionId": "problem_id", "IsCorrect": "correct"})
    df["DateAnswered"] = pd.to_datetime(df.DateAnswered)
    return df.sort_values(["user_id", "DateAnswered", "AnswerId"]).reset_index(drop=True)


def evaluate(df, seed=42):
    """Split by student, run all models, print the console report. Returns the unrounded metrics per model."""
    tr, te = split(df, seed)
    print(
        f"rows={len(df)} students={df.user_id.nunique()} questions={df.problem_id.nunique()} "
        f"skills={df.skill_id.nunique()}"
    )
    return score_models(tr, te)


def make_block(df, scores):
    """The `eedi_2020_task34` block of data/eval_results.json: schema v2 (see `results.dataset_block`),
    with the v1 fields `questions`, `AUC` and `RMSE` kept beside `metrics` because the demo still reads them."""
    from kwan.data.schema import counts
    from kwan.eval import results

    return results.dataset_block(
        DATASET, LICENCE, counts(df), SPLIT, scores, model_order=MODEL_ORDER, legacy=("questions", "AUC", "RMSE")
    )


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Validate Kwan's learner model on Eedi NeurIPS 2020, task 3/4 (80/20 split by student)."
    )
    ap.add_argument("root", help="directory holding train_data/ and metadata/, e.g. raw/eedi/data")
    ap.add_argument(
        "--out", metavar="PATH", help=f"merge the '{KEY}' block into this results JSON, e.g. data/eval_results.json"
    )
    a = ap.parse_args(argv)
    df = load_eedi(a.root)
    scores = evaluate(df)
    block = make_block(df, scores)
    if a.out:
        write_block(KEY, block, a.out)
    return block


if __name__ == "__main__":
    main()
