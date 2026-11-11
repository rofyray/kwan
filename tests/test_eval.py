"""Learner-model validation scripts (eval_kt.py, eval_eedi.py).

Unit tests use small synthetic data and never touch raw/ or the real data/eval_results.json.
The real-dataset acceptance tests (Kwan elo-multi AUC >= 0.72) are skipped when raw/ data is absent.
ASSISTments takes about 2 s and runs by default. Eedi takes longer, carries the `slow` marker and
only runs when the environment has KWAN_RUN_SLOW=1.
"""

import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kwan.eval import eedi as eval_eedi
from kwan.eval import kt as eval_kt
from kwan.eval.kt import BKT, Elo, Elo2

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw"
ASSIST_CSV = RAW / "skill_builder_data_corrected.csv"
EEDI_ROOT = RAW / "eedi" / "data"
EEDI_FILES = [
    EEDI_ROOT / "train_data" / "train_task_3_4.csv",
    EEDI_ROOT / "metadata" / "answer_metadata_task_3_4.csv",
    EEDI_ROOT / "metadata" / "question_metadata_task_3_4.csv",
    EEDI_ROOT / "metadata" / "subject_metadata.csv",
]
RUN_SLOW = os.environ.get("KWAN_RUN_SLOW") == "1"

MODELS = ["global-mean", "skill-mean", "elo", "elo-multi (Kwan)"]
LINE = re.compile(r"^(\S+)\s+AUC=(\d\.\d{4}) RMSE=(\d\.\d{4})$")


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def frame(rows):
    """rows of (user_id, problem_id, skill_id, correct), already in answer order."""
    return pd.DataFrame(rows, columns=["user_id", "problem_id", "skill_id", "correct"])


def synthetic_logs(n_users=60, n_skills=5, n_items=40, per_user=30, seed=0):
    """IRT-like answer log: abler students and easier items give more correct answers."""
    rng = np.random.default_rng(seed)
    ability = rng.normal(0, 1, n_users)
    difficulty = rng.normal(0, 1, n_items)
    rows = []
    for u in range(n_users):
        for it in rng.integers(0, n_items, per_user):
            p = sigmoid(ability[u] - difficulty[it])
            rows.append((100 + u, int(it), int(it % n_skills), int(rng.random() < p)))
    return frame(rows)


@pytest.fixture(scope="module")
def logs():
    return synthetic_logs()


# ---------------------------------------------------------------- split


def test_split_is_80_20_by_student_with_no_overlap(logs):
    tr, te = eval_kt.split(logs)
    train_u, test_u = set(tr.user_id), set(te.user_id)
    assert len(test_u) == int(0.2 * logs.user_id.nunique()) == 12
    assert len(train_u) == 48
    assert train_u.isdisjoint(test_u)
    assert train_u | test_u == set(logs.user_id)
    assert len(tr) + len(te) == len(logs)
    assert sorted(tr.index.tolist() + te.index.tolist()) == logs.index.tolist()


def test_split_is_deterministic_for_seed_42(logs):
    before = logs.copy()
    tr1, te1 = eval_kt.split(logs)
    tr2, te2 = eval_kt.split(logs, seed=42)
    pd.testing.assert_frame_equal(tr1, tr2)
    pd.testing.assert_frame_equal(te1, te2)
    pd.testing.assert_frame_equal(logs, before)  # the input frame is not mutated
    assert set(eval_kt.split(logs, seed=7)[1].user_id) != set(te1.user_id)


def test_split_keeps_row_order_within_each_part(logs):
    tr, te = eval_kt.split(logs)
    assert tr.index.is_monotonic_increasing and te.index.is_monotonic_increasing


# ---------------------------------------------------------------- load (ASSISTments)

ASSIST_ROWS = """order_id,user_id,problem_id,skill_id,correct,original,skill_name
50,2,10,1,1,1,Addition
30,1,11,2,0,1,Café sums
40,1,12,,1,1,
60,1,13,2,1,0,Scaffold step
10,2,14,3,1,1,First copy
10,2,14,7,0,1,Second skill tag of the same attempt
20,1,15,2,1,1,Subtraction
80,3,16,4,0,0,Scaffold with the same order id
80,3,16,4,1,1,Main problem
"""


def test_load_filters_dedupes_and_sorts(tmp_path):
    path = tmp_path / "tiny.csv"
    path.write_bytes(ASSIST_ROWS.encode("latin-1"))  # the real file is latin-1, not UTF-8
    df = eval_kt.load(path)

    assert df.index.tolist() == list(range(5))
    assert df.user_id.tolist() == [1, 1, 2, 2, 3]  # sorted by user_id, then order_id
    assert df.order_id.tolist() == [20, 30, 10, 50, 80]
    assert df.order_id.is_unique  # order_id deduped
    assert 40 not in set(df.order_id)  # missing skill dropped
    assert 60 not in set(df.order_id)  # original == 0 dropped
    assert (df.original == 1).all()
    assert df.skill_id.tolist() == [2, 2, 3, 1, 4]  # first row of a duplicated order_id wins
    assert pd.api.types.is_integer_dtype(df.skill_id)
    assert df.correct.tolist() == [1, 0, 1, 1, 1]  # the original == 1 row of order 80 is kept
    assert set(df.columns) == {"order_id", "user_id", "problem_id", "skill_id", "correct", "original"}


# ---------------------------------------------------------------- Elo


def test_elo_probabilities_are_strictly_between_0_and_1(logs):
    tr, te = eval_kt.split(logs)
    elo = Elo()
    for preds in (elo.run(tr, learn_beta=True), elo.run(te, learn_beta=False)):
        assert len(preds) > 0 and np.isfinite(preds).all()
        assert ((preds > 0) & (preds < 1)).all()


def test_elo_moves_up_after_correct_and_down_after_wrong():
    up = Elo().run(frame([(1, 9, 5, 1)] * 4), learn_beta=False)
    down = Elo().run(frame([(1, 9, 5, 0)] * 4), learn_beta=False)
    assert up[0] == down[0] == 0.5
    assert (np.diff(up) > 0).all() and (np.diff(down) < 0).all()
    # hand-computed: theta = 0 + 0.4 * (1 - 0.5) = 0.2, then K decays to 0.4 / 1.05
    assert up[1] == pytest.approx(sigmoid(0.2))
    assert up[2] == pytest.approx(sigmoid(0.2 + 0.4 / 1.05 * (1 - sigmoid(0.2))))
    assert down[1] == pytest.approx(sigmoid(-0.2))


def test_elo_ability_is_per_student_and_skill():
    preds = Elo().run(frame([(1, 9, 5, 1), (1, 9, 6, 1), (2, 9, 5, 1)]), learn_beta=False)
    assert preds.tolist() == [0.5, 0.5, 0.5]  # new skill or new student starts from 0


def test_elo_learns_difficulty_only_when_asked():
    df = frame([(1, 9, 5, 1), (2, 9, 5, 1), (3, 9, 5, 0)])
    frozen = Elo()
    frozen.run(df, learn_beta=False)
    assert frozen.beta == {}
    learner = Elo()
    learner.run(df[:1], learn_beta=True)
    assert learner.beta == {5: pytest.approx(-0.025)}  # 0 - 0.05 * (1 - 0.5): the skill looks easier
    snapshot = dict(learner.beta)
    learner.run(df, learn_beta=False)
    assert learner.beta == snapshot


# ---------------------------------------------------------------- Elo2 (Kwan's model)


def test_elo2_probabilities_are_strictly_between_0_and_1(logs):
    tr, te = eval_kt.split(logs)
    m = Elo2()
    for preds in (m.run(tr, True), m.run(tr, True), m.run(te, False)):
        assert len(preds) > 0 and np.isfinite(preds).all()
        assert ((preds > 0) & (preds < 1)).all()


def test_elo2_moves_up_after_correct_and_down_after_wrong():
    up = Elo2().run(frame([(1, 9, 5, 1)] * 4), learn=False)
    down = Elo2().run(frame([(1, 9, 5, 0)] * 4), learn=False)
    assert up[0] == down[0] == 0.5
    assert (np.diff(up) > 0).all() and (np.diff(down) < 0).all()
    # hand-computed: theta_u = 0.3 * 0.5 = 0.15 and theta_us = 0.4 * 0.5 = 0.2
    assert up[1] == pytest.approx(sigmoid(0.35))
    assert down[1] == pytest.approx(sigmoid(-0.35))


def test_elo2_student_level_ability_transfers_across_skills():
    preds = Elo2().run(frame([(1, 9, 5, 1), (1, 8, 6, 1), (2, 9, 5, 1)]), learn=False)
    assert preds[1] == pytest.approx(sigmoid(0.15))  # other skill: only the student-level term
    assert preds[2] == 0.5  # other student: nothing transfers


def test_elo2_does_not_learn_item_difficulty_when_learn_is_false():
    df = frame([(1, 9, 5, 1), (2, 9, 5, 0), (3, 8, 6, 1)])
    m = Elo2()
    m.run(df, learn=False)
    assert m.b_item == {} and m.b_skill == {} and m.n_item == {}

    m.run(df[:1], learn=True)
    assert m.b_item == {9: pytest.approx(-0.15)}  # 0 - 0.3 * (1 - 0.5)
    assert m.b_skill == {5: pytest.approx(-0.005)}  # 0 - 0.01 * (1 - 0.5)
    assert m.n_item == {9: 1}
    snapshot = (dict(m.b_item), dict(m.b_skill), dict(m.n_item))
    m.run(df, learn=False)
    assert (m.b_item, m.b_skill, m.n_item) == snapshot


def test_elo2_unseen_item_falls_back_to_skill_difficulty():
    m = Elo2()
    m.b_item, m.b_skill = {9: 1.0}, {5: -2.0}
    preds = m.run(frame([(1, 9, 5, 1), (2, 77, 5, 1), (3, 77, 6, 1)]), learn=False)
    assert preds[0] == pytest.approx(sigmoid(-1.0))  # known item
    assert preds[1] == pytest.approx(sigmoid(2.0))  # unseen item, known skill
    assert preds[2] == 0.5  # unseen item, unseen skill


def test_elo2_beats_the_global_mean_on_synthetic_logs():
    scores = eval_kt.score_models(*eval_kt.split(synthetic_logs(n_users=200, per_user=40, seed=3)))
    assert scores["AUC"]["global-mean"] == 0.5
    assert scores["AUC"]["elo-multi (Kwan)"] > 0.6


# ---------------------------------------------------------------- BKT


def test_bkt_seq_preds_matches_hand_computation():
    # pi=0.4, pt=0.1, ps=0.1, pg=0.2
    # step 1: p = 0.4*0.9 + 0.6*0.2 = 0.48; correct: post = 0.36/0.48 = 0.75; learn: 0.75 + 0.25*0.1 = 0.775
    # step 2: p = 0.775*0.9 + 0.225*0.2 = 0.7425; wrong: post = 0.0775/0.2575; learn: post + (1 - post)*0.1
    # step 3: p = L*0.9 + (1 - L)*0.2 with L = 0.0775/0.2575*0.9 + 0.1
    out = BKT.seq_preds([1, 0, 1], 0.4, 0.1, 0.1, 0.2)
    L = 0.0775 / 0.2575 * 0.9 + 0.1
    assert out == pytest.approx([0.48, 0.7425, L * 0.9 + (1 - L) * 0.2])
    assert out[2] == pytest.approx(0.4596116505)
    assert BKT.seq_preds([], 0.4, 0.1, 0.1, 0.2) == []


def bkt_toy():
    rows = []
    for u in (1, 2, 3):
        rows += [(u, 1, 10, 1)] * 5  # skill 10: everyone always right
        rows += [(u, 2, 20, 0)] * 5  # skill 20: everyone always wrong
    return frame(rows)


def test_bkt_fit_picks_grid_params_that_match_the_data():
    b = BKT()
    b.fit(bkt_toy())
    assert set(b.params) == {10, 20}
    assert all(prm in BKT.GRID for prm in b.params.values())
    pi, pt, ps, pg = b.params[10]
    assert (pi, pt, ps) == (0.6, 0.3, 0.05)  # known early, learned fast, few slips
    pi, pt, ps, pg = b.params[20]
    assert (pi, pt, pg) == (0.2, 0.05, 0.1)  # not known, learned slowly, few guesses


def test_bkt_predict_is_aligned_to_rows_and_uses_default_for_unseen_skill():
    b = BKT()
    b.fit(bkt_toy())
    te = frame([(7, 1, 10, 1), (8, 2, 20, 0), (7, 3, 99, 1), (7, 1, 10, 1), (8, 2, 20, 0), (7, 3, 99, 0)])
    te.index = [40, 10, 30, 60, 20, 50]  # predictions must follow row order, not index
    preds = b.predict(te)
    assert preds.shape == (6,) and ((preds > 0) & (preds < 1)).all()
    assert preds[[0, 3]] == pytest.approx(BKT.seq_preds([1, 1], *b.params[10]))
    assert preds[[1, 4]] == pytest.approx(BKT.seq_preds([0, 0], *b.params[20]))
    assert preds[[2, 5]] == pytest.approx(BKT.seq_preds([1, 0], 0.4, 0.1, 0.1, 0.2))
    assert preds[2] == pytest.approx(0.48)
    assert preds[0] > 0.5 > preds[1]  # always-right skill vs always-wrong skill


# ---------------------------------------------------------------- load_eedi


def make_eedi_dir(root):
    (root / "train_data").mkdir(parents=True)
    (root / "metadata").mkdir()
    (root / "train_data" / "train_task_3_4.csv").write_text(
        "QuestionId,UserId,AnswerId,IsCorrect,CorrectAnswer,AnswerValue\n"
        "0,7,100,1,2,2\n"
        "1,7,101,0,3,1\n"
        "2,5,102,1,1,1\n"
        "0,5,103,0,2,4\n"
        "1,5,104,1,3,3\n"
        "2,7,105,1,1,1\n"
        "0,9,106,1,2,2\n",
        encoding="utf-8",
    )
    (root / "metadata" / "answer_metadata_task_3_4.csv").write_text(
        "AnswerId,DateAnswered,Confidence,GroupId,QuizId,SchemeOfWorkId\n"
        "100,2020-01-06 18:53:00.000,75.0,1,1,52562.0\n"
        "101,2019-10-30 14:34:00.000,,1,1,52562.0\n"
        "102,2020-02-01 09:00:00.000,,1,1,52562.0\n"
        "103,2020-02-01 09:00:00.000,,1,1,52562.0\n"
        "104,2020-01-01 09:00:00.000,,1,1,52562.0\n"
        "105,2020-03-01 09:00:00.000,,1,1,52562.0\n",
        encoding="utf-8",
    )  # answer 106 has no metadata
    (root / "metadata" / "question_metadata_task_3_4.csv").write_text(
        'QuestionId,SubjectId\n0,"[3, 32, 49, 33]"\n1,"[3, 32]"\n2,"[3, 999, 49]"\n', encoding="utf-8"
    )
    (root / "metadata" / "subject_metadata.csv").write_bytes(  # UTF-8 with BOM and CRLF, like the real file
        "\ufeffSubjectId,Name,ParentId,Level\r\n"
        "3,Maths,NULL,0\r\n"
        "32,Number,3,1\r\n"
        "49,BIDMAS,32,2\r\n"
        "33,Basic Arithmetic,49,3\r\n".encode("utf-8")
    )
    return root


def test_load_eedi_picks_deepest_subject_and_handles_bom(tmp_path):
    root = make_eedi_dir(tmp_path / "data")
    assert (root / "metadata" / "subject_metadata.csv").read_bytes().startswith(b"\xef\xbb\xbf")
    df = eval_eedi.load_eedi(root)

    assert {"user_id", "problem_id", "skill_id", "correct"} <= set(df.columns)
    skill = dict(zip(df.problem_id, df.skill_id))
    assert skill == {0: 33, 1: 32, 2: 49}  # deepest Level wins; unknown subject 999 never wins
    assert 106 not in set(df.AnswerId) and len(df) == 6  # answers without metadata are dropped
    # sorted by student, then time answered, then AnswerId as the tie-break
    assert df.user_id.tolist() == [5, 5, 5, 7, 7, 7]
    assert df.AnswerId.tolist() == [104, 102, 103, 101, 100, 105]
    assert df.correct.tolist() == [1, 1, 0, 0, 1, 1]
    assert pd.api.types.is_datetime64_any_dtype(df.DateAnswered)
    assert df.index.tolist() == list(range(6))


# ---------------------------------------------------------------- evaluate, console format, --out writer


def parse_console(text):
    lines = text.strip().splitlines()
    return lines[0], {m.group(1): (float(m.group(2)), float(m.group(3))) for m in map(LINE.match, lines[1:])}


def check_block_matches_console(block, console, bkt):
    printed = dict(console)
    printed["elo-multi (Kwan)"] = printed.pop("elo-multi")  # console name vs results-file name
    expected = MODELS[:3] + (["bkt"] if bkt else []) + MODELS[3:]
    assert list(block["AUC"]) == list(block["RMSE"]) == expected
    for model in expected:
        assert (block["AUC"][model], block["RMSE"][model]) == printed[model]
        for metric in ("AUC", "RMSE"):
            value = block[metric][model]
            assert isinstance(value, float) and value == round(value, 4) and 0 <= value <= 1


V2_FIELDS = ["dataset", "licence", "rows", "students", "items", "skills", "split", "metrics"]


def check_metrics_match_v1_fields(block):
    """`metrics` is {model: {AUC, RMSE, ACC, ECE}} and repeats the v1 `AUC` and `RMSE` tables exactly."""
    assert list(block["metrics"]) == list(block["AUC"]) == list(block["RMSE"])
    for model, got in block["metrics"].items():
        assert list(got) == ["AUC", "RMSE", "ACC", "ECE"]
        assert (got["AUC"], got["RMSE"]) == (block["AUC"][model], block["RMSE"][model])
        assert all(isinstance(v, float) and v == round(v, 4) and 0 <= v <= 1 for v in got.values())


def test_evaluate_prints_the_prd_console_format(logs, capsys):
    scores = eval_kt.evaluate(logs, bkt=True)
    out = capsys.readouterr().out
    header, console = parse_console(out)
    tr, te = eval_kt.split(logs)
    assert header == f"rows={len(logs)} students=60 skills=5 train={len(tr)} test={len(te)}"
    lines = out.splitlines()[1:]
    assert [ln.split()[0] for ln in lines] == ["global-mean", "skill-mean", "elo", "elo-multi", "bkt"]
    assert all(LINE.match(ln) for ln in lines)
    assert lines[0].startswith("global-mean  AUC=0.5000 RMSE=")
    assert lines[2].startswith("elo          AUC=")
    assert lines[3].startswith("elo-multi    AUC=")
    assert list(scores) == ["AUC", "RMSE", "ACC", "ECE"]  # v2: accuracy and calibration error beside AUC and RMSE
    assert all(set(scores[metric]) == set(MODELS) | {"bkt"} for metric in scores)
    check_block_matches_console(eval_kt.make_block(logs, scores), console, bkt=True)


def test_evaluate_without_bkt_has_no_bkt_entry(logs, capsys):
    scores = eval_kt.evaluate(logs)
    assert "bkt" not in capsys.readouterr().out
    assert list(scores["AUC"]) == MODELS and list(scores["RMSE"]) == MODELS


def write_assist_csv(path, df):
    out = df.copy()
    out["order_id"] = np.arange(len(out))[::-1] + 1000  # sort order must come from load(), not the file
    out["original"] = 1
    out.sample(frac=1, random_state=1).to_csv(path, index=False, encoding="latin-1")
    return path


OTHER_BLOCKS = {
    "simulation": {"setup": "kept as is", "new_skills_mastered": {"gapped_p4": {"kwan": 14.0}}},
    "latency": {"recommend_top3_mean_ms": 0.24},
}


def test_out_writes_assistments_block_and_preserves_other_keys(tmp_path, capsys):
    logs = synthetic_logs(n_users=40, per_user=20, seed=5)
    csv = write_assist_csv(tmp_path / "assist.csv", logs)
    out = tmp_path / "results" / "eval_results.json"
    out.parent.mkdir()
    out.write_text(json.dumps({**OTHER_BLOCKS, "assistments_2009": {"stale": True}}), encoding="utf-8")

    block = eval_kt.main([str(csv), "--bkt", "--out", str(out)])
    header, console = parse_console(capsys.readouterr().out)

    saved = json.loads(out.read_text(encoding="utf-8"))
    assert list(saved) == ["simulation", "latency", "assistments_2009"]
    assert {k: saved[k] for k in OTHER_BLOCKS} == OTHER_BLOCKS
    got = saved["assistments_2009"]
    assert got == block
    assert list(got) == V2_FIELDS + ["AUC", "RMSE"]  # schema v2, with the v1 fields kept for the demo
    check_metrics_match_v1_fields(got)
    assert got["items"] == logs.problem_id.nunique()
    assert got["licence"] == "research use; cite Feng, Heffernan & Koedinger (2009)"
    assert got["dataset"] == "ASSISTments 2009 skill-builder (corrected)"
    assert got["split"] == "80/20 by student, seed 42"
    assert (got["rows"], got["students"], got["skills"]) == (len(logs), 40, 5)
    assert all(type(got[k]) is int for k in ("rows", "students", "skills"))
    assert header.startswith(f"rows={len(logs)} students=40 skills=5 train=")
    check_block_matches_console(got, console, bkt=True)
    assert [p.name for p in out.parent.iterdir()] == ["eval_results.json"]  # no temp file left behind


def test_out_without_bkt_omits_bkt_and_creates_the_file(tmp_path, capsys):
    csv = write_assist_csv(tmp_path / "assist.csv", synthetic_logs(n_users=40, per_user=20, seed=5))
    out = tmp_path / "new" / "eval_results.json"  # neither the folder nor the file exists yet
    eval_kt.main([str(csv), "--out", str(out)])
    _, console = parse_console(capsys.readouterr().out)
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert list(saved) == ["assistments_2009"]
    check_block_matches_console(saved["assistments_2009"], console, bkt=False)


def test_no_out_means_no_file_is_written(tmp_path, capsys, monkeypatch):
    csv = write_assist_csv(tmp_path / "assist.csv", synthetic_logs(n_users=40, per_user=20, seed=5))
    monkeypatch.chdir(tmp_path)
    block = eval_kt.main([str(csv)])
    assert block["rows"] == 800
    assert sorted(p.name for p in tmp_path.iterdir()) == ["assist.csv"]


def big_eedi_dir(root, logs):
    """Write `logs` in the Eedi file layout, shuffled, so load_eedi has to restore the order."""
    make_eedi_dir(root)
    n = len(logs)
    stamps = pd.Timestamp("2020-01-01") + pd.to_timedelta(np.arange(n), unit="min")
    answers = pd.DataFrame(
        {
            "QuestionId": logs.problem_id,
            "UserId": logs.user_id,
            "AnswerId": np.arange(n) + 1,
            "IsCorrect": logs.correct,
            "CorrectAnswer": 1,
            "AnswerValue": 1,
        }
    )
    answers.sample(frac=1, random_state=2).to_csv(root / "train_data" / "train_task_3_4.csv", index=False)
    pd.DataFrame(
        {
            "AnswerId": answers.AnswerId,
            "DateAnswered": stamps.strftime("%Y-%m-%d %H:%M:%S.000"),
            "Confidence": "",
            "GroupId": 1,
            "QuizId": 1,
            "SchemeOfWorkId": 52562.0,
        }
    ).to_csv(root / "metadata" / "answer_metadata_task_3_4.csv", index=False)
    subjects = {0: "[3, 32]", 1: "[3, 32, 49]", 2: "[3, 32, 49, 33]"}
    qids = sorted(logs.problem_id.unique())
    pd.DataFrame({"QuestionId": qids, "SubjectId": [subjects[q % 3] for q in qids]}).to_csv(
        root / "metadata" / "question_metadata_task_3_4.csv", index=False
    )
    return root


def test_out_writes_eedi_block_and_preserves_other_keys(tmp_path, capsys):
    logs = synthetic_logs(n_users=40, n_items=12, per_user=20, seed=6)
    root = big_eedi_dir(tmp_path / "data", logs)
    out = tmp_path / "eval_results.json"
    assist = {"dataset": "ASSISTments 2009 skill-builder (corrected)", "AUC": {"elo-multi (Kwan)": 0.7545}}
    out.write_text(json.dumps({"assistments_2009": assist, **OTHER_BLOCKS}), encoding="utf-8")

    block = eval_eedi.main([str(root), "--out", str(out)])
    header, console = parse_console(capsys.readouterr().out)

    saved = json.loads(out.read_text(encoding="utf-8"))
    assert list(saved) == ["assistments_2009", "simulation", "latency", "eedi_2020_task34"]
    assert saved["assistments_2009"] == assist
    assert {k: saved[k] for k in OTHER_BLOCKS} == OTHER_BLOCKS
    got = saved["eedi_2020_task34"]
    assert got == block
    assert list(got) == [*V2_FIELDS[:5], "questions", *V2_FIELDS[5:], "AUC", "RMSE"]
    check_metrics_match_v1_fields(got)
    assert got["items"] == got["questions"]
    assert got["licence"] == "research use; cite Wang et al. (2020) arXiv:2007.12061"
    assert got["dataset"] == "Eedi NeurIPS 2020 Education Challenge, task 3/4 subset"
    assert got["split"] == "80/20 by student, seed 42"
    assert (got["rows"], got["students"], got["questions"], got["skills"]) == (800, 40, logs.problem_id.nunique(), 3)
    assert header == f"rows=800 students=40 questions={logs.problem_id.nunique()} skills=3"
    check_block_matches_console(got, console, bkt=False)


def test_both_scripts_merge_into_one_file(tmp_path, capsys):
    logs = synthetic_logs(n_users=40, n_items=12, per_user=20, seed=6)
    out = tmp_path / "eval_results.json"
    a = eval_kt.main([str(write_assist_csv(tmp_path / "assist.csv", logs)), "--out", str(out)])
    e = eval_eedi.main([str(big_eedi_dir(tmp_path / "data", logs)), "--out", str(out)])
    assert json.loads(out.read_text(encoding="utf-8")) == {"assistments_2009": a, "eedi_2020_task34": e}


def test_cli_runs_as_a_script_from_another_directory(tmp_path):
    csv = write_assist_csv(tmp_path / "assist.csv", synthetic_logs(n_users=40, per_user=20, seed=5))
    out = tmp_path / "eval_results.json"
    run = subprocess.run(
        [sys.executable, "-m", "kwan.eval.kt", str(csv), "--out", str(out)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert run.returncode == 0, run.stderr
    header, console = parse_console(run.stdout)  # stdout carries only the PRD console lines
    assert list(console) == ["global-mean", "skill-mean", "elo", "elo-multi"]
    check_block_matches_console(json.loads(out.read_text(encoding="utf-8"))["assistments_2009"], console, bkt=False)


EM_DASH = chr(0x2014)


def test_user_facing_text_has_no_em_dash():
    for mod in (eval_kt, eval_eedi):
        assert EM_DASH not in Path(mod.__file__).read_text(encoding="utf-8")
    assert EM_DASH not in (ROOT / "scripts" / "download_data.sh").read_text(encoding="utf-8")


# ---------------------------------------------------------------- real datasets (acceptance: AUC >= 0.72)

needs_slow = pytest.mark.skipif(not RUN_SLOW, reason="slow: set KWAN_RUN_SLOW=1 to run the real Eedi evaluation")


@pytest.mark.skipif(not ASSIST_CSV.exists(), reason="raw/skill_builder_data_corrected.csv is absent (run make data)")
def test_real_assistments_kwan_auc_meets_acceptance(capsys):
    df = eval_kt.load(ASSIST_CSV)  # about 2 s in total, so it runs by default
    scores = eval_kt.evaluate(df)  # no BKT: the grid search takes minutes
    header, _ = parse_console(capsys.readouterr().out)
    assert header == "rows=259399 students=4163 skills=111 train=207150 test=52249"
    assert scores["AUC"]["elo-multi (Kwan)"] >= 0.72
    assert scores["AUC"]["elo-multi (Kwan)"] > scores["AUC"]["elo"] > scores["AUC"]["skill-mean"] > 0.5
    assert scores["RMSE"]["elo-multi (Kwan)"] < scores["RMSE"]["global-mean"]


@pytest.mark.slow
@needs_slow
@pytest.mark.skipif(not all(p.exists() for p in EEDI_FILES), reason="raw/eedi/data is absent (run make data)")
def test_real_eedi_kwan_auc_meets_acceptance(capsys):
    df = eval_eedi.load_eedi(EEDI_ROOT)
    scores = eval_eedi.evaluate(df)
    header, _ = parse_console(capsys.readouterr().out)
    assert header == "rows=1382727 students=4918 questions=948 skills=57"
    assert scores["AUC"]["elo-multi (Kwan)"] >= 0.72
    assert scores["AUC"]["elo-multi (Kwan)"] > scores["AUC"]["elo"] > scores["AUC"]["skill-mean"] > 0.5
    assert scores["RMSE"]["elo-multi (Kwan)"] < scores["RMSE"]["global-mean"]
