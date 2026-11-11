"""The evaluation harness v2: splits, blocks of schema v2, `kwan eval <dataset>` and `kwan eval all`.

Everything here runs on small synthetic logs written into temporary folders in each dataset's real
file layout. Nothing reads raw/ or the committed data/eval_results.json.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kwan import cli
from kwan.data import registry
from kwan.eval import ammore as eval_ammore
from kwan.eval import eedi as eval_eedi
from kwan.eval import harness, kt, results

ROOT = Path(__file__).resolve().parent.parent
EM_DASH = chr(0x2014)
MODELS = ["global-mean", "skill-mean", "elo", "elo-multi (Kwan)"]
V2_FIELDS = ["dataset", "licence", "rows", "students", "items", "skills", "split", "metrics"]
RESULT_KEYS = [
    "assistments_2009",
    "eedi_2020_task34",
    "eedi_2020_task12",
    "ammore_2024",
    "assistments_2012",
    "assistments_2015",
    "assistments_2017",
    "xes3g5m",
    "junyi_2015",
]


def synthetic_logs(n_users=60, n_skills=5, n_items=40, per_user=30, seed=0):
    """IRT-like answer log in the interaction schema: abler students and easier items give more right answers."""
    rng = np.random.default_rng(seed)
    ability, difficulty = rng.normal(0, 1, n_users), rng.normal(0, 1, n_items)
    rows = []
    for u in range(n_users):
        for step, it in enumerate(rng.integers(0, n_items, per_user)):
            p = 1 / (1 + math.exp(-(ability[u] - difficulty[it])))
            rows.append((100 + u, int(it), int(it % n_skills), step, int(rng.random() < p)))
    return pd.DataFrame(rows, columns=["user_id", "item_id", "skill_id", "t", "correct"])


@pytest.fixture(scope="module")
def logs():
    return synthetic_logs()


def check_v2_block(block, models=MODELS, extra=()):
    assert list(block) == [*V2_FIELDS, *extra]
    assert all(type(block[k]) is int for k in ("rows", "students", "items", "skills"))
    assert block["split"] == "80/20 by student, seed 42"
    assert list(block["metrics"]) == list(models)
    for got in block["metrics"].values():
        assert list(got) == ["AUC", "RMSE", "ACC", "ECE"]
        assert all(isinstance(v, float) and v == round(v, 4) and 0 <= v <= 1 for v in got.values())


# ---------------------------------------------------------------- the specs


def test_specs_cover_every_results_key_once():
    assert [s.key for s in harness.SPECS] == RESULT_KEYS
    assert len({s.cli for s in harness.SPECS}) == len(harness.SPECS)
    for s in harness.SPECS:
        assert harness.spec(s.key) is s and harness.spec(s.cli) is s
        assert registry.licence(s.key)  # every block can state its licence
        assert EM_DASH not in s.title + s.note
    assert [s.key for s in harness.SPECS if s.bkt] == ["assistments_2009"]  # BKT grid only where v1 ran it
    assert harness.spec("ammore").subsets == ("all", "ghana")
    with pytest.raises(KeyError, match="unknown dataset 'nope'"):
        harness.spec("nope")


def test_v1_titles_and_licences_match_the_v1_scripts():
    assert harness.spec("assistments_2009").title == kt.DATASET and registry.licence("assistments_2009") == kt.LICENCE
    assert harness.spec("eedi_2020_task34").title == eval_eedi.DATASET
    assert registry.licence("eedi_2020_task34") == eval_eedi.LICENCE
    assert harness.spec("ammore_2024").title == eval_ammore.DATASET
    assert registry.licence("ammore_2024") == eval_ammore.LICENCE == "CC BY-NC 4.0"
    assert harness.SPLIT == kt.SPLIT and harness.SEED == 42


# ---------------------------------------------------------------- splits


def test_split_is_the_v1_split(logs):
    tr, te = harness.split(logs)
    tr1, te1 = kt.split(logs)
    pd.testing.assert_frame_equal(tr, tr1)
    pd.testing.assert_frame_equal(te, te1)
    assert te.user_id.nunique() == 12 and set(tr.user_id).isdisjoint(te.user_id)


def test_kfold_by_student_partitions_the_students(logs):
    folds = list(harness.kfold_by_student(logs))
    assert len(folds) == harness.FOLDS == 5
    seen = []
    for tr, te in folds:
        train_u, test_u = set(tr.user_id), set(te.user_id)
        assert train_u.isdisjoint(test_u) and train_u | test_u == set(logs.user_id)
        assert len(test_u) == 12  # 60 students in 5 folds
        assert len(tr) + len(te) == len(logs)
        assert tr.index.is_monotonic_increasing and te.index.is_monotonic_increasing  # row order is kept
        seen += sorted(test_u)
    assert sorted(seen) == sorted(logs.user_id.unique())  # every student is tested exactly once


def test_kfold_is_deterministic_and_depends_on_the_seed(logs):
    before = logs.copy()
    a = [set(te.user_id) for _, te in harness.kfold_by_student(logs)]
    b = [set(te.user_id) for _, te in harness.kfold_by_student(logs, k=5, seed=42)]
    c = [set(te.user_id) for _, te in harness.kfold_by_student(logs, seed=7)]
    assert a == b and a != c
    pd.testing.assert_frame_equal(logs, before)


def test_kfold_handles_uneven_sizes_and_bad_k(logs):
    sizes = [te.user_id.nunique() for _, te in harness.kfold_by_student(logs, k=7)]
    assert sorted(sizes) == [8, 8, 8, 9, 9, 9, 9] and sum(sizes) == 60
    for k in (1, 0, 61):
        with pytest.raises(ValueError, match="k must be between 2"):
            list(harness.kfold_by_student(logs, k=k))


# ---------------------------------------------------------------- evaluate, blocks, cross-validation


def test_evaluate_prints_counts_and_returns_four_metrics(logs, capsys):
    scores = harness.evaluate(logs)
    lines = capsys.readouterr().out.strip().splitlines()
    tr, te = harness.split(logs)
    assert (
        lines[0]
        == f"rows={len(logs)} students=60 items={logs.item_id.nunique()} skills=5 train={len(tr)} test={len(te)}"
    )
    assert [ln.split()[0] for ln in lines[1:]] == ["global-mean", "skill-mean", "elo", "elo-multi"]
    assert list(scores) == ["AUC", "RMSE", "ACC", "ECE"]
    assert all(list(scores[m]) == MODELS for m in scores)
    assert scores["AUC"]["global-mean"] == 0.5
    assert scores["AUC"]["elo-multi (Kwan)"] > scores["AUC"]["skill-mean"] > 0.5
    # the same numbers as the v1 entry point on the same frame
    assert kt.score_models(tr, te) == scores


def test_make_block_is_schema_v2_with_counts(logs, capsys):
    scores = harness.evaluate(logs)
    block = harness.make_block(harness.spec("eedi-12"), logs, scores)
    check_v2_block(block)
    assert block["dataset"] == "Eedi NeurIPS 2020 Education Challenge, task 1/2 (full training set)"
    assert block["licence"] == "research use; cite Wang et al. (2020) arXiv:2007.12061"
    assert (block["rows"], block["students"], block["items"], block["skills"]) == (1800, 60, logs.item_id.nunique(), 5)
    for metric in ("AUC", "RMSE", "ACC", "ECE"):
        assert block["metrics"]["elo-multi (Kwan)"][metric] == round(scores[metric]["elo-multi (Kwan)"], 4)
    assert "AUC" not in block and "questions" not in block  # new datasets carry no v1 fields

    legacy = harness.make_block(harness.spec("eedi-34"), logs, scores)
    assert list(legacy) == [*V2_FIELDS[:5], "questions", *V2_FIELDS[5:], "AUC", "RMSE"]
    assert legacy["questions"] == legacy["items"]
    assert legacy["AUC"] == {m: v["AUC"] for m, v in legacy["metrics"].items()}
    assert legacy["RMSE"] == {m: v["RMSE"] for m, v in legacy["metrics"].items()}

    noted = harness.make_block(harness.spec("junyi"), logs, scores)
    assert list(noted) == [*V2_FIELDS, "note"] and "Evaluation only" in noted["note"]
    assert noted["licence"].startswith("non-commercial")


def test_dataset_block_orders_models_and_skips_missing_metrics():
    scores = {"AUC": {"elo-multi (Kwan)": 0.77123456, "bkt": 0.7, "elo": 0.69, "mine": 0.6}, "RMSE": {"elo": 0.43}}
    counts = {"rows": 10, "students": 2, "items": 3, "skills": 1}
    block = results.dataset_block("Toy", "MIT", counts, "80/20", scores, model_order=kt.MODEL_ORDER, legacy=("AUC",))
    assert list(block["metrics"]) == ["elo", "bkt", "elo-multi (Kwan)", "mine"]  # unknown models go last
    assert block["metrics"]["elo"] == {"AUC": 0.69, "RMSE": 0.43}
    assert block["metrics"]["elo-multi (Kwan)"] == {"AUC": 0.7712}
    assert block["AUC"] == {"elo": 0.69, "bkt": 0.7, "elo-multi (Kwan)": 0.7712, "mine": 0.6} and "RMSE" not in block
    assert results.skipped_block("Toy", "MIT", "download failed") == {
        "dataset": "Toy",
        "licence": "MIT",
        "skipped": "download failed",
    }
    assert results.BLOCK_SCHEMA == 2
    assert "demo" in results.dataset_block.__doc__  # the docstring says why the v1 fields are still there


def test_cross_validate_reports_mean_and_std_per_model(logs, capsys):
    cv = harness.cross_validate(logs, k=3)
    out = capsys.readouterr().out
    assert out.count("[fold ") == 3 and "[fold 3/3]" in out
    assert cv["split"] == "3-fold by student, seed 42" and cv["folds"] == 3
    assert list(cv["metrics"]) == MODELS
    per_fold = [kt.score_models(tr, te) for tr, te in harness.kfold_by_student(logs, k=3)]
    for model in MODELS:
        assert list(cv["metrics"][model]) == ["AUC", "RMSE", "ACC", "ECE"]
        for metric in ("AUC", "RMSE", "ACC", "ECE"):
            values = [f[metric][model] for f in per_fold]
            assert cv["metrics"][model][metric] == {
                "mean": round(float(np.mean(values)), 4),
                "std": round(float(np.std(values)), 4),
            }
    assert cv["metrics"]["global-mean"]["AUC"] == {"mean": 0.5, "std": 0.0}


# ---------------------------------------------------------------- synthetic raw folders


def write_raw(raw, logs):
    """Write `logs` under `raw` in the file layout of every dataset (shuffled, so loaders must sort)."""
    raw.mkdir(parents=True, exist_ok=True)
    n = len(logs)
    stamps = pd.Timestamp("2020-01-01") + pd.to_timedelta(np.arange(n), unit="min")
    shuffled = logs.sample(frac=1, random_state=1).index

    # ASSISTments 2009
    a09 = pd.DataFrame(
        {
            "order_id": np.arange(n) + 1000,
            "user_id": logs.user_id,
            "problem_id": logs.item_id,
            "skill_id": logs.skill_id,
            "correct": logs.correct,
            "original": 1,
        }
    )
    a09.loc[shuffled].to_csv(raw / "skill_builder_data_corrected.csv", index=False, encoding="latin-1")

    # Eedi task 3/4 and 1/2
    eedi = raw / "eedi" / "data"
    (eedi / "train_data").mkdir(parents=True)
    (eedi / "metadata").mkdir()
    answers = pd.DataFrame(
        {
            "QuestionId": logs.item_id,
            "UserId": logs.user_id,
            "AnswerId": np.arange(n) + 1,
            "IsCorrect": logs.correct,
            "CorrectAnswer": 1,
            "AnswerValue": 1,
        }
    )
    subjects = {0: "[3, 32]", 1: "[3, 32, 49]", 2: "[3, 32, 49, 33]"}
    qids = sorted(logs.item_id.unique())
    for task in ("3_4", "1_2"):
        answers.loc[shuffled].to_csv(eedi / "train_data" / f"train_task_{task}.csv", index=False)
        pd.DataFrame(
            {
                "AnswerId": answers.AnswerId.astype(float) if task == "1_2" else answers.AnswerId,
                "DateAnswered": stamps.strftime("%Y-%m-%d %H:%M:%S.000"),
                "Confidence": "",
                "GroupId": 1,
                "QuizId": 1,
                "SchemeOfWorkId": 52562.0,
            }
        ).to_csv(eedi / "metadata" / f"answer_metadata_task_{task}.csv", index=False)
        pd.DataFrame({"QuestionId": qids, "SubjectId": [subjects[q % 3] for q in qids]}).to_csv(
            eedi / "metadata" / f"question_metadata_task_{task}.csv", index=False
        )
    (eedi / "metadata" / "subject_metadata.csv").write_bytes(
        "﻿SubjectId,Name,ParentId,Level\r\n3,Maths,NULL,0\r\n32,Number,3,1\r\n49,BIDMAS,32,2\r\n33,Basic Arithmetic,49,3\r\n".encode()
    )

    # AMMORE: students with an even id are in Ghana
    am = raw / "ammore"
    am.mkdir()
    pd.DataFrame(
        {
            "user_id": "u" + logs.user_id.astype(str),
            "message_inserted_at": stamps.strftime("%m/%d/%y %H:%M"),
            "question": "Question " + logs.item_id.astype(str),
            "question_micro_lesson": "G6.N1.1.1." + logs.skill_id.astype(str),
            "human_label": np.where(logs.correct == 1, "correct_answer", "wrong_answer"),
        }
    ).to_csv(am / "AMMORE_student_answers.csv", index=False)
    users = sorted(logs.user_id.unique())
    pd.DataFrame(
        {"user ID": [f"u{u}" for u in users], "country": ["Ghana" if u % 2 == 0 else "Kenya" for u in users]}
    ).to_csv(am / "AMMORE_demographic_data.csv", index=False)

    # ASSISTments 2012, 2015, 2017
    for folder in ("assist2012", "assist2015", "assist2017"):
        (raw / folder).mkdir()
    pd.DataFrame(
        {
            "problem_log_id": np.arange(n) + 1,
            "problem_id": logs.item_id,
            "user_id": logs.user_id,
            "start_time": stamps.strftime("%Y-%m-%d %H:%M:%S"),
            "correct": logs.correct,
            "skill_id": logs.skill_id,
        }
    ).loc[shuffled].to_csv(raw / "assist2012" / "2012-2013-data-with-predictions-4-final.csv", index=False)
    pd.DataFrame(
        {"user_id": logs.user_id, "log_id": np.arange(n) + 1, "sequence_id": logs.skill_id, "correct": logs.correct}
    ).loc[shuffled].to_csv(raw / "assist2015" / "2015_100_skill_builders_main_problems.csv", index=False)
    pd.DataFrame(
        {
            "studentId": logs.user_id,
            "skill": "skill-" + logs.skill_id.astype(str),
            "problemId": logs.item_id,
            "startTime": 1096470301 + np.arange(n) * 60,
            "correct": logs.correct,
        }
    ).loc[shuffled].to_csv(raw / "assist2017" / "anonymized_full_release_competition_dataset.csv", index=False)

    # XES3G5M: one sequence per student
    xes = raw / "xes3g5m" / "XES3G5M" / "question_level"
    xes.mkdir(parents=True)
    rows = []
    for uid, g in logs.groupby("user_id", sort=True):
        join = lambda values: ",".join(str(v) for v in values)
        rows.append((0, uid, join(g.item_id), join(g.skill_id), join(g.correct), join(g.t * 1000)))
    cols = ["fold", "uid", "questions", "concepts", "responses", "timestamps"]
    pd.DataFrame(rows[:-5], columns=cols).to_csv(xes / "train_valid_sequences_quelevel.csv", index=False)
    pd.DataFrame(rows[-5:], columns=cols).assign(fold=-1).to_csv(xes / "test_quelevel.csv", index=False)

    # Junyi
    (raw / "junyi").mkdir()
    pd.DataFrame(
        {
            "user_id": logs.user_id,
            "exercise": "exercise_" + logs.skill_id.astype(str),
            "problem_type": "t" + logs.item_id.astype(str),
            "problem_number": 1,
            "time_done": 1403098400000000 + np.arange(n) * 1_000_000,
            "correct": np.where(logs.correct == 1, "true", "false"),
        }
    ).loc[shuffled].to_csv(raw / "junyi" / "junyi_ProblemLog_original.csv", index=False)
    return raw


@pytest.fixture(scope="module")
def small_logs():
    return synthetic_logs(n_users=40, n_items=12, n_skills=3, per_user=20, seed=6)


@pytest.fixture(scope="module")
def raw(tmp_path_factory, small_logs):
    return write_raw(tmp_path_factory.mktemp("raw"), small_logs)


# ---------------------------------------------------------------- run_dataset


@pytest.mark.parametrize("name", ["eedi-12", "assistments-2012", "assistments-2017", "xes3g5m", "junyi"])
def test_run_dataset_gives_the_same_block_for_every_layout_of_the_same_log(name, raw, small_logs, capsys):
    """The same answers in five file layouts: counts and metrics must be identical."""
    block = harness.run_dataset(name, raw_dir=raw)
    s = harness.spec(name)
    check_v2_block(block, extra=("note",) if s.note else ())
    assert (block["rows"], block["students"], block["items"], block["skills"]) == (800, 40, 12, 3)
    assert block["dataset"] == s.title and block["licence"] == registry.licence(s.key)
    reference = harness.make_block(s, small_logs, harness.evaluate(small_logs))
    assert block["metrics"] == reference["metrics"]
    out = capsys.readouterr().out
    assert out.startswith("rows=800 students=40 items=12 skills=3 train=")


def test_run_dataset_assistments_2015_has_skill_builders_as_items(raw, capsys):
    block = harness.run_dataset("assistments-2015", raw_dir=raw)
    check_v2_block(block, extra=("note",))
    assert (block["rows"], block["students"], block["items"], block["skills"]) == (800, 40, 3, 3)
    assert "no question ids" in block["note"]


def test_run_dataset_matches_the_v1_scripts(raw, tmp_path, capsys):
    """`kwan eval all` goes through the loaders; the v1 commands go through the v1 scripts. Same blocks."""
    out = tmp_path / "v1.json"
    v1 = {
        "assistments_2009": kt.main([str(raw / "skill_builder_data_corrected.csv"), "--out", str(out)]),
        "eedi_2020_task34": eval_eedi.main([str(raw / "eedi" / "data"), "--out", str(out)]),
        "ammore_2024": eval_ammore.main([str(raw / "ammore"), "--out", str(out)]),
    }
    for key, block in v1.items():
        assert harness.run_dataset(key, raw_dir=raw) == block, key
    assert json.loads(out.read_text(encoding="utf-8")) == v1
    check_v2_block(v1["assistments_2009"], extra=("AUC", "RMSE"))
    assert harness.run_dataset("assistments_2009", raw_dir=raw, bkt=True)["metrics"].keys() == {*MODELS, "bkt"}


def test_ammore_block_reports_ghana_next_to_all(raw, capsys):
    block = harness.run_dataset("ammore", raw_dir=raw)
    out = capsys.readouterr().out
    assert list(block) == ["dataset", "licence", "all", "ghana"]
    assert block["licence"] == "CC BY-NC 4.0"
    for subset in ("all", "ghana"):
        sub = block[subset]
        assert list(sub) == ["rows", "students", "items", "questions", "skills", "split", "metrics", "AUC", "RMSE"]
        assert sub["AUC"]["elo-multi (Kwan)"] == sub["metrics"]["elo-multi (Kwan)"]["AUC"]
        assert f"[{subset}] rows={sub['rows']} students={sub['students']}" in out
    assert (block["all"]["rows"], block["all"]["students"]) == (800, 40)
    assert (block["ghana"]["rows"], block["ghana"]["students"]) == (400, 20)  # the even-numbered students


def test_run_dataset_writes_through_the_single_writer_and_keeps_other_blocks(raw, tmp_path, capsys):
    out = tmp_path / "results" / "eval_results.json"
    out.parent.mkdir()
    others = {"simulation": {"setup": "kept"}, "eedi_2020_task12": {"stale": True}, "latency": {"calls": 3}}
    out.write_text(json.dumps(others), encoding="utf-8")
    block = harness.run_dataset("eedi-12", out=out, raw_dir=raw, folds=2)
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert list(saved) == ["simulation", "eedi_2020_task12", "latency"]
    assert saved["eedi_2020_task12"] == block and saved["simulation"] == others["simulation"]
    assert list(block) == [*V2_FIELDS, "cv"] and block["cv"]["folds"] == 2
    assert [p.name for p in out.parent.iterdir()] == ["eval_results.json"]  # no temp or lock file left behind
    assert "wrote eedi_2020_task12" in capsys.readouterr().err


def test_run_dataset_says_which_files_are_missing(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        harness.run_dataset("junyi", raw_dir=tmp_path)
    assert "junyi_ProblemLog_original.csv" in str(exc.value) and "make data" in str(exc.value)
    assert [p.name for p in harness.missing_files("eedi-12", tmp_path)] == [
        "train_task_1_2.csv",
        "answer_metadata_task_1_2.csv",
        "question_metadata_task_1_2.csv",
        "subject_metadata.csv",
    ]
    assert len(harness.missing_files("xes3g5m", tmp_path)) == 2
    assert harness.default_path("eedi-12", "raw") == Path("raw/eedi/data")


# ---------------------------------------------------------------- run_all and skipped blocks


def test_run_all_evaluates_what_is_present_and_skips_the_rest(raw, tmp_path, capsys):
    partial = tmp_path / "raw"
    partial.mkdir()
    (partial / "ammore").symlink_to(raw / "ammore")
    (partial / "assist2015").symlink_to(raw / "assist2015")
    out = tmp_path / "eval_results.json"
    kept = {"dataset": "old numbers", "metrics": {"elo": {"AUC": 0.7}}}
    out.write_text(json.dumps({"junyi_2015": kept, "xes3g5m": {"dataset": "x", "skipped": "old reason"}}), "utf-8")

    blocks = harness.run_all(partial, out)
    err = capsys.readouterr().err
    assert list(blocks) == RESULT_KEYS
    done = [k for k, b in blocks.items() if "skipped" not in b]
    assert done == ["ammore_2024", "assistments_2015"]
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["ammore_2024"] == blocks["ammore_2024"] and saved["assistments_2015"] == blocks["assistments_2015"]
    assert saved["junyi_2015"] == kept  # a missing download never erases numbers that are already there
    for key in ("eedi_2020_task12", "xes3g5m", "assistments_2009"):
        assert list(saved[key]) == ["dataset", "licence", "skipped"]  # no numbers, a clear reason
        assert "raw data is missing" in saved[key]["skipped"] and "make data" in saved[key]["skipped"]
        assert saved[key]["dataset"] == harness.spec(key).title
    assert err.count("skipping ") == 7 and "skipping junyi: raw data is missing" in err


def test_run_all_only_and_bkt_default(raw, capsys):
    blocks = harness.run_all(raw, only=["assistments-2009", "assistments_2015"])
    assert list(blocks) == ["assistments_2009", "assistments_2015"]
    assert "bkt" in blocks["assistments_2009"]["metrics"]  # the BKT grid runs where v1 ran it
    assert "bkt" not in blocks["assistments_2015"]["metrics"]
    assert "bkt" not in harness.run_all(raw, only=["assistments-2009"], bkt=False)["assistments_2009"]["metrics"]


# ---------------------------------------------------------------- the commands


def test_every_new_dataset_command_has_help(capsys):
    for s in harness.SPECS:
        with pytest.raises(SystemExit) as exc:
            cli.main(["eval", s.cli, "--help"])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert out.startswith(f"usage: kwan eval {s.cli}") and "--out" in out and EM_DASH not in out
    with pytest.raises(SystemExit):
        cli.main(["eval", "all", "--help"])
    out = capsys.readouterr().out
    assert "--raw" in out and "--only" in out and EM_DASH not in out
    with pytest.raises(SystemExit):
        cli.main(["eval", "--help"])
    listed = capsys.readouterr().out
    assert all(s.cli in listed for s in harness.SPECS) and " all " in listed


def test_eval_command_runs_a_new_dataset_and_writes_its_block(raw, tmp_path, capsys):
    out = tmp_path / "r.json"
    assert cli.main(["eval", "assistments-2017", str(raw / "assist2017"), "--out", str(out)]) == 0
    saved = json.loads(out.read_text(encoding="utf-8"))
    check_v2_block(saved["assistments_2017"])
    assert capsys.readouterr().out.startswith("rows=800 students=40 items=12 skills=3")
    assert cli.main(["eval", "junyi", str(tmp_path / "nowhere")]) == 1
    assert "kwan eval junyi: missing" in capsys.readouterr().err


def test_eval_all_command(raw, tmp_path, capsys):
    out = tmp_path / "r.json"
    assert cli.main(["eval", "all", "--raw", str(raw), "--no-bkt", "--out", str(out)]) == 0
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert list(saved) == RESULT_KEYS and not any("skipped" in b for b in saved.values())
    assert "bkt" not in saved["assistments_2009"]["metrics"]
    assert "evaluated 9 dataset(s)" in capsys.readouterr().err

    assert cli.main(["eval", "all", "--raw", str(tmp_path / "empty"), "--only", "ammore,junyi"]) == 0
    err = capsys.readouterr().err
    assert "evaluated 0 dataset(s): none" in err and "skipped 2: ammore_2024, junyi_2015" in err
    assert cli.main(["eval", "all", "--only", "nope"]) == 2
    assert "unknown dataset 'nope'" in capsys.readouterr().err


def test_v1_commands_default_to_the_raw_folder(monkeypatch):
    calls = []
    monkeypatch.setattr(kt, "main", lambda argv=None: calls.append(argv))
    monkeypatch.setattr(eval_eedi, "main", lambda argv=None: calls.append(argv))
    monkeypatch.setattr(eval_ammore, "main", lambda argv=None: calls.append(argv))
    assert cli.main(["eval", "assistments-2009", "--bkt"]) == 0
    assert cli.main(["eval", "eedi-34"]) == 0
    assert cli.main(["eval", "ammore", "--out", "r.json"]) == 0
    assert calls == [
        ["raw/skill_builder_data_corrected.csv", "--bkt"],
        ["raw/eedi/data"],
        ["raw/ammore", "--out", "r.json"],
    ]


# ---------------------------------------------------------------- the results writer


def test_results_update_is_safe_when_two_writers_finish_together(tmp_path):
    """Each writer owns one key; none may lose another's block."""
    import threading

    path = tmp_path / "eval_results.json"
    keys = [f"block_{i}" for i in range(12)]
    threads = [threading.Thread(target=results.update, args=(k, {"n": i}, path)) for i, k in enumerate(keys)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert sorted(saved) == sorted(keys) and all(saved[k] == {"n": i} for i, k in enumerate(keys))
    assert [p.name for p in tmp_path.iterdir()] == ["eval_results.json"]


# ---------------------------------------------------------------- copy rules and the weekly workflow


def test_no_em_dash_in_the_harness_and_loaders():
    files = [
        *sorted((ROOT / "src" / "kwan" / "data").rglob("*.py")),
        *(ROOT / "src" / "kwan" / "eval" / f"{n}.py" for n in ("harness", "metrics", "results", "_cli", "ammore")),
        ROOT / "scripts" / "download_data.sh",
        ROOT / ".github" / "workflows" / "slow-eval.yml",
    ]
    for path in files:
        assert EM_DASH not in path.read_text(encoding="utf-8"), path


def test_weekly_workflow_runs_every_evaluation_and_opens_a_pull_request():
    import yaml

    text = (ROOT / ".github" / "workflows" / "slow-eval.yml").read_text(encoding="utf-8")
    wf = yaml.safe_load(text)
    triggers = wf.get("on", wf.get(True))  # YAML reads the bare key `on` as the boolean true
    assert "schedule" in triggers and "workflow_dispatch" in triggers
    cron = triggers["schedule"][0]["cron"].split()
    assert len(cron) == 5 and cron[2:4] == ["*", "*"] and cron[4] != "*"  # once a week
    steps = [step for job in wf["jobs"].values() for step in job["steps"]]
    runs = "\n".join(step.get("run", "") for step in steps)
    uses = [step.get("uses", "") for step in steps]
    assert any(u.startswith("actions/cache@") for u in uses)  # datasets are cached between weeks
    assert "scripts/download_data.sh learner" in runs
    assert "kwan eval all" in runs and "--out data/eval_results.json" in runs
    assert "KWAN_RUN_SLOW" in text and "pytest -q -m slow" in runs
    assert "gh pr create" in runs or any(u.startswith("peter-evans/create-pull-request@") for u in uses)
    assert "data/eval_results.json" in text
    perms = {**wf.get("permissions", {}), **next(iter(wf["jobs"].values())).get("permissions", {})}
    assert perms.get("contents") == "write" and perms.get("pull-requests") == "write"
    assert "git add raw" not in runs and "git add -A" not in runs and "git add ." not in runs  # never a raw row
