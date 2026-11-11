"""Deep knowledge tracing benchmark (kwan.eval.deep): folds, pyKT sequence conversion, rule D7, results block.

Everything here runs on tiny synthetic data. torch and pyKT are optional (extra `kt`): the tests that
need them skip cleanly when they are not installed.
"""

import importlib
import json
import sys

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from kwan.eval import deep  # noqa: E402

ELO = deep.ELO


def synthetic(n_students=40, seed=0, min_len=1, max_len=12, items=30, skills=5):
    """Random answers in Kwan's interaction form, rows of each student in answer order."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_students):
        for _ in range(int(rng.integers(min_len, max_len + 1))):
            item = int(rng.integers(0, items))
            rows.append((1000 + u, 500 + item, 70 + item % skills, int(rng.random() < 0.6)))
    return pd.DataFrame(rows, columns=["user_id", "problem_id", "skill_id", "correct"])


# ---------------------------------------------------------------------------------------------------
# Import hygiene
# ---------------------------------------------------------------------------------------------------


def test_module_imports_without_torch_or_pykt():
    """Importing the module must not pull in torch or pyKT (the shared environment has neither)."""
    for name in ("torch", "pykt"):
        assert name not in vars(deep), f"{name} is imported at module level"
    source = importlib.util.find_spec("kwan.eval.deep").loader.get_source("kwan.eval.deep")
    top_level = [line for line in source.splitlines() if line.startswith(("import ", "from "))]
    for line in top_level:
        assert not any(word in line for word in ("torch", "pykt", "numpy", "pandas", "sklearn")), line


def test_no_em_dashes_in_user_facing_text():
    source = importlib.util.find_spec("kwan.eval.deep").loader.get_source("kwan.eval.deep")
    assert chr(0x2014) not in source and chr(0x2013) not in source  # em dash, en dash


# ---------------------------------------------------------------------------------------------------
# Fold maker
# ---------------------------------------------------------------------------------------------------


def test_folds_are_five_and_balanced():
    users = list(range(103))
    folds = deep.make_folds(users)
    assert set(folds) == set(users)
    assert set(folds.values()) == {0, 1, 2, 3, 4}
    sizes = [sum(1 for f in folds.values() if f == k) for k in range(5)]
    assert sum(sizes) == 103 and max(sizes) - min(sizes) <= 1


def test_no_student_is_in_two_folds():
    df = synthetic(60)
    folds = deep.make_folds(df.user_id)  # one entry per student, however many rows the student has
    assert len(folds) == df.user_id.nunique()
    row_fold = df.user_id.map(folds)
    assert (df.assign(fold=row_fold).groupby("user_id").fold.nunique() == 1).all()
    for k in range(5):
        held_out = set(df.user_id[row_fold == k])
        training = set(df.user_id[row_fold != k])
        assert held_out and not held_out & training


def test_folds_are_deterministic_and_ignore_row_order():
    users = [f"u{i}" for i in range(50)]
    first = deep.make_folds(users, seed=42)
    assert first == deep.make_folds(users, seed=42)
    assert first == deep.make_folds(list(reversed(users)) + users[:10], seed=42)
    assert first != deep.make_folds(users, seed=43)
    assert deep.make_folds(np.array([3, 1, 2, 5, 4, 3]), k=5) == deep.make_folds([1, 2, 3, 4, 5], k=5)


def test_folds_reject_impossible_requests():
    with pytest.raises(ValueError):
        deep.make_folds([1, 2, 3], k=5)
    with pytest.raises(ValueError):
        deep.make_folds([1, 2, 3], k=1)


def test_validation_students_come_from_training_only_and_are_seeded():
    train = list(range(100))
    valid = deep.validation_users(train, 0.1, seed=7)
    assert len(valid) == 10 and valid <= set(train)
    assert valid == deep.validation_users(list(reversed(train)), 0.1, seed=7)
    assert valid != deep.validation_users(train, 0.1, seed=8)


# ---------------------------------------------------------------------------------------------------
# Sequence conversion (pyKT format)
# ---------------------------------------------------------------------------------------------------


def test_sequences_keep_each_students_answer_order():
    df = pd.DataFrame(
        {
            "user_id": ["a", "b", "a", "a", "b", "b", "a"],  # interleaved on purpose
            "problem_id": [10, 20, 30, 10, 40, 20, 50],
            "skill_id": [7, 8, 9, 7, 8, 8, 9],
            "correct": [1, 0, 0, 1, 1, 0, 1],
        }
    )
    seqs = deep.to_pykt_sequences(df, maxlen=6)
    assert list(seqs["uid"]) == ["a", "b"]
    assert seqs["num_q"] == 5 and seqs["num_c"] == 3  # ids re-indexed from 0, sorted
    a, b = seqs["rows"]
    assert a.tolist() == [0, 2, 3, 6, -1, -1] and b.tolist() == [1, 4, 5, -1, -1, -1]
    assert seqs["questions"][0].tolist() == [0, 2, 0, 4, -1, -1]  # 10, 30, 10, 50
    assert seqs["concepts"][0].tolist() == [0, 2, 0, 2, -1, -1]  # 7, 9, 7, 9
    assert seqs["responses"][0].tolist() == [1, 0, 1, 1, -1, -1]
    assert seqs["responses"][1].tolist() == [0, 1, 0, -1, -1, -1]
    assert seqs["selectmasks"][0].tolist() == [1, 1, 1, 1, -1, -1]


def test_sequences_are_cut_padded_and_short_tails_dropped():
    lengths = {"long": 11, "exact": 8, "short": 2, "tail2": 6, "three": 3}  # maxlen 4
    df = pd.DataFrame(
        [(u, i, i % 3, i % 2) for u, n in lengths.items() for i in range(n)],
        columns=["user_id", "problem_id", "skill_id", "correct"],
    )
    seqs = deep.to_pykt_sequences(df, maxlen=4, min_seq_len=3)
    per_user = {u: int((seqs["uid"] == u).sum()) for u in lengths}
    # long: 4 + 4 + tail of 3 (kept, padded); exact: 4 + 4; short: dropped; tail2: 4, tail of 2 dropped; three: kept
    assert per_user == {"long": 3, "exact": 2, "short": 0, "tail2": 1, "three": 1}
    assert seqs["dropped"] == 2 + 2
    assert seqs["questions"].shape == (7, 4) == seqs["rows"].shape
    last_long = np.where(seqs["uid"] == "long")[0][-1]
    assert seqs["questions"][last_long].tolist() == [8, 9, 10, -1]
    assert seqs["selectmasks"][last_long].tolist() == [1, 1, 1, -1]
    # every kept answer appears exactly once
    kept = seqs["rows"][seqs["rows"] != -1]
    assert len(kept) == len(set(kept.tolist())) == len(df) - seqs["dropped"]
    # padding is in the same places in every array
    for key in ("questions", "concepts", "responses", "selectmasks"):
        assert ((seqs[key] == -1) == (seqs["rows"] == -1)).all(), key


def test_sequences_match_the_source_rows():
    df = synthetic(30, seed=3, max_len=25)
    seqs = deep.to_pykt_sequences(df, maxlen=10)
    real = seqs["rows"] != -1
    rows = seqs["rows"][real]
    assert (df.correct.to_numpy()[rows] == seqs["responses"][real]).all()
    assert (np.repeat(seqs["uid"], real.sum(1)) == df.user_id.to_numpy()[rows]).all()
    for u in df.user_id.unique():  # order within a student is the frame order
        mine = np.concatenate([r[r != -1] for r in seqs["rows"][seqs["uid"] == u]]) if (seqs["uid"] == u).any() else []
        assert list(mine) == sorted(mine)


def test_conversion_rejects_answers_that_are_not_binary():
    df = synthetic(5)
    df.loc[0, "correct"] = 2
    with pytest.raises(ValueError):
        deep.to_pykt_sequences(df)


def test_pykt_frame_has_the_columns_and_text_form_pykt_reads():
    df = synthetic(12, seed=1, min_len=3, max_len=9)
    seqs = deep.to_pykt_sequences(df, maxlen=5)
    folds = deep.make_folds(df.user_id, k=3)
    frame = deep.to_pykt_frame(seqs, folds)
    assert list(frame.columns) == ["fold", "uid", "questions", "concepts", "responses", "selectmasks"]
    assert len(frame) == len(seqs["uid"])
    assert (frame.groupby("uid").fold.nunique() == 1).all()
    for col in ("questions", "concepts", "responses", "selectmasks"):
        parsed = np.array([[int(v) for v in cell.split(",")] for cell in frame[col]])
        assert parsed.shape == (len(frame), 5) and (parsed == seqs[col]).all()


def test_scored_rows_skip_the_first_answer_of_every_sequence():
    df = synthetic(20, seed=5, min_len=3, max_len=14)
    seqs = deep.to_pykt_sequences(df, maxlen=6)
    idx = np.arange(len(seqs["uid"]))
    rows = deep.scored_rows(seqs, idx)
    firsts = set(seqs["rows"][:, 0].tolist())
    assert not firsts & set(rows.tolist())
    assert len(rows) == int((seqs["rows"] != -1).sum()) - len(idx)
    assert rows.tolist() == [r for seq in seqs["rows"] for r in seq[1:] if r != -1]  # row-major order


def test_xes3g5m_loader_joins_pieces_and_drops_padding(tmp_path):
    level = tmp_path / "XES3G5M" / "question_level"
    level.mkdir(parents=True)
    cols = ["fold", "uid", "questions", "concepts", "responses", "timestamps", "selectmasks"]
    pd.DataFrame(
        [
            [0, 7, "5,6,7", "1,2_9,3", "1,0,1", "10,20,30", "1,1,1"],
            [0, 7, "8,9,-1", "4,4,-1", "0,1,-1", "40,50,-1", "1,1,-1"],  # second piece of the same student, padded
            [3, 2, "5,5,-1", "1,1,-1", "1,1,-1", "7,7,-1", "1,1,-1"],
        ],
        columns=cols,
    ).to_csv(level / "train_valid_sequences_quelevel.csv", index=False)
    pd.DataFrame([[-1, 9, "6,5,8,9", "2_9,1,4,4", "0,0,1,1", "3,1,2,4"]], columns=cols[:-1]).to_csv(
        level / "test_quelevel.csv", index=False
    )
    df = deep.load_xes3g5m(tmp_path)
    assert list(df.columns[:4]) == ["user_id", "problem_id", "skill_id", "correct"] and len(df) == 11
    seven = df[df.user_id == 7]
    assert seven.problem_id.tolist() == [5, 6, 7, 8, 9] and seven.skill_id.tolist() == [1, 2, 3, 4, 4]
    assert seven.correct.tolist() == [1, 0, 1, 0, 1]
    assert df[df.user_id == 9].problem_id.tolist() == [5, 8, 6, 9]  # ordered by timestamp
    assert df[df.user_id == 2].problem_id.tolist() == [5, 5] and set(df.correct) == {0, 1}
    with pytest.raises(FileNotFoundError):
        deep.load_xes3g5m(tmp_path / "XES3G5M" / "question_level")


def test_load_dataset_says_when_the_data_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        deep.load_dataset("ammore", tmp_path)


# ---------------------------------------------------------------------------------------------------
# Metrics and elo-multi on a fold
# ---------------------------------------------------------------------------------------------------


def test_score_on_hand_made_predictions():
    pytest.importorskip("sklearn")
    s = deep.score([1, 0, 1, 0], [0.9, 0.2, 0.6, 0.4])
    assert s["AUC"] == 1.0 and s["ACC"] == 1.0
    assert s["RMSE"] == pytest.approx(((0.01 + 0.04 + 0.16 + 0.16) / 4) ** 0.5)
    assert deep.score([1, 0], [0.4, 0.6])["ACC"] == 0.0


def test_summarise_gives_mean_and_sample_sd():
    recs = [
        {"AUC": 0.70, "RMSE": 0.40, "ACC": 0.60},
        {"AUC": 0.74, "RMSE": 0.44, "ACC": 0.70},
        {"AUC": 0.72, "RMSE": 0.42, "ACC": 0.65},
    ]
    s = deep.summarise(recs)
    assert s == {
        "AUC_mean": 0.72,
        "AUC_sd": 0.02,
        "RMSE_mean": 0.42,
        "RMSE_sd": 0.02,
        "ACC_mean": 0.65,
        "ACC_sd": 0.05,
        "folds": 3,
    }
    assert deep.summarise(recs[:1])["AUC_sd"] == 0.0


def test_elo_multi_fold_predicts_only_held_out_rows():
    pytest.importorskip("sklearn")  # kwan.eval.kt imports it
    df = synthetic(40, seed=2, min_len=4)
    folds = deep.make_folds(df.user_id)
    mask = (df.user_id.map(folds) == 0).to_numpy()
    preds, ms = deep.elo_multi_fold(df, mask)
    assert len(preds) == mask.sum() and ((preds > 0) & (preds < 1)).all() and ms > 0
    again, _ = deep.elo_multi_fold(df, mask)
    assert (preds == again).all()


def test_elo_step_is_timed_in_cpu_time():
    pytest.importorskip("sklearn")
    df = synthetic(40, seed=2, min_len=4)
    mask = (df.user_id.map(deep.make_folds(df.user_id)) == 0).to_numpy()
    timing = deep.measure_elo_latency(df, mask, max_rows=30, repeats=3)
    assert timing["steps"] == min(30, int(mask.sum())) and timing["repeats"] == 3
    assert 0 <= timing["ms_min"] <= timing["ms_median"] <= timing["ms_p95"] and timing["ms_wall_median"] > 0


def test_elo_timing_taken_with_the_block_replaces_the_fold_run_figure():
    recs = fold_records("elo", [0.75] * 5)
    assert deep.model_summary(recs)["ms_per_update_cpu"] == 0.002
    timed = deep.model_summary(recs, {"ms_median": 0.00412, "ms_wall_median": 0.05})
    assert timed["ms_per_update_cpu"] == 0.0041 and timed["ms_per_update_wall"] == 0.05


# ---------------------------------------------------------------------------------------------------
# Rule D7
# ---------------------------------------------------------------------------------------------------


def d7_input(aucs, ms):
    """{dataset: {"models": ...}} from {dataset: {model: AUC}} and {model: ms}."""
    return {
        name: {"models": {m: {"AUC_mean": auc, "ms_per_update_cpu": ms.get(m, 0.001)} for m, auc in models.items()}}
        for name, models in aucs.items()
    }


def test_d7_adopts_when_two_datasets_clear_the_margin_and_latency_is_low():
    data = d7_input(
        {
            "ammore": {ELO: 0.77, "akt": 0.81, "dkt": 0.78},
            "eedi_34": {ELO: 0.78, "akt": 0.81, "dkt": 0.77},
            "assistments_2009": {ELO: 0.75, "akt": 0.76, "dkt": 0.74},
        },
        {"akt": 12.0, "dkt": 1.5},
    )
    d = deep.d7_decision(data)
    assert d["adopt"] is True and d["model"] == "akt"
    assert d["per_model"]["akt"]["margins"] == {"assistments_2009": 0.01, "eedi_34": 0.03, "ammore": 0.04}
    assert d["per_model"]["akt"]["datasets_won"] == ["eedi_34", "ammore"]
    assert d["per_model"]["akt"]["meets_rule"] and not d["per_model"]["dkt"]["meets_rule"]
    assert d["thresholds"] == {"min_auc_margin": 0.03, "min_datasets": 2, "max_ms_per_update_cpu": 20.0}


def test_d7_does_not_adopt_when_only_one_dataset_clears_the_margin():
    data = d7_input(
        {
            "ammore": {ELO: 0.77, "akt": 0.82},
            "eedi_34": {ELO: 0.78, "akt": 0.8099},  # 0.0299: just short
            "assistments_2009": {ELO: 0.75, "akt": 0.76},
        },
        {"akt": 5.0},
    )
    d = deep.d7_decision(data)
    assert d["adopt"] is False and d["model"] is None
    assert d["per_model"]["akt"]["datasets_won"] == ["ammore"]
    assert d["per_model"]["akt"]["accuracy_ok"] is False and d["per_model"]["akt"]["latency_ok"] is True
    assert "No deep model meets rule D7" in d["reason"]


def test_d7_does_not_adopt_a_model_that_is_too_slow():
    aucs = {
        "ammore": {ELO: 0.70, "akt": 0.80},
        "eedi_34": {ELO: 0.70, "akt": 0.80},
        "assistments_2009": {ELO: 0.70, "akt": 0.80},
    }
    slow = deep.d7_decision(d7_input(aucs, {"akt": 20.0}))  # "under 20 ms": 20.0 itself fails
    assert slow["adopt"] is False
    assert slow["per_model"]["akt"]["accuracy_ok"] is True and slow["per_model"]["akt"]["latency_ok"] is False
    fast = deep.d7_decision(d7_input(aucs, {"akt": 19.99}))
    assert fast["adopt"] is True and fast["model"] == "akt"


def test_d7_uses_the_slowest_dataset_and_never_counts_a_missing_number():
    data = d7_input({"ammore": {ELO: 0.70, "akt": 0.80}, "eedi_34": {ELO: 0.70, "akt": 0.80}}, {"akt": 5.0})
    data["eedi_34"]["models"]["akt"]["ms_per_update_cpu"] = 31.0
    d = deep.d7_decision(data)
    assert d["per_model"]["akt"]["ms_per_update_cpu"] == 31.0 and d["adopt"] is False
    data["eedi_34"]["models"]["akt"]["ms_per_update_cpu"] = None  # not measured
    d = deep.d7_decision(data)
    assert d["per_model"]["akt"]["latency_ok"] is False and d["adopt"] is False
    assert d["datasets_evaluated"] == ["eedi_34", "ammore"]


def test_d7_ignores_datasets_outside_the_rule_and_picks_the_larger_mean_margin():
    data = d7_input(
        {
            "xes3g5m": {ELO: 0.60, "dkt": 0.90},  # not one of the three D7 datasets
            "ammore": {ELO: 0.70, "dkt": 0.71, "sakt": 0.74, "akt": 0.75},
            "eedi_34": {ELO: 0.70, "dkt": 0.71, "sakt": 0.74, "akt": 0.76},
            "assistments_2009": {ELO: 0.70, "dkt": 0.71, "sakt": 0.70, "akt": 0.69},
        },
        {"dkt": 1.0, "sakt": 2.0, "akt": 9.0},
    )
    d = deep.d7_decision(data)
    assert "xes3g5m" not in d["per_model"]["dkt"]["margins"] and d["per_model"]["dkt"]["meets_rule"] is False
    assert d["per_model"]["sakt"]["meets_rule"] and d["per_model"]["akt"]["meets_rule"]
    assert d["adopt"] is True and d["model"] == "akt"  # mean margin 0.0333 against 0.0267


def test_d7_with_nothing_evaluated():
    d = deep.d7_decision({})
    assert d["adopt"] is False and d["model"] is None and d["per_model"] == {}


def test_d7_says_when_a_dataset_that_was_not_run_could_change_the_outcome():
    three = {
        "ammore": {ELO: 0.77, "dkt": 0.70, "akt": 0.81},
        "eedi_34": {ELO: 0.78, "dkt": 0.77},  # akt was not run here
        "assistments_2009": {ELO: 0.75, "dkt": 0.75, "akt": 0.76},
    }
    d = deep.d7_decision(d7_input(three, {"dkt": 1.0, "akt": 5.0}))
    akt = d["per_model"]["akt"]
    assert akt["datasets_not_run"] == ["eedi_34"] and akt["datasets_won"] == ["ammore"]
    assert akt["meets_rule"] is False and akt["conclusive"] is False  # one win plus one unknown could reach two
    assert d["per_model"]["dkt"]["conclusive"] is True and d["per_model"]["dkt"]["datasets_not_run"] == []
    assert d["adopt"] is False and d["conclusive"] is False and "Not conclusive for akt" in d["reason"]
    # no win on the two datasets that were run: the third cannot bring the count to two
    three["ammore"]["akt"] = 0.78
    d = deep.d7_decision(d7_input(three, {"dkt": 1.0, "akt": 5.0}))
    assert d["per_model"]["akt"]["conclusive"] is True and d["conclusive"] is True
    # too slow is final whatever the missing dataset would show
    three["ammore"]["akt"] = 0.81
    d = deep.d7_decision(d7_input(three, {"dkt": 1.0, "akt": 45.0}))
    assert d["per_model"]["akt"]["conclusive"] is True and d["adopt"] is False


# ---------------------------------------------------------------------------------------------------
# Results block
# ---------------------------------------------------------------------------------------------------


def fold_records(kind, aucs, epochs=4):
    recs = []
    for k, auc in enumerate(aucs):
        rec = {"kind": kind, "fold": k, "AUC": auc, "RMSE": 0.44, "ACC": 0.7, "n_scored": 100 + k}
        if kind == "elo":
            rec.update({"config": {"model": ELO, "fold": k}, "ms_per_update_cpu": 0.002})
            rec.update({"AUC_all": auc - 0.001, "RMSE_all": 0.44, "n_all": 120})
        else:
            rec.update({"config": {"epochs": epochs, "device": "cpu", "trainer": deep.trainer_for("cpu")},
                        "epochs_run": epochs, "best_epoch": epochs - 1, "train_seconds": 60.0, "valid_AUC": auc})  # fmt: skip
        recs.append(rec)
    return recs


def example_block():
    meta = {
        "dataset": "x",
        "licence": "y",
        "rows": 10,
        "students": 5,
        "items": 4,
        "skills": 2,
        "sequences": 5,
        "answers_dropped_by_min_seq_len": 0,
    }
    datasets = {
        name: {
            "meta": dict(meta),
            "folds": {
                ELO: fold_records("elo", [0.75, 0.76, 0.74, 0.75, 0.75]),
                "dkt": fold_records("deep", [0.70, 0.71, 0.72, 0.70, 0.72]),
                "akt": fold_records("deep", [0.76, 0.77, 0.75, 0.76, 0.76], epochs=2),
            },  # fmt: skip
            "latency": {"dkt": {"ms_median": 1.23456, "ms_p95": 2.0}, "akt": {"ms_median": 25.0, "ms_p95": 30.0}},
        }
        for name in ("assistments_2009", "eedi_34", "ammore")
    }
    return deep.make_block(
        datasets, deep.protocol_block(5, 42, 200), {"torch": "0"}, deep.LATENCY_PROTOCOL, {"xes3g5m": "not downloaded"}
    )


def test_block_structure():
    block = example_block()
    assert list(block) == ["task", "implementation", "protocol", "budget", "hyperparameters", "latency_protocol",
                           "versions", "datasets", "skipped", "decision"]  # fmt: skip
    assert (
        block["protocol"]["folds"] == 5 and block["protocol"]["max_seq_len"] == 200 and block["protocol"]["seed"] == 42
    )
    assert set(block["datasets"]) == {"assistments_2009", "eedi_34", "ammore"}
    for entry in block["datasets"].values():
        assert {"dataset", "licence", "rows", "students", "items", "skills", "models"} <= set(entry)
        assert list(entry["models"]) == [ELO, "dkt", "akt"]
        for summary in entry["models"].values():
            assert {"AUC_mean", "AUC_sd", "RMSE_mean", "ACC_mean", "ms_per_update_cpu", "folds", "AUC_folds"} <= set(
                summary
            )
            assert summary["folds"] == 5 and len(summary["AUC_folds"]) == 5
        dkt = entry["models"]["dkt"]
        assert dkt["AUC_mean"] == 0.71 and dkt["AUC_sd"] == 0.01 and dkt["ms_per_update_cpu"] == 1.235
        assert dkt["epochs_max"] == 4 and dkt["train_minutes"] == 5.0 and dkt["device"] == "cpu"
        assert entry["models"][ELO]["ms_per_update_cpu"] == 0.002
        assert entry["models"][ELO]["scored_answers"] == sum(100 + k for k in range(5))
    assert block["budget"]["epochs_max_per_fit"] == [2, 4] and block["budget"]["train_minutes_total"] == 30.0
    assert set(block["skipped"]) == {"xes3g5m", "onnx_backend"} and block["skipped"]["xes3g5m"] == "not downloaded"
    assert set(block["hyperparameters"]) == set(deep.MODELS)
    json.dumps(block)  # plain JSON types only


def test_block_decision_is_computed_from_the_block_numbers():
    block = example_block()
    d = block["decision"]
    assert d["adopt"] is False and d["model"] is None
    assert d["per_model"]["akt"]["margins"] == {"assistments_2009": 0.01, "eedi_34": 0.01, "ammore": 0.01}
    assert d["per_model"]["akt"]["latency_ok"] is False and d["per_model"]["dkt"]["latency_ok"] is True
    assert d == deep.d7_decision(block["datasets"])


def test_block_never_fills_in_a_model_that_was_not_run(tmp_path):
    records, skipped = deep.collect("ammore", ("dkt",), 5, tmp_path)
    assert records == {} and set(skipped) == {f"ammore/{ELO}", "ammore/dkt"}
    for k in range(4):  # one fold short
        deep._save_record(deep._record_path(tmp_path, "ammore", "dkt", k), fold_records("deep", [0.7] * 5)[k])
    records, skipped = deep.collect("ammore", ("dkt",), 5, tmp_path)
    assert "dkt" not in records and "fold(s) [4]" in skipped["ammore/dkt"]
    # complete models are collected; folds made with different budgets are refused, not averaged
    for k in range(5):
        deep._save_record(deep._record_path(tmp_path, "ammore", ELO, k), fold_records("elo", [0.75] * 5)[k])
        deep._save_record(
            deep._record_path(tmp_path, "ammore", "sakt", k), fold_records("deep", [0.7] * 5, epochs=k + 1)[k]
        )
    deep._save_record(deep._record_path(tmp_path, "ammore", "dkt", 4), fold_records("deep", [0.7] * 5)[4])
    records, skipped = deep.collect("ammore", ("dkt", "sakt"), 5, tmp_path)
    assert set(records) == {ELO, "dkt"} and [rec["fold"] for rec in records["dkt"]] == [0, 1, 2, 3, 4]
    assert skipped == {"ammore/sakt": "folds were run with different configurations"}
    assert not list(tmp_path.rglob("*.tmp"))


def test_parse_epochs():
    assert deep.parse_epochs("6") == {m: 6 for m in deep.MODELS}
    assert deep.parse_epochs("dkt=30,sakt=20,simplekt=8,akt=4") == {"dkt": 30, "sakt": 20, "simplekt": 8, "akt": 4}
    with pytest.raises(ValueError):
        deep.parse_epochs("dkt=3")


def test_register_adds_kt_deep_with_the_documented_flags():
    import argparse

    parser = argparse.ArgumentParser(prog="kwan")
    ev = parser.add_subparsers(dest="command").add_parser("eval").add_subparsers(dest="dataset")
    deep.register(ev)
    args = parser.parse_args(
        "eval kt-deep --models dkt,sakt --datasets ammore --epochs 3 --folds 5 --device cpu --out x.json".split()
    )
    assert args.models == "dkt,sakt" and args.datasets == "ammore" and args.epochs == "3" and args.folds == 5
    assert args.device == "cpu" and args.out == "x.json" and args.extra == "kt" and callable(args.func)


# ---------------------------------------------------------------------------------------------------
# End to end on synthetic data (needs torch and pyKT)
# ---------------------------------------------------------------------------------------------------


def test_pykt_models_train_and_line_up_with_scored_rows(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("pykt")
    df = synthetic(60, seed=4, min_len=4, max_len=30)
    seqs = deep.to_pykt_sequences(df, maxlen=20)
    folds = deep.make_folds(df.user_id)
    seq_fold = np.array([folds[u] for u in seqs["uid"]])
    test_idx, valid_idx, train_idx = np.where(seq_fold == 0)[0], np.where(seq_fold == 1)[0], np.where(seq_fold > 1)[0]
    rows = deep.scored_rows(seqs, test_idx)
    for model in deep.MODELS:
        preds, targets, info = deep.fit_and_predict(
            model, seqs, train_idx, valid_idx, test_idx, 1, "cpu", tmp_path / model
        )
        assert len(preds) == len(rows) and ((preds >= 0) & (preds <= 1)).all()
        assert (targets.astype(int) == df.correct.to_numpy()[rows]).all()
        assert info["best_epoch"] == 1 and info["epochs_run"] == 1
        assert not (tmp_path / model / "qid_model.ckpt").exists()  # weights are removed after scoring
    timing = deep.measure_latency("dkt", seqs["num_q"], seqs["num_c"], history=20, reps=5, warmup=1)
    assert timing["ms_median"] > 0 and timing["ms_wall_median"] > 0 and timing["threads"] == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
