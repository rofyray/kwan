"""The fits (`kwan.eval.fit`): tau, the Elo step sizes, item banks, the shipped defaults and their
provenance. Everything runs on tiny synthetic logs; nothing reads raw/."""

import argparse
import json
import math
import random
from pathlib import Path

import pytest

from kwan.data import registry
from kwan.eval import fit
from kwan.learner.params import EloParams
from kwan.learner.scoring import score_stream

ROOT = Path(__file__).resolve().parent.parent
PARAMS_FILE = ROOT / "data" / "params" / "math.params.json"
DAY = 86400.0


def forgetting_rows(
    n_users=120,
    steps=20,
    seed=1,
    tau_true=30.0,
    timestamps=True,
    k_s=0.3,
    k_k=0.4,
    decay=0.05,
    gaps=(0.02, 1, 10, 30, 90, 200),
    skills="abc",
):
    """Answers of learners who learn by the Elo rule and forget each skill with time constant
    `tau_true` days (the model's own forgetting rule, PRD 6.2); items carry their own difficulty."""
    rng = random.Random(seed)
    betas = {f"{s}{i}": rng.gauss(0, 0.7) for s in skills for i in range(6)}
    rows = []
    for u in range(n_users):
        t = 1_700_000_000.0 + u * 7.0
        theta, offsets, last, n, n_s = rng.gauss(0, 0.5), {}, {}, 0, {}
        for _ in range(steps):
            s = rng.choice(skills)
            t += rng.choice(gaps) * DAY
            off = offsets.get(s, 0.0)
            if s in last:
                off *= math.exp(-(t - last[s]) / DAY / tau_true)
            last[s] = t
            item = f"{s}{rng.randrange(6)}"
            p = 1 / (1 + math.exp(-(theta + off - betas[item])))
            y = int(rng.random() < p)
            theta += k_s / (1 + decay * n) * (y - p)
            offsets[s] = off + k_k / (1 + decay * n_s.get(s, 0)) * (y - p)
            n, n_s[s] = n + 1, n_s.get(s, 0) + 1
            rows.append((u, item, s, y, t) if timestamps else (u, item, s, y))
    return rows


def parts_of(rows, share=(0.4, 0.2, 0.4)):
    """{"fit", "val", "train", "test"} by user, in the shape `fit.split_rows` gives."""
    users = sorted({r[0] for r in rows})
    a = int(len(users) * share[0])
    b = a + int(len(users) * share[1])
    fit_u, val_u = set(users[:a]), set(users[a:b])
    return {
        "fit": [r for r in rows if r[0] in fit_u],
        "val": [r for r in rows if r[0] in val_u],
        "train": [r for r in rows if r[0] in fit_u | val_u],
        "test": [r for r in rows if r[0] not in fit_u | val_u],
    }


SMALL_GRID = {"k_s": [0.3], "k_k": [0.4], "decay": [0.05]}


# ----------------------------------------------------------------------------- tau


def test_tau_grid_picks_the_generating_tau_on_simulated_forgetting_data():
    # big true step sizes, so that what a learner forgets over a long gap matters
    true = EloParams(k_s=0.3, k_k=1.5, decay=0.0, tau_days=None)
    parts = parts_of(forgetting_rows(n_users=500, steps=60, tau_true=30.0, k_k=true.k_k, decay=true.decay))
    grid = {"k_s": [true.k_s], "k_k": [true.k_k], "decay": [true.decay]}
    results = fit.run_grid(parts, True, grid, tau_grid=(7.0, 30.0, 180.0, None), base=true)
    curve = fit.tau_curve(results, true)
    assert [r["params"]["tau_days"] for r in curve] == [7.0, 30.0, 180.0, None]  # TAU_GRID order
    assert fit.choose_tau(curve) == 30.0
    off = next(r for r in curve if r["params"]["tau_days"] is None)
    best = next(r for r in curve if r["params"]["tau_days"] == 30.0)
    assert best["val"]["auc"] > off["val"]["auc"] and best["test"]["auc"] > off["test"]["auc"]


def test_forgetting_changes_nothing_without_timestamps():
    rows = forgetting_rows(n_users=60, steps=20, timestamps=False)
    parts = parts_of(rows)
    record, bank = fit.fit_rows(
        "assistments_2009", parts, False, counts={"rows": len(rows), "students": 60, "skills": 4}, grid=SMALL_GRID
    )
    assert record["tau_grid"] == [] and record["tau_chosen"] is None
    assert record["auc_with_forgetting"] == record["auc_without_forgetting"]
    assert record["auc_change_from_forgetting"] == 0.0
    assert all(r["params"]["tau_days"] is None for r in record["grid_results"])
    assert record["auc_v1_defaults"] == record["auc_chosen"]  # one configuration in the grid
    assert len(bank) > 0 and bank.provenance["datasets"] == ["assistments_2009"]


def test_frame_rows_passes_times_only_for_real_timestamps():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(
        {"user_id": [1, 1, 2], "item_id": [5, 6, 5], "skill_id": [1, 1, 1], "t": [10, 20, 30], "correct": [1, 0, 1]}
    )
    assert fit.has_timestamps(df) is False
    assert fit.frame_rows(df) == [(1, 5, 1, 1), (1, 6, 1, 0), (2, 5, 1, 1)]
    df["t"] = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    assert fit.has_timestamps(df) is True
    rows = fit.frame_rows(df)
    assert rows[1][4] - rows[0][4] == DAY and isinstance(rows[0][4], float)


def test_subsample_keeps_whole_students():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"user_id": [u for u in range(10) for _ in range(10)], "x": range(100)})
    same, info = fit.subsample(df, 100)
    assert info is None and len(same) == 100
    part, info = fit.subsample(df, 35)
    assert info["rows_before"] == 100 and info["students_before"] == 10
    assert len(part) == 40  # whole students only: 35 rows need four students of ten rows
    assert part.groupby("user_id").size().eq(10).all()


# ----------------------------------------------------------------------------- the grid and the graph parameters


def test_grade_step_and_transfer_do_nothing_without_a_graph():
    rows = forgetting_rows(n_users=20, steps=15)
    base = score_stream(rows, EloParams(), engine="learner")
    for changed in (EloParams(grade_step=0.0), EloParams(transfer=0.0), EloParams(grade_step=1.0, transfer=1.0)):
        assert score_stream(rows, changed, engine="learner") == base


def test_lookup_finds_a_grid_entry_by_its_parameters():
    parts = parts_of(forgetting_rows(n_users=40, steps=10))
    results = fit.run_grid(parts, True, {"k_s": [0.3, 0.5], "k_k": [0.4], "decay": [0.05]}, tau_grid=(45.0, None))
    assert len(results) == 4
    found = fit.lookup(results, EloParams(k_s=0.5, tau_days=None), True)
    assert found["params"] == {"k_s": 0.5, "k_k": 0.4, "decay": 0.05, "tau_days": None}
    assert fit.lookup(results, EloParams(k_s=0.9), True) is None
    assert fit.lookup(results, {"k_s": 0.3, "tau_days": 45.0}, True)["params"]["tau_days"] == 45.0
    assert results == sorted(results, key=lambda r: -r["val"]["auc"])


# ----------------------------------------------------------------------------- never worse than v1


def test_guard_puts_back_the_parameter_that_regresses():
    # k_k=0.6 costs 0.01 AUC on Eedi; k_s=0.5 and tau help a little everywhere
    def auc_of(p, dataset):
        if dataset == "missing":
            return None
        value = 0.70 + (0.003 if p.k_s == 0.5 else 0.0) + (0.001 if p.tau_days else 0.0)
        if dataset == "eedi_2020_task34" and p.k_k == 0.6:
            value -= 0.01
        return value

    candidate = EloParams(k_s=0.5, k_k=0.6, decay=0.05, tau_days=45.0)
    kept, notes = fit.guard(candidate, auc_of, datasets=("assistments_2009", "eedi_2020_task34", "missing"))
    assert kept == EloParams(k_s=0.5, k_k=0.4, decay=0.05, tau_days=45.0)
    assert len(notes) == 1 and notes[0].startswith("k_k back to the v1 value 0.4")


def test_guard_keeps_a_set_within_tolerance_and_falls_back_to_v1_when_nothing_helps():
    fine = fit.V1_PARAMS.replace(k_s=0.5, tau_days=45.0)
    kept, notes = fit.guard(fine, lambda p, d: 0.7 - (0.0015 if p.k_s == 0.5 else 0.0))
    assert kept == fine and notes == []
    # every tuned parameter hurts: the guard walks back to v1 in as many steps as there are changes
    bad = fit.V1_PARAMS.replace(k_s=0.5, k_k=0.6, decay=0.1, tau_days=45.0)
    kept, notes = fit.guard(
        bad, lambda p, d: 0.7 - 0.01 * sum(getattr(p, f) != getattr(fit.V1_PARAMS, f) for f in fit.GUARD_FIELDS)
    )
    assert kept == fit.V1_PARAMS and len(notes) == 4


# ----------------------------------------------------------------------------- shipped defaults, params file, report


def fit_records(tmp_path, keys=("assistments_2009", "eedi_2020_task34", "ammore_2024")):
    """Fit records of three tiny synthetic datasets written where `report` looks for them."""
    build = tmp_path / "fit"
    build.mkdir()
    for i, key in enumerate(keys):
        timestamps = key != "assistments_2009"
        rows = forgetting_rows(n_users=50, steps=12, seed=i, timestamps=timestamps)
        parts = parts_of(rows)
        record, _ = fit.fit_rows(
            key,
            parts,
            timestamps,
            counts={"rows": len(rows), "students": 50, "skills": 4},
            grid={"k_s": [0.3, 0.5], "k_k": [0.4], "decay": [0.05]},
            tau_grid=(30.0, None),
        )
        record["item_bank"]["saved"] = False
        record["item_bank"]["path"] = None
        (build / f"{key}.json").write_text(json.dumps(record), encoding="utf-8")
    return build


def test_choose_shipped_uses_only_commercial_ok_data(tmp_path):
    fits = fit.load_fits(fit_records(tmp_path))
    assert set(fits) == {"assistments_2009", "eedi_2020_task34", "ammore_2024"}
    shipped = fit.choose_shipped(fits)
    assert shipped["chosen_on"] == ["assistments_2009", "eedi_2020_task34"]
    assert shipped["tau_chosen_on"] == ["eedi_2020_task34"]
    assert (
        "ammore_2024" in shipped["informing_only"]
        and "would_choose_tau_days" in shipped["informing_only"]["ammore_2024"]
    )
    params = EloParams.from_dict(shipped["params"])
    assert params.grade_step == 0.4 and params.transfer == 0.5  # no graph in the public datasets
    assert params.tau_days in (30.0, None)
    assert "eedi_2020_task34" in shipped["held_out"] and "assistments_2009" in shipped["held_out"]
    # the guard holds on the held-out numbers of the guarded datasets
    for key in fit.GUARDED:
        assert shipped["held_out"][key]["gain"] >= -fit.TOLERANCE
    assert "eedi_2020_task12" in shipped["notes"][-1]  # not run, said so
    # a non-commercial record in a deciding slot is refused
    fits["eedi_2020_task34"] = fits["ammore_2024"]
    with pytest.raises(ValueError, match="commercial"):
        fit.choose_shipped(
            {"eedi_2020_task34": fits["ammore_2024"], "ammore_2024": fits["ammore_2024"]}
            | {"assistments_2009": fits["assistments_2009"]}
        )


def test_write_params_refuses_non_commercial_provenance(tmp_path):
    with pytest.raises(ValueError, match="ammore_2024"):
        fit.write_params(EloParams(), {"datasets": ["eedi_2020_task34", "ammore_2024"]}, tmp_path / "p.json")
    with pytest.raises(ValueError, match="junyi"):
        fit.write_params(EloParams(), {"datasets": ["junyi"]}, tmp_path / "p.json")
    with pytest.raises(ValueError, match="must list"):
        fit.write_params(EloParams(), {"datasets": []}, tmp_path / "p.json")
    with pytest.raises(KeyError):
        fit.write_params(EloParams(), {"datasets": ["made-up"]}, tmp_path / "p.json")
    assert not (tmp_path / "p.json").exists()


def test_params_file_round_trip(tmp_path):
    params = EloParams(k_s=0.25, k_k=0.45, decay=0.1, tau_days=60.0)
    path = fit.write_params(params, {"datasets": ["eedi_2020_task34"], "rule": "test"}, tmp_path / "p.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["provenance"] == {"datasets": ["eedi_2020_task34"], "rule": "test"}
    assert EloParams.from_dict(raw) == params  # the provenance block is ignored by from_dict
    assert fit.load_params(path) == params
    assert {k: raw[k] for k in params.to_dict()} == params.to_dict()


def test_report_writes_the_block_and_the_params_file(tmp_path):
    build = fit_records(tmp_path)
    out = tmp_path / "results.json"
    params_path = tmp_path / "params" / "math.params.json"
    block = fit.report(build, out, params_path)
    written = json.loads(out.read_text(encoding="utf-8"))["learner_fit"]
    assert written["shipped"]["params"] == block["shipped"]["params"]
    assert set(written["datasets"]) == {"assistments_2009", "eedi_2020_task34", "ammore_2024"}
    for key, d in written["datasets"].items():
        for field in (
            "rows",
            "students",
            "split",
            "tau_grid",
            "tau_chosen",
            "auc_without_forgetting",
            "auc_with_forgetting",
            "grid_chosen_params",
            "auc_v1_defaults",
            "auc_chosen",
            "rmse_v1_defaults",
            "rmse_chosen",
        ):
            assert field in d, (key, field)
        assert "grid_results" not in d and len(d["grid_top"]) <= 5
    assert written["acceptance"]["passed"] is True
    assert written["acceptance"]["datasets_without_timestamps"] == ["assistments_2009"]
    assert written["item_banks"]["ammore_2024"]["saved"] is False
    saved = json.loads(params_path.read_text(encoding="utf-8"))
    assert EloParams.from_dict(saved) == EloParams.from_dict(block["shipped"]["params"])
    assert all(registry.commercial_ok(d) for d in saved["provenance"]["datasets"])
    assert saved["provenance"]["datasets"] == ["assistments_2009", "eedi_2020_task34"]
    assert chr(0x2014) not in json.dumps(written)


def test_report_needs_fit_records(tmp_path):
    with pytest.raises(FileNotFoundError, match="kwan fit all"):
        fit.report(tmp_path / "empty", None, None)


# ----------------------------------------------------------------------------- the shipped file


def test_shipped_params_file_lists_only_commercial_ok_datasets():
    assert PARAMS_FILE.is_file(), "data/params/math.params.json is a deliverable of the fits"
    raw = json.loads(PARAMS_FILE.read_text(encoding="utf-8"))
    params = EloParams.from_dict(raw)
    assert params == fit.load_params()
    datasets = raw["provenance"]["datasets"]
    assert datasets and all(registry.commercial_ok(d) for d in datasets)
    assert not {registry.dataset_key(d) for d in datasets} & set(registry.non_commercial())
    assert all(registry.commercial_ok(d) for d in raw["provenance"].get("tau_datasets", []))
    assert params.grade_step == 0.4 and params.transfer == 0.5 and params.streak_to_master == 3
    assert "rule" in raw["provenance"] and "held_out_auc" in raw["provenance"]
    for key in fit.GUARDED:
        if key in raw["provenance"]["held_out_auc"]:
            assert raw["provenance"]["held_out_auc"][key]["gain"] >= -fit.TOLERANCE
    assert chr(0x2014) not in PARAMS_FILE.read_text(encoding="utf-8")


# ----------------------------------------------------------------------------- CLI


def test_register_adds_the_fit_commands():
    parser = argparse.ArgumentParser(prog="kwan")
    fit.register(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["fit", "dataset", "eedi-34", "--max-rows", "1000"])
    assert args.func is fit._cmd_dataset and args.extra == "eval" and args.max_rows == 1000
    args = parser.parse_args(["fit", "all", "--only", "eedi-34,ammore"])
    assert args.func is fit._cmd_all
    args = parser.parse_args(["fit", "report", "--out", "x.json"])
    assert args.func is fit._cmd_report and args.out == "x.json"
    with pytest.raises(SystemExit):
        parser.parse_args(["fit", "dataset", "made-up"])


def test_main_report_runs_on_fit_records(tmp_path, capsys):
    build = fit_records(tmp_path)
    out = tmp_path / "results.json"
    code = fit.main(["report", "--build", str(build), "--out", str(out), "--params", str(tmp_path / "p.json")])
    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    assert EloParams.from_dict(printed) == fit.load_params(tmp_path / "p.json")
    assert "learner_fit" in json.loads(out.read_text(encoding="utf-8"))


def test_main_dataset_reports_missing_raw_files(tmp_path, capsys):
    code = fit.main(["dataset", "ammore", str(tmp_path / "nowhere"), "--build", str(tmp_path / "b")])
    assert code == 1
    assert "missing" in capsys.readouterr().err
