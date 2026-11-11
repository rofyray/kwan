"""Phase 4 acceptance: the committed results and, where raw/ data is present, the real datasets.

Fast tests read only data/eval_results.json (aggregate numbers, never rows). Tests marked `slow` read
files under raw/ and are skipped when those are absent; the longer ones also need KWAN_RUN_SLOW=1.
"""

import json
import os
from pathlib import Path

import pytest

from kwan.data import loaders, registry, schema
from kwan.eval import harness

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw"
RESULTS = ROOT / "data" / "eval_results.json"
RUN_SLOW = os.environ.get("KWAN_RUN_SLOW") == "1"
KWAN = "elo-multi (Kwan)"
METRICS = ["AUC", "RMSE", "ACC", "ECE"]
COUNTS = ("rows", "students", "items", "skills")

needs_slow = pytest.mark.skipif(not RUN_SLOW, reason="slow: set KWAN_RUN_SLOW=1 to read the large real datasets")


def committed():
    return json.loads(RESULTS.read_text(encoding="utf-8"))


def flat_blocks(results):
    """(label, block) for every evaluated dataset block of this agent's keys; AMMORE gives two."""
    for s in harness.SPECS:
        block = results.get(s.key)
        if block is None or "skipped" in block:
            continue
        if s.subsets:
            for subset in s.subsets:
                yield f"{s.key}.{subset}", block[subset]
        else:
            yield s.key, block


def have(name):
    s = harness.spec(name)
    return not harness.missing_files(s.key, harness.default_path(s.key, RAW))


def needs_raw(name):
    return pytest.mark.skipif(not have(name), reason=f"raw data for {name} is absent (run make data)")


# ---------------------------------------------------------------- the committed results file (fast)


def test_every_dataset_has_a_block_or_a_skipped_note():
    results = committed()
    for s in harness.SPECS:
        assert s.key in results, f"{s.key} is missing from data/eval_results.json"
        block = results[s.key]
        assert block["dataset"] == s.title
        assert block["licence"] == registry.licence(s.key)
        if "skipped" in block:
            assert list(block) == ["dataset", "licence", "skipped"] and len(block["skipped"]) > 20


def test_committed_blocks_follow_schema_v2():
    results = committed()
    seen = 0
    for label, block in flat_blocks(results):
        seen += 1
        for name in COUNTS:
            assert type(block[name]) is int and block[name] > 0, f"{label}.{name}"
        assert block["students"] <= block["rows"] and block["skills"] <= block["rows"]
        assert block["split"] == "80/20 by student, seed 42"
        models = list(block["metrics"])
        assert models[:3] == ["global-mean", "skill-mean", "elo"] and models[-1] == KWAN, label
        for model, got in block["metrics"].items():
            assert list(got) == METRICS, f"{label}.{model}"
            assert all(isinstance(v, float) and v == round(v, 4) and 0 <= v <= 1 for v in got.values())
        assert block["metrics"]["global-mean"]["AUC"] == 0.5
        best = block["metrics"][KWAN]
        assert best["AUC"] > block["metrics"]["elo"]["AUC"] > 0.5, label  # the Kwan model beats plain Elo everywhere
        assert best["RMSE"] < block["metrics"]["global-mean"]["RMSE"], label
        assert best["ECE"] < 0.1, label  # predictions are roughly calibrated
    assert seen >= 4  # at least ASSISTments 2009, Eedi 3/4 and both AMMORE subsets


def test_v1_fields_are_still_there_for_the_demo():
    results = committed()
    for label, block in flat_blocks(results):
        key = label.split(".")[0]
        if key not in ("assistments_2009", "eedi_2020_task34", "ammore_2024"):
            assert "AUC" not in block and "questions" not in block, label
            continue
        assert block["AUC"] == {m: v["AUC"] for m, v in block["metrics"].items()}, label
        assert block["RMSE"] == {m: v["RMSE"] for m, v in block["metrics"].items()}, label
        if key != "assistments_2009":
            assert block["questions"] == block["items"], label


def test_acceptance_ammore_all_and_ghana():
    """PRD Phase 4: elo-multi AUC >= 0.76 on AMMORE (all) and >= 0.75 on AMMORE (Ghana only)."""
    block = committed()["ammore_2024"]
    assert block["licence"] == "CC BY-NC 4.0"
    assert list(block)[:4] == ["dataset", "licence", "all", "ghana"]
    assert block["all"]["metrics"][KWAN]["AUC"] >= 0.76
    assert block["ghana"]["metrics"][KWAN]["AUC"] >= 0.75
    assert (block["all"]["rows"], block["all"]["students"], block["all"]["skills"]) == (50507, 2388, 151)
    assert (block["ghana"]["rows"], block["ghana"]["students"], block["ghana"]["skills"]) == (16706, 730, 144)
    assert block["ghana"]["rows"] < block["all"]["rows"]


def test_acceptance_no_regression_on_assistments_2009_and_eedi_34():
    """PRD Phase 4: the v1 numbers (0.7545 and 0.7764) must not drop."""
    results = committed()
    a09, e34 = results["assistments_2009"], results["eedi_2020_task34"]
    assert a09["metrics"][KWAN]["AUC"] >= 0.7545 and a09["AUC"][KWAN] >= 0.7545
    assert e34["metrics"][KWAN]["AUC"] >= 0.7764 and e34["AUC"][KWAN] >= 0.7764
    assert (a09["rows"], a09["students"], a09["skills"]) == (259399, 4163, 111)
    assert (e34["rows"], e34["students"], e34["questions"], e34["skills"]) == (1382727, 4918, 948, 57)
    assert a09["metrics"]["bkt"]["AUC"] < a09["metrics"][KWAN]["AUC"]  # BKT runs where v1 ran it


def test_non_commercial_blocks_carry_only_aggregates():
    """AMMORE and Junyi are evaluation only: their blocks hold counts and metrics, never fitted parameters."""
    results = committed()
    allowed = {"dataset", "licence", "note", "skipped", "all", "ghana", "split", "metrics", "cv", "AUC", "RMSE"}
    allowed |= set(COUNTS) | {"questions"}
    for key in ("ammore_2024", "junyi_2015"):
        assert registry.commercial_ok(key) is False
        block = results[key]
        parts = [block, *(block[s] for s in ("all", "ghana") if s in block)]
        for part in parts:
            assert set(part) <= allowed, f"{key}: unexpected field(s) {set(part) - allowed}"


# ---------------------------------------------------------------- real data (slow)


@pytest.mark.slow
@needs_raw("ammore")
def test_real_ammore_reproduces_the_committed_block(capsys):
    """About 5 s: the acceptance numbers come out of the code again, digit for digit."""
    from kwan.eval import ammore

    block = ammore.main([str(RAW / "ammore")])
    out = capsys.readouterr().out
    assert "[all] rows=50507 students=2388 skills=151" in out
    assert "[ghana] rows=16706 students=730 skills=144" in out
    assert block == committed()["ammore_2024"]
    assert block["all"]["AUC"][KWAN] >= 0.76 and block["ghana"]["AUC"][KWAN] >= 0.75


@pytest.mark.slow
@needs_raw("ammore")
def test_real_ammore_loader_follows_the_schema():
    for subset, country in (("all", None), ("ghana", "Ghana")):
        df = loaders.load(f"ammore_2024:{subset}", RAW / "ammore")
        assert schema.validate(df) is df
        assert df.attrs["dataset"]["commercial_params_ok"] is False
        want = committed()["ammore_2024"][subset]
        assert schema.counts(df) == {name: want[name] for name in COUNTS}
        if country:
            assert set(df.meta_country) == {country}
        else:
            assert {"Ghana", "Nigeria", "Kenya", "South Africa"} <= set(df.meta_country.dropna())


@pytest.mark.slow
@needs_slow
@pytest.mark.parametrize("name", ["assistments-2015", "assistments-2017"])
def test_real_evaluation_reproduces_the_committed_block(name, capsys):
    """Two mid-sized datasets are evaluated again in full: the committed block must come out unchanged."""
    if not have(name):
        pytest.skip(f"raw data for {name} is absent (run make data)")
    s = harness.spec(name)
    assert harness.run_dataset(name, raw_dir=RAW) == committed()[s.key]


@pytest.mark.slow
@needs_slow
@pytest.mark.parametrize(
    "name",
    [
        "assistments-2009",
        "eedi-34",
        "eedi-12",
        "assistments-2012",
        "assistments-2015",
        "assistments-2017",
        "xes3g5m",
        "junyi",
    ],
)
def test_real_loader_follows_the_schema_and_matches_the_committed_counts(name):
    """Every real file loads into the one schema, and its counts are the ones the results file reports."""
    if not have(name):
        pytest.skip(f"raw data for {name} is absent (run make data)")
    s = harness.spec(name)
    block = committed()[s.key]
    if "skipped" in block:
        pytest.skip(f"{s.key} was skipped in the committed results: {block['skipped']}")
    df = loaders.load(s.key, harness.default_path(s.key, RAW))
    assert schema.validate(df) is df
    assert list(df.columns)[:5] == list(schema.CORE)
    assert schema.counts(df) == {name_: block[name_] for name_ in COUNTS}
    info = df.attrs["dataset"]
    assert info["licence"] == block["licence"] == registry.licence(s.key)
    assert info["commercial_params_ok"] is registry.commercial_ok(s.key)
