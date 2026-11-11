"""Phase 7 Junyi check (task 7.3) on a made-up exercise table. No Junyi row is used as a fixture.

The real files are read only by the test marked `slow`, which is skipped when `raw/junyi/` is not there.
"""

import csv
import json
from pathlib import Path

import pytest

from kwan.graph import junyi
from kwan.graph.candidates import CrossStrandRules, HashingEmbedder

WORDS = ["apples", "baskets", "coins", "drums", "eggs", "fans", "gourds", "hats", "inks", "jars", "kites", "lamps"]
TOPICS = ["sorting", "sharing", "joining", "halving", "doubling", "ordering"]


def make_table(root: Path) -> Path:
    """A toy table: 6 topics x 12 exercises. In a topic each exercise needs the one before it."""
    rows = []
    for t, topic in enumerate(TOPICS):
        area = "alpha" if t < 3 else "beta"
        for k, word in enumerate(WORDS):
            name = f"{topic}_{word}"
            pre = f"{topic}_{WORDS[k - 1]}" if k else ""
            rows.append({"name": name, "live": "TRUE", "prerequisites": pre, "topic": topic, "area": area})
    rows.append({"name": "steps_1", "live": "TRUE", "prerequisites": "", "topic": "steps", "area": "alpha"})
    rows.append({"name": "steps_2", "live": "TRUE", "prerequisites": "steps_1", "topic": "steps", "area": "alpha"})
    rows.append(
        {
            "name": "steps_10",
            "live": "TRUE",
            "prerequisites": "steps_2,sorting_apples",
            "topic": "steps",
            "area": "alpha",
        }
    )
    with open(root / "junyi_Exercise_table.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    header = "Exercise_A,Exercise_B,Similarity_avg,Similarity_raw,Difficulty_avg,Difficulty_raw,Prerequisite_avg,Prerequisite_raw\n"
    (root / "relationship_annotation_training.csv").write_text(
        header + "sorting_baskets,sorting_apples,8,8_8,2,2_2,8.5,8_9\nsorting_coins,joining_hats,2,2_2,5,5_5,1.5,1_2\n",
        encoding="utf-8",
    )
    (root / "relationship_annotation_testing.csv").write_text(
        header + "sharing_eggs,sharing_drums,7,7_7,3,3_3,7.0,7_7_7\n", encoding="utf-8"
    )
    return root


@pytest.fixture
def root(tmp_path):
    return make_table(tmp_path)


def test_loaders_read_names_prerequisites_and_the_direction_of_annotations(root):
    exercises = junyi.load_exercises(root)
    assert len(exercises) == 75
    by_name = {e["name"]: e for e in exercises}
    assert by_name["steps_10"]["prerequisites"] == ["steps_2", "sorting_apples"]
    assert by_name["sorting_apples"] == {
        "name": "sorting_apples",
        "prerequisites": [],
        "topic": "sorting",
        "area": "alpha",
    }
    ann = junyi.load_annotations(root)
    assert len(ann) == 3
    # a row (Exercise_A, Exercise_B) scores "B is a prerequisite of A": the edge runs B -> A
    assert ann[0] == {
        "source": "sorting_apples",
        "target": "sorting_baskets",
        "prerequisite_avg": 8.5,
        "raters": 2,
        "file": "training",
    }
    assert ann[2]["file"] == "testing" and ann[2]["raters"] == 3


def test_series_get_rule_edges_and_those_links_are_not_scored(root):
    exercises = junyi.load_exercises(root)
    G = junyi.junyi_graph(exercises)
    assert sorted(G.edges()) == [("junyi:steps_1", "junyi:steps_2"), ("junyi:steps_2", "junyi:steps_10")]  # 2 < 10
    node = G.nodes["junyi:sorting_apples"]
    assert node["short_label"] == "sorting apples" and node["description"] == "sorting apples (sorting)"
    assert node["grade"] == 0 and node["strand"] == "alpha"
    assert (
        junyi.junyi_graph(exercises, with_topic=False).nodes["junyi:sorting_apples"]["description"] == "sorting apples"
    )
    links = junyi.expert_links(exercises)
    assert len(links) == 6 * 11 + 3
    assert ("junyi:sorting_apples", "junyi:steps_10") in links
    assert len(links - set(G.edges())) == 6 * 11 + 1


def test_series_edges_put_a_bare_name_first():
    assert junyi.series_edges(["add", "add_2", "add_1", "other"]) == [("add", "add_1"), ("add_1", "add_2")]


def test_split_is_fixed_and_by_target(root):
    names = [e["name"] for e in junyi.load_exercises(root)]
    one, two = junyi.split_targets(names), junyi.split_targets(reversed(names))
    assert one == two and 0 < len(one) < len(names)
    assert 0.15 < len(one) / len(names) < 0.45


def test_choose_threshold_takes_the_lowest_threshold_that_meets_the_bar():
    scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
    labels = [1, 1, 1, 1, 0, 1, 0, 0]
    out = junyi.choose_threshold(scores, labels, bar=0.8, min_support=2)
    assert out["meets_bar"] and out["threshold"] == 0.4 and out["proposed"] == 6 and out["precision"] == 0.8333
    strict = junyi.choose_threshold(scores, labels, bar=0.8, min_support=7)
    assert not strict["meets_bar"] and strict["threshold"] == 0.3 and strict["precision"] == 0.7143


def test_choose_threshold_reports_honestly_when_nothing_meets_the_bar():
    out = junyi.choose_threshold([0.9, 0.8, 0.7, 0.6], [0, 1, 0, 1], bar=0.8, min_support=2)
    assert out == {
        "threshold": 0.6,
        "proposed": 4,
        "correct": 2,
        "precision": 0.5,
        "meets_bar": False,
        "bar": 0.8,
        "min_support": 2,
    }
    empty = junyi.choose_threshold([0.9], [1], bar=0.8, min_support=5)
    assert not empty["meets_bar"] and empty["proposed"] == 0 and empty["precision"] is None


def test_choose_threshold_does_not_split_tied_scores():
    out = junyi.choose_threshold([0.9, 0.5, 0.5, 0.5], [1, 1, 0, 0], bar=0.8, min_support=1)
    assert out["threshold"] == 0.9 and out["proposed"] == 1


def test_pr_table_measures_recall_against_all_true_links():
    rows = junyi.pr_table([0.9, 0.6, 0.4, 0.1], [1, 0, 1, 0], total_positive=4, thresholds=[0.5, 0.3, 0.95])
    assert rows[0] == {"threshold": 0.5, "proposed": 2, "correct": 1, "precision": 0.5, "recall": 0.25}
    assert rows[1]["recall"] == 0.5 and rows[1]["precision"] == 0.6667
    assert rows[2] == {"threshold": 0.95, "proposed": 0, "correct": 0, "precision": None, "recall": 0.0}


def test_benchmark_fits_on_training_targets_and_reports_held_out_numbers_only(root, tmp_path):
    pytest.importorskip("sklearn")
    out = tmp_path / "build" / "augment"
    rules = CrossStrandRules(frozenset({("alpha", "beta")}))
    report = junyi.benchmark(root, HashingEmbedder(), rules, top_k=5, out_dir=out)
    held = set(junyi.split_targets(e["name"] for e in junyi.load_exercises(root)))
    assert report["targets"] == {"train": 75 - len(held), "held_out": len(held)}
    assert report["candidate_pairs"]["held_out"] <= 5 * len(held)
    assert report["expert_links_scored"] == 67 and report["expert_links_already_given_by_rules"] == 2
    assert sum(report["expert_links_scored_by_split"].values()) == 67
    assert report["threshold_choice"]["where"] == "out-of-fold scores of the training targets"
    h = report["held_out"]
    assert set(h) >= {"threshold", "proposed", "correct", "precision", "recall", "meets_precision_bar"}
    assert h["meets_precision_bar"] == (h["precision"] is not None and h["precision"] >= 0.8)
    assert h["threshold"] == round(report["threshold_choice"]["threshold"], 4)
    assert report["cost_usd"] == 0.0 and report["licence"] == "non-commercial use only"
    assert [r["threshold"] for r in report["held_out_curve"]] == sorted(
        r["threshold"] for r in report["held_out_curve"]
    )
    assert report["annotations_held_out"]["direction"] == "Exercise_B -> Exercise_A"

    saved = json.loads((out / "junyi_report.json").read_text(encoding="utf-8"))
    assert saved["held_out"] == h
    artifact = json.loads((out / "local_encoder.json").read_text(encoding="utf-8"))
    assert artifact["threshold"] == pytest.approx(report["threshold_choice"]["threshold"], abs=1e-6)
    assert "do not commit" in artifact["fitted_on"]
    assert artifact["n_pairs"] == report["candidate_pairs"]["train"]  # held-out targets were never fitted on


def test_benchmark_with_a_ready_model_fits_nothing(root, tmp_path):
    from kwan.graph.providers.fake import FakeDecisionModel

    rules = CrossStrandRules(frozenset())
    fake = FakeDecisionModel()
    report = junyi.benchmark(root, HashingEmbedder(), rules, provider="fake", model=fake, top_k=3, out_dir=tmp_path)
    assert report["provider"] == "fake" and "classifier" not in report and report["cost_usd"] is None
    assert not (tmp_path / "local_encoder.json").exists() and (tmp_path / "junyi_report.json").is_file()
    assert len(fake.asked) >= report["candidate_pairs"]["train"] + report["candidate_pairs"]["held_out"]
    with pytest.raises(ValueError, match="needs a ready model"):
        junyi.benchmark(root, HashingEmbedder(), rules, provider="llm-json", write=False)


def test_fitted_files_default_to_the_ignored_build_folder():
    from kwan.graph import augment
    from kwan.graph.providers.local_encoder import default_artifact_path

    assert default_artifact_path().parts[-3:] == ("build", "augment", "local_encoder.json")
    assert augment.default_cache_path().parts[-3:] == ("build", "augment", "cache.jsonl")


RAW = Path(__file__).resolve().parents[1] / "raw" / "junyi"


@pytest.mark.slow
@pytest.mark.skipif(not (RAW / "junyi_Exercise_table.csv").is_file(), reason="raw/junyi is not on this machine")
def test_real_junyi_files_load_and_annotations_point_from_b_to_a():
    exercises = junyi.load_exercises(RAW)
    links = junyi.expert_links(exercises)
    assert len(exercises) > 800 and len(links) > 900
    ann = junyi.load_annotations(RAW)
    assert len(ann) > 1900
    forward = [r["prerequisite_avg"] for r in ann if (junyi.node_id(r["source"]), junyi.node_id(r["target"])) in links]
    backward = [r["prerequisite_avg"] for r in ann if (junyi.node_id(r["target"]), junyi.node_id(r["source"])) in links]
    assert sum(forward) / len(forward) > sum(backward) / len(backward)  # the B -> A reading fits the expert links
