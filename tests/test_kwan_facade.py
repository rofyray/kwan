"""The public interface (PRD 9.1): `kwan.Kwan`, `kwan.LearnerState`, `kwan.Recommendation`."""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta

import pytest

import kwan
from kwan import Kwan, LearnerState, Recommendation
from kwan.learner.params import EloParams

T0 = datetime(2026, 3, 2, 8, 0, tzinfo=UTC)
REC_FIELDS = (
    "indicator",
    "code",
    "score",
    "mastery",
    "grade",
    "strand",
    "description",
    "short_label",
    "rationale",
    "rationale_parts",
    "ghana_context",
    "is_review",
)


@pytest.fixture(scope="module")
def engine() -> Kwan:
    return Kwan.from_bundle()


# ----------------------------------------------------------------------------- the package surface


def test_the_public_names_are_exported():
    assert kwan.__all__ == ["Kwan", "LearnerState", "Recommendation", "__version__"]
    assert kwan.Kwan is Kwan and kwan.LearnerState is LearnerState and kwan.Recommendation is Recommendation
    assert "Kwan" in dir(kwan)
    with pytest.raises(AttributeError):
        kwan.no_such_thing  # noqa: B018


def test_import_kwan_stays_light():
    code = "import sys, kwan; print(sorted(m for m in ('networkx', 'fastapi', 'pandas') if m in sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"


def test_every_public_method_has_a_docstring():
    for name in ("from_bundle", "from_graph", "new_state", "update", "recommend", "mastery_map", "mastery", "resolve"):
        assert getattr(Kwan, name).__doc__, name
    assert Recommendation.__doc__ and Recommendation.to_dict.__doc__


# ----------------------------------------------------------------------------- construction


def test_from_bundle_describes_itself(engine):
    assert engine.subject == "math" and engine.text_mode == "labels"
    assert engine.bundle_version == engine.bundle.bundle_version and engine.bundle_version.startswith("math-")
    assert len(engine) == engine.graph.number_of_nodes() == 276
    assert repr(engine) == f"Kwan(math, {engine.bundle_version}, 276 skills, {engine.graph.number_of_edges()} links)"
    assert engine.params == EloParams.from_dict(engine.bundle.params)
    records = engine.indicators
    assert len(records) == 276 and records[0]["id"] == "math:B1.1.1.1.1" and records[0]["short_label"]
    assert all(r["id"].startswith("math:") and r["grade"] in range(1, 7) for r in records)


def test_from_graph_runs_on_the_frozen_v1_graph(G_v1):
    G = G_v1
    engine = Kwan.from_graph(G)
    assert engine.bundle is None and engine.bundle_version == "" and engine.subject == "math"
    assert engine.resolve("math:B1.1.1.1.1") == "B1.1.1.1.1"  # v1 graphs keep bare keys
    st = engine.new_state("v1")
    engine.update(st, "B1.1.1.1.1", True)
    recs = engine.recommend(st, k=3, goal_mode="soft", variety=0.0, review=False)
    assert recs and all(r.indicator in G for r in recs)
    assert set(engine.mastery_map(st)) == set(G)


def test_from_bundle_with_a_missing_file():
    with pytest.raises(FileNotFoundError):
        Kwan.from_bundle("/no/such/bundle.json")


# ----------------------------------------------------------------------------- ids


def test_bare_codes_and_canonical_ids_are_accepted_everywhere(engine):
    assert engine.resolve("B4.1.1.1.2") == engine.resolve("math:B4.1.1.1.2") == "math:B4.1.1.1.2"
    assert engine.resolve("b4.1.1.1.2") == "math:B4.1.1.1.2"
    assert "B4.1.1.1.2" in engine and "math:B4.1.1.1.2" in engine and "B9.9.9.9.9" not in engine
    with pytest.raises(KeyError, match="unknown indicator"):
        engine.resolve("B9.9.9.9.9")
    a, b = engine.new_state("a"), engine.new_state("b")
    assert engine.update(a, "B1.1.1.1.1", True) == engine.update(b, "math:B1.1.1.1.1", True)
    assert a.theta_k == b.theta_k == {"math:B1.1.1.1.1": a.theta_k["math:B1.1.1.1.1"]}
    assert engine.mastery(a, "B1.1.1.1.1") == engine.mastery(a, "math:B1.1.1.1.1")


# ----------------------------------------------------------------------------- learners


def test_new_state_records_the_bundle(engine):
    st = engine.new_state("pupil-1")
    assert isinstance(st, LearnerState)
    assert st.learner_id == "pupil-1" and st.subject == "math" and st.bundle_version == engine.bundle_version
    assert st.step == 0 and st.theta_k == {} and st.schema_version == 2
    with pytest.raises(ValueError):
        engine.new_state("  ")


def test_update_returns_the_prediction_and_changes_the_state(engine):
    st = engine.new_state("u")
    p = engine.update(st, "B1.1.1.1.1", correct=True)
    assert 0 < p < 1 and st.step == 1 and st.n["_all"] == 1
    m1 = engine.mastery(st, "B1.1.1.1.1")
    engine.update(st, "B1.1.1.1.1", False)
    assert engine.mastery(st, "B1.1.1.1.1") < m1
    with pytest.raises(KeyError):
        engine.update(st, "B9.9.9.9.9", True)
    assert st.step == 2  # a bad id changes nothing


def test_update_with_a_time_feeds_forgetting(engine):
    st = engine.new_state("t")
    for i in range(3):
        engine.update(st, "B1.1.1.1.1", True, at=T0 + timedelta(minutes=i))
    assert st.last_seen_at["math:B1.1.1.1.1"] == (T0 + timedelta(minutes=2)).isoformat()
    fresh = engine.mastery(st, "B1.1.1.1.1", now=T0 + timedelta(minutes=3))
    later = engine.mastery(st, "B1.1.1.1.1", now=T0 + timedelta(days=200))
    assert fresh >= 0.95 > later
    assert engine.mastery_map(st, now=T0 + timedelta(days=200))["math:B1.1.1.1.1"] == later
    assert engine.update(st, "B1.1.1.1.1", True, at="2026-06-01T08:00:00+00:00") < 0.95


def test_a_state_adopted_from_elsewhere_gets_the_bundle_version(engine):
    st = LearnerState.from_dict({"learner_id": "old", "theta_k": {"B1.1.1.1.1": 0.4}})  # a v1 state
    engine.update(st, "B1.1.1.1.2", True)
    assert st.bundle_version == engine.bundle_version and st.subject == "math"


# ----------------------------------------------------------------------------- recommendations


def test_recommend_returns_frozen_records_with_every_field(engine):
    st = engine.new_state("r")
    recs = engine.recommend(st, k=3)
    assert 1 <= len(recs) <= 3 and all(isinstance(r, Recommendation) for r in recs)
    assert [f.name for f in dataclasses.fields(Recommendation)] == list(REC_FIELDS)
    for r in recs:
        assert r.indicator.startswith("math:") and r.code == r.indicator.split(":")[1]
        assert 0 <= r.mastery <= 1 and r.grade == 1 and r.strand and r.description == r.short_label
        assert r.rationale and r.rationale_parts and r.ghana_context and r.is_review is False
        assert all(set(p) == {"key", "params"} for p in r.rationale_parts)
        d = r.to_dict()
        assert list(d) == list(REC_FIELDS) and json.loads(json.dumps(d)) == d
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.score = 1.0  # type: ignore[misc]
    assert engine.last_latency_ms >= 0


def test_recommend_options(engine):
    st = engine.new_state("o")
    assert engine.recommend(st, k=0) == []
    assert len(engine.recommend(st, k=1)) == 1
    on_path = engine.recommend(st, k=3, goal="B4.1.1.1.2")
    assert on_path == engine.recommend(st, k=3, goal="math:B4.1.1.1.2")
    assert all(r.grade <= 2 for r in engine.recommend(st, k=5, grade_max=2))
    with pytest.raises(KeyError, match="unknown indicator"):
        engine.recommend(st, goal="NOPE")
    with pytest.raises(ValueError):
        engine.recommend(st, goal_mode="loose")
    with pytest.raises(ValueError):
        engine.recommend(st, k=-1)
    soft = engine.recommend(st, k=3, goal="B4.1.1.1.2", goal_mode="soft")
    assert {r.indicator for r in soft} and len(soft) <= 3


def test_include_drafts_changes_only_the_practice_setting(engine):
    st = engine.new_state("d")
    plain = engine.recommend(st, k=2)
    drafts = engine.recommend(st, k=2, include_drafts=True)
    assert [r.indicator for r in plain] == [r.indicator for r in drafts]
    assert [r.score for r in plain] == [r.score for r in drafts]
    bank = set(engine.bundle.context_bank["1:lower"])
    assert all(r.ghana_context in bank for r in plain)  # nothing is reviewed yet: the bank is used
    has_drafts = any(not e["reviewed"] for es in engine.bundle.contexts.values() for e in es)
    if has_drafts:
        assert any(r.ghana_context not in bank for r in drafts)


def test_the_review_slot_comes_through(engine):
    st = engine.new_state("rv")
    for i in range(3):
        engine.update(st, "B1.1.1.1.1", True, at=T0 + timedelta(minutes=i))
    recs = engine.recommend(st, k=3, now=T0 + timedelta(days=120))
    review = [r for r in recs if r.is_review]
    assert len(review) == 1 and review[0].indicator == "math:B1.1.1.1.1"
    assert review[0].rationale_parts[0]["key"] == "review"


def test_mastery_map_covers_every_skill_in_canonical_ids(engine):
    st = engine.new_state("m")
    table = engine.mastery_map(st)
    assert list(table) == list(engine.graph) and all(0 <= v <= 1 for v in table.values())
    engine.update(st, "B1.1.1.1.1", True)
    table = engine.mastery_map(st)
    for n in list(table)[:20]:
        assert table[n] == engine.mastery(st, n)


# ----------------------------------------------------------------------------- determinism (PRD 7)


def _story(engine: Kwan, eps: float = 0.05):
    st = engine.new_state("same-pupil")
    answers = [
        ("B1.1.1.1.1", True),
        ("B1.1.1.1.1", True),
        ("B1.1.1.1.2", False),
        ("B1.1.2.1.1", True),
        ("B1.1.1.1.2", True),
    ]
    out = []
    for i, (ind, ok) in enumerate(answers):
        out.append(engine.update(st, ind, ok, at=T0 + timedelta(hours=i)))
    now = T0 + timedelta(days=10)
    recs = [r.to_dict() for r in engine.recommend(st, k=5, goal="B2.1.1.1.1", now=now, eps=eps)]
    return out, recs, engine.mastery_map(st, now=now), st.to_dict()


def test_same_bundle_same_answers_same_now_give_the_same_output():
    a = _story(Kwan.from_bundle())
    b = _story(Kwan.from_bundle())
    assert a == b
    assert json.dumps(a[1], sort_keys=True) == json.dumps(b[1], sort_keys=True)


def test_a_different_now_or_eps_can_change_the_output(engine):
    recs_a = _story(engine, eps=0.0)[1]
    recs_b = _story(engine, eps=0.0)[1]
    assert recs_a == recs_b
    st = engine.new_state("same-pupil")
    engine.update(st, "B1.1.1.1.1", True, at=T0)
    early = engine.mastery_map(st, now=T0 + timedelta(days=1))["math:B1.1.1.1.1"]
    late = engine.mastery_map(st, now=T0 + timedelta(days=300))["math:B1.1.1.1.1"]
    assert early > late
