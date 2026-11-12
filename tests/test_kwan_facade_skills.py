"""The skill catalogue on the facade: labels for people, ids for the engine."""

import pytest

from kwan import Kwan


@pytest.fixture(scope="module")
def engine():
    return Kwan.from_bundle()


def test_skills_cover_the_curriculum_in_order(engine):
    skills = engine.skills()
    assert len(skills) == len(engine) == 276
    assert [s["id"] for s in skills] == [r["id"] for r in engine.indicators]


def test_skills_filter_and_search(engine):
    assert {s["grade"] for s in engine.skills(grade=2)} == {2}
    assert {s["strand_name"] for s in engine.skills(strand=4)} == {"Data"}
    assert [s["code"] for s in engine.skills(q="fractions ORDER", grade=4)] == ["B4.1.3.2.1"]
    assert [s["code"] for s in engine.skills(q="B4.1.3.2.1")] == ["B4.1.3.2.1"]
    assert engine.skills(q="zzz") == []


def test_label_accepts_any_spelling(engine):
    label = engine.label("B4.1.3.2.1")
    assert label == engine.label("math:B4.1.3.2.1") == "Put fractions in order from smallest to largest"
    with pytest.raises(KeyError):
        engine.label("B9.9.9.9.9")


def test_a_label_picked_from_the_catalogue_drives_a_goal(engine):
    (skill,) = engine.skills(q="fractions order", grade=4)
    state = engine.new_state("p1")
    recs = engine.recommend(state, k=3, goal=skill["id"], now="2026-11-12T08:00:00+00:00")
    assert recs
