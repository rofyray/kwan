"""The interaction schema shared by every loader: user_id, item_id, skill_id, t, correct, meta."""

import pandas as pd
import pytest

from kwan.data import schema
from kwan.data.schema import SchemaError


def frame(**overrides):
    data = {
        "user_id": [1, 1, 2, 2, 2],
        "item_id": [10, 11, 10, 12, 11],
        "skill_id": [5, 5, 5, 6, 5],
        "t": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-01", "2024-01-03"]),
        "correct": [1, 0, 1, 1, 0],
    }
    data.update(overrides)
    return pd.DataFrame(data)


def test_schema_names():
    assert schema.CORE == ("user_id", "item_id", "skill_id", "t", "correct")
    assert schema.COLUMNS == (*schema.CORE, "meta")
    assert schema.ALIASES == {"problem_id": "item_id"}


def test_validate_accepts_a_good_frame_and_returns_it():
    df = frame(meta_country=["Ghana"] * 5)
    assert schema.validate(df) is df
    assert schema.is_valid(df)
    assert schema.validate(frame(t=[1, 2, 1, 1, 7])) is not None  # an ordinal that grows with time
    assert schema.validate(frame(correct=[True, False, True, True, False])) is not None
    assert schema.is_valid(frame().iloc[:0])  # an empty log is still a log


@pytest.mark.parametrize(
    "bad, message",
    [
        (lambda: frame().drop(columns=["skill_id", "t"]), "missing column(s): skill_id, t"),
        (lambda: frame().rename(columns={"item_id": "problem_id"}), "missing column(s): item_id"),
        (lambda: frame().assign(country="Ghana"), "outside the schema (name them meta_<field>): country"),
        (lambda: frame(skill_id=[5, None, 5, 6, 5]), "column 'skill_id' has missing values"),
        (lambda: frame(correct=[1, 0, 2, 1, 0]), "column 'correct' must hold only 0 or 1"),
        (lambda: frame(correct=[1.0, 0.5, 1.0, 1.0, 0.0]), "column 'correct' must hold whole numbers 0 or 1"),
        (lambda: frame(t=["mon", "tue", "mon", "mon", "wed"]), "column 't' must be a datetime or a number"),
        (lambda: frame(user_id=[1, 2, 1, 2, 2]), "rows of one student must sit together"),
        (lambda: frame(t=[2, 1, 1, 1, 7]), "rows of one student must be in time order"),
    ],
)
def test_validate_says_what_is_wrong(bad, message):
    df = bad()
    with pytest.raises(SchemaError) as exc:
        schema.validate(df)
    assert message in str(exc.value)
    assert not schema.is_valid(df)
    assert isinstance(exc.value, ValueError)


def test_conform_renames_orders_sorts_and_drops_stray_columns():
    raw = pd.DataFrame(
        {
            "correct": [0, 1, 1, 0],
            "problem_id": [7, 8, 9, 7],  # the v1 name
            "user_id": [2, 1, 2, 1],
            "t": [5, 9, 5, 3],
            "skill_id": [1, 1, 2, 1],
            "junk": ["a", "b", "c", "d"],
            "meta_country": ["GH", "KE", "GH", "KE"],
        },
        index=[40, 30, 20, 10],
    )
    before = raw.copy()
    df = schema.conform(raw, {"key": "toy", "licence": "MIT"})
    assert list(df.columns) == [*schema.CORE, "meta_country"]
    assert df.index.tolist() == [0, 1, 2, 3]
    assert df.user_id.tolist() == [1, 1, 2, 2] and df.t.tolist() == [3, 9, 5, 5]
    assert df.item_id.tolist() == [7, 8, 7, 9]  # equal times keep the order the loader gave (stable sort)
    assert df.attrs["dataset"] == {"key": "toy", "licence": "MIT"}
    pd.testing.assert_frame_equal(raw, before)  # the input is not changed


def test_conform_without_sort_still_checks_the_order():
    with pytest.raises(SchemaError, match="time order"):
        schema.conform(frame(t=[2, 1, 1, 1, 7]), sort=False)
    with pytest.raises(SchemaError, match="missing column"):
        schema.conform(frame().drop(columns="correct"))


def test_records_fold_meta_columns_into_one_dictionary():
    df = frame(meta_country=["Ghana", "Ghana", "Kenya", "Kenya", "Kenya"], meta_row=[0, 1, 2, 3, 4])
    recs = list(schema.records(df))
    assert len(recs) == 5
    assert list(recs[0]) == ["user_id", "item_id", "skill_id", "t", "correct", "meta"] == list(schema.COLUMNS)
    assert recs[0]["meta"] == {"country": "Ghana", "row": 0}
    assert (recs[3]["user_id"], recs[3]["item_id"], recs[3]["correct"]) == (2, 12, 1)
    assert list(schema.records(frame()))[0]["meta"] == {}
    assert schema.meta_columns(df) == ["meta_country", "meta_row"]


def test_problem_id_alias_and_counts():
    df = frame()
    v1 = schema.with_problem_id(df)
    assert v1.problem_id.tolist() == df.item_id.tolist() and "problem_id" not in df.columns
    want = {"rows": 5, "students": 2, "items": 3, "skills": 2}
    assert schema.counts(df) == want
    assert schema.counts(df.rename(columns={"item_id": "problem_id"})) == want  # v1 frames count the same
    assert all(type(v) is int for v in schema.counts(df).values())
