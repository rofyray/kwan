"""Ghana-context bank (PRD 1.2 and 5.3): 21 settings by strand and primary band, picked by CRC32 of the indicator id."""
import zlib
from collections import Counter

import pytest

import context

EM_DASH = chr(0x2014)
CELLS = [(strand, band) for strand in (1, 2, 3, 4) for band in ("lower", "upper")]


def all_settings():
    return [s for options in context.CONTEXTS.values() for s in options]


# ----------------------------------------------------------------------------- the bank

def test_bank_has_exactly_21_settings():
    assert len(all_settings()) == 21
    assert {key: len(v) for key, v in context.CONTEXTS.items()} == {
        (1, "lower"): 3, (1, "upper"): 3, (2, "lower"): 2, (2, "upper"): 2,
        (3, "lower"): 3, (3, "upper"): 3, (4, "lower"): 2, (4, "upper"): 3,
    }


def test_bank_covers_every_strand_and_band_and_nothing_else():
    assert set(context.CONTEXTS) == set(CELLS)


@pytest.mark.parametrize("cell", CELLS)
def test_every_cell_is_non_empty(cell):
    assert len(context.CONTEXTS[cell]) >= 1


def test_settings_are_distinct_complete_sentences():
    settings = all_settings()
    assert len(set(settings)) == len(settings)
    for s in settings:
        assert isinstance(s, str) and s == s.strip() and len(s) > 30
        assert s[0].isupper() and s[-1] in ".?"
        assert "  " not in s and "\n" not in s


def test_no_em_dash_in_the_bank():
    assert not any(EM_DASH in s for s in all_settings())
    assert EM_DASH not in context.__doc__


def test_settings_are_recognisably_ghanaian():
    """A light check that the bank keeps its local colour: cedis, local food, places and games."""
    text = " ".join(all_settings())
    for word in ("cedi", "GH₵", "kente", "trotro", "kenkey", "cocoa", "Adinkra", "Kumasi", "Accra", "ampe", "jollof"):
        assert word in text, word


# ----------------------------------------------------------------------------- band

@pytest.mark.parametrize("grade, expected", [(1, "lower"), (2, "lower"), (3, "lower"),
                                             (4, "upper"), (5, "upper"), (6, "upper")])
def test_band_boundary_is_between_grade_3_and_4(grade, expected):
    assert context.band(grade) == expected


# ----------------------------------------------------------------------------- practice_setting

def test_pick_is_crc32_of_the_indicator_id_modulo_the_cell_size():
    for strand, band in CELLS:
        grade = 2 if band == "lower" else 5
        options = context.CONTEXTS[(strand, band)]
        for i in range(40):
            ind = f"B{grade}.{strand}.1.1.{i}"
            assert context.practice_setting(ind, strand, grade) == options[zlib.crc32(ind.encode()) % len(options)]


def test_same_indicator_always_gets_the_same_setting():
    first = context.practice_setting("B4.1.1.3.3", 1, 4)
    assert all(context.practice_setting("B4.1.1.3.3", 1, 4) == first for _ in range(50))
    assert first in context.CONTEXTS[(1, "upper")]


def test_known_picks():
    """Pinned values: CRC32 is a fixed function, so these never change between runs, machines or Python versions."""
    assert zlib.crc32(b"B1.1.1.1.1") == 1008541236
    assert context.practice_setting("B1.1.1.1.1", 1, 1) == context.CONTEXTS[(1, "lower")][1008541236 % 3]
    assert context.practice_setting("B1.1.1.1.3", 1, 1) == \
        "Use 1 cedi and 2 cedi coins to make 5 cedis in as many ways as you can."
    assert context.practice_setting("B1.3.3.1.1", 3, 1) == \
        "Find circles, squares and triangles in Adinkra symbols printed on cloth at home."


def test_setting_comes_from_the_cell_of_the_strand_and_band():
    for strand, band in CELLS:
        for grade in ((1, 2, 3) if band == "lower" else (4, 5, 6)):
            assert context.practice_setting(f"B{grade}.{strand}.9.9.9", strand, grade) in context.CONTEXTS[(strand, band)]


def test_the_band_not_the_id_decides_the_cell():
    """The id only seeds the pick; strand and grade arguments choose the cell."""
    assert context.practice_setting("B1.1.1.1.1", 1, 6) in context.CONTEXTS[(1, "upper")]
    assert context.practice_setting("B1.1.1.1.1", 4, 1) in context.CONTEXTS[(4, "lower")]


@pytest.mark.parametrize("strand", [0, 5, 99, -1, None, "1"])
def test_unknown_strand_returns_an_empty_string(strand):
    assert context.practice_setting("B1.1.1.1.1", strand, 1) == ""
    assert context.practice_setting("B6.1.1.1.1", strand, 6) == ""


def test_an_emptied_cell_returns_an_empty_string(monkeypatch):
    monkeypatch.setitem(context.CONTEXTS, (2, "lower"), [])
    assert context.practice_setting("B1.2.1.1.1", 2, 1) == ""


# ----------------------------------------------------------------------------- on the real curriculum graph

def test_every_indicator_in_the_graph_gets_a_setting_from_its_own_cell(G):
    assert G.number_of_nodes() >= 250
    for n, d in G.nodes(data=True):
        setting = context.practice_setting(n, d["strand"], d["grade"])
        assert setting, n
        assert setting in context.CONTEXTS[(d["strand"], context.band(d["grade"]))], n
        assert EM_DASH not in setting


def test_every_setting_is_used_by_at_least_one_indicator(G):
    used = Counter(context.practice_setting(n, d["strand"], d["grade"]) for n, d in G.nodes(data=True))
    assert set(used) == set(all_settings())


def test_graph_only_uses_the_four_strands_the_bank_knows(G):
    assert {d["strand"] for _, d in G.nodes(data=True)} == {1, 2, 3, 4}
    assert {d["grade"] for _, d in G.nodes(data=True)} == {1, 2, 3, 4, 5, 6}
