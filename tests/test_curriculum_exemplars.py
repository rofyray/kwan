"""Exemplar splitting and the Ghana term matcher."""

import pytest

from kwan.curriculum.exemplars import CONTEXT_TERMS, EXTRA_TERMS, GhanaTerms, load_terms, read_terms, split_exemplars
from kwan.curriculum.model import TextLine
from kwan.curriculum.pdf import fix_math

from .curriculum_helpers import math_family

FAM = math_family()
TERMS = GhanaTerms(["cedi", "pesewa", "Ho", "Ama", "Cape Coast", "trotro", "mango", "¢", "GH¢"])


def lines(*texts, page=5):
    return [(page, TextLine(y=float(i), x=0.0, text=t)) for i, t in enumerate(texts)]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Ama buys kenkey", True),
        ("Ama's bag", True),
        ("Ama’s bag", True),
        ("a drama club", False),  # "ama" inside a word
        ("ama is not a name in small letters", False),  # names must match their capital letter
        ("a school in Ho", True),
        ("who is next?", False),
        ("How many?", False),
        ("He pays 5 cedis", True),  # plural s
        ("CEDI notes", True),  # small-letter terms match in any case
        ("ten mangoes", True),  # plural es
        ("fifty pesewas", True),
        ("We drove to Cape Coast.", True),
        ("a cape on the coast", False),
        ("Trotros are full", True),
        ("It costs ¢5", True),  # the cedi sign matches anywhere
        ("GH¢20 for the book", True),
        ("exceeding the limit", False),  # "cedi" inside a word
        ("", False),
    ],
)
def test_term_matching_rules(text, expected):
    assert TERMS.matches(text) is expected


def test_found_lists_terms_in_order():
    assert TERMS.found("Ama pays ¢2 and 50 pesewas in Ho") == ["Ama", "¢", "pesewas", "Ho"]


def test_term_files_are_read_without_comments(tmp_path):
    f = tmp_path / "terms.txt"
    f.write_text("# a comment\n\n# [money]\ncedi\n  Kofi  \n", encoding="utf-8")
    assert read_terms(f) == ["cedi", "Kofi"]
    assert load_terms([f, tmp_path / "missing.txt"]).matches("Kofi has one cedi")


def test_the_shipped_lists_flag_cedi_signs_and_common_names():
    terms = load_terms()
    assert EXTRA_TERMS.exists()
    for text in ("A pen costs ¢3", "GH¢10", "Kofi and Ama share 6 cedis", "Count the pesewas"):
        assert terms.matches(text), text
    assert not terms.matches("Count the beads on the string")
    if CONTEXT_TERMS.exists():  # the context package's list is used when it is there
        assert set(read_terms(CONTEXT_TERMS)) <= set(terms.terms)


@pytest.mark.parametrize(
    "marker",
    [
        "E.g. 1. ",
        "E.g.1. ",
        "E.g. 1 ",
        "E.g.1 ",
        "Eg. 1. ",
        "E .g. 1. ",
        "E.g . 1. ",
        "E.g .1 ",
        "E.g. ",
        "E.g. . ",
        "Example 1, ",
        "E.g.3.",
    ],
)
def test_marker_variants_open_an_exemplar(marker):
    out = split_exemplars(lines(marker + "Sort the cards", "by colour"), FAM)
    assert [(e.text, e.marked) for e in out] == [("Sort the cards by colour", True)]


@pytest.mark.parametrize(
    "text",
    ["e.g. the cup is on the left", "examples of shapes are below", "Eggs are sold in crates", "example, when adding"],
)
def test_look_alikes_do_not_open_an_exemplar(text):
    out = split_exemplars(lines("E.g. 1. Describe what you see", text), FAM)
    assert len(out) == 1 and out[0].text.endswith(text)


def test_number_after_the_marker_is_kept_when_it_is_part_of_the_sentence():
    (e,) = split_exemplars(lines("E.g. 123,400 is 10,000 less than 133,400"), FAM)
    assert e.text == "123,400 is 10,000 less than 133,400"


def test_table_cells_and_stray_symbols_are_dropped():
    out = split_exemplars(
        lines("E.g. 1. Fill the table", "4 8 12", "= + −", "then explain it", "E.g. 2. 7 = 3 + 4"), FAM
    )
    assert [e.text for e in out] == ["Fill the table then explain it"]


def test_text_before_the_first_marker_is_one_activity_when_long_enough():
    out = split_exemplars(lines("Pupils act out a short story", "about sharing", "E.g. 1. Share 8 sweets"), FAM)
    assert [(e.text, e.marked) for e in out] == [
        ("Pupils act out a short story about sharing", False),
        ("Share 8 sweets", True),
    ]
    assert split_exemplars(lines("Value Chart", "E.g. 1. Share 8 sweets"), FAM)[0].text == "Share 8 sweets"


def test_each_exemplar_keeps_the_page_it_starts_on_and_its_flag():
    stream = lines("E.g. 1. Count stones", page=7) + lines("with Ama in Ho", "E.g. 2. Count again", page=8)
    out = split_exemplars(stream, FAM, TERMS)
    assert [(e.page, e.ghana_context) for e in out] == [(7, True), (8, False)]


def test_maths_font_letters_are_undoubled_and_made_plain():
    assert fix_math("up to 𝟏𝟏𝟐𝟐 × 𝟏𝟏𝟐𝟐") == "up to 12 × 12"
    assert fix_math("the line 𝑦𝑦 = 𝑥𝑥") == "the line y = x"
    assert fix_math("𝟏𝟏𝟏𝟏") == "11"  # a real double digit is printed four times
    assert fix_math("cm² and ½ stay") == "cm² and ½ stay"
