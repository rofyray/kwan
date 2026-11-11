"""The v2 curriculum parser on synthetic pages: statements, corrections, names, exemplars, both paths."""

import pytest

from kwan.curriculum.document import is_truncated, join_parts, parse_pages, unfinished
from kwan.curriculum.exemplars import GhanaTerms
from kwan.curriculum.text import pages_from_text

from .curriculum_helpers import corrections, english_family, layout_row, line, math_family, page, title_page

FAM = math_family()
BANNERS = ["Strand 1: NUMBER", "Sub-strand 1: Counting things"]


def parse(pages, corr=None, terms=None, family=FAM):
    return parse_pages(pages, family, corr, terms)


def one_page(mid, left=(("100", "B4.1.1.1"), ("112", "Know numbers well")), banners=BANNERS, grade=4):
    return [title_page(1, grade), page(2, mid=mid, left=left, banners=banners)]


def codes(result):
    return [r["code"] for r in result.records]


# ---- records ----------------------------------------------------------------------------------------


def test_full_record_fields():
    res = parse(
        one_page(
            [
                (100, "B4.1.1.1.2 Read numbers aloud", True),
                (112, "up to a limit", True),
                (130, "E.g. 1. Say the number on a card", False),
                (142, "to a friend", False),
            ]
        )
    )
    assert not res.log
    (r,) = res.records
    assert r == {
        "id": "math:B4.1.1.1.2",
        "subject": "math",
        "code": "B4.1.1.1.2",
        "grade": 4,
        "strand": 1,
        "strand_name": "Number",
        "sub_strand": 1,
        "sub_strand_name": "Counting things",
        "content_standard": "B4.1.1.1",
        "cs_key": "1.1.1",
        "indicator_no": 2,
        "aliases": ["B4.1.1.1.2"],
        "description": "Read numbers aloud up to a limit",
        "content_standard_text": "Know numbers well",
        "exemplars": [
            {"text": "Say the number on a card to a friend", "ghana_context": False, "page": 2, "marked": True}
        ],
        "page": 2,
        "printed_page": "2",
        "doc": "doc1",
        "printed_code": "B4.1.1.1.2",
    }


@pytest.mark.parametrize(
    "printed, code",
    [
        ("B4.1.1.1.5 Round numbers", "B4.1.1.1.5"),
        ("B4. 1.1.1.5 Round numbers", "B4.1.1.1.5"),  # stray space inside the code
        ("B4.1.1.1. 5 Round numbers", "B4.1.1.1.5"),
        ("B4.1.1.1.5. Round numbers", "B4.1.1.1.5"),  # trailing dot
        ("B4.1.1.1.5: Round numbers", "B4.1.1.1.5"),  # colon straight after the code
        ("B4.1.1.1.5Round numbers", "B4.1.1.1.5"),  # no space before the sentence
        ("B B4.1.1.1.5 Round numbers", "B4.1.1.1.5"),  # stray letter before the code
        ("B4.1.1.1.12 Round numbers", "B4.1.1.1.12"),  # two-digit indicator number
    ],
)
def test_code_variants(printed, code):
    res = parse(one_page([(100, printed, True)]))
    assert codes(res) == [code]
    assert res.records[0]["description"] == "Round numbers"


def test_code_alone_on_its_line_takes_the_statement_below():
    res = parse(one_page([(100, "B4.1.1.1.1", True), (118, "Count in steps", True), (130, "of ten", True)]))
    assert res.records[0]["description"] == "Count in steps of ten"


def test_a_code_inside_a_sentence_is_not_an_indicator():
    res = parse(
        one_page([(100, "B4.1.1.1.1 Count on", True), (115, "E.g. 1. See the task in B4.1.1.1.4 again", False)])
    )
    assert codes(res) == ["B4.1.1.1.1"]


def test_records_are_sorted_by_code_not_by_page_order():
    mid = [(100, "B4.1.1.1.2 Second", True), (120, "B4.1.1.1.10 Tenth", True), (140, "B4.1.1.1.1 First", True)]
    assert codes(parse(one_page(mid))) == ["B4.1.1.1.1", "B4.1.1.1.2", "B4.1.1.1.10"]


def test_front_matter_before_the_first_grade_heading_is_ignored():
    pages = [page(1, mid=[(100, "B1.1.2.1.1 A code in a diagram", True)]), *one_page([(100, "B4.1.1.1.1 Real", True)])]
    assert codes(parse(pages)) == ["B4.1.1.1.1"]


def test_parsing_stops_at_the_end_heading_and_restarts_for_the_next_document():
    from kwan.curriculum.model import PageText

    pages = [
        title_page(1, 1),
        page(2, mid=[(100, "B1.1.1.1.1 One", True)]),
        PageText(doc="doc1", page=3, banners=["MATHS SUBJECT PANEL MEMBERS"], has_table=False),
        page(4, mid=[(100, "B1.1.1.1.2 A name list that looks like a code", True)]),
        title_page(1, 4, doc="doc2"),
        page(2, mid=[(100, "B4.1.1.1.1 Four", True)], doc="doc2"),
    ]
    assert codes(parse(pages)) == ["B1.1.1.1.1", "B4.1.1.1.1"]


# ---- where a statement ends (pdfplumber path: bold type) -------------------------------------------


def test_bold_statement_ends_where_plain_type_starts():
    mid = [
        (100, "B4.1.1.1.1 Compare two", True),
        (112, "numbers", True),
        (124, "Pupils work in pairs on this task", False),
    ]
    (r,) = parse(one_page(mid)).records
    assert r["description"] == "Compare two numbers"
    assert [e["text"] for e in r["exemplars"]] == ["Pupils work in pairs on this task"]
    assert r["exemplars"][0]["marked"] is False


def test_bold_statement_keeps_a_last_line_of_digits():
    mid = [(100, "B4.1.1.1.1 Count up to", True), (112, "1000", True), (130, "E.g. 1. Count", False)]
    assert parse(one_page(mid)).records[0]["description"] == "Count up to 1000"


def test_statement_ending_inside_a_line_takes_only_the_bold_words():
    tail = line(112, "line [a note in plain type about it]", bold=False)
    tail.lead = "line"
    mid = [(100, "B4.1.1.1.1 Put fractions on a number", True), tail]
    (r,) = parse(one_page(mid)).records
    assert r["description"] == "Put fractions on a number line"
    assert r["exemplars"][0]["text"] == "[a note in plain type about it]"


def test_formula_fragment_between_two_bold_lines_is_skipped():
    mid = [
        (100, "B4.1.1.1.1 Add fractions that are", True),
        (106, "15 15 − 15", False),
        (112, "greater than one", True),
    ]
    assert parse(one_page(mid)).records[0]["description"] == "Add fractions that are greater than one"


def test_formula_placed_lower_by_the_pdf_is_put_back_at_the_end_of_the_statement():
    first = line(100, "B4.1.1.1.1 Know the facts up to", bold=True)
    formula = line(118, "12 × 12", bold=False, x=first.x1 + 3)
    mid = [first, (114, "E.g. 1. Use straws", False), formula]
    (r,) = parse(one_page(mid)).records
    assert r["description"] == "Know the facts up to 12 × 12"
    assert not is_truncated(r["description"])


def test_statement_stops_at_a_large_gap():
    mid = [(100, "B4.1.1.1.1 Name the shapes", True), (160, "A LABEL INSIDE A FIGURE", True)]
    assert parse(one_page(mid)).records[0]["description"] == "Name the shapes"


def test_plain_type_statement_continues_only_while_unfinished():
    mid = [
        (100, "B4.1.1.1.1 Show that moving a thing does not change the", False),
        (112, "length of it", False),
        (124, "Use sticks", False),
    ]
    (r,) = parse(one_page(mid)).records
    assert r["description"] == "Show that moving a thing does not change the length of it"


def test_word_broken_with_a_hyphen_is_joined():
    assert join_parts(["a two-", "dimensional shape"]) == "a two-dimensional shape"
    assert join_parts(["sort by", "size"]) == "sort by size"


@pytest.mark.parametrize(
    "text, cut",
    [
        ("Count up to", True),
        ("Compare, order and", True),
        ("Sort shapes,", True),
        ("", True),
        ("Count up to 100", False),
        ("Name the shapes.", False),
    ],
)
def test_truncation_check(text, cut):
    assert is_truncated(text) is cut
    assert unfinished(text) is (cut and bool(text))


# ---- exemplars --------------------------------------------------------------------------------------


def test_exemplars_run_across_pages_until_the_next_indicator():
    pages = [
        title_page(1, 4),
        page(2, mid=[(100, "B4.1.1.1.1 Count", True), (115, "E.g. 1. First task", False)], banners=BANNERS),
        page(3, mid=[(100, "it carries on here", False), (115, "E.g.2 Second task", False), (130, "4 8 12", False)]),
        page(
            4, mid=[(100, "Eg. 3. Third task", False), (120, "B4.1.1.1.2 Order", True), (135, "E.g. 1. Other", False)]
        ),
    ]
    a, b = parse(pages).records
    assert [(e["text"], e["page"]) for e in a["exemplars"]] == [
        ("First task it carries on here", 2),
        ("Second task", 3),
        ("Third task", 4),
    ]
    assert [e["text"] for e in b["exemplars"]] == ["Other"]


def test_lower_case_eg_inside_an_exemplar_does_not_open_a_new_one():
    mid = [
        (100, "B4.1.1.1.1 Count", True),
        (115, "E.g. 1. Describe where things are;", False),
        (127, "e.g. the cup is left of the bowl", False),
        (139, "examples of this are common", False),
    ]
    (r,) = parse(one_page(mid)).records
    assert len(r["exemplars"]) == 1


def test_exemplars_are_flagged_for_ghanaian_context():
    terms = GhanaTerms(["cedi", "Kofi", "¢"])
    mid = [
        (100, "B4.1.1.1.1 Count money", True),
        (115, "E.g. 1. Kofi has 5 cedis", False),
        (127, "E.g. 2. A pen costs ¢3", False),
        (139, "E.g. 3. Count the beads", False),
    ]
    (r,) = parse(one_page(mid), terms=terms).records
    assert [e["ghana_context"] for e in r["exemplars"]] == [True, True, False]


def test_page_that_carries_on_an_earlier_content_standard_feeds_that_indicator():
    pages = [
        title_page(1, 3),
        page(
            2,
            mid=[(100, "B3.1.2.3.1 Add in your head", True), (115, "E.g. 1. Make tens", False)],
            left=[(100, "B3.1.2.3")],
            banners=BANNERS,
        ),
        page(
            3,
            mid=[(100, "B3.1.2.4.1 Add on paper", True), (115, "E.g. 1. Use columns", False)],
            left=[(100, "B3.1.2.4")],
        ),
        page(4, mid=[(100, "E.g. 2. Make doubles", False)], left=[(100, "B3.1.2.3")]),
        page(5, mid=[(100, "E.g. 2. Use jumps", False)], left=[(100, "B3.1.2.4")]),
    ]
    a, b = parse(pages).records
    assert [e["text"] for e in a["exemplars"]] == ["Make tens", "Make doubles"]
    assert [e["text"] for e in b["exemplars"]] == ["Use columns", "Use jumps"]


# ---- names and content standards --------------------------------------------------------------------


def test_sub_strand_name_goes_to_the_sub_strand_of_the_next_indicator():
    """A heading that prints the wrong number still names the sub-strand that follows it."""
    pages = [
        title_page(1, 5),
        page(2, mid=[(100, "B5.3.3.1.1 Measure angles", True)], banners=["Strand 3: GEOMETRY", "Sub-Strand 3: Angles"]),
        page(3, mid=[(100, "B5.3.4.1.1 Give directions", True)], banners=["Sub-Strand 3: Reasoning"]),
        page(4, mid=[(100, "B5.3.4.1.2 Give more directions", True)], banners=["Sub-Strand 3: A repeated heading"]),
    ]
    res = parse(pages)
    assert [r["sub_strand_name"] for r in res.records] == ["Angles", "Reasoning", "Reasoning"]
    assert res.sub_strand_sources == {"B5.3.3": "printed", "B5.3.4": "printed"}


def test_missing_sub_strand_heading_borrows_the_nearest_grade_then_corrections_win():
    pages = [
        title_page(1, 4),
        page(2, mid=[(100, "B4.1.2.1.1 Multiply", True)], banners=["Strand 1: NUMBER", "Sub-strand 2: Operations"]),
        title_page(3, 5),
        page(4, mid=[(100, "B5.1.2.1.1 Multiply more", True), (130, "B5.1.4.1.1 Use decimals", True)]),
    ]
    res = parse(pages)
    by = {r["code"]: r["sub_strand_name"] for r in res.records}
    assert by == {"B4.1.2.1.1": "Operations", "B5.1.2.1.1": "Operations", "B5.1.4.1.1": "Sub-strand 4"}
    assert res.sub_strand_sources["B5.1.2"] == "same sub-strand in grade 4"
    res = parse(pages, corrections(sub_strand_names={"B5.1.4": "Decimals"}))
    assert {r["code"]: r["sub_strand_name"] for r in res.records}["B5.1.4.1.1"] == "Decimals"
    assert res.sub_strand_sources["B5.1.4"] == "corrections"


def test_heading_in_the_middle_of_a_page_ends_the_exemplars_above_it():
    mid = [
        (100, "B4.3.2.1.1 Say where things are", True),
        (115, "E.g. 1. Use left and right", False),
        (140, "Sub-strand 3: Measuring", True),
        (150, "INDICATORS AND EXEMPLARS", True),
        (170, "B4.3.3.1.1 Measure a line", True),
    ]
    res = parse(one_page(mid, banners=["Strand 3: GEOMETRY", "Sub-strand 2: Position"]))
    a, b = res.records
    assert (a["sub_strand_name"], b["sub_strand_name"]) == ("Position", "Measuring")
    assert [e["text"] for e in a["exemplars"]] == ["Use left and right"]


def test_content_standard_text_comes_from_the_cell_beside_the_indicator():
    left = [(100, "B4.1.1.1"), (112, "Know numbers"), (124, "CONT’D"), (200, "B4.1.1.2 Know Roman"), (212, "numerals.")]
    mid = [(100, "B4.1.1.1.1 Count", True), (205, "B4.1.1.2.1 Read Roman numerals", True)]
    a, b = parse(one_page(mid, left=left)).records
    assert a["content_standard_text"] == "Know numbers"
    assert b["content_standard_text"] == "Know Roman numerals"


def test_strand_name_comes_from_the_family_not_from_the_page():
    res = parse(one_page([(100, "B4.2.1.1.1 Find the rule", True)], banners=["Strand 2: ALGEBRA AS PRINTED"]))
    assert res.records[0]["strand_name"] == "Algebra"


# ---- misprints and corrections ----------------------------------------------------------------------


def test_same_code_and_same_statement_twice_is_one_indicator():
    pages = [
        title_page(1, 3),
        page(2, mid=[(100, "B3.3.3.3.1 Time an event", True), (115, "E.g. 1. Clap and count", False)], banners=BANNERS),
        page(
            3,
            mid=[
                (100, "B3.3.3.3.1 Time an event.", True),
                (115, "E.g. 1. Clap and count", False),
                (130, "E.g. 2. Use a song", False),
            ],
        ),
    ]
    res = parse(pages)
    assert codes(res) == ["B3.3.3.3.1"]
    assert [e["text"] for e in res.records[0]["exemplars"]] == ["Clap and count", "Use a song"]
    assert [e["kind"] for e in res.log] == ["repeat_merged"]
    assert not res.errors()


def test_same_code_for_a_different_statement_is_an_error_until_corrected():
    mid = [
        (100, "B6.3.3.5.1 Give compass directions", True),
        (130, "B6.3.3.5.1 Reflect a shape", True),
        (160, "B6.3.3.5.2 Spot a reflection", True),
    ]
    res = parse(one_page(mid, grade=6))
    assert [e["kind"] for e in res.errors()] == ["duplicate_id"]
    fix = corrections(
        {
            "printed": "B6.3.3.5.1",
            "corrected": "B6.3.3.6.1",
            "occurrence": 2,
            "alias": False,
            "reason": "second standard",
        },
        {"printed": "B6.3.3.5.2", "corrected": "B6.3.3.6.2", "reason": "moves with its standard"},
    )
    res = parse(one_page(mid, grade=6), fix)
    assert not res.errors()
    by = {r["code"]: r for r in res.records}
    assert sorted(by) == ["B6.3.3.5.1", "B6.3.3.6.1", "B6.3.3.6.2"]
    assert by["B6.3.3.5.1"]["description"] == "Give compass directions"
    assert by["B6.3.3.6.1"]["aliases"] == ["B6.3.3.6.1"]  # the printed code belongs to the first indicator
    assert by["B6.3.3.6.2"]["aliases"] == ["B6.3.3.6.2", "B6.3.3.5.2"]
    assert by["B6.3.3.6.1"]["content_standard"] == "B6.3.3.6" and by["B6.3.3.6.1"]["cs_key"] == "3.3.6"


def test_code_with_the_wrong_grade_is_an_error_until_corrected():
    mid = [(100, "B3.1.1.3.1 Use negative numbers", True), (130, "B4.1.1.3.2 Count through zero", True)]
    res = parse(one_page(mid, grade=3))
    assert [e["kind"] for e in res.errors()] == ["grade_mismatch"]
    fix = corrections(
        {"printed": "B4.1.1.3.2", "corrected": "B3.1.1.3.2", "doc": "doc1", "alias": False, "reason": "grade digit"}
    )
    res = parse(one_page(mid, grade=3), fix)
    assert not res.errors()
    assert codes(res) == ["B3.1.1.3.1", "B3.1.1.3.2"]
    assert res.records[1]["aliases"] == ["B3.1.1.3.2"]


def test_a_correction_for_one_document_does_not_touch_the_same_code_in_another():
    pages = [
        title_page(1, 3),
        page(2, mid=[(100, "B4.1.1.3.2 Count through zero", True)], banners=BANNERS),
        title_page(1, 4, doc="doc2"),
        page(2, mid=[(100, "B4.1.1.3.2 Find common factors", True)], banners=BANNERS, doc="doc2"),
    ]
    fix = corrections(
        {"printed": "B4.1.1.3.2", "corrected": "B3.1.1.3.2", "doc": "doc1", "occurrence": 1, "alias": False}
    )
    res = parse(pages, fix)
    assert not res.errors()
    assert {r["code"]: r["description"] for r in res.records} == {
        "B3.1.1.3.2": "Count through zero",
        "B4.1.1.3.2": "Find common factors",
    }


def test_printed_code_becomes_an_alias_only_when_free():
    mid = [(100, "B6.1.1.6.1 Order integers", True), (130, "P6.1.2.6.2 Add integers", True)]
    fix = corrections(
        {"printed": "B6.1.1.6.1", "corrected": "B6.1.2.6.1", "reason": "digit"},
        {"printed": "P6.1.2.6.2", "corrected": "B6.1.2.6.2", "reason": "letter"},
    )
    res = parse(one_page(mid, grade=6), fix)
    assert [r["aliases"] for r in res.records] == [["B6.1.2.6.1", "B6.1.1.6.1"], ["B6.1.2.6.2", "P6.1.2.6.2"]]
    names = [a for r in res.records for a in r["aliases"]]
    assert len(names) == len(set(names))


def test_four_part_code_is_plain_text_unless_a_correction_names_it():
    mid = [(100, "B6.4.1.2.1 Pick a way to collect data", True), (130, "B6.5.2.2 Write a questionnaire", True)]
    left = [(100, "B6.4.1.2")]
    res = parse(one_page(mid, left=left, grade=6))
    assert codes(res) == ["B6.4.1.2.1"]
    assert [e["kind"] for e in res.log] == ["unresolved_short_code"]
    fix = corrections({"printed": "B6.5.2.2", "corrected": "B6.4.1.2.2", "alias": False, "reason": "four parts"})
    res = parse(one_page(mid, left=left, grade=6), fix)
    assert codes(res) == ["B6.4.1.2.1", "B6.4.1.2.2"]
    assert res.records[1]["description"] == "Write a questionnaire"
    assert res.records[1]["aliases"] == ["B6.4.1.2.2"]


def test_merge_adds_only_new_exemplars_to_the_named_indicator():
    pages = [
        title_page(1, 4),
        page(
            2, mid=[(100, "B4.3.2.1.1 Use north and south", True), (115, "E.g. 1. Walk north", False)], banners=BANNERS
        ),
        page(3, mid=[(100, "B4.3.3.1.1 Estimate perimeter", True), (115, "E.g. 1. Use a hand span", False)]),
        page(
            4,
            mid=[
                (100, "B4.3.3.1.1", True),
                (112, "Use north and south", True),
                (125, "E.g. 1. Walk north", False),
                (137, "E.g. 2. Walk east", False),
            ],
        ),
    ]
    fix = corrections({"printed": "B4.3.3.1.1", "corrected": "B4.3.2.1.1", "action": "merge", "occurrence": 2})
    res = parse(pages, fix)
    assert not res.errors()
    by = {r["code"]: [e["text"] for e in r["exemplars"]] for r in res.records}
    assert by == {"B4.3.2.1.1": ["Walk north", "Walk east"], "B4.3.3.1.1": ["Use a hand span"]}


def test_drop_heading_keeps_the_text_under_it_with_the_running_indicator():
    pages = [
        title_page(1, 5),
        page(
            2,
            mid=[(100, "B5.1.2.1.1 Multiply in your head", True), (115, "E.g. 1. Double it", False)],
            left=[(100, "B5.1.2.1")],
            banners=BANNERS,
        ),
        page(
            3,
            mid=[(100, "B5.1.4.1.1 Write decimals", True), (115, "E.g. 1. Use a chart", False)],
            left=[(100, "B5.1.4.1")],
        ),
        page(
            4,
            mid=[(100, "B5.1.2.1.1. A stray heading", True), (115, "E.g. 2. Shade a grid", False)],
            left=[(100, "B5.1.4.1")],
        ),
    ]
    fix = corrections({"printed": "B5.1.2.1.1", "corrected": None, "action": "drop_heading", "occurrence": 2})
    res = parse(pages, fix)
    assert not res.errors()
    by = {r["code"]: [e["text"] for e in r["exemplars"]] for r in res.records}
    assert by == {"B5.1.2.1.1": ["Double it"], "B5.1.4.1.1": ["Use a chart", "Shade a grid"]}


def test_mismatch_between_indicator_and_content_standard_cell_is_reported():
    res = parse(one_page([(100, "B4.1.2.6.1 Solve word problems", True)], left=[(100, "B4.1.1.1")]))
    assert [(e["level"], e["kind"]) for e in res.log] == [("warning", "cs_mismatch")]


# ---- the pdftotext path and a second subject (PRD tasks 2.2 and 2.9) --------------------------------


def layout_text(*pages):
    return "\f".join("\n".join(p) for p in pages) + "\f"


def test_text_path_reads_three_columns_and_gives_the_same_records():
    header = layout_row("INDICATORS AND EXEMPLARS", "CONTENT STANDARDS", "CORE COMPETENCIES")
    text = layout_text(
        ["", "BASIC 4".center(80), "", "7", "© Somebody 2019"],
        [
            "Strand 1: NUMBER".center(90),
            "Sub-strand 1: Counting things".center(90),
            header,
            "",
            layout_row("B4.1.1.1.2 Read numbers aloud", "B4.1.1.1", "Learners develop:"),
            layout_row("        up to a limit", "Know numbers well", "Critical Thinking"),
            layout_row("E.g. 1. Say the number on a card"),
            layout_row("        to a friend"),
            layout_row("E.g. 2. Kofi counts 12 cedis"),
            "",
            layout_row("B4.1.1.1.3 Compare two", "", "Critical Thinking"),
            layout_row("numbers"),
            "",
            layout_row("E.g. 1. Use a number line"),
            "",
            "8".rjust(120),
            "© Somebody 2019",
        ],
    )
    pages = pages_from_text(text, FAM, "doc1")
    assert pages[1].printed_page == "8"
    assert [ln.text for ln in pages[1].left] == ["B4.1.1.1", "Know numbers well"]
    assert all("Critical" not in ln.text and "Learners" not in ln.text for ln in pages[1].mid)
    res = parse(pages, terms=GhanaTerms(["cedi", "Kofi"]))
    a, b = res.records
    assert (a["code"], a["description"], a["sub_strand_name"]) == (
        "B4.1.1.1.2",
        "Read numbers aloud up to a limit",
        "Counting things",
    )
    assert a["content_standard_text"] == "Know numbers well"
    assert [(e["text"], e["ghana_context"]) for e in a["exemplars"]] == [
        ("Say the number on a card to a friend", False),
        ("Kofi counts 12 cedis", True),
    ]
    assert (b["description"], [e["text"] for e in b["exemplars"]]) == ("Compare two numbers", ["Use a number line"])
    assert (a["page"], a["printed_page"]) == (2, "8")


def test_second_family_parses_with_configuration_only():
    """Another subject (other code letter, headings, column titles, exemplar marker) needs no code change."""
    fam = english_family()
    header = layout_row("WHAT PUPILS DO", "STANDARD", "SKILLS")
    text = layout_text(
        ["CLASS 1".center(80)],
        [
            "Strand 2: Reading".center(90),
            "Topic 1: Letters and sounds".center(90),
            header,
            layout_row("E1.2.1.1.1 Say the sound of each letter", "E1.2.1.1", "Pupils build confidence"),
            layout_row("Try 1. Point at a letter and say it", "Know letters"),
            layout_row("Try 2. Ama finds the letter that starts her name"),
            layout_row("E1.2.1.1.2 Blend two sounds"),
            layout_row("Try 1: Clap each sound"),
            layout_row("B4.1.1.1.1 A maths code that this family must not read"),
        ],
        ["THE END".center(80)],
        [header, layout_row("E1.2.1.1.3 After the end")],
    )
    res = parse_pages(pages_from_text(text, fam, "e12"), fam, None, GhanaTerms(["Ama"]))
    assert not res.errors()
    assert [r["id"] for r in res.records] == ["eng:E1.2.1.1.1", "eng:E1.2.1.1.2"]
    first = res.records[0]
    assert (first["strand_name"], first["sub_strand_name"], first["content_standard"]) == (
        "Reading",
        "Letters and sounds",
        "E1.2.1.1",
    )
    assert first["content_standard_text"] == "Know letters"
    assert [(e["text"], e["ghana_context"]) for e in first["exemplars"]] == [
        ("Point at a letter and say it", False),
        ("Ama finds the letter that starts her name", True),
    ]
    assert len(res.records[1]["exemplars"]) == 1 and res.records[1]["exemplars"][0]["text"].startswith(
        "Clap each sound"
    )
    assert (
        fam.path("structure_file", root=__import__("pathlib").Path("/x")).as_posix()
        == "/x/data/curriculum/eng.structure.json"
    )
