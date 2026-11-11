"""Curriculum extraction: parser behaviour on synthetic layout text, then invariants on the committed data."""

import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from kwan.curriculum import parse as pc

ROOT = Path(__file__).resolve().parent.parent
RAW = [ROOT / "raw" / "m13.txt", ROOT / "raw" / "m46.txt"]
FIELDS = [
    "id",
    "grade",
    "strand",
    "strand_name",
    "sub_strand",
    "content_standard",
    "cs_key",
    "indicator_no",
    "description",
]  # PRD 5.1, in file order
ID = re.compile(r"B([1-6])\.([1-4])\.(\d)\.(\d)\.(\d{1,2})")
# PRD task 2.4 heuristic: a description whose last token is one of these stops mid-sentence
DANGLING = {"to", "of", "a", "the", "and", "in", "for", "with", "by", "or"}
MID, RIGHT = 30, 110  # offsets of the indicator column and the competencies column


def row(mid="", left="", right=""):
    """One pdftotext -layout line: content standard | indicator | competencies."""
    assert len(left) <= MID - 3, "left column must stay 3+ spaces clear of the indicator column"
    line = left.ljust(MID) + mid
    if right:
        assert len(line) <= RIGHT - 3
        line = line.ljust(RIGHT) + right
    return line.rstrip()


def cont(text, **kw):
    """A wrapped line, indented to the description offset like most of the PDF."""
    return row(" " * 11 + text, **kw)


def parse_lines(tmp_path, lines, name="curriculum.txt"):
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return pc.parse([path])


def describe(tmp_path, lines):
    recs = parse_lines(tmp_path, lines)
    return recs[0]["description"]


def dangling_rate(recs):
    return sum(r["description"].split()[-1].lower() in DANGLING for r in recs) / len(recs)


# ---------------------------------------------------------------- code line


def test_record_schema_matches_prd(tmp_path):
    (rec,) = parse_lines(tmp_path, [row("B4.1.1.1.2 Read and write the made-up numbers in figures and in words")])
    assert list(rec) == FIELDS
    assert rec == {
        "id": "B4.1.1.1.2",
        "grade": 4,
        "strand": 1,
        "strand_name": "Number",
        "sub_strand": 1,
        "content_standard": "B4.1.1.1",
        "cs_key": "1.1.1",
        "indicator_no": 2,
        "description": "Read and write the made-up numbers in figures and in words",
    }


@pytest.mark.parametrize("strand, name", [(1, "Number"), (2, "Algebra"), (3, "Geometry and Measurement"), (4, "Data")])
def test_strand_names(tmp_path, strand, name):
    (rec,) = parse_lines(tmp_path, [row(f"B2.{strand}.1.1.1 Some indicator text")])
    assert (rec["strand"], rec["strand_name"]) == (strand, name)


@pytest.mark.parametrize(
    "written, code, number",
    [
        ("B4.1.1.1.5", "B4.1.1.1.5", 5),
        ("B4. 1.1.1.5", "B4.1.1.1.5", 5),  # space after the grade
        ("B1.3.2.1. 1", "B1.3.2.1.1", 1),  # space before the indicator number
        ("B4.1.1.1.6.", "B4.1.1.1.6", 6),  # trailing dot after the code
        ("B4. 1.1.3.3.", "B4.1.1.3.3", 3),  # both
        ("B5.2.1.1.12", "B5.2.1.1.12", 12),  # two-digit indicator number
        ("B5.2.1.1.07", "B5.2.1.1.7", 7),  # leading zero is normalised
    ],
)
def test_code_variants(tmp_path, written, code, number):
    (rec,) = parse_lines(tmp_path, [row(f"{written} Skip count forwards and backwards")])
    assert rec["id"] == code and rec["indicator_no"] == number
    assert rec["description"] == "Skip count forwards and backwards"


def test_content_standard_code_alone_is_not_an_indicator(tmp_path):
    assert parse_lines(tmp_path, [row("Count objects in a set", left="B1.1.1.1")]) == []


def test_description_is_cut_at_the_next_layout_column(tmp_path):
    lines = [row("B1.1.1.1.1 Count objects in a set", left="B1.1.1.1", right="Learners develop:")]
    assert describe(tmp_path, lines) == "Count objects in a set"


def test_single_and_double_spaces_stay_inside_the_description(tmp_path):
    assert describe(tmp_path, [row("B1.1.1.1.1 Count  objects in  a set")]) == "Count objects in a set"


def test_trailing_punctuation_is_stripped(tmp_path):
    assert describe(tmp_path, [row("B1.1.1.1.1 Count objects in a set.;:")]) == "Count objects in a set"


def test_duplicate_codes_keep_the_first_occurrence(tmp_path):
    recs = parse_lines(
        tmp_path,
        [
            row("B1.1.1.1.1 First wording of the indicator"),
            row("B1.1.1.1.2 Another indicator"),
            row("B1.1.1.1.1 Second wording of the indicator"),
        ],
    )
    assert [r["id"] for r in recs] == ["B1.1.1.1.1", "B1.1.1.1.2"]
    assert recs[0]["description"] == "First wording of the indicator"


def test_duplicate_codes_across_files_keep_the_first_file(tmp_path):
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text(row("B4.1.1.3.2 Wording in the first file") + "\n", encoding="utf-8")
    b.write_text(row("B4.1.1.3.2 Wording in the second file") + "\n", encoding="utf-8")
    (rec,) = pc.parse([a, b])
    assert rec["description"] == "Wording in the first file"


def test_descriptions_shorter_than_four_characters_are_skipped(tmp_path):
    recs = parse_lines(
        tmp_path,
        [
            row("B1.1.1.1.1 abc"),  # 3 characters: skipped
            row("B1.1.1.1.2 abcd"),  # 4 characters: kept
            row("B1.1.1.1.3 a."),  # punctuation does not count
            cont("long enough continuation text"),  # and a wrapped line does not rescue it
        ],
    )
    assert [(r["id"], r["description"]) for r in recs] == [("B1.1.1.1.2", "abcd")]


def test_a_skipped_short_occurrence_does_not_block_a_later_one(tmp_path):
    recs = parse_lines(tmp_path, [row("B1.1.1.1.1 ab"), row("B1.1.1.1.1 Count objects in a set")])
    assert [r["description"] for r in recs] == ["Count objects in a set"]


def test_output_is_sorted_numerically(tmp_path):
    codes = [
        "B2.1.1.1.1",
        "B1.2.1.1.1",
        "B1.1.2.1.1",
        "B1.1.1.2.1",
        "B1.1.1.1.10",
        "B1.1.1.1.2",
        "B1.1.1.1.1",
        "B1.1.1.9.1",
        "B1.1.9.1.1",
        "B1.4.1.1.1",
        "B6.1.1.1.1",
    ]
    recs = parse_lines(tmp_path, [row(f"{c} Some indicator text") for c in codes])
    assert [r["id"] for r in recs] == [
        "B1.1.1.1.1",
        "B1.1.1.1.2",
        "B1.1.1.1.10",  # 10 after 2: numeric, not lexical
        "B1.1.1.2.1",
        "B1.1.1.9.1",
        "B1.1.2.1.1",
        "B1.1.9.1.1",
        "B1.2.1.1.1",
        "B1.4.1.1.1",
        "B2.1.1.1.1",
        "B6.1.1.1.1",
    ]


def test_cli_writes_json_and_prints_the_summary_line(tmp_path):
    src, out = tmp_path / "in.txt", tmp_path / "out.json"
    src.write_text(
        "\n".join(
            [
                row("B1.1.1.1.1 Count objects in a set"),
                row("B2.1.1.1.1 Count objects “up to 1000”"),
                row("B2.1.1.1.2 Compare two numbers"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    done = subprocess.run(
        [sys.executable, "-m", "kwan.curriculum.parse", str(src), str(out)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert done.stdout.strip() == "3 Counter({2: 2, 1: 1})"
    assert json.loads(out.read_text(encoding="utf-8")) == pc.parse([src])
    assert "“up to 1000”" in out.read_text(encoding="utf-8")  # written as UTF-8, not escaped


def test_cli_without_an_output_file_writes_under_build_not_data(tmp_path, monkeypatch, capsys):
    """The records carry NaCCA text (decision D5): the default output is build/curriculum/, never data/."""
    src = tmp_path / "in.txt"
    src.write_text(row("B1.1.1.1.1 Count objects in a set") + "\n", encoding="utf-8")
    monkeypatch.setenv("KWAN_DATA_DIR", str(tmp_path / "data"))
    assert pc.default_out() == tmp_path / "build" / "curriculum" / "math.indicators.json"
    recs = pc.main([str(src)])
    assert json.loads(pc.default_out().read_text(encoding="utf-8")) == recs == pc.parse([src])
    assert not (tmp_path / "data").exists()
    assert capsys.readouterr().out.strip() == "1 Counter({1: 1})"
    assert "data/indicators.json" not in (pc.__doc__ or "") and "fallback" in (pc.__doc__ or "")


# ---------------------------------------------------------------- continuation (PRD task 2.4)


def test_wrapped_line_at_the_description_offset_is_appended(tmp_path):
    lines = [
        row("B1.1.1.1.1 Use number names and how to count to", left="B1.1.1.1", right="Learners develop:"),
        cont("find out “how many?”", right="Problem Solving Skills;"),
    ]
    assert describe(tmp_path, lines) == "Use number names and how to count to find out “how many?”"


def test_wrapped_line_at_the_code_offset_is_appended(tmp_path):
    lines = [row("B1.1.1.1.1 Use number names and how to count to"), row("find out how many")]
    assert describe(tmp_path, lines) == "Use number names and how to count to find out how many"


def test_other_columns_on_a_wrapped_line_are_ignored(tmp_path):
    lines = [
        row("B1.1.1.1.4 Use comparative language to describe the relationship between"),
        row(
            "quantities up to 100 using place value",
            left="Describe numbers and the",
            right="Critical Thinking; Justification",
        ),
        row("E.g. 1. Use matching to solve problems", left="relationship between"),
    ]
    assert describe(tmp_path, lines) == (
        "Use comparative language to describe the relationship between quantities up to 100 using place value"
    )


def test_several_wrapped_lines(tmp_path):
    lines = [
        row("B1.4.1.1.2 Organise a given set of data into three categories,"),
        row("find the total number of data points and determine how many"),
        row("are in each category and compare the number in any two"),
        row("categories."),
    ]
    assert describe(tmp_path, lines) == (
        "Organise a given set of data into three categories, find the total number of data points "
        "and determine how many are in each category and compare the number in any two categories"
    )


def test_inner_punctuation_is_kept(tmp_path):
    lines = [row("B1.1.2.1.1 Understand subtraction as finding how many left;"), row("numbers 0 to 20.")]
    assert describe(tmp_path, lines) == "Understand subtraction as finding how many left; numbers 0 to 20"


def test_a_word_broken_with_a_hyphen_is_rejoined(tmp_path):
    lines = [
        row("B1.3.1.1.1 Distinguish between attributes that define a two-"),
        cont("dimensional figure and a three-dimensional figure"),
    ]
    assert describe(tmp_path, lines) == (
        "Distinguish between attributes that define a two-dimensional figure and a three-dimensional figure"
    )


def test_a_spaced_dash_is_not_treated_as_a_broken_word(tmp_path):
    lines = [row("B2.3.3.1.1 Measure lengths, capacities or mass -"), row("directly or indirectly")]
    assert describe(tmp_path, lines) == "Measure lengths, capacities or mass - directly or indirectly"


STOPS = {
    "next indicator code": row("B1.1.1.1.2 Compare two sets of objects"),
    "next code with a spaced variant": row("B1. 1.1.1.2. Compare two sets of objects"),
    "blank line": "",
    "whitespace-only line": "      ",
    "E.g. line": row("E.g. 1. Count the objects in the basket"),
    "E.g. line without a space": row("E.g.1. Count the objects in the basket"),
    "E.g line with a stray space": row("E.g   .1 Use analogue clocks to tell time"),
    "Eg. line": row("Eg. count the objects in the basket"),
    "Example line": row("Example: count the objects in the basket"),
    "Note line": row("(Note: The focus is on developing strategies)"),
    "NB line": row("NB: exclude notation here"),
    "dash bullet": row("- walk round the classroom"),
    "minus bullet": row("− skip count up by one or two groups"),
    "roman list item": row("ii. second item of a list"),
    "bracketed list item": row("(i) first item of a list"),
    "numbered list item": row("1. picking a black bottle top is impossible"),
    "code-like start": row("B1.1.1.2 another content standard"),
    "copyright footer": "© NaCCA, Ministry of Education 2019",
    "table header": " CONTENT STANDARDS" + " " * 40 + "INDICATORS AND EXEMPLARS",
    "table header, second row": " " * RIGHT + "AND CORE COMPETENCIES",
    "strand banner": " " * 60 + "Strand 2: ALGEBRA",
    "sub-strand banner": " " * 50 + "Sub-strand 1: Patterns and Relationships",
    "form feed": "\f" + row("and by shape"),
    "page number": " " * 170 + "12",
    "formula glyphs": cont("𝟏𝟏 𝟖𝟖"),
    "another column running into the indicator column": "x" * 20 + " and text that crosses the column edge",
}


@pytest.mark.parametrize("stop", STOPS.values(), ids=STOPS.keys())
def test_continuation_stops(tmp_path, stop):
    lines = [
        row("B1.1.1.1.1 Sort a set of objects by", left="B1.1.1.1", right="Learners develop:"),
        cont("colour and size"),
        stop,
        cont("and this line must never be appended"),
    ]
    assert describe(tmp_path, lines) == "Sort a set of objects by colour and size"


def test_example_right_after_the_code_line_leaves_the_description_alone(tmp_path):
    lines = [
        row("B4.1.1.3.5 Generate and analyse made-up patterns in bead numbers"),
        row("E.g. 1 Put learners into convenient groups"),
        cont("and provide each group with bottle tops"),
    ]
    assert describe(tmp_path, lines) == "Generate and analyse made-up patterns in bead numbers"


def test_text_far_to_the_right_of_the_indicator_offset_is_not_appended(tmp_path):
    lines = [
        row("B6.4.1.1.3 Use a given table of values to interpret and"),
        row(" " * 40 + "draw conclusions", left="Create, label, and"),
        row("E.g. 1. Ask learners to draw more line graphs"),
    ]
    assert describe(tmp_path, lines) == "Use a given table of values to interpret and"


def test_competency_text_is_never_appended(tmp_path):
    lines = [
        row("B1.1.1.1.1 Count objects in a set", right="Learners develop:"),
        row("", left="Describe numbers and the", right="problem solving skills; critical thinking"),
        row("", right="justification of ideas"),
    ]
    assert describe(tmp_path, lines) == "Count objects in a set"


def test_lines_with_text_only_in_other_columns_are_skipped(tmp_path):
    lines = [
        row("B1.1.2.4.2 Solve one-step made-up problems involving", left="B1.1.2.4", right="Learners develop:"),
        row("adding and taking away beads up to 20 using a variety of"),
        row("", left="Apply tricks for adding", right="Problem Solving Skills; Critical"),
        row("tricks"),
        row("", left="and taking away to 20", right="Thinking; Justification of Ideas;"),
        row("E.g. 1. Use a variety of tricks"),
    ]
    assert describe(tmp_path, lines) == (
        "Solve one-step made-up problems involving adding and taking away beads up to 20 using a variety of tricks"
    )


@pytest.mark.parametrize(
    "after_gap", ["Display 2D cut out shapes and have learners sort them", "100 objects in a line", "(see the chart)"]
)
def test_after_a_skipped_line_only_a_lower_case_start_continues(tmp_path, after_gap):
    lines = [
        row("B1.3.1.1.1 Distinguish between attributes that define a shape"),
        row("", left="dimensional shapes and", right="Critical Thinking;"),
        row(after_gap),
    ]
    assert describe(tmp_path, lines) == "Distinguish between attributes that define a shape"


def test_too_many_skipped_lines_stop_the_continuation(tmp_path):
    gap = row("", left="Describe numbers", right="Critical Thinking;")
    start = row("B1.1.1.1.1 Count objects in sets of")
    assert (
        describe(tmp_path, [start] + [gap] * pc.MAX_GAP + [row("ten and more")])
        == "Count objects in sets of ten and more"
    )
    assert describe(tmp_path, [start] + [gap] * (pc.MAX_GAP + 1) + [row("ten and more")]) == "Count objects in sets of"


def test_a_new_sentence_after_a_finished_one_is_not_appended(tmp_path):
    lines = [
        row("B4.1.4.1.2 Round decimals to the nearest tenth."),
        row("Learners explain the rule for rounding decimals"),
    ]
    assert describe(tmp_path, lines) == "Round decimals to the nearest tenth"


def test_an_inline_example_is_cut_with_its_bracket(tmp_path):
    lines = [
        row("B3.1.3.1.2 Understand that fractions can be used to"),
        row("represent parts of a group of objects or distances on a number"),
        row("line [Read and write fractions using words. (E.g. one-half, two halves,"),
        row("fifths etc.)]"),
    ]
    assert describe(tmp_path, lines) == (
        "Understand that fractions can be used to represent parts of a group of objects or distances on a number line"
    )


def test_lower_case_eg_inside_a_sentence_is_part_of_the_indicator(tmp_path):
    lines = [row("B4.1.3.1.1 Locate a unit fraction, e.g. one-eighth, on a number"), cont("line (e.g. from 0 to 1)")]
    assert describe(tmp_path, lines) == "Locate a unit fraction, e.g. one-eighth, on a number line (e.g. from 0 to 1)"


@pytest.mark.parametrize(
    "first, rest, expected",
    [
        (
            "Recognise Ghanaian coins by name, including one pesewa,",
            "five pesewas and one cedi",
            "Recognise Ghanaian coins by name, including one pesewa, five pesewas and one cedi",
        ),
        (
            "Use different denominations of money to buy and",
            "give change",
            "Use different denominations of money to buy and give change",
        ),
    ],
)
def test_one_blank_line_is_bridged_when_the_text_is_unfinished(tmp_path, first, rest, expected):
    lines = [row(f"B1.1.4.1.1 {first}", right="Learners develop:"), "", row(rest), "", row("and never this")]
    assert describe(tmp_path, lines) == expected


def test_blank_line_bridge_is_narrow(tmp_path):
    first = row("B1.1.4.1.1 Recognise Ghanaian coins by name, including one pesewa,")
    kept = "Recognise Ghanaian coins by name, including one pesewa,"
    assert describe(tmp_path, [first, "", "", row("five pesewas and one cedi")]) == kept  # two blank lines
    assert describe(tmp_path, [first, "", row("Display the various coins")]) == kept  # upper-case start
    assert describe(tmp_path, [first, "", row("E.g. 1. display the coins")]) == kept  # example
    assert describe(tmp_path, [first, "", row("B1.1.4.1.2 five pesewas")]) == kept  # next code


@pytest.mark.parametrize(
    "text, expected",
    [
        ("count to", True),
        ("including one pesewa,", True),
        ("notes an", True),
        ("and The", True),
        ("find out how many", False),
        ("up to 100", False),
        ("", False),
    ],
)
def test_unfinished(text, expected):
    assert pc.unfinished(text) is expected


def test_continuation_never_reads_past_the_cap(tmp_path):
    lines = [row("B1.1.1.1.1 Count")] + [row(f"line {i} and") for i in range(pc.MAX_LINES + 5)]
    assert describe(tmp_path, lines).count("line") == pc.MAX_LINES


def test_continuation_of_a_duplicate_occurrence_is_not_used(tmp_path):
    lines = [
        row("B1.1.1.1.1 Count objects in a set"),
        "",
        row("B1.1.1.1.1 Count objects in a set"),
        row("of up to one hundred things"),
    ]
    assert [r["description"] for r in parse_lines(tmp_path, lines)] == ["Count objects in a set"]


# ---------------------------------------------------------------- the frozen v1 indicator set
# tests/fixtures/indicators_v1.json: the v1 records with every NaCCA description replaced by the Kwan short
# label (decision D5). Ids, numbers and order are the parser's; the checks on the text itself run on a fresh
# parse of the raw dumps when they are present.


def test_frozen_v1_count_and_grades(indicators_v1):
    assert len(indicators_v1) == 266
    assert Counter(r["grade"] for r in indicators_v1) == {1: 24, 2: 28, 3: 39, 4: 71, 5: 63, 6: 41}


def test_frozen_v1_ids_are_unique(indicators_v1):
    assert len({r["id"] for r in indicators_v1}) == len(indicators_v1)


def test_frozen_v1_records_follow_the_schema(indicators_v1):
    for r in indicators_v1:
        assert list(r) == FIELDS, r["id"]
        m = ID.fullmatch(r["id"])
        assert m, r["id"]
        g, s, ss, cs, i = map(int, m.groups())
        assert (r["grade"], r["strand"], r["sub_strand"], r["indicator_no"]) == (g, s, ss, i), r["id"]
        assert r["strand_name"] == pc.STRANDS[s]
        assert r["content_standard"] == f"B{g}.{s}.{ss}.{cs}"
        assert r["cs_key"] == f"{s}.{ss}.{cs}"
        assert r["id"] == f"{r['content_standard']}.{i}"
        assert isinstance(r["description"], str) and r["description"].strip()


def test_frozen_v1_order(indicators_v1):
    key = lambda r: (
        r["grade"],
        r["strand"],
        r["sub_strand"],
        int(r["content_standard"].split(".")[-1]),
        r["indicator_no"],
    )
    assert indicators_v1 == sorted(indicators_v1, key=key)


# ---------------------------------------------------------------- the raw text dumps (when downloaded)


@pytest.fixture(scope="module")
def parsed():
    """A fresh parse of raw/m13.txt and raw/m46.txt (never committed: the records carry NaCCA text)."""
    if not all(p.exists() for p in RAW):
        pytest.skip("raw/m13.txt and raw/m46.txt not downloaded")
    return pc.parse(RAW)


@pytest.mark.slow
def test_reparsing_the_raw_text_gives_the_frozen_v1_set_apart_from_the_text(parsed, indicators_v1):
    without_text = lambda recs: [{k: v for k, v in r.items() if k != "description"} for r in recs]
    assert without_text(parsed) == without_text(indicators_v1)
    assert all(r["description"] for r in parsed)
    # the fixture holds labels, not the parsed sentences
    assert not any(a["description"] == b["description"] for a, b in zip(parsed, indicators_v1))


@pytest.mark.slow
def test_parsed_descriptions_are_clean(parsed):
    junk = re.compile(
        r"Learners develop|Problem Solving|Critical Thinking|Justification of|Collaborative|"
        r"Personal Development|Attention to Precision|NaCCA|©|CONT’D|\f|\s{2,}",
        re.I,
    )
    for r in parsed:
        d = r["description"]
        assert isinstance(d, str) and len(d) >= 4, r["id"]
        assert d == d.strip(" .;:"), r["id"]
        assert "E.g." not in d and not re.search(r"\bE\s?\.\s?g\b", d), r["id"]  # no exemplar text
        assert not junk.search(d), (r["id"], d)
        assert not ID.search(d), (r["id"], d)  # no swallowed indicator code
        assert len(d) <= 300, (r["id"], len(d))  # longest real indicator is under 300 characters


@pytest.mark.slow
def test_parsed_dangling_rate_is_under_ten_percent(parsed):
    assert dangling_rate(parsed) < 0.10


@pytest.mark.slow
def test_continuation_only_extends_the_code_line_text(parsed):
    """Every description still starts with what the code line alone gives (the pre-2.4 output)."""
    first_line = {}
    for path in RAW:
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = pc.IND.search(line)
                if not m:
                    continue
                code = "B{}.{}.{}.{}.{}".format(*m.groups()[:4], int(m.group(5)))
                desc = pc.clean(m.group(6))
                if code not in first_line and len(desc) >= 4:
                    first_line[code] = desc
    assert list(first_line) != [] and set(first_line) == {r["id"] for r in parsed}
    before = [{**r, "description": first_line[r["id"]]} for r in parsed]
    assert all(r["description"].startswith(b["description"]) for r, b in zip(parsed, before))
    assert dangling_rate(before) > 0.20 > 0.10 > dangling_rate(parsed)
