"""Synthetic curriculum pages for the curriculum tests. Nothing here is NaCCA text."""

from kwan.curriculum.corrections import corrections_from_dict
from kwan.curriculum.families import family_from_dict
from kwan.curriculum.model import PageText, TextLine

MATH = {
    "subject": "math",
    "title": "Synthetic maths family",
    "grades": [1, 2, 3, 4, 5, 6],
    "strands": {1: "Number", 2: "Algebra", 3: "Geometry and Measurement", 4: "Data"},
    "code_prefix": "B",
    "code_pattern": r"[BP]\s?(?P<grade>[1-6])\s?\.\s?(?P<strand>[1-4])\s?\.\s?(?P<sub_strand>\d)\s?\.\s?(?P<cs>\d)\s?\.\s?(?P<no>\d{1,2})(?!\d)",
    "short_code_pattern": r"[BP]\s?(?P<grade>[1-6])\s?\.\s?(?P<strand>\d)\s?\.\s?(?P<sub_strand>\d)\s?\.\s?(?P<cs>\d)(?![\d.]*\d)",
    "grade_heading": r"^\s*BASIC\s+(?P<grade>[1-6])\s*$",
    "end_heading": r"SUBJECT\s+PANEL\s+MEMBERS",
    "header_words": ["CONTENT", "STANDARD", "STANDARDS", "INDICATORS", "AND", "EXEMPLARS", "CORE", "COMPETENCIES"],
    "footer_pattern": r"©\s*Somebody",
    "competency_words": ["Learners develop", "Critical Thinking"],
    "documents": [{"id": "doc1", "grades": [1, 2, 3]}, {"id": "doc2", "grades": [4, 5, 6]}],
}

# A made-up second subject: other prefix, other strands, other headings, other column words (PRD task 2.9).
ENGLISH = {
    "subject": "eng",
    "title": "Synthetic English family",
    "grades": [1, 2],
    "strands": {1: "Oral Language", 2: "Reading", 3: "Writing"},
    "code_prefix": "E",
    "code_pattern": r"E\s?(?P<grade>[1-2])\s?\.\s?(?P<strand>[1-3])\s?\.\s?(?P<sub_strand>\d)\s?\.\s?(?P<cs>\d)\s?\.\s?(?P<no>\d{1,2})(?!\d)",
    "short_code_pattern": r"E\s?(?P<grade>[1-2])\s?\.\s?(?P<strand>\d)\s?\.\s?(?P<sub_strand>\d)\s?\.\s?(?P<cs>\d)(?![\d.]*\d)",
    "grade_heading": r"^\s*CLASS\s+(?P<grade>[1-2])\s*$",
    "end_heading": r"THE\s+END",
    "strand_heading": r"^\s*Strand\s*(?P<no>\d)\s*:\s*(?P<name>.+)$",
    "sub_strand_heading": r"^\s*Topic\s*(?P<no>\d)\s*:\s*(?P<name>.+)$",
    "header_anchor": r"WHAT\s+PUPILS\s+DO",
    "header_words": ["STANDARD", "WHAT", "PUPILS", "DO", "SKILLS"],
    "footer_pattern": r"©\s*Nobody",
    "exemplar_marker": r"^\s*Try\s*(?P<label>\d{1,2})?\s*[.:)]\s*",
    "competency_words": ["Pupils build"],
    "documents": [{"id": "e12", "grades": [1, 2]}],
}


def math_family():
    return family_from_dict("synthetic-math", MATH)


def english_family():
    return family_from_dict("synthetic-english", ENGLISH)


def corrections(*entries, **extra):
    return corrections_from_dict({"corrections": list(entries), **extra})


def line(y, text, bold=False, x=210.0):
    """One line of the middle column as the pdfplumber path gives it."""
    return TextLine(y=float(y), x=float(x), text=text, bold=bold, x1=float(x) + 5.0 * len(text))


def page(number, mid=(), left=(), banners=(), doc="doc1", printed=None):
    """A table page. `mid` and `left` are lists of (y, text, bold) or TextLine."""

    def lines(items, x):
        return [
            it if isinstance(it, TextLine) else line(it[0], it[1], it[2] if len(it) > 2 else False, x) for it in items
        ]

    return PageText(
        doc=doc,
        page=number,
        printed_page=printed or str(number),
        banners=list(banners),
        left=lines(left, 70.0),
        mid=lines(mid, 210.0),
        has_table=True,
    )


def title_page(number, grade, doc="doc1"):
    """The page that opens a grade: only "BASIC n"."""
    return PageText(doc=doc, page=number, banners=[f"BASIC {grade}"], has_table=False)


LEFT, MID, RIGHT = 0, 30, 110


def layout_row(mid="", left="", right=""):
    """One `pdftotext -layout` line: content standard | indicator | competencies."""
    out = left.ljust(MID) + mid
    if right:
        out = out.ljust(RIGHT) + right
    return out.rstrip()
