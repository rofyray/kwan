"""Plain data types shared by the two extraction paths (pdfplumber and pdftotext) and the parser."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TextLine:
    """One line of text inside one table column."""

    y: float  # vertical position on the page (PDF points, or the line number in a text dump)
    x: float  # where the line starts (PDF points, or the character offset in a text dump)
    text: str
    bold: bool | None = None  # None when the extraction path cannot see fonts
    x1: float = 0.0  # where the line ends
    lead: str = ""  # the bold words that open a line which then carries on in plain type


@dataclass
class PageText:
    """One page, already split into the three table columns."""

    doc: str
    page: int  # 1-based position in the PDF
    printed_page: str | None = None  # the page number printed at the foot of the page
    banners: list[str] = field(default_factory=list)  # lines above the table header (grade, strand, sub-strand)
    left: list[TextLine] = field(default_factory=list)  # content standards
    mid: list[TextLine] = field(default_factory=list)  # indicators and exemplars
    right: list[TextLine] = field(default_factory=list)  # core competencies
    has_table: bool = False


@dataclass
class Exemplar:
    """One "E.g." block or activity printed under an indicator."""

    text: str
    page: int
    ghana_context: bool = False
    marked: bool = True  # opened by an "E.g." marker; False for an unmarked activity or note


@dataclass
class Occurrence:
    """One printed indicator statement, before corrections are applied."""

    printed: str  # the code as printed, normalised ("B4. 1.1.1.5" -> "B4.1.1.1.5")
    doc: str
    page: int
    printed_page: str | None
    y: float
    description: str
    lines: list[tuple[int, TextLine]] = field(default_factory=list)  # (page, line) that follow the description
    cs_printed: str | None = None  # content-standard code printed beside it
    cs_text: str = ""
    index: int = 1  # 1-based count of this printed code within its document
