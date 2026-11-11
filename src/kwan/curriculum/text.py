"""Extraction path 2 (fallback): `pdftotext -layout` text cut into the same three columns.

`pdftotext -layout` keeps every word at its horizontal place, so each page of the dump is a grid of
characters. This module finds, page by page, the character offset where the indicator column starts
(the leftmost indicator code or exemplar marker) and where the competencies column starts, and slices
every line at those two offsets. It sees no fonts, so the parser falls back on sentence shape to find
the end of a statement. The QA report compares this path with the pdfplumber path.

The v1 parser (`kwan.curriculum.parse`) also reads these dumps; it is kept unchanged for the v1 tests.
"""

from __future__ import annotations

import re
from pathlib import Path

from kwan.curriculum.families import Family
from kwan.curriculum.model import PageText, TextLine
from kwan.curriculum.pdf import PAGE_NUMBER, fix_math

SEGMENT = re.compile(r"\S+(?:\s{1,2}\S+)*")


def split_pages(text: str) -> list[list[str]]:
    """The lines of each page of a dump (pages end with a form feed)."""
    pages = text.split("\f")
    if pages and not pages[-1].strip():
        pages.pop()
    return [p.split("\n") for p in pages]


def _offsets(lines: list[str], family: Family, previous: tuple[int, int] | None) -> tuple[int, int] | None:
    """Character offsets where the middle and the right column start on this page."""
    start = re.compile(r"(?:" + family.code_re.pattern + r")|(?:" + family.exemplar_re.pattern.lstrip("^") + r")")
    short = re.compile(family.short_code_re.pattern)
    mids: list[int] = []
    rights: list[int] = []
    for line in lines:
        for i, m in enumerate(SEGMENT.finditer(line)):
            seg = m.group()
            hit = start.match(seg)
            if hit and hit.end() > 0 and not (i == 0 and short.match(seg) and not family.code_re.match(seg)):
                if i > 0 or m.start() > 12:
                    mids.append(m.start())
            if any(seg.startswith(w) for w in family.competency_words):
                rights.append(m.start())
    if not mids:
        return previous
    mid = min(mids)
    far = [r for r in rights if r >= mid + 40]
    right = min(far) if far else max((len(ln) for ln in lines), default=mid) + 1
    return mid, right


def page_text(lines: list[str], family: Family, doc: str, number: int, previous: tuple[int, int] | None) -> PageText:
    """One page of a dump as banners and column streams."""
    out = PageText(doc=doc, page=number)
    lines = [fix_math(ln.rstrip()) for ln in lines]
    end = len(lines)
    for i, ln in enumerate(lines):
        if family.footer_re.search(ln):
            end = i
            break
    j = end - 1
    while j >= 0 and not lines[j].strip():
        j -= 1
    if j >= 0 and PAGE_NUMBER.match(lines[j].strip()):
        out.printed_page = lines[j].strip()
        end = j
    body = lines[:end]
    anchor = next((i for i, ln in enumerate(body) if family.header_anchor_re.search(ln)), None)
    first = 0
    if anchor is not None:
        header = {anchor}
        for i in range(max(0, anchor - 2), min(len(body), anchor + 3)):
            tokens = body[i].split()
            if tokens and all(t.strip(".,:").upper() in family.header_words for t in tokens):
                header.add(i)
        out.banners = [ln.strip() for ln in body[: min(header)] if ln.strip()]
        first = max(header) + 1
    offsets = _offsets(body[first:], family, previous)
    out.has_table = anchor is not None or (offsets is not None and offsets != previous)
    if not out.has_table or offsets is None:
        if anchor is None:
            out.banners = [ln.strip() for ln in body if ln.strip()]
        out.has_table = False
        return out
    mid, right = offsets
    for y, line in enumerate(body[first:], start=first):
        if not line.strip():
            continue
        cells = (line[: max(mid - 1, 0)], line[max(mid - 1, 0) : max(right - 1, 0)], line[max(right - 1, 0) :])
        starts = (0, max(mid - 1, 0), max(right - 1, 0))
        for cell, base, stream in zip(cells, starts, (out.left, out.mid, out.right)):
            if cell.strip():
                x = base + len(cell) - len(cell.lstrip())
                text = re.sub(r"\s+", " ", cell).strip()
                stream.append(TextLine(y=float(y), x=float(x), text=text, bold=None, x1=float(x + len(cell.strip()))))
    return out


def extract_pages(text_path: Path | str, family: Family, doc: str) -> list[PageText]:
    """All pages of one `pdftotext -layout` dump as column streams."""
    text = Path(text_path).read_text(encoding="utf-8", errors="ignore")
    return pages_from_text(text, family, doc)


def pages_from_text(text: str, family: Family, doc: str) -> list[PageText]:
    """Column streams from layout text held in memory (used by tests with synthetic text)."""
    pages: list[PageText] = []
    previous: tuple[int, int] | None = None
    for number, lines in enumerate(split_pages(text), start=1):
        page = page_text(lines, family, doc, number, previous)
        pages.append(page)
    return pages
