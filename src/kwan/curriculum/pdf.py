"""Extraction path 1: pdfplumber with explicit column boxes (PRD task 2.2).

Each table page is cut into three boxes (content standards | indicators and exemplars | core
competencies) and every box is read on its own, so text from one column can never run into another.
The box edges come from the page's own table rulings when four tall vertical rulings are found, and
from the family configuration otherwise.

Reading a PDF with pdfplumber is slow, so the raw words of each page are cached under
`build/curriculum/cache/` (git-ignored), keyed by the SHA-256 of the PDF.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from kwan._paths import repo_root
from kwan.curriculum.families import Family
from kwan.curriculum.model import PageText, TextLine

CACHE_VERSION = 2
LINE_TOLERANCE = 3.6  # points between the vertical centres of two words on the same line
MATH_CHARS = re.compile("[\U0001d400-\U0001d7ff]")
DOUBLED_MATH = re.compile("([\U0001d400-\U0001d7ff])\\1")
PAGE_NUMBER = re.compile(r"^(?:\d{1,3}|[ivxlc]{1,7})$", re.IGNORECASE)

# word tuple layout in the cache
TEXT, X0, X1, TOP, BOTTOM, FONT, SIZE = range(7)


def sha256_of(path: Path | str) -> str:
    """SHA-256 of a file, in hexadecimal."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def default_cache_dir() -> Path:
    """`build/curriculum/cache` in the repository (or under the current directory)."""
    return (repo_root() or Path.cwd()) / "build" / "curriculum" / "cache"


def fix_math(text: str) -> str:
    """Undo the doubled letters of the PDF's maths font and turn its styled digits into plain ones."""
    if not MATH_CHARS.search(text):
        return text
    text = DOUBLED_MATH.sub(r"\1", text)
    return MATH_CHARS.sub(lambda m: unicodedata.normalize("NFKC", m.group()), text)


def read_words(
    pdf_path: Path | str, cache_dir: Path | str | None = None, use_cache: bool = True
) -> list[dict[str, Any]]:
    """The words and vertical rulings of every page, from the cache when it is there.

    Returns one dict per page: `{"page", "width", "height", "words": [[text, x0, x1, top, bottom, font,
    size], ...], "rulings": [[x, top, bottom], ...]}`.
    """
    pdf_path = Path(pdf_path)
    cache = Path(cache_dir) if cache_dir else default_cache_dir()
    target = cache / f"{sha256_of(pdf_path)[:16]}.v{CACHE_VERSION}.words.json"
    if use_cache and target.exists():
        return json.loads(target.read_text(encoding="utf-8"))

    import pdfplumber

    pages: list[dict[str, Any]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for number, page in enumerate(pdf.pages, start=1):
            words = page.extract_words(x_tolerance=1.5, y_tolerance=3, extra_attrs=["fontname", "size"])
            rulings = [
                [round(ln["x0"], 1), round(ln["top"], 1), round(ln["bottom"], 1)]
                for ln in page.lines
                if abs(ln["x0"] - ln["x1"]) < 1.5
            ]
            rulings += [
                [round(r["x0"], 1), round(r["top"], 1), round(r["bottom"], 1)]
                for r in page.rects
                if r["width"] < 2.5 and r["height"] > 20
            ]
            pages.append(
                {
                    "page": number,
                    "width": float(page.width),
                    "height": float(page.height),
                    "words": [
                        [
                            w["text"],
                            round(w["x0"], 1),
                            round(w["x1"], 1),
                            round(w["top"], 1),
                            round(w["bottom"], 1),
                            w["fontname"],
                            round(w["size"], 1),
                        ]
                        for w in words
                    ],
                    "rulings": rulings,
                }
            )
            page.flush_cache()
    cache.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")
    return pages


def column_edges(page: dict[str, Any], family: Family, previous: tuple[float, float] | None) -> tuple[float, float]:
    """The two inner edges of the table: (content standards | indicators, indicators | competencies).

    Uses the page's tall vertical rulings when the family allows it and four of them fit the layout,
    then the edges of the previous page, then the family's configured boxes.
    """
    fallback = previous or (family.columns["indicators"][0], family.columns["indicators"][1])
    if not family.snap_to_rulings or not page["rulings"]:
        return fallback
    spans: dict[float, float] = {}
    for x, top, bottom in sorted(page["rulings"]):
        key = next((k for k in spans if abs(k - x) <= 3), x)
        spans[key] = spans.get(key, 0.0) + (bottom - top)
    if not spans:
        return fallback
    tallest = max(spans.values())
    xs = sorted(x for x, h in spans.items() if h >= 0.6 * tallest and h >= 60)
    if len(xs) < 4:
        return fallback
    left, right = xs[0], xs[-1]
    c1 = next((x for x in xs[1:-1] if 80 <= x - left <= 185), None)
    c2 = next((x for x in reversed(xs[1:-1]) if 110 <= right - x <= 200), None)
    if c1 is None or c2 is None or c2 - c1 < 150:
        return fallback
    return c1, c2


def _is_bold(font: str) -> bool:
    return "Bold" in font


def _is_math(font: str) -> bool:
    return "Math" in font


def _lines(words: list[list[Any]]) -> list[TextLine]:
    """Group the words of one column into lines, top to bottom, left to right."""
    rows: list[list[list[Any]]] = []
    centres: list[float] = []
    for w in sorted(words, key=lambda w: ((w[TOP] + w[BOTTOM]) / 2, w[X0])):
        cy = (w[TOP] + w[BOTTOM]) / 2
        if rows and abs(cy - centres[-1]) <= LINE_TOLERANCE:
            rows[-1].append(w)
            centres[-1] += (cy - centres[-1]) / len(rows[-1])
        else:
            rows.append([w])
            centres.append(cy)
    out: list[TextLine] = []
    for row in rows:
        row.sort(key=lambda w: w[X0])
        plain = [w for w in row if not _is_math(w[FONT])]
        kept = []
        for w in row:
            if _is_math(w[FONT]) and any(min(w[X1], p[X1]) - max(w[X0], p[X0]) > 0.5 * (w[X1] - w[X0]) for p in plain):
                continue  # a formula fragment printed on top of ordinary text: misplaced by the PDF
            kept.append(w)
        if not kept:
            continue
        text = fix_math(" ".join(w[TEXT] for w in kept)).strip()
        if not text:
            continue
        inked = [w for w in plain if any(c.isalnum() for c in w[TEXT])]
        bold_chars = sum(len(w[TEXT]) for w in inked if _is_bold(w[FONT]))
        all_chars = sum(len(w[TEXT]) for w in inked)
        bold = all_chars > 0 and bold_chars >= 0.6 * all_chars
        lead = ""
        if not bold:
            run = []
            for w in kept:
                if not _is_bold(w[FONT]):
                    break
                run.append(w[TEXT])
            lead = " ".join(run) if len(run) < len(kept) else ""
        out.append(
            TextLine(
                y=round(min(w[TOP] for w in kept), 1),
                x=kept[0][X0],
                text=text,
                bold=bold,
                x1=max(w[X1] for w in kept),
                lead=lead,
            )
        )
    return out


def page_text(page: dict[str, Any], family: Family, doc: str, edges: tuple[float, float]) -> PageText:
    """Split one cached page into banners and the three column streams."""
    words = [w for w in page["words"] if w[TEXT].strip()]
    out = PageText(doc=doc, page=page["page"])
    everything = _lines(words)
    # footer: the copyright line, the printed page number and anything below them
    cut = page["height"]
    for ln in everything:
        if family.footer_re.search(ln.text):
            cut = min(cut, ln.y - 1)
    for ln in everything:
        if ln.y >= cut - 22 and ln.y > page["height"] * 0.8:
            tokens = ln.text.split()
            numbers = [t for t in tokens if PAGE_NUMBER.match(t)]
            if numbers and (len(tokens) == 1 or ln.y < cut):
                out.printed_page = numbers[-1]
                if len(tokens) == 1:
                    cut = min(cut, ln.y - 1)
    body = [w for w in words if w[TOP] < cut and not (out.printed_page == w[TEXT] and w[TOP] > page["height"] * 0.8)]
    full = _lines(body)
    # table header: the line that names the middle column, plus the header-only lines around it
    anchor = next((ln for ln in full if family.header_anchor_re.search(ln.text)), None)
    top = 0.0
    if anchor is not None:
        out.has_table = True
        block = [
            ln
            for ln in full
            if abs(ln.y - anchor.y) <= 30
            and all(t.strip(".,:").upper() in family.header_words for t in ln.text.split())
        ]
        top = min([anchor.y] + [ln.y for ln in block]) - 1
        bottom = max([anchor.y] + [ln.y for ln in block]) + 3
        out.banners = [ln.text for ln in full if ln.y < top]
        body = [w for w in body if w[TOP] >= top and not (w[TOP] <= bottom and _in_header(w, family))]
    else:
        out.has_table = len(page["rulings"]) >= 4 and len(words) > 20
        if not out.has_table:
            out.banners = [ln.text for ln in full]
            return out
    c1, c2 = edges
    cols: tuple[list[list[Any]], list[list[Any]], list[list[Any]]] = ([], [], [])
    for w in body:
        start = w[X0] + 2  # a word belongs to the column it starts in, even when it is long
        cols[0 if start < c1 else 1 if start < c2 else 2].append(w)
    out.left, out.mid, out.right = _lines(cols[0]), _lines(cols[1]), _lines(cols[2])
    return out


def _in_header(word: list[Any], family: Family) -> bool:
    return word[TEXT].strip(".,:").upper() in family.header_words


def extract_pages(
    pdf_path: Path | str, family: Family, doc: str, cache_dir: Path | str | None = None, use_cache: bool = True
) -> list[PageText]:
    """All pages of one PDF as column streams."""
    pages: list[PageText] = []
    previous: tuple[float, float] | None = None
    for raw in read_words(pdf_path, cache_dir, use_cache):
        edges = column_edges(raw, family, previous)
        if raw["rulings"]:
            previous = edges
        pages.append(page_text(raw, family, doc, edges))
    return pages
