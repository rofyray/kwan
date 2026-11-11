"""The v2 curriculum parser: column streams in, indicator records out (PRD tasks 2.3 to 2.5).

Both extraction paths (`kwan.curriculum.pdf` and `kwan.curriculum.text`) produce `PageText` objects.
This module walks them in document order and finds every indicator statement, its description, the
exemplars printed under it, the content standard beside it and the strand and sub-strand names. All
document-specific knowledge comes from the `Family` configuration and the corrections file, so the same
code reads any subject that follows the same table layout.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from kwan.curriculum.corrections import Corrections
from kwan.curriculum.exemplars import HAS_WORD, GhanaTerms, clean, split_exemplars
from kwan.curriculum.families import Family
from kwan.curriculum.model import Occurrence, PageText, TextLine

MAX_DESCRIPTION_LINES = 8
PDF_MAX_GAP = 30.0  # points between two lines of one wrapped statement (pdfplumber path)
MAX_PAGES_BACK = 4  # how far back a page may reach to carry on the exemplars of an earlier content standard
COLUMN_TITLE = re.compile(r"^\W*CONTENT\s+STANDAR\w*\W*$", re.IGNORECASE)  # a table header repeated mid-page
# a description that ends like this stops mid-sentence (also the QA truncation stop-list)
STOP_WORDS = frozenset(
    "a an the and or of to in for with by as on at from that which is are be into than up e.g i.e".split()
)
LIST_START = re.compile(r"\(?(?:\d{1,2}|[ivx]{1,4}|[a-z])[.)]\s|[-•▪●–−]\s")
CONTINUED = re.compile(r"\bCONT\s?[’'`]?\s?D\b\.?", re.IGNORECASE)


@dataclass
class ParseResult:
    """Everything the parser found."""

    records: list[dict[str, Any]]  # full records (no labels yet), sorted by code
    log: list[dict[str, Any]] = field(default_factory=list)  # corrections applied, merges, warnings, errors
    sub_strand_sources: dict[str, str] = field(default_factory=dict)  # "B4.1.4" -> printed | corrections | ...

    def errors(self) -> list[dict[str, Any]]:
        """Log entries that make the result untrustworthy (duplicate ids, unknown codes, wrong grade)."""
        return [e for e in self.log if e["level"] == "error"]


def last_word(text: str) -> str:
    """The last word of `text` in small letters, without punctuation."""
    words = text.split()
    return words[-1].strip(".,;:()[]\"'“”‘’").lower() if words else ""


def unfinished(text: str) -> bool:
    """True when `text` visibly stops mid-sentence: a trailing comma or hyphen, or a trailing stop word."""
    text = text.rstrip()
    return bool(text) and (text.endswith((",", "-", "–")) or last_word(text) in STOP_WORDS)


def is_truncated(description: str) -> bool:
    """The QA truncation check (PRD task 2.8): the description is empty or its last word is in the stop-list."""
    return not description.strip() or unfinished(description)


def join_parts(parts: list[str]) -> str:
    """Join wrapped pieces with a space, except after a word broken with a hyphen."""
    text = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        glue = "" if not text or re.search(r"\w-$", text) and part[:1].islower() else " "
        text += glue + part
    return clean(text)


def _same_statement(a: str, b: str) -> bool:
    na, nb = (re.sub(r"[^a-z0-9]+", "", s.lower()) for s in (a, b))
    return bool(na) and bool(nb) and (na == nb or na.startswith(nb) or nb.startswith(na))


class _Walker:
    """State of one pass over the pages of a family."""

    def __init__(self, family: Family, corrections: Corrections, terms: GhanaTerms | None):
        self.family = family
        self.corrections = corrections
        self.terms = terms
        self.code_start = re.compile(r"^\W{0,3}(?:[A-Z]\s)?(" + family.code_re.pattern + r")")
        self.short_start = re.compile(r"^\W{0,3}(" + family.short_code_re.pattern + r")")
        self.cell_start = re.compile(r"^\W{0,3}(" + family.short_code_re.pattern + r")")
        self.records: dict[str, Occurrence] = {}
        self.codes: dict[int, str] = {}  # id(occurrence) -> canonical code
        self.printed_of: dict[str, list[str]] = {}  # canonical code -> printed codes to keep as aliases
        self.last_by_cs: dict[str, Occurrence] = {}
        self.seen: Counter[tuple[str, str]] = Counter()
        self.sub_strand_names: dict[tuple[int, int, int], str] = {}
        self.log: list[dict[str, Any]] = []
        self.grade: int | None = None
        self.strand: int | None = None
        self.current: Occurrence | None = None
        self.pending: tuple[int, str] | None = None  # sub-strand heading waiting for its first indicator

    # ---- helpers -------------------------------------------------------------------------------

    def note(self, level: str, kind: str, code: str, page: PageText, detail: str = "") -> None:
        self.log.append(
            {
                "level": level,
                "kind": kind,
                "code": code,
                "doc": page.doc,
                "page": page.page,
                "printed_page": page.printed_page,
                "detail": detail,
            }
        )

    def cs_of(self, code: str) -> str:
        return code.rsplit(".", 1)[0]

    @staticmethod
    def last_page(occ: Occurrence) -> int:
        return occ.lines[-1][0] if occ.lines else occ.page

    def heading(self, text: str, page: PageText) -> bool:
        """Handle a strand or sub-strand heading; True when `text` was one."""
        m = self.family.sub_strand_heading_re.match(text)
        if m:
            self.flush_pending()
            self.pending = (int(m.group("no")), self.tidy_name(m.group("name")))
            self.current = None
            return True
        m = self.family.strand_heading_re.match(text)
        if m:
            self.flush_pending()
            self.strand = int(m.group("no"))
            self.current = None
            return True
        return False

    @staticmethod
    def tidy_name(name: str) -> str:
        name = clean(name).strip(" .:-–")
        return re.sub(r"\s*/\s*", " / ", name)

    def flush_pending(self) -> None:
        """A heading with no indicator after it names the sub-strand number it prints, if that is free."""
        if self.pending and self.grade is not None and self.strand is not None:
            self.sub_strand_names.setdefault((self.grade, self.strand, self.pending[0]), self.pending[1])
        self.pending = None

    def cells(self, page: PageText) -> list[tuple[float, str, str]]:
        """Content-standard cells of the left column: (y, printed code, text)."""
        out: list[tuple[float, str, list[str]]] = []
        for ln in page.left:
            if COLUMN_TITLE.match(ln.text):
                continue
            m = self.cell_start.match(ln.text)
            if m:
                rest = ln.text[m.end() :].strip(" .:`")
                out.append((ln.y, self.family.short_code(m), [rest] if rest else []))
            elif out:
                out[-1][2].append(ln.text)
        cells = []
        for y, code, parts in out:
            text = clean(CONTINUED.sub(" ", " ".join(parts))).strip(" .`")
            cells.append((y, code, text))
        return cells

    # ---- statements ----------------------------------------------------------------------------

    def description(self, lines: list[TextLine], i: int, rest: str) -> tuple[str, int]:
        """The statement that starts on lines[i] (`rest` is its text after the code) and the lines it uses."""
        fam = self.family
        first = lines[i]
        parts = [rest] if rest else []
        last_y = first.y
        j = i + 1
        while j < len(lines) and len(parts) < MAX_DESCRIPTION_LINES:
            ln = lines[j]
            text = ln.text.strip()
            m = fam.exemplar_re.match(text)
            if (m and m.end() > 0) or self.code_start.match(text) or self.short_start.match(text):
                break
            if fam.sub_strand_heading_re.match(text) or fam.header_anchor_re.search(text):
                break
            joined = join_parts(parts)
            gap = ln.y - last_y
            if first.bold is not None:  # pdfplumber path: the statement is printed in bold
                if gap > PDF_MAX_GAP:
                    break
                if not ln.bold and not HAS_WORD.search(text):
                    j += 1  # a formula fragment or table cell that sits between two lines of the statement
                    continue
                if first.bold and not ln.bold:
                    if ln.lead and parts:  # the statement ends inside this line, then plain text follows
                        parts.append(ln.lead)
                        lines[j] = TextLine(y=ln.y, x=ln.x, text=text[len(ln.lead) :].strip(), bold=False, x1=ln.x1)
                    break
                if not first.bold and parts and (ln.bold or not (unfinished(joined) or text[:1].islower())):
                    break
            else:  # pdftotext path: no fonts, so go by blank lines and sentence shape
                if gap > 2 or (gap > 1 and parts and not unfinished(joined)):
                    break
                if LIST_START.match(text) or not re.search(r"[A-Za-z]", text):
                    break
                if parts and joined.endswith((".", "?", "!")) and text[:1].isupper():
                    break
                if parts and gap > 1 and not text[:1].islower():
                    break
            parts.append(text)
            last_y = ln.y
            j += 1
        # wordless lines skipped at the very end were not part of the statement: hand them back
        while j > i + 1 and not lines[j - 1].bold and not HAS_WORD.search(lines[j - 1].text) and first.bold is not None:
            if lines[j - 1].text.strip() in parts[-1:]:
                break
            j -= 1
        desc = join_parts(parts).strip(" ;:")
        if first.bold is not None and first.x1 and unfinished(desc):
            # the statement ends in a formula that the PDF places a little lower: take it from where the line stopped
            for k in range(j, min(j + 4, len(lines))):
                cand = lines[k]
                if not HAS_WORD.search(cand.text) and re.search(r"\d", cand.text) and -6 <= cand.x - first.x1 <= 40:
                    if cand.y - first.y <= PDF_MAX_GAP:
                        desc = f"{desc} {cand.text.strip()}"
                        del lines[k]
                    break
        return desc, j - i

    def statement(
        self,
        printed: str,
        normal: str | None,
        desc: str,
        line: TextLine,
        page: PageText,
        cells: list[tuple[float, str, str]],
    ) -> str:
        """Register one printed statement, applying corrections and merging repeats.

        Returns what happened: "new", "merged", "dropped" (not an indicator) or "error".
        """
        self.seen[(page.doc, printed)] += 1
        index = self.seen[(page.doc, printed)]
        corr = self.corrections.find(printed, page.doc, index)
        if corr is None and normal is None:
            return "dropped"
        action = corr.action if corr else "recode"
        code = (corr.corrected if corr and corr.corrected else normal) or ""
        if corr:
            detail = f"printed {printed} (use {index}): {corr.reason}"
            self.note("info", f"correction:{action}", code or printed, page, detail)
        if action == "drop_heading":
            return "dropped"
        if action == "merge":
            target = self.records.get(code)
            if target is None:
                self.note("error", "merge_target_missing", code, page, f"printed {printed}")
                return "error"
            self.current = target
            return "merged"
        try:
            grade, strand, sub_strand, _, _ = self.family.split_code(code)
        except ValueError:
            self.note("error", "bad_code", code, page, f"printed {printed}")
            return "error"
        if code in self.records:
            first = self.records[code]
            if _same_statement(first.description, desc):
                self.note("info", "repeat_merged", code, page, f"same statement as on page {first.page}")
                self.current = first
                return "merged"
            self.note(
                "error", "duplicate_id", code, page, f"a different statement already has this code (page {first.page})"
            )
            self.current = None
            return "error"
        if self.grade is not None and grade != self.grade:
            self.note("error", "grade_mismatch", code, page, f"found in the Basic {self.grade} part of the document")
        cell = None
        if cells:
            slack = 40 if line.bold is not None else 3  # points on the PDF path, lines on the text path
            before = [c for c in cells if c[0] <= line.y + slack]
            cell = before[-1] if before else cells[0]
        occ = Occurrence(
            printed=printed,
            doc=page.doc,
            page=page.page,
            printed_page=page.printed_page,
            y=line.y,
            description=desc,
            cs_printed=cell[1] if cell else None,
            cs_text=cell[2] if cell else "",
            index=index,
        )
        if cell and cell[1] != self.cs_of(code):
            self.note("warning", "cs_mismatch", code, page, f"printed beside content standard {cell[1]}")
        if not desc:
            self.note("warning", "empty_description", code, page)
        self.records[code] = occ
        self.codes[id(occ)] = code
        self.last_by_cs[self.cs_of(code)] = occ
        if printed != code and (corr is None or corr.alias):
            self.printed_of.setdefault(code, []).append(printed)
        if self.pending:
            self.sub_strand_names.setdefault((grade, strand, sub_strand), self.pending[1])
            self.pending = None
        self.strand = strand
        self.current = occ
        return "new"

    # ---- pages ---------------------------------------------------------------------------------

    def page(self, page: PageText) -> bool:
        """Read one page; False once the end heading is reached."""
        fam = self.family
        for banner in page.banners:
            if fam.end_heading_re.search(banner) and self.grade is not None:
                return False
            m = fam.grade_heading_re.match(banner)
            if m:
                self.flush_pending()
                self.grade = int(m.group("grade"))
                self.current = None
        if self.grade is None or not page.has_table:
            return True
        for banner in page.banners:
            self.heading(banner, page)
        cells = self.cells(page)
        target = self.current
        if cells and target is not None:
            first_cs = cells[0][1]
            own = (self.cs_of(self.codes[id(target)]), target.cs_printed)
            earlier = self.last_by_cs.get(first_cs)
            if first_cs not in own and earlier is not None and page.page - self.last_page(earlier) <= MAX_PAGES_BACK:
                target = earlier  # the page carries on an earlier content standard
        lines = page.mid
        i = 0
        while i < len(lines):
            ln = lines[i]
            text = ln.text.strip()
            m = self.code_start.match(text)
            short = None if m else self.short_start.match(text)
            if short and not self.corrections.has(re.sub(r"\s+", "", short.group(1)), page.doc):
                if ln.bold:
                    code = re.sub(r"\s+", "", short.group(1))
                    self.note("warning", "unresolved_short_code", code, page, "kept as plain text")
                short = None
            hit = m or short
            if hit:
                printed = re.sub(r"\s+", "", hit.group(1))
                normal = fam.code(fam.code_re.match(hit.group(1))) if m else None  # type: ignore[arg-type]
                desc, used = self.description(lines, i, text[hit.end() :].lstrip(" .:").strip())
                status = self.statement(printed, normal, desc, ln, page, cells)
                if status in ("new", "merged"):
                    target = self.current
                elif status == "error":
                    target = None
                i += used
                continue
            if fam.header_anchor_re.search(text) or self.heading(text, page):
                if self.current is None:
                    target = None
                i += 1
                continue
            if target is not None:
                target.lines.append((page.page, ln))
            i += 1
        return True


def parse_pages(
    pages: list[PageText], family: Family, corrections: Corrections | None = None, terms: GhanaTerms | None = None
) -> ParseResult:
    """Turn the pages of one family (all documents, in order) into full indicator records.

    Records carry everything read from the document: `description`, `content_standard_text`,
    `exemplars`, `page`. They do not carry `short_label` yet (see `kwan.curriculum.build`).
    """
    corrections = corrections or Corrections()
    walker = _Walker(family, corrections, terms)
    doc = None
    stopped = False
    for page in pages:
        if page.doc != doc:  # a new document starts again at its front matter
            walker.flush_pending()
            doc, stopped = page.doc, False
            walker.grade, walker.strand, walker.current = None, None, None
        if not stopped:
            stopped = not walker.page(page)
    walker.flush_pending()

    # content-standard text: the cell printed beside the first indicator that names the same standard
    cs_text: dict[str, str] = {}
    for code, occ in walker.records.items():
        cs = walker.cs_of(code)
        if occ.cs_text and (occ.cs_printed == cs) and cs not in cs_text:
            cs_text[cs] = occ.cs_text
    for code, occ in walker.records.items():
        if occ.cs_text:
            cs_text.setdefault(walker.cs_of(code), occ.cs_text)

    canonical = set(walker.records)
    sources: dict[str, str] = {}
    records = []
    for code, occ in walker.records.items():
        grade, strand, sub_strand, cs_no, no = family.split_code(code)
        key = f"{family.code_prefix}{grade}.{strand}.{sub_strand}"
        name, source = _sub_strand_name(walker.sub_strand_names, corrections, family, grade, strand, sub_strand)
        sources[key] = source
        aliases = [code] + [p for p in dict.fromkeys(walker.printed_of.get(code, [])) if p not in canonical]
        exemplars = _dedupe(split_exemplars(occ.lines, family, terms))
        records.append(
            {
                "id": f"{family.subject}:{code}",
                "subject": family.subject,
                "code": code,
                "grade": grade,
                "strand": strand,
                "strand_name": family.strands.get(strand, f"Strand {strand}"),
                "sub_strand": sub_strand,
                "sub_strand_name": name,
                "content_standard": walker.cs_of(code),
                "cs_key": f"{strand}.{sub_strand}.{cs_no}",
                "indicator_no": no,
                "aliases": aliases,
                "description": occ.description,
                "content_standard_text": cs_text.get(walker.cs_of(code), ""),
                "exemplars": [
                    {"text": e.text, "ghana_context": e.ghana_context, "page": e.page, "marked": e.marked}
                    for e in exemplars
                ],
                "page": occ.page,
                "printed_page": occ.printed_page,
                "doc": occ.doc,
                "printed_code": occ.printed,
            }
        )
    records.sort(
        key=lambda r: (r["grade"], r["strand"], r["sub_strand"], int(r["cs_key"].split(".")[-1]), r["indicator_no"])
    )
    return ParseResult(records=records, log=walker.log, sub_strand_sources=sources)


def _dedupe(exemplars: list) -> list:
    seen: set[str] = set()
    out = []
    for e in exemplars:
        key = re.sub(r"[^a-z0-9]+", "", e.text.lower())
        if key and key not in seen:
            seen.add(key)
            out.append(e)
    return out


def _sub_strand_name(
    printed: dict[tuple[int, int, int], str], corrections: Corrections, family: Family, grade: int, strand: int, ss: int
) -> tuple[str, str]:
    """The name of a sub-strand and where it came from: the corrections file, the heading printed in the
    document, or (when the document prints none) the same sub-strand of the nearest grade."""
    key = f"{family.code_prefix}{grade}.{strand}.{ss}"
    if key in corrections.sub_strand_names:
        return corrections.sub_strand_names[key], "corrections"
    if (grade, strand, ss) in printed:
        return printed[(grade, strand, ss)], "printed"
    for other in sorted(family.grades, key=lambda g: (abs(g - grade), g)):
        if (other, strand, ss) in printed:
            return printed[(other, strand, ss)], f"same sub-strand in grade {other}"
    return f"Sub-strand {ss}", "none"
