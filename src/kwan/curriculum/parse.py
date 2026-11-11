"""The fallback parser: NaCCA B1-B6 Mathematics curriculum text (pdftotext -layout) into v1 indicator
records (bare codes with the indicator description).

The curriculum pipeline (`kwan curriculum parse`, `kwan.curriculum.build`) reads the PDFs with pdfplumber
and falls back to this parser on the pdftotext dumps. The records carry NaCCA text, so the output goes
under the git-ignored `build/curriculum/` folder, never under `data/` (PRD 5.3, decision D5).

Usage: python -m kwan.curriculum.parse raw/m13.txt raw/m46.txt [build/curriculum/math.indicators.json]
"""

import json
import re
import sys
from collections import Counter, OrderedDict
from pathlib import Path

DEFAULT_OUT_NAME = "math.indicators.json"

STRANDS = {1: "Number", 2: "Algebra", 3: "Geometry and Measurement", 4: "Data"}
# Indicator code like B4.1.1.1.2 (tolerates "B4. 1.1.1.5" and trailing dot)
IND = re.compile(r"B\s?([1-6])\s?\.\s?([1-4])\s?\.\s?(\d)\s?\.\s?(\d)\s?\.\s?(\d{1,2})\.?\s+(.*)")

# --- description continuation (PRD task 2.4) -------------------------------------------------
# The PDF is a three-column table: content standard | indicators and exemplars | competencies.
# pdftotext -layout keeps the columns apart with runs of 3+ spaces, so a line is a list of
# "segments" with a start offset. A wrapped indicator continues on the following lines in a
# segment that starts near the indicator's own code/description offset.
SEGMENT = re.compile(r"\S+(?:\s{1,2}\S+)*")
WINDOW = 2  # a continuation starts between (code offset - 2) and (description offset + 2)
MAX_GAP = 2  # lines with text only in the other columns that may sit inside a wrapped indicator
MAX_LINES = 8  # hard cap on appended lines
# exemplar / note / list-item / code starts: never part of the indicator sentence
STOP_START = re.compile(
    r"E\s?[.,]?\s?g\b|Example|\(?\s*Note\b|NB\b|B\s?[1-6]\s?\.\s?\d"
    r"|\(?(?:\d{1,2}|[ivx]{1,4}|[a-z])[.)]\s|[^\w\s(\[\"'“‘<>=+]"
)
EXAMPLE = re.compile(r"\bE\s?\.\s?g\b|\bEg\.")
MATH_GLYPH = re.compile("[\U0001d400-\U0001d7ff]")  # pieces of a typeset formula, garbled by pdftotext
# page furniture: footer, table header, strand banners, form feed
FURNITURE = re.compile(
    r"\f|©|NaCCA|CONTENT STANDARDS|INDICATORS AND EXEMPLARS|CORE COMPETENCIES"
    r"|SUBJECT SPECIFIC PRACTICES|^\s*(?i:(?:sub-)?strand)\s+\d"
)
# a description ending like this is visibly unfinished (used to bridge one blank line, see below)
DANGLING = frozenset("a an the and or of to in for with by as on at from".split())


def clean(text: str) -> str:
    # keep only the indicator column: cut at 3+ spaces (next layout column)
    text = re.split(r"\s{3,}", text.strip())[0]
    return re.sub(r"\s+", " ", text).strip(" .;:")


def unfinished(text: str) -> bool:
    """True when `text` stops mid-sentence: trailing comma or a trailing connector word."""
    words = text.split()
    return bool(words) and (words[-1].endswith(",") or words[-1].lower() in DANGLING)


def _before_example(text: str) -> str:
    """Cut `text` at an inline "E.g." and drop the bracket the example was opened in."""
    m = EXAMPLE.search(text)
    if not m:
        return text
    text, opened = text[: m.start()], []
    for pos, ch in enumerate(text):
        if ch in "([":
            opened.append(pos)
        elif ch in ")]" and opened:
            opened.pop()
    return text[: opened[0]] if opened else text


def continuation(lines, i, col, descol, first=""):
    """Wrapped description text that follows the indicator found on lines[i].

    `col` is the offset of the indicator code, `descol` the offset of its description and
    `first` the description text on the code line. Returns the list of pieces to append.

    Scanning stops at a blank line, the next indicator code, an "E.g." / note / list line,
    page furniture, or anything ambiguous: when in doubt it stops early rather than append
    text from another column. Lines that only have text in the other columns are skipped
    (at most MAX_GAP in a row), and one blank line is bridged when the text so far is
    visibly unfinished; after either, only a piece starting in lower case is accepted.
    """
    lo, hi = col - WINDOW, descol + WINDOW
    parts, prev, gap, blank = [], first, 0, 0
    for line in lines[i + 1 :]:
        line = line.rstrip("\n")
        if len(parts) >= MAX_LINES or FURNITURE.search(line) or IND.search(line):
            break
        if not line.strip():
            if blank or not unfinished(prev):
                break
            blank = 1
            continue
        segs = [(m.start(), m.end(), m.group()) for m in SEGMENT.finditer(line)]
        cand = next((text for start, _, text in segs if lo <= start <= hi), None)
        if cand is None:
            # text of another column runs into the indicator column, or a lone page number: ambiguous
            if any(start < lo and end > col for start, end, _ in segs) or line.strip().isdigit():
                break
            gap += 1  # only the other columns have text on this line
            if gap > MAX_GAP:
                break
            continue
        if STOP_START.match(cand) or MATH_GLYPH.search(cand):
            break
        if (gap or blank) and not cand[0].islower():
            break  # after a skipped line only an obvious mid-sentence wrap is accepted
        if cand[0].isupper() and prev.rstrip().endswith((".", "?", "!")):
            break  # previous sentence is finished and a new one starts
        cut = _before_example(cand).strip()
        if cut:
            parts.append(cut)
        if cut != cand:
            break
        prev, gap, blank = cand, 0, 0
    return parts


def join(parts) -> str:
    """Join wrapped pieces with a space, except after a word broken with a hyphen ("two-" + "dimensional")."""
    text = ""
    for part in parts:
        glue = "" if re.search(r"\w-$", text) or not text else " "
        text += glue + part
    return text


def parse(paths):
    out = OrderedDict()
    for p in paths:
        with open(p, encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        for n, line in enumerate(lines):
            for m in IND.finditer(line):
                g, s, ss, cs, i, rest = m.groups()
                code = f"B{g}.{s}.{ss}.{cs}.{int(i)}"
                desc = clean(rest)
                if code in out or len(desc) < 4:  # decided on the code line only, as before
                    continue
                first = re.split(r"\s{3,}", rest.strip())[0]
                desc = clean(join([first] + continuation(lines, n, m.start(), m.start(6), first)))
                out[code] = {
                    "id": code,
                    "grade": int(g),
                    "strand": int(s),
                    "strand_name": STRANDS[int(s)],
                    "sub_strand": int(ss),
                    "content_standard": f"B{g}.{s}.{ss}.{cs}",
                    "cs_key": f"{s}.{ss}.{cs}",  # grade-agnostic key used for vertical edges
                    "indicator_no": int(i),
                    "description": desc,
                }
    return sorted(
        out.values(),
        key=lambda r: (
            r["grade"],
            r["strand"],
            r["sub_strand"],
            int(r["content_standard"].split(".")[-1]),
            r["indicator_no"],
        ),
    )


def default_out() -> Path:
    """Where the records go when no output file is named: `build/curriculum/math.indicators.json`."""
    from kwan._paths import data_dir

    return data_dir().parent / "build" / "curriculum" / DEFAULT_OUT_NAME


def main(argv=None):
    """Parse the text dumps and write the indicators file: `TEXT [TEXT ...] [INDICATORS_JSON]`.

    The last argument is the output file when it ends in `.json`; otherwise every argument is a text
    dump and the records go to `default_out()`."""
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[-1].lower().endswith(".json"):
        sources, out = argv[:-1], Path(argv[-1])
    else:
        sources, out = argv, default_out()
    recs = parse(sources)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(recs, f, indent=1, ensure_ascii=False)
    print(len(recs), Counter(r["grade"] for r in recs))
    return recs


if __name__ == "__main__":
    main()
