"""Exemplars (PRD task 2.5): cut the text under an indicator into "E.g." blocks and flag Ghanaian context."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from kwan.curriculum.families import Family
from kwan.curriculum.model import Exemplar, TextLine

CONTEXT_TERMS = Path(__file__).resolve().parent.parent / "context" / "ghana_terms.txt"  # owned by kwan.context
EXTRA_TERMS = Path(__file__).with_name("ghana_terms_extra.txt")  # symbols and a small fallback list
HAS_WORD = re.compile(r"[A-Za-z]{2}")
MIN_ACTIVITY_WORDS = 5  # an unmarked block shorter than this is a stray label, not an activity


def read_terms(path: Path | str) -> list[str]:
    """Terms of one list file: one per line, `#` starts a comment."""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


class GhanaTerms:
    """Finds Ghanaian context words (cedi, pesewa, local names, places, foods) in a piece of text.

    Matching follows the rules written at the top of `kwan/context/ghana_terms.txt`: whole words only;
    a term that starts with a capital letter must match exactly (a following 's is allowed); a term in
    small letters matches in any case and with a plural s or es. A term without letters or digits
    (the cedi sign) matches anywhere.
    """

    def __init__(self, terms: Iterable[str]):
        names, words, symbols = [], [], []
        for term in dict.fromkeys(t.strip() for t in terms if t.strip()):
            if not any(c.isalnum() for c in term) or not term[-1].isalnum() or not term[0].isalnum():
                symbols.append(term)
            elif term[0].isupper():
                names.append(term)
            else:
                words.append(term)

        def alternation(items: list[str]) -> str:
            return "|".join(re.escape(t) for t in sorted(items, key=len, reverse=True))

        self.terms = names + words + symbols
        self._names = re.compile(rf"(?<![\w])(?:{alternation(names)})(?:['’]s)?(?![\w])") if names else None
        self._words = (
            re.compile(rf"(?<![\w])(?:{alternation(words)})(?:es|s)?(?![\w])", re.IGNORECASE) if words else None
        )
        self._symbols = re.compile(alternation(symbols)) if symbols else None

    def found(self, text: str) -> list[str]:
        """Every term found in `text`, in order of appearance."""
        hits: list[tuple[int, str]] = []
        for rx in (self._names, self._words, self._symbols):
            if rx is not None:
                hits += [(m.start(), m.group()) for m in rx.finditer(text)]
        return [t for _, t in sorted(hits)]

    def matches(self, text: str) -> bool:
        """True when `text` uses at least one term."""
        return any(rx is not None and rx.search(text) for rx in (self._names, self._words, self._symbols))


def load_terms(paths: Iterable[Path | str] | None = None) -> GhanaTerms:
    """The term list: the context package's file when it exists, plus this package's extra list."""
    files = list(paths) if paths is not None else [CONTEXT_TERMS, EXTRA_TERMS]
    terms: list[str] = []
    for f in files:
        if Path(f).exists():
            terms += read_terms(f)
    return GhanaTerms(terms)


def clean(text: str) -> str:
    """Collapse runs of white space."""
    return re.sub(r"\s+", " ", text).strip()


def split_exemplars(
    lines: Iterable[tuple[int, TextLine]], family: Family, terms: GhanaTerms | None = None
) -> list[Exemplar]:
    """Cut the lines printed under an indicator into exemplars.

    A line that starts with an exemplar marker ("E.g. 2.", "Eg.", "Example 3") opens a new exemplar and
    the lines after it belong to it, across page breaks. Text before the first marker is kept as one
    activity when it has at least `MIN_ACTIVITY_WORDS` words. Lines without any word (cells of a
    number table, stray symbols) are dropped.
    """
    blocks: list[tuple[int, bool, list[str]]] = []  # (page, opened by a marker, parts)
    for page, line in lines:
        text = clean(line.text)
        m = family.exemplar_re.match(text)
        if m and m.end() > 0:
            blocks.append((page, True, [text[m.end() :]]))
        elif not HAS_WORD.search(text):
            continue
        elif blocks:
            blocks[-1][2].append(text)
        else:
            blocks.append((page, False, [text]))
    out = []
    for page, marked, parts in blocks:
        text = clean(" ".join(parts))
        if not HAS_WORD.search(text) or (not marked and len(text.split()) < MIN_ACTIVITY_WORDS):
            continue
        out.append(Exemplar(text=text, page=page, ghana_context=bool(terms and terms.matches(text)), marked=marked))
    return out
