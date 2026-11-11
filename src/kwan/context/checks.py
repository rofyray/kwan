"""Automatic checks for practice settings and short labels (PRD 8.3).

Every "Try it in Ghana" setting must pass these before it can be merged, imported from a review
sheet or shipped. The checks catch what a program can catch. They do not replace the teacher review:
a wrong price, a town in the wrong region or a stereotype written in harmless words needs a person.

Checks on one setting
    form           one line, no stray spaces, starts with a capital or a digit, ends with . ? or !
    length         at most 220 characters
    em_dash        no em dash
    reading_level  Flesch-Kincaid grade at most 5 for Basic 1 to 3 and at most 7 for Basic 4 to 6
    banned         no term from `banned_terms.txt` (brands, unsafe topics, stereotypes, foreign units)
    currency       money written `GH₵ 7.50`; a named coin or note must be one that exists
    ghana_term     at least one term from `ghana_terms.txt`
    duplicate      not a near copy of another setting of the same indicator

How the reading level is worked out
    grade = 0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59

    Sentences end at ".", "?" or "!" followed by a space and anything but a small letter, so "GH₵ 7.50",
    "10.15 a.m. and" and "3.4 m" do not end a sentence.

    Words are runs of letters (a hyphenated word is one word). A number is one word too: it adds to
    the length of its sentence. "GH₵ 7.50" is one number.

    Syllables are counted by `syllables()`: groups of vowels, with corrections for silent endings
    ("make", "shared", "plates"), for "-le" and "-tre" endings ("table", "metre"), for "ia" and
    "-ing" after a vowel ("triangle", "playing"), and a short table of maths words the rules get wrong.

    Names must not push the grade up. A pupil in Bolgatanga reads "Bolgatanga" and "Atampoka" with
    ease, and the formula was not made for them. So these words count towards sentence length but are
    left out of the syllables-per-word average (they are "neutral"):
      - every term from `ghana_terms.txt` (places, personal names, foods, games and so on);
      - a word that starts with a capital letter and does not start a sentence (a proper noun);
      - a sentence-opening word with a capital that is possessive ("Abugri's") or that also appears
        with its capital inside a sentence of the same setting;
      - numbers, and words in capitals such as Roman numerals ("XIV") and "HCF".
    One limit: a name that is not in `ghana_terms.txt` and stands alone as the first word of a sentence
    cannot be told from an ordinary word, so it is counted like one. Add real names to the term file.

Only the standard library is used.
"""

from __future__ import annotations

import difflib
import functools
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

MAX_CHARS = 220
MAX_LABEL_CHARS = 60
FK_LIMIT_LOWER = 5.0  # Basic 1 to Basic 3
FK_LIMIT_UPPER = 7.0  # Basic 4 to Basic 6
EM_DASH = chr(0x2014)
CEDI = "GH₵"

NOTES = (1, 2, 5, 10, 20, 50, 100, 200)  # cedi notes in use
CEDI_COINS = (1, 2)  # cedi coins in use
PESEWA_COINS = (1, 5, 10, 20, 50)  # pesewa coins in use

# near-duplicate thresholds for two settings of one indicator
DUPLICATE_RATIO = 0.85  # difflib ratio of the normalised texts
DUPLICATE_JACCARD = 0.75  # shared words / all words

CHECKS = ("form", "length", "em_dash", "reading_level", "banned", "currency", "ghana_term", "duplicate")

_HERE = Path(__file__).resolve().parent
GHANA_TERMS_PATH = _HERE / "ghana_terms.txt"
BANNED_TERMS_PATH = _HERE / "banned_terms.txt"


@dataclass(frozen=True)
class Failure:
    """One failed check: the name of the check and a message a writer can act on."""

    check: str
    message: str

    def __str__(self) -> str:
        return f"{self.check}: {self.message}"


# ----------------------------------------------------------------------------- term files


def _read_term_file(path: Path) -> list[tuple[str, str]]:
    """`(section, term)` pairs of a term file: one term per line, `#` comments, `# [name]` sections."""
    out: list[tuple[str, str]] = []
    section = ""
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            m = re.fullmatch(r"#\s*\[(.+)\]\s*", line)
            if m:
                section = m.group(1).strip()
            continue
        out.append((section, line))
    return out


@functools.lru_cache(maxsize=4)
def load_ghana_terms(path: Path | None = None) -> tuple[str, ...]:
    """Every term in `ghana_terms.txt`, in file order."""
    return tuple(term for _, term in _read_term_file(Path(path) if path else GHANA_TERMS_PATH))


@functools.lru_cache(maxsize=4)
def ghana_term_sections(path: Path | None = None) -> dict[str, tuple[str, ...]]:
    """`{section name: terms}` of `ghana_terms.txt` (for reports and for people)."""
    out: dict[str, list[str]] = {}
    for section, term in _read_term_file(Path(path) if path else GHANA_TERMS_PATH):
        out.setdefault(section, []).append(term)
    return {k: tuple(v) for k, v in out.items()}


@functools.lru_cache(maxsize=4)
def _ghana_patterns(path: Path | None = None) -> tuple[re.Pattern[str] | None, re.Pattern[str] | None, bool]:
    """(pattern for names, pattern for everyday words, whether GH₵ is listed)."""
    exact: list[str] = []
    loose: list[str] = []
    cedi = False
    for term in load_ghana_terms(path):
        if term == CEDI:
            cedi = True
        elif term[0].isupper():
            exact.append(term)
        else:
            loose.append(term)

    def alternation(terms: list[str]) -> str:
        return "|".join(re.escape(t) for t in sorted(set(terms), key=len, reverse=True))

    p_exact = re.compile(rf"(?<!\w)(?:{alternation(exact)})(?:'s|’s)?(?!\w)") if exact else None
    p_loose = re.compile(rf"(?<!\w)(?:{alternation(loose)})(?:es|s)?(?!\w)", re.IGNORECASE) if loose else None
    return p_exact, p_loose, cedi


def ghana_term_spans(text: str, path: Path | None = None) -> list[tuple[int, int, str]]:
    """`(start, end, matched text)` for every Ghana term found in `text`."""
    p_exact, p_loose, cedi = _ghana_patterns(path)
    spans = [(m.start(), m.end(), m.group(0)) for p in (p_exact, p_loose) if p is not None for m in p.finditer(text)]
    if cedi:
        spans += [(m.start(), m.end(), CEDI) for m in re.finditer(re.escape(CEDI), text)]
    return sorted(spans)


def find_ghana_terms(text: str, path: Path | None = None) -> list[str]:
    """The Ghana terms found in `text`, in order of appearance (empty when there are none)."""
    return [found for _, _, found in ghana_term_spans(text, path)]


@functools.lru_cache(maxsize=4)
def load_banned_terms(path: Path | None = None) -> tuple[tuple[str, str, re.Pattern[str]], ...]:
    """`(category, term as written in the file, compiled pattern)` for every banned term."""
    out = []
    for category, raw in _read_term_file(Path(path) if path else BANNED_TERMS_PATH):
        term = raw
        flags = re.IGNORECASE
        if term.startswith("="):
            term, flags = term[1:].strip(), 0
        tail = r"(?!\w)"
        if term.endswith("*"):
            term, tail = term[:-1], r"\w*"
        body = r"\s+".join(re.escape(part) for part in term.split())
        out.append((category, raw, re.compile(rf"(?<!\w){body}{tail}", flags)))
    return tuple(out)


def find_banned_terms(text: str, path: Path | None = None) -> list[tuple[str, str]]:
    """`(category, matched text)` for every banned term in `text`."""
    found = []
    for category, _, pattern in load_banned_terms(path):
        for m in pattern.finditer(text):
            found.append((category, m.group(0)))
    return found


# ----------------------------------------------------------------------------- syllables

_VOWELS = "aeiouy"
# words the rules below get wrong; plurals in -s and -es are looked up through their singular
_SYLLABLE_OVERRIDES = {
    "area": 3,
    "idea": 3,
    "create": 2,
    "created": 3,
    "science": 2,
    "poem": 2,
    "fuel": 2,
    "video": 3,
    "radio": 3,
    "ratio": 3,
    "period": 3,
    "million": 2,
    "billion": 2,
    "onion": 2,
    "lion": 2,
    "usual": 3,
    "usually": 4,
    "hundred": 2,
    "ninety": 2,
    "nineteen": 2,
    "something": 2,
    "sometimes": 2,
    "someone": 2,
    "likely": 2,
    "unlikely": 3,
    "every": 2,
    "everyone": 3,
    "everything": 3,
    "evening": 2,
    "vegetable": 3,
    "pineapple": 3,
    "notebook": 2,
    "prism": 2,
    "continuous": 4,
    "theoretical": 5,
    "player": 2,
    "quiet": 2,
    "heaviest": 3,
    "stadium": 3,
    "january": 4,
    "february": 4,
    "wednesday": 2,
    "does": 1,
    "goes": 1,
    "eye": 1,
    "maybe": 2,
}


def _count_syllables(w: str) -> int:
    if w in _SYLLABLE_OVERRIDES:
        return _SYLLABLE_OVERRIDES[w]
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        if w[:-1] in _SYLLABLE_OVERRIDES:
            return _SYLLABLE_OVERRIDES[w[:-1]]
        if not w.endswith("es"):
            return _count_syllables(w[:-1])
        if not w[:-2].endswith(("s", "x", "z", "ch", "sh", "c", "g")):
            return _count_syllables(w[:-1])  # "plates" reads like "plate", "tables" like "table"
        # "boxes", "oranges", "pieces": the ending is a sound of its own, counted below
    groups = re.findall(rf"[{_VOWELS}]+", w)
    n = len(groups)
    # "ia" is two sounds ("tri-an-gle", "di-a-gram") unless it follows c, t or s ("special", "initial")
    n += len(re.findall(r"(?<![cts])ia", w))
    # "-ing" after a vowel adds a sound ("play-ing", "go-ing")
    if len(w) > 4 and w.endswith("ing") and w[-4] in _VOWELS:
        n += 1
    if n > 1:
        if w.endswith("ed") and len(w) > 3 and w[-3] not in "td" and w[-3] not in _VOWELS:
            n -= 1  # "shared", "packed"; "counted" and "added" keep the sound
        elif w.endswith("e") and w[-2] not in _VOWELS:
            keeps = (w.endswith("le") and len(w) > 2 and w[-3] not in _VOWELS) or w.endswith("tre")
            if not keeps:
                n -= 1  # silent e: "make", "whole"; "table" and "metre" keep the sound
    return max(n, 1)


def syllables(word: str) -> int:
    """Number of syllables in an English word (a rule of thumb, right for the words pupils meet).

    A hyphenated word is the sum of its parts. A token with no vowel ("km", "cm") counts as one.
    """
    w = unicodedata.normalize("NFKD", word).lower()
    w = re.sub(r"['’]s$", "", w)
    parts = [re.sub(r"[^a-z]", "", part) for part in re.split(r"[-\s]+", w)]
    parts = [p for p in parts if p]
    if not parts:
        return 1
    return sum(_count_syllables(p) for p in parts)


# ----------------------------------------------------------------------------- reading level

_SENTENCE_END = re.compile(r"[.?!]+[\"'’”)\]]*(?=\s+[^a-z\s]|\s*$)")
_NUMBER = r"\d+(?:[.,:/]\d+)*(?:st|nd|rd|th|s)?(?![^\W\d_])"
_WORD = r"[^\W\d_]+(?:['’\-][^\W\d_]+)*"
_TOKEN = re.compile(rf"(?P<num>{_NUMBER})|(?P<word>{_WORD})")


def split_sentences(text: str) -> list[str]:
    """The sentences of a setting (see the module docstring for what ends a sentence)."""
    out, start = [], 0
    for m in _SENTENCE_END.finditer(text):
        piece = text[start : m.end()].strip()
        if piece:
            out.append(piece)
        start = m.end()
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


@dataclass(frozen=True)
class Readability:
    """The numbers behind a Flesch-Kincaid grade."""

    grade: float
    sentences: int
    words: int
    counted_words: int
    syllables: int
    neutral: tuple[str, ...] = ()


def readability(text: str) -> Readability:
    """Flesch-Kincaid grade of a setting, with names and numbers treated as the module docstring says."""
    clean = text.replace(CEDI, " ")
    # blank out the terms from ghana_terms.txt but remember them: each word in them is neutral
    term_spans = [(s, e) for s, e, _ in ghana_term_spans(clean)]

    def in_term(start: int, end: int) -> bool:
        return any(s <= start and end <= e for s, e in term_spans)

    sentences = 0
    tokens: list[tuple[str, int, int, bool, bool]] = []  # (text, start, end, is number, opens a sentence)
    pos = 0
    for sentence in split_sentences(clean):
        offset = clean.index(sentence, pos)
        pos = offset + len(sentence)
        found = list(_TOKEN.finditer(sentence))
        if not found:
            continue
        sentences += 1
        first_word_seen = False
        for m in found:
            is_num = m.group("num") is not None
            opens = not first_word_seen
            first_word_seen = True
            tokens.append((m.group(0), offset + m.start(), offset + m.end(), is_num, opens))
    if not tokens:
        return Readability(0.0, 0, 0, 0, 0)

    inside_capitals = {
        t.removesuffix("'s").removesuffix("’s")
        for t, _, _, is_num, opens in tokens
        if not is_num and not opens and t[0].isupper()
    }
    counted = syl = 0
    neutral: list[str] = []
    for t, start, end, is_num, opens in tokens:
        bare = t.removesuffix("'s").removesuffix("’s")
        possessive = bare != t
        letters = [c for c in t if c.isalpha()]
        if (
            is_num
            or in_term(start, end)
            or (len(letters) >= 2 and all(c.isupper() for c in letters))
            or (t[0].isupper() and not opens)
            or (t[0].isupper() and opens and (possessive or bare in inside_capitals) and len(bare) > 1)
        ):
            neutral.append(t)
            continue
        counted += 1
        syl += syllables(t)
    words = len(tokens)
    asw = syl / counted if counted else 1.0
    grade = 0.39 * (words / sentences) + 11.8 * asw - 15.59
    return Readability(round(grade, 2), sentences, words, counted, syl, tuple(neutral))


def fk_grade(text: str) -> float:
    """Flesch-Kincaid grade of a setting (names and numbers do not push it up)."""
    return readability(text).grade


def fk_limit(grade: int) -> float:
    """The highest Flesch-Kincaid grade allowed for a class: 5 for Basic 1 to 3, 7 for Basic 4 to 6."""
    return FK_LIMIT_LOWER if grade <= 3 else FK_LIMIT_UPPER


def grade_of(indicator_id: str) -> int:
    """The class of an indicator from its code: `grade_of("math:B4.1.1.1.2") -> 4`."""
    code = indicator_id.rpartition(":")[2]
    m = re.match(r"[A-Za-z]+(\d+)\.", code)
    if not m:
        raise ValueError(f"cannot read the class from indicator {indicator_id!r}")
    return int(m.group(1))


# ----------------------------------------------------------------------------- single checks

_AMOUNT_TAIL = r"(?:\d{1,3}(?:,\d{3})+|\d{1,3})(?:\.\d{2})?"
_GOOD_AMOUNT = re.compile(rf"{re.escape(CEDI)} {_AMOUNT_TAIL}(?![\d]|[.,]\d)")
_ANY_AMOUNT = r"(?:GH₵ ?[\d,]+(?:\.\d+)?|\d+ pesewas?)"
_AMOUNT_LIST = rf"{_ANY_AMOUNT}(?:(?:,? and |,? or |, ){_ANY_AMOUNT})*"
_NAMED_AFTER = re.compile(rf"({_AMOUNT_LIST}) (?:play |paper |toy |cedi |cedis )?(notes?|coins?|bills?)\b")
_NAMED_BEFORE = re.compile(rf"\b(notes?|coins?|bills?) (?:of|worth) ({_AMOUNT_LIST})")
_ONE_AMOUNT = re.compile(r"GH₵ ?(?P<cedis>[\d,]+(?:\.\d+)?)|(?P<pesewas>\d+) pesewas?")


def check_form(text: str) -> list[Failure]:
    """One clean line that reads as a sentence."""
    out = []
    if not text or not text.strip():
        return [Failure("form", "the setting is empty")]
    if text != text.strip():
        out.append(Failure("form", "remove the spaces at the start or the end"))
    if re.search(r"[\n\r\t]", text):
        out.append(Failure("form", "write the setting on one line (no line breaks or tabs)"))
    if "  " in text:
        out.append(Failure("form", "remove the double space"))
    body = text.strip()
    if body and not (body[0].isupper() or body[0].isdigit() or body[0] in "\"'‘“"):
        out.append(Failure("form", "start with a capital letter"))
    if body and not re.search(r"[.?!][\"'’”)]?$", body):
        out.append(Failure("form", "end with a full stop or a question mark"))
    return out


def check_length(text: str, limit: int = MAX_CHARS) -> list[Failure]:
    """At most `limit` characters, spaces and punctuation included."""
    n = len(text)
    return [Failure("length", f"{n} characters; the limit is {limit}")] if n > limit else []


def check_em_dash(text: str) -> list[Failure]:
    """No em dash."""
    return [Failure("em_dash", "replace the em dash with a comma, a colon or a full stop")] if EM_DASH in text else []


def check_reading_level(text: str, grade: int) -> list[Failure]:
    """Flesch-Kincaid grade within the limit for the class."""
    limit = fk_limit(grade)
    r = readability(text)
    if r.grade <= limit:
        return []
    return [
        Failure(
            "reading_level",
            f"Flesch-Kincaid grade {r.grade:.1f}; the limit for Basic {grade} is {limit:g}. "
            "Use shorter sentences and shorter words.",
        )
    ]


def check_banned(text: str) -> list[Failure]:
    """No term from the banned list."""
    advice = {
        "brand": "brand name: write what the thing is",
        "unsafe": "not suitable for a young child",
        "stereotype": "stereotype",
        "not_ghana": "not how it is said or measured in Ghana",
    }
    seen: set[tuple[str, str]] = set()
    out = []
    for category, found in find_banned_terms(text):
        if (category, found.lower()) in seen:
            continue
        seen.add((category, found.lower()))
        out.append(Failure("banned", f'"{found}" ({advice.get(category, category)})'))
    return out


def _amount_value(match: re.Match[str]) -> tuple[str, float]:
    if match.group("cedis") is not None:
        return "cedi", float(match.group("cedis").replace(",", ""))
    return "pesewa", float(match.group("pesewas"))


def check_currency(text: str) -> list[Failure]:
    """Money written `GH₵ 7.50` and `50 pesewas`; a named coin or note must exist."""
    out: list[Failure] = []
    # 1. every GH₵ must open a well-formed amount
    for m in re.finditer(re.escape(CEDI), text):
        if not _GOOD_AMOUNT.match(text, m.start()):
            shown = re.match(rf"{re.escape(CEDI)}\s*\S*", text[m.start() :]).group(0).rstrip(".,;:?!")  # type: ignore[union-attr]
            out.append(
                Failure(
                    "currency",
                    f'"{shown}": write GH₵, one space, the cedis, then a point and two digits for '
                    "pesewas (GH₵ 7.50, GH₵ 12, GH₵ 1,250.00)",
                )
            )
    # 2. other ways of writing cedis and pesewas
    wrong = [
        (r"(?<![A-Za-z])(?:GH[SC¢]|Gh[SsCc¢₵]|gh[sc₵])(?![a-z])", "write GH₵"),
        (r"¢", "write GH₵"),
        (r"(?<![A-Za-z])GH(?:\s?\?|\s+(?=\d))", "the cedi sign is missing or broken: write GH₵"),
        (r"[$£€]", "use cedis: write GH₵"),
        (r"\d\s*cedis?\b", "write amounts as GH₵ 1 and GH₵ 5, not as a number followed by the word cedi"),
        (r"\b\d+\s?p\b(?!\.\s?m)", "write 50 pesewas, not 50p"),
        (r"\bGp\b", "write pesewas"),
    ]
    for pattern, message in wrong:
        m = re.search(pattern, text)
        if m:
            out.append(Failure("currency", f'"{m.group(0)}": {message}'))
    for m in re.finditer(r"\b(\d+) (pesewas?)\b", text):
        n, word = int(m.group(1)), m.group(2)
        if (n == 1) != (word == "pesewa"):
            out.append(Failure("currency", f'"{m.group(0)}": write 1 pesewa and {max(n, 2)} pesewas'))
    # 3. named coins and notes must exist
    named = [(m.group(1), m.group(2)) for m in _NAMED_AFTER.finditer(text)]
    named += [(m.group(2), m.group(1)) for m in _NAMED_BEFORE.finditer(text)]
    for amounts, kind in named:
        is_note = kind.startswith(("note", "bill"))
        for am in _ONE_AMOUNT.finditer(amounts):
            unit, value = _amount_value(am)
            if is_note:
                ok = unit == "cedi" and value in NOTES
                real = "GH₵ 1, 2, 5, 10, 20, 50, 100 and 200"
            else:
                ok = (unit == "cedi" and value in CEDI_COINS) or (unit == "pesewa" and value in PESEWA_COINS)
                real = "1, 5, 10, 20 and 50 pesewas, GH₵ 1 and GH₵ 2"
            if not ok:
                what = "note" if is_note else "coin"
                out.append(Failure("currency", f'there is no "{am.group(0)}" {what}; the {what}s are {real}'))
    return out


def check_ghana_term(text: str) -> list[Failure]:
    """At least one term from `ghana_terms.txt`."""
    if find_ghana_terms(text):
        return []
    return [
        Failure(
            "ghana_term",
            "no Ghanaian term found: name a real place, a local food or game, or use cedis (see ghana_terms.txt)",
        )
    ]


def normalise(text: str) -> str:
    """Text reduced to small letters, digits and single spaces, for comparing two settings."""
    t = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t)).strip()


def similarity(a: str, b: str) -> tuple[float, float]:
    """(difflib ratio, share of common words) of two settings after `normalise`."""
    na, nb = normalise(a), normalise(b)
    ratio = difflib.SequenceMatcher(None, na, nb).ratio()
    wa, wb = set(na.split()), set(nb.split())
    jaccard = len(wa & wb) / len(wa | wb) if wa | wb else 1.0
    return ratio, jaccard


def is_near_duplicate(a: str, b: str) -> bool:
    """True when two settings are close enough to count as one."""
    ratio, jaccard = similarity(a, b)
    return ratio >= DUPLICATE_RATIO or jaccard >= DUPLICATE_JACCARD


def check_duplicate(text: str, siblings: Iterable[str]) -> list[Failure]:
    """Not a near copy of another setting of the same indicator."""
    for other in siblings:
        if is_near_duplicate(text, other):
            return [Failure("duplicate", f'too close to another setting of this indicator: "{other[:60]}..."')]
    return []


# ----------------------------------------------------------------------------- all checks


def check_setting(text: str, grade: int, siblings: Iterable[str] = ()) -> list[Failure]:
    """Run every automatic check on one English setting. An empty list means it passes.

    `grade` is the class (1 to 6). `siblings` are the other settings of the same indicator.
    """
    out = check_form(text)
    if not text or not text.strip():
        return out
    out += check_length(text)
    out += check_em_dash(text)
    out += check_reading_level(text, grade)
    out += check_banned(text)
    out += check_currency(text)
    out += check_ghana_term(text)
    out += check_duplicate(text, siblings)
    return out


def check_label(label: str) -> list[Failure]:
    """Checks on a short label: one clean line, at most 60 characters, no em dash, no banned term."""
    if not label or not label.strip():
        return [Failure("form", "the label is empty")]
    out = []
    if label != label.strip() or "  " in label or re.search(r"[\n\r\t]", label):
        out.append(Failure("form", "write the label on one line with single spaces"))
    if len(label) > MAX_LABEL_CHARS:
        out.append(Failure("length", f"{len(label)} characters; the limit for a label is {MAX_LABEL_CHARS}"))
    out += check_em_dash(label)
    out += check_banned(label)
    if label.strip().endswith("."):
        out.append(Failure("form", "a label has no full stop at the end"))
    return out


def _numbers(text: str) -> Counter[str]:
    return Counter(re.findall(r"\d+(?:[.,:/]\d+)*", text))


def check_translation(text: str, source: str) -> list[Failure]:
    """Checks on a translated setting: form, length, no em dash, money format and the same numbers.

    The reading-level formula and the term lists are made for English, so they are not used here.
    A fluent teacher judges the language.
    """
    out = check_form(text)
    if not text or not text.strip():
        return out
    out += check_length(text)
    out += check_em_dash(text)
    out += [f for f in check_currency(text) if "there is no" not in f.message]
    want, have = _numbers(source), _numbers(text)
    if want != have:
        missing = sorted((want - have).elements())
        extra = sorted((have - want).elements())
        parts = []
        if missing:
            parts.append("missing " + ", ".join(missing))
        if extra:
            parts.append("not in the English " + ", ".join(extra))
        out.append(Failure("numbers", "the numbers must be the same as in the English setting: " + "; ".join(parts)))
    return out


@dataclass
class Report:
    """Result of running the checks on many settings."""

    total: int = 0
    passed: int = 0
    by_check: dict[str, int] = field(default_factory=lambda: {c: 0 for c in CHECKS})
    failures: list[tuple[str, int, str, list[Failure]]] = field(default_factory=list)  # (indicator, index, text, why)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 1.0

    def summary_lines(self) -> list[str]:
        """A per-check summary, one line per check."""
        lines = [f"{self.total} settings checked: {self.passed} pass, {self.failed} fail ({self.pass_rate:.1%} pass)"]
        for check in self.by_check:
            n = self.by_check[check]
            lines.append(f"  {check:<14} {self.total - n:>4} pass  {n:>4} fail")
        return lines


def check_texts(texts_by_indicator: Mapping[str, Sequence[str]]) -> Report:
    """Run `check_setting` on every text of every indicator. Keys are indicator ids or bare codes."""
    report = Report()
    for indicator, texts in texts_by_indicator.items():
        grade = grade_of(indicator)
        for i, text in enumerate(texts):
            failures = check_setting(text, grade, siblings=texts[:i])
            report.total += 1
            if failures:
                for name in {f.check for f in failures}:
                    report.by_check[name] = report.by_check.get(name, 0) + 1
                report.failures.append((indicator, i, text, failures))
            else:
                report.passed += 1
    return report


def check_contexts(settings: Mapping[str, Sequence[Mapping[str, object]]]) -> Report:
    """Run the checks on the `settings` mapping of `data/contexts/<subject>.json` (English entries)."""
    texts = {
        indicator: [str(e["text"]) for e in entries if e.get("lang", "en") == "en"]
        for indicator, entries in settings.items()
    }
    return check_texts(texts)
