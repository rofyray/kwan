"""Automatic checks for practice settings (PRD 8.3): each check on its own, then the term files."""

import re
import subprocess
import sys

import pytest

from kwan.context import checks

EM_DASH = chr(0x2014)

# the six good examples of docs/context-style-guide.md, with their class
GOOD = [
    (1, "Akua has 8 oranges on her tray at Makola Market. She sells 3. How many oranges are left on the tray?"),
    (2, "Yao has one GH₵ 2 coin and three 50 pesewas coins. How much money does he have altogether?"),
    (
        3,
        "An oware board has 12 houses. Tetteh puts 4 seeds in each house to start a game. How many seeds does he use?",
    ),
    (
        4,
        "Esinam and her father catch 36 tilapia on the lake at Kpando. They put 9 fish in each basin. How many basins do they fill?",
    ),
    (
        5,
        "Fuseini buys 3 exercise books at GH₵ 4.50 each in Tamale. He pays with a GH₵ 20 note. How much change does he get?",
    ),
    (
        6,
        "The road from Tamale to Bolgatanga is about 160 km. Lamisi's bus has covered three quarters of the way. How many kilometres are left?",
    ),
]
BAD_FOREIGN = "Tom has 5 dollars. He buys a hot dog for 2 dollars at the baseball game. How much money is left?"
BAD_MONEY = "Kofi pays GHC3.5 for bread with a GH₵ 3 note and a 25 pesewas coin."
BAD_BRAND = "Kwame buys 4 tins of Milo at the supermarket. Each tin costs GH₵ 60. How much does he pay?"
BAD_LONG = (
    "During the annual harvest celebration, which is traditionally attended by numerous dignitaries, a prosperous "
    "agricultural entrepreneur distributes approximately 1,250 kilograms of produce proportionally among several "
    "neighbouring communities, so calculate each community's allocation."
)


def names(failures):
    return {f.check for f in failures}


# ----------------------------------------------------------------------------- style guide examples


@pytest.mark.parametrize("grade, text", GOOD)
def test_the_good_examples_of_the_style_guide_pass(grade, text):
    assert checks.check_setting(text, grade) == []


def test_foreign_setting_fails_on_banned_words_and_has_no_ghana_term():
    failures = checks.check_setting(BAD_FOREIGN, 4)
    assert names(failures) == {"banned", "ghana_term"}
    found = " ".join(f.message for f in failures)
    for word in ("dollars", "hot dog", "baseball"):
        assert word in found


def test_wrong_money_fails_on_format_and_on_coins_and_notes_that_do_not_exist():
    messages = [f.message for f in checks.check_currency(BAD_MONEY)]
    assert any("GHC" in m for m in messages)
    assert any('no "GH₵ 3" note' in m for m in messages)
    assert any('no "25 pesewas" coin' in m for m in messages)


def test_brand_name_fails():
    failures = checks.check_setting(BAD_BRAND, 4)
    assert names(failures) == {"banned"}
    assert "brand" in failures[0].message


def test_long_hard_sentence_fails_length_and_reading_level():
    assert {"length", "reading_level"} <= names(checks.check_setting(BAD_LONG, 6))


# ----------------------------------------------------------------------------- length, form, em dash


def test_length_limit_is_220_characters():
    base = "Ama counts in Accra. "
    ok = (base * 11)[:219] + "."
    assert len(ok) == 220 and checks.check_length(ok) == []
    assert names(checks.check_length(ok + "x")) == {"length"}
    assert "221" in checks.check_length(ok + "x")[0].message


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        " Ama has 3 yams in Ho.",
        "Ama has 3 yams in Ho. ",
        "Ama has 3  yams in Ho.",
        "Ama has 3 yams\nin Ho.",
        "ama has 3 yams in Ho.",
        "Ama has 3 yams in Ho",
    ],
)
def test_form_check_refuses_untidy_text(text):
    assert names(checks.check_form(text)) == {"form"}


@pytest.mark.parametrize(
    "text",
    [
        "Ama has 3 yams in Ho.",
        "3 okadas wait at a junction in Wa. How many wheels?",
        "Solve 8 + □ = 15.",
        "Is he right?",
        "'Sixty votes came in,' says a host in Wa.",
    ],
)
def test_form_check_accepts_clean_sentences(text):
    assert checks.check_form(text) == []


def test_em_dash_is_refused():
    assert names(checks.check_em_dash(f"Ama has 3 yams {EM_DASH} and 2 more.")) == {"em_dash"}
    assert checks.check_em_dash("Ama has 3 yams - and 2 more.") == []


def test_no_em_dash_in_any_message_or_term_file():
    for path in (checks.GHANA_TERMS_PATH, checks.BANNED_TERMS_PATH):
        assert EM_DASH not in path.read_text(encoding="utf-8")
    messages = [str(f) for f in checks.check_setting(BAD_MONEY + " " + BAD_LONG + EM_DASH, 2)]
    assert messages and not any(EM_DASH in m for m in messages[1:] if "em_dash" not in m)


# ----------------------------------------------------------------------------- syllables and reading level

SYLLABLES = {
    "make": 1,
    "table": 2,
    "whole": 1,
    "triangle": 3,
    "rectangle": 3,
    "square": 1,
    "people": 2,
    "oranges": 3,
    "plates": 1,
    "tables": 2,
    "mangoes": 2,
    "tomatoes": 3,
    "shared": 1,
    "counted": 2,
    "packed": 1,
    "hundred": 2,
    "metre": 2,
    "kilometres": 4,
    "centimetre": 4,
    "litres": 2,
    "playing": 2,
    "going": 2,
    "thing": 1,
    "area": 3,
    "equal": 2,
    "pieces": 2,
    "minutes": 2,
    "classmates": 2,
    "exercise": 3,
    "favourite": 3,
    "numbers": 2,
    "altogether": 4,
    "fraction": 2,
    "denominator": 5,
    "perimeter": 4,
    "multiply": 3,
    "multiplication": 5,
    "probability": 5,
    "impossible": 4,
    "estimate": 3,
    "diagonal": 4,
    "symmetrical": 4,
    "km": 1,
    "the": 1,
    "does": 1,
    "boxes": 2,
    "halves": 1,
    "school": 1,
    "water": 2,
    "bicycle": 3,
    "many": 2,
    "money": 2,
    "colour": 2,
    "quarter": 2,
    "seven": 2,
    "eleven": 3,
    "twelve": 1,
    "thousand": 2,
    "ninety": 2,
    "measured": 2,
    "divided": 3,
    "hand-span": 2,
    "degrees": 2,
    "temperature": 4,
    "equivalent": 4,
    "likely": 2,
    "change": 1,
    "friends": 1,
    "equally": 3,
    "different": 3,
    "being": 2,
    "pineapples": 3,
    "classes": 2,
    "leaves": 1,
    "circle": 2,
    "double": 2,
    "scale": 1,
    "three": 1,
    "value": 2,
    "ratio": 3,
    "million": 2,
}


@pytest.mark.parametrize("word, expected", sorted(SYLLABLES.items()))
def test_syllable_counter_on_words_pupils_meet(word, expected):
    assert checks.syllables(word) == expected


def test_syllables_ignore_a_possessive_ending_and_never_return_zero():
    assert checks.syllables("Ama's") == checks.syllables("Ama")
    assert checks.syllables("cm") == 1
    assert checks.syllables("□") == 1


def test_sentences_do_not_break_inside_money_times_or_decimals():
    text = "An oware game in Ada starts at 10.15 a.m. and ends at 10.40 a.m. How long did it last? It cost GH₵ 7.50."
    assert checks.split_sentences(text) == [
        "An oware game in Ada starts at 10.15 a.m. and ends at 10.40 a.m.",
        "How long did it last?",
        "It cost GH₵ 7.50.",
    ]


def test_reading_grade_uses_the_flesch_kincaid_formula():
    # 2 sentences, 8 words, no names or numbers: "The cat sat here. The dog ran away."
    r = checks.readability("The cat sat here. The dog ran away.")
    assert (r.sentences, r.words, r.counted_words, r.syllables) == (2, 8, 8, 9)
    assert r.grade == pytest.approx(0.39 * (8 / 2) + 11.8 * (9 / 8) - 15.59, abs=0.01)


def test_long_ghanaian_names_do_not_raise_the_grade():
    short = "Ama buys 3 yams in Ho. She sells 2 of them. How many yams are left?"
    long = "Asantewaa buys 3 yams in Bolgatanga. She sells 2 of them. How many yams are left?"
    assert checks.fk_grade(long) == checks.fk_grade(short)


def test_a_name_the_term_list_does_not_hold_is_neutral_inside_a_sentence():
    # made-up names: neutral after the first word of a sentence, or when written as a possessive
    known = "A girl named Ama buys 3 yams in Ho. How many yams does she have?"
    unknown = "A girl named Nhyirabaaa buys 3 yams in Akwatialinee. How many yams does she have?"
    assert checks.fk_grade(unknown) == checks.fk_grade(known)
    assert "Nhyirabaaa's" in checks.readability("Nhyirabaaa's yams are big. She has 3.").neutral
    # at the very start of a sentence an unknown name is read as an ordinary word (documented limit)
    assert "Nhyirabaaa" not in checks.readability("Nhyirabaaa buys 3 yams.").neutral
    assert "Nhyirabaaa" in checks.readability("Nhyirabaaa buys 3 yams. Then Nhyirabaaa sells 2.").neutral


def test_names_numbers_and_roman_numerals_are_neutral_words():
    r = checks.readability("Kofi's seat at a durbar in Kumasi is XIV. Seat 20 is next to it.")
    assert {"Kofi's", "durbar", "Kumasi", "XIV", "20"} <= set(r.neutral)
    assert r.words == 15 and r.counted_words == 10


def test_reading_limits_by_class():
    assert [checks.fk_limit(g) for g in range(1, 7)] == [5.0, 5.0, 5.0, 7.0, 7.0, 7.0]
    text = "Korkor holds 4 oranges in each hand at Somanya market. Double 4. How many oranges is she holding?"
    grade = checks.fk_grade(text)
    assert 5.0 < grade <= 7.0
    assert names(checks.check_reading_level(text, 3)) == {"reading_level"}
    assert checks.check_reading_level(text, 4) == []
    assert "Basic 3" in checks.check_reading_level(text, 3)[0].message


def test_grade_of_reads_the_class_from_the_code():
    assert checks.grade_of("B4.1.1.1.2") == 4
    assert checks.grade_of("math:B1.1.1.1.1") == 1
    with pytest.raises(ValueError):
        checks.grade_of("math:nonsense")


# ----------------------------------------------------------------------------- currency


@pytest.mark.parametrize(
    "text",
    [
        "Kofi pays GH₵ 7.50 for kenkey.",
        "A bag costs GH₵ 12 in Wa.",
        "A bag costs GH₵ 12.00 in Wa.",
        "The school raised GH₵ 1,250.00.",
        "The town raised GH₵ 240,000 for a clinic.",
        "One pack costs GH₵ 0.",
        "Ama has 50 pesewas and 1 pesewa.",
        "A ball of banku costs b cedis in Aflao.",
        "The fund has reached six hundred and five cedis.",
        "Ama reads 27°C at 9 a.m. and 33°C at 1 p.m. in Sunyani.",
        "Is GH₵ 5 more than GH₵ 4.50?",
    ],
)
def test_well_written_money_passes(text):
    assert checks.check_currency(text) == []


@pytest.mark.parametrize(
    "text",
    [
        "A bag costs GHS 7.50.",
        "A bag costs GHC7.50.",
        "A bag costs Ghc 7.5.",
        "A bag costs ¢7.50.",
        "A bag costs 7.50 cedis.",
        "A bag costs $7.50.",
        "A bag costs GH₵12.",
        "A bag costs GH₵ 12.0.",
        "A bag costs GH₵ 12.5 today.",
        "The school raised GH₵ 1250.00.",
        "The school raised GH₵ 1.250,00.",
        "A sweet costs 50p.",
        "A sweet costs GH₵ .50.",
        "A sweet costs 50 Gp.",
        "Toss a 50 pesewa coin.",
        "Ama has 1 pesewas.",
        "Use 1 cedi and 2 cedi coins to make 5 cedis.",
    ],
)
def test_badly_written_money_fails(text):
    assert names(checks.check_currency(text)) == {"currency"}


@pytest.mark.parametrize(
    "text",
    [
        "Pay with a GH₵ 1 note, a GH₵ 2 note, a GH₵ 5 note or a GH₵ 10 note.",
        "She has GH₵ 20, GH₵ 50, GH₵ 100 and GH₵ 200 notes.",
        "He has a GH₵ 1 coin and three GH₵ 2 coins.",
        "Sort 1 pesewa, 5 pesewas, 10 pesewas, 20 pesewas and 50 pesewas coins.",
        "Wumpini has only GH₵ 100 and GH₵ 10 play notes.",
        "Show GH₵ 25 with GH₵ 5, GH₵ 10 and GH₵ 20 notes.",
    ],
)
def test_real_coins_and_notes_pass(text):
    assert checks.check_currency(text) == []


@pytest.mark.parametrize(
    "text, missing",
    [
        ("He pays with a GH₵ 3 note.", "GH₵ 3"),
        ("He pays with a GH₵ 25 note.", "GH₵ 25"),
        ("He pays with a GH₵ 500 note.", "GH₵ 500"),
        ("She has GH₵ 5, GH₵ 15 and GH₵ 20 notes.", "GH₵ 15"),
        ("She has a GH₵ 5 coin.", "GH₵ 5"),
        ("She has a 25 pesewas coin.", "25 pesewas"),
        ("She has two 2 pesewas coins.", "2 pesewas"),
        ("He gives a note of GH₵ 30.", "GH₵ 30"),
        ("He finds a 50 pesewas note.", "50 pesewas"),
    ],
)
def test_coins_and_notes_that_do_not_exist_fail(text, missing):
    messages = [f.message for f in checks.check_currency(text)]
    assert any(f'there is no "{missing}"' in m for m in messages), messages


def test_an_amount_is_not_a_coin_or_a_note():
    # GH₵ 15 is a fine price; it only fails when it is called a note or a coin
    assert checks.check_currency("Zakari buys waakye for GH₵ 15 and pays with a GH₵ 20 note.") == []


# ----------------------------------------------------------------------------- banned terms


@pytest.mark.parametrize(
    "text, word",
    [
        ("Kwame drinks a tin of Milo.", "Milo"),
        ("Kwame drinks a tin of milo.", "milo"),
        ("Ama sends money by MTN.", "MTN"),
        ("He buys a tin of Peak milk.", "Peak milk"),
        ("Kofi bets GH₵ 5 on the match.", "bets"),
        ("She wins the lottery.", "lottery"),
        ("Abugri's raffle ticket has a number.", "raffle"),
        ("The men are gambling.", "gambling"),
        ("He drinks beer after work.", "beer"),
        ("A thief took 3 yams.", "thief"),
        ("The boys are fighting.", "fighting"),
        ("Two cars crashed on the road.", "crashed"),
        ("He works at a galamsey site.", "galamsey"),
        ("She throws like a girl.", "like a girl"),
        ("The house girl sweeps 3 rooms.", "house girl"),
        ("The road is 5 miles long.", "miles"),
        ("A gallon holds 4 litres.", "gallon"),
        ("It costs 5 dollars.", "dollars"),
        ("They play in the snow.", "snow"),
    ],
)
def test_banned_words_are_found(text, word):
    found = [w for _, w in checks.find_banned_terms(text)]
    assert word in found


@pytest.mark.parametrize(
    "text",
    [
        "Esi sells smoked fish at Elmina.",
        "Kwame cuts a stick of sugar cane in two.",
        "Roll a ludo die 30 times.",
        "Auntie Ama pounds fufu for 6 people.",
        "Mansa walks 5 steps backward.",
        "The bus leaves at 1 p.m. from Accra.",
        "The peak of Mount Afadja is 885 metres high.",
        "Pupils play tug of war at a school in Wa.",
        "A drummer beats the drum 8 times.",
        "The class votes for a game.",
        "A baby at a clinic in Nandom weighs 4 kg.",
        "The notice is on the bulletin board.",
        "Demonstrate how to fold the paper.",
        "Pupils build new skills each term.",
    ],
)
def test_honest_sentences_are_not_banned(text):
    assert checks.find_banned_terms(text) == []


def test_banned_categories_are_the_four_documented_ones():
    categories = {category for category, _, _ in checks.load_banned_terms()}
    assert categories == {"brand", "unsafe", "stereotype", "not_ghana"}


# ----------------------------------------------------------------------------- Ghana terms


@pytest.mark.parametrize(
    "text, expected",
    [
        ("A school in Ho has 20 desks.", ["Ho"]),
        ("Three Trotros leave Kaneshie.", ["Trotros", "Kaneshie"]),
        ("Ama's mangoes cost GH₵ 5.", ["Ama's", "GH₵"]),
        ("They eat tuo zaafi in Wa.", ["tuo zaafi", "Wa"]),
        ("She saves 50 pesewas.", ["pesewas"]),
        ("A bus goes to Sekondi-Takoradi.", ["Sekondi-Takoradi"]),
        ("Pupils play ampe and oware.", ["ampe", "oware"]),
        ("He sells garden eggs and yams.", ["garden eggs", "yams"]),
    ],
)
def test_ghana_terms_are_found(text, expected):
    assert checks.find_ghana_terms(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Who has more?",
        "What is the total?",
        "She waves at a friend and adds 3 to 4.",
        "ho ho ho, says the man.",
        "Tom has 5 apples.",
    ],
)
def test_names_match_whole_words_with_their_capital_only(text):
    assert checks.find_ghana_terms(text) == []
    assert names(checks.check_ghana_term(text)) == {"ghana_term"}


def test_ghana_terms_cover_all_16_regions_and_every_kind_of_term():
    sections = checks.ghana_term_sections()
    regions = {name.split(": ", 1)[1] for name in sections if name.startswith("places: ")}
    assert regions == {
        "Ahafo",
        "Ashanti",
        "Bono",
        "Bono East",
        "Central",
        "Eastern",
        "Greater Accra",
        "North East",
        "Northern",
        "Oti",
        "Savannah",
        "Upper East",
        "Upper West",
        "Volta",
        "Western",
        "Western North",
    }
    for name, terms in sections.items():
        if name.startswith("places: "):
            assert len(terms) >= 5, name
    for kind in ("currency", "food and drink", "transport", "games"):
        assert sections[kind], kind
    assert any(name.startswith("names: ") for name in sections)
    capitals = "Goaso Kumasi Sunyani Techiman Koforidua Accra Nalerigu Tamale Dambai Damongo Bolgatanga Wa Ho".split()
    all_terms = set(checks.load_ghana_terms())
    assert set(capitals) <= all_terms
    assert {"Cape Coast", "Sekondi-Takoradi", "Sefwi Wiawso"} <= all_terms
    assert {"trotro", "okada", "ampe", "oware", "chaskele", "kenkey", "waakye", "cedi", "pesewa", "GH₵"} <= all_terms


def test_term_files_are_tidy():
    ghana = checks.load_ghana_terms()
    assert len(ghana) == len(set(ghana)), [t for t in ghana if ghana.count(t) > 1]
    assert len(ghana) > 300
    for term in ghana:
        assert term == term.strip() and "  " not in term
    banned = [raw for _, raw, _ in checks.load_banned_terms()]
    assert len(banned) == len(set(banned)), [t for t in banned if banned.count(t) > 1]
    # no Ghana term is itself banned
    for term in ghana:
        assert checks.find_banned_terms(term) == [], term


def test_every_name_in_the_style_guide_tables_is_a_ghana_term():
    from pathlib import Path

    guide = (Path(__file__).resolve().parent.parent / "docs" / "context-style-guide.md").read_text(encoding="utf-8")
    terms = set(checks.load_ghana_terms())
    # region table: | Region | Capital | Other towns |
    rows = re.findall(r"^\| ([A-Z][A-Za-z ]+) \| ([A-Z][A-Za-z\- ]+) \| ([A-Z][^|]+) \|$", guide, flags=re.M)
    regions = {name.split(": ", 1)[1] for name in checks.ghana_term_sections() if name.startswith("places: ")}
    rows = [row for row in rows if row[0] in regions]
    towns = [t.strip() for _, capital, others in rows for t in [capital, *others.split(",")]]
    assert len(rows) == 16 and len(towns) == 80
    assert [t for t in towns if t not in terms] == []


# ----------------------------------------------------------------------------- duplicates, labels, translations


def test_near_duplicates_within_an_indicator_fail():
    a = "Ama has 8 oranges at Makola Market. She sells 3. How many oranges are left?"
    b = "Ama has 9 oranges at Makola Market. She sells 4. How many oranges are left?"
    c = "A trotro in Kumasi has 14 people. 5 more get in. How many people are in it now?"
    assert checks.is_near_duplicate(a, b)
    assert not checks.is_near_duplicate(a, c)
    assert names(checks.check_setting(b, 1, siblings=[a])) == {"duplicate"}
    assert checks.check_setting(c, 2, siblings=[a]) == []
    assert names(checks.check_setting(a.upper().replace("AMA", "Ama"), 1, siblings=[a])) >= {"duplicate"}


def test_check_texts_counts_each_failing_setting_once_per_check():
    report = checks.check_texts(
        {
            "B1.1.1.1.1": [GOOD[0][1], BAD_FOREIGN],
            "math:B4.1.1.1.1": [BAD_BRAND, GOOD[3][1], GOOD[3][1]],
        }
    )
    assert (report.total, report.passed, report.failed) == (5, 2, 3)
    assert report.by_check["banned"] == 2
    assert report.by_check["ghana_term"] == 1
    assert report.by_check["duplicate"] == 1
    assert report.by_check["length"] == 0
    lines = report.summary_lines()
    assert lines[0].startswith("5 settings checked: 2 pass, 3 fail")
    assert len(lines) == 1 + len(checks.CHECKS)
    assert [f[0] for f in report.failures] == ["B1.1.1.1.1", "math:B4.1.1.1.1", "math:B4.1.1.1.1"]


def test_check_contexts_reads_the_settings_mapping():
    settings = {"math:B1.1.1.1.1": [{"text": GOOD[0][1], "lang": "en", "reviewed": False}]}
    report = checks.check_contexts(settings)
    assert (report.total, report.passed) == (1, 1)


@pytest.mark.parametrize(
    "label, ok",
    [
        ("Read and write numbers to 100,000", True),
        ("x" * 60, True),
        ("x" * 61, False),
        ("", False),
        (" Count to 100", False),
        ("Count to 100.", False),
        (f"Count {EM_DASH} to 100", False),
        ("Count tins of Milo", False),
    ],
)
def test_label_checks(label, ok):
    assert (checks.check_label(label) == []) is ok


def test_translation_must_keep_the_numbers_and_the_money_format():
    english = "Fati buys waakye for GH₵ 15 in Wa. She pays with a GH₵ 20 note. How much change does she get?"
    # not a real translation: the test only looks at numbers, length, form and money format
    same_numbers = "Fati xxx waakye GH₵ 15 xxx Wa. Xxx GH₵ 20 xxx. Xxx xxx?"
    assert checks.check_translation(same_numbers, english) == []
    lost = checks.check_translation("Fati xxx waakye GH₵ 15 xxx Wa. Xxx GH₵ 50 xxx?", english)
    assert names(lost) == {"numbers"}
    assert "20" in lost[0].message and "50" in lost[0].message
    assert names(checks.check_translation("Fati xxx GHC15 xxx 20?", english)) >= {"currency"}
    assert names(checks.check_translation("x" * 230 + " 15 20.", english)) >= {"length"}


# ----------------------------------------------------------------------------- dependencies


def test_context_modules_need_only_the_standard_library():
    code = (
        "import sys\n"
        "import kwan.context.checks, kwan.context.settings, kwan.context.drafts\n"
        "import kwan.context.sheets, kwan.context.translate, kwan.context.coverage, kwan.context._cli\n"
        "heavy = {'pandas', 'numpy', 'sklearn', 'yaml', 'httpx', 'requests', 'networkx', 'pdfplumber'}\n"
        "print(sorted(heavy & set(sys.modules)))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
