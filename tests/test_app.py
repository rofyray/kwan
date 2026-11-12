"""Headless tests for the Streamlit demo (demo/app.py), driven through streamlit.testing.v1.AppTest.

They follow PRD Phase 10 (10.1, 10.2) and the UI contract of the v1.1 app. Nothing here asserts on a
curriculum label or on a metric value: every expectation comes from the shipped bundle through the public API
(`kwan.Kwan`) or from data/eval_results.json at test time.
"""

import ast
import logging
import math
import re
import sys
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from kwan import Kwan
from kwan.eval import results, simulate

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "demo" / "app.py"

TABS = ["Pupil view", "Curriculum map", "Evidence"]
LOAD, RESET = "Load a P4 pupil with gaps in earlier classes", "Start over with a new pupil"
RIGHT, WRONG = "✅ Got it right", "❌ Got it wrong"
SIMULATE = "Run simulator v2 now (a small run, takes a moment)"
MIX, DRAFTS = "Mix in other skills", "Show unreviewed draft settings"
LOADED_FORM = re.compile(
    r"Loaded (?P<all>\d+) past answers: (?P<strong>\d+) skills learned well (?P<weeks>\d+) weeks ago "
    r"\(3 right answers each\), (?P<lower>\d+) on other B1 to B3 skills \d+ weeks ago "
    r"\((?P<lower_tone>mostly right|mostly wrong)\) and (?P<b4>\d+) on B4 skills \d+ days ago "
    r"\((?P<b4_tone>mostly right|mostly wrong)\)\."
)
REVIEW_NOTE = "🔁 **Review:**"
DISCLAIMER = (
    "Kwan uses the NaCCA Standards-Based Curriculum to organise skills. "
    "Kwan is not reviewed, approved or endorsed by NaCCA or the Ministry of Education."
)
KWAN_MODEL = "elo-multi (Kwan)"
AUC_MODELS = ["global-mean", "skill-mean", "bkt", "elo", KWAN_MODEL]
DATASET_NAMES = {
    "assistments_2009": "ASSISTments 2009",
    "assistments_2012": "ASSISTments 2012",
    "assistments_2015": "ASSISTments 2015",
    "assistments_2017": "ASSISTments 2017",
    "eedi_2020_task34": "Eedi 2020 (tasks 3 and 4)",
    "eedi_2020_task12": "Eedi 2020 (tasks 1 and 2)",
    "xes3g5m": "XES3G5M",
    "junyi_2015": "Junyi 2015",
    "ammore_2024/all": "AMMORE 2024 (all)",
    "ammore_2024/ghana": "AMMORE 2024 (Ghana)",
}
NOT_RUN = "not run yet"
CARD = re.compile(r"^\*\*(?P<code>B\d[\d.]*)\*\* · B(?P<grade>\d) · ")
EM_DASH = chr(0x2014)
EPS = simulate.EXPLORE_EPS
TOL = 1e-3  # the app stamps answers with the clock; a few seconds of forgetting move mastery far less than this


# ---------------------------------------------------------------- helpers


@pytest.fixture(scope="module")
def K():
    return Kwan.from_bundle()


@pytest.fixture(scope="module")
def G(K):
    return K.graph


def open_app(timeout=60):
    at = AppTest.from_file(str(APP), default_timeout=timeout).run()
    assert_healthy(at)
    return at


def assert_healthy(at):
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def rerun(at, **kw):
    at.run(**kw)
    assert_healthy(at)
    return at


def button(at, label, nth=0):
    return [b for b in at.button if b.label == label][nth]


def click(at, label, nth=0, **kw):
    button(at, label, nth).click()
    return rerun(at, **kw)


def toggle(at, label):
    return [t for t in at.toggle if t.label == label][0]


def goal_box(at):
    return [s for s in at.selectbox if s.label.startswith("Goal skill")][0]


def language_box(at):
    return [s for s in at.selectbox if s.label == "Language"][0]


def cards(at, K):
    """The recommendation cards in page order: [(canonical id, grade)]."""
    return [(K.resolve(m["code"]), int(m["grade"])) for m in (CARD.match(x.value) for x in at.markdown) if m]


def card_ids(at, K):
    return [i for i, _ in cards(at, K)]


def whys(at):
    return [m.value for m in at.markdown if m.value.startswith("**Why this skill:**")]


def settings(at):
    return [i.value for i in at.info if i.value.startswith("**Try it in Ghana:** ")]


def table(at, has_column=None, index_name=None):
    """The one st.dataframe that has this column (or this index name)."""
    found = [
        d.value
        for d in at.dataframe
        if (has_column is not None and has_column in list(d.value.columns))
        or (index_name is not None and d.value.index.name == index_name)
    ]
    assert len(found) == 1, f"expected one table with {has_column or index_name!r}, found {len(found)}"
    return found[0]


def mastery_table(at):
    return table(at, index_name="strand")


def auc_table(at):
    return table(at, has_column=KWAN_MODEL)


def pupil_mastery(at, K):
    """Mastery of every skill for the pupil held in the session, through the public API, timestamps ignored."""
    return K.mastery_map(at.session_state["learner"])


def mastery_after(K, events):
    """Mastery of every skill for a new pupil who gave exactly these (indicator, correct[, when]) answers."""
    S = K.new_state("expected")
    for ind, correct, *when in events:
        K.update(S, ind, correct, at=when[0] if when else None)
    return K.mastery_map(S)


def rendered_text(at):
    """Every piece of text the page shows: prose, labels, options, table headers and table cells."""
    out = [
        x.value
        for kind in ("title", "header", "subheader", "markdown", "caption", "info", "success", "warning", "error")
        for x in getattr(at, kind)
    ]
    out += [t.label for t in at.tabs] + [b.label for b in at.button] + [p.proto.text for p in at.get("progress")]
    out += [text for kind in ("slider", "selectbox", "toggle") for w in getattr(at, kind) for text in (w.label, w.help)]
    out += [str(o) for s in at.selectbox for o in s.options]
    out += [f"{m.label} {m.value}" for m in at.metric]
    for d in at.dataframe:
        df = d.value
        out += [str(c) for c in df.columns] + [str(i) for i in df.index] + [str(df.index.name)]
        out += [v for col in df.columns for v in df[col].tolist() if isinstance(v, str)]
    return [t for t in out if t]


def fake_results(monkeypatch, fake):
    monkeypatch.setattr(results, "load", fake if callable(fake) else (lambda *a, **k: fake))


# ---------------------------------------------------------------- page structure


def test_app_runs_on_its_own_and_shows_the_three_tabs(K):
    at = open_app()
    assert [t.label for t in at.tabs] == TABS
    assert not at.title  # the wordmark image is the page heading
    first = next(iter(at.main.children.values()))
    assert first.type == "image" and first.proto.imgs[0].url.endswith(".png")
    assert [s.value for s in at.subheader] == ["What to learn next", "Mastery by class and strand"]


def test_header_counts_come_from_the_bundle(K, G):
    at = open_app()
    head = at.caption[0].value
    assert head.startswith(f"NaCCA B1 to B6 Mathematics · {G.number_of_nodes()} skills · bundle {K.bundle_version}")
    by_type = {}
    for *_, d in G.edges(data=True):
        by_type[d["type"]] = by_type.get(d["type"], 0) + 1
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Skills (NaCCA indicators)"] == str(G.number_of_nodes())
    assert metrics["Prerequisite links"] == str(G.number_of_edges())
    assert {k: v for k, v in metrics.items() if k.endswith(" links") and k != "Prerequisite links"} == {
        f"{kind} links": str(n) for kind, n in by_type.items()
    }
    assert len(table(at, has_column="type")) == G.number_of_edges()


def test_three_cards_each_with_label_reason_and_ghana_setting(K, G):
    at = open_app()
    shown = cards(at, K)
    assert len(shown) == 3 and len(set(shown)) == 3
    assert all(i in G and G.nodes[i]["grade"] == g for i, g in shown)
    assert len(whys(at)) == 3
    assert len(settings(at)) == 3 and all(len(s) > 30 for s in settings(at))
    bars = at.get("progress")
    assert len(bars) == 3 and all(re.fullmatch(r"Mastery so far: \d+%", b.proto.text) for b in bars)
    assert [b.label for b in at.button if b.label in (RIGHT, WRONG)] == [RIGHT, WRONG] * 3
    # each card is drawn as: skill line with its short label, then its reason
    for m in at.markdown:
        if CARD.match(m.value):
            code = CARD.match(m.value)["code"]
            assert m.value.endswith(G.nodes[K.resolve(code)]["short_label"].replace("*", "\\*"))
    pupil_tab = [m.value for m in at.markdown if CARD.match(m.value) or m.value.startswith("**Why this skill:**")]
    assert [bool(CARD.match(v)) for v in pupil_tab] == [True, False] * 3


def test_goal_options_are_every_skill_shown_by_code_and_label(K, G):
    at = open_app()
    box = goal_box(at)
    assert box.value == "(no goal)"
    assert box.options == ["(no goal)"] + [
        f"{G.nodes[n]['code']} · {G.nodes[n]['short_label']}" for n in sorted(G.nodes)
    ]
    assert "strictly" in box.help and MIX in box.help


def test_the_same_pupil_always_gets_the_same_picks(K):
    """Exploration is on (eps > 0) but seeded: two sessions with the same pupil show the same cards."""
    a, b = open_app(), open_app()
    assert cards(a, K) == cards(b, K)
    click(a, LOAD)
    click(b, LOAD)
    assert cards(a, K) == cards(b, K)
    rerun(a)
    assert cards(a, K) == cards(b, K)


def test_cards_are_what_the_engine_picks_with_exploration_on(K):
    """The demo asks the engine with eps = 0.05 (PRD 6.3); a fresh pupil has no dates, so `now` plays no part."""
    at = open_app()
    state = at.session_state["learner"]
    assert card_ids(at, K) == [r.indicator for r in K.recommend(state, k=3, eps=EPS)]
    assert EPS == 0.05
    click(at, LOAD)
    assert card_ids(at, K) == reference_ids(K, at, None, "strict")


# ---------------------------------------------------------------- answering


@pytest.mark.parametrize(
    "label, correct, nth", [(RIGHT, True, 1), (WRONG, False, 2), (RIGHT, True, 0), (WRONG, False, 0)]
)
def test_answer_button_updates_the_skill_on_its_own_card(K, label, correct, nth):
    at = open_app()
    skill = card_ids(at, K)[nth]
    before, table_before = pupil_mastery(at, K), mastery_table(at)

    click(at, label, nth)

    after = pupil_mastery(at, K)
    assert after == pytest.approx(mastery_after(K, [(skill, correct)]), abs=TOL)  # exactly one answer, on that skill
    assert (after[skill] > before[skill]) if correct else (after[skill] < before[skill])
    assert not mastery_table(at).equals(table_before)
    if skill in card_ids(at, K):  # still recommended: its bar shows the new value
        bar = at.get("progress")[card_ids(at, K).index(skill)].proto.text
        assert abs(int(re.fullmatch(r"Mastery so far: (\d+)%", bar).group(1)) - 100 * after[skill]) <= 1


def test_answers_accumulate_and_change_the_recommendations(K):
    at = open_app()
    first, events = card_ids(at, K), []
    for _ in range(4):  # four right answers on the top card
        events.append((card_ids(at, K)[0], True))
        click(at, RIGHT, 0)
    assert pupil_mastery(at, K) == pytest.approx(mastery_after(K, events), abs=TOL)
    assert card_ids(at, K) != first


def test_click_lands_on_the_card_it_was_made_on_even_if_the_cards_change_in_the_same_rerun(K):
    """A goal change and a click on an old card can reach the server together. The click belongs to the skill
    that was on the card, not to whichever skill takes that position after the change."""
    at, probe = open_app(), open_app()
    old = card_ids(at, K)
    for goal in sorted(K.graph.nodes):  # a goal that reshuffles the cards
        goal_box(probe).set_value(goal)
        moved = [i for i, (a, b) in enumerate(zip(old, card_ids(rerun(probe), K))) if a != b]
        if moved:
            break
    else:
        pytest.skip("no goal changes the cards of a new pupil")
    nth = moved[-1]

    goal_box(at).set_value(goal)
    button(at, WRONG, nth).click()
    rerun(at)

    assert pupil_mastery(at, K) == pytest.approx(mastery_after(K, [(old[nth], False)]), abs=TOL)


# ---------------------------------------------------------------- loading and resetting the pupil


def test_load_p4_pupil_changes_mastery_and_keeps_its_message(K):
    at = open_app()
    fresh_table, fresh = mastery_table(at), pupil_mastery(at, K)
    assert not at.success

    click(at, LOAD)
    loaded = pupil_mastery(at, K)
    assert len(at.success) == 1 and LOADED_FORM.fullmatch(at.success[0].value), at.success[0].value
    message = at.success[0].value
    assert not mastery_table(at).equals(fresh_table)
    assert loaded != pytest.approx(fresh)

    at.slider[0].set_value(5)  # any later rerun keeps the message
    rerun(at)
    assert [s.value for s in at.success] == [message]

    click(at, LOAD)  # same pupil again, not a second history
    assert pupil_mastery(at, K) == pytest.approx(loaded, abs=TOL)
    assert [s.value for s in at.success] == [message]


def test_load_replaces_the_current_pupil(K):
    at, other = open_app(), open_app()
    click(at, WRONG, 0)
    click(at, LOAD)
    click(other, LOAD)
    assert pupil_mastery(at, K) == pytest.approx(pupil_mastery(other, K), abs=TOL)


def test_loaded_message_counts_equal_the_seeded_history(K, G):
    at = open_app()
    click(at, LOAD)
    history = list(at.session_state["seeded_history"])
    said = LOADED_FORM.fullmatch(at.success[0].value)
    assert said, at.success[0].value
    repeated = {n for n, _, _ in history if sum(1 for m, _, _ in history if m == n) > 1}
    strong = [(n, ok) for n, ok, _ in history if n in repeated]
    lower = [ok for n, ok, _ in history if G.nodes[n]["grade"] <= 3 and n not in repeated]
    b4 = [ok for n, ok, _ in history if G.nodes[n]["grade"] == 4]
    assert (
        (int(said["all"]), int(said["strong"]), int(said["lower"]), int(said["b4"]))
        == (
            len(history),
            len(repeated),
            len(lower),
            len(b4),
        )
        == (44, 4, 24, 8)
    )
    assert len(strong) + len(lower) + len(b4) == len(history)  # nothing above B4 is seeded
    assert all(ok for _, ok in strong) and all(G.nodes[n]["grade"] in (2, 3) for n, _ in strong)
    assert said["lower_tone"] == "mostly right" and 0.8 <= sum(lower) / len(lower) <= 0.9  # about 85% right
    assert said["b4_tone"] == "mostly wrong" and 0.1 <= sum(b4) / len(b4) <= 0.3  # about 20% right
    stamps = [when for _, _, when in history]
    assert stamps == sorted(stamps)  # oldest answer first
    span = (max(stamps) - min(stamps)).days  # the strong skills were learned the said number of weeks ago
    assert 7 * int(said["weeks"]) - 7 < span <= 7 * int(said["weeks"])

    # the pupil in the session is exactly this history, answered once
    pupil = at.session_state["learner"]
    assert pupil.n["_all"] == len(history)
    assert {n: c for n, c in pupil.n.items() if n != "_all"} == {
        **{n: 1 for n, _, _ in history if n not in repeated},
        **{n: 3 for n in repeated},
    }
    assert pupil_mastery(at, K) == pytest.approx(mastery_after(K, history), abs=TOL)  # replayed with its dates


def test_loaded_pupil_gets_a_review_card(K, G):
    """The skills learned well weeks ago have started to fade: one card brings one back, with the badge."""
    at = open_app()
    assert not [m for m in at.markdown if m.value.startswith(REVIEW_NOTE)]
    click(at, LOAD)
    history = list(at.session_state["seeded_history"])
    repeated = {n for n, _, _ in history if sum(1 for m, _, _ in history if m == n) > 1}
    notes = [m.value for m in at.markdown if m.value.startswith(REVIEW_NOTE)]
    assert len(notes) == 1 and "learned before" in notes[0]
    # the badge sits on exactly the card the engine marks as the review slot, and that is a faded strong skill
    state = at.session_state["learner"]
    latest = max(state.last_seen_at.values())
    recs = K.recommend(state, k=3, now=latest, eps=EPS)
    review = [r.indicator for r in recs if r.is_review]
    assert len(review) == 1 and review[0] in repeated
    lines = [m.value for m in at.markdown if CARD.match(m.value) or m.value.startswith(REVIEW_NOTE)]
    position = [bool(CARD.match(v)) for v in lines].index(False) - 1  # the badge follows its card's title line
    assert K.resolve(CARD.match(lines[position])["code"]) == review[0]


def test_pressing_load_twice_keeps_one_history(K):
    at = open_app()
    click(at, LOAD)
    history, once = list(at.session_state["seeded_history"]), pupil_mastery(at, K)
    click(at, RIGHT, 0)  # the pupil moves on, then the button is pressed again
    click(at, LOAD)
    assert [(n, ok) for n, ok, _ in at.session_state["seeded_history"]] == [(n, ok) for n, ok, _ in history]
    assert at.session_state["learner"].n["_all"] == len(history) == 44
    assert pupil_mastery(at, K) == pytest.approx(once, abs=TOL)
    click(at, LOAD)
    assert at.session_state["learner"].n["_all"] == 44


def test_start_over_returns_to_the_initial_recommendations(K):
    at = open_app()
    first_cards, first_whys, first_bars = cards(at, K), whys(at), [b.proto.text for b in at.get("progress")]
    first_table, first = mastery_table(at), pupil_mastery(at, K)

    click(at, LOAD)
    click(at, RIGHT, 0)
    click(at, WRONG, 1)
    assert pupil_mastery(at, K) != pytest.approx(first)

    click(at, RESET)
    assert cards(at, K) == first_cards and whys(at) == first_whys
    assert [b.proto.text for b in at.get("progress")] == first_bars
    assert mastery_table(at).equals(first_table)
    assert pupil_mastery(at, K) == pytest.approx(first)
    assert not at.success  # the "Loaded ..." note goes with the pupil


# ---------------------------------------------------------------- class cap, goal, toggles, language


def test_class_cap_limits_the_grade_of_every_card(K, G):
    at = open_app()
    click(at, LOAD)  # a pupil who is offered work above B1
    top = max(g for _, g in cards(at, K))
    for cap in sorted({6, 5, 4, 3, 2, 1, max(top - 1, 1)}, reverse=True):
        at.slider[0].set_value(cap)
        rerun(at)
        shown = cards(at, K)
        assert shown, f"no card at class cap {cap}"
        assert all(g <= cap and G.nodes[i]["grade"] <= cap for i, g in shown), (cap, shown)


def test_goal_changes_the_recommendations_and_the_reason_mentions_it(K, G):
    at = open_app()
    no_goal = card_ids(at, K)
    assert not any("goal" in w.lower() for w in whys(at))

    for goal in sorted(G.nodes):  # first goal that pulls a new skill in
        goal_box(at).set_value(goal)
        if card_ids(rerun(at), K) != no_goal and any("goal" in w.lower() for w in whys(at)):
            break
    else:
        pytest.fail("no goal skill changes the recommendations of a new pupil and is named in a reason")

    towards = [i for i, w in zip(card_ids(at, K), whys(at)) if "goal" in w.lower()]
    assert towards and set(towards) <= nx.ancestors(G, goal)  # only skills that lead to the goal say so

    goal_box(at).set_value("(no goal)")
    assert card_ids(rerun(at), K) == no_goal


def reference_ids(K, at, goal, goal_mode):
    """What the engine recommends for the pupil in the session, asked directly with the demo's settings."""
    state = at.session_state["learner"]
    latest = max(state.last_seen_at.values()) if state.last_seen_at else None
    return [r.indicator for r in K.recommend(state, k=3, goal=goal, goal_mode=goal_mode, now=latest, eps=EPS)]


def test_goal_is_followed_strictly_unless_other_skills_are_mixed_in(K, G):
    at = open_app()
    click(at, LOAD)
    mix = toggle(at, MIX)
    assert mix.value is False

    for goal in sorted(G.nodes):  # first goal where the two modes pick different cards
        if reference_ids(K, at, goal, "strict") != reference_ids(K, at, goal, "soft"):
            break
    else:
        pytest.fail("no goal separates strict from soft for the loaded pupil")
    path = nx.ancestors(G, goal) | {goal}

    goal_box(at).set_value(goal)
    strict = card_ids(rerun(at), K)
    assert strict == reference_ids(K, at, goal, "strict")
    on = [n in path for n in strict]
    assert on == sorted(on, reverse=True)  # every card on the path comes before any card off it

    toggle(at, MIX).set_value(True)
    soft = card_ids(rerun(at), K)
    assert soft == reference_ids(K, at, goal, "soft") != strict

    toggle(at, MIX).set_value(False)
    assert card_ids(rerun(at), K) == strict


def test_mix_toggle_changes_nothing_without_a_goal(K):
    at = open_app()
    before = cards(at, K)
    toggle(at, MIX).set_value(True)
    assert cards(rerun(at), K) == before


def test_draft_settings_toggle_is_off_by_default_and_labelled_as_unreviewed(K):
    at = open_app()
    drafts = toggle(at, DRAFTS)
    assert drafts.value is False
    assert "no teacher has checked" in drafts.help
    assert not at.warning
    before, before_settings = card_ids(at, K), settings(at)

    drafts.set_value(True)
    rerun(at)
    assert [w.value for w in at.warning] == [
        "Draft settings are on. The 'Try it in Ghana' text may not have been checked by a teacher."
    ]
    assert card_ids(at, K) == before  # drafts change the practice setting text, never the picks
    state = at.session_state["learner"]
    expected = [r.ghana_context for r in K.recommend(state, k=3, eps=EPS, include_drafts=True)]
    assert [s.removeprefix("**Try it in Ghana:** ") for s in settings(at)] == expected

    toggle(at, DRAFTS).set_value(False)
    rerun(at)
    assert not at.warning and settings(at) == before_settings


def test_language_picker_offers_reviewed_languages_only_and_says_why(K):
    at = open_app()
    box = language_box(at)
    reviewed = {"en"} | {e["lang"] for entries in K.bundle.contexts.values() for e in entries if e.get("reviewed")}
    names = {"en": "English", "tw": "Twi", "ee": "Ewe", "gaa": "Ga"}
    assert set(box.options) == {names.get(code, code) for code in reviewed} and box.value == "en"
    why = [c.value for c in at.caption if c.value.startswith("Only languages with practice settings checked")]
    assert len(why) == 1
    assert ("today that is English alone" in why[0]) == (reviewed == {"en"})


def test_counts_in_the_copy_are_pluralised_by_one_helper():
    source = APP.read_text(encoding="utf-8")
    namespace = {}
    exec(source[source.index("def plural(") : source.index("def p4_history(")], namespace)
    plural = namespace["plural"]
    assert plural(1, "skill") == "1 skill" and plural(2, "skill") == "2 skills" and plural(0, "skill") == "0 skills"
    assert plural(1, "past answer") == "1 past answer" and plural(32, "past answer") == "32 past answers"
    # every count that reaches the page goes through it: no hand-written "N skills" or "(s)" in the demo copy
    assert "(s)" not in source
    assert not re.search(r"\}\s+(skills?|answers?|pupils?|turns?|steps?|weeks?|days?)\b", source)


# ---------------------------------------------------------------- evidence tab


def block_at(ev, key):
    node = ev
    for part in key.split("/"):
        node = node.get(part) if isinstance(node, dict) else None
    return node if isinstance(node, dict) else None


def file_auc(block, model):
    row = (block.get("metrics") or {}).get(model) or {}
    value = row.get("AUC")
    return value if value is not None else (block.get("AUC") or {}).get(model)


def test_evidence_shows_auc_per_model_per_dataset_from_the_results_file():
    at, ev = open_app(), results.load()
    auc = auc_table(at)
    present = [
        (key, name) for key, name in DATASET_NAMES.items() if block_at(ev, key) and not block_at(ev, key).get("skipped")
    ]
    assert auc["Dataset"].tolist() == [name for _, name in present]
    assert list(auc.columns)[:2] == ["Dataset", "Answers"] and list(auc.columns)[2:7] == AUC_MODELS
    for (key, _name), (_, row) in zip(present, auc.iterrows()):
        block = block_at(ev, key)
        assert re.fullmatch(r"\d+k|\d+\.\d\dM|\d+", row["Answers"]) and row["Answers"] != "0"
        for model in auc.columns[2:]:
            want = file_auc(block, model)
            assert math.isnan(row[model]) if want is None else row[model] == pytest.approx(want), (key, model)
    if "ammore_2024" in ev:
        assert {"AMMORE 2024 (all)", "AMMORE 2024 (Ghana)"} <= set(auc["Dataset"])
    missing = [name for key, name in DATASET_NAMES.items() if block_at(ev, key) is None]
    said = [c.value for c in at.caption if c.value.startswith("Not run yet: ")]
    assert said == ([f"Not run yet: {', '.join(missing)}."] if missing else [])
    assert any("AUC is 0.5 for a coin flip and 1.0 for perfect prediction." in m.value for m in at.markdown)


def complete_results():
    """The real results file with every block the Evidence tab reads filled in."""
    ev = results.load()
    for key in DATASET_NAMES:
        top, _, sub = key.partition("/")
        ev.setdefault(top, {})
        if sub:
            ev[top].setdefault(sub, {"rows": 1200, "metrics": {"elo": {"AUC": 0.61}, KWAN_MODEL: {"AUC": 0.7}}})
        elif "metrics" not in ev[top] and "AUC" not in ev[top]:
            ev[top] = {"rows": 1200, "metrics": {"elo": {"AUC": 0.61}, KWAN_MODEL: {"AUC": 0.7}}}
    ev.setdefault(
        "learner_fit",
        {
            "protocol": "p",
            "datasets": {"assistments_2009": {"timestamps": False, "tau_chosen": None, "auc_chosen": 0.75}},
            "shipped": {"params": {"k_s": 0.3, "k_k": 0.4, "tau_days": 30.0}},
            "acceptance": {"rule": "r", "passed": True},
        },
    )
    ev.setdefault("kt_deep", {})
    ev["kt_deep"].setdefault(
        "decision", {"rule": "a rule", "adopt": False, "per_model": {"dkt": {"margins": {"ammore": -0.01}}}}
    )
    ev.setdefault("goal_path", {"setup": "s", "on_path_share": {"strict": 0.8, "soft": 0.4}})
    ev.setdefault("latency", {"recommend_top3_mean_ms": 0.2, "recommend_top3_p95_ms": 0.3, "calls": 10})
    ev.setdefault("simulation", {})
    ev["simulation"].setdefault(
        "v2",
        {
            "setup": "s",
            "new_skills_mastered": {"gapped_p4": {"kwan-strict": {"mean": 9.0, "sd": 1.0}}},
            "skills_retained": {"gapped_p4": {"kwan-strict": {"mean": 8.0, "sd": 1.0}}},
            "steps_to_goal": {"gapped_p4": {"kwan-strict": {"mean": 30.0, "sd": 2.0, "reached": 1.0}}},
            "acceptance": {"met": True},
        },
    )
    return ev


def test_evidence_with_every_block_present_shows_each_section_and_never_says_not_run(monkeypatch):
    ev = complete_results()
    fake_results(monkeypatch, ev)
    at = open_app()
    assert not [t for t in rendered_text(at) if NOT_RUN in t.lower()]
    auc = auc_table(at)
    assert auc["Dataset"].tolist() == list(DATASET_NAMES.values())
    assert len(table(at, has_column="Licence")) == len(DATASET_NAMES)
    fit = table(at, has_column="Tuned settings")
    assert len(fit) == len(ev["learner_fit"]["datasets"])
    shipped = table(at, has_column="Setting")
    assert dict(zip(shipped["Setting"], shipped["Value"])) == {
        k: str(v) for k, v in ev["learner_fit"]["shipped"]["params"].items()
    }
    deep = table(at, has_column="meets the rule")
    assert deep["Model"].tolist() == list(ev["kt_deep"]["decision"]["per_model"])
    verdict = [m.value for m in at.markdown if m.value.startswith("Decision: ")]
    assert len(verdict) == 1 and verdict[0].startswith("Decision: adopt") == bool(ev["kt_deep"]["decision"]["adopt"])
    metrics = {m.label: m.value for m in at.metric}
    share = ev["goal_path"]["on_path_share"]
    assert metrics["On the path to the goal (strict)"] == f"{share['strict']:.0%}"
    assert metrics["On the path with other skills mixed in"] == f"{share['soft']:.0%}"
    assert metrics["Top 3 picks, average"] == f"{ev['latency']['recommend_top3_mean_ms']} ms"
    sim = table(at, has_column="still known after the break")
    v2 = ev["simulation"]["v2"]
    assert len(sim) == sum(len(p) for p in v2["new_skills_mastered"].values())
    assert set(sim["pupil type"]) <= {
        "Beginner in P1",
        "P4 pupil with gaps in B1 to B3",
        "P6 pupil with gaps in B1 to B5",
        "P4 pupil weak in one strand",
    }
    assert any(c.value.endswith("is met." if v2["acceptance"]["met"] else "is not met.") for c in at.caption)


def broken_json(*a, **k):
    raise ValueError("eval_results.json is not valid JSON")


@pytest.mark.parametrize(
    "fake",
    [
        {},  # file missing: results.load() gives {}
        {"simulation": {"setup": "only this block"}, "kt_deep": {"task": "no decision yet"}},  # v1 shapes only
        {"assistments_2009": {"rows": 10, "AUC": {"elo": 0.6}}},  # one block, most models missing
        {"assistments_2009": {"rows": 10}, "eedi_2020_task34": {"AUC": None}, "ammore_2024": {"all": {}}},
        {"assistments_2009": None, "eedi_2020_task34": [], "latency": "fast", "goal_path": {"on_path_share": 1}},
        {"junyi_2015": {"skipped": "download failed"}},
        broken_json,
    ],
    ids=["empty", "no-v2-blocks", "one-model", "no-auc", "wrong-shape", "skipped", "unreadable"],
)
def test_evidence_says_not_run_yet_for_missing_blocks_and_never_crashes(monkeypatch, fake, K):
    fake_results(monkeypatch, fake)
    at = open_app()
    given = {} if callable(fake) else fake
    auc = auc_table(at)
    assert list(auc.columns)[:7] == ["Dataset", "Answers", *AUC_MODELS]
    shown = [
        (key, name)
        for key, name in DATASET_NAMES.items()
        if block_at(given, key) is not None and not block_at(given, key).get("skipped")
    ]
    assert auc["Dataset"].tolist() == [name for _, name in shown]
    for (key, _), (_, row) in zip(shown, auc.iterrows()):
        have = block_at(given, key)
        for model in AUC_MODELS:
            want = file_auc(have, model)
            assert math.isnan(row[model]) if want is None else row[model] == want  # empty, never invented
    text = rendered_text(at)
    missing = [name for key, name in DATASET_NAMES.items() if block_at(given, key) is None]
    if missing:
        assert f"Not run yet: {', '.join(missing)}." in text
    for key, label in [
        ("learner_fit", "Forgetting and tuned parameters"),
        ("kt_deep/decision", "Deep knowledge tracing benchmark"),
        ("goal_path/on_path_share", "Goal path check"),
        ("simulation/v2", "Simulator v2"),
    ]:
        if block_at(given, key) is None:
            assert f"{label}: {NOT_RUN}." in text
    if not isinstance(
        (given.get("latency") or {}).get("recommend_top3_mean_ms") if isinstance(given.get("latency"), dict) else None,
        (int, float),
    ):
        assert f"Latency benchmark: {NOT_RUN}." in text
    skipped = [name for key, name in DATASET_NAMES.items() if (block_at(given, key) or {}).get("skipped")]
    for name in skipped:
        assert any(t.startswith(f"Skipped. {name}: ") for t in text)
    assert len(cards(at, K)) == 3  # the rest of the page is unaffected


def test_bkt_note_is_shown_only_when_bkt_ran_on_some_datasets_but_not_all(monkeypatch):
    both = {k: {"rows": 5000, "AUC": {"bkt": 0.7, "elo": 0.6}} for k in ("assistments_2009", "eedi_2020_task34")}
    fake_results(monkeypatch, both)
    assert not any("BKT was run on" in c.value for c in open_app().caption)
    only_a = {**both, "eedi_2020_task34": {"rows": 5000, "AUC": {"elo": 0.6}}}
    fake_results(monkeypatch, only_a)
    assert any(
        c.value.endswith("the classic baseline. BKT was run on ASSISTments 2009 only.") for c in open_app().caption
    )


def test_simulation_button_runs_simulator_v2_with_readable_labels(G):
    at = open_app(timeout=600)
    sims = [d for d in at.dataframe if "still known after the break" in d.value.columns]
    click(at, SIMULATE, timeout=600)
    after = [d.value for d in at.dataframe if "still known after the break" in d.value.columns]
    assert len(after) == len(sims) + 1
    sim = after[-1]
    assert list(sim.columns) == [
        "pupil type",
        "how they practised",
        "new skills mastered",
        "still known after the break",
        "steps to reach a goal",
        "reached the goal",
    ]
    assert len(sim) == len(simulate.PROFILES_V2) * len(simulate.POLICIES_V2)
    assert sorted(set(sim["pupil type"])) == sorted(
        [
            "Beginner in P1",
            "P4 pupil with gaps in B1 to B3",
            "P6 pupil with gaps in B1 to B5",
            "P4 pupil weak in one strand",
        ]
    )
    assert sorted(set(sim["how they practised"])) == sorted(
        [
            "Following Kwan",
            "Following Kwan, with exploration",
            "Following Kwan, top pick only",
            "Syllabus order",
            "Random practice",
            "Class syllabus, gaps first",
        ]
    )
    for cell in sim["new skills mastered"]:
        mean, sd = (float(x) for x in cell.split(" ± "))
        assert 0 <= mean <= G.number_of_nodes() and sd >= 0
    assert all(re.fullmatch(r"\d+%", v) for v in sim["reached the goal"])

    click(at, RIGHT, 0)  # the table survives later clicks
    assert [d.value for d in at.dataframe if "still known after the break" in d.value.columns][-1].equals(sim)


# ---------------------------------------------------------------- house rules


def test_disclaimer_is_in_the_footer():
    at = open_app()
    assert at.markdown[-1].value == DISCLAIMER
    assert at.caption[-1].value.startswith("Bundle ")


def test_no_em_dash_anywhere_on_the_page(K, G):
    at = open_app(timeout=600)
    seen = rendered_text(at)
    click(at, LOAD)
    click(at, WRONG, 0)
    toggle(at, DRAFTS).set_value(True)
    goal_box(at).set_value(sorted(G.nodes)[-1])
    at.slider[0].set_value(4)
    rerun(at)
    click(at, SIMULATE, timeout=600)
    seen += rendered_text(at)
    assert len(seen) > 600  # the collector really sees the page
    assert not [t for t in seen if EM_DASH in t]
    for name in ("app.py", "README.md", "Dockerfile", "smoke_test.py", "requirements-demo.txt"):
        assert EM_DASH not in (ROOT / "demo" / name).read_text(encoding="utf-8"), name


ALLOWED_IMPORTS = {
    "kwan": {"Kwan"},
    "kwan.eval": {"results", "simulate"},
    "kwan.eval.results": None,
    "kwan.eval.simulate": None,
    "kwan.graph.export": None,
    "pandas": None,
    "streamlit": None,
    "streamlit.components.v1": None,
}


def test_app_imports_only_the_public_api_and_the_allowed_modules():
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.partition(".")[0]
                assert top in sys.stdlib_module_names or alias.name in ALLOWED_IMPORTS, alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.partition(".")[0] in sys.stdlib_module_names:
                continue
            assert module in ALLOWED_IMPORTS, module
            allowed = ALLOWED_IMPORTS[module]
            if allowed is not None:
                assert {a.name for a in node.names} <= allowed, (module, [a.name for a in node.names])
    source = APP.read_text(encoding="utf-8")
    assert "kwan.learner" not in source and "kwan.recommend" not in source and "kwan.bundle" not in source


class _Records(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@pytest.fixture
def streamlit_warnings():
    """Everything Streamlit logs at WARNING or above while the test runs (its loggers do not all propagate)."""
    handler = _Records()
    loggers = [
        logging.getLogger(n)
        for n in list(logging.root.manager.loggerDict)
        if n == "streamlit" or n.startswith("streamlit.")
    ]
    for lg in loggers:
        lg.addHandler(handler)
    yield handler.messages
    for lg in loggers:
        lg.removeHandler(handler)


def test_no_deprecation_noise_from_streamlit(streamlit_warnings):
    at = open_app(timeout=600)
    click(at, LOAD)
    click(at, SIMULATE, timeout=600)
    noise = [
        m
        for m in streamlit_warnings
        if "deprecat" in m.lower() or "please replace" in m.lower() or "will be removed" in m.lower()
    ]
    assert not noise, noise

    # the check above is only worth something if this handler really hears Streamlit's deprecation logger
    probe = AppTest.from_string("import streamlit as st\nst.dataframe([1], use_container_width=True)").run()
    assert not probe.exception
    import inspect

    import streamlit as st

    if inspect.signature(st.dataframe).parameters["width"].default == "stretch":  # versions that deprecate it
        assert any("use_container_width" in m for m in streamlit_warnings)


def test_mastery_table_cells_are_coloured_with_readable_text():
    """The class-by-strand table colours cells by band; every fill carries dark text so contrast stays AA."""
    source = APP.read_text(encoding="utf-8")
    namespace = {"pd": pd}
    exec(source[source.index("BANDS = ") : source.index("CELL_TEXT = ")], namespace)
    exec(source[source.index("CELL_TEXT = ") : source.index("\n", source.index("CELL_TEXT = "))], namespace)
    exec(source[source.index("def cell_style(") : source.index("def styled(")], namespace)
    cell_style = namespace["cell_style"]
    assert cell_style(float("nan")) == ""
    fills = {cell_style(v).split(";")[0] for v in (0.0, 0.49, 0.5, 0.69, 0.7, 0.84, 0.85, 1.0)}
    assert len(fills) == 4
    for v in (0.0, 0.5, 0.7, 0.85, 1.0):
        assert cell_style(v).endswith(f"color: {namespace['CELL_TEXT']}")
        fill = re.search(r"background-color: (#[0-9a-f]{6})", cell_style(v)).group(1)
        assert contrast(fill, namespace["CELL_TEXT"]) >= 4.5, (v, fill)


def contrast(a, b):
    """WCAG contrast ratio of two hex colours."""

    def lum(h):
        rgb = [int(h[i : i + 2], 16) / 255 for i in (1, 3, 5)]
        lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
        return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]

    la, lb = sorted((lum(a), lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)
