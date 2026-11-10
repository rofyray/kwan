"""Headless tests for the Streamlit demo (app.py), driven through streamlit.testing.v1.AppTest.

They follow the acceptance checks of PRD Phase 8.1 and the UI contract of PRD Section 5.5. Nothing here asserts
on a curriculum description or on a metric value: those belong to data/graph.json and data/eval_results.json,
so every expectation is derived from the data files or from the engine at test time.
"""
import ast
import logging
import math
import re
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import engine
import graph
import results

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "app.py")

TABS = ["Pupil view", "Curriculum map", "Evidence"]
LOAD, RESET = "Load a P4 pupil with gaps in earlier classes", "Start over with a new pupil"
RIGHT, WRONG = "✅ Got it right", "❌ Got it wrong"
SIMULATE = "Run the simulation (about 10 seconds)"
LOADED_MSG = "Loaded about 20 past answers: strong on most B1 to B3 skills, weak on B4."
MODEL_HEADER = "Model (AUC, higher is better)"
AUC_MODELS = ["global-mean", "skill-mean", "bkt", "elo", "elo-multi (Kwan)"]
CARD = re.compile(r"^\*\*(?P<id>B\d[\d.]*)\*\* · B(?P<grade>\d) · ")
EM_DASH = chr(0x2014)


# ---------------------------------------------------------------- helpers

@pytest.fixture(scope="module")
def G():
    return graph.load()


def open_app(timeout=30):
    at = AppTest.from_file(APP, default_timeout=timeout).run()
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


def cards(at):
    """The recommendation cards in page order: [(indicator id, grade)]."""
    return [(m["id"], int(m["grade"])) for m in (CARD.match(x.value) for x in at.markdown) if m]


def card_ids(at):
    return [i for i, _ in cards(at)]


def whys(at):
    return [m.value for m in at.markdown if m.value.startswith("**Why this skill:**")]


def table(at, has_column=None, index_name=None):
    """The one st.dataframe that has this column (or this index name)."""
    found = [d.value for d in at.dataframe
             if (has_column is not None and has_column in list(d.value.columns))
             or (index_name is not None and d.value.index.name == index_name)]
    assert len(found) == 1, f"expected one table with {has_column or index_name!r}, found {len(found)}"
    return found[0]


def mastery_table(at):
    return table(at, index_name="strand")


def pupil_mastery(at, G):
    """Mastery of every skill for the pupil held in the session, read through the engine's public interface."""
    L = engine.EloLearner(G=G)
    return {n: L.mastery(at.session_state["learner"], n) for n in G}


def mastery_after(G, events):
    """Mastery of every skill for a new pupil who gave exactly these (indicator, correct) answers."""
    L, S = engine.EloLearner(G=G), engine.LearnerState("expected")
    for ind, correct in events:
        L.update(S, ind, correct)
    return {n: L.mastery(S, n) for n in G}


def rendered_text(at):
    """Every piece of text the page shows: prose, labels, options, table headers and table cells."""
    out = [x.value for kind in ("title", "header", "subheader", "markdown", "caption", "info", "success", "warning", "error")
           for x in getattr(at, kind)]
    out += [t.label for t in at.tabs] + [b.label for b in at.button] + [p.proto.text for p in at.get("progress")]
    out += [w.label for kind in ("slider", "selectbox") for w in getattr(at, kind)]
    out += [str(o) for s in at.selectbox for o in s.options]
    out += [f"{m.label} {m.value}" for m in at.metric]
    for d in at.dataframe:
        df = d.value
        out += [str(c) for c in df.columns] + [str(i) for i in df.index] + [str(df.index.name)]
        out += [v for col in df.columns for v in df[col].tolist() if isinstance(v, str)]
    return [t for t in out if t]


# ---------------------------------------------------------------- page structure

def test_app_runs_and_shows_the_three_tabs(G):
    at = open_app()
    assert [t.label for t in at.tabs] == TABS
    assert at.title[0].value == "Kwan 🧭"
    assert [s.value for s in at.subheader] == ["What to learn next", "Mastery by class and strand"]


def test_header_skill_count_comes_from_the_graph(G):
    at = open_app()
    assert at.caption[0].value == (f"NaCCA B1 to B6 Mathematics · {G.number_of_nodes()} skills · "
                                   "runs on one CPU core in under a millisecond")
    assert {m.label: m.value for m in at.metric} == {"Skills (NaCCA indicators)": str(G.number_of_nodes()),
                                                    "Prerequisite links": str(G.number_of_edges())}
    assert len(table(at, has_column="type")) == G.number_of_edges()


def test_three_cards_each_with_reason_and_ghana_setting(G):
    at = open_app()
    shown = cards(at)
    assert len(shown) == 3 and len(set(shown)) == 3
    assert all(i in G and G.nodes[i]["grade"] == g for i, g in shown)
    assert len(whys(at)) == 3
    assert len(at.info) == 3 and all(i.value.startswith("**Try it in Ghana:** ") and len(i.value) > 30 for i in at.info)
    bars = at.get("progress")
    assert len(bars) == 3 and all(re.fullmatch(r"Mastery so far: \d+%", b.proto.text) for b in bars)
    assert [b.label for b in at.button if b.label in (RIGHT, WRONG)] == [RIGHT, WRONG] * 3
    # each card is drawn as: skill line, then its reason
    pupil_tab = [m.value for m in at.markdown if CARD.match(m.value) or m.value.startswith("**Why this skill:**")]
    assert [bool(CARD.match(v)) for v in pupil_tab] == [True, False] * 3


# ---------------------------------------------------------------- answering

@pytest.mark.parametrize("label, correct, nth", [(RIGHT, True, 1), (WRONG, False, 2), (RIGHT, True, 0), (WRONG, False, 0)])
def test_answer_button_updates_the_skill_on_its_own_card(G, label, correct, nth):
    at = open_app()
    skill = card_ids(at)[nth]
    before, table_before = pupil_mastery(at, G), mastery_table(at)

    click(at, label, nth)

    after = pupil_mastery(at, G)
    assert after == pytest.approx(mastery_after(G, [(skill, correct)]))      # exactly one answer, on that skill
    assert (after[skill] > before[skill]) if correct else (after[skill] < before[skill])
    assert not mastery_table(at).equals(table_before)
    if skill in card_ids(at):                                                # still recommended: its bar shows the new value
        bar = at.get("progress")[card_ids(at).index(skill)].proto.text
        assert abs(int(re.fullmatch(r"Mastery so far: (\d+)%", bar).group(1)) - 100 * after[skill]) <= 1


def test_answers_accumulate_and_change_the_recommendations(G):
    at = open_app()
    first, events = card_ids(at), []
    for _ in range(4):                                                       # four right answers on the top card
        events.append((card_ids(at)[0], True))
        click(at, RIGHT, 0)
    assert pupil_mastery(at, G) == pytest.approx(mastery_after(G, events))
    assert card_ids(at) != first


def test_click_lands_on_the_card_it_was_made_on_even_if_the_cards_change_in_the_same_rerun(G):
    """A goal change and a click on an old card can reach the server together. The click belongs to the skill
    that was on the card, not to whichever skill takes that position after the change."""
    at, probe = open_app(), open_app()
    old = card_ids(at)
    for goal in probe.selectbox[0].options[1:]:                              # a goal that reshuffles the cards
        probe.selectbox[0].set_value(goal)
        moved = [i for i, (a, b) in enumerate(zip(old, card_ids(rerun(probe)))) if a != b]
        if moved:
            break
    else:
        pytest.skip("no goal changes the cards of a new pupil")
    nth = moved[-1]

    at.selectbox[0].set_value(goal)
    button(at, WRONG, nth).click()
    rerun(at)

    assert pupil_mastery(at, G) == pytest.approx(mastery_after(G, [(old[nth], False)]))


# ---------------------------------------------------------------- loading and resetting the pupil

def test_load_p4_pupil_changes_mastery_and_keeps_its_message(G):
    at = open_app()
    fresh_table, fresh = mastery_table(at), pupil_mastery(at, G)
    assert not at.success

    click(at, LOAD)
    loaded = pupil_mastery(at, G)
    assert [s.value for s in at.success] == [LOADED_MSG]
    assert not mastery_table(at).equals(fresh_table)
    assert loaded != pytest.approx(fresh)

    at.slider[0].set_value(5)                                                # any later rerun keeps the message
    rerun(at)
    assert [s.value for s in at.success] == [LOADED_MSG]

    click(at, LOAD)                                                          # same pupil again, not a second history
    assert pupil_mastery(at, G) == pytest.approx(loaded)
    assert [s.value for s in at.success] == [LOADED_MSG]


def test_load_replaces_the_current_pupil(G):
    at, other = open_app(), open_app()
    click(at, WRONG, 0)
    click(at, LOAD)
    click(other, LOAD)
    assert pupil_mastery(at, G) == pytest.approx(pupil_mastery(other, G))


def test_start_over_returns_to_the_initial_recommendations(G):
    at = open_app()
    first_cards, first_whys, first_bars = cards(at), whys(at), [b.proto.text for b in at.get("progress")]
    first_table, first = mastery_table(at), pupil_mastery(at, G)

    click(at, LOAD)
    click(at, RIGHT, 0)
    click(at, WRONG, 1)
    assert pupil_mastery(at, G) != pytest.approx(first)

    click(at, RESET)
    assert cards(at) == first_cards and whys(at) == first_whys
    assert [b.proto.text for b in at.get("progress")] == first_bars
    assert mastery_table(at).equals(first_table)
    assert pupil_mastery(at, G) == pytest.approx(first)
    assert not at.success                                                    # the "Loaded ..." note goes with the pupil


# ---------------------------------------------------------------- class cap and goal

def test_class_cap_limits_the_grade_of_every_card(G):
    at = open_app()
    click(at, LOAD)                                                          # a pupil who is offered work above B1
    top = max(g for _, g in cards(at))
    for cap in sorted({6, 5, 4, 3, 2, 1, max(top - 1, 1)}, reverse=True):
        at.slider[0].set_value(cap)
        rerun(at)
        shown = cards(at)
        assert shown, f"no card at class cap {cap}"
        assert all(g <= cap and G.nodes[i]["grade"] <= cap for i, g in shown), (cap, shown)


def test_goal_changes_the_recommendations_and_the_reason_mentions_it(G):
    import networkx as nx
    at = open_app()
    no_goal = card_ids(at)
    assert not any("goal" in w.lower() for w in whys(at))
    assert at.selectbox[0].options == ["(no goal)"] + sorted(G.nodes)

    for goal in at.selectbox[0].options[1:]:                                 # first goal that pulls a new skill in
        at.selectbox[0].set_value(goal)
        if card_ids(rerun(at)) != no_goal and any("goal" in w.lower() for w in whys(at)):
            break
    else:
        pytest.fail("no goal skill changes the recommendations of a new pupil and is named in a reason")

    towards = [i for i, w in zip(card_ids(at), whys(at)) if "goal" in w.lower()]
    assert towards and set(towards) <= nx.ancestors(G, goal)                 # only skills that lead to the goal say so

    at.selectbox[0].set_value("(no goal)")
    assert card_ids(rerun(at)) == no_goal


# ---------------------------------------------------------------- evidence tab

def test_evidence_shows_the_auc_table_from_the_results_file():
    at, ev = open_app(), results.load()
    auc = table(at, has_column=MODEL_HEADER)
    assert auc[MODEL_HEADER].tolist() == AUC_MODELS
    heads = list(auc.columns)
    assert len(heads) == 3
    assert re.fullmatch(r"ASSISTments 2009 \(\S+ answers\)", heads[1]) and re.fullmatch(r"Eedi 2020 \(\S+ answers\)", heads[2])
    for head, key in zip(heads[1:], ("assistments_2009", "eedi_2020_task34")):
        for model, shown in zip(AUC_MODELS, auc[head].tolist()):
            want = ev[key]["AUC"].get(model)
            assert (math.isnan(shown) if want is None else shown == pytest.approx(want)), (key, model, shown, want)
    assert any("AUC is 0.5 for a coin flip and 1.0 for perfect prediction." in m.value for m in at.markdown)


@pytest.mark.parametrize("rows_a, rows_e, head_a, head_e", [
    (259399, 1382727, "ASSISTments 2009 (259k answers)", "Eedi 2020 (1.38M answers)"),   # the wording of PRD A.8
    (48210, 2000000, "ASSISTments 2009 (48k answers)", "Eedi 2020 (2.00M answers)"),
    (512, None, "ASSISTments 2009 (512 answers)", "Eedi 2020"),
])
def test_evidence_headers_take_dataset_sizes_from_the_results_file(monkeypatch, rows_a, rows_e, head_a, head_e):
    fake = {"assistments_2009": {"rows": rows_a, "AUC": {"elo": 0.61}}, "eedi_2020_task34": {"rows": rows_e, "AUC": {"elo": 0.62}}}
    monkeypatch.setattr(results, "load", lambda *a, **k: fake)
    auc = table(open_app(), has_column=MODEL_HEADER)
    assert list(auc.columns) == [MODEL_HEADER, head_a, head_e]
    assert auc.set_index(MODEL_HEADER).loc["elo"].tolist() == [0.61, 0.62]


def broken_json(*a, **k):
    raise ValueError("eval_results.json is not valid JSON")


@pytest.mark.parametrize("fake", [
    {},                                                                              # file missing: results.load() gives {}
    {"simulation": {"setup": "only this block"}},                                    # both dataset blocks missing
    {"assistments_2009": {"rows": 10, "AUC": {"elo": 0.6}}},                         # one block, most models missing
    {"assistments_2009": {"rows": 10}, "eedi_2020_task34": {"AUC": None}},           # blocks without AUC
    {"assistments_2009": None, "eedi_2020_task34": []},                              # blocks of the wrong shape
    broken_json,
], ids=["empty", "no-datasets", "one-model", "no-auc", "wrong-shape", "unreadable"])
def test_evidence_does_not_crash_on_missing_blocks_or_models(monkeypatch, fake):
    monkeypatch.setattr(results, "load", fake if callable(fake) else (lambda *a, **k: fake))
    at = open_app()
    auc = table(at, has_column=MODEL_HEADER)
    assert auc[MODEL_HEADER].tolist() == AUC_MODELS and auc.shape == (5, 3)
    given = {} if callable(fake) else fake
    for head, key in zip(auc.columns[1:], ("assistments_2009", "eedi_2020_task34")):
        block = given.get(key) if isinstance(given.get(key), dict) else {}
        have = block.get("AUC") or {}
        assert [None if pd.isna(v) else v for v in auc[head]] == [have.get(m) for m in AUC_MODELS]   # empty, never invented
    assert len(cards(at)) == 3                                                       # the rest of the page is unaffected


def test_bkt_note_is_shown_only_when_it_is_true(monkeypatch):
    both = {k: {"rows": 5000, "AUC": {"bkt": 0.7, "elo": 0.6}} for k in ("assistments_2009", "eedi_2020_task34")}
    monkeypatch.setattr(results, "load", lambda *a, **k: both)
    assert not any("BKT was run on ASSISTments only." in c.value for c in open_app().caption)
    only_a = {**both, "eedi_2020_task34": {"rows": 5000, "AUC": {"elo": 0.6}}}
    monkeypatch.setattr(results, "load", lambda *a, **k: only_a)
    assert any(c.value.endswith("the classic baseline. BKT was run on ASSISTments only.") for c in open_app().caption)


def test_simulation_button_shows_six_labelled_rows(G):
    at = open_app(timeout=180)
    assert not [d for d in at.dataframe if "pupil type" in d.value.columns]
    click(at, SIMULATE, timeout=180)
    sim = table(at, has_column="pupil type")
    assert list(sim.columns) == ["pupil type", "how they practised", "new skills mastered (avg)", "spread (sd)"]
    assert len(sim) == 6
    assert sorted(sim["how they practised"]) == sorted(["Following Kwan", "Syllabus order", "Random practice"] * 2)
    assert sim["pupil type"].value_counts().to_dict() == {"Beginner in P1": 3, "P4 pupil with gaps in B1 to B3": 3}
    assert sim["new skills mastered (avg)"].between(0, G.number_of_nodes()).all() and (sim["spread (sd)"] >= 0).all()

    click(at, RIGHT, 0)                                                              # the table survives later clicks
    assert table(at, has_column="pupil type").equals(sim)


# ---------------------------------------------------------------- house rules

def test_no_em_dash_anywhere_on_the_page(G):
    at = open_app(timeout=180)
    seen = rendered_text(at)
    click(at, LOAD)
    click(at, WRONG, 0)
    at.selectbox[0].set_value(sorted(G.nodes)[-1])
    at.slider[0].set_value(4)
    rerun(at)
    click(at, SIMULATE, timeout=180)
    seen += rendered_text(at)
    assert len(seen) > 600                                                           # the collector really sees the page
    assert not [t for t in seen if EM_DASH in t]


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
    loggers = [logging.getLogger(n) for n in list(logging.root.manager.loggerDict) if n == "streamlit" or n.startswith("streamlit.")]
    for lg in loggers:
        lg.addHandler(handler)
    yield handler.messages
    for lg in loggers:
        lg.removeHandler(handler)


def test_no_deprecation_noise_from_streamlit(streamlit_warnings):
    at = open_app(timeout=180)
    click(at, LOAD)
    click(at, SIMULATE, timeout=180)
    noise = [m for m in streamlit_warnings if "deprecat" in m.lower() or "please replace" in m.lower() or "will be removed" in m.lower()]
    assert not noise, noise

    # the check above is only worth something if this handler really hears Streamlit's deprecation logger
    probe = AppTest.from_string("import streamlit as st\nst.dataframe([1], use_container_width=True)").run()
    assert not probe.exception
    import inspect, streamlit as st
    if inspect.signature(st.dataframe).parameters["width"].default == "stretch":     # versions that deprecate it
        assert any("use_container_width" in m for m in streamlit_warnings)


# ---------------------------------------------------------------- Hugging Face Space bundle

def local_imports(path, seen=None):
    """Names of the repo's own modules that `path` imports, directly or through another of them."""
    seen = set() if seen is None else seen
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
        for name in (n.split(".")[0] for n in names):
            if name not in seen and (ROOT / f"{name}.py").exists():
                seen.add(name)
                local_imports(ROOT / f"{name}.py", seen)
    return seen


def test_space_bundle_lists_every_module_the_app_needs():
    script = (ROOT / "scripts" / "build_space.sh").read_text(encoding="utf-8")
    needed = {"app"} | local_imports(ROOT / "app.py")
    assert {"graph", "engine", "context", "simulate", "results"} <= needed
    missing = [m for m in sorted(needed) if not re.search(rf"(?<![\w/]){m}\.py\b", script)]
    assert not missing, f"scripts/build_space.sh does not copy: {missing}"
    for data in ("data/graph.json", "data/indicators.json", "data/eval_results.json", "requirements.txt"):
        assert data in script


def test_space_readme_front_matter():
    text = (ROOT / "deploy" / "space" / "README.md").read_text(encoding="utf-8")
    assert EM_DASH not in text
    assert text.startswith("---\n")
    front = dict(line.split(":", 1) for line in text.split("---\n")[1].strip().splitlines())
    front = {k.strip(): v.strip().strip("\"'") for k, v in front.items()}
    assert {k: front.get(k) for k in ("title", "emoji", "colorFrom", "colorTo", "sdk", "app_file", "pinned", "license")} == {
        "title": "Kwan", "emoji": "🧭", "colorFrom": "yellow", "colorTo": "green", "sdk": "streamlit",
        "app_file": "app.py", "pinned": "false", "license": "mit"}
    assert re.fullmatch(r"\d+\.\d+\.\d+", front["sdk_version"])
