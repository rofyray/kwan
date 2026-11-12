"""The Gradio demo (demo/gradio_app.py): its handlers, without a browser.

The browser check is demo/gradio_smoke_test.py, run by space-deploy.yml against the live Space.
"""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("gradio", reason="the Gradio demo needs the gradio extra (uv sync --extra gradio)")

ROOT = Path(__file__).resolve().parent.parent
EM_DASH = chr(0x2014)
CONTROLS = (6, "", False, False, "en")  # grade_max, goal, mix, drafts, lang


@pytest.fixture(scope="module")
def app():
    spec = importlib.util.spec_from_file_location("kwan_gradio_app", ROOT / "demo" / "gradio_app.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def unpack(app, out):
    learner, shown, head, drafts_note = out[:4]
    cards = out[4 : 4 + 2 * app.CARDS]
    return learner, shown, head, drafts_note, cards[1::2], out[4 + 2 * app.CARDS]


def test_the_page_builds_with_three_tabs(app):
    config = app.demo.get_config_file()
    labels = {c.get("props", {}).get("label") for c in config["components"]}
    assert {"Pupil view", "Curriculum map", "Evidence"} <= labels
    assert config["title"].startswith("Kwan")


def test_a_new_pupil_gets_three_cards_and_a_mastery_grid(app):
    out = app.new_pupil(*CONTROLS)
    learner, shown, head, drafts_note, cards, grid = unpack(app, out[:-1])
    assert len(shown) == 3 and all(s.startswith("math:") for s in shown)
    assert all("Why this skill" in c and "Mastery so far" in c for c in cards)
    assert "Picked in" in head and drafts_note == "" and out[-1] == ""
    assert "<table" in grid and "B6" in grid and "Number" in grid


def test_an_answer_lands_on_the_skill_its_card_showed(app):
    learner, shown, *_ = unpack(app, app.new_pupil(*CONTROLS)[:-1])
    target = shown[1]
    before = app.engine().mastery_map(learner)[target]
    after_out = app.answer(1, True)(learner, shown, *CONTROLS)
    learner2 = after_out[0]
    assert app.engine().mastery_map(learner2)[target] > before


def test_a_stale_slot_is_ignored(app):
    learner, shown, *_ = unpack(app, app.new_pupil(*CONTROLS)[:-1])
    state = learner.to_dict()
    app.answer(2, True)(learner, shown[:1], *CONTROLS)  # the third card is gone: nothing is recorded
    assert learner.to_dict() == state


def test_the_loaded_pupil_and_its_note(app):
    out = app.load_p4_pupil(*CONTROLS)
    assert out[-1].startswith("Loaded 44 past answers")
    assert "mostly wrong" in out[-1] and "mostly right" in out[-1]


def test_a_goal_is_picked_by_its_label_and_shown_by_name(app):
    choices = dict((value, label) for label, value in app.goal_choices())
    goal = "math:B4.1.3.2.1"
    assert "Put fractions in order" in choices[goal] and choices[""] == "No goal"
    head = unpack(app, app.new_pupil(6, goal, False, False, "en")[:-1])[2]
    assert "**Goal:** Put fractions in order from smallest to largest" in head


def test_drafts_show_a_warning(app):
    assert "Draft settings are on" in app.new_pupil(6, "", False, True, "en")[3]


def test_the_evidence_tables_read_the_results_file(app):
    ev = app.evidence()
    table, _missing = app.dataset_rows(ev)
    assert "AMMORE 2024 (Ghana)" in set(table["Dataset"])
    assert "Decision:" in app.deep_kt_text(ev)
    assert "ms" in app.latency_text(app.block_at(ev, "latency") or {})
    assert app.dataset_rows({})[0].empty and app.deep_kt_text({}).endswith("not run yet.")


def test_the_map_is_an_iframe_with_the_graph_page(app):
    page = app.map_page()
    assert page.startswith("<iframe srcdoc=") and "Interactive map" in page


def test_copy_has_no_em_dash_and_carries_the_disclaimer(app):
    text = (ROOT / "demo" / "gradio_app.py").read_text(encoding="utf-8")
    assert EM_DASH not in text
    assert "Kwan is not reviewed, approved or endorsed by NaCCA or the Ministry of Education." in app.DISCLAIMER
