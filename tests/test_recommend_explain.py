"""Reasons as data (PRD 6.3 and Phase 6.2): `rationale_parts`, the English templates, labels and settings."""

import json
import random
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import networkx as nx
import pytest

from kwan.context import bank
from kwan.learner.elo import EloLearner, LearnerState
from kwan.recommend import explain
from kwan.recommend.recommender import Recommender

EN_FILE = Path(__file__).resolve().parent.parent / "data" / "i18n" / "en.json"
EM_DASH = chr(0x2014)
GOAL = "B4.1.1.3.3"
EVERYTHING = 10_000
T0 = datetime(2026, 9, 14, 8, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def G(G_v1):
    """The seeded learners and goals below use bare codes (the v1 contract): the frozen v1 graph."""
    return G_v1


@pytest.fixture(scope="module")
def L(G):
    return EloLearner(G=G)


@pytest.fixture(scope="module")
def R(G, L):
    return Recommender(G, L)


@pytest.fixture(scope="module")
def states(G, L):
    """A new learner, a learner with a streak, and three seeded learners with timed answers."""
    out = [LearnerState("fresh")]
    st = LearnerState("b1")
    for i, ind in enumerate(["B1.1.1.1.1", "B1.1.1.1.2"] * 3):
        L.update(st, ind, True, at=T0 + timedelta(minutes=i))
    out.append(st)
    syllabus = sorted(G, key=lambda x: (G.nodes[x]["grade"], x))
    for seed in (1, 2, 3):
        rng = random.Random(seed)
        st = LearnerState(f"s{seed}")
        for i in range(120):
            L.update(st, rng.choice(syllabus[: 40 * seed]), rng.random() < 0.8, at=T0 + timedelta(hours=5 * i))
        out.append(st)
    return out


def all_recommendations(R, states):
    for st in states:
        for goal in (None, GOAL, "B2.1.1.1.1"):
            for now in (None, T0 + timedelta(days=70), T0 + timedelta(days=400)):
                yield from R.recommend(st, k=EVERYTHING, goal=goal, now=now)["recommendations"]


# ----------------------------------------------------------------------------- the templates


def test_the_built_in_english_templates_equal_the_data_file():
    """The package must work without `data/` on disk (a wheel install), so the code carries a copy."""
    on_disk = json.loads(EN_FILE.read_text(encoding="utf-8"))
    assert on_disk == explain.EN
    assert list(on_disk) == list(explain.EN)  # same order too
    assert explain.load_templates(EN_FILE) == explain.EN


def test_the_english_wording_of_the_v1_reasons_is_unchanged():
    assert explain.EN == {
        "builds_on": "Builds on {prerequisite}, where you are at {pct}%.",
        "starting_skill": "A starting skill with nothing before it.",
        "mastery_new": "Still new to you ({pct}%), so expect a few tries.",
        "mastery_close": "You are at {pct}%, close enough to master with practice.",
        "mastery_nearly": "Nearly there at {pct}%. A few correct answers will lock it in.",
        "goal_steps_one": "On the way to your goal, {count} step away.",
        "goal_steps_other": "On the way to your goal, {count} steps away.",
        "unlocks_one": "Opens the way to {count} later skill.",
        "unlocks_other": "Opens the way to {count} later skills.",
        "review": "You learned this before. A quick check keeps it fresh.",
    }


def test_plural_keys_come_in_pairs():
    for key in explain.PLURAL_KEYS:
        assert f"{key}_one" in explain.EN and f"{key}_other" in explain.EN and key not in explain.EN
    paired = {k.rsplit("_", 1)[0] for k in explain.EN if k.endswith(("_one", "_other"))}
    assert paired == set(explain.PLURAL_KEYS)


def test_templates_are_plain_short_and_have_no_em_dash():
    for key, text in explain.EN.items():
        assert EM_DASH not in text and ";" not in text, key
        for sentence in filter(None, re.split(r"(?<=[.!?])\s+", text)):
            assert len(sentence.split()) <= 12, (key, sentence)  # short sentences, P4 reading level
        assert set(re.findall(r"\{(\w+)\}", text)) <= {"prerequisite", "pct", "count"}, key


def test_the_data_file_is_flat_text_only():
    data = json.loads(EN_FILE.read_text(encoding="utf-8"))
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in data.items())
    assert EM_DASH not in EN_FILE.read_text(encoding="utf-8")


# ----------------------------------------------------------------------------- render()


def test_render_one_part_of_each_kind():
    r = explain.render
    assert r([explain.part("builds_on", prerequisite="B1.1.1.1.2", pct=95)]) == (
        "Builds on B1.1.1.1.2, where you are at 95%."
    )
    assert r([explain.part("starting_skill")]) == "A starting skill with nothing before it."
    assert r([explain.part("mastery_new", pct=7)]) == "Still new to you (7%), so expect a few tries."
    assert r([explain.part("mastery_close", pct=60)]) == "You are at 60%, close enough to master with practice."
    assert r([explain.part("mastery_nearly", pct=84)]) == "Nearly there at 84%. A few correct answers will lock it in."
    assert r([explain.part("goal_steps", count=1)]) == "On the way to your goal, 1 step away."
    assert r([explain.part("goal_steps", count=19)]) == "On the way to your goal, 19 steps away."
    assert r([explain.part("unlocks", count=1)]) == "Opens the way to 1 later skill."
    assert r([explain.part("unlocks", count=56)]) == "Opens the way to 56 later skills."
    assert r([explain.part("unlocks", count=0)]) == "Opens the way to 0 later skills."
    assert r([explain.part("review", pct=70, days=21)]) == "You learned this before. A quick check keeps it fresh."
    assert r([]) == ""


def test_render_joins_parts_with_one_space():
    parts = [explain.part("starting_skill"), explain.part("mastery_close", pct=55), explain.part("unlocks", count=45)]
    assert explain.render(parts, explain.EN) == (
        "A starting skill with nothing before it. You are at 55%, close enough to master with practice. "
        "Opens the way to 45 later skills."
    )


def test_parts_survive_json():
    parts = [explain.part("builds_on", prerequisite="B1.1.1.1.2", prerequisite_id="math:B1.1.1.1.2", pct=95)]
    assert json.loads(json.dumps(parts)) == parts
    assert explain.render(json.loads(json.dumps(parts))) == "Builds on B1.1.1.1.2, where you are at 95%."


def test_a_plain_key_wins_over_the_plural_pair():
    templates = {**explain.EN, "unlocks": "{count} more to come."}
    assert explain.render([explain.part("unlocks", count=1)], templates) == "1 more to come."
    assert explain.render([explain.part("unlocks", count=5)], templates) == "5 more to come."


def test_missing_template_or_field_is_a_clear_error():
    with pytest.raises(KeyError, match="no template for rationale key 'nope'"):
        explain.render([explain.part("nope")])
    with pytest.raises(KeyError, match="no template for rationale key 'unlocks'"):
        explain.render([explain.part("unlocks")])  # a plural key without its count
    with pytest.raises(KeyError, match="needs the field 'pct'"):
        explain.render([explain.part("mastery_new")])


def test_render_does_not_evaluate_format_expressions():
    """Templates are filled by name only: no attribute access, no format specs."""
    templates = {"x": "{pct.__class__} {pct!r} {pct:>5} {{pct}} {pct}"}
    assert explain.render([explain.part("x", pct=7)], templates) == "{pct.__class__} {pct!r} {pct:>5} {7} 7"


def test_resolve_templates_accepts_none_a_flat_table_or_languages():
    assert explain.resolve_templates(None) == {"en": explain.EN}
    assert explain.resolve_templates({}) == {"en": explain.EN}
    flat = explain.resolve_templates({"review": "Try this one again."})
    assert (
        flat["en"]["review"] == "Try this one again." and flat["en"]["starting_skill"] == explain.EN["starting_skill"]
    )
    multi = explain.resolve_templates({"en": explain.EN, "tw": {"starting_skill": "Mfiase."}})
    assert multi["tw"]["starting_skill"] == "Mfiase."
    assert multi["tw"]["review"] == explain.EN["review"]  # a missing key falls back to English
    assert explain.resolve_templates({"tw": {"review": "x"}})["en"] == explain.EN


# ----------------------------------------------------------------------------- parts in the recommender


def test_rendering_the_parts_equals_the_rationale_for_every_branch_on_the_real_graph(R, states):
    seen = set()
    count = 0
    for r in all_recommendations(R, states):
        assert explain.render(r["rationale_parts"], explain.EN) == r["rationale"], r["indicator"]
        assert explain.render(r["rationale_parts"]) == r["rationale"]
        for p in r["rationale_parts"]:
            assert set(p) == {"key", "params"}
            key = p["key"]
            if "count" in p["params"] and key in explain.PLURAL_KEYS:
                key += "_one" if p["params"]["count"] == 1 else "_other"
            seen.add(key)
            assert all(isinstance(v, (str, int)) and not isinstance(v, bool) for v in p["params"].values())
        assert json.loads(json.dumps(r["rationale_parts"])) == r["rationale_parts"]
        count += 1
    assert count > 1000
    assert seen == set(explain.EN)  # every template was used at least once, both plural forms included


def test_part_order_and_content(R, states):
    for r in all_recommendations(R, states):
        keys = [p["key"] for p in r["rationale_parts"]]
        if r["is_review"]:
            assert keys == ["review"]
            assert set(r["rationale_parts"][0]["params"]) == {"pct", "days"}
            assert r["rationale_parts"][0]["params"]["days"] >= 7
            continue
        assert keys[0] in ("builds_on", "starting_skill")
        assert keys[1] in ("mastery_new", "mastery_close", "mastery_nearly")
        assert keys[2:] in ([], ["goal_steps"], ["unlocks"], ["goal_steps", "unlocks"])
        band = r["rationale_parts"][1]
        pct = band["params"]["pct"]
        assert isinstance(pct, int) and 0 <= pct <= 100
        assert (band["key"] == "mastery_new") == (r["mastery"] < 0.5)
        if keys[0] == "builds_on":
            assert set(r["rationale_parts"][0]["params"]) == {"prerequisite", "prerequisite_id", "pct"}


def test_review_picks_show_up_on_the_real_graph(R, states):
    reviews = [r for r in all_recommendations(R, states) if r["is_review"]]
    assert reviews
    assert {r["rationale"] for r in reviews} == {"You learned this before. A quick check keeps it fresh."}


def test_custom_templates_and_language_fallback(G, L, states):
    twi_like = {"starting_skill": "MFIASE.", "unlocks": "+{count}."}
    r = Recommender(G, L, i18n={"en": explain.EN, "xx": twi_like})
    st = states[0]
    en = r.recommend(st, k=3)["recommendations"]
    xx = r.recommend(st, k=3, lang="xx")["recommendations"]
    assert [x["rationale_parts"] for x in xx] == [x["rationale_parts"] for x in en]  # the data is the same
    assert xx[0]["rationale"] == "MFIASE. Still new to you (38%), so expect a few tries. +58."
    assert [x["rationale"] for x in r.recommend(st, k=3, lang="zz")["recommendations"]] == [x["rationale"] for x in en]
    for item in xx:
        assert explain.render(item["rationale_parts"], r.templates["xx"]) == item["rationale"]
    flat = Recommender(G, L, i18n={"starting_skill": "Start here."})
    assert flat.recommend(st, k=1)["recommendations"][0]["rationale"].startswith("Start here. Still new to you")


def test_the_default_recommender_uses_the_built_in_english(R):
    assert R.templates == {"en": explain.EN}


# ----------------------------------------------------------------------------- short_label, ids and settings


def namespaced(G):
    """The v1 graph dressed as a v2 graph: `math:` ids, a `code` and a `short_label` on every node."""
    H = nx.relabel_nodes(G, {n: f"math:{n}" for n in G}, copy=True)
    for n, d in H.nodes(data=True):
        d["code"] = n.split(":", 1)[1]
        d["short_label"] = f"Label of {d['code']}"
    return H


def test_short_label_is_the_node_label_when_present_else_the_description(G, R, states):
    for r in R.recommend(states[1], k=5)["recommendations"]:
        assert r["short_label"] == r["description"] == G.nodes[r["indicator"]]["description"]
    H = namespaced(G)
    for r in Recommender(H, EloLearner(G=H)).recommend(LearnerState("x"), k=5)["recommendations"]:
        assert r["short_label"] == f"Label of {r['indicator'].split(':')[1]}"
        assert r["description"] == H.nodes[r["indicator"]]["description"] != r["short_label"]


def test_a_namespaced_graph_gives_the_same_picks_reasons_and_settings(G, L, R):
    H = namespaced(G)
    LH = EloLearner(G=H)
    RH = Recommender(H, LH)
    st, sth = LearnerState("p"), LearnerState("p")
    rng = random.Random(5)
    nodes = sorted(G)
    for _ in range(90):
        n, ok = rng.choice(nodes[:70]), rng.random() < 0.8
        L.update(st, n, ok)
        LH.update(sth, f"math:{n}", ok)
    for kwargs in ({}, {"goal": GOAL}, {"eps": 0.2}, {"goal": GOAL, "goal_mode": "soft", "variety": 0.0}):
        bare = R.recommend(st, k=EVERYTHING, **kwargs)["recommendations"]
        kw = {**kwargs, "goal": f"math:{kwargs['goal']}"} if "goal" in kwargs else kwargs
        spaced = RH.recommend(sth, k=EVERYTHING, **kw)["recommendations"]
        assert [x["indicator"] for x in spaced] == [f"math:{x['indicator']}" for x in bare]
        for a, b in zip(bare, spaced):
            assert (a["score"], a["mastery"], a["rationale"], a["ghana_context"]) == (
                b["score"],
                b["mastery"],
                b["rationale"],
                b["ghana_context"],
            )
            first = b["rationale_parts"][0]
            if first["key"] == "builds_on":  # the sentence shows the bare code, the id is there for programs
                assert first["params"]["prerequisite_id"] == f"math:{first['params']['prerequisite']}"


def test_the_default_setting_is_the_v1_bank_pick_on_the_bare_code(G):
    H = namespaced(G)
    for r in Recommender(H, EloLearner(G=H)).recommend(LearnerState("x"), k=EVERYTHING)["recommendations"]:
        node = H.nodes[r["indicator"]]
        assert r["ghana_context"] == bank.practice_setting(node["code"], node["strand"], node["grade"])


class RecordingSource:
    """A practice-setting source with the `ContextSource` shape."""

    def __init__(self):
        self.calls = []

    def pick(self, indicator_id, strand, grade, lang="en", include_drafts=False):
        self.calls.append((indicator_id, strand, grade, lang, include_drafts))
        return f"setting for {indicator_id} in {lang}{' (draft)' if include_drafts else ''}"


def test_a_context_source_is_asked_for_each_recommendation(G, L):
    source = RecordingSource()
    r = Recommender(G, L, contexts=source)
    recs = r.recommend(LearnerState("x"), k=3)["recommendations"]
    assert [x["ghana_context"] for x in recs] == [f"setting for {x['indicator']} in en" for x in recs]
    assert source.calls == [(x["indicator"], G.nodes[x["indicator"]]["strand"], x["grade"], "en", False) for x in recs]
    source.calls.clear()
    recs = r.recommend(LearnerState("x"), k=2, lang="tw", include_drafts=True)["recommendations"]
    assert [c[3:] for c in source.calls] == [("tw", True)] * 2
    assert recs[0]["ghana_context"].endswith("in tw (draft)")
    assert len(source.calls) == 2  # only the skills that are returned, not every candidate


def test_the_real_context_store_plugs_in_when_it_exists(G, L):
    settings = pytest.importorskip("kwan.context.settings")
    store_cls = getattr(settings, "ContextStore", None)
    if store_cls is None:
        pytest.skip("ContextStore is not there yet")
    try:
        store = store_cls({})
    except TypeError:
        pytest.skip("ContextStore has a different constructor")
    for r in Recommender(G, L, contexts=store).recommend(LearnerState("x"), k=3)["recommendations"]:
        assert isinstance(r["ghana_context"], str)
