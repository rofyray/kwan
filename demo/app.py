"""Kwan demo UI (Streamlit). Run: uv run streamlit run demo/app.py

The app uses only the public package API (`kwan.Kwan` on the shipped bundle), the graph page export and the
evaluation results file. Nothing about a visitor is stored: the pupil is a LearnerState in the session.
"""

import inspect
import random
import re
from collections import Counter
from datetime import UTC, datetime, timedelta
from importlib.resources import files
from time import perf_counter

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from kwan import Kwan
from kwan.eval import results, simulate
from kwan.graph.export import graph_html

ASSETS = files("kwan") / "assets"  # the logos ship inside the package, so the Docker image has them too
LOGO = str(ASSETS / "kwan-logo.png")  # wordmark on its cream tile: reads the same in light and dark themes
MARK = str(ASSETS / "kwan-mark.png")

st.set_page_config(page_title="Kwan · the next step in maths", page_icon=MARK, layout="wide")

NO_GOAL = "(no goal)"
GOAL_HELP = (
    "Kwan follows the path to this skill strictly: skills on the way to the goal come first. "
    "Turn on 'Mix in other skills' to let other ready skills in."
)
MIX_LABEL = "Mix in other skills"
MIX_HELP = "With a goal set, also offer ready skills that are not on the path to it."
DRAFTS_LABEL = "Show unreviewed draft settings"
DRAFTS_HELP = (
    "Practice settings drafted for Kwan that no teacher has checked yet. They may contain mistakes. "
    "Off: only teacher-reviewed settings and the standard examples by strand are used."
)
DRAFTS_WARNING = "Draft settings are on. The 'Try it in Ghana' text may not have been checked by a teacher."
LANG_NAMES = {"en": "English", "tw": "Twi", "ee": "Ewe", "gaa": "Ga"}
REVIEW_NOTE = "🔁 **Review:** learned before and starting to fade. A quick check keeps it fresh."
DISCLAIMER = (
    "Kwan uses the NaCCA Standards-Based Curriculum to organise skills. "
    "Kwan is not reviewed, approved or endorsed by NaCCA or the Ministry of Education."
)
EPS = simulate.EXPLORE_EPS  # exploration strength in the demo (PRD 6.3); seeded by pupil and step
SEED = 7  # the loaded pupil is always the same pupil
STRONG_SKILLS, STRONG_WEEKS = 4, 6  # B2 to B3 skills answered right three times, this many weeks ago
LOWER_ANSWERS, LOWER_RIGHT, LOWER_WEEKS = 24, 20, 3  # answers on other B1 to B3 skills (about 85% right)
B4_ANSWERS, B4_RIGHT, B4_DAYS = 8, 2, 2  # answers on B4 skills (about 20% right)
SIM_LEARNERS, SIM_STEPS, SIM_GOAL_STEPS = 3, simulate.STEPS, simulate.STEPS  # the small run behind the button
AUC_MODELS = ["global-mean", "skill-mean", "bkt", "elo", "elo-multi (Kwan)"]
DATASETS = [  # results key (a slash reaches into a sub-block) and the name shown
    ("assistments_2009", "ASSISTments 2009"),
    ("assistments_2012", "ASSISTments 2012"),
    ("assistments_2015", "ASSISTments 2015"),
    ("assistments_2017", "ASSISTments 2017"),
    ("eedi_2020_task34", "Eedi 2020 (tasks 3 and 4)"),
    ("eedi_2020_task12", "Eedi 2020 (tasks 1 and 2)"),
    ("xes3g5m", "XES3G5M"),
    ("junyi_2015", "Junyi 2015"),
    ("ammore_2024/all", "AMMORE 2024 (all)"),
    ("ammore_2024/ghana", "AMMORE 2024 (Ghana)"),
]
PROFILE_LABELS = {
    "beginner_p1": "Beginner in P1",
    "gapped_p4": "P4 pupil with gaps in B1 to B3",
    "gapped_p6": "P6 pupil with gaps in B1 to B5",
    "uneven_strands": "P4 pupil weak in one strand",
}
POLICY_LABELS = {
    "kwan-strict": "Following Kwan",
    "kwan-explore": "Following Kwan, with exploration",
    "kwan-top1": "Following Kwan, top pick only",
    "syllabus": "Syllabus order",
    "random": "Random practice",
    "remedial-first": "Class syllabus, gaps first",
}
NOT_RUN = "not run yet"
MAP_HEIGHT = 820
# mastery bands of the class-by-strand table: (upper bound, background). Dark text on these pale fills keeps
# the WCAG AA contrast ratio in both Streamlit themes.
BANDS = [(0.5, "#f6c6c6"), (0.7, "#ffe3a3"), (0.85, "#d9efc2"), (1.01, "#9fdc9f")]
CELL_TEXT = "#1f1f1f"

# st.dataframe changed its API between the oldest Streamlit we allow (pyproject.toml) and current releases:
# width="stretch" replaced use_container_width (deprecated, logs a warning on every call) and `placeholder`
# was added to choose what a missing value looks like. Ask the installed version which one it speaks.
_DF_PARAMS = inspect.signature(st.dataframe).parameters
FULL_WIDTH = {"width": "stretch"} if _DF_PARAMS["width"].default == "stretch" else {"use_container_width": True}
BLANK_IF_MISSING = {"placeholder": ""} if "placeholder" in _DF_PARAMS else {}


@st.cache_resource(show_spinner=False)
def load_engine():
    """The engine on the shipped bundle, built once per server and shared by every session.
    It stores nothing about a pupil: a pupil is the LearnerState kept in st.session_state."""
    return Kwan.from_bundle()


@st.cache_resource(show_spinner=False)
def map_page(bundle_version):
    """The interactive graph page, drawn once per bundle."""
    return graph_html(load_engine().graph)


@st.cache_data(show_spinner=False)
def small_simulation(bundle_version):
    """Simulator v2 on a few seeded pupils (the same result every time), labelled for the table."""
    rep = simulate.report(load_engine().graph, n_learners=SIM_LEARNERS, steps=SIM_STEPS, goal_max_steps=SIM_GOAL_STEPS)
    return simulation_rows(rep)


def now():
    return datetime.now(UTC)


def plural(n, word, shown=None):
    """The one place a count meets its noun: plural(1, "skill") -> '1 skill', plural(3, "skill") -> '3 skills'.
    `shown` replaces the number as written, for short forms: plural(259399, "answer", "259k") -> '259k answers'."""
    return f"{n if shown is None else shown} {word}{'' if n == 1 else 's'}"


def p4_history(K, at):
    """The fixed history of the loaded pupil, oldest answer first: [(indicator, answered right, when)].

    Four B2 to B3 skills answered right three times six weeks ago (learned well, now fading), one answer on
    each of 24 other skills from B1 to B3 three weeks ago (about 85% right), then one answer on each of 8
    skills from B4 two days ago (about 20% right). The skills, and which answers are right, come from one
    seeded generator, so the history never changes; the dates sit at fixed distances from `at`."""
    G = K.graph
    rng = random.Random(SEED)

    def answers_on(skills, count, right):
        picked = sorted(rng.sample(sorted(skills), count))
        correct = set(rng.sample(picked, right))
        return [(n, n in correct) for n in picked]

    lower = answers_on((n for n in G if G.nodes[n]["grade"] <= 3), LOWER_ANSWERS, LOWER_RIGHT)
    b4 = answers_on((n for n in G if G.nodes[n]["grade"] == 4), B4_ANSWERS, B4_RIGHT)
    taken = {n for n, _ in lower}
    strong = sorted(rng.sample(sorted(n for n in G if G.nodes[n]["grade"] in (2, 3) and n not in taken), STRONG_SKILLS))
    t_strong, t_lower, t_b4 = (
        at - timedelta(weeks=STRONG_WEEKS),
        at - timedelta(weeks=LOWER_WEEKS),
        at - timedelta(days=B4_DAYS),
    )
    history = [(n, True, t_strong + timedelta(minutes=3 * j + i)) for j, n in enumerate(strong) for i in range(3)]
    history += [(n, ok, t_lower) for n, ok in lower]
    return history + [(n, ok, t_b4) for n, ok in b4]


def mostly(outcomes):
    """'mostly right' or 'mostly wrong' for a list of answers, from the answers themselves."""
    right = sum(outcomes)
    return (
        "mostly right" if 2 * right > len(outcomes) else "mostly wrong" if 2 * right < len(outcomes) else "half right"
    )


def loaded_message(K, history):
    """The note under the load button, counted from the history that was loaded."""
    G = K.graph
    strong = {n for n, _, _ in history if sum(1 for m, _, _ in history if m == n) > 1}
    lower = [ok for n, ok, _ in history if G.nodes[n]["grade"] <= 3 and n not in strong]
    b4 = [ok for n, ok, _ in history if G.nodes[n]["grade"] == 4]
    return (
        f"Loaded {plural(len(history), 'past answer')}: {plural(len(strong), 'skill')} learned well "
        f"{plural(STRONG_WEEKS, 'week')} ago (3 right answers each), {len(lower)} on other B1 to B3 skills "
        f"{plural(LOWER_WEEKS, 'week')} ago ({mostly(lower)}) and {len(b4)} on B4 skills "
        f"{plural(B4_DAYS, 'day')} ago ({mostly(b4)})."
    )


def start_over():
    st.session_state.learner = load_engine().new_state("demo")
    st.session_state.pop("notice", None)
    st.session_state.pop("seeded_history", None)


def load_p4_pupil():
    """Always builds the same pupil on a fresh state, so pressing the button twice does not stack two histories."""
    K = load_engine()
    pupil, history = K.new_state("demo"), p4_history(K, now())
    for n, correct, when in history:
        K.update(pupil, n, correct, at=when)
    st.session_state.learner = pupil
    st.session_state.seeded_history = history
    st.session_state.notice = loaded_message(K, history)


def answer(indicator, correct):
    """Bound to one card's skill when the card is drawn, so a click can never land on a different skill."""
    load_engine().update(st.session_state.learner, indicator, correct, at=now())


def plain(text):
    """Curriculum text is shown as written: escape the characters Markdown would treat as formatting."""
    return re.sub(r"([\\`*_\[\]$~<>#])", r"\\\1", str(text))


def answers(n):
    """259399 -> '259k', 1382727 -> '1.38M'."""
    return f"{n / 1e6:.2f}M" if n >= 999_500 else f"{n / 1e3:.0f}k" if n >= 1000 else f"{n:.0f}"


def reviewed_languages(K):
    """Languages a visitor may pick: English, plus every language with a teacher-reviewed practice setting."""
    langs = {"en"}
    if K.bundle is not None:
        langs |= {str(e.get("lang")) for entries in K.bundle.contexts.values() for e in entries if e.get("reviewed")}
    return sorted(langs, key=lambda code: LANG_NAMES.get(code, code))


def cell_style(value):
    """Background and text colour of one mastery cell."""
    if pd.isna(value):
        return ""
    fill = next(colour for bound, colour in BANDS if value < bound)
    return f"background-color: {fill}; color: {CELL_TEXT}"


def styled(df, func):
    """`Styler.map` (pandas 2.1 and later) or its old name `applymap`."""
    style = df.style
    return (style.map if hasattr(style, "map") else style.applymap)(func)


def embed_map(page):
    """Show the graph page in an iframe: `st.iframe` where Streamlit has it, else the older components call."""
    if hasattr(st, "iframe"):
        st.iframe(page, height=MAP_HEIGHT, alt="Interactive map of the curriculum graph, one node per indicator")
    else:
        components.html(page, height=MAP_HEIGHT, scrolling=True)


def evidence():
    """data/eval_results.json as written by the evaluation scripts, or {} if it is missing or unreadable."""
    try:
        ev = results.load()
    except (OSError, ValueError):
        return {}
    return ev if isinstance(ev, dict) else {}


def block_at(ev, key):
    """The block at a results key; `a/b` reaches into a sub-block. Anything that is not a dict counts as missing."""
    node = ev
    for part in key.split("/"):
        node = node.get(part) if isinstance(node, dict) else None
    return node if isinstance(node, dict) else None


def auc_of(block, model):
    """AUC of one model in a dataset block, from `metrics` (schema v2) or the v1 `AUC` table; None if absent."""
    metrics = block.get("metrics") if isinstance(block.get("metrics"), dict) else {}
    row = metrics.get(model) if isinstance(metrics.get(model), dict) else {}
    value = row.get("AUC")
    if value is None and isinstance(block.get("AUC"), dict):
        value = block["AUC"].get(model)
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def dataset_rows(ev):
    """One row per evaluated dataset (AUC per model), plus the names of datasets not run and skipped."""
    models = list(AUC_MODELS)
    present, missing, skipped = [], [], []
    for key, name in DATASETS:
        block = block_at(ev, key)
        if block is None:
            missing.append(name)
            continue
        if block.get("skipped"):
            skipped.append(f"{name}: {block['skipped']}")
            continue
        for m in block.get("metrics") or {}:
            if m not in models:
                models.append(m)
        present.append((name, block))
    rows = []
    for name, block in present:
        n = block.get("rows")
        sized = isinstance(n, (int, float)) and not isinstance(n, bool) and n > 0
        row = {"Dataset": name, "Answers": answers(n) if sized else ""}
        row.update({m: auc_of(block, m) for m in models})
        rows.append(row)
    table = pd.DataFrame(rows, columns=["Dataset", "Answers", *models])
    for m in models:
        table[m] = pd.to_numeric(table[m], errors="coerce")
    return table, missing, skipped


def dataset_details(ev):
    """Size, split and licence of every evaluated dataset, as written in the results file."""
    rows = []
    for key, name in DATASETS:
        block = block_at(ev, key)
        if block is None or block.get("skipped"):
            continue
        rows.append(
            {
                "Dataset": name,
                "Answers": block.get("rows"),
                "Pupils": block.get("students"),
                "Items": block.get("items"),
                "Skills": block.get("skills"),
                "Split": block.get("split"),
                "Licence": block.get("licence"),
                "Note": block.get("note"),
            }
        )
    return pd.DataFrame(rows)


def fit_rows(block):
    """One row per dataset of the learner_fit block: forgetting and tuned parameters, as fitted."""
    names = dict(DATASETS)
    rows = []
    for key, d in (block.get("datasets") or {}).items():
        if not isinstance(d, dict):
            continue
        chosen = d.get("grid_chosen_params") if isinstance(d.get("grid_chosen_params"), dict) else {}
        rows.append(
            {
                "Dataset": names.get(key, key),
                "Has dates": d.get("timestamps"),
                "Forgetting half-way (tau, days)": d.get("tau_chosen"),
                "AUC without forgetting": d.get("auc_without_forgetting"),
                "AUC with forgetting": d.get("auc_with_forgetting"),
                "AUC with v1 settings": d.get("auc_v1_defaults"),
                "AUC with tuned settings": d.get("auc_chosen"),
                "Tuned settings": ", ".join(f"{k} {v}" for k, v in chosen.items()),
                "May ship": d.get("commercial_params_ok"),
            }
        )
    return pd.DataFrame(rows)


def shipped_params(block):
    """The parameters the bundle ships with, from the learner_fit block, as a two-column table."""
    params = (block.get("shipped") or {}).get("params") if isinstance(block.get("shipped"), dict) else None
    params = params if isinstance(params, dict) else {}
    return pd.DataFrame([{"Setting": k, "Value": str(v)} for k, v in params.items()], columns=["Setting", "Value"])


def deep_kt_rows(decision):
    """One row per deep model: AUC margin over Kwan on each dataset, speed, and whether the rule was met."""
    models = {m: d for m, d in (decision.get("per_model") or {}).items() if isinstance(d, dict)}
    datasets = list(decision.get("datasets_in_rule") or [])
    for d in models.values():
        datasets += [ds for ds in (d.get("margins") or {}) if ds not in datasets]
    rows = []
    for model, d in models.items():
        row = {"Model": model}
        row.update({f"AUC margin on {ds}": (d.get("margins") or {}).get(ds) for ds in datasets})
        row["ms per update (CPU)"] = d.get("ms_per_update_cpu")
        row["fast enough"] = d.get("latency_ok")
        row["meets the rule"] = d.get("meets_rule")
        rows.append(row)
    return pd.DataFrame(rows)


def mean_sd(cell):
    return f"{cell['mean']} ± {cell['sd']}" if isinstance(cell, dict) and "mean" in cell else ""


def simulation_rows(rep):
    """The simulator v2 report as table rows with readable labels."""
    rows = []
    for prof, by_pol in (rep.get("new_skills_mastered") or {}).items():
        for pol, new in by_pol.items():
            kept = (rep.get("skills_retained") or {}).get(prof, {}).get(pol)
            goal = (rep.get("steps_to_goal") or {}).get(prof, {}).get(pol)
            rows.append(
                {
                    "pupil type": PROFILE_LABELS.get(prof, prof),
                    "how they practised": POLICY_LABELS.get(pol, pol),
                    "new skills mastered": mean_sd(new),
                    "still known after the break": mean_sd(kept),
                    "steps to reach a goal": mean_sd(goal),
                    "reached the goal": f"{goal['reached']:.0%}"
                    if isinstance(goal, dict) and "reached" in goal
                    else "",
                }
            )
    return rows


K = load_engine()
G = K.graph
if "learner" not in st.session_state:
    st.session_state.learner = K.new_state("demo")
ev = evidence()
latency = block_at(ev, "latency") or {}

st.image(LOGO, width=280)
st.markdown(
    "**Kwan** is Twi for *the path*. It reads Ghana's Basic 1 to Basic 6 maths curriculum, keeps track of what "
    "a pupil has mastered, and picks the next skill to learn, with a reason and an example from everyday "
    "Ghanaian life."
)
speed = (
    f" · top 3 picked in {latency['recommend_top3_mean_ms']} ms on average ({latency.get('hardware', 'one CPU core')})"
    if isinstance(latency.get("recommend_top3_mean_ms"), (int, float))
    else ""
)
st.caption(f"NaCCA B1 to B6 Mathematics · {plural(G.number_of_nodes(), 'skill')} · bundle {K.bundle_version}{speed}")

tab1, tab2, tab3 = st.tabs(["Pupil view", "Curriculum map", "Evidence"])
with tab1:
    c1, c2 = st.columns([1, 2])
    with c1:
        grade_max = st.slider("Recommend skills up to class (B1 to B6)", 1, 6, 6)
        goal = st.selectbox(
            "Goal skill, e.g. next term's topic (optional)",
            [NO_GOAL] + sorted(G.nodes),
            format_func=lambda n: (
                n if n == NO_GOAL else f"{G.nodes[n].get('code', n)} · {G.nodes[n].get('short_label', '')}"
            ),
            help=GOAL_HELP,
        )
        goal = None if goal == NO_GOAL else goal
        goal_mode = "soft" if st.toggle(MIX_LABEL, value=False, help=MIX_HELP) else "strict"
        include_drafts = st.toggle(DRAFTS_LABEL, value=False, help=DRAFTS_HELP)
        if include_drafts:
            st.warning(DRAFTS_WARNING)
        langs = reviewed_languages(K)
        lang = st.selectbox("Language", langs, format_func=lambda code: LANG_NAMES.get(code, code))
        st.caption(
            "Only languages with practice settings checked by a teacher are listed"
            + (": today that is English alone. " if langs == ["en"] else ". ")
            + "Other languages appear here once their translations are reviewed."
        )
        st.button("Load a P4 pupil with gaps in earlier classes", on_click=load_p4_pupil)
        if st.session_state.get("notice"):
            st.success(st.session_state.notice)
        st.button("Start over with a new pupil", on_click=start_over)
    S = st.session_state.learner
    t = now()
    with c2:
        tick = perf_counter()
        recs = K.recommend(
            S,
            k=3,
            goal=goal,
            grade_max=grade_max,
            goal_mode=goal_mode,
            now=t,
            eps=EPS,
            lang=lang,
            include_drafts=include_drafts,
        )
        took = (perf_counter() - tick) * 1000
        st.subheader("What to learn next")
        st.caption(
            f"Picked in {took:.2f} ms. Mark an answer right or wrong to watch the path change. Review and "
            "exploration are on: one card may bring back a skill learned before, and close contenders are "
            "shuffled a little (the same pupil always gets the same picks)."
        )
        for r in recs:
            with st.container(border=True):
                st.markdown(f"**{r.code}** · B{r.grade} · {r.strand}  \n{plain(r.short_label)}")
                if r.description and r.description != r.short_label:
                    st.caption(plain(r.description))
                if r.is_review:
                    st.markdown(REVIEW_NOTE)
                st.progress(min(max(r.mastery, 0.0), 1.0), text=f"Mastery so far: {r.mastery:.0%}")
                st.markdown(f"**Why this skill:** {r.rationale}")
                if r.ghana_context:
                    st.info(f"**Try it in Ghana:** {r.ghana_context}")
                a, b = st.columns(2)
                # keyed by skill, not by card position: a late or repeated click still belongs to its own skill
                a.button("✅ Got it right", key=f"right_{r.indicator}", on_click=answer, args=(r.indicator, True))
                b.button("❌ Got it wrong", key=f"wrong_{r.indicator}", on_click=answer, args=(r.indicator, False))
    mastery = K.mastery_map(S, now=t)
    m = pd.DataFrame(
        [
            {"indicator": n, "grade": d["grade"], "strand": d["strand_name"], "mastery": mastery[n]}
            for n, d in G.nodes(data=True)
        ]
    )
    st.subheader("Mastery by class and strand")
    st.caption(
        "Average estimated mastery across the skills in each cell. Red is below 50%, amber is 50% to 70%, "
        "light green is 70% to 85% and green is 85% and above, which Kwan counts as mastered."
    )
    st.dataframe(
        styled(m.pivot_table(index="strand", columns="grade", values="mastery", aggfunc="mean"), cell_style).format(
            "{:.0%}"
        ),
        **FULL_WIDTH,
    )
with tab2:
    st.markdown(
        "Every NaCCA learning indicator is a skill. Kwan links each skill to the ones that should come before it: "
        "the next indicator in the same standard, the same standard in the next class, and the next standard in "
        "the same sub-strand. Dashed links were suggested by the decision model and accepted by a reviewer."
    )
    by_type = Counter(str(d.get("type")) for _, _, d in G.edges(data=True))
    cols = st.columns(2 + len(by_type))
    cols[0].metric("Skills (NaCCA indicators)", G.number_of_nodes())
    cols[1].metric("Prerequisite links", G.number_of_edges())
    for col, (kind, count) in zip(cols[2:], sorted(by_type.items())):
        col.metric(f"{kind} links", count)
    embed_map(map_page(K.bundle_version))
    with st.expander("The same links as a table"):
        st.dataframe(
            pd.DataFrame(
                [
                    {"from": G.nodes[a].get("code", a), "to": G.nodes[b].get("code", b), "type": d.get("type")}
                    for a, b, d in G.edges(data=True)
                ]
            ),
            height=300,
            **FULL_WIDTH,
        )
with tab3:
    st.markdown(
        "**How well does Kwan predict a pupil's next answer?** We tested the learner model on public datasets "
        "of real student answers, holding out 20% of students it never saw in training. AUC is 0.5 for a coin "
        "flip and 1.0 for perfect prediction. AMMORE comes from a WhatsApp maths tutor used in West Africa; its "
        "Ghana rows are the pupils in Ghana."
    )
    auc, missing, skipped = dataset_rows(ev)
    st.dataframe(auc, hide_index=True, **FULL_WIDTH, **BLANK_IF_MISSING)
    ran_bkt = [row["Dataset"] for _, row in auc.iterrows() if pd.notna(row.get("bkt"))] if "bkt" in auc else []
    note = "Global-mean and skill-mean are simple averages. BKT is Bayesian Knowledge Tracing, the classic baseline."
    if ran_bkt and len(ran_bkt) < len(auc):
        note += f" BKT was run on {', '.join(ran_bkt)} only."
    st.caption(note)
    if missing:
        st.caption(f"{NOT_RUN.capitalize()}: {', '.join(missing)}.")
    for line in skipped:
        st.caption(f"Skipped. {line}")
    if len(auc):
        with st.expander("Dataset details and licences"):
            st.dataframe(dataset_details(ev), hide_index=True, **FULL_WIDTH, **BLANK_IF_MISSING)

    st.markdown(
        "**Forgetting and tuned parameters.** How fast a skill fades without practice, and the learner-model "
        "settings chosen on the training splits."
    )
    fit = block_at(ev, "learner_fit")
    if fit is None:
        st.caption(f"Forgetting and tuned parameters: {NOT_RUN}.")
    else:
        fitted = fit_rows(fit)
        if len(fitted):
            st.dataframe(fitted, hide_index=True, **FULL_WIDTH, **BLANK_IF_MISSING)
        st.caption(str(fit.get("protocol", "")))
        shipped = shipped_params(fit)
        if len(shipped):
            st.caption("Settings shipped in the bundle (chosen on datasets whose licence allows it)")
            st.dataframe(shipped, hide_index=True, **FULL_WIDTH, **BLANK_IF_MISSING)
        acc = fit.get("acceptance") if isinstance(fit.get("acceptance"), dict) else {}
        if "passed" in acc:
            st.caption(f"Check: {acc.get('rule', '')}. " + ("Passed." if acc["passed"] else "Not passed."))

    st.markdown(
        "**Would a deep learning model do better?** Deep knowledge-tracing models were trained on the same splits "
        "and judged by one rule written before the runs."
    )
    deep = block_at(ev, "kt_deep/decision")
    if deep is None:
        st.caption(f"Deep knowledge tracing benchmark: {NOT_RUN}.")
    else:
        verdict = (
            f"Decision: adopt {deep['model']}."
            if deep.get("adopt")
            else "Decision: keep Kwan's own learner model. No deep model met the rule."
        )
        st.markdown(f"{verdict}  \nThe rule: {deep.get('rule', '')}")
        table = deep_kt_rows(deep)
        if len(table):
            st.dataframe(table, hide_index=True, **FULL_WIDTH, **BLANK_IF_MISSING)

    st.markdown(
        "**Does a strict goal keep the path on track?** With a goal set, the share of the top picks that lead to "
        "it, strictly and when other skills are mixed in."
    )
    gp = block_at(ev, "goal_path")
    share = gp.get("on_path_share") if gp and isinstance(gp.get("on_path_share"), dict) else None
    if share is None:
        st.caption(f"Goal path check: {NOT_RUN}.")
    else:
        g1, g2 = st.columns(2)
        g1.metric("On the path to the goal (strict)", f"{share.get('strict', 0):.0%}")
        g2.metric("On the path with other skills mixed in", f"{share.get('soft', 0):.0%}")
        st.caption(str(gp.get("setup", "")))

    st.markdown("**How fast is it?**")
    if isinstance(latency.get("recommend_top3_mean_ms"), (int, float)):
        l1, l2 = st.columns(2)
        l1.metric("Top 3 picks, average", f"{latency['recommend_top3_mean_ms']} ms")
        l2.metric("Top 3 picks, slowest 5%", f"{latency.get('recommend_top3_p95_ms', '')} ms")
        where = latency.get("hardware", "one CPU core")
        st.caption(f"{plural(latency.get('calls', 0), 'call')} on {where}, in-process, no network.")
    else:
        st.caption(f"Latency benchmark: {NOT_RUN}.")

    st.markdown(
        "**Does following Kwan help pupils learn more?** The simulator gives seeded virtual pupils a daily practice "
        "routine, lets them forget, and counts the new skills they master under each way of practising, how many "
        "they still know after a month off, and how long a goal takes. Everything in it is an assumption, not a "
        "measurement of Ghanaian pupils."
    )
    sim = block_at(ev, "simulation/v2")
    if sim is None:
        st.caption(f"Simulator v2: {NOT_RUN}.")
    else:
        st.caption(str(sim.get("setup", "")))
        st.dataframe(pd.DataFrame(simulation_rows(sim)), hide_index=True, **FULL_WIDTH)
        acc = sim.get("acceptance") if isinstance(sim.get("acceptance"), dict) else {}
        if "met" in acc:
            st.caption(
                "The acceptance check (Kwan beats syllabus order and random practice for pupils with gaps, and "
                "stays within one skill of syllabus order for every pupil type) is "
                + ("met." if acc["met"] else "not met.")
            )
    st.markdown(
        f"Run it here on {plural(SIM_LEARNERS, 'simulated pupil')} per pupil type and way of practising, "
        f"{plural(SIM_STEPS, 'practice step')} each. The same small run gives the same table every time."
    )
    if st.button("Run simulator v2 now (a small run, takes a moment)"):
        with st.spinner("Simulating..."):
            st.session_state.sim_rows = small_simulation(K.bundle_version)
    if st.session_state.get("sim_rows"):  # kept for the session, so the table survives the next click elsewhere
        st.dataframe(pd.DataFrame(st.session_state.sim_rows), hide_index=True, **FULL_WIDTH)

st.divider()
st.markdown(DISCLAIMER)
st.caption(
    f"Bundle {K.bundle_version} ({K.text_mode} mode) · code MIT, Kwan-written content CC BY 4.0 · "
    "source and docs at github.com/rofyray/kwan"
)
