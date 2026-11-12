"""Kwan demo UI (Gradio), for the free Hugging Face Space. Run: uv run --extra gradio demo/gradio_app.py

The same three tabs as the Streamlit demo (demo/app.py, which the Docker image still runs): the pupil view,
the curriculum map and the evidence. It uses only the public package API (`kwan.Kwan` on the shipped bundle),
the graph page export and the evaluation results file. Nothing about a visitor is stored: the pupil is a
LearnerState held in the browser session (gr.State) and is gone when the tab closes.
"""

from __future__ import annotations

import base64
import html
import os
import random
from collections import Counter
from datetime import UTC, datetime, timedelta
from functools import cache
from importlib.resources import files
from time import perf_counter

import gradio as gr
import pandas as pd

from kwan import Kwan
from kwan.eval import results, simulate
from kwan.graph.export import graph_html

TITLE = "Kwan · the next step in maths"
ASSETS = files("kwan") / "assets"  # the logos ship inside the package (and in the Space's unpacked copy)
MARK = str(ASSETS / "kwan-mark.png")  # the cream tile: the browser tab icon
LOGO_ALT = "Kwan logo: a green path through gold steps to a red goal under a star"
# the logo's green path and gold steps
THEME = gr.themes.Default(primary_hue=gr.themes.colors.green, secondary_hue=gr.themes.colors.yellow)
# Gradio puts the class `dark` on the page in dark mode: show the wordmark drawn for that background.
CSS = """
.kwan-logo img { max-width: 300px; width: 100%; height: auto; }
.kwan-logo .on-dark { display: none; }
.dark .kwan-logo .on-light { display: none; }
.dark .kwan-logo .on-dark { display: inline; }
"""
INTRO = (
    "**Kwan** is Twi for *the path*. It reads Ghana's Basic 1 to Basic 6 maths curriculum, keeps track of what "
    "a pupil has mastered, and picks the next skill to learn, with a reason and an example from everyday "
    "Ghanaian life."
)
DISCLAIMER = (
    "Kwan uses the NaCCA Standards-Based Curriculum to organise skills. "
    "Kwan is not reviewed, approved or endorsed by NaCCA or the Ministry of Education."
)
NO_GOAL = ""
CARDS = 3
EPS = simulate.EXPLORE_EPS  # exploration strength in the demo; seeded by pupil and step
SEED = 7  # the loaded pupil is always the same pupil
STRONG_SKILLS, STRONG_WEEKS = 4, 6
LOWER_ANSWERS, LOWER_RIGHT, LOWER_WEEKS = 24, 20, 3
B4_ANSWERS, B4_RIGHT, B4_DAYS = 8, 2, 2
SIM_LEARNERS, SIM_STEPS = 3, simulate.STEPS
LANG_NAMES = {"en": "English", "tw": "Twi", "ee": "Ewe", "gaa": "Ga"}
AUC_MODELS = ["global-mean", "skill-mean", "bkt", "elo", "elo-multi (Kwan)"]
DATASETS = [
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
BANDS = [(0.5, "#f6c6c6"), (0.7, "#ffe3a3"), (0.85, "#d9efc2"), (1.01, "#9fdc9f")]
CELL_TEXT = "#1f1f1f"


# ----------------------------------------------------------------------------- engine and data


@cache
def engine() -> Kwan:
    """The engine on the shipped bundle, built once per server and shared by every session."""
    return Kwan.from_bundle()


@cache
def map_page() -> str:
    """The interactive graph page, drawn once, inside an iframe so its scripts run."""
    page = graph_html(engine().graph)
    return (
        f'<iframe srcdoc="{html.escape(page, quote=True)}" title="Interactive map of the curriculum graph" '
        f'style="width:100%;height:{MAP_HEIGHT}px;border:0"></iframe>'
    )


def evidence() -> dict:
    try:
        ev = results.load()
    except (OSError, ValueError):
        return {}
    return ev if isinstance(ev, dict) else {}


def block_at(ev: dict, key: str) -> dict | None:
    node: object = ev
    for part in key.split("/"):
        node = node.get(part) if isinstance(node, dict) else None
    return node if isinstance(node, dict) else None


def now() -> datetime:
    return datetime.now(UTC)


def plural(n, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def logo_html() -> str:
    """The wordmark as the page heading, in a light-background and a dark-background version."""

    def data_uri(name: str) -> str:
        return "data:image/png;base64," + base64.b64encode((ASSETS / name).read_bytes()).decode("ascii")

    return (
        f'<h1 class="kwan-logo" style="margin:0">'
        f'<img class="on-light" src="{data_uri("kwan-logo-light.png")}" alt="{LOGO_ALT}">'
        f'<img class="on-dark" src="{data_uri("kwan-logo-dark.png")}" alt="{LOGO_ALT}"></h1>'
    )


# ----------------------------------------------------------------------------- the pupil view


def goal_choices() -> list[tuple[str, str]]:
    """What people pick from: the short label, with the code; the value is the id the engine uses."""
    return [("No goal", NO_GOAL)] + [(f"{s['code']} · {s['short_label']}", s["id"]) for s in engine().skills()]


def reviewed_languages() -> list[str]:
    K = engine()
    langs = {"en"}
    if K.bundle is not None:
        langs |= {str(e.get("lang")) for entries in K.bundle.contexts.values() for e in entries if e.get("reviewed")}
    return sorted(langs, key=lambda code: LANG_NAMES.get(code, code))


def p4_history(at: datetime) -> list[tuple[str, bool, datetime]]:
    """The fixed history of the loaded pupil, oldest first (the same seeded pupil as the Streamlit demo)."""
    G = engine().graph
    rng = random.Random(SEED)

    def answers_on(skills, count, right):
        picked = sorted(rng.sample(sorted(skills), count))
        correct = set(rng.sample(picked, right))
        return [(n, n in correct) for n in picked]

    grade = lambda n: int(G.nodes[n]["grade"])  # noqa: E731
    lower = answers_on((n for n in G if grade(n) <= 3), LOWER_ANSWERS, LOWER_RIGHT)
    b4 = answers_on((n for n in G if grade(n) == 4), B4_ANSWERS, B4_RIGHT)
    taken = {n for n, _ in lower}
    strong = sorted(rng.sample(sorted(n for n in G if grade(n) in (2, 3) and n not in taken), STRONG_SKILLS))
    t_strong, t_lower, t_b4 = (
        at - timedelta(weeks=STRONG_WEEKS),
        at - timedelta(weeks=LOWER_WEEKS),
        at - timedelta(days=B4_DAYS),
    )
    history = [(n, True, t_strong + timedelta(minutes=3 * j + i)) for j, n in enumerate(strong) for i in range(3)]
    history += [(n, ok, t_lower) for n, ok in lower]
    return history + [(n, ok, t_b4) for n, ok in b4]


def mostly(outcomes: list[bool]) -> str:
    right = sum(outcomes)
    return (
        "mostly right" if 2 * right > len(outcomes) else "mostly wrong" if 2 * right < len(outcomes) else "half right"
    )


def loaded_message(history) -> str:
    G = engine().graph
    strong = {n for n, _, _ in history if sum(1 for m, _, _ in history if m == n) > 1}
    lower = [ok for n, ok, _ in history if int(G.nodes[n]["grade"]) <= 3 and n not in strong]
    b4 = [ok for n, ok, _ in history if int(G.nodes[n]["grade"]) == 4]
    return (
        f"Loaded {plural(len(history), 'past answer')}: {plural(len(strong), 'skill')} learned well "
        f"{plural(STRONG_WEEKS, 'week')} ago (3 right answers each), {len(lower)} on other B1 to B3 skills "
        f"{plural(LOWER_WEEKS, 'week')} ago ({mostly(lower)}) and {len(b4)} on B4 skills "
        f"{plural(B4_DAYS, 'day')} ago ({mostly(b4)})."
    )


def card_markdown(r) -> str:
    lines = [f"### {r.short_label}", f"`{r.code}` · B{r.grade} · {r.strand}"]
    if r.description and r.description != r.short_label:
        lines.append(f"*{r.description}*")
    if r.is_review:
        lines.append("🔁 **Review:** learned before and starting to fade. A quick check keeps it fresh.")
    lines.append(f"**Mastery so far:** {r.mastery:.0%}")
    lines.append(f"**Why this skill:** {r.rationale}")
    if r.ghana_context:
        lines.append(f"> **Try it in Ghana:** {r.ghana_context}")
    return "\n\n".join(lines)


def cell_style(value) -> str:
    if pd.isna(value):
        return ""
    fill = next(colour for bound, colour in BANDS if value < bound)
    return f"background-color: {fill}; color: {CELL_TEXT}"


def mastery_table(learner, at: datetime) -> str:
    """Average mastery by strand (rows) and class (columns) as a coloured HTML table."""
    K, G = engine(), engine().graph
    mastery = K.mastery_map(learner, now=at)
    cells: dict[tuple[str, int], list[float]] = {}
    for n, d in G.nodes(data=True):
        cells.setdefault((str(d["strand_name"]), int(d["grade"])), []).append(mastery[n])
    strands = list(dict.fromkeys(s for s, _ in cells))
    grades = sorted({g for _, g in cells})
    head = "".join(f"<th>B{g}</th>" for g in grades)
    rows = []
    for s in strands:
        tds = []
        for g in grades:
            values = cells.get((s, g))
            if not values:
                tds.append("<td></td>")
                continue
            mean = sum(values) / len(values)
            tds.append(f'<td style="{cell_style(mean)};text-align:center">{mean:.0%}</td>')
        rows.append(f"<tr><th style='text-align:left'>{html.escape(s)}</th>{''.join(tds)}</tr>")
    return (
        '<table style="width:100%;border-collapse:collapse" aria-label="Mastery by class and strand">'
        f"<thead><tr><th></th>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def view(learner, grade_max, goal, mix, drafts, lang):
    """Everything the pupil view shows for this learner and these controls."""
    K, at = engine(), now()
    tick = perf_counter()
    recs = K.recommend(
        learner,
        k=CARDS,
        goal=goal or None,
        grade_max=int(grade_max),
        goal_mode="soft" if mix else "strict",
        now=at,
        eps=EPS,
        lang=lang or "en",
        include_drafts=bool(drafts),
    )
    took = (perf_counter() - tick) * 1000
    head = (
        f"Picked in {took:.2f} ms. Mark an answer right or wrong to watch the path change. Review and exploration "
        "are on: one card may bring back a skill learned before, and close contenders are shuffled a little."
    )
    if goal:
        head = f"**Goal:** {K.label(goal)}\n\n" + head
    if not recs:
        head += "\n\nNo skill is ready under these settings. Raise the class cap or pick another goal."
    shown = [r.indicator for r in recs]
    cards = []
    for i in range(CARDS):
        if i < len(recs):
            cards += [gr.Column(visible=True), card_markdown(recs[i])]
        else:
            cards += [gr.Column(visible=False), ""]
    drafts_note = (
        "⚠️ Draft settings are on. The 'Try it in Ghana' text may not have been checked by a teacher." if drafts else ""
    )
    return [learner, shown, head, drafts_note, *cards, mastery_table(learner, at)]


def new_pupil(*controls):
    return [*view(engine().new_state("demo"), *controls), ""]


def load_p4_pupil(*controls):
    K = engine()
    pupil, history = K.new_state("demo"), p4_history(now())
    for n, correct, when in history:
        K.update(pupil, n, correct, at=when)
    return [*view(pupil, *controls), loaded_message(history)]


def answer(slot: int, correct: bool):
    def handler(learner, shown, *controls):
        if learner is None:
            learner = engine().new_state("demo")
        if slot < len(shown):  # the skill this card showed when it was drawn, never a later one
            engine().update(learner, shown[slot], correct, at=now())
        return view(learner, *controls)

    return handler


# ----------------------------------------------------------------------------- the evidence


def answers(n) -> str:
    return f"{n / 1e6:.2f}M" if n >= 999_500 else f"{n / 1e3:.0f}k" if n >= 1000 else f"{n:.0f}"


def auc_of(block: dict, model: str):
    metrics = block.get("metrics") if isinstance(block.get("metrics"), dict) else {}
    row = metrics.get(model) if isinstance(metrics.get(model), dict) else {}
    value = row.get("AUC")
    if value is None and isinstance(block.get("AUC"), dict):
        value = block["AUC"].get(model)
    return round(value, 3) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def dataset_rows(ev: dict):
    models = list(AUC_MODELS)
    present, missing = [], []
    for key, name in DATASETS:
        block = block_at(ev, key)
        if block is None or block.get("skipped"):
            missing.append(name)
            continue
        models += [m for m in (block.get("metrics") or {}) if m not in models]
        present.append((name, block))
    rows = []
    for name, block in present:
        n = block.get("rows")
        row = {"Dataset": name, "Answers": answers(n) if isinstance(n, (int, float)) and n > 0 else ""}
        row.update({m: auc_of(block, m) for m in models})
        rows.append(row)
    return pd.DataFrame(rows, columns=["Dataset", "Answers", *models]), missing


def mean_sd(cell) -> str:
    return f"{cell['mean']} ± {cell['sd']}" if isinstance(cell, dict) and "mean" in cell else ""


def simulation_rows(rep: dict) -> pd.DataFrame:
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
    return pd.DataFrame(rows)


def deep_kt_text(ev: dict) -> str:
    deep = block_at(ev, "kt_deep/decision")
    if deep is None:
        return f"Deep knowledge tracing benchmark: {NOT_RUN}."
    verdict = (
        f"Decision: adopt {deep['model']}."
        if deep.get("adopt")
        else "Decision: keep Kwan's own learner model. No deep model met the rule."
    )
    return f"{verdict}\n\nThe rule: {deep.get('rule', '')}"


def goal_path_text(ev: dict) -> str:
    gp = block_at(ev, "goal_path")
    share = gp.get("on_path_share") if gp and isinstance(gp.get("on_path_share"), dict) else None
    if share is None:
        return f"Goal path check: {NOT_RUN}."
    return (
        f"On the path to the goal (strict): **{share.get('strict', 0):.0%}** · with other skills mixed in: "
        f"**{share.get('soft', 0):.0%}**\n\n{gp.get('setup', '')}"
    )


def latency_text(latency: dict) -> str:
    if not isinstance(latency.get("recommend_top3_mean_ms"), (int, float)):
        return f"Latency benchmark: {NOT_RUN}."
    return (
        f"Top 3 picks: **{latency['recommend_top3_mean_ms']} ms** on average, "
        f"{latency.get('recommend_top3_p95_ms', '')} ms for the slowest 5%. "
        f"{plural(latency.get('calls', 0), 'call')} on {latency.get('hardware', 'one CPU core')}, in-process."
    )


@cache
def small_simulation() -> pd.DataFrame:
    rep = simulate.report(engine().graph, n_learners=SIM_LEARNERS, steps=SIM_STEPS, goal_max_steps=SIM_STEPS)
    return simulation_rows(rep)


# ----------------------------------------------------------------------------- the page


def build() -> gr.Blocks:
    K, G = engine(), engine().graph
    ev = evidence()
    latency = block_at(ev, "latency") or {}
    with gr.Blocks(title=TITLE) as demo:
        gr.HTML(logo_html())
        gr.Markdown(INTRO)
        gr.Markdown(f"NaCCA B1 to B6 Mathematics · {plural(G.number_of_nodes(), 'skill')} · bundle {K.bundle_version}")
        learner = gr.State(None)
        shown = gr.State([])
        with gr.Tabs():
            with gr.Tab("Pupil view"):
                with gr.Row():
                    with gr.Column(scale=1):
                        grade_max = gr.Slider(1, 6, value=6, step=1, label="Recommend skills up to class (B1 to B6)")
                        goal = gr.Dropdown(
                            goal_choices(),
                            value=NO_GOAL,
                            label="Goal skill, e.g. next term's topic (optional)",
                            info="Type to search. Skills on the way to the goal come first.",
                            filterable=True,
                        )
                        mix = gr.Checkbox(
                            False, label="Mix in other skills", info="Also offer ready skills off the goal's path."
                        )
                        drafts = gr.Checkbox(
                            False,
                            label="Show unreviewed draft settings",
                            info="Practice settings no teacher has checked yet. They may contain mistakes.",
                        )
                        langs = reviewed_languages()
                        lang = gr.Dropdown(
                            [(LANG_NAMES.get(c, c), c) for c in langs],
                            value="en",
                            label="Language",
                            info="Only languages with teacher-reviewed settings are listed"
                            + (": today that is English alone." if langs == ["en"] else "."),
                        )
                        load_btn = gr.Button("Load a P4 pupil with gaps in earlier classes", variant="primary")
                        reset_btn = gr.Button("Start over with a new pupil")
                        notice = gr.Markdown()
                    with gr.Column(scale=2):
                        gr.Markdown("## What to learn next")
                        head = gr.Markdown()
                        drafts_note = gr.Markdown()
                        slots, slot_md, buttons = [], [], []
                        for i in range(CARDS):
                            with gr.Column(visible=False, variant="panel") as col:
                                md = gr.Markdown()
                                with gr.Row():
                                    right = gr.Button("✅ Got it right", size="sm")
                                    wrong = gr.Button("❌ Got it wrong", size="sm")
                            slots.append(col)
                            slot_md.append(md)
                            buttons.append((i, right, wrong))
                gr.Markdown(
                    "## Mastery by class and strand\n\nAverage estimated mastery across the skills in each cell. "
                    "Red is below 50%, amber 50% to 70%, light green 70% to 85% and green 85% and above, "
                    "which Kwan counts as mastered."
                )
                grid = gr.HTML()

            with gr.Tab("Curriculum map"):
                by_type = Counter(str(d.get("type")) for _, _, d in G.edges(data=True))
                kinds = ", ".join(f"{count} {kind}" for kind, count in sorted(by_type.items()))
                gr.Markdown(
                    "Every NaCCA learning indicator is a skill. Kwan links each skill to the ones that should come "
                    "before it: the next indicator in the same standard, the same standard in the next class, and "
                    "the next standard in the same sub-strand.\n\n"
                    f"**{G.number_of_nodes()}** skills · **{G.number_of_edges()}** prerequisite links ({kinds})"
                )
                gr.HTML(map_page())

            with gr.Tab("Evidence"):
                gr.Markdown(
                    "### How well does Kwan predict a pupil's next answer?\n\nThe learner model was tested on public "
                    "datasets of real student answers, holding out 20% of students it never saw. AUC is 0.5 for a "
                    "coin flip and 1.0 for perfect prediction. AMMORE comes from a WhatsApp maths tutor used in West "
                    "Africa; its Ghana rows are the pupils in Ghana."
                )
                auc, missing = dataset_rows(ev)
                gr.Dataframe(auc, interactive=False, show_label=False)
                if missing:
                    gr.Markdown(f"{NOT_RUN.capitalize()} or skipped: {', '.join(missing)}.")
                gr.Markdown("### Would a deep learning model do better?\n\n" + deep_kt_text(ev))
                gr.Markdown("### Does a strict goal keep the path on track?\n\n" + goal_path_text(ev))
                gr.Markdown("### How fast is it?\n\n" + latency_text(latency))
                gr.Markdown(
                    "### Does following Kwan help pupils learn more?\n\nThe simulator gives seeded virtual pupils a "
                    "daily practice routine, lets them forget, and counts the new skills they master. Everything in "
                    "it is an assumption, not a measurement of Ghanaian pupils."
                )
                sim = block_at(ev, "simulation/v2")
                if sim is None:
                    gr.Markdown(f"Simulator v2: {NOT_RUN}.")
                else:
                    gr.Markdown(str(sim.get("setup", "")))
                    gr.Dataframe(simulation_rows(sim), interactive=False, show_label=False)
                sim_btn = gr.Button(
                    f"Run the simulator now ({plural(SIM_LEARNERS, 'pupil')} per type, {SIM_STEPS} steps)"
                )
                sim_out = gr.Dataframe(interactive=False, show_label=False, visible=False)
                sim_btn.click(lambda: gr.Dataframe(small_simulation(), visible=True), outputs=sim_out)

        gr.Markdown(
            f"---\n\n{DISCLAIMER}\n\nBundle {K.bundle_version} ({K.text_mode} mode) · code MIT, Kwan-written "
            "content CC BY 4.0 · source and docs at [github.com/rofyray/kwan](https://github.com/rofyray/kwan)"
        )

        controls = [grade_max, goal, mix, drafts, lang]
        cards = [x for pair in zip(slots, slot_md) for x in pair]
        outputs = [learner, shown, head, drafts_note, *cards, grid]
        demo.load(new_pupil, inputs=controls, outputs=[*outputs, notice])
        reset_btn.click(new_pupil, inputs=controls, outputs=[*outputs, notice])
        load_btn.click(load_p4_pupil, inputs=controls, outputs=[*outputs, notice])
        for control in controls:
            control.change(view_with_none, inputs=[learner, *controls], outputs=outputs)
        for i, right, wrong in buttons:
            right.click(answer(i, True), inputs=[learner, shown, *controls], outputs=outputs)
            wrong.click(answer(i, False), inputs=[learner, shown, *controls], outputs=outputs)
    return demo


def view_with_none(learner, *controls):
    return view(learner if learner is not None else engine().new_state("demo"), *controls)


try:  # On a ZeroGPU Space the `spaces` package expects one GPU function at start-up; Kwan needs none.
    import spaces

    @spaces.GPU(duration=1)
    def _no_gpu_needed() -> None:  # never called
        return None
except ImportError:
    pass


demo = build()

if __name__ == "__main__":
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        css=CSS,
        favicon_path=MARK,
        theme=THEME,
    )
