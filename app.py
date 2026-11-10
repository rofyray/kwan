"""Kwan demo UI (Streamlit). Run: streamlit run app.py"""
import inspect, random, re
import pandas as pd, streamlit as st
import engine, graph, results, simulate

st.set_page_config(page_title="Kwan · the next step in maths", page_icon="🧭", layout="wide")

PROFILE_LABELS = {"beginner_p1": "Beginner in P1", "gapped_p4": "P4 pupil with gaps in B1 to B3"}
POLICY_LABELS = {"kwan": "Following Kwan", "syllabus": "Syllabus order", "random": "Random practice"}
AUC_MODELS = ["global-mean", "skill-mean", "bkt", "elo", "elo-multi (Kwan)"]
NO_GOAL = "(no goal)"
LOADED_MSG = "Loaded about 20 past answers: strong on most B1 to B3 skills, weak on B4."
SIM_PUPILS, SIM_TURNS = getattr(simulate, "N_LEARNERS", 30), getattr(simulate, "STEPS", 150)   # what compare() runs

# st.dataframe changed its API between the oldest Streamlit we allow (requirements.txt) and current releases:
# width="stretch" replaced use_container_width (deprecated, logs a warning on every call) and `placeholder`
# was added to choose what a missing value looks like. Ask the installed version which one it speaks.
_DF_PARAMS = inspect.signature(st.dataframe).parameters
FULL_WIDTH = {"width": "stretch"} if _DF_PARAMS["width"].default == "stretch" else {"use_container_width": True}
BLANK_IF_MISSING = {"placeholder": ""} if "placeholder" in _DF_PARAMS else {}


@st.cache_resource(show_spinner=False)
def load_engine():
    """Curriculum graph, learner model and recommender, built once per server and shared by every session.
    None of them stores anything about a pupil: a pupil is the LearnerState kept in st.session_state."""
    G = graph.load()
    L = engine.EloLearner(G=G)
    return G, L, engine.Recommender(G, L)


@st.cache_data(show_spinner=False)
def simulation_rows():
    """Run the policy simulation (seeded, so the result is the same every time) and label it for the table."""
    G = load_engine()[0]
    return [{"pupil type": PROFILE_LABELS.get(prof, prof), "how they practised": POLICY_LABELS.get(pol, pol),
             "new skills mastered (avg)": round(mu, 1), "spread (sd)": round(sd, 1)}
            for prof in simulate.PROFILES for pol, (mu, sd) in simulate.compare(G, profile=prof).items()]


def start_over():
    st.session_state.learner = engine.LearnerState("demo")
    st.session_state.pop("notice", None)


def load_p4_pupil():
    """Always builds the same pupil on a fresh state, so pressing the button twice does not stack two histories."""
    G, L, _ = load_engine()
    pupil, rng = engine.LearnerState("demo"), random.Random(7)
    for n in sorted(G, key=lambda x: (G.nodes[x]["grade"], x))[:60]:
        if rng.random() < 0.35:
            L.update(pupil, n, rng.random() < (0.85 if G.nodes[n]["grade"] <= 3 else 0.2))
    st.session_state.learner = pupil
    st.session_state.notice = LOADED_MSG


def answer(indicator, correct):
    """Bound to one card's skill when the card is drawn, so a click can never land on a different skill."""
    load_engine()[1].update(st.session_state.learner, indicator, correct)


def plain(text):
    """Curriculum text is shown as written: escape the characters Markdown would treat as formatting."""
    return re.sub(r"([\\`*_\[\]$~<>#])", r"\\\1", str(text))


def answers(n):
    """259399 -> '259k', 1382727 -> '1.38M'."""
    return f"{n / 1e6:.2f}M" if n >= 999_500 else f"{n / 1e3:.0f}k" if n >= 1000 else f"{n:.0f}"


def evidence():
    """data/eval_results.json as written by the evaluation scripts, or {} if it is missing or unreadable."""
    try:
        ev = results.load()
    except (OSError, ValueError):
        return {}
    return ev if isinstance(ev, dict) else {}


def auc_column(block, name):
    """Column header and AUC per model for one dataset block. Whatever the file does not have stays empty."""
    block = block if isinstance(block, dict) else {}
    rows, auc = block.get("rows"), block.get("AUC")
    auc = auc if isinstance(auc, dict) else {}
    sized = isinstance(rows, (int, float)) and not isinstance(rows, bool) and rows > 0
    return (f"{name} ({answers(rows)} answers)" if sized else name,
            pd.to_numeric(pd.Series([auc.get(m) for m in AUC_MODELS], dtype="object"), errors="coerce"))


G, L, R = load_engine()
if "learner" not in st.session_state:
    st.session_state.learner = engine.LearnerState("demo")

st.title("Kwan 🧭")
st.markdown("**Kwan** is Twi for *the path*. It reads Ghana's Basic 1 to Basic 6 maths curriculum, keeps track of what a pupil has mastered, and picks the next skill to learn, with a reason and an example from everyday Ghanaian life.")
st.caption(f"NaCCA B1 to B6 Mathematics · {G.number_of_nodes()} skills · runs on one CPU core in under a millisecond")

tab1, tab2, tab3 = st.tabs(["Pupil view", "Curriculum map", "Evidence"])
with tab1:
    c1, c2 = st.columns([1, 2])
    with c1:
        grade_max = st.slider("Recommend skills up to class (B1 to B6)", 1, 6, 6)
        goal = st.selectbox("Goal skill, e.g. next term's topic (optional)", [NO_GOAL] + sorted(G.nodes))
        goal = None if goal == NO_GOAL else goal
        st.button("Load a P4 pupil with gaps in earlier classes", on_click=load_p4_pupil)
        if st.session_state.get("notice"):
            st.success(st.session_state.notice)
        st.button("Start over with a new pupil", on_click=start_over)
    S = st.session_state.learner
    with c2:
        res = R.recommend(S, k=3, goal=goal, grade_max=grade_max)
        st.subheader("What to learn next")
        st.caption(f"Picked in {res['latency_ms']} ms. Mark an answer right or wrong to watch the path change.")
        for r in res["recommendations"]:
            ind = r["indicator"]
            with st.container(border=True):
                st.markdown(f"**{ind}** · B{r['grade']} · {r['strand']}  \n{plain(r['description'])}")
                st.progress(min(max(r["mastery"], 0.0), 1.0), text=f"Mastery so far: {r['mastery']:.0%}")
                st.markdown(f"**Why this skill:** {r['rationale']}")
                if r.get("ghana_context"): st.info(f"**Try it in Ghana:** {r['ghana_context']}")
                a, b = st.columns(2)
                # keyed by skill, not by card position: a late or repeated click still belongs to the skill it was made on
                a.button("✅ Got it right", key=f"right_{ind}", on_click=answer, args=(ind, True))
                b.button("❌ Got it wrong", key=f"wrong_{ind}", on_click=answer, args=(ind, False))
    m = pd.DataFrame([{"indicator": n, "grade": d["grade"], "strand": d["strand_name"], "mastery": L.mastery(S, n)}
                      for n, d in G.nodes(data=True)])
    st.subheader("Mastery by class and strand")
    st.caption("Average estimated mastery across the skills in each cell. Green is strong, red needs work.")
    st.dataframe(m.pivot_table(index="strand", columns="grade", values="mastery", aggfunc="mean").style.format("{:.0%}")
                 .background_gradient(cmap="RdYlGn", vmin=0, vmax=1), **FULL_WIDTH)
with tab2:
    st.markdown("Every NaCCA learning indicator is a skill. Kwan links each skill to the ones that should come before it: the next indicator in the same standard, the same standard in the next class, and the next standard in the same sub-strand.")
    c3, c4 = st.columns(2); c3.metric("Skills (NaCCA indicators)", G.number_of_nodes()); c4.metric("Prerequisite links", G.number_of_edges())
    st.dataframe(pd.DataFrame([{"from": a, "to": b, "type": d["type"]} for a, b, d in G.edges(data=True)]),
                 height=300, **FULL_WIDTH)
with tab3:
    st.markdown("**How well does Kwan predict a pupil's next answer?** We tested the learner model on two public datasets of real student answers, holding out 20% of students it never saw in training. AUC is 0.5 for a coin flip and 1.0 for perfect prediction.")
    ev = evidence()
    (head_a, auc_a), (head_e, auc_e) = auc_column(ev.get("assistments_2009"), "ASSISTments 2009"), auc_column(ev.get("eedi_2020_task34"), "Eedi 2020")
    st.dataframe(pd.DataFrame({"Model (AUC, higher is better)": AUC_MODELS, head_a: auc_a, head_e: auc_e}),
                 hide_index=True, **FULL_WIDTH, **BLANK_IF_MISSING)
    bkt = AUC_MODELS.index("bkt")
    st.caption("Global-mean and skill-mean are simple averages. BKT is Bayesian Knowledge Tracing, the classic baseline."
               + (" BKT was run on ASSISTments only." if pd.notna(auc_a[bkt]) and pd.isna(auc_e[bkt]) else ""))
    st.markdown(f"**Does following Kwan help pupils learn more?** The simulation gives {SIM_PUPILS} virtual pupils {SIM_TURNS} practice turns each and counts the new skills they master under Kwan, under the syllabus order, and under random practice.")
    if st.button("Run the simulation (about 10 seconds)"):
        st.session_state.sim_rows = simulation_rows()
    if st.session_state.get("sim_rows"):      # kept for the session, so the table survives the next click elsewhere
        st.dataframe(pd.DataFrame(st.session_state.sim_rows), hide_index=True, **FULL_WIDTH)
