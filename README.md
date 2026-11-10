# Kwan 🧭

**Kwan** is Twi for *the path*. It tells a Ghanaian pupil which maths skill to learn next, explains why, and sets the practice in everyday Ghanaian life.

Built for the DataNova 2026 Global Data-thon (problem DN-ED-01, Adaptive Learning Path Recommendation). Kwan is the first piece of the personalization engine for **Syto**, an AI learning platform for Ghanaian primary pupils.

- **Live demo:** [DEMO URL]
- **Video:** [VIDEO URL]

## Why we built it

Most learning apps and AI tutors used across Africa were designed somewhere else. Their word problems feature snow days and prices in dollars, and the order of topics follows a foreign syllabus. A child in a village in the Volta Region has to decode the story before she can start on the maths, and that extra step is where many children stop.

Kwan starts from what Ghanaian pupils already study. Its skill map is the NaCCA Standards-Based Mathematics Curriculum for Basic 1 to Basic 6, the same document teachers plan their lessons from, so a recommendation lines up with the lesson on the blackboard. Each recommended skill comes with a setting a Ghanaian child will recognise: sharing groundnuts, giving change for kenkey and fish, continuing a kente pattern, working out a trotro fare.

## What it does

1. **Maps the curriculum.** 266 NaCCA learning indicators (for example `B4.1.1.1.2`, "Read and write numbers in figures and in words up to 100,000") linked by 287 prerequisite connections.
2. **Tracks the pupil.** After each answer, Kwan updates its estimate of how well the pupil knows every skill. Evidence flows along the curriculum map, so doing well on a skill also raises the estimate for the skills that build on it.
3. **Picks the next skills.** It looks for skills whose foundations are in place and that sit at a productive level of challenge, leans toward a goal skill if the teacher sets one, and can stay within a class level.
4. **Explains each pick** in plain language, using the same signals that produced the ranking. Example: *"Builds on B2.1.1.1.6, where you are at 74%. You are at 62%, close enough to master with practice. Opens the way to 46 later skills."*
5. **Grounds it in Ghana.** *"Try it in Ghana: Use 1 cedi and 2 cedi coins to make 5 cedis in as many ways as you can."*

A top-3 recommendation takes about 0.18 ms on one CPU core (0.20 ms at the 95th percentile). Kwan makes no network calls when it recommends, which matters for schools with patchy internet.

## Results

How well the learner model predicts a pupil's next answer, tested on students it never saw in training (AUC: 0.5 is a coin flip, 1.0 is perfect):

| Dataset | Skill average | BKT | Classic Elo | **Kwan** |
|---|---|---|---|---|
| ASSISTments 2009 (259k answers, 4,163 students) | 0.624 | 0.700 | 0.668 | **0.755** |
| Eedi NeurIPS 2020, task 3/4 (1.38M answers, 4,918 students) | 0.576 | not run | 0.684 | **0.776** |

New skills mastered in a simulation of 30 pupils with 150 practice turns each:

| Pupil | Following Kwan | Syllabus order | Random practice |
|---|---|---|---|
| P4 pupil with gaps in B1 to B3 | **14.0** | 8.8 | 10.9 |
| Beginner in P1 | **25.6** | 24.0 | 10.4 |

Kwan helps most where pupils have gaps. For a beginner, the syllabus order is already a good path, and Kwan roughly matches it.

**Simulator assumptions:** each virtual pupil has a hidden true state for every skill. Practising a skill whose prerequisites are truly known succeeds with probability 0.35 per turn, otherwise 0.03. Slip rate is 0.10 and guess rate 0.20. The P4 profile starts knowing 75% of B1 to B3 skills and 5% above that; the P1 profile knows 15% of B1 to B2 and 3% above. These are modelling choices, so treat the simulation as a sanity check on the policy. The real-data AUC results stand on their own.

## How it works

- **Curriculum parsing** (`parse_curriculum.py`): reads the NaCCA PDFs (via `pdftotext -layout`) and extracts each indicator code and description, following the description across wrapped lines in the same layout column.
- **Curriculum map** (`graph.py`): links the next indicator in the same standard, the same standard in the next class, and the next standard in the same sub-strand. The result is checked to be a directed acyclic graph.
- **Learner model** (`engine.py`): multi-level Elo. The chance of a correct answer is `σ(θ_pupil + θ_pupil,skill − β_item)`. Skills the pupil has not tried borrow evidence from their prerequisites, adjusted for class level. Three correct answers in a row, or an 85% predicted chance, counts as mastered.
- **Recommender** (`engine.py`): scores ready skills on how solid their foundations are (0.30), how close they are to a 60% chance of success (0.25), how close they are to the goal (0.25), how many later skills they open (0.15), and how long since they were last practised (0.05).
- **Ghana context** (`context.py`): a bank of 21 everyday settings grouped by strand and by lower or upper primary.
- **Interfaces:** `api.py` (FastAPI) and `app.py` (Streamlit demo).
- **Evidence** (`eval_kt.py`, `eval_eedi.py`, `simulate.py`, `bench_latency.py`): each script writes its own block of `data/eval_results.json` through `results.py`. Every number in this README comes from that file.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # also needs poppler for pdftotext
make all                                 # download data, parse, build graph, evaluate, simulate, test
make app                                 # Streamlit demo on http://localhost:8501
make api                                 # REST API on http://localhost:8000
```

`make all` runs `data`, `curriculum`, `graph`, `eval`, `sim`, `bench` and `test` in order. `make data` fetches about 670 MB and is safe to run again: files that are already complete are skipped. The parsed curriculum, the graph and the results file are committed, so the demo, the API and the tests run without downloading anything:

```bash
make test                                # full test suite (tests that need raw data skip themselves)
KWAN_RUN_SLOW=1 make test                # also re-check the Eedi result when raw/ is present
make space                               # assemble the Hugging Face Space bundle in build/space/
```

API example:

```bash
curl -s -X POST localhost:8000/attempt -H 'content-type: application/json' \
     -d '{"learner_id":"ama","indicator":"B1.1.1.1.1","correct":true}'
curl -s "localhost:8000/recommend/ama?k=3&goal=B4.1.1.3.3"
```

| Method | Path | What it does |
|---|---|---|
| GET | `/health` | Liveness check with the number of skills and links |
| POST | `/attempt` | Record one answer; returns the predicted chance of a correct answer and the new mastery |
| GET | `/recommend/{learner_id}` | Top `k` next skills (1 to 50, default 3), with optional `goal` and `grade_max` (1 to 6) |
| GET | `/mastery/{learner_id}` | Estimated mastery of every skill |
| DELETE | `/learner/{learner_id}` | Forget a learner (204, or 404 if nothing is stored for that id) |

A new `learner_id` needs no registration: a pupil the API has never seen gets starting recommendations. An unknown indicator or goal returns 404, and out-of-range parameters return 422. Interactive documentation is at `/docs`. Learner state is held in memory, so run a single worker.

## Data and credits

- NaCCA Standards-Based Mathematics Curriculum, Basic 1 to 3 and Basic 4 to 6 (Ministry of Education, Ghana), from curriculumresources.edu.gh. The PDFs are downloaded by `scripts/download_data.sh` and not stored in this repo.
- ASSISTments 2009 skill-builder data: Feng, Heffernan and Koedinger (2009).
- Eedi NeurIPS 2020 Education Challenge: Wang et al. (2020), arXiv:2007.12061.

## Limitations

- A few indicators are still missed by the parser (B1 Algebra in particular), and three descriptions that wrap around a typeset fraction or a centred line are still cut short (B3.1.3.1.1, B4.1.3.1.1, B6.4.1.1.3). Fixes are planned.
- A pupil who does very well on early skills raises the estimate for skills they have not tried yet. In rare cases an untried skill in the next class can already count as mastered and is then not recommended.
- The Ghana-context settings are hand-written and chosen per strand, so a setting fits the area of maths but not always the exact indicator.
- There are no Ghanaian classroom logs yet. The learner model is validated on US and UK data; recalibrating on Ghanaian pupils is part of the Syto pilot.

## What's next

- Use a structured decision model to suggest prerequisite links across strands, reviewed by teachers.
- Benchmark deep knowledge tracing (pyKT) against Kwan's Elo model.
- Replace the context bank with examples drawn from the NaCCA documents, reviewed by Ghanaian teachers, with Twi, Ewe and Ga phrasing for lower primary.
- Add English Language B1 to B6, then JHS Mathematics.
- Run on-device for offline schools, and plug into Syto.

## Team

[Names and roles]

MIT License.
