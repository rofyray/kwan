# Quickstart

Three ways to use Kwan: from Python, from the command line, and over a REST API.

## Install

Kwan needs Python 3.11, 3.12 or 3.13. The core needs only the standard library and `networkx`.

<!-- VERIFY:START id=install-pypi lang=bash -->
```bash
pip install kwan                 # the engine
pip install 'kwan[api]'          # plus the REST API
```
<!-- VERIFY:END -->

From source:

<!-- VERIFY:START id=install-source lang=bash -->
```bash
git clone https://github.com/rofyray/kwan.git
cd kwan
pip install -e '.[dev,api,demo,eval,curriculum]'
```
<!-- VERIFY:END -->

## The bundle

Everything Kwan needs for a subject is in one file, the **bundle**: the skill map, the model parameters, the practice settings and the text templates. For maths it is `kwan-bundle.math.json`. You can download it from a Kwan release or build it yourself (see [Command line](#command-line)).

The bundle for maths ships inside the package, so `Kwan.from_bundle()` with no path just works. It also holds the learner-model parameters and a record of the datasets they were chosen on (see the [Model card](model-card.md)), and it refuses to carry parameters fitted on a non-commercial dataset. A bundle comes in one of two text modes:

- `labels`: skills are shown with Kwan's own short labels. This mode needs nothing from NaCCA, and it is the one the project publishes.
- `full`: skills also carry the NaCCA descriptions and exemplars. `kwan bundle` builds it only when `data/curriculum/permissions.json` records NaCCA's written permission. See [Governance](governance.md).

## Python

### Load Kwan and record an answer

<!-- VERIFY:START id=python-basic lang=python -->
```python
from kwan import Kwan, LearnerState

kwan = Kwan.from_bundle()  # the bundle shipped with the package; pass a path to use another one

# The learner id is any opaque string you choose. Never use a name or a phone number.
state = kwan.new_state("pupil-0042")

# The pupil answered a question on skill B1.1.1.1.1 and got it right.
kwan.update(state, "math:B1.1.1.1.1", correct=True)
```
<!-- VERIFY:END -->

`new_state` gives a `LearnerState`, plain data that records which bundle it was built with. `update` returns the probability the model gave the answer before seeing it, and takes two optional arguments:

- `item`: the id of the question, if you have an item bank with its difficulty;
- `at`: when the answer was given. Pass it if you want forgetting to work.

<!-- VERIFY:START id=python-update-at lang=python -->
```python
from datetime import datetime, timezone

kwan.update(
    state,
    "math:B1.1.1.1.2",
    correct=False,
    item="q-17",
    at=datetime(2026, 11, 10, 9, 30, tzinfo=timezone.utc),
)
```
<!-- VERIFY:END -->

Plain codes such as `B1.1.1.1.2` are accepted as well as `math:B1.1.1.1.2`.

### Ask what to learn next

<!-- VERIFY:START id=python-recommend lang=python -->
```python
recommendations = kwan.recommend(
    state,
    k=3,                              # how many skills to return
    goal="math:B4.1.1.1.2",           # optional: a skill to work towards
    grade_max=4,                      # optional: nothing above Basic 4
    goal_mode="strict",               # "strict" (default) or "soft"
    now=datetime.now(timezone.utc),   # optional: needed for forgetting and review
)

for rec in recommendations:
    print(rec.code, rec.short_label)
    print("Why:", rec.rationale)
    print("Try it in Ghana:", rec.ghana_context)
```
<!-- VERIFY:END -->

Each recommendation carries the skill (`indicator` is the canonical id `math:B1.1.1.1.2`, `code` the bare code), a score, the pupil's current mastery of it, the short label, a reason (`rationale`), the same reason as translatable parts (`rationale_parts`), a practice setting (`ghana_context`), and `is_review`, which is true for a skill learned before that is starting to fade. `rec.to_dict()` gives plain data.

With `goal_mode="strict"`, skills on the path to the goal come first. With `"soft"`, the goal is one signal among several.

### See the whole picture

<!-- VERIFY:START id=python-mastery lang=python -->
```python
mastery = kwan.mastery_map(state)     # {canonical indicator id: estimated mastery from 0 to 1}
print(mastery["math:B1.1.1.1.1"])
```
<!-- VERIFY:END -->

### Save and load a learner

Learner state is plain data. Keep it wherever you like.

<!-- VERIFY:START id=python-state-roundtrip lang=python -->
```python
import json

saved = json.dumps(state.to_dict())
state = LearnerState.from_dict(json.loads(saved))
```
<!-- VERIFY:END -->

Kwan also comes with two ready-made stores, one in memory and one on SQLite. Every store has a delete operation.

Kwan gives the same output for the same bundle, the same answers and the same `now`.

## Command line

Installing Kwan adds the `kwan` command.

<!-- VERIFY:START id=cli-overview lang=bash -->
```bash
kwan --version

# Curriculum: download the official PDFs, parse them, write the QA report
kwan curriculum fetch
kwan curriculum parse
kwan curriculum qa

# Skill map: build it, suggest extra links, and exchange review sheets with teachers
kwan graph build
kwan graph augment --dry-run
kwan graph review-export
kwan graph review-import

# Evidence: evaluate on a dataset, run the simulator, time the recommender
kwan eval <dataset>
kwan sim --out data/eval_results.json
kwan bench

# Bundle: write kwan-bundle.math.json
kwan bundle --subject math --text labels --out build/bundles/
```
<!-- VERIFY:END -->

Some commands need extra packages: `kwan curriculum` needs `kwan[curriculum]`, and `kwan eval` needs `kwan[eval]`. The command tells you what to install if something is missing. Run `kwan <command> --help` for the options of each command.

There are a few more: `kwan graph validate` checks the map and writes the skill map report, `kwan fit` chooses the forgetting time, the step sizes and the item banks on the public datasets, and `kwan context` runs the checks on practice settings, writes and imports teacher review sheets, and reports coverage. `kwan --help` lists them all. After any command that changes `data/eval_results.json`, `python scripts/fill_docs.py` refreshes the result tables in the README and the docs.

`kwan curriculum fetch` downloads the NaCCA PDFs to your machine. They are © NaCCA, Ministry of Education. Do not commit them or text taken from them.

## REST API

The API is an optional extra. It keeps learner state in a SQLite file by default (`kwan.sqlite3` in the working directory; set `KWAN_STORE=sqlite:///path/to/file` or `KWAN_STORE=memory` to change that).

### Start the server

<!-- VERIFY:START id=api-start lang=bash -->
```bash
pip install 'kwan[api]'
export KWAN_API_KEYS="change-me-to-a-long-random-string"
uvicorn kwan.api.app:app --port 8000
```
<!-- VERIFY:END -->

Every request except `/v1/health` must carry one of the keys in `KWAN_API_KEYS` in the `X-API-Key` header. When `KWAN_API_KEYS` is not set the API logs a warning and accepts every request, which is fine on your own machine only.

### Routes

| Method | Path | What it does |
|---|---|---|
| `GET` | `/v1/health` | Says the service is up |
| `POST` | `/v1/attempts` | Records one answer |
| `GET` | `/v1/learners/{id}/recommendations` | The next skills for a learner |
| `GET` | `/v1/learners/{id}/mastery` | The estimated mastery of every skill |
| `DELETE` | `/v1/learners/{id}` | Forgets a learner |

### Example

<!-- VERIFY:START id=api-curl lang=bash -->
```bash
KEY="change-me-to-a-long-random-string"

curl -s localhost:8000/v1/health

curl -s -X POST localhost:8000/v1/attempts \
     -H "X-API-Key: $KEY" -H "content-type: application/json" \
     -d '{"learner_id": "pupil-0042", "indicator": "math:B1.1.1.1.1", "correct": true}'

curl -s -H "X-API-Key: $KEY" \
     "localhost:8000/v1/learners/pupil-0042/recommendations?k=3&goal=math:B4.1.1.1.2"

curl -s -H "X-API-Key: $KEY" localhost:8000/v1/learners/pupil-0042/mastery

curl -s -X DELETE -H "X-API-Key: $KEY" localhost:8000/v1/learners/pupil-0042
```
<!-- VERIFY:END -->

A learner id needs no registration. A pupil the API has never seen gets starting recommendations. Send an `idempotency_key` with each attempt (or an `Idempotency-Key` header) and a retried request is applied once. Interactive documentation is served at `/docs` while the server runs. The Round 1 routes (`/attempt`, `/recommend/{id}`, ...) still answer, with a `Deprecation` header and a `Link` to the v1 route that replaces them.

## Keep pupils safe

- Use learner ids that mean nothing outside your system. Never a name, a phone number or an index number.
- Kwan stores no names, emails or free text. Do not add them to requests.
- Delete a learner's state when it is no longer needed.

More on this in the [Model card](model-card.md).

## Next steps

- [Method](method.md): how the map, the learner model and the recommender work.
- [Evidence](evidence.md): how well it works.
- [Context style guide](context-style-guide.md): how practice settings are written.
- [CONTRIBUTING.md](https://github.com/rofyray/kwan/blob/main/CONTRIBUTING.md): how to help.
