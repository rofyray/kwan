# Changelog

All notable changes to Kwan are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the package follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Numbers are never written in this file. Every result lives in `data/eval_results.json` and is rendered into the README and the docs by `uv run scripts/fill_docs.py`.

## [Unreleased]

- **Wording:** Kwan is Twi for "path" in the README, the docs, both demos and the citation file, and the README no longer tells how the project began.

## [1.0.1] - 2026-11-12

Documentation and packaging only; the engine, the model parameters and the skill map are unchanged.

- **README** for the released package: PyPI, docs and demo links without pre-release notes, badges, the Kwan logo (light and dark versions), and the why-Kwan story set "in a village somewhere in Ghana".
- **Package links** on PyPI now include the documentation site and the changelog.
- **Docs site** at https://rofyray.github.io/kwan/ is published on every push to `main`.
- The bundle version follows the package: `math-1.0.1`.

## [1.0.0] - 2026-11-11

- **Skill catalogue for people.** `GET /v1/skills` lists every skill with a short human-readable label, filtered by `grade`, `strand` and `q`; recommendations echo the goal with its label. `Kwan.skills()` and `Kwan.label()` do the same in Python. Codes stay the only thing the engine accepts.
- **Gradio demo.** The public Space is now a Gradio app (`demo/gradio_app.py`, extra `gradio`), which runs on the free Hugging Face plan; the Streamlit demo and its Docker image stay. `space-deploy.yml` deploys the Gradio Space and stops quietly when `HF_TOKEN` or `HF_SPACE` is not set.
- **uv.** Setup, CI, release and the Docker image use uv; development tools live in the `dev` and `docs` dependency groups, pinned by `uv.lock`.
- **Maintainer tasks** page: releasing, the Space, teacher review and the decision model, step by step.

## [1.0.0rc1] - 2026-11-11

The first release candidate of Kwan 1.0: the Round 1 prototype rebuilt as a Python package, with complete curriculum data, a learner model validated on African pupils, a recommender that follows goals strictly, a practice-setting pipeline with teacher review, a bundle format, a store, a versioned API, a Docker demo and a docs site.

### Added

- **Package.** `kwan` is now a package under `src/kwan`, installable with `uv add kwan` (or `pip install kwan`) and extras `api`, `demo`, `eval`, `kt`, `graph-ai`, `curriculum` and `postgres`; development tools live in the `dev` and `docs` dependency groups, pinned by `uv.lock`. The core imports only the standard library and `networkx`. The public interface is `kwan.Kwan` (`from_bundle`, `new_state`, `update`, `recommend`, `mastery_map`, `resolve`), `kwan.LearnerState` and `kwan.Recommendation`.
- **Namespaced ids.** Indicator ids are `math:B4.1.1.1.2`; the bare code is accepted everywhere as an alias (`kwan.ids`).
- **Curriculum pipeline v2** (`kwan.curriculum`): `kwan curriculum fetch` records the URL, size, SHA-256 and date of each official PDF; `parse` reads every indicator with strand and sub-strand names, descriptions and exemplars through pdfplumber with a pdftotext cross-check; `qa` compares the parse with a hand count per class and strand and writes an HTML report. A corrections file records misprinted codes and how each is handled. NaCCA text goes only to a git-ignored `build/` folder; the committed structure file holds codes, numbers, names and Kwan-written short labels.
- **Graph schema v2** (`kwan.graph.schema`): nodes carry the indicator record, every edge carries a type (`sequence`, `vertical`, `bridge`, `augmented`) and provenance (what made it, when, review status). `kwan graph build`, `validate` (writes the skill map report) and `export` (interactive HTML).
- **AI-augmented graph** (`kwan.graph.augment`, `candidates`, `review`, `providers`, `junyi`): candidate pairs from cross-strand rules, a decision-model interface with a local sentence-encoder provider, a benchmark against Junyi Academy's expert links with a precision bar, `kwan graph augment`, `review-export` and `review-import`. Proposed edges stay pending until a teacher accepts them; only accepted edges ship.
- **Learner model v2** (`kwan.learner`): forgetting with a fitted time constant, streak expiry, an item-difficulty bank, a BKT baseline, a fast graph-free scorer, and a parameters file with provenance. Timestamps are accepted on every update and request.
- **Fits** (`kwan.eval.fit`, `kwan fit`): grid search for the forgetting time and the Elo step sizes on validation students, item-bank fits, and a report that chooses the shipped defaults and refuses anything fitted on non-commercial data.
- **Deep knowledge tracing benchmark** (`kwan.eval.deep`, `kwan eval kt-deep`): DKT, SAKT, simpleKT and AKT from pyKT on the same folds as Kwan's Elo, with the D7 adoption rule applied mechanically and the decision recorded in the results file.
- **Recommender v2** (`kwan.recommend`): strict goal mode by default, seeded exploration (Mulberry32 seeded by CRC32 of `learner_id:step`), a review slot for skills that are starting to fade, strand variety, and reasons built from parts rendered through `data/i18n/en.json` so they can be translated. A v1 equivalence mode reproduces the Round 1 output exactly.
- **Simulator v2** (`kwan sim --v2`): forgetting in the hidden state, four pupil profiles, six policies including a remedial-first rule, retention after a rest, and steps to a goal. The goal-path check (strict against soft) is written to the results file.
- **Data and evaluation harness v2** (`kwan.data`, `kwan.eval`): a dataset registry (`data/datasets.yaml`) with licences and a `commercial_params_ok` flag, loaders for ASSISTments 2009, 2012-13, 2015 and 2017, Eedi task 3/4 and task 1/2, AMMORE (all countries and Ghana only), XES3G5M and Junyi, one harness, and `kwan eval <dataset>` or `kwan eval all`.
- **Ghana context v2** (`kwan.context`): per-indicator practice settings with review status, drafts for every indicator, automatic checks (form, length, reading level, banned terms, money format, Ghanaian terms, duplicates), teacher review sheets (`kwan context review-export` and `review-import`), machine-drafted Twi, Ewe and Ga translations that stay hidden until reviewed, label reviews, and a coverage report. A style guide and a teacher review guide in the docs.
- **Bundle** (`kwan.bundle`, `kwan bundle`): one versioned file per subject with the graph, parameters, contexts, context bank, language templates and provenance; `text_mode` records whether NaCCA text is present. The shipped maths bundle lives in the package.
- **Store** (`kwan.store`): `MemoryStore`, `SQLiteStore` and an optional `PostgresStore`, with optimistic versioning, idempotent attempts and a delete operation.
- **REST API v1** (`kwan.api.v1`): `/v1/health`, `/v1/attempts`, `/v1/learners/{id}/recommendations`, `/v1/learners/{id}/mastery` and `DELETE /v1/learners/{id}`, API-key auth through `KWAN_API_KEYS`, a persistent store through `KWAN_STORE`, idempotency keys, machine-readable error codes and JSON logs.
- **Demo** (`demo/`): the Streamlit app rebuilt on the public package interface, with a strict goal, a "mix in other skills" switch, a labelled switch for unreviewed draft settings, a language choice, review picks marked, a Dockerfile for the Hugging Face Space and a smoke test.
- **Workflows:** `release.yml` (PyPI trusted publishing on a `v*` tag, with the bundle and the results file attached to the GitHub release), `space-deploy.yml` (deploys the demo to the Space), `slow-eval.yml` (weekly full evaluation that opens a pull request when the results file changes) and `docs.yml` (`mkdocs build --strict` on every push, deploy to GitHub Pages by hand).
- **Docs site** (`mkdocs.yml`, `docs/`): overview, quickstart (Python, CLI, API), method, evidence, data card, model card, context style guide, teacher review guide, governance, roadmap, and the two QA reports.
- **Community files:** `CONTRIBUTING.md` with a developer track and a teacher track, `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1), `CONTRIBUTORS.md`, `CITATION.cff`, `LICENSE-CONTENT` (CC BY 4.0 for Kwan-written content), issue templates (bug, feature, teacher review, curriculum error) and a pull request template.
- **Results plumbing:** `scripts/fill_docs.py` renders every RESULTS block in the README and the docs from `data/eval_results.json`, and `--check` fails when they drift; `tests/test_readme.py` runs it.
- **Examples:** `examples/quickstart.py`, `examples/simulate_a_class.py` and `examples/evaluate_on_ammore.py`.

### Changed

- The README was rewritten for 1.0: results as generated tables, install and quickstart on the package interface, the CLI and the v1 API, the bundle, the limitations, the Round 1 team section replaced by Maintainers and Contributors, the demo address, and the NaCCA disclaimer.
- Strict goal mode is the default. `goal_mode="soft"` keeps the Round 1 behaviour.
- Rounding is half-up and the practice-setting pick uses CRC32 of the bare code, so a port in another language can match Kwan exactly (decision D8). The Round 1 plural fix ("1 later skill") is in.
- The demo message about the seeded pupil is computed from the seeded history instead of being typed.
- The Hugging Face Space uses the Docker SDK; the Streamlit SDK configuration and the old Space build script are gone.
- Learner state is schema version 2 (`last_seen_at`, `bundle_version`, `subject`); version 1 state is read and upgraded on load.
- The Round 1 API routes (`/health`, `/attempt`, `/recommend/{id}`, `/mastery/{id}`, `/learner/{id}`) still answer for one minor release, with a `Deprecation` header and a `Link` to the v1 route that replaces each one.
- `Makefile` targets drive the package CLI (`curriculum`, `graph`, `eval`, `fit`, `sim`, `bench`, `bundle`, `docs`); `make all` runs the whole pipeline.
- `data/eval_results.json` is written only through `kwan.eval.results.update`, one top-level key per script, as before, and now holds blocks for every dataset, the fits, the deep benchmark, the simulator, the goal path, the latency benchmark, the curriculum, the graph, the augmented graph and the contexts.

### Removed

- The Round 1 top-level modules (`engine.py`, `graph.py`, `api.py`, `app.py`, `context.py`, `parse_curriculum.py`, `eval_kt.py`, `eval_eedi.py`, `simulate.py`, `bench_latency.py`, `results.py`) and their tests. Everything lives under `src/kwan` and `tests/`.
- The Round 1 data files `data/graph.json` and `data/indicators.json`, which held NaCCA text. The graph is `data/graph/math.graph.json` and the structure file is `data/curriculum/math.structure.json`.
- The README placeholders for a demo address, a video and team names, and the video line itself.

## [0.1.0] - 2026-11-10

The first build of Kwan.

### Added

- A parser for the NaCCA Mathematics B1 to B6 PDFs, a rule-built prerequisite graph checked to be acyclic, a multi-level Elo learner model with graph transfer, a recommender with five weighted signals and a soft goal boost, plain-language reasons, and a bank of Ghanaian practice settings by strand and class band.
- A FastAPI service, a Streamlit demo, evaluation on ASSISTments 2009 and Eedi task 3/4 with skill-average, BKT and classic Elo baselines, a learner simulator with two profiles, a latency benchmark, and a results file written by a single writer.
- A test suite and CI on Python 3.11 to 3.13.

[1.0.1]: https://github.com/rofyray/kwan/releases/tag/v1.0.1
[1.0.0]: https://github.com/rofyray/kwan/releases/tag/v1.0.0
[1.0.0rc1]: https://github.com/rofyray/kwan/releases/tag/v1.0.0rc1
[0.1.0]: https://github.com/rofyray/kwan/commit/a86195d
