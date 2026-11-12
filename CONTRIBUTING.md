# Contributing to Kwan

Thank you for wanting to help. Kwan (Twi for "the path") tells a pupil which maths skill to learn next, explains why, and sets the practice in everyday Ghanaian life. It follows the NaCCA Standards-Based Curriculum for Mathematics, Basic 1 to Basic 6.

## 1. Welcome

Kwan is for Ghanaian pupils, the teachers who teach them, and the people who build learning tools for them. There are two ways to contribute, and both matter equally.

- **Teacher track.** You teach, train teachers or work in education in Ghana. You check that skill links make sense, write or review "Try it in Ghana" practice settings and short skill labels, and review Twi, Ewe and Ga translations. No coding is needed. Go to [section 5](#5-teacher-track).
- **Developer track.** You write code or work with data. You help with curriculum parsing, the learner model, the recommender, evaluation, the API and the demo. Go to [section 3](#3-developer-setup).

Not sure where to start? Open an issue and ask. Issues labelled `good first issue` are a good first step for developers.

## 2. Code of Conduct

Everyone who takes part follows the [Code of Conduct](CODE_OF_CONDUCT.md) (Contributor Covenant 2.1). Be kind, be patient, and assume the other person means well. The Code of Conduct explains how to report a problem in private.

## 3. Developer setup

Kwan needs Python 3.11, 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/), which installs the right Python for you if it is missing.

```bash
git clone https://github.com/rofyray/kwan.git
cd kwan
uv sync --extra api --extra demo --extra eval --extra curriculum --extra gradio
uv run pytest -q
```

`uv sync` creates `.venv`, installs Kwan in editable mode with those extras plus the `dev` group (ruff, mypy, pytest), and pins everything to `uv.lock`. Prefix commands with `uv run`, or activate `.venv` once with `source .venv/bin/activate`. After you change a dependency in `pyproject.toml`, run `uv lock` and commit `uv.lock`.

The curriculum structure, the skill map and the results file are committed, so the tests, the API and the demo run without downloading anything. Tests that need raw data skip themselves.

To rebuild from public sources:

```bash
make data        # downloads the public datasets listed in data/datasets.yaml
make all         # rebuilds every committed artifact
```

`make data` downloads a lot. It is safe to run again: files that are already complete are skipped. Raw data goes into `raw/`, which git ignores.

Before you open a pull request, run the same checks as CI:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -q
```

To work on the documentation site:

```bash
uv run --group docs mkdocs serve             # preview on http://127.0.0.1:8000
uv run --group docs mkdocs build --strict    # the check CI runs
```

## 4. Rules that keep Kwan honest

These rules are not style preferences. A pull request that breaks one of them will not be merged.

1. **Numbers come only from `data/eval_results.json`.** Never type a metric by hand in the README, the docs, the demo or a release note. Each evaluation script writes its own block of that file, and the text reads from it.
2. **Never commit raw datasets.** Not a full file, not a sample of real rows, not in a test fixture. Tests use small made-up data.
3. **Never commit NaCCA curriculum text.** Indicator descriptions and exemplars are © NaCCA, Ministry of Education. Kwan downloads them from the official PDFs at build time. The repository holds only the indicator codes, the structure and Kwan's own short labels. This changes only if NaCCA gives written permission. See [docs/governance.md](docs/governance.md).
4. **Never ship parameters fitted on non-commercial data.** AMMORE and Junyi Academy are for evaluation only. Model parameters and item-difficulty banks in a Kwan bundle must come from datasets marked `commercial_params_ok: true` in `data/datasets.yaml`. The bundle builder refuses anything else. See [docs/data-card.md](docs/data-card.md).
5. **Keep outputs deterministic.** The same bundle, the same answers and the same `now` must give the same recommendations. Exploration is seeded from the learner id and the step. Rounding is half-up and the practice setting is picked with CRC32, so a future port in another language can match Kwan exactly. Do not use unseeded randomness, the system clock or the order of a `set` in anything that affects a result.
6. **No personal data.** Kwan never stores a pupil's name, email or free text. Learner state is keyed by an opaque id chosen by whoever uses Kwan.
7. **Do not claim NaCCA approval.** Kwan uses the NaCCA Standards-Based Curriculum to organise skills. Kwan is not reviewed, approved or endorsed by NaCCA or the Ministry of Education. Nothing in the code, the docs or the demo may suggest otherwise.
8. **Write for a Ghanaian teacher.** Plain words, short sentences, no em dashes in anything a user reads, no invented statistics, and Ghanaian examples that are true: real places, real cedi coins and notes, everyday foods and activities. The [context style guide](docs/context-style-guide.md) has the details.

## 5. Teacher track

You do not need a GitHub workflow, a terminal or any code. You need a spreadsheet program (Google Sheets, Excel or LibreOffice) and your knowledge of how children learn maths.

**How to start.** Open a ["Teacher review" issue](https://github.com/rofyray/kwan/issues/new?template=teacher-review.yml) and tell us which classes you teach and what you would like to review. A maintainer will reply and send you a review sheet. Please do not post your phone number or email address in a public issue.

**How review sheets work.** A review sheet is a spreadsheet with one row per item. You read each row, choose a decision, add a note if you want, and send the sheet back. A maintainer imports your decisions, and your reviewer id and the date are recorded next to each one. The [teacher review guide](docs/teacher-review-guide.md) explains every column.

**What to check in a skill link.** Each row shows two skills, A and B, and asks one question: *Would a pupil really need A before B?* Accept the link if a child who cannot yet do A would struggle to learn B. Reject it if B can be learned without A, even when the two skills are on a similar topic.

**Practice settings and short labels.** A practice setting is the "Try it in Ghana" line that comes with a recommended skill. A short label is a plain name for a skill in 60 characters or fewer. For each one you choose accept, edit or reject. Follow the [context style guide](docs/context-style-guide.md): real places, real coins and notes, prices a child would meet at the market, girls and boys equally, and never a sentence copied from the NaCCA curriculum.

**Translations.** For Basic 1 to Basic 3, Kwan offers practice settings in Twi, Ewe and Ga. A first draft is made by machine and stays hidden until a fluent teacher approves it. If you are fluent in one of these languages, you can review them.

**How credit is given.** Every reviewer and translator is listed in [CONTRIBUTORS.md](CONTRIBUTORS.md) under the name they choose. You may also list the region where you teach. Tell us if you would rather not be named.

## 6. Pull requests

- **Conventional Commits.** Start the pull request title and each commit message with a type: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `perf:`, `build:`, `ci:` or `chore:`. Example: `fix: count B1 Algebra indicators that wrap across two lines`.
- **One topic per pull request.** Small pull requests get reviewed faster.
- **Tests are required.** New behaviour needs a test. A bug fix needs a test that fails without the fix.
- **CI must be green.** Lint, type check and tests on Python 3.11 to 3.13, and the docs build.
- **One review.** A maintainer or another contributor reviews before merge.

The pull request template has this checklist:

- [ ] The title follows Conventional Commits and the pull request covers one topic.
- [ ] Tests are added or updated, and `uv run pytest -q` passes.
- [ ] `uv run ruff check .`, `uv run ruff format --check .` and `uv run mypy` pass.
- [ ] Every number in user-facing text comes from `data/eval_results.json`.
- [ ] No raw dataset rows and no NaCCA curriculum text are added.
- [ ] No parameters or item banks fitted on non-commercial data are shipped.
- [ ] Outputs stay deterministic (seeded randomness, half-up rounding, no clock or `set` order in results).
- [ ] No pupil names, emails or free text are stored or logged.
- [ ] User-facing text is in plain language, with no em dashes and no claim of NaCCA approval.
- [ ] New or changed practice settings follow the context style guide and are marked reviewed only after a teacher has reviewed them.
- [ ] Docs are updated where behaviour changed, and `uv run --group docs mkdocs build --strict` passes.

## 7. Reporting curriculum errors

If Kwan has a skill wrong (a missing indicator, a description that is cut short, a wrong strand name, a link that makes no sense), please tell us. Use the ["Curriculum error" issue template](https://github.com/rofyray/kwan/issues/new?template=curriculum-error.yml) and give:

- the **indicator code**, for example `B4.1.1.1.2`;
- the **PDF** (Mathematics B1 to B3 or Mathematics B4 to B6) and the **page number** where the indicator appears;
- what Kwan shows and what the curriculum says. A few words are enough. Please do not paste whole pages of the curriculum.

## Licences

By contributing you agree that your code is released under the [MIT licence](LICENSE), and that practice settings, short labels, translations and documentation you write are released under [CC BY 4.0](LICENSE-CONTENT).
