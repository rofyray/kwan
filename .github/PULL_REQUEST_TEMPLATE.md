<!-- Title: use Conventional Commits, for example "fix: count B1 Algebra indicators that wrap across two lines". -->

## What this changes

<!-- One or two sentences. One topic per pull request. -->

## Why

<!-- Link the issue if there is one: Closes #123 -->

## How it was tested

<!-- The commands you ran and what you saw. -->

## Checklist

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

See [CONTRIBUTING.md](https://github.com/rofyray/kwan/blob/main/CONTRIBUTING.md) for the reasons behind each line.
