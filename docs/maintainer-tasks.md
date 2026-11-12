# Maintainer tasks

The jobs only a maintainer can do because they need an account, a secret or a person: releasing, the demo
Space, teacher review, and the decision model for extra skill links. The first two are one-time set-up;
the last two are ongoing work with no deadline. Every command runs from a source checkout after
`uv sync --extra api --extra demo --extra eval --extra curriculum --extra gradio`.

| Task | When | Needs |
|---|---|---|
| [Release to PyPI](#release-to-pypi) | each release | the `pypi` trusted publisher (set up once) |
| [Demo Space](#demo-space-on-hugging-face) | set up once, then automatic | a Hugging Face account and token |
| [Teacher review](#teacher-review) | ongoing, before the drafts can show by default | teachers who review sheets |
| [Decision model](#decision-model-for-extra-skill-links) | later, when a provider is chosen | an API key for a chat model |
| NaCCA permission | once | the letter in [Governance](governance.md#kwan-and-nacca) |

## Release to PyPI

Set up once: on pypi.org, add a pending trusted publisher with project `kwan`, owner `rofyray`, repository
`kwan`, workflow `release.yml` and environment `pypi`; on GitHub, create the environment `pypi` (Settings,
Environments).

For each release:

1. Set the version in `src/kwan/_version.py` (for example `1.0.0`). Then rebuild the bundle so its
   `bundle_version` matches, refresh the docs, and run the checks:

    ```bash
    make bundle fill-docs test lint
    ```

2. Move the `Unreleased` notes in `CHANGELOG.md` under the new version and date, update `version` in
   `CITATION.cff`, commit and push to `main`, and wait for CI to pass.
3. Tag the commit and push the tag:

    ```bash
    git tag v1.0.0 && git push origin v1.0.0
    ```

`release.yml` checks that the tag matches the package version, runs the tests, builds with `uv build`,
publishes to PyPI, creates the GitHub release with the bundle and `eval_results.json` attached, and then
starts `space-deploy.yml`. If the Space is not set up yet, that last step says so and stops without failing.

## Demo Space on Hugging Face

The public demo is a **Gradio** Space, which runs on the free plan (Docker Spaces are paid). The app is
`demo/gradio_app.py`; the Streamlit app with `demo/Dockerfile` stays for anyone who hosts it in Docker.

Set up once:

1. On huggingface.co, choose **New Space**: owner `rofyray`, name `kwan`, SDK **Gradio**, template
   **Blank**, hardware **CPU Basic** (free; Kwan needs no GPU, and the free ZeroGPU option also works),
   visibility **Public**. Create it empty: the workflow uploads every file.
2. Under **Settings, Access Tokens**, create a **fine-grained** token with **write** access to the Space
   `rofyray/kwan` (nothing else).
3. In the GitHub repository, open **Settings, Secrets and variables, Actions**. Add the repository secret
   `HF_TOKEN` (the token) and, on the **Variables** tab, the variable `HF_SPACE` = `rofyray/kwan`.
4. Open **Actions, space-deploy, Run workflow** on `main`. It builds the wheel, uploads `app.py`, the
   `kwan` package, the results file, `README.md` (the Space settings) and `requirements.txt`, waits up to
   ten minutes for the Space to answer, and runs `demo/gradio_smoke_test.py` against it in a browser.

After that, every published release deploys the Space again. To try the Gradio app on your machine:

```bash
make app-gradio
```

## Teacher review

Ongoing. The 828 practice settings and 276 short labels are drafts written for Kwan; until a teacher
approves one, Kwan shows the general setting for the strand and class band, and the demo shows drafts
only behind a labelled switch. The [Teacher review guide](teacher-review-guide.md) is what you send to
reviewers; this is the maintainer's side.

1. Write a review sheet (100 settings and labels per sheet; `--grade 4` for one class, `--batch 2` for the
   next sheet, `--all` for every sheet):

    ```bash
    uv run kwan context review-export --out-dir build/review
    ```

2. Send the CSV to a teacher. They fill in the decision and note columns and send it back.
3. Check what the sheet would change, then apply it:

    ```bash
    uv run kwan context review-import build/review/filled.csv --dry-run
    uv run kwan context review-import build/review/filled.csv
    ```

4. Rebuild the bundle, refresh the docs and commit the changed files under `data/contexts/`, the bundle
   and the docs:

    ```bash
    make bundle fill-docs
    ```

5. Credit the reviewer in `CONTRIBUTORS.md` if they agree to be named.

Skill links proposed by the decision model are reviewed the same way, with
`uv run kwan graph review-export` and `uv run kwan graph review-import <sheet> --reviewer <id>`. Today
there are 4 proposed links, none reviewed. Translations (Twi, Ewe, Ga) use
`uv run kwan context review-export --lang tw`; none are drafted yet.

`uv run kwan context coverage` shows how much is drafted, checked, reviewed and translated.

## Decision model for extra skill links

Later, once you choose a provider. The rule-based skill map is complete and is what Kwan ships. A
decision model can propose extra links between strands (for example, a measurement skill that needs a
fractions skill). The only model that runs without a key, a local sentence encoder, reached 0.31 precision
on the Junyi benchmark, below the 0.80 bar, so nothing it proposed ships.

1. Choose any service with an OpenAI-style chat completions endpoint (a hosted model, or a local server
   such as Ollama). Set three environment variables; the key stays on your machine:

    ```bash
    export KWAN_LLM_ENDPOINT="https://example.com/v1/chat/completions"
    export KWAN_LLM_MODEL="model-id"
    export KWAN_LLM_API_KEY="your-key"
    ```

2. Download the Junyi benchmark and the curriculum PDFs, and parse the full curriculum text. The full text
   stays in the git-ignored `build/` folder:

    ```bash
    bash scripts/download_data.sh junyi
    uv run kwan curriculum fetch && uv run kwan curriculum parse
    ```

3. Measure the model on Junyi first. It must reach 0.80 precision:

    ```bash
    uv run kwan graph augment --provider llm-json --junyi raw/junyi --results data/eval_results.json
    ```

4. If it passes, propose links on the NaCCA curriculum. They are written to
   `data/graph/math.augmented.json` as pending and do not reach learners until a reviewer accepts them:

    ```bash
    uv run kwan graph augment --provider llm-json --results data/eval_results.json
    ```

5. Review the proposed links with teachers (see [Teacher review](#teacher-review)), then run
   `make graph bundle fill-docs` and commit.

Calls are cached in `build/augment/cache.jsonl`, so a rerun costs nothing for pairs already asked.
