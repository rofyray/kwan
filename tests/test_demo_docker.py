"""Static checks of the demo's Docker Space files and the two release workflows (PRD Phase 10.3, 10.4, 12.2).

No Docker and no network in CI: the Dockerfile, the Space README front matter, the pinned requirements and
the workflow files are read as text and parsed, and actionlint runs on the workflows when it is installed.
"""

import json
import py_compile
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "demo"
WORKFLOWS = ROOT / ".github" / "workflows"
DOCKERFILE = DEMO / "Dockerfile"
SPACE_README = DEMO / "README.md"
REQUIREMENTS = DEMO / "requirements-demo.txt"
SMOKE = DEMO / "smoke_test.py"
DEPLOY = WORKFLOWS / "space-deploy.yml"
RELEASE = WORKFLOWS / "release.yml"
EM_DASH = chr(0x2014)
SPACE_URL = "https://huggingface.co/spaces/rofyray/kwan"
FRONT_MATTER = {  # PRD Appendix A.3, with the 1.0 scope (Mathematics only) in the short description
    "title": "Kwan",
    "emoji": "⭐",
    "colorFrom": "yellow",
    "colorTo": "green",
    "sdk": "docker",
    "app_port": 7860,
    "pinned": False,
    "license": "mit",
    "short_description": "Next maths skill for Ghanaian pupils, with a reason.",
}
DISCLAIMER = (
    "Kwan uses the NaCCA Standards-Based Curriculum to organise skills. Kwan is not reviewed, approved or "
    "endorsed by NaCCA or the Ministry of Education."
)


def lines(path):
    """Non-blank lines of a text file, comments left out."""
    out = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    return [ln for ln in out if ln and not ln.startswith("#")]


def instructions(path):
    """Dockerfile instructions with continuation lines joined: [(INSTRUCTION, argument text)]."""
    joined, buffer = [], ""
    for ln in path.read_text(encoding="utf-8").splitlines():
        if ln.strip().startswith("#") or not ln.strip():
            continue
        buffer += ln.rstrip("\\") if ln.rstrip().endswith("\\") else ln
        if not ln.rstrip().endswith("\\"):
            joined.append(buffer.strip())
            buffer = ""
    out = []
    for text in joined:
        word, _, rest = text.partition(" ")
        out.append((word, rest.strip()))
    return out


def workflow(path):
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    doc["on"] = doc.pop(True, doc.get("on"))  # PyYAML reads the bare `on` key as boolean True
    return doc


def steps_text(doc):
    return "\n".join(
        str(step.get("run", "")) + "\n" + str(step.get("uses", "")) + "\n" + json.dumps(step.get("with", {}))
        for job in doc["jobs"].values()
        for step in job["steps"]
    )


# ---------------------------------------------------------------- Dockerfile


def test_dockerfile_follows_the_space_layout_of_appendix_a3():
    ins = instructions(DOCKERFILE)
    froms = [arg for word, arg in ins if word == "FROM"]
    assert froms and all(arg.startswith("python:3.12-slim") for arg in froms)
    assert any(word == "RUN" and "useradd -m -u 1000 user" in arg for word, arg in ins)
    assert ("WORKDIR", "/home/user/app") in ins
    assert ("USER", "user") in ins
    assert ("EXPOSE", "7860") in ins
    words = [word for word, _ in ins]
    assert words.index("USER") < words.index("CMD")  # the app never runs as root
    assert words.index("USER") > max(i for i, (w, a) in enumerate(ins) if w == "RUN" and "uv pip install" in a)


def test_dockerfile_runs_streamlit_headless_on_port_7860():
    cmd = [arg for word, arg in instructions(DOCKERFILE) if word == "CMD"]
    assert len(cmd) == 1
    argv = json.loads(cmd[0])
    assert argv[:3] == ["streamlit", "run", "app.py"]
    assert {
        "--server.port=7860",
        "--server.address=0.0.0.0",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
    } <= set(argv)


def test_dockerfile_installs_the_demo_extra_from_a_wheel_and_nothing_heavy():
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "[demo]" in text and "kwan-*.whl" in text
    assert "uv build --wheel" in text  # built from the repository when no wheel is given
    assert "requirements-demo.txt" in text
    for banned in ("torch", "pykt", "sentence-transformers", "apt-get"):
        assert banned not in text, banned
    assert re.search(r"PYTHONDONTWRITEBYTECODE=1\s+PYTHONUNBUFFERED=1\s+UV_NO_CACHE=1", text)
    assert "wheels/kwan-*.whl" in text and "data/eval_results.json" in text
    assert "no wheel under" in text  # a context without wheel or source fails loudly


def test_dockerignore_files_keep_raw_data_and_the_venv_out_of_the_image():
    root_context = lines(DEMO / "Dockerfile.dockerignore")
    assert root_context[0] == "*" and "!src/" in root_context and "!demo/app.py" in root_context
    assert "!data/eval_results.json" in root_context
    assert not any(ln.startswith("!raw") or ln.startswith("!.venv") or ln.startswith("!build") for ln in root_context)
    space_context = lines(DEMO / ".dockerignore")
    assert ".git" in space_context and "smoke_test.py" in space_context


def test_requirements_demo_is_fully_pinned():
    pins = [ln for ln in lines(REQUIREMENTS) if ln and not ln.startswith("#")]
    assert pins and all(re.fullmatch(r"[A-Za-z0-9_.-]+==\d+(\.\d+)*", ln) for ln in pins), pins
    names = {ln.split("==")[0].lower() for ln in pins}
    assert {"streamlit", "pyvis"} <= names
    assert "kwan" not in names  # kwan comes from the wheel, so the file never goes stale on a release


# ---------------------------------------------------------------- Space README


def front_matter():
    text = SPACE_README.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    head, _, body = text[4:].partition("\n---\n")
    return yaml.safe_load(head), body


def test_space_readme_front_matter_is_exactly_appendix_a3_with_the_maths_only_description():
    head, _ = front_matter()
    assert head == FRONT_MATTER
    assert list(head) == list(FRONT_MATTER)  # same order, so the diff against the PRD is readable


def test_space_readme_has_the_live_url_the_disclaimer_and_plain_copy():
    _, body = front_matter()
    assert SPACE_URL in body
    assert DISCLAIMER in body
    assert "English" not in body or "English" in body and "English is not" not in body  # maths only in 1.0
    assert "github.com/rofyray/kwan" in body


def test_no_em_dash_in_any_demo_or_workflow_file():
    files = [p for p in DEMO.iterdir() if p.is_file()] + [DEPLOY, RELEASE]
    for path in files:
        assert EM_DASH not in path.read_text(encoding="utf-8"), path.name


# ---------------------------------------------------------------- smoke test


def test_smoke_test_compiles_and_explains_itself():
    py_compile.compile(str(SMOKE), doraise=True)
    out = subprocess.run([sys.executable, str(SMOKE), "--help"], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0 and "--health-only" in out.stdout and "url" in out.stdout


def test_smoke_test_fails_fast_when_nothing_listens():
    out = subprocess.run(
        [sys.executable, str(SMOKE), "http://127.0.0.1:9", "--health-only", "--wait", "0"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode == 1 and out.stderr.startswith("FAIL: health endpoint")


def test_smoke_test_checks_all_three_tabs_and_an_answer():
    text = SMOKE.read_text(encoding="utf-8")
    assert 'TABS = ["Pupil view", "Curriculum map", "Evidence"]' in text
    for tab in ("Pupil view", "Curriculum map", "Evidence"):
        assert f'open_tab(page, "{tab}")' in text
    assert "Got it right" in text and "did not change after one right answer" in text
    assert "sys.exit(main())" in text


# ---------------------------------------------------------------- workflows


def test_workflows_parse_and_lint():
    for path in (DEPLOY, RELEASE):
        workflow(path)
    linter = shutil.which("actionlint")
    if not linter:
        pytest.skip("actionlint is not installed")
    out = subprocess.run([linter, str(DEPLOY), str(RELEASE)], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stdout + out.stderr


def test_space_deploy_runs_on_release_or_by_hand_and_uploads_the_gradio_folder():
    doc = workflow(DEPLOY)
    assert doc["on"]["release"]["types"] == ["published"]
    assert "workflow_dispatch" in doc["on"]
    text = steps_text(doc)
    assert "uv build --wheel" in text
    assert "cp demo/gradio_app.py space/app.py" in text
    assert "demo/space/README.md demo/space/requirements.txt space/" in text
    assert "unzip -q dist/kwan-*.whl 'kwan/*' -d space/" in text
    assert "data/eval_results.json space/data/" in text
    assert "upload_folder(" in text and 'repo_type="space"' in text and 'folder_path="space"' in text
    raw = DEPLOY.read_text(encoding="utf-8")
    assert "${{ secrets.HF_TOKEN }}" in raw and "${{ vars.HF_SPACE }}" in raw
    assert 'HfApi(token="${{' not in raw  # the token goes through the environment, never into code text


def test_space_deploy_stops_quietly_without_the_settings():
    doc = workflow(DEPLOY)
    steps = doc["jobs"]["deploy"]["steps"]
    assert steps[0]["id"] == "cfg" and "ok=false" in steps[0]["run"]
    for step in steps[1:]:
        assert step.get("if") in ("steps.cfg.outputs.ok == 'true'", "failure()"), step


def test_space_deploy_polls_up_to_ten_minutes_then_runs_the_smoke_test():
    doc = workflow(DEPLOY)
    assert doc["env"]["POLL_SECONDS"] == "600"
    text = steps_text(doc)
    assert ".hf.space" in text
    assert "playwright install --with-deps chromium" in text
    assert "--with playwright demo/gradio_smoke_test.py" in text and "steps.space.outputs.url" in text
    assert '--wait "$POLL_SECONDS"' in text
    # the smoke test waits for the uploaded commit to run, never the old container
    names = [step.get("name", "") for step in doc["jobs"]["deploy"]["steps"]]
    wait = names.index("Wait until the Space runs the uploaded commit")
    assert names.index("Upload to the Space") < wait < names.index("Wait for the Space, then run the smoke test")
    assert 'runtime.get("sha")' in text and "steps.upload.outputs.commit" in DEPLOY.read_text(encoding="utf-8")
    assert doc["permissions"] == {"contents": "read"}


# ---------------------------------------------------------------- the Gradio Space folder

GRADIO_README = DEMO / "space" / "README.md"
GRADIO_REQUIREMENTS = DEMO / "space" / "requirements.txt"
GRADIO_SMOKE = DEMO / "gradio_smoke_test.py"


def test_gradio_space_front_matter_runs_the_app_on_python_3_12():
    text = GRADIO_README.read_text(encoding="utf-8")
    head, _, body = text[4:].partition("\n---\n")
    meta = yaml.safe_load(head)
    assert meta["sdk"] == "gradio" and meta["app_file"] == "app.py" and meta["python_version"] == "3.12"
    assert re.fullmatch(r"\d+\.\d+\.\d+", str(meta["sdk_version"]))
    for key in ("title", "emoji", "colorFrom", "colorTo", "license", "short_description"):
        assert meta[key] == FRONT_MATTER[key]
    assert DISCLAIMER in " ".join(body.split()) and "github.com/rofyray/kwan" in body
    assert EM_DASH not in text


def test_gradio_space_requirements_are_pinned_and_leave_kwan_to_the_wheel():
    pins = lines(GRADIO_REQUIREMENTS)
    assert pins and all(re.fullmatch(r"[A-Za-z0-9_.-]+==\d+(\.\d+)*", ln) for ln in pins), pins
    names = {ln.split("==")[0].lower() for ln in pins}
    assert {"networkx", "pyvis", "pandas"} <= names and "kwan" not in names and "gradio" not in names


def test_gradio_smoke_test_compiles_and_fails_fast_when_nothing_listens():
    py_compile.compile(str(GRADIO_SMOKE), doraise=True)
    out = subprocess.run(
        [sys.executable, str(GRADIO_SMOKE), "http://127.0.0.1:9", "--health-only", "--wait", "0"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode == 1 and out.stderr.startswith("FAIL: health endpoint")
    assert EM_DASH not in GRADIO_SMOKE.read_text(encoding="utf-8")


def test_release_publishes_to_pypi_with_trusted_publishing():
    doc = workflow(RELEASE)
    assert doc["on"] == {"push": {"tags": ["v*"]}}
    pypi = doc["jobs"]["pypi"]
    assert pypi["environment"]["name"] == "pypi"
    assert pypi["permissions"] == {"id-token": "write"}
    uses = [step.get("uses", "") for step in pypi["steps"]]
    assert any(u.startswith("pypa/gh-action-pypi-publish@") for u in uses)
    assert not any("password" in json.dumps(step.get("with", {})) for step in pypi["steps"])  # no API token
    build = doc["jobs"]["build"]
    assert "uv build --sdist --wheel" in steps_text({"jobs": {"build": build}})
    assert doc["jobs"]["check"]["steps"] and "kwan.__version__" in steps_text({"jobs": {"check": doc["jobs"]["check"]}})


def test_release_attaches_the_bundle_and_results_then_dispatches_the_space_deploy():
    doc = workflow(RELEASE)
    gh = doc["jobs"]["github-release"]
    assert gh["permissions"] == {"contents": "write"}
    text = steps_text({"jobs": {"gh": gh}})
    assert "src/kwan/bundles/kwan-bundle.math.json release/" in text and "data/eval_results.json release/" in text
    assert any(step.get("uses", "").startswith("softprops/action-gh-release@") for step in gh["steps"])
    deploy = doc["jobs"]["deploy-space"]
    assert deploy["permissions"] == {"actions": "write"}
    assert "gh workflow run space-deploy.yml" in steps_text({"jobs": {"d": deploy}})
    order = ["check", "build", "pypi", "github-release", "deploy-space"]
    assert list(doc["jobs"]) == order
    for earlier, later in zip(order, order[1:]):
        assert doc["jobs"][later]["needs"] == earlier
