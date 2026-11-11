"""The example scripts run, and the Python blocks of docs/quickstart.md are true (PRD 9.5)."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
QUICKSTART = ROOT / "docs" / "quickstart.md"
VERIFY = re.compile(r"<!-- VERIFY:START id=([\w-]+) lang=(\w+) -->\n```\w+\n(.*?)\n```\n<!-- VERIFY:END -->", re.S)


def run(script: str, *args: str, cwd: Path = ROOT, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(EXAMPLES / script), *args], capture_output=True, text=True, cwd=cwd, timeout=timeout
    )


# ----------------------------------------------------------------------------- examples/


def test_quickstart_runs(tmp_path):
    r = run("quickstart.py", cwd=tmp_path)  # from any directory: the bundle comes with the package
    assert r.returncode == 0, r.stderr
    assert "Next skills:" in r.stdout and "why:" in r.stdout and "try:" in r.stdout
    assert "Strongest skills right now:" in r.stdout and "4 answers recorded" in r.stdout
    assert chr(0x2014) not in r.stdout


def test_simulate_a_class_runs():
    r = run("simulate_a_class.py", "--pupils", "8", "--days", "4")
    assert r.returncode == 0, r.stderr
    assert "8 pupils, 4 days" in r.stdout
    for profile in ("beginner_p1", "gapped_p4", "gapped_p6", "uneven_strands"):
        assert profile in r.stdout
    again = run("simulate_a_class.py", "--pupils", "8", "--days", "4")
    assert again.stdout == r.stdout  # seeded: the same run twice


def test_evaluate_on_ammore_explains_how_to_get_the_data(tmp_path):
    r = run("evaluate_on_ammore.py", "--raw", str(tmp_path / "ammore"))
    assert r.returncode == 0, r.stderr
    assert "git clone https://github.com/owenhenkel/ammore_dataset" in r.stdout and "CC BY-NC" in r.stdout


def test_examples_have_no_em_dash():
    for path in EXAMPLES.glob("*.py"):
        assert chr(0x2014) not in path.read_text(encoding="utf-8"), path.name


# ----------------------------------------------------------------------------- docs/quickstart.md


def _blocks(lang: str) -> list[tuple[str, str]]:
    text = QUICKSTART.read_text(encoding="utf-8")
    return [(m.group(1), m.group(3)) for m in VERIFY.finditer(text) if m.group(2) == lang]


def test_the_quickstart_python_blocks_run_in_order(tmp_path, monkeypatch):
    blocks = _blocks("python")
    assert [name for name, _ in blocks] == [
        "python-basic",
        "python-update-at",
        "python-recommend",
        "python-mastery",
        "python-state-roundtrip",
    ]
    monkeypatch.chdir(tmp_path)  # the blocks must not depend on the checkout
    namespace: dict[str, object] = {}
    for name, code in blocks:
        exec(compile(code, f"quickstart.md:{name}", "exec"), namespace)  # noqa: S102
    assert namespace["state"].to_dict()["step"] == 2
    assert namespace["recommendations"] and namespace["mastery"]["math:B1.1.1.1.1"] > 0


def test_the_quickstart_cli_lines_name_real_commands():
    from kwan import cli

    parser, skipped = cli.build_parser()
    assert not skipped, skipped
    ((_, block),) = [b for b in _blocks("bash") if b[0] == "cli-overview"]
    subparsers = next(a for a in parser._actions if a.dest == "command")
    for line in block.splitlines():
        if line.startswith("kwan ") and not line.startswith("kwan --"):
            words = line.split()
            assert words[1] in subparsers.choices, line
            if words[1] == "bundle":
                subparsers.choices["bundle"].parse_args(words[2:])


def test_the_quickstart_no_longer_carries_the_integration_note():
    text = QUICKSTART.read_text(encoding="utf-8")
    assert "NOTE FOR THE INTEGRATION STEP" not in text and "Release candidate" not in text
    assert "Kwan.from_bundle()" in text


@pytest.mark.parametrize("name", ["api-start", "api-curl"])
def test_the_quickstart_api_blocks_match_the_routes(name):
    ((_, block),) = [b for b in _blocks("bash") if b[0] == name]
    from kwan.api.app import app

    paths = set(app.openapi()["paths"])
    for path in re.findall(r"localhost:8000(/v1/[\w/-]+)", block):
        template = re.sub(r"/pupil-0042", "/{learner_id}", path)
        assert template in paths, path
